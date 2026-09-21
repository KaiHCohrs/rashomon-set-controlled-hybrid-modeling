import sys
import os
import itertools
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import random as orandom
import copy

import jax
from jax import grad, vmap, random, jit
from jax import numpy as jnp
from jax.lax import cond
from jax.numpy import sin, cos, exp

from .building_blocks import create_network

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from hybridmodeling.datasets.loaders import CustomLoader, CustomBootstrapLoader
from flax.training import train_state

from tqdm import trange
import optax

from flax.training import train_state
from typing import Any

from typing import NamedTuple

import chex
from sympy import symbols, sympify, lambdify
from scipy.optimize import minimize


class TrainState(train_state.TrainState):
    batch_stats: Any


class EnsembleHybridJNN(BaseEstimator, RegressorMixin):
    def __init__(self, model_config, trainer_config, equation_config, seed=1):
        # Initialize model, trainer, and equation configurations
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.equation_config = equation_config
        self.parameters = equation_config["parameter_values"].keys()
        self.fp = equation_config["fp"]

        # Prepare expressions for symbolic evaluation
        for i in range(10):
            self.fp = self.fp.replace(f"x{i}", f"x[{i}]")

        # Adjust final layer nonlinearity based on equation type
        if (
            equation_config["type"] == "fp / fa"
            or equation_config["type"] == "a0 / (fp + fa)"
        ):
            self.model_config["final_nonlin"] = True
        if "comment" in equation_config.keys():
            if equation_config["comment"] == "positive":
                self.model_config["final_nonlin"] = True

        # Set seed for reproducibility
        self.seed = model_config.get("seed", seed)

        # Define input dimensions and model parameters
        self.model_config["layers"][0] = len(self.equation_config["fa_input"])
        self.equation = equation_config["type"]
        self.fa_input = jnp.array(equation_config["fa_input"])

        # Set model configuration
        self.layers = self.model_config["layers"]
        self.final_nonlin = self.model_config["final_nonlin"]
        self.p = self.model_config["dropout_p"]

        # Regularization: only variance (l2_net) regularization
        self.l2_net_decay = self.trainer_config.get("l2_net_decay", 0.0)

        self.split = self.trainer_config["split"]
        self.ensemble_size = self.model_config["ensemble_size"]
        self.iterations = self.trainer_config["iterations"]

        # Initialize network
        self.init_net, self.apply_net = create_network(self.model_config)

        # Initialize logs and counters
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.train_GT_log = []
        self.val_GT_log = []
        self.test_GT_log = []
        self.loss_test_log = [jnp.array(self.ensemble_size * [jnp.inf])]
        self.params_log = []

    def _effective_l2_net_decay(self):
        """Variance regularization coefficient (subclasses may override for Rashomon)."""
        return self.l2_net_decay

    # Apply hybrid model combining physical and neural network outputs
    def apply_hybrid(self, fa_output, params, inputs):
        x = inputs
        fa = fa_output
        locals().update({param: params[i] for i, param in enumerate(self.parameters)})
        fp = eval(self.fp)
        return eval(self.equation)

    # Forward pass during training
    def net_forward(self, net_params, params, state, inputs):
        fa_outputs, updated_states = self.state.apply_fn(
            {"params": net_params, "batch_stats": state.batch_stats},
            x=inputs[:, self.fa_input],
            train=True,
            mutable=["batch_stats"],
        )
        outputs = vmap(self.apply_hybrid, in_axes=(0, None, 0))(
            fa_outputs, params, inputs
        )
        return outputs, updated_states

    # Forward pass during testing
    def net_forward_test(self, net_params, params, state, inputs):
        fa_outputs = self.state.apply_fn(
            {"params": net_params, "batch_stats": state.batch_stats},
            x=inputs[:, self.fa_input],
            train=False,
        )
        outputs = vmap(self.apply_hybrid, in_axes=(0, None, 0))(
            fa_outputs, params, inputs
        )
        return outputs

    def _zero_reg(self, net_params, _):
        """No weight decay; only variance regularization is used."""
        return 0.0

    # Variance (output) regularization only
    def l2_net_regularization(self, net_params, state, inputs, l2_net_decay):
        fa_outputs = self.state.apply_fn(
            {"params": net_params, "batch_stats": state.batch_stats},
            x=inputs[:, self.fa_input],
            train=False,
        )
        return l2_net_decay * jnp.sqrt(jnp.sum(jnp.square(fa_outputs)))

    # Compute loss and update state
    def loss(self, net_params, params, state, batch):
        inputs, targets = batch
        outputs, updated_states = self.net_forward(net_params, params, state, inputs)
        l2_net = self._effective_l2_net_decay()
        regularization = self.l2_net_regularization(net_params, state, inputs, l2_net)
        loss = jnp.mean((targets - outputs) ** 2) + regularization
        return loss, updated_states

    def loss_with_fixed_net_params(self, params, net_params, all_state, batch):
        loss_value, _ = self.loss(net_params, params, all_state, batch)
        return loss_value

    # Compute and log loss during in training mode
    def monitor_loss(self, net_params, params, state, batch):
        inputs, targets = batch
        loss_value, _ = self.loss(net_params, params, state, batch)
        l2_net = self._effective_l2_net_decay()
        regularization = self.l2_net_regularization(net_params, state, inputs, l2_net)
        RMSE_value = loss_value - regularization
        return RMSE_value, loss_value

    # Compute and log loss during in test mode
    def monitor_loss_test(self, net_params, params, state, batch):
        inputs, targets = batch
        outputs = self.net_forward_test(net_params, params, state, inputs)
        RMSE_value = jnp.mean((targets - outputs) ** 2)
        l2_net = self._effective_l2_net_decay()
        regularization = self.l2_net_regularization(net_params, state, inputs, l2_net)
        loss_value = RMSE_value + regularization
        return RMSE_value, loss_value

    # Single optimization step
    @partial(jit, static_argnums=(0,))
    def step(self, i, state, params, opt_state_params, batch):
        (l, updated_state), grads_net = jax.value_and_grad(
            self.loss, argnums=0, has_aux=True
        )(state.params, params, state, batch)
        state = state.apply_gradients(grads=grads_net)
        state = state.replace(batch_stats=updated_state["batch_stats"])

        (loss_value, updated_param_state), grads_params = jax.value_and_grad(
            self.loss, argnums=1, has_aux=True
        )(state.params, params, state, batch)
        updates_params, opt_state_params = self.optimizer_params.update(
            grads_params,
            opt_state_params,
            params,
            value=loss_value,
            grad=grads_params,
            value_fn=self.loss_with_fixed_net_params,
            net_params=state.params,
            all_state=state,
            batch=batch,
        )
        params = optax.apply_updates(params, updates_params)

        return state, params, opt_state_params, grads_params

    # Functions for storing best states
    def update_best_weights(
        self, net, params, it, net_best, params_best, it_best, counter
    ):
        counter = jnp.array(0, dtype=counter.dtype)
        return net, params, it, counter

    def keep_best_weights(
        self, net, params, it, net_best, params_best, it_best, counter
    ):
        counter = counter + 1
        return net_best, params_best, it_best, counter

    def store_best(
        self, update, net, params, it, net_best, params_best, it_best, counter
    ):
        return jax.lax.cond(
            update,
            self.update_best_weights,
            self.keep_best_weights,
            net,
            params,
            it,
            net_best,
            params_best,
            it_best,
            counter,
        )

    # Functions for reseting model upon decay
    def keep_current_weights(self, net, params, net_best, params_best):
        return net, params

    def restore_best_weights(self, net, params, net_best, params_best):
        return net_best, params_best

    def restore_best(self, decay, net, params, net_best, params_best):
        return jax.lax.cond(
            decay,
            self.restore_best_weights,
            self.keep_current_weights,
            net,
            params,
            net_best,
            params_best,
        )

    # Functions for updating learning rates
    def update_lr(self, learning_rates, counter, decay_factor):
        learning_rates = learning_rates * decay_factor
        counter = jnp.array(0, dtype=counter.dtype)
        return learning_rates, counter

    def keep_lr(self, learning_rates, counter, decay_factor):
        return learning_rates, counter

    def decay_step(self, decay, learning_rates, counter, decay_factor):
        return jax.lax.cond(
            decay, self.update_lr, self.keep_lr, learning_rates, counter, decay_factor
        )

    def batch_normalize(self, data_val, norm_const):
        X, y = data_val

        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y

        return [X, y]

    def _fit_one_run(self, X, y, warm_start=False):
        """Single training pass; used by fit() and by RashomonRobustEnsembleHybridJNN."""
        nIter = self.iterations
        if not warm_start:
            rng_key = random.PRNGKey(self.seed)
            rng_key1, rng_key2, self.rng_key_fit = random.split(rng_key, 3)
            (
                k1,
                k2,
            ) = random.split(rng_key1, 2)
            keys_1 = random.split(k1, self.ensemble_size)
            test_input = jnp.ones(X[:, self.fa_input].shape)
            self.net = vmap(self.init_net, in_axes=(0, None, None))(
                keys_1, test_input, True
            )
            self.net_params = self.net["params"]
            self.net_batch_stats = self.net["batch_stats"]
            schedule_net_params = self.trainer_config["init_lr_net"]
            self.optimizer_net_params = optax.chain(
                optax.inject_hyperparams(optax.adamw)(
                    learning_rate=schedule_net_params,
                    weight_decay=0.0,
                )
            )
            self.state = vmap(
                lambda params, batch_stats: TrainState.create(
                    apply_fn=self.apply_net,
                    params=params,
                    batch_stats=batch_stats,
                    tx=self.optimizer_net_params,
                )
            )(self.net_params, self.net_batch_stats)
            if "initial_guess" in self.equation_config.keys():
                self.params = jax.random.normal(
                    k2, (self.ensemble_size, len(self.parameters))
                )
                self.params = np.array(
                    list(self.equation_config["uncertainty"].values())
                ).reshape(1, -1) * self.params + np.array(
                    list(self.equation_config["initial_guess"].values())
                ).reshape(
                    1, -1
                )
            else:
                self.params = jax.random.uniform(
                    k2,
                    (self.ensemble_size, len(self.parameters)),
                    minval=-10,
                    maxval=10,
                )
            schedule_params = self.trainer_config["init_lr_params"]
            self.optimizer_params = optax.chain(
                optax.inject_hyperparams(optax.adam)(learning_rate=schedule_params),
                optax.inject_hyperparams(optax.scale_by_zoom_linesearch)(
                    max_linesearch_steps=100, initial_guess_strategy="one"
                ),
            )
            self.opt_state_params = vmap(self.optimizer_params.init)(self.params)
            self.net_best = copy.deepcopy(self.net_params)
            self.params_best = self.params
        else:
            self.net_params = self.net_best
            self.params = np.array(self.params_best)
            self.opt_state_params = vmap(self.optimizer_params.init)(self.params_best)
            self.state = vmap(
                lambda params, batch_stats: TrainState.create(
                    apply_fn=self.apply_net,
                    params=params,
                    batch_stats=batch_stats,
                    tx=self.optimizer_net_params,
                )
            )(self.net_best, self.state_best.batch_stats)
            self.state = self.state.replace(params=self.net_best)
            self.it_best = jnp.zeros(self.ensemble_size)
            self.counter = jnp.zeros(self.ensemble_size)
            self.lr_net = (
                jnp.ones(self.ensemble_size) * self.trainer_config["init_lr_net"]
            )
            self.lr_params = (
                jnp.ones(self.ensemble_size) * self.trainer_config["init_lr_params"]
            )
        if len(y.shape) == 1:
            y = y[:, None]
        X, y = jnp.array(X), jnp.array(y)
        if not warm_start:
            rng_key, rng_key_loader = random.split(self.rng_key_fit, 2)
            self._rng_key_loader = rng_key_loader
        dataset = CustomBootstrapLoader(
            X,
            y,
            self.trainer_config["batch_size"],
            self.ensemble_size,
            split=self.split,
            rng_key=self._rng_key_loader,
        )
        if not warm_start:
            self.it_best = jnp.zeros(self.ensemble_size)
            self.counter = jnp.zeros(self.ensemble_size)
            self.lr_net = (
                jnp.ones(self.ensemble_size) * self.trainer_config["init_lr_net"]
            )
            self.lr_params = (
                jnp.ones(self.ensemble_size) * self.trainer_config["init_lr_params"]
            )
        self.maximize = -1 if self.trainer_config["maximize"] else 1
        finalized = jnp.zeros(self.ensemble_size, dtype=bool)
        data = iter(dataset)
        self.norm_const = dataset.norm_const
        (self.mu_X, self.sigma_X), (self.mu_y, self.sigma_y) = self.norm_const
        v_step = jit(vmap(self.step, in_axes=(None, 0, 0, 0, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes=(0, 0, 0, 0)))
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes=(0, 0, 0, 0)))
        v_store_best = vmap(self.store_best, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
        v_restore_best = vmap(self.restore_best, in_axes=(0, 0, 0, 0, 0))
        v_decay_step = vmap(self.decay_step, in_axes=(0, 0, 0, None))
        v_batch_normalize = vmap(self.batch_normalize, in_axes=(0, 0))
        data_train = v_batch_normalize(dataset.data_train, dataset.norm_const)
        data_val = v_batch_normalize(dataset.data_val, dataset.norm_const)
        self.data_train = data_train
        self.data_val = data_val
        self.loss_test_log = [jnp.array(self.ensemble_size * [jnp.inf])]
        rng_key = (
            self.rng_key_fit if not warm_start else random.fold_in(self.rng_key_fit, 1)
        )
        pbar = trange(nIter)
        for it in pbar:
            rng_key, *rng_keys = random.split(rng_key, self.ensemble_size + 1)
            batch = next(data)
            self.state, self.params, self.opt_state_params, grads_params = v_step(
                it, self.state, self.params, self.opt_state_params, batch
            )
            self.params_log.append(self.params)
            RMSE_value, loss_value = v_monitor_loss(
                self.state.params, self.params, self.state, batch
            )
            self.loss_log.append(loss_value)
            if it % 10 == 0:
                RMSE_test_value, loss_test_value = v_monitor_loss_test(
                    self.state.params, self.params, self.state, data_val
                )
                update = (jnp.array(self.loss_test_log) * self.maximize).min(
                    axis=0
                ) - self.trainer_config["tolerance"] > loss_test_value * self.maximize
                self.loss_test_log.append(loss_test_value)
                RMSE_train_value, loss_train_value = v_monitor_loss_test(
                    self.state.params, self.params, self.state, data_train
                )
                self.train_log.append(loss_train_value)
                RMSE_val_value, loss_val_value = v_monitor_loss_test(
                    self.state.params, self.params, self.state, data_val
                )
                self.val_log.append(loss_val_value)
                (
                    self.net_best,
                    self.params_best,
                    self.it_best,
                    self.counter,
                ) = v_store_best(
                    update,
                    self.state.params,
                    self.params,
                    jnp.ones(self.ensemble_size) * it,
                    self.net_best,
                    self.params_best,
                    self.it_best,
                    self.counter,
                )
                self.state_best = self.state.replace(params=self.net_best)
                decay = jnp.array(self.trainer_config["patience"]) < self.counter
                decay = ~finalized * decay
                self.lr_net, self.counter = v_decay_step(
                    decay,
                    self.lr_net,
                    self.counter,
                    self.trainer_config["decay_factor"],
                )
                self.lr_params, self.counter = v_decay_step(
                    decay,
                    self.lr_params,
                    self.counter,
                    self.trainer_config["decay_factor"],
                )
                self.state.opt_state[0].hyperparams["learning_rate"] = self.lr_net
                self.opt_state_params[0].hyperparams["learning_rate"] = self.lr_params
                self.net_params, self.params = v_restore_best(
                    decay,
                    self.state.params,
                    self.params,
                    self.net_best,
                    self.params_best,
                )
                self.state = self.state.replace(params=self.net_params)
                finalized = (
                    jnp.array(self.lr_net)
                    < self.trainer_config["init_lr_net"]
                    * self.trainer_config["decay_factor"]
                    ** self.trainer_config["decay_steps"]
                )
                best_test_RMSE, best_test_loss = v_monitor_loss_test(
                    self.state_best.params, self.params_best, self.state_best, data_val
                )
                pbar.set_postfix(
                    {
                        "Min/Max loss": f"{RMSE_value.min()}/{RMSE_value.max()}",
                        "Min/Max test loss": f"{RMSE_test_value.min()}/{RMSE_test_value.max()}",
                        "Min/Max best test loss": f"{best_test_RMSE.min()}/{best_test_RMSE.max()}",
                    }
                )
                if finalized.all():
                    break
        self.state_best = self.state.replace(params=self.net_best)
        self.state = self.state.replace(params=self.net_best)
        self.params_best = np.array(self.params_best)

    def fit(self, X, y):
        self._fit_one_run(X, y, warm_start=False)
        print(self.params_best)

    def _compute_train_val_joint_mse(self):
        """Compute train/val/joint MSE for Rashomon set (requires fit to have been run)."""
        if not hasattr(self, "data_val") or not hasattr(self, "data_train"):
            raise ValueError("Must call fit() before _compute_train_val_joint_mse()")
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes=(0, 0, 0, 0)))
        train_vals, _ = v_monitor_loss_test(
            self.state_best.params, self.params_best, self.state_best, self.data_train
        )
        val_vals, _ = v_monitor_loss_test(
            self.state_best.params, self.params_best, self.state_best, self.data_val
        )
        # monitor_loss_test returns (MSE, loss_value); use MSE for joint
        train_MSE_np = np.array(train_vals)
        val_MSE_np = np.array(val_vals)
        n_train = self.data_train[1].shape[1]
        n_val = self.data_val[1].shape[1]
        finite_both = np.isfinite(train_MSE_np) & np.isfinite(val_MSE_np)
        joint_MSE_np = np.where(
            finite_both,
            (n_train * train_MSE_np + n_val * val_MSE_np) / (n_train + n_val),
            np.nan,
        )
        self._train_MSE_np = train_MSE_np
        self._val_MSE_np = val_MSE_np
        self._joint_MSE_np = joint_MSE_np
        self._val_RMSE_np = np.array(np.sqrt(np.maximum(val_MSE_np, 0.0)))

    def compute_rashomon_set(self, epsilon):
        """Rashomon set: ensemble members with joint MSE <= best_joint_MSE + epsilon."""
        if not hasattr(self, "data_val"):
            raise ValueError("Must call fit() before compute_rashomon_set()")
        self._compute_train_val_joint_mse()
        joint_MSE = jnp.array(self._joint_MSE_np)
        val_RMSE = jnp.array(self._val_RMSE_np)
        mse_finite = jnp.where(jnp.isfinite(joint_MSE), joint_MSE, jnp.inf)
        best_joint_mse = float(jnp.min(mse_finite))
        best_idx = int(jnp.argmin(mse_finite))
        if np.isinf(best_joint_mse) or best_idx < 0:
            best_joint_mse = float(np.nan)
            best_idx = 0
        best_val_rmse = (
            float(val_RMSE[best_idx])
            if jnp.isfinite(val_RMSE[best_idx])
            else float(np.nan)
        )
        rashomon_mask = (joint_MSE <= (best_joint_mse + epsilon)) & jnp.isfinite(
            joint_MSE
        )
        rashomon_mask = rashomon_mask.at[best_idx].set(True)
        rashomon_indices = jnp.where(rashomon_mask)[0]
        rashomon_params = self.params_best[rashomon_indices]
        RMSE_values = val_RMSE
        rashomon_rmse = RMSE_values[rashomon_indices]

        def extract_params(params_tree, indices):
            def extract_leaf(p):
                if (
                    hasattr(p, "__getitem__")
                    and hasattr(p, "shape")
                    and len(p.shape) > 0
                ):
                    if p.shape[0] == self.ensemble_size:
                        return p[indices]
                return p

            return jax.tree_util.tree_map(extract_leaf, params_tree)

        rashomon_net_params = extract_params(self.state_best.params, rashomon_indices)
        param_names = list(self.parameters)
        estimated_params = self.equation_config.get("parameters") or param_names
        if not estimated_params:
            estimated_params = param_names
        try:
            estimated_indices = [param_names.index(p) for p in estimated_params]
        except ValueError:
            estimated_indices = list(range(len(param_names)))
        rashomon_params_estimated = rashomon_params[:, estimated_indices]
        n_rashomon = len(rashomon_indices)
        if n_rashomon > 1:
            param_diff = (
                rashomon_params_estimated[:, jnp.newaxis, :]
                - rashomon_params_estimated[jnp.newaxis, :, :]
            )
            param_distances = jnp.linalg.norm(param_diff, axis=2)
            upper_triangle_mask = jnp.triu(
                jnp.ones((n_rashomon, n_rashomon), dtype=bool), k=1
            )
            pairwise_distances = param_distances[upper_triangle_mask]
            rashomon_diameter = float(jnp.max(pairwise_distances))
        else:
            rashomon_diameter = 0.0
        self.rashomon_diameter_parameter_names = [
            param_names[i] for i in estimated_indices
        ]
        self.rashomon_indices = rashomon_indices
        self.rashomon_params = rashomon_params
        self.rashomon_net_params = rashomon_net_params
        self.rashomon_rmse = rashomon_rmse
        self.rashomon_losses = rashomon_rmse
        self.rashomon_best_val_loss = best_joint_mse
        self.rashomon_best_idx = best_idx
        self.rashomon_epsilon = epsilon
        self.rashomon_diameter = rashomon_diameter
        return (
            rashomon_indices,
            rashomon_params,
            rashomon_net_params,
            rashomon_rmse,
            best_val_rmse,
        )

    def compute_epsilon_estimates(self, noise_level=None):
        """Epsilon for Rashomon: eps1 (from aggregate), eps3 (best 20% spread), eps4 = max(eps1, eps3). The code uses eps4."""
        if not hasattr(self, "data_val") or not hasattr(self, "data_train"):
            raise ValueError("Must call fit() before compute_epsilon_estimates()")
        self._compute_train_val_joint_mse()
        joint_MSE_np = self._joint_MSE_np
        eps1 = getattr(
            self, "_initial_rashomon_epsilon", getattr(self, "rashomon_epsilon", np.nan)
        )
        if not np.isfinite(eps1):
            eps1 = np.nan
        finite_joint = np.isfinite(joint_MSE_np)
        if np.sum(finite_joint) > 0:
            sorted_joint = np.sort(joint_MSE_np[finite_joint])
            n = len(sorted_joint)
            k = max(1, int(np.ceil(0.2 * n)))
            spread = float(sorted_joint[k - 1] - sorted_joint[0])
            eps3 = spread
        else:
            eps3 = np.nan
        valid = [c for c in [eps1, eps3] if np.isfinite(c)]
        eps4 = float(max(valid)) if valid else np.nan
        self.epsilon_estimate_1 = eps1
        self.epsilon_estimate_3 = eps3
        self.epsilon_estimate_4 = eps4
        return eps1, eps3, eps4

    # Evaluates predictions at test points
    # @partial(jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def posterior(self, x):
        normalize = vmap(lambda x, mu, std: (x - mu) / std, in_axes=(0, 0, 0))
        denormalize = vmap(lambda x, mu, std: x * std + mu, in_axes=(0, 0, 0))

        x = jnp.tile(x[jnp.newaxis, :, :], (self.ensemble_size, 1, 1))
        inputs = normalize(x, self.mu_X, self.sigma_X)

        # net_forward_batched = vmap(self.net_forward_test, (None, None, None, 0))
        samples = vmap(self.net_forward_test, (0, 0, 0, 0))(
            self.state_best.params, self.params_best, self.state_best, inputs
        )
        samples = denormalize(samples, self.mu_y, self.sigma_y)

        return samples

    # @partial(jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def predict(self, x):
        # accepts and returns un-normalized data
        samples = self.posterior(x)
        return samples.mean(0).reshape(-1), samples.std(0)

    def score(self, x, y):
        y_pred = self.predict(x)
        return mean_squared_error(y, y_pred, squared=False), r2_score(y, y_pred)

    def v_mean_squared_error(self, y_true, y_pred):
        y_true = jnp.broadcast_to(y_true, y_pred.shape)
        squared_diff = (y_pred - y_true) ** 2
        mse = jnp.mean(squared_diff, axis=1)
        rmse = jnp.sqrt(mse)
        return rmse

    def v_r2_score(self, y_true, y_pred):
        y_true = jnp.broadcast_to(y_true, y_pred.shape)
        y_mean = jnp.mean(y_true, axis=1)
        total_ss = jnp.sum((y_true - y_mean[:, None]) ** 2, axis=1)
        residual_ss = jnp.sum((y_pred - y_true) ** 2, axis=1)
        r2_score = 1 - (residual_ss / total_ss)
        return r2_score

    def v_l2_loss(self):
        v_reg = vmap(self._zero_reg, in_axes=(0, None))
        return v_reg(self.state.params, 1)

    def v_score(self, x, y):
        y_pred = self.posterior(x)
        return self.v_mean_squared_error(y, y_pred), self.v_r2_score(y, y_pred)


class RashomonRobustEnsembleHybridJNN(EnsembleHybridJNN):
    """
    Hybrid JNN with Rashomon-set fitting: shrink set diameter by increasing
    regularization until diameter <= max_diameter. Rashomon set defined by
    validation (joint train+val) MSE.
    """

    def __init__(
        self,
        model_config,
        trainer_config,
        equation_config,
        max_diameter,
        h_lambda,
        rashomon_epsilon=0.01,
        regularizer="l2_net",
        seed=1,
        reduction_max_k=4,
        lambda_min=1e-10,
    ):
        super().__init__(model_config, trainer_config, equation_config, seed=seed)
        self.max_diameter = max_diameter
        self.h_lambda = h_lambda
        self.rashomon_epsilon = rashomon_epsilon
        self._initial_rashomon_epsilon = float(rashomon_epsilon)
        self._regularizer = regularizer
        self._current_lambda = 0.0
        self.lambda_runs = []
        self._reduction_max_k = int(reduction_max_k)
        self._lambda_min = float(lambda_min)

    def _effective_l2_net_decay(self):
        return self._current_lambda

    def _store_run(self, lam, k=None):
        rashomon_mask = np.zeros(self.ensemble_size, dtype=bool)
        rashomon_mask[np.array(self.rashomon_indices)] = True
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes=(0, 0, 0, 0)))
        train_vals, _ = v_monitor_loss_test(
            self.state_best.params, self.params_best, self.state_best, self.data_train
        )
        train_MSE = np.array(train_vals)
        run_dict = {
            "lambda": lam,
            "parameters": np.array(self.params_best),
            "net_params": copy.deepcopy(self.net_best),
            "in_rashomon": rashomon_mask,
            "best_idx": int(self.rashomon_best_idx),
            "diameter": self.rashomon_diameter,
            "min_performance": self.rashomon_best_val_loss,
            "train_MSE": train_MSE,
        }
        if k is not None:
            run_dict["k"] = int(k)
        self.lambda_runs.append(run_dict)

    def _print_rashomon_parameters(self):
        param_names = list(self.parameters)
        params = np.array(self.params_best)
        in_idx = np.array(self.rashomon_indices)
        out_mask = np.ones(self.ensemble_size, dtype=bool)
        out_mask[in_idx] = False
        out_idx = np.where(out_mask)[0]
        print(f"[Rashomon] Parameters IN set ({len(in_idx)} members):")
        if len(in_idx) > 0:
            for idx in in_idx:
                pvec = params[idx]
                pstr = "  ".join(
                    f"{k}={pvec[j]:.6g}" for j, k in enumerate(param_names)
                )
                print(f"    member {idx}: {pstr}")
        else:
            print("    (none)")
        print(f"[Rashomon] Parameters OUTSIDE set ({len(out_idx)} members):")
        if len(out_idx) > 0:
            for idx in out_idx:
                pvec = params[idx]
                pstr = "  ".join(
                    f"{k}={pvec[j]:.6g}" for j, k in enumerate(param_names)
                )
                print(f"    member {idx}: {pstr}")
        else:
            print("    (none)")

    def fit(self, X, y, noise_level=None, compute_delta=None):
        self._termination_case = None
        self._final_lambda = None
        self._noise_level = noise_level
        lam = 0.0
        print(f"[Rashomon] Setting {self._regularizer} lambda = {lam}")
        self._current_lambda = lam
        self._fit_one_run(X, y, warm_start=False)
        eps1, eps3, eps4 = self.compute_epsilon_estimates(noise_level=noise_level)
        if np.isfinite(eps4):
            self.rashomon_epsilon = float(eps4)
            print(
                f"[Rashomon] Epsilon: eps1={eps1:.6g}, eps3={eps3:.6g}, eps4=max(eps1,eps3)={self.rashomon_epsilon:.6g}"
            )
        else:
            print(
                f"[Rashomon] Epsilon not finite; using initial rashomon_epsilon={self.rashomon_epsilon}"
            )
        if compute_delta is not None:
            out = compute_delta(self.rashomon_epsilon)
            delta_val = out[0] if isinstance(out, (tuple, list)) else out
            if np.isfinite(delta_val) and delta_val > 0:
                self.max_diameter = float(delta_val)
                print(f"[Rashomon] max_diameter (= delta) = {self.max_diameter:.6g}")
        self.compute_rashomon_set(self.rashomon_epsilon)
        self._store_run(lam)
        n_in = int(np.sum(self.lambda_runs[-1]["in_rashomon"]))
        n_out = self.ensemble_size - n_in
        print(
            f"[Rashomon] diameter = {self.rashomon_diameter:.6f}  |  in set: {n_in}  |  outside: {n_out}"
        )
        self._print_rashomon_parameters()
        if self.rashomon_diameter <= self.max_diameter:
            print(
                f"[Rashomon] Terminating: diameter <= max_diameter ({self.max_diameter}) after lambda=0"
            )
            self._termination_case = 1
            self._final_lambda = 0.0
            self._current_lambda = 0.0
            return
        increase_lambdas = [10.0, 10.0**2, 10.0**3, 10.0**4, 10.0**5, 10.0**6]
        reached_small_diameter = False
        for inc_idx, lam in enumerate(increase_lambdas):
            print(
                f"[Rashomon] Increase step {inc_idx + 1}/{len(increase_lambdas)}: {self._regularizer} lambda = {lam}"
            )
            self._current_lambda = lam
            self._fit_one_run(X, y, warm_start=True)
            self.compute_rashomon_set(self.rashomon_epsilon)
            self._store_run(lam)
            n_in = int(np.sum(self.lambda_runs[-1]["in_rashomon"]))
            n_out = self.ensemble_size - n_in
            print(
                f"[Rashomon] diameter = {self.rashomon_diameter:.6f}  |  in set: {n_in}  |  outside: {n_out}"
            )
            self._print_rashomon_parameters()
            if self.rashomon_diameter <= self.max_diameter:
                reached_small_diameter = True
                print(f"[Rashomon] Diameter <= max_diameter; starting reduction phase")
                break
        if not reached_small_diameter:
            best_run_idx = int(np.argmin([r["diameter"] for r in self.lambda_runs]))
            best_run = self.lambda_runs[best_run_idx]
            self.net_best = best_run["net_params"]
            self.params_best = best_run["parameters"]
            self.state_best = self.state_best.replace(params=self.net_best)
            self.state = self.state.replace(params=self.net_best)
            self.rashomon_indices = np.where(best_run["in_rashomon"])[0]
            self.rashomon_diameter = float(best_run["diameter"])
            self.rashomon_best_val_loss = float(best_run["min_performance"])
            self.rashomon_best_idx = int(best_run["best_idx"])
            print(
                f"[Rashomon] Terminating: restored run with smallest diameter: lambda={best_run['lambda']}, diameter={best_run['diameter']:.6f}"
            )
            self._termination_case = 2
            self._final_lambda = float(best_run["lambda"])
            self._current_lambda = 0.0
            return
        lam = self.lambda_runs[-1]["lambda"]
        last_admissible_idx = len(self.lambda_runs) - 1
        k = 1
        max_k = self._reduction_max_k
        while k <= max_k:
            reduction_factor = 1.0 - 0.5**k
            lam_new = lam * reduction_factor
            if lam_new < self._lambda_min:
                self._termination_case = 3
                prev_run = self.lambda_runs[last_admissible_idx]
                self._final_lambda = float(prev_run["lambda"])
                print(
                    f"[Rashomon] lambda below minimum; keeping smallest admissible lambda = {self._final_lambda:.6g}"
                )
                self.net_best = prev_run["net_params"]
                self.params_best = prev_run["parameters"]
                self.state_best = self.state_best.replace(params=self.net_best)
                self.state = self.state.replace(params=self.net_best)
                self.rashomon_indices = np.where(prev_run["in_rashomon"])[0]
                self.rashomon_diameter = float(prev_run["diameter"])
                self.rashomon_best_val_loss = float(prev_run["min_performance"])
                self.rashomon_best_idx = int(prev_run["best_idx"])
                break
            print(
                f"[Rashomon] Reduction (k={k}): factor = 1 - 1/2^{k} = {reduction_factor:.4g}, {self._regularizer} lambda = {lam_new:.6g}"
            )
            self._current_lambda = lam_new
            self._fit_one_run(X, y, warm_start=True)
            self.compute_rashomon_set(self.rashomon_epsilon)
            self._store_run(lam_new, k=k)
            n_in = int(np.sum(self.lambda_runs[-1]["in_rashomon"]))
            n_out = self.ensemble_size - n_in
            print(
                f"[Rashomon] diameter = {self.rashomon_diameter:.6f}  |  in set: {n_in}  |  outside: {n_out}"
            )
            self._print_rashomon_parameters()
            if self.rashomon_diameter > self.max_diameter:
                print(
                    f"[Rashomon] diameter > max_diameter; restoring last admissible run"
                )
                prev_run = self.lambda_runs[last_admissible_idx]
                self.net_best = prev_run["net_params"]
                self.params_best = prev_run["parameters"]
                self.state_best = self.state_best.replace(params=self.net_best)
                self.state = self.state.replace(params=self.net_best)
                self.rashomon_indices = np.where(prev_run["in_rashomon"])[0]
                self.rashomon_diameter = float(prev_run["diameter"])
                self.rashomon_best_val_loss = float(prev_run["min_performance"])
                self.rashomon_best_idx = int(prev_run["best_idx"])
                lam = float(prev_run["lambda"])
                k += 1
                if k > max_k:
                    self._termination_case = 3
                    self._final_lambda = float(prev_run["lambda"])
                    print(f"[Rashomon] Terminating: k reached max_k={max_k}")
                    break
            else:
                last_admissible_idx = len(self.lambda_runs) - 1
                lam = lam_new
        self._current_lambda = 0.0


class EnsembleJNN(BaseEstimator, RegressorMixin):
    def __init__(self, model_config, trainer_config, equation_config, seed=1):
        # Initialize model, trainer, and equation configurations
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.equation_config = equation_config

        # Set seed for reproducibility
        self.seed = model_config.get("seed", seed)

        # Set model configuration
        self.model_config["layers"][0] = len(self.equation_config["input_variables"])
        self.equation = equation_config["type"]
        self.layers = self.model_config["layers"]
        self.final_nonlin = self.model_config["final_nonlin"]
        self.p = self.model_config["dropout_p"]

        # Regularization: only variance (l2_net)
        self.l2_net_decay = self.trainer_config.get("l2_net_decay", 0.0)

        self.split = self.trainer_config["split"]
        self.ensemble_size = self.model_config["ensemble_size"]
        self.iterations = self.trainer_config["iterations"]

        # Initialize network
        self.init_net, self.apply_net = create_network(self.model_config)

        # Initialize logs and counters
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.train_GT_log = []
        self.val_GT_log = []
        self.test_GT_log = []
        self.loss_test_log = [jnp.array(self.ensemble_size * [jnp.inf])]

    # Forward pass during training
    def net_forward(self, net_params, state, inputs):
        outputs, updated_states = self.state.apply_fn(
            {"params": net_params, "batch_stats": state.batch_stats},
            x=inputs,
            train=True,
            mutable=["batch_stats"],
        )
        return outputs, updated_states

    # Forward pass during testing
    def net_forward_test(self, net_params, state, inputs):
        outputs = self.state.apply_fn(
            {"params": net_params, "batch_stats": state.batch_stats},
            x=inputs,
            train=False,
        )
        return outputs

    # Variance (output) regularization only
    def l2_net_regularization(self, net_params, state, inputs, l2_net_decay):
        fa_outputs = self.state.apply_fn(
            {"params": net_params, "batch_stats": state.batch_stats},
            x=inputs,
            train=False,
        )
        return l2_net_decay * jnp.sqrt(jnp.sum(jnp.square(fa_outputs)))

    # Compute loss and update state
    def loss(self, net_params, state, batch):
        inputs, targets = batch
        outputs, updated_states = self.net_forward(net_params, state, inputs)
        regularization = self.l2_net_regularization(
            net_params, state, inputs, self.l2_net_decay
        )
        loss = jnp.mean((targets - outputs) ** 2) + regularization
        return loss, updated_states

    def loss_with_fixed_net_params(self, net_params, all_state, batch):
        loss_value, _ = self.loss(net_params, all_state, batch)
        return loss_value  # Only return the scalar loss value for linesearch

    # Compute and log loss during in training mode
    def monitor_loss(self, net_params, state, batch):
        inputs, targets = batch
        loss_value, _ = self.loss(net_params, state, batch)
        regularization = self.l2_net_regularization(
            net_params, state, inputs, self.l2_net_decay
        )
        RMSE_value = loss_value - regularization
        return RMSE_value, loss_value

    # Compute and log loss during in test mode
    def monitor_loss_test(self, net_params, state, batch):
        inputs, targets = batch
        outputs = self.net_forward_test(net_params, state, inputs)
        RMSE_value = jnp.mean((targets - outputs) ** 2)
        regularization = self.l2_net_regularization(
            net_params, state, inputs, self.l2_net_decay
        )
        loss_value = RMSE_value + regularization
        return RMSE_value, loss_value

    # Single optimization step
    @partial(jit, static_argnums=(0,))
    def step(self, i, state, batch):
        # Update the network parameters
        (l, updated_state), grads_net = jax.value_and_grad(
            self.loss, argnums=0, has_aux=True
        )(state.params, state, batch)
        state = state.apply_gradients(grads=grads_net)
        state = state.replace(batch_stats=updated_state["batch_stats"])

        return state

    # Functions for storing best states
    def update_best_weights(self, net, it, net_best, it_best, counter):
        counter = jnp.array(0, dtype=counter.dtype)
        return net, it, counter

    def keep_best_weights(self, net, it, net_best, it_best, counter):
        counter = counter + 1
        return net_best, it_best, counter

    def store_best(self, update, net, it, net_best, it_best, counter):
        return jax.lax.cond(
            update,
            self.update_best_weights,
            self.keep_best_weights,
            net,
            it,
            net_best,
            it_best,
            counter,
        )

    # Functions for reseting model upon decay
    def keep_current_weights(self, net, net_best):
        return net

    def restore_best_weights(self, net, net_best):
        return net_best

    def restore_best(self, decay, net, net_best):
        return jax.lax.cond(
            decay, self.restore_best_weights, self.keep_current_weights, net, net_best
        )

    # Functions for updating learning rates
    def update_lr(self, learning_rates, counter, decay_factor):
        learning_rates = learning_rates * decay_factor
        counter = jnp.array(0, dtype=counter.dtype)
        return learning_rates, counter

    def keep_lr(self, learning_rates, counter, decay_factor):
        return learning_rates, counter

    def decay_step(self, decay, learning_rates, counter, decay_factor):
        return jax.lax.cond(
            decay, self.update_lr, self.keep_lr, learning_rates, counter, decay_factor
        )

    def batch_normalize(self, data_val, norm_const):
        X, y = data_val

        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y

        return [X, y]

    # Full training loop
    def fit(self, X, y):
        rng_key = random.PRNGKey(self.seed)
        nIter = self.iterations

        # Initialize random keys
        rng_key1, rng_key2, self.rng_key_fit = random.split(rng_key, 3)
        (
            k1,
            k2,
        ) = random.split(rng_key1, 2)
        keys_1 = random.split(k1, self.ensemble_size)

        # Initialize network
        test_input = jnp.ones(X.shape)
        self.net = vmap(self.init_net, in_axes=(0, None, None))(
            keys_1, test_input, True
        )

        # Split variables into params and batch_stats
        self.net_params = self.net["params"]
        self.net_batch_stats = self.net["batch_stats"]

        # Define optimizer for network parameters (Adam only; no weight decay)
        schedule_net_params = self.trainer_config["init_lr_net"]
        self.optimizer_net_params = optax.chain(
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=schedule_net_params,
                weight_decay=0.0,
            )
        )

        self.state = vmap(
            lambda params, batch_stats: TrainState.create(
                apply_fn=self.apply_net,
                params=params,
                batch_stats=batch_stats,
                tx=self.optimizer_net_params,
            )
        )(self.net_params, self.net_batch_stats)

        # Potentially reformat y
        if len(y.shape) == 1:
            y = y[:, None]
        X, y = jnp.array(X), jnp.array(y)

        rng_key, rng_key_loader = random.split(self.rng_key_fit, 2)

        dataset = CustomLoader(
            X,
            y,
            self.trainer_config["batch_size"],
            self.ensemble_size,
            split=self.split,
            rng_key=rng_key_loader,
        )  # , normalize=True
        self.net_best = copy.deepcopy(self.net_params)

        self.it_best = jnp.zeros(self.ensemble_size)
        self.counter = jnp.zeros(self.ensemble_size)
        self.lr_net = jnp.ones(self.ensemble_size) * self.trainer_config["init_lr_net"]
        self.maximize = -1 if self.trainer_config["maximize"] else 1
        finalized = jnp.zeros(self.ensemble_size, dtype=bool)

        # Data normalization (as normalize=False, constanst are 0 and 1)
        data = iter(dataset)
        self.norm_const = dataset.norm_const
        (self.mu_X, self.sigma_X), (self.mu_y, self.sigma_y) = self.norm_const

        # Define vectorized optimization steps across the entire ensemble
        v_step = jit(vmap(self.step, in_axes=(None, 0, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes=(0, 0, 0)))
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes=(0, 0, 0)))
        v_store_best = vmap(self.store_best, in_axes=(0, 0, 0, 0, 0, 0))
        v_restore_best = vmap(self.restore_best, in_axes=(0, 0, 0))
        v_decay_step = vmap(self.decay_step, in_axes=(0, 0, 0, None))
        v_batch_normalize = vmap(self.batch_normalize, in_axes=(0, 0))

        data_train = v_batch_normalize(dataset.data_train, dataset.norm_const)
        data_val = v_batch_normalize(dataset.data_val, dataset.norm_const)

        # Track performance over nIter
        pbar = trange(nIter)
        for it in pbar:
            # Batch data sampling
            rng_key, *rng_keys = random.split(rng_key, self.ensemble_size + 1)
            batch = next(data)
            self.state = v_step(it, self.state, batch)

            RMSE_value, loss_value = v_monitor_loss(
                self.state.params, self.state, batch
            )
            self.loss_log.append(loss_value)

            # Normalization optional for this path.
            if it % 10 == 0:
                RMSE_test_value, loss_test_value = v_monitor_loss_test(
                    self.state.params, self.state, data_val
                )
                update = (jnp.array(self.loss_test_log) * self.maximize).min(
                    axis=0
                ) - self.trainer_config["tolerance"] > loss_test_value * self.maximize

                self.loss_test_log.append(loss_test_value)

                # Record training performance
                RMSE_train_value, loss_train_value = v_monitor_loss_test(
                    self.state.params, self.state, data_train
                )
                self.train_log.append(loss_train_value)
                RMSE_val_value, loss_val_value = v_monitor_loss_test(
                    self.state.params, self.state, data_val
                )
                self.val_log.append(loss_val_value)

                # Model selection
                self.net_best, self.it_best, self.counter = v_store_best(
                    update,
                    self.state.params,
                    jnp.ones(self.ensemble_size) * it,
                    self.net_best,
                    self.it_best,
                    self.counter,
                )
                self.state_best = self.state.replace(params=self.net_best)

                # Update LRs
                decay = jnp.array(self.trainer_config["patience"]) < self.counter
                decay = ~finalized * decay

                self.lr_net, self.counter = v_decay_step(
                    decay,
                    self.lr_net,
                    self.counter,
                    self.trainer_config["decay_factor"],
                )
                self.state.opt_state[0].hyperparams["learning_rate"] = self.lr_net

                # Reset models with decay
                self.net = v_restore_best(decay, self.state.params, self.net_best)
                self.state = self.state.replace(params=self.net)

                # Mark finalized trainings
                finalized = (
                    jnp.array(self.lr_net)
                    < self.trainer_config["init_lr_net"]
                    * self.trainer_config["decay_factor"]
                    ** self.trainer_config["decay_steps"]
                )

                best_test_RMSE, best_test_loss = v_monitor_loss_test(
                    self.state_best.params, self.state_best, data_val
                )
                pbar.set_postfix(
                    {
                        "Min/Max loss": f'{RMSE_value.min()}"/"{RMSE_value.max()}',
                        "Min/Max test loss": f'{RMSE_test_value.min()}"/"{RMSE_test_value.max()}',
                        "Min/Max best test loss": f'{best_test_RMSE.min()}"/"{best_test_RMSE.max()}',
                    }
                )

                if finalized.all():
                    break

        # set to best params
        self.state_best = self.state.replace(params=self.net_best)
        self.state = self.state.replace(params=self.net_best)

    # @partial(jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def posterior(self, x):
        normalize = vmap(lambda x, mu, std: (x - mu) / std, in_axes=(0, 0, 0))
        denormalize = vmap(lambda x, mu, std: x * std + mu, in_axes=(0, 0, 0))

        x = jnp.tile(x[jnp.newaxis, :, :], (self.ensemble_size, 1, 1))
        inputs = normalize(x, self.mu_X, self.sigma_X)

        samples = vmap(self.net_forward_test, (0, 0, 0))(
            self.state_best.params, self.state_best, inputs
        )
        samples = denormalize(samples, self.mu_y, self.sigma_y)

        return samples

    # @partial(jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def predict(self, x):
        # accepts and returns un-normalized data
        samples = self.posterior(x)
        return samples.mean(0).reshape(-1), samples.std(0)

    def score(self, x, y):
        y_pred = self.predict(x)
        return mean_squared_error(y, y_pred, squared=False), r2_score(y, y_pred)

    def v_mean_squared_error(self, y_true, y_pred):
        y_true = jnp.broadcast_to(y_true, y_pred.shape)
        squared_diff = (y_pred - y_true) ** 2
        mse = jnp.mean(squared_diff, axis=1)
        rmse = jnp.sqrt(mse)
        return rmse

    def v_r2_score(self, y_true, y_pred):
        y_true = jnp.broadcast_to(y_true, y_pred.shape)
        y_mean = jnp.mean(y_true, axis=1)
        total_ss = jnp.sum((y_true - y_mean[:, None]) ** 2, axis=1)
        residual_ss = jnp.sum((y_pred - y_true) ** 2, axis=1)
        r2_score = 1 - (residual_ss / total_ss)
        return r2_score

    def v_l2_loss(self):
        return np.zeros(self.ensemble_size)

    def v_score(self, x, y):
        y_pred = self.posterior(x)
        return self.v_mean_squared_error(y, y_pred), self.v_r2_score(y, y_pred)


class EnsemblePartialPhysics(BaseEstimator, RegressorMixin):
    def __init__(self, model_config, equation_config, seed=1):
        # Load configs
        self.model_config = model_config
        self.equation_config = equation_config
        self.parameters = equation_config["parameter_values"].keys()
        self.fp = equation_config["fp"]

        # Set seed for reproducibility
        self.seed = model_config.get("seed", seed)

        self.equation = equation_config["type"]
        self.fa_input = jnp.array(equation_config["fa_input"])
        self.ensemble_size = self.model_config["ensemble_size"]

        parameters = ", ".join(self.parameters)
        inputs = ", ".join(equation_config["input_variables"])
        fp_str = self.equation_config["type"].replace("fp", self.equation_config["fp"])

        a = symbols(parameters)
        if not isinstance(a, tuple):
            a = a
        x = symbols(inputs)
        if not isinstance(x, tuple):
            x = x
        fp = sympify(fp_str)

        self.fp_numeric = lambdify((*a, *x), fp, modules="numpy")

    def loss(self, params, X, y):
        return np.mean(
            (self.fp_numeric(*params, *[X[:, i] for i in range(X.shape[1])]) - y) ** 2
        )

    def fit(self, X, y):
        np.random.seed(self.seed)
        orandom.seed(self.seed)

        if "initial_guess" in self.equation_config.keys():
            self.params = np.random.normal(
                size=(self.ensemble_size, len(self.parameters))
            )
            self.params = np.array(
                list(self.equation_config["uncertainty"].values())
            ).reshape(1, -1) * self.params + np.array(
                list(self.equation_config["initial_guess"].values())
            ).reshape(
                1, -1
            )
        else:
            self.params = np.random.uniform(
                size=(self.ensemble_size, len(self.parameters)), low=-10, high=10
            )

        def loss_fixed(params):
            return self.loss(params, X, y)

        self.params_best = np.zeros((self.ensemble_size, len(self.parameters)))
        self.results = []

        for i in range(self.ensemble_size):
            res = minimize(loss_fixed, self.params[i], method="L-BFGS-B")
            self.params_best[i] = res.x
            self.results.append(res)

        self.it_best = [res["nit"] for res in self.results]

    # @partial(jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def posterior(self, x):
        x = [x[:, i].reshape(1, -1) for i in range(x.shape[1])]
        params = [
            np.array(self.params_best[:, i]).reshape(-1, 1)
            for i in range(len(self.parameters))
        ]
        return self.fp_numeric(*params, *x)

    def predict(self, x):
        samples = self.posterior(x)
        return samples.mean(0).reshape(-1), samples.std(0)

    def score(self, x, y):
        y_pred, *_ = self.predict(x)
        return mean_squared_error(y, y_pred, squared=False), r2_score(y, y_pred)

    def v_mean_squared_error(self, y_true, y_pred):
        y_true = jnp.broadcast_to(y_true, y_pred.shape)
        squared_diff = (y_pred - y_true) ** 2
        mse = jnp.mean(squared_diff, axis=1)
        rmse = jnp.sqrt(mse)
        return rmse

    def v_r2_score(self, y_true, y_pred):
        y_true = jnp.broadcast_to(y_true, y_pred.shape)
        y_mean = jnp.mean(y_true, axis=1)
        total_ss = jnp.sum((y_true - y_mean[:, None]) ** 2, axis=1)
        residual_ss = jnp.sum((y_pred - y_true) ** 2, axis=1)
        r2_score = 1 - (residual_ss / total_ss)
        return r2_score

    def v_l2_loss(self):
        v_l2_regularization = np.zeros(self.ensemble_size)
        return v_l2_regularization

    def v_score(self, x, y):
        y_pred = self.posterior(x)
        return self.v_mean_squared_error(y, y_pred), self.v_r2_score(y, y_pred)


class LearningRateSchedule:
    def __init__(self, initial_lr):
        self.lr = initial_lr

    def get_lr(self):
        return self.lr

    def set_lr(self, new_lr):
        self.lr = new_lr

    def __call__(self):
        return self.lr  # This callable returns the current learning rate
