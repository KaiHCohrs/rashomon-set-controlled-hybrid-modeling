"""
Tools for loading a synthetic or real dataset.
"""

import numpy as onp

from pathlib import Path
import pandas as pd
from torch.utils import data
from jax import vmap, random, jit
from jax import numpy as jnp
from functools import partial
import random as orandom


class CustomLoader(data.Dataset):
    def __init__(
        self,
        X,
        y,
        batch_size=128,
        ensemble_size=32,
        split=0.8,
        rng_key=random.PRNGKey(1234),
        normalize=False,
    ):
        # Initialize the CustomLoader with dataset and configuration.
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.split = split
        self.key = rng_key
        self.normalize = normalize

        if self.N < self.batch_size:
            self.batch_size = self.N

        # Split the dataset for each ensemble member
        keys = random.split(rng_key, ensemble_size)
        if split < 1:
            self.data_train, self.data_val = vmap(self.__split, (None, None, 0))(
                X, y, keys
            )
            (self.X_train, self.y_train) = self.data_train
        else:
            self.data_train, self.data_val = vmap(self.__train_only, (None, None, 0))(
                X, y, keys
            )
            (self.X_train, self.y_train) = self.data_train

        # Compute normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0, 0))(
            self.X_train, self.y_train
        )

    def normalization_constants(self, X, y):
        if self.normalize:
            mu_X, sigma_X = X.mean(0), X.std(0)
            mu_y, sigma_y = jnp.zeros(
                y.shape[1],
            ), jnp.abs(
                y
            ).max(0) * jnp.ones(
                y.shape[1],
            )
        else:
            mu_X, sigma_X = 0, 1
            mu_y, sigma_y = 0, 1

        return (mu_X, sigma_X), (mu_y, sigma_y)

    def __split(self, X, y, key):
        # Split data into training and validation sets
        idx = random.choice(key, self.N, (self.N,), replace=False)
        idx_train = idx[: jnp.floor(self.N * self.split).astype(int)]
        idx_test = idx[jnp.floor(self.N * self.split).astype(int) :]

        inputs_train = X[idx_train, :]
        targets_train = y[idx_train, :]

        inputs_test = X[idx_test, :]
        targets_test = y[idx_test, :]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    def __train_only(self, X, y, key):
        idx = random.choice(key, self.N, (self.N,), replace=False).sort()

        inputs_train = X[idx]
        targets_train = y[idx]

        inputs_test = X[idx]
        targets_test = y[idx]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, X, y, norm_const):
        # Generate a batch of normalized data
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        X = X[idx, :]
        y = y[idx, :]
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y
        return X, y

    def __getitem__(self, index):
        # Generate one batch of data
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, targets = vmap(self.__data_generation, (0, 0, 0, 0))(
            keys, self.X_train, self.y_train, self.norm_const
        )
        return inputs, targets


class CustomBootstrapLoader(data.Dataset):
    """Loader with train/val split per ensemble member (used by Rashomon routine)."""

    def __init__(
        self,
        X,
        y,
        batch_size=128,
        ensemble_size=32,
        split=0.8,
        rng_key=random.PRNGKey(1234),
        normalize=False,
    ):
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.split = split
        self.key = rng_key
        self.normalize = normalize
        if self.N < self.batch_size:
            self.batch_size = self.N
        keys = random.split(rng_key, ensemble_size)
        if split < 1:
            self.data_train, self.data_val = vmap(self._bootstrap, (None, None, 0))(
                X, y, keys
            )
            (self.X_train, self.y_train) = self.data_train
        else:
            self.data_train, self.data_val = vmap(
                self._bootstrap_train_only, (None, None, 0)
            )(X, y, keys)
            (self.X_train, self.y_train) = self.data_train
        self.norm_const = vmap(self.normalization_constants, in_axes=(0, 0))(
            self.X_train, self.y_train
        )

    def normalization_constants(self, X, y):
        if self.normalize:
            mu_X, sigma_X = X.mean(0), X.std(0)
            mu_y, sigma_y = (
                jnp.zeros(y.shape[1]),
                jnp.abs(y).max(0) * jnp.ones(y.shape[1]),
            )
        else:
            mu_X, sigma_X = 0, 1
            mu_y, sigma_y = 0, 1
        return (mu_X, sigma_X), (mu_y, sigma_y)

    def _bootstrap(self, X, y, key):
        idx = random.choice(key, self.N, (self.N,), replace=False)
        idx_train = idx[: int(jnp.floor(self.N * self.split))]
        idx_test = idx[int(jnp.floor(self.N * self.split)) :]
        inputs_train = X[idx_train, :]
        targets_train = y[idx_train, :]
        inputs_test = X[idx_test, :]
        targets_test = y[idx_test, :]
        return (inputs_train, targets_train), (inputs_test, targets_test)

    def _bootstrap_train_only(self, X, y, key):
        idx = random.choice(key, self.N, (self.N,), replace=False)
        idx = jnp.sort(idx)
        inputs_train = X[idx, :]
        targets_train = y[idx, :]
        inputs_test = X[idx, :]
        targets_test = y[idx, :]
        return (inputs_train, targets_train), (inputs_test, targets_test)

    @partial(jit, static_argnums=(0,))
    def _data_generation(self, key, X, y, norm_const):
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        n = X.shape[0]
        batch_size = min(self.batch_size, n)
        idx = random.choice(key, n, (batch_size,), replace=False)
        X_b = X[idx, :]
        y_b = y[idx, :]
        X_b = (X_b - mu_X) / sigma_X
        y_b = (y_b - mu_y) / sigma_y
        return X_b, y_b

    def __getitem__(self, index):
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, targets = vmap(self._data_generation, (0, 0, 0, 0))(
            keys, self.X_train, self.y_train, self.norm_const
        )
        return inputs, targets
