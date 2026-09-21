from jax import grad, vmap, random, jit
from jax import numpy as jnp
import jax
import flax.linen as nn
from typing import Sequence


def create_network(model_config):
    layers = model_config["layers"]
    final_nonlin = model_config["final_nonlin"]

    model = MLP_flax(layers[1:], dropout_rate=0, final_nonlin=final_nonlin)
    return model.init, model.apply


class MLP_flax(nn.Module):
    features: list[int]
    dropout_rate: float
    deterministic: bool = False
    final_nonlin: bool = False

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
        self.batch_norms = [
            nn.BatchNorm(use_running_average=None) for _ in self.features
        ]
        # self.dropout = nn.Dropout(rate=self.dropout_rate, deterministic=None)
        if self.final_nonlin:
            self.final_activation = nn.softplus
        else:
            self.final_activation = lambda x: x

    def __call__(self, x, train: bool):
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            if i < len(self.layers) - 1:
                x = layer(x)
                x = bn(x, use_running_average=not train)
                x = nn.relu(x)
            else:
                x = layer(x)
        x = self.final_activation(x)
        return x
