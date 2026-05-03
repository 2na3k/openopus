"""Simple neural network kernel implementations using MLX."""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn


def swish(x: mx.array) -> mx.array:
    """Swish activation function: x * sigmoid(x)."""
    return x * mx.sigmoid(x)


class MLP(nn.Module):  # type: ignore[misc]
    """A simple multi-layer perceptron kernel.

    Args:
        dims: List of layer dimensions (input, hidden..., output).
        activation: Activation function between layers (default: swish).
    """

    def __init__(
        self,
        dims: list[int],
        activation: Callable[[mx.array], mx.array] = swish,
    ) -> None:
        super().__init__()

        self.layers: list[nn.Linear] = []
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(layer)

        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
