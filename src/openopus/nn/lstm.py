"""LSTM kernel implementation using MLX primitives."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class LSTMCell(nn.Module):  # type: ignore[misc]
    """A single LSTM cell implemented from scratch.

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden state features.
        bias: Whether to use bias terms (default: True).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_weights = mx.random.uniform(low=-0.1, high=0.1, shape=(4 * hidden_size, input_size))
        self.h_weights = mx.random.uniform(low=-0.1, high=0.1, shape=(4 * hidden_size, hidden_size))

        self.bias: Optional[mx.array] = None
        if bias:
            self.bias = mx.zeros((4 * hidden_size,))

    def __call__(
        self,
        x: mx.array,
        h: mx.array,
        c: mx.array,
    ) -> tuple[mx.array, mx.array]:
        gates = x @ self.x_weights.T + h @ self.h_weights.T
        if self.bias is not None:
            gates = gates + self.bias

        chunk_size = self.hidden_size
        i = mx.sigmoid(gates[..., :chunk_size])
        f = mx.sigmoid(gates[..., chunk_size : 2 * chunk_size])
        g = mx.tanh(gates[..., 2 * chunk_size : 3 * chunk_size])
        o = mx.sigmoid(gates[..., 3 * chunk_size :])

        c = f * c + i * g
        h = o * mx.tanh(c)

        return h, c


class LSTM(nn.Module):  # type: ignore[misc]
    """Multi-layer LSTM kernel.

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden state features per layer.
        num_layers: Number of recurrent layers (default: 1).
        bias: Whether to use bias terms (default: True).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.cells: list[LSTMCell] = []
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            cell = LSTMCell(in_dim, hidden_size, bias=bias)
            self.cells.append(cell)

    def __call__(
        self,
        x: mx.array,
        h0: Optional[mx.array] = None,
        c0: Optional[mx.array] = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        batch, seq_len, _ = x.shape
        num_layers = len(self.cells)
        hidden_size = self.cells[0].hidden_size

        if h0 is None:
            h0 = mx.zeros((num_layers, batch, hidden_size))
        if c0 is None:
            c0 = mx.zeros((num_layers, batch, hidden_size))

        h_state = [h0[layer] for layer in range(num_layers)]
        c_state = [c0[layer] for layer in range(num_layers)]
        outputs: list[mx.array] = []

        for t in range(seq_len):
            xt = x[:, t, :]
            for layer in range(num_layers):
                ht, ct = self.cells[layer](xt, h_state[layer], c_state[layer])
                h_state[layer] = ht
                c_state[layer] = ct
                xt = ht
            outputs.append(xt)

        output = mx.stack(outputs, axis=1)
        hn = mx.stack(h_state, axis=0)
        cn = mx.stack(c_state, axis=0)
        return output, (hn, cn)
