"""Tests for LSTM kernel."""

import pytest

pytest.importorskip("mlx")

import mlx.core as mx  # noqa: E402

from openopus.nn import LSTM, LSTMCell  # noqa: E402


def test_lstm_cell_shapes() -> None:
    cell = LSTMCell(input_size=4, hidden_size=8)
    x = mx.random.normal(shape=(4,))
    h = mx.zeros((8,))
    c = mx.zeros((8,))
    h_next, c_next = cell(x, h, c)
    assert h_next.shape == (8,)
    assert c_next.shape == (8,)


def test_lstm_cell_parameters() -> None:
    cell = LSTMCell(input_size=3, hidden_size=5)
    params = cell.parameters()
    assert "x_weights" in params
    assert "h_weights" in params
    assert params["x_weights"].shape == (20, 3)
    assert params["h_weights"].shape == (20, 5)


def test_lstm_forward() -> None:
    model = LSTM(input_size=4, hidden_size=8, num_layers=1)
    x = mx.random.normal(shape=(2, 3, 4))
    output, (hn, cn) = model(x)
    assert output.shape == (2, 3, 8)
    assert hn.shape == (1, 2, 8)
    assert cn.shape == (1, 2, 8)


def test_lstm_multi_layer() -> None:
    model = LSTM(input_size=4, hidden_size=8, num_layers=2)
    x = mx.random.normal(shape=(1, 5, 4))
    output, (hn, cn) = model(x)
    assert output.shape == (1, 5, 8)
    assert hn.shape == (2, 1, 8)
    assert cn.shape == (2, 1, 8)


def test_lstm_with_state() -> None:
    model = LSTM(input_size=2, hidden_size=3, num_layers=1)
    x = mx.random.normal(shape=(1, 4, 2))
    h0 = mx.random.normal(shape=(1, 1, 3))
    c0 = mx.random.normal(shape=(1, 1, 3))
    output, (hn, cn) = model(x, h0, c0)
    assert output.shape == (1, 4, 3)
