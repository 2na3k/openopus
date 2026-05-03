"""Tests for neural network kernel."""

import pytest

mlx = pytest.importorskip("mlx")

import mlx.core as mx  # noqa: E402

from openopus.nn import MLP, swish  # noqa: E402


def test_swish() -> None:
    x = mx.array([0.0, 1.0, -1.0])
    y = swish(x)
    assert y.shape == (3,)
    assert y.dtype == mx.float32


def test_mlp_forward() -> None:
    model = MLP(dims=[4, 8, 2])
    x = mx.random.normal(shape=(2, 4))
    y = model(x)
    assert y.shape == (2, 2)


def test_mlp_shapes() -> None:
    model = MLP(dims=[16, 32, 16, 4])
    x = mx.random.normal(shape=(1, 16))
    y = model(x)
    assert y.shape == (1, 4)


def test_mlp_parameters() -> None:
    model = MLP(dims=[3, 5, 2])
    params = model.parameters()
    assert "layers" in params
    assert len(params["layers"]) == 2
    assert params["layers"][0]["weight"].shape == (5, 3)
    assert params["layers"][1]["weight"].shape == (2, 5)
