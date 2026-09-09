import numpy as np
import pytest

from nn_scratch.layers import Dense
from nn_scratch.gradcheck import check_gradient


def _projected_loss(layer: Dense, x: np.ndarray, upstream: np.ndarray):
    """L = <upstream, forward(x)>  ->  dL/dx = backward(upstream)."""

    def f(_ignored=None) -> float:
        return float(np.sum(upstream * layer.forward(x)))

    return f


@pytest.mark.parametrize("in_f,out_f,batch", [(3, 4, 5), (1, 1, 1), (6, 2, 8)])
def test_dense_gradients_match_numerical(in_f, out_f, batch):
    rng = np.random.default_rng(42)
    layer = Dense(in_f, out_f, seed=1)
    x = rng.normal(size=(batch, in_f))
    upstream = rng.normal(size=(batch, out_f))

    out = layer.forward(x)
    assert out.shape == (batch, out_f)
    grad_x = layer.backward(upstream)
    assert grad_x.shape == x.shape

    f = _projected_loss(layer, x, upstream)

    # IMPORTANT: check_gradient perturbs the exact array object it is given
    # and re-evaluates f on it, so f's closure must read *that same array*
    # (here: `x`, `layer.W`, `layer.b`) rather than a `.copy()` of it --
    # perturbing a copy while f keeps reading the original silently checks
    # nothing (numerical gradient comes back ~0, a bug caught during
    # development: every check here failed with relative error == 1.0
    # until the `.copy()` calls below were removed).
    passed, err = check_gradient(f, x, grad_x)
    assert passed, f"dL/dx relative error too high: {err}"

    passed, err = check_gradient(f, layer.W, layer.dW)
    assert passed, f"dL/dW relative error too high: {err}"

    passed, err = check_gradient(f, layer.b, layer.db)
    assert passed, f"dL/db relative error too high: {err}"


def test_dense_forward_is_affine():
    layer = Dense(2, 2, seed=0)
    layer.W = np.array([[1.0, 0.0], [0.0, 1.0]])
    layer.b = np.array([1.0, -1.0])
    x = np.array([[2.0, 3.0]])
    out = layer.forward(x)
    np.testing.assert_allclose(out, [[3.0, 2.0]])


def test_dense_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        Dense(0, 4)
    with pytest.raises(ValueError):
        Dense(4, -1)


def test_dense_backward_before_forward_raises():
    layer = Dense(2, 2)
    with pytest.raises(RuntimeError):
        layer.backward(np.ones((1, 2)))


def test_dense_grads_before_backward_raises():
    layer = Dense(2, 2)
    layer.forward(np.ones((1, 2)))
    with pytest.raises(RuntimeError):
        layer.grads()


def test_dense_unknown_weight_init_raises():
    with pytest.raises(ValueError):
        Dense(2, 2, weight_init="nonsense")
