import numpy as np
import pytest

from nn_scratch.activations import ReLU, Sigmoid, Tanh, Softmax, get_activation
from nn_scratch.gradcheck import check_gradient


@pytest.mark.parametrize("cls", [ReLU, Sigmoid, Tanh, Softmax])
def test_activation_gradients_match_numerical(cls):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 4))
    # ReLU has a kink at 0 where finite differences are unreliable; nudge
    # away from exactly 0 so the numerical check is well-defined.
    x = np.where(np.abs(x) < 1e-2, x + 0.5, x)
    upstream = rng.normal(size=(5, 4))

    act = cls()

    def f(_ignored=None) -> float:
        return float(np.sum(upstream * act.forward(x)))

    out = act.forward(x)
    grad = act.backward(upstream)
    assert out.shape == x.shape
    assert grad.shape == x.shape

    # f's closure reads `x` directly, so `x` (not a copy) must be the array
    # check_gradient perturbs -- see the comment in test_layers.py for why.
    passed, err = check_gradient(f, x, grad, eps=1e-6, tol=1e-3)
    assert passed, f"{cls.__name__} backward relative error too high: {err}"


def test_relu_zeros_negative_inputs():
    act = ReLU()
    out = act.forward(np.array([[-1.0, 0.0, 2.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.0, 2.0]])


def test_sigmoid_output_in_unit_interval():
    act = Sigmoid()
    out = act.forward(np.array([[-100.0, 0.0, 100.0]]))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    np.testing.assert_allclose(out[0, 1], 0.5)


def test_tanh_output_in_range():
    act = Tanh()
    out = act.forward(np.array([[-50.0, 0.0, 50.0]]))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-12)


def test_softmax_rows_sum_to_one_and_are_positive():
    act = Softmax()
    out = act.forward(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(np.sum(out, axis=1), [1.0, 1.0], atol=1e-10)
    assert np.all(out > 0)


def test_backward_before_forward_raises_for_all_activations():
    for cls in (ReLU, Sigmoid, Tanh, Softmax):
        act = cls()
        with pytest.raises(RuntimeError):
            act.backward(np.ones((1, 2)))


def test_get_activation_known_and_unknown():
    assert isinstance(get_activation("relu"), ReLU)
    assert isinstance(get_activation("SIGMOID"), Sigmoid)
    with pytest.raises(ValueError):
        get_activation("not_a_real_activation")
