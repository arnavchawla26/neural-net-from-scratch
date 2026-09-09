import numpy as np
import pytest

from nn_scratch.losses import MSELoss, CrossEntropyLoss, get_loss
from nn_scratch.gradcheck import check_gradient


def test_mse_forward_known_value():
    loss = MSELoss()
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([[1.0, 0.0], [0.0, 4.0]])
    # squared diffs: 0, 4, 9, 0 -> mean = 13/4
    val = loss.forward(y_pred, y_true)
    assert val == pytest.approx(13.0 / 4.0)


def test_mse_gradient_matches_numerical():
    rng = np.random.default_rng(3)
    y_pred = rng.normal(size=(6, 3))
    y_true = rng.normal(size=(6, 3))
    loss = MSELoss()

    def f(pred):
        return loss.forward(pred, y_true)

    val = loss.forward(y_pred, y_true)
    grad = loss.backward()
    assert isinstance(val, float)

    passed, err = check_gradient(f, y_pred.copy(), grad)
    assert passed, f"MSE gradient relative error too high: {err}"


def test_mse_shape_mismatch_raises():
    loss = MSELoss()
    with pytest.raises(ValueError):
        loss.forward(np.zeros((2, 3)), np.zeros((2, 4)))


def test_mse_backward_before_forward_raises():
    loss = MSELoss()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_cross_entropy_matches_known_value_for_confident_correct_prediction():
    loss = CrossEntropyLoss()
    # huge logit on the correct class -> probability ~1 -> loss ~0
    logits = np.array([[50.0, 0.0, 0.0]])
    y_true = np.array([0])
    val = loss.forward(logits, y_true)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_cross_entropy_uniform_logits_gives_log_k():
    loss = CrossEntropyLoss()
    logits = np.zeros((4, 5))
    y_true = np.array([0, 1, 2, 3])
    val = loss.forward(logits, y_true)
    assert val == pytest.approx(np.log(5), abs=1e-8)


def test_cross_entropy_gradient_matches_numerical():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(7, 4))
    y_true = rng.integers(0, 4, size=7)
    loss = CrossEntropyLoss()

    def f(lg):
        return loss.forward(lg, y_true)

    loss.forward(logits, y_true)
    grad = loss.backward()

    passed, err = check_gradient(f, logits.copy(), grad)
    assert passed, f"CrossEntropy gradient relative error too high: {err}"


def test_cross_entropy_rejects_1d_logits():
    loss = CrossEntropyLoss()
    with pytest.raises(ValueError):
        loss.forward(np.zeros(5), np.array([0]))


def test_cross_entropy_backward_before_forward_raises():
    loss = CrossEntropyLoss()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_get_loss_known_and_unknown():
    assert isinstance(get_loss("mse"), MSELoss)
    assert isinstance(get_loss("cross_entropy"), CrossEntropyLoss)
    with pytest.raises(ValueError):
        get_loss("nonsense")
