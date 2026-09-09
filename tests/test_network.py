import numpy as np
import pytest
import tempfile
import os

from nn_scratch.network import MLP
from nn_scratch.losses import CrossEntropyLoss, MSELoss
from nn_scratch.gradcheck import check_gradient
from nn_scratch.data import make_spiral, train_test_split


def test_forward_shape():
    model = MLP([3, 5, 4, 2], seed=0)
    x = np.random.default_rng(0).normal(size=(7, 3))
    out = model.forward(x)
    assert out.shape == (7, 2)


def test_rejects_too_short_layer_sizes():
    with pytest.raises(ValueError):
        MLP([5])


def test_full_network_gradient_matches_numerical():
    """The strongest correctness check: verify every W and b in a multi-
    layer network against finite differences of the *actual* end-to-end
    loss, not just each component in isolation."""
    rng = np.random.default_rng(11)
    model = MLP([4, 6, 5, 3], hidden_activation="tanh", seed=2)
    loss_fn = CrossEntropyLoss()

    X = rng.normal(size=(9, 4))
    y = rng.integers(0, 3, size=9)

    def full_loss(_ignored=None) -> float:
        logits = model.forward(X)
        return loss_fn.forward(logits, y)

    logits = model.forward(X)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())

    for i, layer in enumerate(model.dense_layers):
        passed, err = check_gradient(lambda _p: full_loss(), layer.W, layer.dW, tol=1e-3)
        assert passed, f"layer {i} dW relative error too high: {err}"
        passed, err = check_gradient(lambda _p: full_loss(), layer.b, layer.db, tol=1e-3)
        assert passed, f"layer {i} db relative error too high: {err}"


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_full_network_gradient_all_activations(activation):
    rng = np.random.default_rng(21)
    model = MLP([3, 4, 2], hidden_activation=activation, seed=1)
    loss_fn = CrossEntropyLoss()
    X = rng.normal(size=(5, 3)) + 0.3  # nudge off ReLU's kink at 0
    y = rng.integers(0, 2, size=5)

    def full_loss(_ignored=None) -> float:
        return loss_fn.forward(model.forward(X), y)

    loss_fn.forward(model.forward(X), y)
    model.backward(loss_fn.backward())

    for layer in model.dense_layers:
        passed, err = check_gradient(lambda _p: full_loss(), layer.W, layer.dW, tol=1e-3)
        assert passed, f"{activation} dW relative error too high: {err}"


def test_regression_network_gradient_with_mse():
    rng = np.random.default_rng(33)
    model = MLP([3, 4, 1], hidden_activation="tanh", seed=4)
    loss_fn = MSELoss()
    X = rng.normal(size=(6, 3))
    y = rng.normal(size=(6, 1))

    def full_loss(_ignored=None) -> float:
        return loss_fn.forward(model.forward(X), y)

    loss_fn.forward(model.forward(X), y)
    model.backward(loss_fn.backward())

    for layer in model.dense_layers:
        passed, err = check_gradient(lambda _p: full_loss(), layer.W, layer.dW, tol=1e-3)
        assert passed, f"regression dW relative error too high: {err}"


def test_training_reduces_loss_and_solves_spiral():
    """End-to-end: a from-scratch MLP should comfortably solve the spiral
    dataset (which a linear model cannot), and loss should trend down."""
    X, y = make_spiral(n_per_class=100, n_classes=3, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_frac=0.2, seed=0)

    model = MLP([2, 16, 16, 3], hidden_activation="relu", seed=0)
    # lr=0.05 (not the more aggressive 0.5 tried first): with momentum=0.9
    # and He-initialized 16-16 ReLU layers, lr=0.5 measurably diverges on
    # this dataset (train loss *increases* run over run instead of
    # decreasing) -- found by sweeping lr and watching actual loss curves,
    # not assumed from the loss decreasing at all in the first few epochs.
    history = model.fit(
        X_train,
        y_train,
        loss_name="cross_entropy",
        optimizer_name="momentum",
        epochs=300,
        batch_size=32,
        lr=0.05,
        momentum=0.9,
        l2_reg=1e-4,
        seed=0,
    )

    assert history["loss"][-1] < history["loss"][0]
    # loss should mostly trend down: compare early vs late average
    early = np.mean(history["loss"][:10])
    late = np.mean(history["loss"][-10:])
    assert late < early

    test_acc = model.accuracy(X_test, y_test)
    assert test_acc > 0.9, f"expected >90% test accuracy on spiral, got {test_acc}"


def test_linear_model_cannot_solve_spiral():
    """Sanity check that the spiral dataset is genuinely non-linear: a
    model with *no* hidden layer (pure softmax regression) should do
    meaningfully worse than the hidden-layer model above."""
    X, y = make_spiral(n_per_class=100, n_classes=3, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_frac=0.2, seed=0)

    linear_model = MLP([2, 3], seed=0)
    linear_model.fit(
        X_train, y_train, epochs=300, batch_size=32, lr=0.05, optimizer_name="momentum", seed=0
    )
    linear_acc = linear_model.accuracy(X_test, y_test)
    assert linear_acc < 0.9


def test_save_and_load_round_trip_predictions():
    rng = np.random.default_rng(0)
    model = MLP([3, 5, 2], seed=0)
    X = rng.normal(size=(4, 3))
    preds_before = model.predict(X)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        model.save(path)
        loaded = MLP.load(path)

    preds_after = loaded.predict(X)
    np.testing.assert_array_equal(preds_before, preds_after)
    np.testing.assert_allclose(model.forward(X), loaded.forward(X))


def test_accuracy_helper():
    model = MLP([2, 2], seed=0)
    model.dense_layers[0].W = np.array([[10.0, -10.0], [-10.0, 10.0]])
    model.dense_layers[0].b = np.zeros(2)
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    assert model.accuracy(X, y) == 1.0
