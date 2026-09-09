import numpy as np
import pytest

from nn_scratch.optim import SGD, SGDMomentum, get_optimizer


def test_sgd_step_matches_hand_computed_update():
    w = np.array([1.0, 2.0])
    b = np.array([0.5])
    dw = np.array([0.1, -0.2])
    db = np.array([0.05])

    opt = SGD(lr=0.1)
    opt.step([w, b], [dw, db])

    np.testing.assert_allclose(w, [1.0 - 0.1 * 0.1, 2.0 - 0.1 * -0.2])
    np.testing.assert_allclose(b, [0.5 - 0.1 * 0.05])


def test_sgd_rejects_bad_lr():
    with pytest.raises(ValueError):
        SGD(lr=0.0)
    with pytest.raises(ValueError):
        SGD(lr=-1.0)


def test_sgd_rejects_mismatched_lengths():
    opt = SGD(lr=0.1)
    with pytest.raises(ValueError):
        opt.step([np.zeros(2)], [np.zeros(2), np.zeros(2)])


def test_momentum_matches_hand_computed_two_steps():
    w = np.array([0.0])
    opt = SGDMomentum(lr=0.1, momentum=0.9)

    # step 1: v = 0.9*0 - 0.1*1.0 = -0.1; w += v -> w = -0.1
    opt.step([w], [np.array([1.0])])
    np.testing.assert_allclose(w, [-0.1])

    # step 2: v = 0.9*(-0.1) - 0.1*1.0 = -0.19; w += v -> w = -0.29
    opt.step([w], [np.array([1.0])])
    np.testing.assert_allclose(w, [-0.29])


def test_momentum_rejects_bad_momentum():
    with pytest.raises(ValueError):
        SGDMomentum(momentum=1.0)
    with pytest.raises(ValueError):
        SGDMomentum(momentum=-0.1)


def test_momentum_accelerates_beyond_plain_sgd_for_consistent_gradient():
    """With a constant gradient direction, momentum's cumulative step should
    exceed plain SGD's after a few iterations -- that's the entire point of
    momentum, so verify it actually happens rather than assuming it does."""
    w_sgd = np.array([0.0])
    w_mom = np.array([0.0])
    sgd = SGD(lr=0.1)
    mom = SGDMomentum(lr=0.1, momentum=0.9)

    for _ in range(5):
        sgd.step([w_sgd], [np.array([1.0])])
        mom.step([w_mom], [np.array([1.0])])

    assert abs(w_mom[0]) > abs(w_sgd[0])


def test_get_optimizer_known_and_unknown():
    assert isinstance(get_optimizer("sgd", lr=0.1), SGD)
    assert isinstance(get_optimizer("momentum", lr=0.1, momentum=0.5), SGDMomentum)
    with pytest.raises(ValueError):
        get_optimizer("adam_but_not_implemented")
