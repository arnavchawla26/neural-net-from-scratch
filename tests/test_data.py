import numpy as np
import pytest

from nn_scratch.data import make_spiral, make_moons, train_test_split


def test_make_spiral_shapes_and_labels():
    X, y = make_spiral(n_per_class=50, n_classes=3, seed=1)
    assert X.shape == (150, 2)
    assert y.shape == (150,)
    assert set(np.unique(y).tolist()) == {0, 1, 2}
    assert np.sum(y == 0) == 50
    assert np.sum(y == 1) == 50
    assert np.sum(y == 2) == 50


def test_make_spiral_deterministic_with_seed():
    X1, y1 = make_spiral(n_per_class=20, n_classes=3, seed=99)
    X2, y2 = make_spiral(n_per_class=20, n_classes=3, seed=99)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_make_spiral_different_seeds_differ():
    X1, _ = make_spiral(n_per_class=20, seed=1)
    X2, _ = make_spiral(n_per_class=20, seed=2)
    assert not np.allclose(X1, X2)


def test_make_spiral_rejects_bad_args():
    with pytest.raises(ValueError):
        make_spiral(n_per_class=0)
    with pytest.raises(ValueError):
        make_spiral(n_classes=0)


def test_make_moons_shapes_and_labels():
    X, y = make_moons(n_samples=100, seed=0)
    assert X.shape[0] == 100
    assert X.shape[1] == 2
    assert set(np.unique(y).tolist()) == {0, 1}


def test_make_moons_rejects_bad_args():
    with pytest.raises(ValueError):
        make_moons(n_samples=0)


def test_train_test_split_proportions_and_no_overlap():
    # Use a synthetic dataset with guaranteed-unique rows (spiral data is
    # NOT unique here: every arm starts at r=0, i.e. the exact point (0, 0),
    # so all n_classes arms share that one duplicate row by construction --
    # discovered when this test first used make_spiral and failed on a
    # spurious "overlap" that was actually the shared origin point).
    n = 300
    X = np.column_stack([np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64) * 2])
    y = np.arange(n) % 3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_frac=0.25, seed=1)

    assert X_train.shape[0] + X_test.shape[0] == n
    assert X_test.shape[0] == round(n * 0.25)
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    train_set = {tuple(row) for row in X_train}
    test_set = {tuple(row) for row in X_test}
    assert train_set.isdisjoint(test_set)


def test_spiral_arms_share_the_origin_point():
    """Documents a real property of make_spiral worth knowing before relying
    on row-uniqueness elsewhere: every class's arm starts at radius 0, so
    the exact point (0, 0) appears once per class."""
    X, y = make_spiral(n_per_class=50, n_classes=3, seed=0)
    origin_count = int(np.sum(np.all(np.isclose(X, 0.0, atol=1e-9), axis=1)))
    assert origin_count == 3


def test_train_test_split_deterministic_with_seed():
    X, y = make_spiral(n_per_class=30, seed=0)
    a = train_test_split(X, y, test_frac=0.2, seed=42)
    b = train_test_split(X, y, test_frac=0.2, seed=42)
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_train_test_split_rejects_bad_fraction():
    X, y = make_spiral(n_per_class=10, seed=0)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_frac=0.0)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_frac=1.0)
