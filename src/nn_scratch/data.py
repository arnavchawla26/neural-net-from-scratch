"""Synthetic classification datasets, generated on the fly (no external
downloads, no dataset dependency) so the project has zero non-NumPy
runtime dependencies.

The flagship dataset is the classic multi-class "spiral" used throughout
neural-network teaching material (e.g. CS231n): K arms of points spiraling
outward from the origin, each arm one class. It is not linearly separable,
so a plain linear (softmax) classifier tops out well below 100% accuracy
and a hidden layer is required to solve it -- which makes it a good
end-to-end demo for a from-scratch backprop implementation.
"""
from __future__ import annotations

import numpy as np


def make_spiral(
    n_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the classic K-armed 2D spiral classification dataset.

    Returns
    -------
    X : (n_per_class * n_classes, 2) float64 array of point coordinates.
    y : (n_per_class * n_classes,) int64 array of class labels in
        [0, n_classes).
    """
    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)

    for class_idx in range(n_classes):
        ix = range(n_per_class * class_idx, n_per_class * (class_idx + 1))
        # radius grows linearly from ~0 to 1 along each arm
        r = np.linspace(0.0, 1.0, n_per_class)
        # angle sweeps `4` radians (~0.64 turns) per class, offset so each
        # class's arm starts where the previous one's ended -- the standard
        # CS231n-style spiral formula. An earlier version of this function
        # multiplied `t` by an extra 2.5 "for more texture"; that extra
        # rotation was never validated against anything and, it turns out,
        # over-rotates each arm into the next class's territory, destroying
        # separability -- caught by training a network on the data (test
        # accuracy stuck at ~35-70% no matter how the optimizer was tuned)
        # and confirmed by reverting to the textbook formula below, which
        # trains to >=95% test accuracy in a couple hundred epochs. Left as
        # a cautionary example: don't embellish a well-known synthetic
        # generator without checking the embellishment against a trained
        # model.
        t = (
            np.linspace(class_idx * 4, (class_idx + 1) * 4, n_per_class)
            + rng.normal(0.0, noise, n_per_class)
        )
        X[ix] = np.column_stack([r * np.sin(t), r * np.cos(t)])
        y[ix] = class_idx

    return X, y


def make_moons(
    n_samples: int = 200,
    noise: float = 0.15,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the classic two-interleaving-half-circles binary dataset."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    theta_out = np.linspace(0, np.pi, n_out)
    outer_x = np.cos(theta_out)
    outer_y = np.sin(theta_out)

    theta_in = np.linspace(0, np.pi, n_in)
    inner_x = 1 - np.cos(theta_in)
    inner_y = 1 - np.sin(theta_in) - 0.5

    X = np.column_stack(
        [np.concatenate([outer_x, inner_x]), np.concatenate([outer_y, inner_y])]
    )
    X += rng.normal(0.0, noise, X.shape)
    y = np.concatenate([np.zeros(n_out, dtype=np.int64), np.ones(n_in, dtype=np.int64)])
    return X, y


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int | None = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split (X, y) into train/test subsets."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
