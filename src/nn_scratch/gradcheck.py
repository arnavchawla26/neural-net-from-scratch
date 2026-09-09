"""Numerical gradient checking: the standard way to verify a hand-derived
backward pass without trusting it on faith. For a scalar function f(x) and
a claimed analytic gradient, the centered finite-difference estimate

    d f/dx_i ~= (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)

should match the analytic gradient to a small relative error. This module
is used by the test suite (checking every layer, activation, and loss in
isolation) and is also exposed via the CLI so a user can sanity-check the
math themselves.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Centered finite-difference gradient of scalar function f at x."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + eps
        f_plus = f(x)

        x[idx] = original - eps
        f_minus = f(x)

        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """Max elementwise relative error, robust to both being ~0."""
    denom = np.maximum(np.abs(analytic) + np.abs(numerical), 1e-12)
    return float(np.max(np.abs(analytic - numerical) / denom))


def check_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    analytic_grad: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> tuple[bool, float]:
    """Return (passed, relative_error) comparing analytic_grad to a
    numerically-estimated gradient of f at x."""
    numerical = numerical_gradient(f, x, eps=eps)
    err = relative_error(analytic_grad, numerical)
    return err < tol, err
