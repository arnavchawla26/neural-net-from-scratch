"""Gradient-descent optimizers. Each mutates parameter arrays in place
(numpy's `-=` keeps the same array object, so a layer's `self.W` stays a
live view of the same memory the optimizer updates -- no need to copy
updated values back into the layer).
"""
from __future__ import annotations

import numpy as np


class SGD:
    """Vanilla mini-batch stochastic gradient descent.

    theta <- theta - lr * grad
    """

    def __init__(self, lr: float = 0.1) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must be the same length")
        for p, g in zip(params, grads):
            p -= self.lr * g


class SGDMomentum:
    """SGD with classical (heavy-ball) momentum.

    v <- momentum * v - lr * grad
    theta <- theta + v

    Momentum accumulates a running velocity so consistent gradient
    directions accelerate and oscillating ones damp out, which typically
    speeds convergence and helps escape shallow local structure compared
    to plain SGD -- at the cost of one extra float array of state per
    parameter tensor.
    """

    def __init__(self, lr: float = 0.1, momentum: float = 0.9) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[int, np.ndarray] = {}

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must be the same length")
        for p, g in zip(params, grads):
            key = id(p)
            v = self._velocity.get(key)
            if v is None or v.shape != p.shape:
                v = np.zeros_like(p)
            v = self.momentum * v - self.lr * g
            self._velocity[key] = v
            p += v


OPTIMIZERS = {"sgd": SGD, "momentum": SGDMomentum}


def get_optimizer(name: str, **kwargs):
    try:
        return OPTIMIZERS[name.lower()](**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer {name!r}; choices are {sorted(OPTIMIZERS)}"
        ) from exc
