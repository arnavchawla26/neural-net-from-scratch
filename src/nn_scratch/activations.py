"""Activation functions with hand-derived forward and backward passes.

Every activation follows the same tiny interface used throughout this
project (see layers.py): `forward(x)` computes and caches what backward
needs, and `backward(grad_output)` returns d(loss)/d(input) given
d(loss)/d(output), via the chain rule -- no autodiff, every derivative
below was worked out by hand.
"""
from __future__ import annotations

import numpy as np


class Activation:
    """Base class: stateful forward/backward pair over a batch of inputs."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []


class ReLU(Activation):
    """f(x) = max(0, x); f'(x) = 1 if x > 0 else 0 (subgradient 0 at x=0)."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("backward() called before forward()")
        return grad_output * self._mask


class Sigmoid(Activation):
    """f(x) = 1 / (1 + e^-x); f'(x) = f(x) * (1 - f(x))."""

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # clip to avoid overflow in exp for very negative x
        out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
            np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))),
        )
        self._out = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("backward() called before forward()")
        return grad_output * self._out * (1.0 - self._out)


class Tanh(Activation):
    """f(x) = tanh(x); f'(x) = 1 - tanh(x)^2."""

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self._out = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("backward() called before forward()")
        return grad_output * (1.0 - self._out**2)


class Softmax(Activation):
    """Row-wise softmax. Not meant to be paired with backward() directly in
    training: when used as the output layer for classification, pair it
    with `CrossEntropyLoss`, whose `backward()` already returns
    d(loss)/d(pre-softmax logits) in closed form (softmax_output - one_hot),
    which is far simpler and more numerically stable than composing
    Softmax.backward with a separate cross-entropy backward. Softmax's own
    backward() (the full Jacobian-vector product) is provided for
    completeness / gradient-checking in isolation.
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        out = exp / np.sum(exp, axis=1, keepdims=True)
        self._out = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Full Jacobian-vector product per row: for row s (softmax output)
        # and upstream gradient g, dL/dx = s * (g - sum(g * s)).
        if self._out is None:
            raise RuntimeError("backward() called before forward()")
        s = self._out
        dot = np.sum(grad_output * s, axis=1, keepdims=True)
        return s * (grad_output - dot)


ACTIVATIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}


def get_activation(name: str) -> Activation:
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown activation {name!r}; choices are {sorted(ACTIVATIONS)}"
        ) from exc
