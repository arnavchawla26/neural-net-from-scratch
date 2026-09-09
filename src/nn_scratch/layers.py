"""A fully-connected (dense/affine) layer with a hand-derived backward pass.

Forward:   Y = X @ W + b            X: (N, in), W: (in, out), b: (out,)
Backward, given dL/dY (grad_output), by the matrix-calculus chain rule:
    dL/dW = X.T @ dL/dY
    dL/db = sum over the batch axis of dL/dY
    dL/dX = dL/dY @ W.T
These three identities are the entire content of backprop through a linear
layer; everything else in this project composes them with activation and
loss derivatives.
"""
from __future__ import annotations

import numpy as np


class Dense:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int | None = None,
        weight_init: str = "he",
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        rng = np.random.default_rng(seed)
        if weight_init == "he":
            # He (2015) init: good default when followed by ReLU.
            scale = np.sqrt(2.0 / in_features)
        elif weight_init == "xavier":
            # Xavier/Glorot init: good default for tanh/sigmoid/softmax.
            scale = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"Unknown weight_init {weight_init!r}")

        self.W = rng.normal(0.0, scale, size=(in_features, out_features))
        self.b = np.zeros(out_features)

        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward() called before forward()")
        self.dW = self._x.T @ grad_output
        self.db = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        if self.dW is None or self.db is None:
            raise RuntimeError("grads() called before backward()")
        return [self.dW, self.db]
