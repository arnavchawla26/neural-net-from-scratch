"""Loss functions with hand-derived gradients w.r.t. their input.

Both losses use *mean* reduction over the batch, so the returned gradient
is already correctly scaled for a plain (non-batch-size-adjusted) SGD
update -- this matters: a sum-reduction loss's gradient would need to be
divided by batch size separately, and forgetting to do so is a classic
bug where larger batches silently blow up the effective learning rate.
"""
from __future__ import annotations

import numpy as np


class MSELoss:
    """Mean squared error: L = mean((y_pred - y_true)^2).

    dL/dy_pred = 2 * (y_pred - y_true) / N, N = number of elements
    averaged over (batch * output_dims), matching np.mean's own
    normalization so forward and backward stay consistent.
    """

    def __init__(self) -> None:
        self._diff: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
            )
        self._diff = y_pred - y_true
        return float(np.mean(self._diff**2))

    def backward(self) -> np.ndarray:
        if self._diff is None:
            raise RuntimeError("backward() called before forward()")
        return 2.0 * self._diff / self._diff.size


class CrossEntropyLoss:
    """Softmax + negative-log-likelihood, fused for numerical stability.

    Takes raw *logits* (pre-softmax scores), not probabilities -- the
    network's final Dense layer should feed straight into this loss with
    no Softmax activation in between. Internally:

        p = softmax(logits)
        L = -mean_i( log(p[i, y_true[i]]) )

    The famously clean gradient of softmax+cross-entropy combined is
        dL/dlogits = (p - one_hot(y_true)) / N
    which is *not* the same as separately computing Softmax.backward and
    then a plain NLL backward and composing them (that also works, via the
    chain rule, but is more code and less numerically stable near 0/1
    probabilities) -- this fused form is the standard textbook shortcut,
    derived by expanding d/dz_k[-log(softmax(z)_y)] and observing most
    terms cancel.
    """

    def __init__(self) -> None:
        self._probs: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        if logits.ndim != 2:
            raise ValueError("logits must be a 2D (batch, classes) array")
        if y_true.shape[0] != logits.shape[0]:
            raise ValueError("y_true batch size must match logits")

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self._probs = probs
        self._y_true = y_true

        n = logits.shape[0]
        correct_class_probs = probs[np.arange(n), y_true]
        # clip to avoid log(0) if a prediction is exactly saturated
        log_likelihood = -np.log(np.clip(correct_class_probs, 1e-12, 1.0))
        return float(np.mean(log_likelihood))

    def backward(self) -> np.ndarray:
        if self._probs is None or self._y_true is None:
            raise RuntimeError("backward() called before forward()")
        n = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(n), self._y_true] -= 1.0
        return grad / n

    def predict_proba(self) -> np.ndarray:
        """Return the softmax probabilities computed on the last forward()."""
        if self._probs is None:
            raise RuntimeError("forward() has not been called yet")
        return self._probs


LOSSES = {"mse": MSELoss, "cross_entropy": CrossEntropyLoss}


def get_loss(name: str):
    try:
        return LOSSES[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown loss {name!r}; choices are {sorted(LOSSES)}") from exc
