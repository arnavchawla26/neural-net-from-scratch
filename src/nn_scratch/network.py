"""The MLP itself: stacks Dense layers and activations, and drives forward
and backward passes by literally calling each sub-module's forward/backward
in sequence/reverse -- there is no autodiff graph, this loop *is* backprop.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .activations import Activation, get_activation
from .layers import Dense
from .losses import CrossEntropyLoss, MSELoss, get_loss
from .optim import get_optimizer


class MLP:
    """A feed-forward multilayer perceptron.

    `layer_sizes = [n_in, h1, h2, ..., n_out]` describes a Dense layer
    between each consecutive pair, so `len(layer_sizes) - 1` Dense layers
    total. `hidden_activation` is applied after every Dense layer except
    the last (the output layer is left as raw logits/linear output --
    for classification, pair with CrossEntropyLoss, which applies softmax
    internally; for regression, pair with MSELoss directly on the linear
    output).
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: str = "relu",
        seed: int | None = None,
        weight_init: str | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least [n_in, n_out]")
        if weight_init is None:
            weight_init = "he" if hidden_activation == "relu" else "xavier"

        rng = np.random.default_rng(seed)
        self.dense_layers: list[Dense] = []
        self.activations: list[Activation] = []
        for i in range(len(layer_sizes) - 1):
            layer_seed = int(rng.integers(0, 2**31 - 1))
            self.dense_layers.append(
                Dense(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    seed=layer_seed,
                    weight_init=weight_init,
                )
            )
            is_last = i == len(layer_sizes) - 2
            if not is_last:
                self.activations.append(get_activation(hidden_activation))

        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for i, dense in enumerate(self.dense_layers):
            out = dense.forward(out)
            if i < len(self.activations):
                out = self.activations[i].forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for i in range(len(self.dense_layers) - 1, -1, -1):
            if i < len(self.activations):
                grad = self.activations[i].backward(grad)
            grad = self.dense_layers[i].backward(grad)
        return grad

    def params(self) -> list[np.ndarray]:
        out = []
        for layer in self.dense_layers:
            out.extend(layer.params())
        return out

    def grads(self) -> list[np.ndarray]:
        out = []
        for layer in self.dense_layers:
            out.extend(layer.grads())
        return out

    def weight_params(self) -> list[np.ndarray]:
        """Just the W matrices (not biases) -- used for L2 regularization,
        which by convention is not applied to bias terms."""
        return [layer.W for layer in self.dense_layers]

    # -- persistence ---------------------------------------------------
    def save(self, path: str) -> None:
        arrays = {}
        for i, layer in enumerate(self.dense_layers):
            arrays[f"W{i}"] = layer.W
            arrays[f"b{i}"] = layer.b
        np.savez(
            path,
            layer_sizes=np.array(self.layer_sizes),
            hidden_activation=self.hidden_activation,
            **arrays,
        )

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path, allow_pickle=False)
        layer_sizes = data["layer_sizes"].tolist()
        hidden_activation = str(data["hidden_activation"])
        model = cls(layer_sizes, hidden_activation=hidden_activation)
        for i, layer in enumerate(model.dense_layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        return model

    # -- training --------------------------------------------------------
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Classification only: softmax over the network's raw output."""
        logits = self.forward(x)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Classification only: argmax class prediction."""
        return np.argmax(self.forward(x), axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_name: str = "cross_entropy",
        optimizer_name: str = "sgd",
        epochs: int = 100,
        batch_size: int | None = 32,
        lr: float = 0.1,
        momentum: float = 0.9,
        l2_reg: float = 0.0,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        seed: int | None = None,
        verbose: bool = False,
        log_every: int = 10,
    ) -> dict:
        """Train with mini-batch gradient descent. Returns a history dict
        with per-epoch train loss (and accuracy / val loss / val accuracy
        when the loss is classification-shaped / X_val is given).
        """
        loss_fn = get_loss(loss_name)
        is_classification = isinstance(loss_fn, CrossEntropyLoss)
        if isinstance(loss_fn, MSELoss) and y.ndim == 1:
            y = y.reshape(-1, 1)

        if optimizer_name == "momentum":
            optimizer = get_optimizer("momentum", lr=lr, momentum=momentum)
        else:
            optimizer = get_optimizer(optimizer_name, lr=lr)

        rng = np.random.default_rng(seed)
        n = X.shape[0]
        bs = batch_size or n

        history: dict[str, list[float]] = {"loss": []}
        if is_classification:
            history["accuracy"] = []
        if X_val is not None and y_val is not None:
            history["val_loss"] = []
            if is_classification:
                history["val_accuracy"] = []

        for epoch in range(epochs):
            perm = rng.permutation(n)
            X_shuffled, y_shuffled = X[perm], y[perm]

            epoch_losses = []
            for start in range(0, n, bs):
                end = start + bs
                xb, yb = X_shuffled[start:end], y_shuffled[start:end]

                logits = self.forward(xb)
                batch_loss = loss_fn.forward(logits, yb)

                if l2_reg > 0:
                    reg = sum(np.sum(w**2) for w in self.weight_params())
                    batch_loss = batch_loss + 0.5 * l2_reg * reg

                grad = loss_fn.backward()
                self.backward(grad)

                if l2_reg > 0:
                    for layer in self.dense_layers:
                        layer.dW = layer.dW + l2_reg * layer.W

                optimizer.step(self.params(), self.grads())
                epoch_losses.append(batch_loss)

            mean_loss = float(np.mean(epoch_losses))
            history["loss"].append(mean_loss)
            if is_classification:
                history["accuracy"].append(self.accuracy(X, y))

            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val)
                val_loss = loss_fn.forward(val_logits, y_val)
                history["val_loss"].append(val_loss)
                if is_classification:
                    history["val_accuracy"].append(self.accuracy(X_val, y_val))

            if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                msg = f"epoch {epoch:4d}  loss={mean_loss:.4f}"
                if is_classification:
                    msg += f"  acc={history['accuracy'][-1]:.4f}"
                if "val_loss" in history:
                    msg += f"  val_loss={history['val_loss'][-1]:.4f}"
                print(msg)

        return history
