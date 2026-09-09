"""`nnscratch` command-line interface.

Subcommands
-----------
train      Train an MLP on a synthetic dataset (spiral or moons), print
           per-epoch progress, report test accuracy, optionally save the
           trained weights and/or an ASCII decision-boundary plot.
gradcheck  Numerically verify the hand-derived backward pass of a freshly
           initialized network against finite-difference gradients.
predict    Load a saved model and predict classes for point(s) given on
           the command line.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import data as data_mod
from .gradcheck import check_gradient
from .losses import CrossEntropyLoss
from .network import MLP


def _build_dataset(name: str, seed: int):
    if name == "spiral":
        X, y = data_mod.make_spiral(n_per_class=100, n_classes=3, seed=seed)
        n_classes = 3
    elif name == "moons":
        X, y = data_mod.make_moons(n_samples=300, seed=seed)
        n_classes = 2
    else:
        raise ValueError(f"Unknown dataset {name!r}")
    return X, y, n_classes


def _ascii_decision_boundary(
    model: MLP, X: np.ndarray, y: np.ndarray, width: int = 60, height: int = 24
) -> str:
    """Render the model's predicted class regions as an ASCII grid, with
    the actual training points overlaid as digits, purely as a dependency-
    free (no matplotlib) way to eyeball whether training worked."""
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3

    xs = np.linspace(x_min, x_max, width)
    ys = np.linspace(y_min, y_max, height)
    grid = np.array([[gx, gy] for gy in ys for gx in xs])
    preds = model.predict(grid).reshape(height, width)

    palette = " .:-=+*#%@"
    n_classes = int(preds.max()) + 1
    symbols = [palette[min(c * (len(palette) - 1) // max(n_classes - 1, 1), len(palette) - 1)] for c in range(n_classes)]

    # rasterize training points on top of the region symbols
    canvas = [[symbols[preds[row, col]] for col in range(width)] for row in range(height)]
    for (px, py), label in zip(X, y):
        col = int(round((px - x_min) / (x_max - x_min) * (width - 1)))
        row = int(round((py - y_min) / (y_max - y_min) * (height - 1)))
        if 0 <= row < height and 0 <= col < width:
            canvas[row][col] = str(int(label))

    return "\n".join("".join(row) for row in reversed(canvas))


def cmd_train(args: argparse.Namespace) -> int:
    X, y, n_classes = _build_dataset(args.dataset, seed=args.seed)
    X_train, X_test, y_train, y_test = data_mod.train_test_split(
        X, y, test_frac=0.2, seed=args.seed
    )

    layer_sizes = [2, *args.hidden, n_classes]
    model = MLP(layer_sizes, hidden_activation=args.activation, seed=args.seed)

    print(f"dataset={args.dataset}  layers={layer_sizes}  activation={args.activation}")
    print(f"train={X_train.shape[0]}  test={X_test.shape[0]}")

    history = model.fit(
        X_train,
        y_train,
        loss_name="cross_entropy",
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        l2_reg=args.l2,
        X_val=X_test,
        y_val=y_test,
        seed=args.seed,
        verbose=True,
        log_every=max(args.epochs // 10, 1),
    )

    test_acc = model.accuracy(X_test, y_test)
    print(f"\nfinal train loss: {history['loss'][-1]:.4f}")
    print(f"final test accuracy: {test_acc:.4f}")

    if args.plot:
        print("\ndecision boundary (region = predicted class, digit = true label):\n")
        print(_ascii_decision_boundary(model, X_train, y_train))

    if args.save:
        model.save(args.save)
        print(f"\nsaved model to {args.save}")

    return 0


def cmd_gradcheck(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    layer_sizes = [args.in_dim, *args.hidden, args.out_dim]
    model = MLP(layer_sizes, hidden_activation=args.activation, seed=args.seed)
    loss_fn = CrossEntropyLoss()

    X = rng.normal(size=(args.batch, args.in_dim))
    y = rng.integers(0, args.out_dim, size=args.batch)

    def full_loss(_ignored=None) -> float:
        logits = model.forward(X)
        return loss_fn.forward(logits, y)

    # analytic pass: populates every layer's dW/db
    logits = model.forward(X)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())

    print(f"gradient check: layers={layer_sizes} activation={args.activation} batch={args.batch}")
    all_passed = True
    for i, layer in enumerate(model.dense_layers):
        for name, param, analytic in (("W", layer.W, layer.dW), ("b", layer.b, layer.db)):
            passed, err = check_gradient(lambda _p: full_loss(), param, analytic, tol=args.tol)
            status = "PASS" if passed else "FAIL"
            all_passed &= passed
            print(f"  layer {i} {name:>2s}  rel_error={err:.2e}  {status}")

    if all_passed:
        print("\nall gradients verified against numerical estimates.")
    else:
        print("\nSOME GRADIENTS FAILED numerical verification.", file=sys.stderr)
    return 0 if all_passed else 1


def cmd_predict(args: argparse.Namespace) -> int:
    model = MLP.load(args.model)
    points = np.array(args.points, dtype=np.float64).reshape(-1, model.layer_sizes[0])
    preds = model.predict(points)
    probs = model.predict_proba(points)
    for point, pred, prob in zip(points, preds, probs):
        coords = ", ".join(f"{c:.3f}" for c in point)
        print(f"({coords}) -> class {pred}  probs={np.round(prob, 4).tolist()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nnscratch", description="Train and inspect a from-scratch NumPy MLP."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train an MLP on a synthetic dataset")
    p_train.add_argument("--dataset", choices=["spiral", "moons"], default="spiral")
    p_train.add_argument("--hidden", type=int, nargs="+", default=[16, 16])
    p_train.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    p_train.add_argument("--optimizer", choices=["sgd", "momentum"], default="momentum")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=0.05)
    p_train.add_argument("--momentum", type=float, default=0.9)
    p_train.add_argument("--l2", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--plot", action="store_true", help="print an ASCII decision-boundary plot")
    p_train.add_argument("--save", type=str, default=None, help="path to save trained weights (.npz)")
    p_train.set_defaults(func=cmd_train)

    p_gc = sub.add_parser("gradcheck", help="numerically verify the backward pass")
    p_gc.add_argument("--in-dim", type=int, default=4)
    p_gc.add_argument("--out-dim", type=int, default=3)
    p_gc.add_argument("--hidden", type=int, nargs="+", default=[5, 4])
    p_gc.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="tanh")
    p_gc.add_argument("--batch", type=int, default=6)
    p_gc.add_argument("--tol", type=float, default=1e-4)
    p_gc.add_argument("--seed", type=int, default=0)
    p_gc.set_defaults(func=cmd_gradcheck)

    p_pred = sub.add_parser("predict", help="predict classes with a saved model")
    p_pred.add_argument("model", type=str, help="path to a .npz saved by `train --save`")
    p_pred.add_argument(
        "points",
        type=float,
        nargs="+",
        help="flattened point coordinates, e.g. 0.1 0.2 -0.5 0.4 for two 2D points",
    )
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
