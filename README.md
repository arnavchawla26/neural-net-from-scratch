# neural-net-from-scratch

A small multilayer perceptron (MLP) with **hand-derived NumPy backprop** —
no autodiff framework (no PyTorch, no TensorFlow, no JAX). Every forward
pass and every gradient in this repo was worked out by hand and is
verified against numerical (finite-difference) gradients in the test
suite, not just eyeballed.

This continues a "from-scratch ML" series: earlier repos in the same
series implement a [decision tree](https://github.com/arnavchawla26/decision-tree-from-scratch),
[naive Bayes classifier](https://github.com/arnavchawla26/naive-bayes-text-classifier),
[k-means](https://github.com/arnavchawla26/kmeans-clustering-from-scratch),
[word2vec](https://github.com/arnavchawla26/word2vec-from-scratch), and
[gradient-boosted trees](https://github.com/arnavchawla26/gradient-boosted-trees-from-scratch)
without their usual libraries either.

## What it does

- A `Dense` (fully-connected/affine) layer with a hand-derived forward
  (`Y = X @ W + b`) and backward pass (the three matrix-calculus
  identities `dW = X.T @ dY`, `db = sum(dY)`, `dX = dY @ W.T`).
- Activations — ReLU, Sigmoid, Tanh, Softmax — each with its own
  hand-derived derivative.
- Losses — mean squared error (regression) and a fused softmax +
  cross-entropy (classification), the latter using the classic
  `dL/dlogits = (softmax(logits) - one_hot(y)) / N` shortcut instead of
  composing two separate backward passes.
- An `MLP` class that stacks `Dense` + activation layers, runs forward
  and backward passes through the whole stack, and trains with mini-batch
  SGD (plain or with classical momentum), optional L2 weight decay, and
  a train/validation history.
- Two synthetic, dependency-free datasets generated on the fly (no
  downloads, no scikit-learn): the classic multi-arm "spiral" (not
  linearly separable — needs a hidden layer to solve) and "two moons".
- A **numerical gradient checker** (`gradcheck.py`): every layer,
  activation, loss, and the end-to-end network is tested by comparing its
  analytic gradient against a centered finite-difference estimate. This
  is the actual proof that the hand-derived math is correct, not a
  test that merely checks the code runs.
- A CLI (`nnscratch`) for training, gradient-checking, and predicting.

## Tech stack

Python 3.10+, NumPy only for the library itself (`pytest`/`pyflakes` for
development). No autodiff framework, no ML framework, no dataset
dependency — everything the network trains on is generated in
`src/nn_scratch/data.py`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# train an MLP on the spiral dataset and print an ASCII decision-boundary plot
nnscratch train --dataset spiral --hidden 16 16 --plot

# train on the two-moons dataset and save the weights
nnscratch train --dataset moons --save model.npz

# predict with a saved model
nnscratch predict model.npz 0.1 0.2 -0.3 0.5

# numerically verify the backward pass of a freshly initialized network
nnscratch gradcheck --hidden 6 4 --activation tanh

# run the test suite (includes gradient checks for every component)
pytest
```

Library usage:

```python
from nn_scratch import MLP
from nn_scratch.data import make_spiral, train_test_split

X, y = make_spiral(n_per_class=100, n_classes=3, seed=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=0)

model = MLP([2, 16, 16, 3], hidden_activation="relu", seed=0)
history = model.fit(
    X_train, y_train,
    loss_name="cross_entropy", optimizer_name="momentum",
    epochs=200, batch_size=32, lr=0.05, momentum=0.9, l2_reg=1e-4,
)
print(model.accuracy(X_test, y_test))  # ~0.90-0.97 depending on seed
```

## Design notes / things that went wrong during development (and how they were caught)

- **The synthetic spiral generator initially had a bug that silently
  capped test accuracy around 35-70% no matter how training was tuned.**
  An early version multiplied the angle parameter by an extra `2.5` "for
  more texture" on top of the standard CS231n-style spiral formula. That
  extra rotation was never checked against anything and, it turns out,
  over-rotates each class's arm into the next class's territory,
  destroying separability. It was caught by actually training a network
  on the data and sweeping learning rates: loss plateaued near `ln(3)`
  (chance level for 3 classes) or, worse, *increased* over training no
  matter the learning rate. Reverting to the textbook formula (no `2.5`
  factor) immediately let the same architecture reach 95%+ test accuracy
  in a couple hundred epochs, confirming the dataset — not the network —
  had been the problem. See the comment in `data.py::make_spiral`.
- **A too-aggressive default learning rate (`lr=0.5` with momentum=0.9)
  measurably diverged** on the (now-fixed) spiral dataset with He-init
  ReLU layers — train loss *increased* over hundreds of epochs instead of
  decreasing. Found the same way: by watching the actual loss curve
  rather than assuming a couple of early epochs of improvement meant
  training was healthy. A learning-rate sweep settled on `lr=0.05` as the
  default, which converges cleanly to >90% test accuracy.
- **A gradient-checking test-design mistake** (not a bug in the actual
  backprop math): several tests originally checked a parameter's gradient
  by passing `layer.W.copy()` into the numerical gradient checker while
  the loss function being differentiated still read the *original*
  `layer.W` through a closure. Perturbing the copy therefore had zero
  effect on the measured loss, so every affected test's "numerical"
  gradient came back ~0 and failed with relative error exactly `1.0`
  against the (correct) analytic gradient. Fixed by perturbing the exact
  array object the forward pass reads, not a detached copy — see the
  comment in `tests/test_layers.py`. This is a good example of why
  gradient checking a whole network end-to-end (`tests/test_network.py`)
  matters in addition to checking each component in isolation: the
  end-to-end check is what would catch a case like this if the isolated
  checks were ever silently broken again.
- **`make_spiral`'s own points aren't all unique**: every class's arm
  starts at radius 0 — i.e. the exact point `(0, 0)` — so with `K`
  classes, `(0, 0)` appears `K` times, once per class. A train/test-split
  test that assumed spiral rows were unique (to check for train/test
  overlap) failed on this shared origin point, not on an actual bug in
  `train_test_split`. Fixed by testing the overlap property on a
  synthetic dataset with guaranteed-unique rows instead, and added a
  small test documenting the origin-sharing property so it doesn't
  surprise anyone again (`test_spiral_arms_share_the_origin_point`).

## Verification performed

- `pytest` (68 tests): numerical gradient checks for `Dense`, every
  activation, every loss, and the full network end-to-end (both
  classification and regression heads, across ReLU/Tanh/Sigmoid); hand-
  computed optimizer update checks for SGD and momentum; dataset shape/
  determinism/train-test-split checks; an end-to-end training test that
  asserts loss trends down and the network reaches >90% test accuracy on
  the spiral dataset; a linear-model sanity check confirming the spiral
  data is genuinely non-linear (a hidden-layer-free model does
  meaningfully worse); model save/load round-trip; CLI smoke tests for
  `train`, `gradcheck`, and `predict`.
- `pyflakes` clean over `src/` and `tests/`.
- Fresh-venv install (`pip install -e ".[dev]"`) followed by a full
  `pytest` run, matching what a new clone would see.
- Manual CLI runs of all three subcommands (`train --plot`, `gradcheck`,
  `predict`) with real output inspected, not just exit codes.

## Current status

**v1 — functional and tested.** Dense layers, ReLU/Sigmoid/Tanh/Softmax
activations, MSE and cross-entropy losses, SGD and momentum optimizers,
full training loop with mini-batching and L2 regularization, model
save/load, CLI, and a 68-test suite including numerical gradient checks
for every component are all in place and passing.

Possible future extensions (not yet built): additional optimizers (Adam,
RMSProp), dropout / batch normalization, a convolutional layer, and a
matplotlib-based (rather than ASCII) decision-boundary plot as an
optional extra.

## License

MIT — see [LICENSE](LICENSE).
