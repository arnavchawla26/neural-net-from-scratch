import os
import tempfile

import pytest

from nn_scratch.cli import main, build_parser, _ascii_decision_boundary
from nn_scratch.network import MLP
from nn_scratch.data import make_spiral


def test_build_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_train_smoke(capsys):
    rc = main(["train", "--epochs", "3", "--hidden", "4", "--dataset", "moons"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "final test accuracy" in out


def test_train_with_plot_and_save(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.npz")
        rc = main(
            [
                "train",
                "--epochs",
                "3",
                "--hidden",
                "4",
                "--dataset",
                "spiral",
                "--plot",
                "--save",
                path,
            ]
        )
        assert rc == 0
        assert os.path.exists(path)
        out = capsys.readouterr().out
        assert "decision boundary" in out
        assert "saved model to" in out


def test_gradcheck_passes(capsys):
    rc = main(["gradcheck", "--hidden", "3", "2", "--batch", "4"])
    out = capsys.readouterr().out
    assert "PASS" in out
    assert rc == 0


def test_predict_after_train(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.npz")
        main(["train", "--epochs", "2", "--dataset", "moons", "--save", path])
        capsys.readouterr()  # discard training output

        rc = main(["predict", path, "0.1", "0.2"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "class" in out


def test_ascii_decision_boundary_matches_grid_shape():
    X, y = make_spiral(n_per_class=20, seed=0)
    model = MLP([2, 4, 3], seed=0)
    art = _ascii_decision_boundary(model, X, y, width=20, height=10)
    lines = art.split("\n")
    assert len(lines) == 10
    assert all(len(line) == 20 for line in lines)


def test_main_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["not-a-real-command"])
