import numpy as np

from nn_scratch.gradcheck import numerical_gradient, relative_error, check_gradient


def test_numerical_gradient_of_quadratic():
    # f(x) = sum(x^2)  ->  grad = 2x
    x = np.array([1.0, -2.0, 3.0])

    def f(x):
        return float(np.sum(x**2))

    grad = numerical_gradient(f, x.copy())
    np.testing.assert_allclose(grad, 2 * x, atol=1e-4)


def test_numerical_gradient_leaves_input_unchanged():
    x = np.array([1.0, 2.0, 3.0])
    x_copy = x.copy()

    def f(x):
        return float(np.sum(x**3))

    numerical_gradient(f, x)
    np.testing.assert_array_equal(x, x_copy)


def test_relative_error_zero_for_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert relative_error(a, a.copy()) == 0.0


def test_relative_error_handles_both_zero():
    a = np.zeros(3)
    assert relative_error(a, a.copy()) == 0.0


def test_check_gradient_detects_wrong_gradient():
    x = np.array([1.0, 2.0])

    def f(x):
        return float(np.sum(x**2))

    correct = 2 * x
    wrong = np.array([100.0, 100.0])

    passed, _ = check_gradient(f, x.copy(), correct)
    assert passed

    passed, _ = check_gradient(f, x.copy(), wrong)
    assert not passed
