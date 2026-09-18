"""
test_exercises.py
Day 43 -- test suite for numeric_derivative.py and derivative_anatomy.py
"""

import numpy as np
import numeric_derivative as nd
import derivative_anatomy as da


def test_derivative_of_square():
    result = nd.numeric_derivative(lambda x: x ** 2, 3.0)
    assert abs(result - 6.0) < 1e-6


def test_derivative_of_cube():
    result = nd.numeric_derivative(lambda x: x ** 3, 2.0)
    assert abs(result - 12.0) < 1e-6


def test_derivative_of_sin():
    x = np.pi / 4
    result = nd.numeric_derivative(np.sin, x)
    assert abs(result - np.cos(x)) < 1e-6


def test_derivative_of_exp():
    result = nd.numeric_derivative(np.exp, 1.0)
    assert abs(result - np.exp(1.0)) < 1e-6


def test_derivative_of_log():
    result = nd.numeric_derivative(np.log, 2.0)
    assert abs(result - 0.5) < 1e-6


def test_works_on_array_input():
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = nd.numeric_derivative(lambda x: x ** 2, xs)
    expected = 2 * xs
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_run_all_tests_error_is_small():
    max_error = nd.run_all_tests(h=1e-5)
    assert max_error < 1e-6


def test_smaller_h_is_not_always_better_due_to_float_precision():
    # sanity check: a reasonable h (1e-5) beats an absurdly tiny h (1e-14)
    # which suffers from floating point cancellation
    f = lambda x: x ** 2
    good = abs(nd.numeric_derivative(f, 3.0, h=1e-5) - 6.0)
    tiny = abs(nd.numeric_derivative(f, 3.0, h=1e-14) - 6.0)
    assert good <= tiny + 1e-6  # good h should not be meaningfully worse


# ---------------------------------------------------------------- derivative_anatomy.py
def test_average_slope_matches_manual_calc():
    f = lambda x: x ** 2
    result = da.average_slope(f, 3.0, 0.1)
    expected = ((3.1 ** 2) - (3.0 ** 2)) / 0.1
    assert abs(result - expected) < 1e-9


def test_average_slope_converges_to_derivative_as_h_shrinks():
    f = lambda x: x ** 2
    x = 3.0
    true_derivative = 6.0
    errors = [abs(da.average_slope(f, x, h) - true_derivative)
              for h in [1.0, 0.1, 0.01, 0.0001]]
    # error should shrink monotonically as h shrinks
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))


def test_central_difference_more_accurate_than_forward():
    f = np.exp
    x, h = 1.0, 0.01
    true_derivative = np.exp(x)
    forward = (f(x + h) - f(x)) / h
    central = (f(x + h) - f(x - h)) / (2 * h)
    assert abs(central - true_derivative) < abs(forward - true_derivative)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}  ->  {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
