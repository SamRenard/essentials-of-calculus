"""
test_exercises.py
Day 45 -- test suite for gradient_descent_2d.py and partial_derivatives_anatomy.py
"""

import os
import numpy as np
import gradient_descent_2d as gd
import partial_derivatives_anatomy as pd


def test_gradient_is_zero_at_minimum():
    grad = gd.grad_f(3.0, 2.0)
    np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-4)


def test_gradient_points_away_from_minimum():
    grad = gd.grad_f(8.0, 5.0)
    # moving further from (3,2) should increase f, so gradient should
    # have a positive component pointing away from the minimum
    assert grad[0] > 0
    assert grad[1] > 0


def test_descent_converges_close_to_true_minimum():
    path = gd.gradient_descent((8.0, -3.0), learning_rate=0.3, n_steps=60)
    final = path[-1]
    assert abs(final[0] - 3.0) < 0.01
    assert abs(final[1] - 2.0) < 0.01


def test_descent_reduces_f_monotonically_enough():
    path = gd.gradient_descent((8.0, -3.0), learning_rate=0.3, n_steps=60)
    values = [gd.f(x, y) for x, y in path]
    # allow the very first couple steps to overshoot slightly, but the
    # overall trend must be strongly decreasing
    assert values[-1] < values[0] * 0.001


def test_descent_from_different_start_reaches_same_minimum():
    path = gd.gradient_descent((-1.0, 6.0), learning_rate=0.3, n_steps=80)
    final = path[-1]
    assert abs(final[0] - 3.0) < 0.05
    assert abs(final[1] - 2.0) < 0.05


def test_plot_descent_creates_file(tmp_name="test_gradient_descent.png"):
    path = gd.gradient_descent((8.0, -3.0), n_steps=20)
    saved_to = gd.plot_descent(path, save_path=tmp_name)
    assert os.path.exists(saved_to)
    os.remove(saved_to)


def test_learning_rate_too_small_makes_slow_progress():
    slow_path = gd.gradient_descent((8.0, -3.0), learning_rate=0.01, n_steps=20)
    fast_path = gd.gradient_descent((8.0, -3.0), learning_rate=0.3, n_steps=20)
    slow_final_f = gd.f(*slow_path[-1])
    fast_final_f = gd.f(*fast_path[-1])
    assert fast_final_f < slow_final_f


# ---------------------------------------------------------------- partial_derivatives_anatomy.py
def test_partial_derivative_matches_analytical():
    f = lambda x, y: x ** 2 + y ** 3
    result = pd.numeric_partial_derivative(f, [2.0, 3.0], var_index=0)
    assert abs(result - 4.0) < 1e-4


def test_gradient_vector_matches_analytical():
    f = lambda x, y: x ** 2 + y ** 2
    grad = pd.numeric_gradient(f, [3.0, 4.0])
    np.testing.assert_allclose(grad, [6.0, 8.0], atol=1e-4)


def test_negative_gradient_step_decreases_function():
    f = lambda x, y: (x - 1) ** 2 + (y - 2) ** 2
    point = np.array([4.0, 6.0])
    grad = pd.numeric_gradient(f, point)
    downhill_point = point - 0.1 * grad
    assert f(*downhill_point) < f(*point)


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
