"""
test_exercises.py
Day 46 -- test suite for linear_regression_gd.py and learning_rate_anatomy.py
"""

import os
import numpy as np
import linear_regression_gd as lr
import learning_rate_anatomy as la


def test_mse_loss_zero_for_perfect_fit():
    x = np.array([1.0, 2.0, 3.0])
    y = 2 * x + 1
    assert lr.mse_loss(2.0, 1.0, x, y) < 1e-9


def test_mse_gradient_zero_at_perfect_fit():
    x = np.array([1.0, 2.0, 3.0])
    y = 2 * x + 1
    dw, db = lr.mse_gradient(2.0, 1.0, x, y)
    assert abs(dw) < 1e-9
    assert abs(db) < 1e-9


def test_training_recovers_true_parameters():
    x, y = lr.make_dataset(n=300, true_w=2.0, true_b=1.0, noise=0.3, seed=1)
    w, b, history = lr.train_linear_regression(x, y, learning_rate=0.1, n_epochs=200)
    assert abs(w - 2.0) < 0.1
    assert abs(b - 1.0) < 0.1


def test_loss_decreases_with_good_learning_rate():
    x, y = lr.make_dataset(seed=2)
    _, _, history = lr.train_linear_regression(x, y, learning_rate=0.1, n_epochs=100)
    assert history[-1] < history[0]
    assert history[-1] < 1.0


def test_too_small_lr_makes_slow_progress():
    x, y = lr.make_dataset(seed=3)
    _, _, slow_history = lr.train_linear_regression(x, y, learning_rate=0.0005, n_epochs=100)
    _, _, fast_history = lr.train_linear_regression(x, y, learning_rate=0.1, n_epochs=100)
    assert slow_history[-1] > fast_history[-1]


def test_too_large_lr_diverges():
    x, y = lr.make_dataset(seed=4)
    _, _, history = lr.train_linear_regression(x, y, learning_rate=1.0, n_epochs=50)
    assert history[-1] > history[0]   # loss grew instead of shrinking


def test_plot_loss_curves_creates_file(tmp_name="test_loss_curves.png"):
    x, y = lr.make_dataset(seed=5)
    results = {0.01: lr.train_linear_regression(x, y, 0.01, 20)[2],
               0.1: lr.train_linear_regression(x, y, 0.1, 20)[2]}
    saved_to = lr.plot_loss_curves(results, save_path=tmp_name)
    assert os.path.exists(saved_to)
    os.remove(saved_to)


# ---------------------------------------------------------------- learning_rate_anatomy.py
def test_1d_descent_converges_with_good_lr():
    path = la.gradient_descent_1d(0.0, lr=0.4, n_steps=15)
    assert abs(path[-1] - 4.0) < 1e-3


def test_1d_descent_diverges_with_too_large_lr():
    path = la.gradient_descent_1d(0.0, lr=1.1, n_steps=10)
    assert abs(path[-1] - 4.0) > abs(path[1] - 4.0)  # distance grew


def test_divergence_threshold_boundary():
    # |1 - 2*lr| < 1 for lr in (0, 1); exactly 1.0 is the boundary
    assert abs(1 - 2 * 0.9) < 1.0
    assert abs(1 - 2 * 1.0) == 1.0
    assert abs(1 - 2 * 1.05) > 1.0


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
