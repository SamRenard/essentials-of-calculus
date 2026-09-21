"""
learning_rate_anatomy.py
Day 46 -- Gradient Descent Variants & the Critical Effect of Learning Rate
================================================================================

Gradient descent's update rule looks simple:

    point = point - learning_rate * gradient

but that one number, the learning rate, decides whether training
converges smoothly, crawls forever, or blows up entirely. This script
demonstrates the three regimes on a simple 1D bowl, then shows the
main GD variants (batch, stochastic, mini-batch, momentum) side by
side on the same problem.

Run directly:
    python learning_rate_anatomy.py
"""

import numpy as np


def _rule_line():
    print("=" * 70)


def f(x):
    """A simple 1D bowl, minimum at x = 4."""
    return (x - 4) ** 2


def grad_f(x):
    """Analytical derivative of f: d/dx (x-4)^2 = 2(x-4)."""
    return 2 * (x - 4)


def gradient_descent_1d(start, lr, n_steps=20):
    x = start
    path = [x]
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        path.append(x)
    return path


def demo_too_small_too_large_just_right():
    """
    Three learning rates on the same bowl:
      - too small  -> converges, but painfully slowly
      - just right -> converges quickly and smoothly
      - too large  -> overshoots, oscillates, and can diverge outright
    """
    _rule_line()
    print("1) Learning rate: too small vs just right vs too large")
    _rule_line()
    start = 0.0
    for label, lr in [("too small", 0.01), ("just right", 0.4), ("too large", 1.1)]:
        path = gradient_descent_1d(start, lr, n_steps=10)
        print(f"\nlr = {lr:<5} ({label})")
        print("  steps:", [f"{p:.3f}" for p in path])
        print(f"  final distance from minimum (x=4): {abs(path[-1] - 4):.4f}")
    print("\nNotice: lr=1.1 doesn't shrink the distance -- it GROWS, because")
    print("each step overshoots the minimum by more than the last step did.\n")


def demo_divergence_threshold():
    """
    For f(x) = (x-4)^2, the update becomes
        x_new = x - lr*2*(x-4) = (1 - 2*lr)*(x-4) + 4
    so the distance-to-minimum gets multiplied by |1 - 2*lr| every
    step. That factor must be < 1 for convergence: 0 < lr < 1.
    """
    _rule_line()
    print("2) Where exactly does it start diverging?")
    _rule_line()
    for lr in [0.3, 0.5, 0.9, 1.0, 1.05]:
        shrink_factor = abs(1 - 2 * lr)
        verdict = "converges" if shrink_factor < 1 else "diverges (or oscillates forever)"
        print(f"lr = {lr:<5}  |1 - 2*lr| = {shrink_factor:.2f}  -> {verdict}")
    print()


def demo_gd_variants():
    """
    The three classic variants differ in HOW MUCH DATA is used to
    compute the gradient at each step, and momentum adds "memory" of
    past gradients to smooth out the path.
    """
    _rule_line()
    print("3) GD variants (conceptual, on a linear regression toy problem)")
    _rule_line()
    rng = np.random.default_rng(0)
    n = 200
    x_data = rng.uniform(-3, 3, n)
    y_data = 2.0 * x_data + 1.0 + rng.normal(scale=0.5, size=n)

    def mse_grad(w, b, xs, ys):
        preds = w * xs + b
        err = preds - ys
        dw = 2 * np.mean(err * xs)
        db = 2 * np.mean(err)
        return dw, db

    # BATCH: uses the whole dataset every step -- smooth but one
    # gradient computation is expensive on large datasets.
    w, b = 0.0, 0.0
    for _ in range(50):
        dw, db = mse_grad(w, b, x_data, y_data)
        w, b = w - 0.05 * dw, b - 0.05 * db
    print(f"BATCH GD (all {n} points/step)      -> w={w:.3f}, b={b:.3f}")

    # STOCHASTIC: one random point per step -- noisy but cheap, and
    # the noise itself can help escape shallow local minima.
    w, b = 0.0, 0.0
    for _ in range(50):
        i = rng.integers(0, n)
        dw, db = mse_grad(w, b, x_data[i:i+1], y_data[i:i+1])
        w, b = w - 0.05 * dw, b - 0.05 * db
    print(f"STOCHASTIC GD (1 point/step)        -> w={w:.3f}, b={b:.3f}")

    # MINI-BATCH: a compromise, small random batches each step.
    w, b = 0.0, 0.0
    batch_size = 20
    for _ in range(50):
        idx = rng.choice(n, batch_size, replace=False)
        dw, db = mse_grad(w, b, x_data[idx], y_data[idx])
        w, b = w - 0.05 * dw, b - 0.05 * db
    print(f"MINI-BATCH GD ({batch_size} points/step)       -> w={w:.3f}, b={b:.3f}")

    # MOMENTUM: keeps a running average of past gradients, which
    # smooths oscillations and can speed up convergence.
    w, b, vw, vb = 0.0, 0.0, 0.0, 0.0
    momentum_coeff = 0.9
    for _ in range(50):
        dw, db = mse_grad(w, b, x_data, y_data)
        vw = momentum_coeff * vw + (1 - momentum_coeff) * dw
        vb = momentum_coeff * vb + (1 - momentum_coeff) * db
        w, b = w - 0.05 * vw, b - 0.05 * vb
    print(f"MOMENTUM GD (batch + velocity)      -> w={w:.3f}, b={b:.3f}")
    print(f"\ntrue underlying line: w=2.0, b=1.0\n")


if __name__ == "__main__":
    demo_too_small_too_large_just_right()
    demo_divergence_threshold()
    demo_gd_variants()
