"""
linear_regression_gd.py
Day 46 -- Practical Coding: Linear Regression from Scratch (NumPy GD)
===========================================================================

A single-feature linear regression y = w*x + b, trained purely with
gradient descent implemented by hand in NumPy (no sklearn). Runs the
SAME training data through several learning rates and plots every
resulting loss curve on one chart, so the "too small / just right /
too large" story from today's theory is visible on a realistic
training problem, not just a toy 1D bowl.

Run directly:
    python linear_regression_gd.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_dataset(n=200, true_w=2.0, true_b=1.0, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = true_w * x + true_b + rng.normal(scale=noise, size=n)
    return x, y


def mse_loss(w, b, x, y):
    preds = w * x + b
    return np.mean((preds - y) ** 2)


def mse_gradient(w, b, x, y):
    """Analytical gradient of mean squared error w.r.t. w and b."""
    preds = w * x + b
    error = preds - y
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)
    return dw, db


def train_linear_regression(x, y, learning_rate, n_epochs=100):
    """
    Batch gradient descent on w and b. Returns the final (w, b) and
    the full loss history, one value per epoch.
    """
    w, b = 0.0, 0.0
    loss_history = []
    for _ in range(n_epochs):
        loss_history.append(mse_loss(w, b, x, y))
        dw, db = mse_gradient(w, b, x, y)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b, loss_history


def plot_loss_curves(results, save_path="loss_curves.png"):
    """
    results: dict of {learning_rate: loss_history}. Curves that
    diverge get clipped on the y-axis so the chart stays readable.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for lr, history in results.items():
        capped = np.clip(history, 0, 50)   # keep exploding curves visible but bounded
        ax.plot(capped, label=f"lr = {lr}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (capped at 50 for readability)")
    ax.set_title("Gradient descent loss curves at different learning rates")
    ax.legend()
    ax.set_ylim(bottom=-1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    x, y = make_dataset()
    true_w, true_b = 2.0, 1.0

    learning_rates = [0.001, 0.01, 0.05, 0.15, 0.35]
    results = {}

    print("=" * 78)
    print(f"{'learning rate':>14} | {'final w':>10} | {'final b':>10} | "
          f"{'final loss':>12} | {'verdict':>12}")
    print("=" * 78)
    for lr in learning_rates:
        w, b, history = train_linear_regression(x, y, lr, n_epochs=100)
        results[lr] = history
        final_loss = history[-1]
        if not np.isfinite(final_loss) or final_loss > 1e6:
            verdict = "diverged"
        elif final_loss > 1.0:
            verdict = "too slow"
        else:
            verdict = "converged"
        print(f"{lr:>14} | {w:>10.4f} | {b:>10.4f} | {final_loss:>12.4f} | {verdict:>12}")

    print(f"\ntrue parameters: w = {true_w}, b = {true_b}")

    saved_to = plot_loss_curves(results)
    print(f"\nSaved loss curve comparison to: {saved_to}")
