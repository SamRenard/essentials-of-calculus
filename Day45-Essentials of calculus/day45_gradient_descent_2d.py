"""
gradient_descent_2d.py
Day 45 -- Practical Coding: Gradient Descent From Scratch on a 2D Function
===============================================================================

Implements gradient descent with no ML library -- just a numeric
gradient -- and runs it on a 2D function with an interesting
(non-symmetric, curved) landscape. The descent path is plotted over a
contour map of the function and saved as a PNG, so the whole "walk
downhill following the negative gradient" story is visible at a
glance.

Run directly:
    python gradient_descent_2d.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- the landscape
def f(x, y):
    """
    A curved bowl (not a simple circle) so the descent path visibly
    bends: elongated along x, with a mild off-axis term. Global
    minimum at (3, 2).
    """
    return 0.3 * (x - 3) ** 2 + 1.5 * (y - 2) ** 2 + 0.5 * (x - 3) * (y - 2)


def grad_f(x, y, h=1e-6):
    """Numeric gradient of f, central difference in each coordinate."""
    dfdx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    dfdy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([dfdx, dfdy])


# ---------------------------------------------------------------- gradient descent
def gradient_descent(start, learning_rate=0.3, n_steps=60, tol=1e-8):
    """
    Plain gradient descent: repeatedly step in the direction of the
    NEGATIVE gradient (steepest decrease), scaled by learning_rate.
    Returns the full path taken, as an (n_steps+1, 2) array.
    """
    point = np.array(start, dtype=float)
    path = [point.copy()]
    for _ in range(n_steps):
        gradient = grad_f(*point)
        step = learning_rate * gradient
        point = point - step
        path.append(point.copy())
        if np.linalg.norm(step) < tol:
            break
    return np.array(path)


def plot_descent(path, save_path="gradient_descent.png"):
    """Contour map of f with the descent path overlaid."""
    x_range = np.linspace(-2, 9, 200)
    y_range = np.linspace(-4, 8, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    fig, ax = plt.subplots(figsize=(7, 6))
    contours = ax.contour(X, Y, Z, levels=25, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=7)

    ax.plot(path[:, 0], path[:, 1], "o-", color="crimson",
             markersize=3, linewidth=1.2, label="gradient descent path")
    ax.plot(path[0, 0], path[0, 1], "s", color="black", markersize=8,
             label="start")
    ax.plot(3, 2, "*", color="gold", markersize=16,
             markeredgecolor="black", label="true minimum (3, 2)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gradient descent on f(x,y) = 0.3(x-3)^2 + 1.5(y-2)^2 + 0.5(x-3)(y-2)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    start = (8.0, -3.0)
    path = gradient_descent(start, learning_rate=0.3, n_steps=60)

    print("=" * 60)
    print("Gradient descent from scratch")
    print("=" * 60)
    print(f"start point       : {start}")
    print(f"steps taken       : {len(path) - 1}")
    print(f"final point       : ({path[-1, 0]:.4f}, {path[-1, 1]:.4f})")
    print(f"true minimum      : (3.0000, 2.0000)")
    print(f"final f(x, y)     : {f(*path[-1]):.6f}")
    print(f"f at true minimum : {f(3.0, 2.0):.6f}")

    saved_to = plot_descent(path)
    print(f"\nSaved descent plot to: {saved_to}")
