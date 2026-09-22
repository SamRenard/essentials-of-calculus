"""
Day 47 — Visualizing Vanilla GD vs Momentum GD Convergence
NIZAM AI · 150-Day AI Engineering Protocol

Generates two comparison plots:
  1. Convex bowl (f=x^2): loss curve, vanilla vs momentum.
  2. Narrow 2D valley: trajectory paths overlaid on contours, showing
     momentum's smoother path vs vanilla GD's zig-zag oscillation.

Saves to PNG instead of plt.show() for headless environments.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentum_gd import (
    vanilla_gd, momentum_gd, convex_bowl, convex_bowl_grad,
    narrow_valley, vanilla_gd_2d, momentum_gd_2d,
)


def plot_convex_convergence(filename: str):
    traj_v = vanilla_gd(convex_bowl_grad, x0=4.0, lr=0.15, steps=30)
    traj_m = momentum_gd(convex_bowl_grad, x0=4.0, lr=0.15, beta=0.8, steps=30)

    loss_v = [convex_bowl(x) for x in traj_v]
    loss_m = [convex_bowl(x) for x in traj_m]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(loss_v, label="Vanilla GD", color="#DD8452", linewidth=2)
    ax.plot(loss_m, label="Momentum GD (beta=0.8)", color="#55A868", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("f(x) = x^2  (log scale)")
    ax.set_title("Convex bowl: momentum converges in fewer steps")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"Saved {filename}")


def plot_narrow_valley_paths(filename: str):
    tv = vanilla_gd_2d(x0=-3.0, y0=1.0, lr=0.015, steps=60)
    tm = momentum_gd_2d(x0=-3.0, y0=1.0, lr=0.015, beta=0.7, steps=60)

    xs = np.linspace(-3.5, 0.5, 200)
    ys = np.linspace(-1.2, 1.2, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = narrow_valley(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, traj, title, color in [
        (axes[0], tv, "Vanilla GD -- stalls before reaching the minimum", "#DD8452"),
        (axes[1], tm, "Momentum GD -- oscillates but reaches the minimum", "#55A868"),
    ]:
        ax.contour(X, Y, Z, levels=25, cmap="Purples", alpha=0.5)
        xs_t = [p[0] for p in traj]
        ys_t = [p[1] for p in traj]
        ax.plot(xs_t, ys_t, color=color, linewidth=1.3, marker="o", markersize=2)
        ax.scatter([0], [0], color="gold", marker="*", s=180, zorder=5, label="minimum")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"Saved {filename}")


if __name__ == "__main__":
    plot_convex_convergence("convex_convergence.png")
    plot_narrow_valley_paths("narrow_valley_paths.png")
