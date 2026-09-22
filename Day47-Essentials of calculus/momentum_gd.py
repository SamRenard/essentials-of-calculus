"""
Day 47 — Convexity, Local/Global Minima & Momentum-based Gradient Descent
NIZAM AI · 150-Day AI Engineering Protocol

Goal: implement vanilla gradient descent and momentum-based gradient
descent from scratch, then compare their convergence speed and their
ability to escape shallow local minima on a non-convex function.

Core intuition: vanilla GD only looks at the current gradient at each
step -- it can be slow in narrow valleys (zig-zagging) and can get
stuck in shallow local minima. Momentum keeps a running "velocity"
term, like a ball rolling downhill: v = beta*v - lr*grad(x), then
x += v. This lets it build up speed through flat regions, dampen
oscillation in narrow valleys, and sometimes coast straight through
a shallow local minimum into a deeper one.
"""

from __future__ import annotations
from typing import Callable, List
import math


def vanilla_gd(
    grad_fn: Callable[[float], float],
    x0: float,
    lr: float = 0.1,
    steps: int = 100,
) -> List[float]:
    """
    Standard gradient descent: x_{t+1} = x_t - lr * grad(x_t).
    Only ever looks at the current gradient -- no memory of past steps.
    """
    x = x0
    trajectory = [x]
    for _ in range(steps):
        g = grad_fn(x)
        x = x - lr * g
        trajectory.append(x)
    return trajectory


def momentum_gd(
    grad_fn: Callable[[float], float],
    x0: float,
    lr: float = 0.1,
    beta: float = 0.9,
    steps: int = 100,
) -> List[float]:
    """
    Momentum-based gradient descent:
      v_{t+1} = beta * v_t - lr * grad(x_t)
      x_{t+1} = x_t + v_{t+1}

    beta controls how much of the previous velocity carries over
    (0 = vanilla GD, closer to 1 = more "inertia").
    """
    x = x0
    v = 0.0
    trajectory = [x]
    for _ in range(steps):
        g = grad_fn(x)
        v = beta * v - lr * g
        x = x + v
        trajectory.append(x)
    return trajectory


def is_convex_numerically(f: Callable[[float], float], a: float, b: float, samples: int = 50) -> bool:
    """
    Checks the definition of convexity directly: for random t in [0,1] and
    points x1, x2 in [a, b], verify f(t*x1 + (1-t)*x2) <= t*f(x1) + (1-t)*f(x2).
    A single violation proves the function is NOT convex on this interval.
    """
    import random
    for _ in range(samples):
        x1 = random.uniform(a, b)
        x2 = random.uniform(a, b)
        t = random.uniform(0, 1)
        lhs = f(t * x1 + (1 - t) * x2)
        rhs = t * f(x1) + (1 - t) * f(x2)
        if lhs > rhs + 1e-9:  # small tolerance for float error
            return False
    return True


# ---------------------------------------------------------------------
# Two test functions: a convex bowl, and a non-convex function with a
# shallow local minimum next to a deeper global minimum.
# ---------------------------------------------------------------------

def convex_bowl(x: float) -> float:
    """f(x) = x^2 -- the textbook convex function, one global minimum at x=0."""
    return x ** 2


def convex_bowl_grad(x: float) -> float:
    return 2 * x


def non_convex_landscape(x: float) -> float:
    """
    An asymmetric double-well function: f(x) = 0.25*x^4 - 2*x^2 - 0.5*x
    Two local minima -- a shallow one near x=-2 (f=-3) and a deeper,
    global one near x=+2 (f=-5) -- separated by a local maximum near x=0.
    """
    return 0.25 * x ** 4 - 2 * x ** 2 - 0.5 * x


def non_convex_landscape_grad(x: float) -> float:
    """Derivative of non_convex_landscape: f'(x) = x^3 - 4x - 0.5"""
    return x ** 3 - 4 * x - 0.5


def narrow_valley(x: float, y: float) -> float:
    """
    A steep, narrow valley: f(x,y) = x^2 + 50*y^2
    Curvature along y is 50x sharper than along x. Vanilla GD with a
    learning rate large enough to make progress along x will overshoot
    and oscillate along y; momentum dampens that oscillation while
    still building speed along the shallow x direction.
    """
    return x ** 2 + 50 * y ** 2


def narrow_valley_grad(x: float, y: float) -> tuple:
    return (2 * x, 100 * y)


def vanilla_gd_2d(x0: float, y0: float, lr: float, steps: int):
    x, y = x0, y0
    traj = [(x, y)]
    for _ in range(steps):
        gx, gy = narrow_valley_grad(x, y)
        x, y = x - lr * gx, y - lr * gy
        traj.append((x, y))
    return traj


def momentum_gd_2d(x0: float, y0: float, lr: float, beta: float, steps: int):
    x, y = x0, y0
    vx, vy = 0.0, 0.0
    traj = [(x, y)]
    for _ in range(steps):
        gx, gy = narrow_valley_grad(x, y)
        vx, vy = beta * vx - lr * gx, beta * vy - lr * gy
        x, y = x + vx, y + vy
        traj.append((x, y))
    return traj


if __name__ == "__main__":
    print("=" * 70)
    print("Convexity check")
    print("=" * 70)
    print(f"convex_bowl (x^2) is convex on [-5,5]: {is_convex_numerically(convex_bowl, -5, 5)}")
    print(f"non_convex_landscape is convex on [-5,5]: {is_convex_numerically(non_convex_landscape, -5, 5)}")

    print("\n" + "=" * 70)
    print("Convex bowl: vanilla GD vs momentum GD (both reach x=0 easily)")
    print("=" * 70)
    traj_v = vanilla_gd(convex_bowl_grad, x0=4.0, lr=0.15, steps=30)
    traj_m = momentum_gd(convex_bowl_grad, x0=4.0, lr=0.15, beta=0.8, steps=30)
    print(f"vanilla GD:  final x = {traj_v[-1]:.6f}  (steps to |x|<0.01: "
          f"{next((i for i, x in enumerate(traj_v) if abs(x) < 0.01), 'never')})")
    print(f"momentum GD: final x = {traj_m[-1]:.6f}  (steps to |x|<0.01: "
          f"{next((i for i, x in enumerate(traj_m) if abs(x) < 0.01), 'never')})")

    print("\n" + "=" * 70)
    print("Non-convex double-well: convergence direction and speed")
    print("=" * 70)
    x_start = -1.8  # in the shallow well's basin, near the local max boundary
    traj_v2 = vanilla_gd(non_convex_landscape_grad, x0=x_start, lr=0.08, steps=200)
    traj_m2 = momentum_gd(non_convex_landscape_grad, x0=x_start, lr=0.08, beta=0.9, steps=200)
    print(f"vanilla GD final x:  {traj_v2[-1]:.4f}  (f = {non_convex_landscape(traj_v2[-1]):.4f})")
    print(f"momentum GD final x: {traj_m2[-1]:.4f}  (f = {non_convex_landscape(traj_m2[-1]):.4f})")
    print("global minimum is at x=2 (f=-5); shallow local minimum is at x=-2 (f=-3).")

    print("\n" + "=" * 70)
    print("Narrow valley (2D): momentum dampens oscillation, GD zig-zags")
    print("=" * 70)
    steps_2d = 60
    tv2d = vanilla_gd_2d(x0=-3.0, y0=1.0, lr=0.015, steps=steps_2d)
    tm2d = momentum_gd_2d(x0=-3.0, y0=1.0, lr=0.015, beta=0.7, steps=steps_2d)
    print(f"vanilla GD final (x,y):  ({tv2d[-1][0]:.4f}, {tv2d[-1][1]:.4f})  "
          f"f = {narrow_valley(*tv2d[-1]):.6f}")
    print(f"momentum GD final (x,y): ({tm2d[-1][0]:.4f}, {tm2d[-1][1]:.4f})  "
          f"f = {narrow_valley(*tm2d[-1]):.6f}")
    # Count sign flips in y as a proxy for oscillation
    def count_sign_flips(traj, idx):
        vals = [p[idx] for p in traj]
        return sum(1 for i in range(1, len(vals)) if vals[i - 1] * vals[i] < 0)
    print(f"vanilla GD  y-oscillations (sign flips): {count_sign_flips(tv2d, 1)}")
    print(f"momentum GD y-oscillations (sign flips): {count_sign_flips(tm2d, 1)}")
    print("-> at this learning rate, vanilla GD's steps are too small to make "
          "real progress along x before y damps out -- it crawls slowly toward "
          "the valley floor. Momentum keeps accumulating velocity along x even "
          "while y oscillates, so it reaches the minimum in far fewer steps "
          "and with much lower final loss.")
