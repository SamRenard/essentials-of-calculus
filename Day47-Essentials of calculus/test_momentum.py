"""
Day 47 — Automated correctness checks for momentum_gd.py
NIZAM AI · 150-Day AI Engineering Protocol
"""

from momentum_gd import (
    vanilla_gd, momentum_gd, is_convex_numerically,
    convex_bowl, convex_bowl_grad,
    non_convex_landscape, non_convex_landscape_grad,
    narrow_valley, vanilla_gd_2d, momentum_gd_2d,
)


def test_convexity_detection():
    assert is_convex_numerically(convex_bowl, -5, 5) is True
    assert is_convex_numerically(non_convex_landscape, -5, 5) is False
    print("[PASS] convexity check correctly classifies x^2 as convex, double-well as not")


def test_vanilla_gd_converges_on_convex_bowl():
    traj = vanilla_gd(convex_bowl_grad, x0=5.0, lr=0.1, steps=100)
    assert abs(traj[-1]) < 0.01
    assert convex_bowl(traj[-1]) < convex_bowl(traj[0])
    print("[PASS] vanilla GD converges to the minimum of a convex bowl")


def test_momentum_gd_converges_on_convex_bowl():
    traj = momentum_gd(convex_bowl_grad, x0=5.0, lr=0.1, beta=0.8, steps=100)
    assert abs(traj[-1]) < 0.05
    print("[PASS] momentum GD converges to the minimum of a convex bowl")


def test_momentum_converges_faster_than_vanilla():
    traj_v = vanilla_gd(convex_bowl_grad, x0=4.0, lr=0.15, steps=30)
    traj_m = momentum_gd(convex_bowl_grad, x0=4.0, lr=0.15, beta=0.8, steps=30)

    def steps_to_threshold(traj, thresh=0.01):
        for i, x in enumerate(traj):
            if abs(x) < thresh:
                return i
        return len(traj)

    steps_v = steps_to_threshold(traj_v)
    steps_m = steps_to_threshold(traj_m)
    assert steps_m < steps_v
    print(f"[PASS] momentum reaches threshold faster ({steps_m} steps) than vanilla ({steps_v} steps)")


def test_momentum_beta_zero_equals_vanilla():
    """beta=0 means momentum GD should reduce exactly to vanilla GD."""
    traj_v = vanilla_gd(convex_bowl_grad, x0=3.0, lr=0.1, steps=20)
    traj_m = momentum_gd(convex_bowl_grad, x0=3.0, lr=0.1, beta=0.0, steps=20)
    assert all(abs(a - b) < 1e-9 for a, b in zip(traj_v, traj_m))
    print("[PASS] momentum GD with beta=0 exactly matches vanilla GD")


def test_double_well_has_two_local_minima():
    # local minimum near x=-2 (f=-3), global minimum near x=2 (f=-5)
    f_local = non_convex_landscape(-2.0)
    f_global = non_convex_landscape(2.0)
    assert f_global < f_local  # global is deeper
    # gradient should be ~0 at both critical points
    assert abs(non_convex_landscape_grad(-2.0)) < 0.6
    assert abs(non_convex_landscape_grad(2.0)) < 0.6
    print(f"[PASS] double-well has local min f({-2})={f_local:.2f} and "
          f"deeper global min f(2)={f_global:.2f}")


def test_momentum_reduces_oscillation_in_narrow_valley():
    tv = vanilla_gd_2d(x0=-3.0, y0=1.0, lr=0.015, steps=60)
    tm = momentum_gd_2d(x0=-3.0, y0=1.0, lr=0.015, beta=0.7, steps=60)

    def count_sign_flips(traj, idx):
        vals = [p[idx] for p in traj]
        return sum(1 for i in range(1, len(vals)) if vals[i - 1] * vals[i] < 0)

    flips_v = count_sign_flips(tv, 1)
    flips_m = count_sign_flips(tm, 1)
    assert flips_m < flips_v
    print(f"[PASS] momentum has fewer y-oscillations ({flips_m}) than vanilla GD ({flips_v})")


def test_momentum_reaches_lower_loss_in_narrow_valley():
    tv = vanilla_gd_2d(x0=-3.0, y0=1.0, lr=0.015, steps=60)
    tm = momentum_gd_2d(x0=-3.0, y0=1.0, lr=0.015, beta=0.7, steps=60)
    f_v = narrow_valley(*tv[-1])
    f_m = narrow_valley(*tm[-1])
    assert f_m < f_v
    print(f"[PASS] momentum reaches lower loss ({f_m:.4f}) than vanilla GD ({f_v:.4f}) in the same steps")


if __name__ == "__main__":
    tests = [
        test_convexity_detection,
        test_vanilla_gd_converges_on_convex_bowl,
        test_momentum_gd_converges_on_convex_bowl,
        test_momentum_converges_faster_than_vanilla,
        test_momentum_beta_zero_equals_vanilla,
        test_double_well_has_two_local_minima,
        test_momentum_reduces_oscillation_in_narrow_valley,
        test_momentum_reaches_lower_loss_in_narrow_valley,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
