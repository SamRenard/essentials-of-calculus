"""
derivative_anatomy.py
Day 43 -- Derivative Intuition: Instantaneous Rate of Change
================================================================

A derivative is the slope of a curve at a single point -- the rate of
change "in the instant", not averaged over some interval. You can't
literally divide by zero to get an instant, so calculus approximates
it: take the average slope over a tiny interval h, then watch what
happens as h shrinks toward zero. This script builds that intuition
step by step, in the spirit of 3Blue1Brown's "Essence of Calculus"
episodes 1-3.

Run directly:
    python derivative_anatomy.py
"""

import numpy as np


def _rule_line():
    print("=" * 70)


def average_slope(f, x, h):
    """The slope of the secant line through (x, f(x)) and (x+h, f(x+h))."""
    return (f(x + h) - f(x)) / h


def demo_average_vs_instantaneous():
    """
    For f(x) = x^2, the average slope over [x, x+h] is NOT the
    instantaneous slope at x -- but as h shrinks, the two converge.
    The known instantaneous derivative of x^2 is 2x.
    """
    _rule_line()
    print("1) AVERAGE slope over a shrinking interval -> INSTANTANEOUS slope")
    _rule_line()
    f = lambda x: x ** 2
    x = 3.0
    true_derivative = 2 * x   # calculus fact: d/dx(x^2) = 2x
    print(f"f(x) = x^2, evaluating the slope at x = {x}")
    print(f"true instantaneous derivative (2x) = {true_derivative}\n")
    for h in [1.0, 0.1, 0.01, 0.0001, 0.000001]:
        approx = average_slope(f, x, h)
        print(f"h = {h:<10}  average slope = {approx:.6f}  "
              f"error = {abs(approx - true_derivative):.6f}")
    print()


def demo_dx_notation():
    """
    Leibniz's dx/dy notation is exactly this limit written compactly:
    dy/dx = lim(h -> 0) [f(x+h) - f(x)] / h
    It is NOT literally a fraction of two infinitely small numbers --
    it is shorthand for "the limit of this ratio as h shrinks to 0".
    """
    _rule_line()
    print("2) dy/dx is a LIMIT, not a literal tiny fraction")
    _rule_line()
    f = lambda x: np.sin(x)
    x = np.pi / 4
    true_derivative = np.cos(x)   # d/dx(sin x) = cos x
    print(f"f(x) = sin(x), x = pi/4")
    print(f"true derivative (cos x) = {true_derivative:.6f}\n")
    hs = [0.5, 0.1, 0.01, 0.001, 0.00001]
    approxes = [average_slope(f, x, h) for h in hs]
    for h, a in zip(hs, approxes):
        print(f"h = {h:<10}  [f(x+h)-f(x)]/h = {a:.6f}")
    print(f"\nAs h -> 0 the ratio approaches cos(x) -- that limit IS")
    print(f"the derivative, dy/dx.\n")


def demo_three_finite_difference_schemes():
    """
    In practice (numerically) there are three common ways to pick the
    tiny interval around x: forward, backward, and central difference.
    Central difference is the most accurate for the same step size h,
    because its error shrinks with h^2 instead of just h.
    """
    _rule_line()
    print("3) Forward vs backward vs central difference")
    _rule_line()
    f = lambda x: np.exp(x)
    x = 1.0
    true_derivative = np.exp(x)   # d/dx(e^x) = e^x
    h = 0.01
    forward = (f(x + h) - f(x)) / h
    backward = (f(x) - f(x - h)) / h
    central = (f(x + h) - f(x - h)) / (2 * h)
    print(f"f(x) = e^x, x = 1, h = {h}")
    print(f"true derivative       : {true_derivative:.6f}")
    print(f"forward difference    : {forward:.6f}  (error {abs(forward-true_derivative):.6f})")
    print(f"backward difference   : {backward:.6f}  (error {abs(backward-true_derivative):.6f})")
    print(f"central difference    : {central:.6f}  (error {abs(central-true_derivative):.6f})")
    print("central difference is consistently the closest for the same h\n")


def demo_derivative_as_a_function():
    """
    The derivative isn't just a number at one point -- it's itself a
    new function, mapping every x to the slope of the original
    function at that x. Sampling many x values and taking the
    central-difference slope at each one approximates that whole
    derivative function, e.g. for f(x) = x^3 the derivative is 3x^2.
    """
    _rule_line()
    print("4) The derivative as a function, not just a single slope")
    _rule_line()
    f = lambda x: x ** 3
    xs = np.linspace(-2, 2, 5)
    h = 1e-5
    approx_derivative = (f(xs + h) - f(xs - h)) / (2 * h)
    true_derivative = 3 * xs ** 2
    print(f"{'x':>6} | {'approx f_prime':>15} | {'true f_prime (3x^2)':>20}")
    print("-" * 50)
    for x_val, approx, true in zip(xs, approx_derivative, true_derivative):
        print(f"{x_val:>6.2f} | {approx:>15.6f} | {true:>20.6f}")
    print()


if __name__ == "__main__":
    demo_average_vs_instantaneous()
    demo_dx_notation()
    demo_three_finite_difference_schemes()
    demo_derivative_as_a_function()
