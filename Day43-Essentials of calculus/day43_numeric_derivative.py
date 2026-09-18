"""
numeric_derivative.py
Day 43 -- Practical Coding: Numeric Derivative via Finite Differences
==========================================================================

A general-purpose numeric derivative function using the central
difference formula:

    f'(x) ~= [f(x+h) - f(x-h)] / (2h)

Tested against 5 functions with known analytical derivatives, so the
accuracy of the numeric approximation can be checked directly instead
of just trusted.

Run directly:
    python numeric_derivative.py
"""

import numpy as np


def numeric_derivative(f, x, h=1e-5):
    """
    Central-difference numeric derivative of f at point(s) x.
    Works for a scalar x or a numpy array of x values.
    """
    x = np.asarray(x, dtype=float)
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------- 5 test functions
# Each entry: (name, function, analytical derivative, a few sample x values)
TEST_FUNCTIONS = [
    ("f(x) = x^2",       lambda x: x ** 2,        lambda x: 2 * x,             [-3, -1, 0, 1, 3]),
    ("f(x) = x^3",       lambda x: x ** 3,        lambda x: 3 * x ** 2,        [-2, -1, 0, 1, 2]),
    ("f(x) = sin(x)",    lambda x: np.sin(x),     lambda x: np.cos(x),         [0, np.pi / 4, np.pi / 2, np.pi]),
    ("f(x) = e^x",       lambda x: np.exp(x),     lambda x: np.exp(x),         [-1, 0, 1, 2]),
    ("f(x) = ln(x)",     lambda x: np.log(x),     lambda x: 1.0 / x,           [0.5, 1, 2, 5]),
]


def run_all_tests(h=1e-5):
    print("=" * 78)
    print(f"{'function':<14} | {'x':>8} | {'numeric':>12} | {'analytical':>12} | {'abs error':>10}")
    print("=" * 78)
    max_error = 0.0
    for name, f, f_prime, xs in TEST_FUNCTIONS:
        for x in xs:
            numeric = float(numeric_derivative(f, x, h))
            analytical = float(f_prime(x))
            error = abs(numeric - analytical)
            max_error = max(max_error, error)
            print(f"{name:<14} | {x:>8.4f} | {numeric:>12.6f} | "
                  f"{analytical:>12.6f} | {error:>10.2e}")
        print("-" * 78)
    print(f"\nLargest error across all 5 functions: {max_error:.2e}")
    return max_error


if __name__ == "__main__":
    run_all_tests()
