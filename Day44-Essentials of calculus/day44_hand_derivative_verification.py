"""
hand_derivative_verification.py
Day 44 -- Practical Coding: Hand-Derive Complex Functions, Verify Numerically
=================================================================================

Five composite functions. For each one the derivative is worked out
by hand using the chain rule (written out step by step in the
docstring and the `hand_derivative` function), then checked against a
numeric (finite-difference) derivative. If the two disagree, the hand
derivation has a mistake -- this is exactly the workflow you'd use to
sanity-check a backprop implementation.

Run directly:
    python hand_derivative_verification.py
"""

import numpy as np


def numeric_derivative(f, x, h=1e-6):
    """Central-difference numeric derivative, used as the ground truth check."""
    return (f(x + h) - f(x - h)) / (2 * h)


# =====================================================================
# 1. y = sin(x^2 + 3x)
#    Let u = x^2 + 3x, y = sin(u)
#    dy/dx = cos(u) * du/dx = cos(x^2 + 3x) * (2x + 3)
# =====================================================================
def f1(x):
    return np.sin(x ** 2 + 3 * x)


def f1_hand_derivative(x):
    u = x ** 2 + 3 * x
    du_dx = 2 * x + 3
    return np.cos(u) * du_dx


# =====================================================================
# 2. y = e^(-x^2)          (the un-normalized Gaussian bump)
#    Let u = -x^2, y = e^u
#    dy/dx = e^u * du/dx = e^(-x^2) * (-2x)
# =====================================================================
def f2(x):
    return np.exp(-x ** 2)


def f2_hand_derivative(x):
    u = -x ** 2
    du_dx = -2 * x
    return np.exp(u) * du_dx


# =====================================================================
# 3. y = ln(sin(x)^2 + 1)
#    Let v = sin(x), u = v^2 + 1, y = ln(u)
#    dv/dx = cos(x)
#    du/dv = 2v  ->  du/dx = 2*sin(x)*cos(x)
#    dy/du = 1/u
#    dy/dx = (1/u) * du/dx = 2*sin(x)*cos(x) / (sin(x)^2 + 1)
# =====================================================================
def f3(x):
    return np.log(np.sin(x) ** 2 + 1)


def f3_hand_derivative(x):
    v = np.sin(x)
    u = v ** 2 + 1
    du_dx = 2 * v * np.cos(x)
    return du_dx / u


# =====================================================================
# 4. y = sqrt(x^3 + 2x)             (a square root of a polynomial)
#    Let u = x^3 + 2x, y = u^(1/2)
#    dy/du = 1/(2*sqrt(u))
#    du/dx = 3x^2 + 2
#    dy/dx = (3x^2 + 2) / (2*sqrt(x^3 + 2x))
# =====================================================================
def f4(x):
    return np.sqrt(x ** 3 + 2 * x)


def f4_hand_derivative(x):
    u = x ** 3 + 2 * x
    du_dx = 3 * x ** 2 + 2
    return du_dx / (2 * np.sqrt(u))


# =====================================================================
# 5. y = sigmoid(w*x)^2, the squared output of a single neuron
#    sigmoid(z) = 1/(1+e^-z), sigmoid'(z) = sigmoid(z)*(1-sigmoid(z))
#    Let s = sigmoid(w*x), y = s^2
#    dy/ds = 2s
#    ds/dx = sigmoid'(w*x) * w
#    dy/dx = 2*sigmoid(w*x) * sigmoid'(w*x) * w
# =====================================================================
W = 1.7   # fixed weight for this exercise


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def f5(x):
    return _sigmoid(W * x) ** 2


def f5_hand_derivative(x):
    s = _sigmoid(W * x)
    ds_dx = s * (1 - s) * W
    return 2 * s * ds_dx


FUNCTIONS = [
    ("y = sin(x^2 + 3x)",       f1, f1_hand_derivative, [-2.0, 0.0, 0.5, 1.5]),
    ("y = e^(-x^2)",            f2, f2_hand_derivative, [-1.5, 0.0, 0.5, 2.0]),
    ("y = ln(sin(x)^2 + 1)",    f3, f3_hand_derivative, [0.2, 1.0, 2.0, 3.0]),
    ("y = sqrt(x^3 + 2x)",      f4, f4_hand_derivative, [0.5, 1.0, 2.0, 3.0]),
    ("y = sigmoid(1.7x)^2",     f5, f5_hand_derivative, [-2.0, -0.5, 0.5, 2.0]),
]


def run_all_verifications(h=1e-6, tolerance=1e-4):
    print("=" * 82)
    print(f"{'function':<22} | {'x':>6} | {'hand derivative':>16} | "
          f"{'numeric derivative':>19} | {'match':>6}")
    print("=" * 82)
    all_match = True
    for name, f, hand_derivative, xs in FUNCTIONS:
        for x in xs:
            hand = float(hand_derivative(x))
            numeric = float(numeric_derivative(f, x, h))
            match = abs(hand - numeric) < tolerance
            all_match = all_match and match
            print(f"{name:<22} | {x:>6.2f} | {hand:>16.6f} | "
                  f"{numeric:>19.6f} | {'OK' if match else 'MISMATCH':>6}")
        print("-" * 82)
    print(f"\nAll hand derivatives verified: {all_match}")
    return all_match


if __name__ == "__main__":
    run_all_verifications()
