"""
partial_derivatives_anatomy.py
Day 45 -- Partial Derivatives and Gradients: The Multivariable World
============================================================================

Real functions (a loss function, a neural network) almost never
depend on just one variable. Once there are several inputs, "the
derivative" isn't a single number anymore -- you need a PARTIAL
derivative for each input, and the GRADIENT collects all of them into
one vector that points in the direction of steepest increase.

Run directly:
    python partial_derivatives_anatomy.py
"""

import numpy as np


def _rule_line():
    print("=" * 70)


def numeric_partial_derivative(f, point, var_index, h=1e-6):
    """
    Numeric partial derivative of f with respect to one variable:
    nudge only var_index by +-h, hold every other coordinate fixed.
    """
    point = np.array(point, dtype=float)
    forward = point.copy()
    backward = point.copy()
    forward[var_index] += h
    backward[var_index] -= h
    return (f(*forward) - f(*backward)) / (2 * h)


def numeric_gradient(f, point, h=1e-6):
    """The full gradient: one partial derivative per input variable."""
    point = np.array(point, dtype=float)
    return np.array([
        numeric_partial_derivative(f, point, i, h)
        for i in range(len(point))
    ])


def demo_one_variable_at_a_time():
    """
    f(x, y) = x^2 + y^3.
    The partial derivative with respect to x treats y as a constant
    (df/dx = 2x); the partial derivative with respect to y treats x as
    a constant (df/dy = 3y^2). Each one answers "how does f change if
    ONLY this variable moves?"
    """
    _rule_line()
    print("1) Partial derivatives -- one variable moves, the rest are frozen")
    _rule_line()
    f = lambda x, y: x ** 2 + y ** 3
    x, y = 2.0, 3.0

    df_dx_analytical = 2 * x
    df_dy_analytical = 3 * y ** 2
    df_dx_numeric = numeric_partial_derivative(f, [x, y], var_index=0)
    df_dy_numeric = numeric_partial_derivative(f, [x, y], var_index=1)

    print(f"f(x, y) = x^2 + y^3, at (x, y) = ({x}, {y})")
    print(f"df/dx (treat y as constant) : analytical={df_dx_analytical:.6f}, "
          f"numeric={df_dx_numeric:.6f}")
    print(f"df/dy (treat x as constant) : analytical={df_dy_analytical:.6f}, "
          f"numeric={df_dy_numeric:.6f}\n")


def demo_the_gradient_vector():
    """
    The gradient, written grad(f) or nabla-f, is just the vector of
    all partial derivatives: [df/dx, df/dy, ...]. It points in the
    direction where f increases fastest, and its length tells you how
    steep that increase is.
    """
    _rule_line()
    print("2) The gradient vector -- all partials, collected together")
    _rule_line()
    f = lambda x, y: x ** 2 + y ** 2   # a bowl, minimum at (0,0)
    point = [3.0, 4.0]
    grad = numeric_gradient(f, point)
    print(f"f(x, y) = x^2 + y^2, at point {point}")
    print(f"gradient = {grad}")
    print(f"gradient magnitude (steepness here) = {np.linalg.norm(grad):.6f}")
    print("The gradient [6, 8] points directly away from the origin --")
    print("exactly the direction of steepest INCREASE of the bowl.\n")


def demo_gradient_points_uphill_negative_gradient_downhill():
    """
    Because the gradient points toward the steepest increase, walking
    in the OPPOSITE direction (negative gradient) is the steepest way
    DOWN -- which is exactly the idea behind gradient descent.
    """
    _rule_line()
    print("3) Gradient = uphill, negative gradient = downhill")
    _rule_line()
    f = lambda x, y: (x - 1) ** 2 + (y - 2) ** 2   # bowl centered at (1,2)
    point = [4.0, 6.0]
    grad = numeric_gradient(f, point)

    step = 0.1
    uphill_point = np.array(point) + step * grad
    downhill_point = np.array(point) - step * grad

    print(f"f(x, y) = (x-1)^2 + (y-2)^2, minimum at (1, 2)")
    print(f"current point = {point}, f = {f(*point):.4f}")
    print(f"gradient here = {grad}")
    print(f"step +gradient -> {uphill_point}, f = {f(*uphill_point):.4f}  (went UP)")
    print(f"step -gradient -> {downhill_point}, f = {f(*downhill_point):.4f}  (went DOWN)\n")


def demo_multivariable_chain_rule():
    """
    The chain rule extends naturally: if z depends on x and y, and x
    and y both depend on t, then dz/dt sums the contribution through
    every path -- this is exactly what backprop does across every
    weight in a network with multiple connections.
    """
    _rule_line()
    print("4) Multivariable chain rule -- summing contributions over every path")
    _rule_line()
    # z = x*y, x = t^2, y = t+1  =>  z(t) = t^2 * (t+1)
    x_of_t = lambda t: t ** 2
    y_of_t = lambda t: t + 1
    z = lambda x, y: x * y
    z_of_t = lambda t: z(x_of_t(t), y_of_t(t))

    t = 2.0
    x, y = x_of_t(t), y_of_t(t)
    dz_dx = y                 # d/dx(x*y) = y
    dz_dy = x                 # d/dy(x*y) = x
    dx_dt = 2 * t              # d/dt(t^2)
    dy_dt = 1.0                # d/dt(t+1)
    dz_dt_chain = dz_dx * dx_dt + dz_dy * dy_dt

    def numeric_derivative(f, t, h=1e-6):
        return (f(t + h) - f(t - h)) / (2 * h)

    dz_dt_numeric = numeric_derivative(z_of_t, t)

    print(f"z = x*y, x = t^2, y = t+1, at t = {t}")
    print(f"dz/dt = dz/dx * dx/dt + dz/dy * dy/dt")
    print(f"      = {dz_dx} * {dx_dt} + {dz_dy} * {dy_dt} = {dz_dt_chain}")
    print(f"numeric dz/dt = {dz_dt_numeric:.6f}")
    print(f"match: {abs(dz_dt_chain - dz_dt_numeric) < 1e-4}\n")


if __name__ == "__main__":
    demo_one_variable_at_a_time()
    demo_the_gradient_vector()
    demo_gradient_points_uphill_negative_gradient_downhill()
    demo_multivariable_chain_rule()
