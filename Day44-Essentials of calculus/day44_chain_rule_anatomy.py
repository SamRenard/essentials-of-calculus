"""
chain_rule_anatomy.py
Day 44 -- The Chain Rule: The Heart of Backpropagation
================================================================

If y = f(g(x)), the chain rule says:

    dy/dx = f'(g(x)) * g'(x)

In words: the rate of change of the outer function, evaluated at the
inner function's output, multiplied by the rate of change of the
inner function. This single rule is what makes neural network
training possible: a network is nothing but a long chain of function
compositions (layer after layer), and backpropagation is just the
chain rule applied repeatedly, from the loss backward to every
weight.

Run directly:
    python chain_rule_anatomy.py
"""

import numpy as np


def _rule_line():
    print("=" * 70)


def numeric_derivative(f, x, h=1e-6):
    """Central-difference numeric derivative, used to check hand rules."""
    return (f(x + h) - f(x - h)) / (2 * h)


def demo_single_composition():
    """
    y = f(g(x)) where g(x) = x^2 and f(u) = sin(u).
    By the chain rule: dy/dx = cos(g(x)) * 2x = cos(x^2) * 2x
    """
    _rule_line()
    print("1) A single composition: y = sin(x^2)")
    _rule_line()
    g = lambda x: x ** 2
    f = lambda u: np.sin(u)
    y = lambda x: f(g(x))

    x = 1.5
    g_prime = 2 * x                 # d/dx(x^2) = 2x
    f_prime_at_g = np.cos(g(x))     # d/du(sin u) at u = g(x)
    chain_rule_result = f_prime_at_g * g_prime

    numeric_result = numeric_derivative(y, x)

    print(f"x = {x}")
    print(f"g(x) = x^2 = {g(x):.6f},  g'(x) = 2x = {g_prime:.6f}")
    print(f"f(u) = sin(u),  f'(g(x)) = cos(g(x)) = {f_prime_at_g:.6f}")
    print(f"chain rule: f'(g(x)) * g'(x) = {chain_rule_result:.6f}")
    print(f"numeric derivative of y = sin(x^2)  = {numeric_result:.6f}")
    print(f"match: {abs(chain_rule_result - numeric_result) < 1e-4}\n")


def demo_three_link_chain():
    """
    Chains can have more than two links. For
    y = h(f(g(x))), with g(x)=2x, f(u)=u^3, h(v)=ln(v):

        dy/dx = h'(f(g(x))) * f'(g(x)) * g'(x)

    Every layer just multiplies in its own local derivative.
    """
    _rule_line()
    print("2) Three links: y = ln((2x)^3)")
    _rule_line()
    g = lambda x: 2 * x
    f = lambda u: u ** 3
    h = lambda v: np.log(v)
    y = lambda x: h(f(g(x)))

    x = 2.0
    g_prime = 2.0                        # d/dx(2x) = 2
    f_prime_at_g = 3 * g(x) ** 2          # d/du(u^3) at u=g(x)
    h_prime_at_f = 1.0 / f(g(x))          # d/dv(ln v) at v=f(g(x))
    chain_rule_result = h_prime_at_f * f_prime_at_g * g_prime

    numeric_result = numeric_derivative(y, x)

    print(f"x = {x}")
    print(f"g'(x) = {g_prime}")
    print(f"f'(g(x)) = {f_prime_at_g}")
    print(f"h'(f(g(x))) = {h_prime_at_f:.6f}")
    print(f"chain rule product = {chain_rule_result:.6f}")
    print(f"numeric derivative  = {numeric_result:.6f}")
    print(f"match: {abs(chain_rule_result - numeric_result) < 1e-4}\n")


def demo_neuron_as_a_chain():
    """
    A single artificial neuron is exactly a 2-link chain:
        z = w*x + b          (linear combination)
        a = sigmoid(z)        (activation)
    da/dw is what backprop needs to update the weight, and the chain
    rule gives it directly: da/dw = sigmoid'(z) * x
    """
    _rule_line()
    print("3) A single neuron IS a chain: a = sigmoid(w*x + b)")
    _rule_line()
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    sigmoid_prime = lambda z: sigmoid(z) * (1 - sigmoid(z))

    w, x, b = 0.8, 2.0, -0.3
    z = w * x + b
    a = sigmoid(z)

    dz_dw = x                              # d/dw(w*x + b) = x
    da_dz = sigmoid_prime(z)               # d/dz(sigmoid(z))
    da_dw = da_dz * dz_dw                  # chain rule

    # verify numerically by nudging w directly
    def a_of_w(w_val):
        return sigmoid(w_val * x + b)
    numeric_da_dw = numeric_derivative(a_of_w, w)

    print(f"w={w}, x={x}, b={b}")
    print(f"z = w*x + b = {z:.6f}")
    print(f"a = sigmoid(z) = {a:.6f}")
    print(f"dz/dw = x = {dz_dw}")
    print(f"da/dz = sigmoid'(z) = {da_dz:.6f}")
    print(f"chain rule: da/dw = da/dz * dz/dw = {da_dw:.6f}")
    print(f"numeric da/dw = {numeric_da_dw:.6f}")
    print(f"match: {abs(da_dw - numeric_da_dw) < 1e-4}")
    print("\nThis is literally one step of backpropagation: to know how")
    print("much a weight contributed to the output, multiply the local")
    print("derivatives along the path from that weight to the output.\n")


def demo_backprop_through_two_neurons():
    """
    Two neurons stacked: a1 = sigmoid(w1*x), a2 = sigmoid(w2*a1).
    To get da2/dw1 you must chain THROUGH a1: the gradient flows
    backward from a2, through a1's local derivative, down to w1.
    This is exactly the "back" in backpropagation.
    """
    _rule_line()
    print("4) Two stacked neurons -- gradient flowing backward")
    _rule_line()
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    sigmoid_prime = lambda z: sigmoid(z) * (1 - sigmoid(z))

    x, w1, w2 = 1.0, 0.5, 1.2
    z1 = w1 * x
    a1 = sigmoid(z1)
    z2 = w2 * a1
    a2 = sigmoid(z2)

    # da2/dw1 = da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
    da2_dz2 = sigmoid_prime(z2)
    dz2_da1 = w2
    da1_dz1 = sigmoid_prime(z1)
    dz1_dw1 = x
    da2_dw1 = da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1

    def a2_of_w1(w1_val):
        z1_ = w1_val * x
        a1_ = sigmoid(z1_)
        z2_ = w2 * a1_
        return sigmoid(z2_)

    numeric_result = numeric_derivative(a2_of_w1, w1)

    print(f"a1 = {a1:.6f}, a2 = {a2:.6f}")
    print(f"chain: da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1")
    print(f"     = {da2_dz2:.4f} * {dz2_da1:.4f} * {da1_dz1:.4f} * {dz1_dw1:.4f}")
    print(f"     = {da2_dw1:.6f}")
    print(f"numeric da2/dw1 = {numeric_result:.6f}")
    print(f"match: {abs(da2_dw1 - numeric_result) < 1e-4}\n")


if __name__ == "__main__":
    demo_single_composition()
    demo_three_link_chain()
    demo_neuron_as_a_chain()
    demo_backprop_through_two_neurons()
