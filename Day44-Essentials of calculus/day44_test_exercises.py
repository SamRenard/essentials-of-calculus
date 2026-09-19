"""
test_exercises.py
Day 44 -- test suite for hand_derivative_verification.py and chain_rule_anatomy.py
"""

import numpy as np
import hand_derivative_verification as hd
import chain_rule_anatomy as cr


def _check(f, hand_derivative, x, tol=1e-4):
    numeric = hd.numeric_derivative(f, x)
    hand = hand_derivative(x)
    assert abs(numeric - hand) < tol, f"x={x}: hand={hand}, numeric={numeric}"


def test_f1_sin_of_polynomial():
    for x in [-2.0, 0.0, 0.5, 1.5]:
        _check(hd.f1, hd.f1_hand_derivative, x)


def test_f2_gaussian_bump():
    for x in [-1.5, 0.0, 0.5, 2.0]:
        _check(hd.f2, hd.f2_hand_derivative, x)


def test_f3_log_of_sin_squared():
    for x in [0.2, 1.0, 2.0, 3.0]:
        _check(hd.f3, hd.f3_hand_derivative, x)


def test_f4_sqrt_of_polynomial():
    for x in [0.5, 1.0, 2.0, 3.0]:
        _check(hd.f4, hd.f4_hand_derivative, x)


def test_f5_sigmoid_squared():
    for x in [-2.0, -0.5, 0.5, 2.0]:
        _check(hd.f5, hd.f5_hand_derivative, x)


def test_run_all_verifications_reports_true():
    assert hd.run_all_verifications() is True


# ---------------------------------------------------------------- chain_rule_anatomy.py
def test_single_composition_matches_numeric():
    g = lambda x: x ** 2
    f = lambda u: np.sin(u)
    y = lambda x: f(g(x))
    x = 1.5
    chain_result = np.cos(g(x)) * (2 * x)
    numeric_result = cr.numeric_derivative(y, x)
    assert abs(chain_result - numeric_result) < 1e-4


def test_three_link_chain_matches_numeric():
    g = lambda x: 2 * x
    f = lambda u: u ** 3
    h = lambda v: np.log(v)
    y = lambda x: h(f(g(x)))
    x = 2.0
    chain_result = (1.0 / f(g(x))) * (3 * g(x) ** 2) * 2.0
    numeric_result = cr.numeric_derivative(y, x)
    assert abs(chain_result - numeric_result) < 1e-4


def test_neuron_chain_rule_matches_numeric():
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    sigmoid_prime = lambda z: sigmoid(z) * (1 - sigmoid(z))
    w, x, b = 0.8, 2.0, -0.3
    z = w * x + b
    da_dw = sigmoid_prime(z) * x

    def a_of_w(w_val):
        return sigmoid(w_val * x + b)

    numeric_result = cr.numeric_derivative(a_of_w, w)
    assert abs(da_dw - numeric_result) < 1e-4


def test_two_stacked_neurons_matches_numeric():
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    sigmoid_prime = lambda z: sigmoid(z) * (1 - sigmoid(z))
    x, w1, w2 = 1.0, 0.5, 1.2
    z1 = w1 * x
    a1 = sigmoid(z1)
    z2 = w2 * a1
    da2_dw1 = sigmoid_prime(z2) * w2 * sigmoid_prime(z1) * x

    def a2_of_w1(w1_val):
        z1_ = w1_val * x
        a1_ = sigmoid(z1_)
        z2_ = w2 * a1_
        return sigmoid(z2_)

    numeric_result = cr.numeric_derivative(a2_of_w1, w1)
    assert abs(da2_dw1 - numeric_result) < 1e-4


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}  ->  {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
