# Day 44 — Calculus Basics (Chain Rule / Backprop's Heart)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 2/5 · 4 hours total

## What this covers

Today's theory is the chain rule — `dy/dx = f'(g(x)) * g'(x)` — built
up from a single composition to a 3-link chain to an actual sigmoid
neuron, ending with two stacked neurons to show exactly how a gradient
flows *backward* through a network (the "back" in backpropagation).
The practical block hand-derives 5 composite functions with the chain
rule and verifies every derivative numerically.

## Files

| File | Description | Status |
|---|---|---|
| `chain_rule_anatomy.py` | Chain rule from a single composition up to a 2-neuron backprop example, each checked against a numeric derivative | ✅ runs clean, all match |
| `hand_derivative_verification.py` | 5 composite functions hand-derived step by step, verified against finite-difference derivatives | ✅ 20/20 checks OK |
| `test_exercises.py` | 10 tests covering every hand derivation + every chain-rule demo | ✅ 10/10 PASS |
| `notes.md` | Short Azerbaijani konspekt | ✅ |
| `README.md` | This file | ✅ |

## The core idea

```
y = f(g(x))   =>   dy/dx = f'(g(x)) * g'(x)
```

A neural network is a long chain of compositions (layer after layer).
Backpropagation is the chain rule applied repeatedly: to find how much
a weight deep inside the network affected the final output, multiply
the local derivatives along the path from that weight to the output.

## How to run

```bash
python chain_rule_anatomy.py             # prints the composition -> neuron demos
python hand_derivative_verification.py   # prints hand vs numeric derivative table
python test_exercises.py                 # runs the 10-test suite (or use pytest)
```

## Sources used today

- 3Blue1Brown — "Essence of Calculus", episode 4 (chain rule)
- Khan Academy — Calculus
