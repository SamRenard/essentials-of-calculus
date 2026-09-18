# Day 43 — Calculus Basics (Derivative Intuition)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 1/5 (new block) · 4 hours total

## What this covers

The opening theory block of a new math sub-block: the derivative as
an instantaneous rate of change, following the intuition built in
3Blue1Brown's "Essence of Calculus" episodes 1–3. The practical block
implements a general numeric derivative function using finite
differences and validates it against 5 functions with known
analytical derivatives.

## Files

| File | Description | Status |
|---|---|---|
| `derivative_anatomy.py` | Average vs instantaneous slope, dy/dx as a limit, forward/backward/central difference comparison, derivative-as-a-function | ✅ runs clean |
| `numeric_derivative.py` | Central-difference numeric derivative, tested on 5 functions (x², x³, sin, eˣ, ln) | ✅ max error ~2.6e-10 |
| `test_exercises.py` | 11 tests covering accuracy, convergence, and the finite-difference schemes | ✅ 11/11 PASS |
| `notes.md` | Short Azerbaijani konspekt | ✅ |
| `README.md` | This file | ✅ |

## The core idea

```
f'(x) ≈ [f(x+h) - f(x-h)] / (2h)     (central difference)
```

As h shrinks toward 0, this ratio converges to the true derivative —
central difference converges faster (error ~ h²) than forward or
backward difference (error ~ h).

## How to run

```bash
python derivative_anatomy.py     # prints the intuition-building demos
python numeric_derivative.py     # prints numeric vs analytical derivative table
python test_exercises.py         # runs the 11-test suite (or use pytest)
```

## Sources used today

- 3Blue1Brown — "Essence of Calculus", episodes 1–3
- Khan Academy — Calculus
