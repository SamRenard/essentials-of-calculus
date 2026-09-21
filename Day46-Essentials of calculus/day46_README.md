# Day 46 — Calculus Basics (Gradient Descent Variants & Learning Rate)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 4/5 · 4 hours total

## What this covers

Today's theory is the single most practical knob in gradient-based
training: the learning rate. Too small and convergence crawls; too
large and it overshoots, oscillates, or diverges outright; there's an
exact threshold where the behavior flips. The theory file also
compares batch, stochastic, mini-batch, and momentum GD side by side.
The practical block builds linear regression entirely from scratch
with NumPy and plots loss curves across five learning rates on the
same dataset.

## Files

| File | Description | Status |
|---|---|---|
| `learning_rate_anatomy.py` | Too-small/just-right/too-large demo, the exact divergence threshold, and batch/SGD/mini-batch/momentum compared | ✅ runs clean |
| `linear_regression_gd.py` | From-scratch linear regression trained with batch GD at 5 learning rates, saves `loss_curves.png` | ✅ runs clean |
| `test_exercises.py` | 10 tests covering convergence, divergence, and the plotting function | ✅ 10/10 PASS |
| `notes.md` | Short Azerbaijani konspekt | ✅ |
| `README.md` | This file | ✅ |
| `loss_curves.png` | Generated when `linear_regression_gd.py` runs | generated on run |

## Measured result (this run)

| learning rate | final loss | verdict |
|---|---|---|
| 0.001 | 4.5046 | too slow |
| 0.01 | 0.3862 | converged |
| 0.05 | 0.3752 | converged |
| 0.15 | 0.3752 | converged |
| 0.35 | ~4.3 × 10²⁶ | diverged |

True parameters: w = 2.0, b = 1.0. Noise floor loss ≈ 0.38 (the best
any fit can do given the added noise), so lr = 0.05 and lr = 0.15
both essentially reached the true optimum.

## The exact divergence threshold (1D bowl case)

For `f(x) = (x-a)²`, each step multiplies the distance to the minimum
by `|1 - 2·lr|`. Convergence needs this factor `< 1`, i.e. `0 < lr < 1`
— cross that boundary and the distance grows every step instead of
shrinking.

## How to run

```bash
python learning_rate_anatomy.py   # prints the lr regimes + GD variants demo
python linear_regression_gd.py    # trains at 5 lrs, saves loss_curves.png
python test_exercises.py          # runs the 10-test suite (or use pytest)
```

## Sources used today

- 3Blue1Brown — "Essence of Calculus" (gradient descent)
- Khan Academy — Calculus
