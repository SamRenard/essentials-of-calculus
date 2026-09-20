# Day 45 — Calculus Basics (Partial Derivatives & Gradients)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 3/5 · 4 hours total

## What this covers

Today's theory moves from one variable to many: partial derivatives
(how f changes if only one input moves), the gradient vector (all
partials collected together, pointing toward steepest increase), and
the multivariable chain rule. The practical block implements gradient
descent from scratch on a 2D function and plots the descent path over
a contour map with matplotlib.

## Files

| File | Description | Status |
|---|---|---|
| `partial_derivatives_anatomy.py` | Partial derivatives, the gradient vector, uphill vs downhill, multivariable chain rule | ✅ runs clean, all match |
| `gradient_descent_2d.py` | Gradient descent from scratch on a curved 2D bowl, saves `gradient_descent.png` | ✅ converges to true minimum |
| `test_exercises.py` | 10 tests covering gradients, convergence, and the plotting function | ✅ 10/10 PASS |
| `notes.md` | Short Azerbaijani konspekt | ✅ |
| `README.md` | This file | ✅ |
| `gradient_descent.png` | Generated when `gradient_descent_2d.py` runs — contour map + descent path | generated on run |

## The core idea

```
gradient = [df/dx, df/dy, ...]           -> points toward steepest INCREASE
new_point = point - learning_rate * gradient   -> steepest DECREASE (descent)
```

Starting at `(8, -3)` on a curved bowl with true minimum at `(3, 2)`,
60 steps of gradient descent (learning rate 0.3) land within
`0.0003` of the true minimum.

## How to run

```bash
python partial_derivatives_anatomy.py   # prints partial/gradient demos
python gradient_descent_2d.py           # runs descent, saves gradient_descent.png
python test_exercises.py                # runs the 10-test suite (or use pytest)
```

## Sources used today

- 3Blue1Brown — "Essence of Calculus" (gradient descent connection)
- Khan Academy — multivariable calculus
