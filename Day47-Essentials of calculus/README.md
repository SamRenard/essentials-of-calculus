# Day 47 — Convexity, Local/Global Minima & Momentum-based Gradient Descent

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 5/5

## 📌 Concept

A **convex function** is one where the line between any two points on
it always stays above the function itself — practically, a single
smooth bowl: wherever you start, going downhill leads straight to the
global minimum. A **local minimum** is lower than its neighbors, but
a **global minimum** is the lowest point overall; non-convex surfaces
(like a deep network's loss landscape) can have many local minima.

**Momentum** borrows a physics idea: imagine a ball rolling downhill.
Instead of only looking at the current gradient, it keeps a running
"velocity": `v = β·v - η·∇f(x)`, then `x += v`. The real payoff shows
up on **narrow valleys** — surfaces much steeper in one direction than
another. Vanilla GD has to keep its step size small enough to stay
safe in the steep direction, so it barely progresses in the shallow
direction and can stall partway to the minimum. Momentum's accumulated
velocity keeps carrying it forward in the shallow direction even while
it oscillates, so it reaches the minimum — vanilla GD, at the same
learning rate, does not.

## 📂 Files

| File | Description |
|---|---|
| `momentum_gd.py` | From-scratch vanilla GD and momentum GD (1D and 2D), a numeric convexity checker, a convex bowl, an asymmetric double-well (local vs global minimum), and a narrow 2D valley |
| `test_momentum.py` | 8 automated assertion-based tests |
| `visualize_convergence.py` | Generates two comparison plots (loss curve on the convex bowl; trajectory paths on the narrow valley) |
| `convex_convergence.png` | Generated output — momentum reaches a given loss threshold in far fewer steps |
| `narrow_valley_paths.png` | Generated output — vanilla GD stalls before the minimum; momentum, despite oscillating, reaches it |

## ▶️ Usage

```bash
pip install numpy matplotlib
python3 momentum_gd.py            # run all demonstrations with printed output
python3 test_momentum.py          # automated correctness checks
python3 visualize_convergence.py  # generate the two comparison plots
```

## ✅ Test Results

```
8/8 checks passed
```

Covers: convexity detection on both a convex and a non-convex function,
convergence on the convex bowl for both algorithms, momentum reaching
a loss threshold in fewer steps, `beta=0` reducing momentum exactly to
vanilla GD, correct local/global minimum values on the double-well,
and — the key result — momentum both oscillating less *and* reaching
dramatically lower loss than vanilla GD on the narrow valley (0.0000
vs 0.2327 after 60 identical steps).

## 📚 Sources

- 3Blue1Brown — Essence of Calculus
- Khan Academy — Calculus

## 🔁 Refactor Log

- Initial implementation: 1D vanilla/momentum GD plus a numeric convexity check
- Added an asymmetric double-well function to demonstrate local vs. global minima
- Added a 2D narrow-valley function after discovering the 1D double-well doesn't
  cleanly separate the two algorithms — the valley is where momentum's advantage
  is unambiguous and matches the textbook explanation
- Tuned `lr`/`beta` by grid search so the comparison shows a real, non-trivial gap
- Added matplotlib visualizations for both scenarios

---
*NIZAM AI Day 47/150*
