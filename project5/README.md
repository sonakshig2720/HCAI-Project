
# Project 5 (RLHF) — drop-in Django app

## Quick wire-up
1) Add to `INSTALLED_APPS` in `settings.py`:
```python
'project5',
```

2) In the root `urls.py`, include:
```python
path('project5/', include('project5.urls')),
```

3) Ensure your base template `base.html` exists (provided in the skeleton).

## What it does
- Trains a baseline policy with REINFORCE in a 5×5 grid (mouse, walls, traps, normal & organic cheese).
- Samples trajectories; you provide pairwise preferences.
- Fits a learned reward with a Bradley–Terry preference model.
- Fine-tunes the policy with a KL penalty toward the baseline (simple, pedagogical version).

> Numpy-only implementation (no torch dependency) for portability in grading.
