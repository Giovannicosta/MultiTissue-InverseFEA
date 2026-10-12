"""Smoke tests for stable legacy Gaussian behavior and radial displacements."""
import numpy as np
import pandas as pd
from noise_models import GEOMETRY_COLS, BOTTOM_COLS, SHAPE_COLS, add_geometry_noise
from noise_comparison import compare_noise_methods

rng = np.random.default_rng(5)
N = 48
x = pd.DataFrame(rng.normal(0, .1, (N, len(GEOMETRY_COLS))), columns=GEOMETRY_COLS)
for prefix, r in (("innerShape", .4), ("outerShape", .9)):
    for i, t in enumerate(np.linspace(0, 2*np.pi, 9, endpoint=False), 1):
        x[f"{prefix}_x{i}"] = r*np.cos(t) + rng.normal(0, .002, N)
        x[f"{prefix}_y{i}"] = r*np.sin(t) + rng.normal(0, .002, N)

old_rng = np.random.default_rng(42)
expected = x.copy()
expected[BOTTOM_COLS] += old_rng.uniform(-.01, .01, size=(N, len(BOTTOM_COLS)))
expected[SHAPE_COLS] += old_rng.normal(0, .05, size=(N, len(SHAPE_COLS)))
got = add_geometry_noise(x, "gaussian", .05, 42)
assert np.allclose(got[GEOMETRY_COLS], expected[GEOMETRY_COLS])
assert got.equals(add_geometry_noise(x, "gaussian", .05, 42))
rad = add_geometry_noise(x, "radial", .05, 42)
assert rad.shape == x.shape and rad.index.equals(x.index)
assert not np.allclose(x[SHAPE_COLS], rad[SHAPE_COLS])
assert np.allclose(rad[BOTTOM_COLS], got[BOTTOM_COLS])
assert rad.equals(add_geometry_noise(x, "radial", .05, 42))
rad_zero = add_geometry_noise(x, "radial", 0, 42, bottom_jitter=0)
assert np.allclose(rad_zero[GEOMETRY_COLS], x[GEOMETRY_COLS])
for prefix in ("innerShape", "outerShape"):
    px = x[[f"{prefix}_x{i}" for i in range(1,10)]].to_numpy()
    py = x[[f"{prefix}_y{i}" for i in range(1,10)]].to_numpy()
    qx = rad[[f"{prefix}_x{i}" for i in range(1,10)]].to_numpy()
    qy = rad[[f"{prefix}_y{i}" for i in range(1,10)]].to_numpy()
    ux = px-px.mean(axis=1, keepdims=True)
    uy = py-py.mean(axis=1, keepdims=True)
    vx, vy = qx-px, qy-py
    assert np.max(np.abs(ux*vy - uy*vx)) < 1e-10
train, test = x.iloc[:38], x.iloc[38:]
y = pd.DataFrame({"Part1_E": x["inner_y1"]*0.5 + 0.12}, index=x.index)
results, preds = compare_noise_methods(
    train, test, y.iloc[:38], y.iloc[38:], n_estimators=3
)
assert results.shape == (2, 6), list(results.columns)
assert set(preds) == {"gaussian", "radial"}
assert all(len(p) == len(test) for p in preds.values())
print("All geometry-noise and comparison smoke tests passed")
