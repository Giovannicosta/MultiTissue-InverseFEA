"""Exact spline preprocessing from Notebook 02, Step 2.

Inputs: raw original dataset DataFrame with 54 named geometry features.
Outputs: DataFrame with the same feature names and original row index.
Bottom coordinate columns are unchanged. The 9-point inner/outer contours
are periodically interpolated on a *uniform point-index parameter*, not
chord length (Notebook 01 used a different parameterization).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

BOTTOM_COLS = ([f"inner_y{i}" for i in range(1, 10)] +
               [f"inner_z{i}" for i in range(1, 10)])
INNER_COLS = ([f"innerShape_x{i}" for i in range(1, 10)] +
              [f"innerShape_y{i}" for i in range(1, 10)])
OUTER_COLS = ([f"outerShape_x{i}" for i in range(1, 10)] +
              [f"outerShape_y{i}" for i in range(1, 10)])
GEOMETRY_COLS = BOTTOM_COLS + INNER_COLS + OUTER_COLS


def smooth_contour(x, y, n_points=9):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.linspace(0, 1, len(x) + 1)
    x_closed = np.r_[x, x[0]]
    y_closed = np.r_[y, y[0]]
    spline_x = CubicSpline(t, x_closed, bc_type="periodic")
    spline_y = CubicSpline(t, y_closed, bc_type="periodic")
    t_new = np.linspace(0, 1, n_points, endpoint=False)
    return spline_x(t_new), spline_y(t_new)


def preprocess_geometry(part):
    """Transform original geometry into Notebook-02 expert input space."""
    if not isinstance(part, pd.DataFrame):
        raise TypeError("Provide original geometry as a pandas DataFrame")
    missing = [c for c in GEOMETRY_COLS if c not in part.columns]
    if missing:
        raise ValueError(f"Missing geometry columns: {missing}")
    X = part[GEOMETRY_COLS].copy()
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Geometry contains non-finite values")
    for prefix in ["innerShape", "outerShape"]:
        x_cols = [f"{prefix}_x{i}" for i in range(1, 10)]
        y_cols = [f"{prefix}_y{i}" for i in range(1, 10)]
        x_values = X[x_cols].to_numpy()
        y_values = X[y_cols].to_numpy()
        processed = [smooth_contour(x, y) for x, y in zip(x_values, y_values)]
        if processed:
            X.loc[:, x_cols] = np.array([item[0] for item in processed])
            X.loc[:, y_cols] = np.array([item[1] for item in processed])
    return X
