"""Noise generators for already spline-resampled inverse-FEA geometry.

Both generators preserve DataFrame columns and sample ordering. The radial mode
applies a smooth angular displacement along center-to-contour-point rays; it
does not require explicit intersections between arbitrary rays and the spline.
"""

import numpy as np
import pandas as pd

BOTTOM_COLS = ([f"inner_y{i}" for i in range(1, 10)] +
               [f"inner_z{i}" for i in range(1, 10)])
SHAPES = ("innerShape", "outerShape")


def shape_columns(prefix):
    """The x and y columns for one nine-point closed contour."""
    return ([f"{prefix}_x{i}" for i in range(1, 10)] +
            [f"{prefix}_y{i}" for i in range(1, 10)])


SHAPE_COLS = shape_columns("innerShape") + shape_columns("outerShape")
GEOMETRY_COLS = BOTTOM_COLS + SHAPE_COLS


def add_geometry_noise(
    X_base,
    method="gaussian",
    noise_level=0.05,
    seed=42,
    bottom_jitter=0.01,
    harmonics=3,
):
    """Add Gaussian coordinate noise or smooth center-ray (radial) noise.

    The Gaussian branch reproduces the original add_fast_noise() RNG sequence:
    uniform bottom jitter followed by Gaussian noise on 36 shape coordinates.

    Radial mode perturbs the radii of existing spline-resampled contour points
    with a low-frequency Fourier field in polar angle. Its displacement is
    normalized so the per-sample shape-coordinate RMS is approximately equal
    to `noise_level`, making it comparable to Gaussian standard deviation.
    Negative radii are prevented by limiting inward displacement to 90% of
    each point's original radius. This safety limit can lower the actual RMS,
    so measured perturbation RMS should be reported with model metrics.

    Both modes use identical bottom jitter drawn from the same seed.
    A zero noise_level does *not* remove bottom jitter unless bottom_jitter=0.
    """
    if not isinstance(X_base, pd.DataFrame):
        raise TypeError("X_base must be a pandas DataFrame")
    if method not in ("gaussian", "radial"):
        raise ValueError("method must be 'gaussian' or 'radial'")
    if noise_level < 0 or bottom_jitter < 0:
        raise ValueError("noise_level and bottom_jitter must be nonnegative")
    if not isinstance(harmonics, int) or harmonics < 1:
        raise ValueError("harmonics must be a positive integer")
    missing = [c for c in GEOMETRY_COLS if c not in X_base.columns]
    if missing:
        raise ValueError(f"Missing geometry columns: {missing[:6]}")

    noisy = X_base.copy(deep=True)
    rng = np.random.default_rng(seed)
    noisy.loc[:, BOTTOM_COLS] = (
        X_base[BOTTOM_COLS].to_numpy(dtype=float)
        + rng.uniform(-bottom_jitter, bottom_jitter,
                      size=(len(X_base), len(BOTTOM_COLS)))
    )

    if method == "gaussian":
        noisy.loc[:, SHAPE_COLS] = (
            X_base[SHAPE_COLS].to_numpy(dtype=float)
            + rng.normal(0.0, noise_level,
                         size=(len(X_base), len(SHAPE_COLS)))
        )
        return noisy

    # One independent, smoothly varying angular displacement field per shape.
    k = np.arange(1, harmonics + 1, dtype=float)
    weights = 1.0 / (k ** 1.5)  # suppress jagged high-frequency deformations
    for prefix in SHAPES:
        x_cols = [f"{prefix}_x{i}" for i in range(1, 10)]
        y_cols = [f"{prefix}_y{i}" for i in range(1, 10)]
        x = X_base[x_cols].to_numpy(dtype=float)
        y = X_base[y_cols].to_numpy(dtype=float)
        cx = x.mean(axis=1, keepdims=True)
        cy = y.mean(axis=1, keepdims=True)
        dx, dy = x - cx, y - cy
        radius = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)

        # sin/cos harmonics yield a smooth, periodic function of angle.
        a = rng.normal(size=(len(X_base), harmonics)) * weights
        b = rng.normal(size=(len(X_base), harmonics)) * weights
        angles = theta[:, :, None] * k[None, None, :]
        field = (np.cos(angles) * a[:, None, :]
                 + np.sin(angles) * b[:, None, :]).sum(axis=2)
        field_rms = np.sqrt(np.mean(field ** 2, axis=1, keepdims=True))
        delta_radius = (field / np.maximum(field_rms, 1e-12)
                        * (np.sqrt(2.0) * noise_level))

        # Never let a ray's endpoint cross through the chosen center.
        delta_radius = np.maximum(delta_radius, -0.9 * radius)
        unit_x = np.divide(dx, radius, out=np.zeros_like(dx),
                           where=radius > 1e-12)
        unit_y = np.divide(dy, radius, out=np.zeros_like(dy),
                           where=radius > 1e-12)
        noisy.loc[:, x_cols] = x + delta_radius * unit_x
        noisy.loc[:, y_cols] = y + delta_radius * unit_y
    return noisy
