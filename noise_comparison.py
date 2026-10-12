"""Matched training/testing comparison of two geometry-noise generators."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from noise_models import GEOMETRY_COLS, SHAPE_COLS, SHAPES, add_geometry_noise


def compare_noise_methods(
    X_train_clean, X_test_clean, y_train, y_test,
    target="Part1_E", noise_level=0.05,
    train_seed=42, test_seed=43,
    n_estimators=100, bottom_jitter=0.01,
):
    """Train the same RF separately with Gaussian and radial geometry noise.

    Same clean split, target, features, model hyperparameters, and seed policy.
    Returns (metrics DataFrame, predicted target arrays by method).
    """
    results, predictions = [], {}
    true = np.asarray(y_test[target]).ravel()
    for method in ("gaussian", "radial"):
        noisy_train = add_geometry_noise(
            X_train_clean, method=method, noise_level=noise_level,
            seed=train_seed, bottom_jitter=bottom_jitter
        )
        noisy_test = add_geometry_noise(
            X_test_clean, method=method, noise_level=noise_level,
            seed=test_seed, bottom_jitter=bottom_jitter
        )
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
        model.fit(noisy_train[GEOMETRY_COLS], y_train[target])
        predicted = model.predict(noisy_test[GEOMETRY_COLS])
        predictions[method] = predicted
        perturb = (noisy_test[SHAPE_COLS].to_numpy(dtype=float)
                   - X_test_clean[SHAPE_COLS].to_numpy(dtype=float))
        results.append({
            "method": method,
            "noise_level": noise_level,
            "R2": r2_score(true, predicted),
            "prediction_MAE": mean_absolute_error(true, predicted),
            "shape_coordinate_MAE": np.mean(np.abs(perturb)),
            "shape_coordinate_RMS": np.sqrt(np.mean(perturb ** 2)),
        })
    return pd.DataFrame(results), predictions


def plot_noise_example(X_clean, sample=0, noise_level=0.05, seed=43):
    """Overlay clean, Gaussian, radial contours for one held-out sample."""
    import matplotlib.pyplot as plt

    if not (0 <= sample < len(X_clean)):
        raise IndexError("sample is out of range")
    versions = {
        "Clean": X_clean,
        "Gaussian": add_geometry_noise(X_clean, "gaussian", noise_level, seed),
        "Radial": add_geometry_noise(X_clean, "radial", noise_level, seed),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, prefix in zip(axes, SHAPES):
        for name, frame in versions.items():
            x = frame[[f"{prefix}_x{i}" for i in range(1, 10)]].iloc[sample].to_numpy()
            y = frame[[f"{prefix}_y{i}" for i in range(1, 10)]].iloc[sample].to_numpy()
            ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], "-o", markersize=3, label=name)
        ax.set_title(prefix)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f"Geometry noise comparison | level={noise_level} | sample={sample}")
    fig.tight_layout()
    return fig, axes
