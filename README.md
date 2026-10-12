# MultiTissue-InverseFEA
Multi-expert machine learning framework for predicting tissue material properties through inverse finite element analysis (Inverse FEA).


## Gaussian versus radial geometry noise

- [`noise_models.py`](noise_models.py): reusable `add_geometry_noise(..., method="gaussian"|"radial")` for already spline-resampled 54-column geometry.
- [`noise_comparison.py`](noise_comparison.py): matched Random Forest comparison (`compare_noise_methods`) and a shape overlay (`plot_noise_example`).
- [`test_noise_models.py`](test_noise_models.py): deterministic synthetic smoke tests.

From an existing Colab session containing `X_train_spline_base`, `X_test_spline_base`, `y_train`, and `y_test`:

```python
from noise_comparison import plot_noise_example, compare_noise_methods

plot_noise_example(X_test_spline_base, sample=0, noise_level=0.05)
results, predictions = compare_noise_methods(
    X_train_spline_base, X_test_spline_base,
    y_train, y_test, target="Part1_E", noise_level=0.05
)
display(results)
```

The two generators hold fixed the split, target, Random Forest settings, and bottom-coordinate jitter. Radial noise smoothly perturbs distances from the contour centroid along existing spline-sample rays. Coordinate RMS and MAE are reported because matching a requested Gaussian standard deviation does not guarantee identical realized corruption amplitudes. Further details and experimental history appear in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).
