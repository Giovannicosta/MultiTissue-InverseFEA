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


## Part1_E shape analysis and geometry denoising (2026-11-03)

The exploratory findings from Colab `04_Shape_Analysis_Comparison.ipynb` (Steps 1–32) are recorded in [**Part1_E Shape Analysis and Denoising — Research Findings**](research/PART1E_SHAPE_ANALYSIS_2026-11-03.md).

On the repeatedly explored 1,302-row validation split under radial noise 0.05 (with bottom jitter 0.01), a Bottom + Inner geometry denoiser followed by an **out-of-fold reconstruction-aware Part1_E Random Forest** achieved **R² = 0.7480, MAE = 0.006762**, compared with **R² = 0.632186, MAE = 0.008694** for denoising followed by a clean-trained RF and **R² = 0.507630, MAE = 0.010204** for direct radial-noise-specialist prediction. This is an **exploratory validation result, not a final generalization claim**. The report covers clean/PCA baselines, noise ablations, Kernel PCA/autoencoder comparisons, geometry descriptors, regional feature importance, reconstruction diagnostics, limitations, and the completed five-fold cross-fitted Step 32 experiment.

The Colab notebook, raw simulation dataset, trained denoiser, and generated figures are not tracked in this repository yet.