# Development Log

This file documents the step-by-step development of the MultiTissue-InverseFEA pipeline.

## Current direction

The project is being rebuilt as a cleaner inverse FEA pipeline that can later support multiple specialized expert models for different tissue configurations.

### Validated preprocessing findings

- Source dataset: `Spring_25/2025_4_14_intermediate.csv` from `SomeoneJN/BME_Muscle_Strength_Prediction_FA25`.
- Dataset size: 6,510 rows and 99 columns.
- Variable material-property targets:
  - `Part1_E`
  - `Part3_E`
  - `Part11_E`
- `Part7_E` and `Part10_E` are constant at 1.0 and are not useful regression targets.
- Geometry subset currently used: 54 coordinates from Bottom, Inner Shape, and Outer Shape.
- Train/test split: 5,208 training samples and 1,302 test samples.
- The older Ver11 preprocessing code incorrectly fits PCA separately on training and test data.
- Correct approach: fit centering + PCA on clean training geometry only, then use the same fitted transformation for noisy training/test observations.
- Clean geometry is highly low-dimensional:
  - Bottom: 1 PC >= 95% variance, 2 PCs >= 99%.
  - Inner Shape: 1 PC >= 95% variance, 2 PCs >= 99%.
  - Outer Shape: 1 PC >= 99% variance.
- Noise level 0.3 injects strong high-dimensional variation and causes Inner/Outer PCA to require about 16 PCs for 90% noisy variance.
- Fitting PCA on clean geometry and projecting noisy observations preserves the dominant shape mode well.
- At noise level 0.05:
  - Inner PC1 clean/noisy correlation: 0.9903.
  - Outer PC1 clean/noisy correlation: 0.9885.
  - Bottom PC1 correlation: 0.9999.
  - Bottom PC2 correlation: 0.9937.
- Current candidate compact representation:
  - Bottom PC1 + PC2
  - Inner PC1
  - Outer PC1
  - Total: 4 features instead of 54 raw geometry coordinates.

## Clean Colab rebuild

A fresh notebook is now being built step-by-step so only validated code is retained.

### Cell 1 - Load the source dataset

```python
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/SomeoneJN/BME_Muscle_Strength_Prediction_FA25/main/Spring_25/2025_4_14_intermediate.csv"

df = pd.read_csv(url)

print("Dataset shape:", df.shape)
df.head()
```

Validated output:

```text
Dataset shape: (6510, 99)
```

### Cell 2 - Define regression targets and geometry features

```python
target_columns = [
    "Part1_E",
    "Part3_E",
    "Part11_E"
]

feature_columns = (
    [f"inner_y{i}" for i in range(1, 10)] +
    [f"inner_z{i}" for i in range(1, 10)] +
    [f"innerShape_x{i}" for i in range(1, 10)] +
    [f"innerShape_y{i}" for i in range(1, 10)] +
    [f"outerShape_x{i}" for i in range(1, 10)] +
    [f"outerShape_y{i}" for i in range(1, 10)]
)

X = df[feature_columns].copy()
y = df[target_columns].copy()

print("Geometry inputs:", X.shape)
print("Targets:", y.shape)
print("Missing input values:", X.isna().sum().sum())
print("Missing target values:", y.isna().sum().sum())
```

Validated output:

```text
Geometry inputs: (6510, 54)
Targets: (6510, 3)
Missing input values: 0
Missing target values: 0
```

### Cell 3 - Reproducible train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```

Validated output:

```text
X_train: (5208, 54)
X_test: (1302, 54)
y_train: (5208, 3)
y_test: (1302, 3)
```

### Cell 4 - Define geometry groups

```python
bottom_cols = (
    [f"inner_y{i}" for i in range(1, 10)] +
    [f"inner_z{i}" for i in range(1, 10)]
)

inner_shape_cols = (
    [f"innerShape_x{i}" for i in range(1, 10)] +
    [f"innerShape_y{i}" for i in range(1, 10)]
)

outer_shape_cols = (
    [f"outerShape_x{i}" for i in range(1, 10)] +
    [f"outerShape_y{i}" for i in range(1, 10)]
)

print("Bottom columns:", len(bottom_cols))
print("Inner shape columns:", len(inner_shape_cols))
print("Outer shape columns:", len(outer_shape_cols))
```

Validated output:

```text
Bottom columns: 18
Inner shape columns: 18
Outer shape columns: 18
```

### Cell 5 - Define spline resampling helpers

```python
from scipy.interpolate import splprep, splev

def resample_closed_shape(row, shape_x_cols, shape_y_cols):
    x = row[shape_x_cols].to_numpy(dtype=float)
    y = row[shape_y_cols].to_numpy(dtype=float)

    tck, _ = splprep([x, y], s=0, per=True)

    u_new = np.linspace(0, 1, 100)
    x_spline, y_spline = splev(u_new, tck)

    indices = np.linspace(0, 99, 10, endpoint=True).astype(int)

    return x_spline[indices][:9], y_spline[indices][:9]


def build_spline_base(X):
    base = X.copy()

    for idx, row in X.iterrows():
        inner_x, inner_y = resample_closed_shape(
            row,
            inner_shape_cols[:9],
            inner_shape_cols[9:]
        )

        outer_x, outer_y = resample_closed_shape(
            row,
            outer_shape_cols[:9],
            outer_shape_cols[9:]
        )

        base.loc[idx, inner_shape_cols[:9]] = inner_x
        base.loc[idx, inner_shape_cols[9:]] = inner_y
        base.loc[idx, outer_shape_cols[:9]] = outer_x
        base.loc[idx, outer_shape_cols[9:]] = outer_y

    return base
```

Validated: Cell 5 ran successfully with no errors.

### Cell 6 - Build clean spline bases

```python
X_train_spline_base = build_spline_base(X_train)
X_test_spline_base = build_spline_base(X_test)

print("Train spline base:", X_train_spline_base.shape)
print("Test spline base:", X_test_spline_base.shape)
```

Validated output:

```text
Train spline base: (5208, 54)
Test spline base: (1302, 54)
```

### Cell 7 - Define fast noise injection

```python
def add_fast_noise(X_base, noise_level=0.05, seed=42):
    rng = np.random.default_rng(seed)
    noisy = X_base.copy()

    # Bottom geometry uses small uniform measurement noise.
    noisy[bottom_cols] += rng.uniform(
        -0.01,
        0.01,
        size=noisy[bottom_cols].shape
    )

    # Inner and outer shapes use Gaussian coordinate noise.
    shape_cols = inner_shape_cols + outer_shape_cols
    noisy[shape_cols] += rng.normal(
        0,
        noise_level,
        size=noisy[shape_cols].shape
    )

    return noisy
```

Validated: Cell 7 ran successfully with no errors.

### Cell 8 - Create noisy train/test observations

```python
X_train_noisy = add_fast_noise(
    X_train_spline_base,
    noise_level=0.05,
    seed=42
)

X_test_noisy = add_fast_noise(
    X_test_spline_base,
    noise_level=0.05,
    seed=43
)

print("Noisy train:", X_train_noisy.shape)
print("Noisy test:", X_test_noisy.shape)
```

Validated output:

```text
Noisy train: (5208, 54)
Noisy test: (1302, 54)
```

### Cell 9 - Fit PCA on clean training geometry only

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def fit_clean_pca(X_clean, cols, n_components=3):
    pipeline = Pipeline([
        ("center", StandardScaler(with_std=False)),
        ("pca", PCA(n_components=n_components))
    ])

    pipeline.fit(X_clean[cols])
    return pipeline

bottom_pca_clean = fit_clean_pca(
    X_train_spline_base,
    bottom_cols,
    n_components=3
)

inner_pca_clean = fit_clean_pca(
    X_train_spline_base,
    inner_shape_cols,
    n_components=3
)

outer_pca_clean = fit_clean_pca(
    X_train_spline_base,
    outer_shape_cols,
    n_components=3
)

print("PCA models fitted.")
```

Validated output:

```text
PCA models fitted.
```

### Cell 10 - Project noisy train/test data into the fixed clean PCA spaces

```python
bottom_train_scores = bottom_pca_clean.transform(
    X_train_noisy[bottom_cols]
)

bottom_test_scores = bottom_pca_clean.transform(
    X_test_noisy[bottom_cols]
)

inner_train_scores = inner_pca_clean.transform(
    X_train_noisy[inner_shape_cols]
)

inner_test_scores = inner_pca_clean.transform(
    X_test_noisy[inner_shape_cols]
)

outer_train_scores = outer_pca_clean.transform(
    X_train_noisy[outer_shape_cols]
)

outer_test_scores = outer_pca_clean.transform(
    X_test_noisy[outer_shape_cols]
)

print("Bottom train/test:", bottom_train_scores.shape, bottom_test_scores.shape)
print("Inner train/test:", inner_train_scores.shape, inner_test_scores.shape)
print("Outer train/test:", outer_train_scores.shape, outer_test_scores.shape)
```

Validated output:

```text
Bottom train/test: (5208, 3) (1302, 3)
Inner train/test: (5208, 3) (1302, 3)
Outer train/test: (5208, 3) (1302, 3)
```

### Cell 11 - Build the first compact 4-feature representation

```python
X_model_train = pd.DataFrame({
    "Bottom_PC1": bottom_train_scores[:, 0],
    "Bottom_PC2": bottom_train_scores[:, 1],
    "Inner_PC1": inner_train_scores[:, 0],
    "Outer_PC1": outer_train_scores[:, 0]
}, index=X_train.index)

X_model_test = pd.DataFrame({
    "Bottom_PC1": bottom_test_scores[:, 0],
    "Bottom_PC2": bottom_test_scores[:, 1],
    "Inner_PC1": inner_test_scores[:, 0],
    "Outer_PC1": outer_test_scores[:, 0]
}, index=X_test.index)

print("Model train:", X_model_train.shape)
print("Model test:", X_model_test.shape)
print(X_model_train.head())
```

Validated output:

```text
Model train: (5208, 4)
Model test: (1302, 4)
```

The compact representation uses:
- Bottom PC1
- Bottom PC2
- Inner PC1
- Outer PC1

This reduces the model input from 54 raw geometry coordinates to 4 PCA-derived features.

### Cell 12 - Train baseline Random Forest regressors

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

baseline_results = {}

for target in target_columns:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_model_train, y_train[target])

    predictions = model.predict(X_model_test)

    r2 = r2_score(y_test[target], predictions)
    mae = mean_absolute_error(y_test[target], predictions)
    rmse = mean_squared_error(
        y_test[target],
        predictions
    ) ** 0.5

    baseline_results[target] = {
        "model": model,
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }

    print(target)
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
```

Validated baseline results for the 4-feature representation:

```text
Part1_E
  R2:   -0.0422
  MAE:  0.015607
  RMSE: 0.018305

Part3_E
  R2:   0.9946
  MAE:  0.004655
  RMSE: 0.006378

Part11_E
  R2:   0.9561
  MAE:  0.006744
  RMSE: 0.008884
```

Interpretation:
- The compact PCA representation is extremely predictive for `Part3_E`.
- It is also strongly predictive for `Part11_E`.
- It does not currently recover `Part1_E` (negative R2), indicating that the dominant PCs retained so far likely omit information needed for that target, or that Part1_E is less identifiable from these geometry features alone.

### Cell 13 - Test all 9 retained PCA components

This diagnostic checks whether Part1_E information is present in the first three clean PCA components of each geometry group, but was lost by reducing to four features.

```python
X_model_train_9 = pd.DataFrame({
    "Bottom_PC1": bottom_train_scores[:, 0],
    "Bottom_PC2": bottom_train_scores[:, 1],
    "Bottom_PC3": bottom_train_scores[:, 2],
    "Inner_PC1": inner_train_scores[:, 0],
    "Inner_PC2": inner_train_scores[:, 1],
    "Inner_PC3": inner_train_scores[:, 2],
    "Outer_PC1": outer_train_scores[:, 0],
    "Outer_PC2": outer_train_scores[:, 1],
    "Outer_PC3": outer_train_scores[:, 2]
}, index=X_train.index)

X_model_test_9 = pd.DataFrame({
    "Bottom_PC1": bottom_test_scores[:, 0],
    "Bottom_PC2": bottom_test_scores[:, 1],
    "Bottom_PC3": bottom_test_scores[:, 2],
    "Inner_PC1": inner_test_scores[:, 0],
    "Inner_PC2": inner_test_scores[:, 1],
    "Inner_PC3": inner_test_scores[:, 2],
    "Outer_PC1": outer_test_scores[:, 0],
    "Outer_PC2": outer_test_scores[:, 1],
    "Outer_PC3": outer_test_scores[:, 2]
}, index=X_test.index)

for target in target_columns:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_model_train_9, y_train[target])
    predictions = model.predict(X_model_test_9)

    r2 = r2_score(y_test[target], predictions)
    mae = mean_absolute_error(y_test[target], predictions)
    rmse = mean_squared_error(y_test[target], predictions) ** 0.5

    print(target)
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
```

Validated 9-feature diagnostic results:

```text
Part1_E
  R2:   0.0781
  MAE:  0.014600
  RMSE: 0.017216

Part3_E
  R2:   0.9938
  MAE:  0.004995
  RMSE: 0.006819

Part11_E
  R2:   0.9576
  MAE:  0.006643
  RMSE: 0.008732
```

Comparison with the 4-feature representation:
- Part1_E improves slightly from R2 = -0.0422 to R2 = 0.0781, but remains poorly predicted.
- Part3_E remains essentially unchanged and excellent.
- Part11_E remains essentially unchanged and strong.

Interpretation:
- Adding PC2/PC3 from Inner and Outer plus Bottom PC3 does not recover enough information to solve Part1_E.
- The failure is therefore not primarily caused by reducing from 9 PCA features to 4.
- Part1_E may depend on information outside the current geometry-only PCA representation, possibly raw low-variance geometric detail or known simulation/boundary-condition variables such as Pressure, Inner_Radius, or Outer_Radius.

### Cell 14 - Test whether known simulation parameters help Part1_E

Add the known simulation parameters to the 9-PC representation as a diagnostic:

```python
context_cols = [
    "Pressure",
    "Inner_Radius",
    "Outer_Radius"
]

X_context_train = df.loc[X_train.index, context_cols]
X_context_test = df.loc[X_test.index, context_cols]

X_model_train_context = pd.concat([
    X_model_train_9,
    X_context_train
], axis=1)

X_model_test_context = pd.concat([
    X_model_test_9,
    X_context_test
], axis=1)

print("Train with context:", X_model_train_context.shape)
print("Test with context:", X_model_test_context.shape)

for target in target_columns:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_model_train_context, y_train[target])
    predictions = model.predict(X_model_test_context)

    r2 = r2_score(y_test[target], predictions)
    mae = mean_absolute_error(y_test[target], predictions)
    rmse = mean_squared_error(y_test[target], predictions) ** 0.5

    print(target)
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
```

Validated simulation-context diagnostic:

```text
Train with context: (5208, 12)
Test with context: (1302, 12)

Part1_E
  R2:   0.0793
  MAE:  0.014593
  RMSE: 0.017205

Part3_E
  R2:   0.9938
  MAE:  0.004992
  RMSE: 0.006820

Part11_E
  R2:   0.9576
  MAE:  0.006639
  RMSE: 0.008728
```

Interpretation:
- Adding Pressure, Inner_Radius, and Outer_Radius produces essentially no meaningful improvement for Part1_E.
- Part3_E and Part11_E remain unchanged.
- The Part1_E problem is therefore not explained by omission of those three context variables.

### Cell 15 - Test raw noisy geometry directly

This diagnostic checks whether Part1_E information is being discarded by PCA at all. Train the same Random Forest on the full 54 noisy geometry coordinates.

```python
for target in target_columns:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_noisy, y_train[target])
    predictions = model.predict(X_test_noisy)

    r2 = r2_score(y_test[target], predictions)
    mae = mean_absolute_error(y_test[target], predictions)
    rmse = mean_squared_error(y_test[target], predictions) ** 0.5

    print(target)
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
```

Validated raw-geometry baseline:

```text
Part1_E
  R2:   0.0768
  MAE:  0.014744
  RMSE: 0.017228

Part3_E
  R2:   0.9948
  MAE:  0.004532
  RMSE: 0.006233

Part11_E
  R2:   0.9411
  MAE:  0.008051
  RMSE: 0.010288
```

Interpretation:
- Part1_E remains poorly predicted even with all 54 noisy geometry coordinates, so PCA compression is not the main cause.
- Part3_E remains extremely predictable from noisy geometry.
- Part11_E performs slightly worse with raw geometry than with the compact PCA representation, suggesting the clean-PCA projection may be filtering harmful noise.

### Cell 16 - Test Part1_E on clean geometry

This diagnostic removes measurement noise entirely. If Part1_E is still poorly predicted, the issue is likely identifiability from geometry rather than noise.

```python
for target in target_columns:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_spline_base, y_train[target])
    predictions = model.predict(X_test_spline_base)

    r2 = r2_score(y_test[target], predictions)
    mae = mean_absolute_error(y_test[target], predictions)
    rmse = mean_squared_error(y_test[target], predictions) ** 0.5

    print(target)
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
```

Validated clean-geometry baseline:

```text
Part1_E
  R2:   0.9896
  MAE:  0.001313
  RMSE: 0.001830

Part3_E
  R2:   0.9999
  MAE:  0.000483
  RMSE: 0.000647

Part11_E
  R2:   0.9983
  MAE:  0.001166
  RMSE: 0.001744
```

Interpretation:
- Part1_E is highly identifiable from clean geometry (R2 = 0.9896).
- Its failure under noisy geometry is therefore caused by noise sensitivity rather than an inherent lack of geometric information.
- Part3_E and Part11_E are far more robust to the current noise model.
- This suggests Part1_E depends on subtler geometric features that are being obscured by the injected noise.

### Cell 17 - Measure Part1_E sensitivity across noise levels

```python
noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]

for noise in noise_levels:
    if noise == 0.0:
        X_train_level = X_train_spline_base
        X_test_level = X_test_spline_base
    else:
        X_train_level = add_fast_noise(
            X_train_spline_base,
            noise_level=noise,
            seed=42
        )

        X_test_level = add_fast_noise(
            X_test_spline_base,
            noise_level=noise,
            seed=43
        )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_level, y_train["Part1_E"])
    predictions = model.predict(X_test_level)

    r2 = r2_score(y_test["Part1_E"], predictions)
    mae = mean_absolute_error(y_test["Part1_E"], predictions)

    print(f"Noise {noise:.3f}")
    print(f"  R2:  {r2:.4f}")
    print(f"  MAE: {mae:.6f}")
```

Validated Part1_E noise-sensitivity sweep:

```text
Noise 0.000
  R2:  0.9896
  MAE: 0.001313
Noise 0.005
  R2:  0.7093
  MAE: 0.007898
Noise 0.010
  R2:  0.4934
  MAE: 0.010506
Noise 0.020
  R2:  0.2498
  MAE: 0.012991
Noise 0.030
  R2:  0.1390
  MAE: 0.014137
Noise 0.050
  R2:  0.0768
  MAE: 0.014744
```

Interpretation:
- Part1_E performance degrades very rapidly even at small noise levels.
- R2 falls from 0.9896 with clean geometry to 0.7093 at noise 0.005 and below 0.5 by noise 0.01.
- This confirms Part1_E depends on subtle geometry information that the current independent coordinate-noise model quickly destroys.
- The next priority should be testing a denoising/projection strategy rather than simply increasing model complexity.

### Cell 18 - Compare raw noisy geometry vs clean-PCA projection for Part1_E across noise levels

This experiment tests whether projecting noisy observations into PCA spaces learned from clean geometry improves robustness.

```python
noise_levels = [0.005, 0.01, 0.02, 0.03, 0.05]

for noise in noise_levels:
    X_train_level = add_fast_noise(
        X_train_spline_base,
        noise_level=noise,
        seed=42
    )

    X_test_level = add_fast_noise(
        X_test_spline_base,
        noise_level=noise,
        seed=43
    )

    # Raw noisy geometry
    raw_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    raw_model.fit(X_train_level, y_train["Part1_E"])
    raw_pred = raw_model.predict(X_test_level)
    raw_r2 = r2_score(y_test["Part1_E"], raw_pred)

    # Project noisy data into fixed clean PCA spaces
    bottom_train_level = bottom_pca_clean.transform(
        X_train_level[bottom_cols]
    )
    bottom_test_level = bottom_pca_clean.transform(
        X_test_level[bottom_cols]
    )

    inner_train_level = inner_pca_clean.transform(
        X_train_level[inner_shape_cols]
    )
    inner_test_level = inner_pca_clean.transform(
        X_test_level[inner_shape_cols]
    )

    outer_train_level = outer_pca_clean.transform(
        X_train_level[outer_shape_cols]
    )
    outer_test_level = outer_pca_clean.transform(
        X_test_level[outer_shape_cols]
    )

    X_train_pca_level = np.column_stack([
        bottom_train_level[:, :3],
        inner_train_level[:, :3],
        outer_train_level[:, :3]
    ])

    X_test_pca_level = np.column_stack([
        bottom_test_level[:, :3],
        inner_test_level[:, :3],
        outer_test_level[:, :3]
    ])

    pca_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    pca_model.fit(X_train_pca_level, y_train["Part1_E"])
    pca_pred = pca_model.predict(X_test_pca_level)
    pca_r2 = r2_score(y_test["Part1_E"], pca_pred)

    print(f"Noise {noise:.3f}")
    print(f"  Raw geometry R2: {raw_r2:.4f}")
    print(f"  Clean-PCA R2:    {pca_r2:.4f}")
```

Validated raw-vs-clean-PCA robustness comparison for Part1_E:

```text
Noise 0.005
  Raw geometry R2: 0.7093
  Clean-PCA R2:    0.7909

Noise 0.010
  Raw geometry R2: 0.4934
  Clean-PCA R2:    0.6521

Noise 0.020
  Raw geometry R2: 0.2498
  Clean-PCA R2:    0.3912

Noise 0.030
  Raw geometry R2: 0.1390
  Clean-PCA R2:    0.2280

Noise 0.050
  Raw geometry R2: 0.0768
  Clean-PCA R2:    0.0781
```

Interpretation:
- Clean-trained PCA projection clearly improves Part1_E robustness at low-to-moderate noise levels.
- The gain is strongest around noise 0.01 to 0.03.
- At noise 0.05 the benefit disappears; the signal has been degraded too severely for the current 3-PC-per-group representation to recover it.
- This supports clean-PCA projection as a useful denoising step, but not as a complete solution for high noise.

### Cell 19 - Compare different PCA dimensionalities for Part1_E

This diagnostic checks whether retaining more clean-trained PCs helps Part1_E under noise, especially around noise = 0.01 and 0.02.

```python
from sklearn.decomposition import PCA

def fit_clean_pca_n(X_clean, cols, n_components):
    pipeline = Pipeline([
        ("center", StandardScaler(with_std=False)),
        ("pca", PCA(n_components=n_components))
    ])
    pipeline.fit(X_clean[cols])
    return pipeline

noise_levels = [0.01, 0.02]
component_counts = [1, 2, 3, 5, 9, 18]

for noise in noise_levels:
    print(f"\n========== Noise {noise:.3f} ==========")

    X_train_level = add_fast_noise(
        X_train_spline_base,
        noise_level=noise,
        seed=42
    )
    X_test_level = add_fast_noise(
        X_test_spline_base,
        noise_level=noise,
        seed=43
    )

    for n in component_counts:
        bottom_pca_n = fit_clean_pca_n(
            X_train_spline_base,
            bottom_cols,
            n
        )
        inner_pca_n = fit_clean_pca_n(
            X_train_spline_base,
            inner_shape_cols,
            n
        )
        outer_pca_n = fit_clean_pca_n(
            X_train_spline_base,
            outer_shape_cols,
            n
        )

        train_features = np.column_stack([
            bottom_pca_n.transform(X_train_level[bottom_cols]),
            inner_pca_n.transform(X_train_level[inner_shape_cols]),
            outer_pca_n.transform(X_train_level[outer_shape_cols])
        ])

        test_features = np.column_stack([
            bottom_pca_n.transform(X_test_level[bottom_cols]),
            inner_pca_n.transform(X_test_level[inner_shape_cols]),
            outer_pca_n.transform(X_test_level[outer_shape_cols])
        ])

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            train_features,
            y_train["Part1_E"]
        )

        pred = model.predict(test_features)

        r2 = r2_score(
            y_test["Part1_E"],
            pred
        )

        print(f"{n:>2} PCs per group -> R2: {r2:.4f}")
```

Validated PCA dimensionality sweep for Part1_E:

```text
========== Noise 0.010 ==========
 1 PCs per group -> R2: -0.0954
 2 PCs per group -> R2: 0.6280
 3 PCs per group -> R2: 0.6521
 5 PCs per group -> R2: 0.6421
 9 PCs per group -> R2: 0.6308
18 PCs per group -> R2: 0.6093

========== Noise 0.020 ==========
 1 PCs per group -> R2: -0.0738
 2 PCs per group -> R2: 0.3319
 3 PCs per group -> R2: 0.3912
 5 PCs per group -> R2: 0.3869
 9 PCs per group -> R2: 0.3817
18 PCs per group -> R2: 0.3710
```

Interpretation:
- One PC per geometry group is insufficient for Part1_E.
- Three PCs per group performs best at both tested noise levels.
- Keeping more than three PCs slightly hurts performance, consistent with weaker PCs carrying more noise than useful signal.
- This supports a target-specific representation for Part1_E: clean-trained PCA with approximately three components per geometry group.
- The earlier 4-feature representation is still attractive for Part3_E and Part11_E, but Part1_E clearly benefits from a richer 9-feature PCA representation.

### Cell 20 - Compare target-specific PCA feature sets

Use 9 PCA features for Part1_E and the compact 4-feature representation for Part3_E and Part11_E.

```python
target_feature_sets = {
    "Part1_E": (
        X_model_train_9,
        X_model_test_9
    ),
    "Part3_E": (
        X_model_train,
        X_model_test
    ),
    "Part11_E": (
        X_model_train,
        X_model_test
    )
}

for target in target_columns:
    X_train_target, X_test_target = target_feature_sets[target]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_target,
        y_train[target]
    )

    predictions = model.predict(
        X_test_target
    )

    r2 = r2_score(
        y_test[target],
        predictions
    )

    mae = mean_absolute_error(
        y_test[target],
        predictions
    )

    rmse = mean_squared_error(
        y_test[target],
        predictions
    ) ** 0.5

    print(target)
    print(f"  Features: {X_train_target.shape[1]}")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
```

Validated target-specific specialist baseline:

```text
Part1_E
  Features: 9
  R2:   0.0781
  MAE:  0.014600
  RMSE: 0.017216

Part3_E
  Features: 4
  R2:   0.9946
  MAE:  0.004655
  RMSE: 0.006378

Part11_E
  Features: 4
  R2:   0.9561
  MAE:  0.006744
  RMSE: 0.008884
```

Interpretation:
- The target-specific feature selection reproduces the earlier baselines as expected.
- Part3_E and Part11_E are already strong with the compact 4-feature representation.
- Part1_E remains the unresolved target and requires a robustness strategy rather than simply more PCA components.
- This establishes a clean specialist baseline before introducing noise-augmented training.

### Cell 21 - Train Part1_E with multiple noise levels

This experiment augments the Part1_E training data using several noise levels while keeping the test set fixed at noise = 0.05.

```python
train_noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]

augmented_features = []
augmented_targets = []

for noise in train_noise_levels:
    if noise == 0.0:
        X_train_level = X_train_spline_base
    else:
        X_train_level = add_fast_noise(
            X_train_spline_base,
            noise_level=noise,
            seed=42 + int(noise * 1000)
        )

    bottom_level = bottom_pca_clean.transform(
        X_train_level[bottom_cols]
    )
    inner_level = inner_pca_clean.transform(
        X_train_level[inner_shape_cols]
    )
    outer_level = outer_pca_clean.transform(
        X_train_level[outer_shape_cols]
    )

    X_level = np.column_stack([
        bottom_level[:, :3],
        inner_level[:, :3],
        outer_level[:, :3]
    ])

    augmented_features.append(X_level)
    augmented_targets.append(y_train["Part1_E"].to_numpy())

X_train_augmented = np.vstack(augmented_features)
y_train_augmented = np.concatenate(augmented_targets)

print("Augmented train:", X_train_augmented.shape)
print("Augmented targets:", y_train_augmented.shape)

part1_aug_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

part1_aug_model.fit(
    X_train_augmented,
    y_train_augmented
)

part1_test_pred = part1_aug_model.predict(
    X_model_test_9
)

r2 = r2_score(
    y_test["Part1_E"],
    part1_test_pred
)

mae = mean_absolute_error(
    y_test["Part1_E"],
    part1_test_pred
)

rmse = mean_squared_error(
    y_test["Part1_E"],
    part1_test_pred
) ** 0.5

print(f"Part1_E augmented training")
print(f"  R2:   {r2:.4f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
```

Validated multi-noise augmentation result for Part1_E:

```text
Part1_E augmented training
  R2:   0.0626
  MAE:  0.014469
  RMSE: 0.017361
```

A scikit-learn warning was also observed because the Random Forest was fitted on a NumPy array without feature names and then asked to predict from a DataFrame with feature names. This warning does not change the numeric result.

Interpretation:
- Mixing many training noise levels did not improve the noise = 0.05 test case.
- R2 decreased slightly from 0.0781 to 0.0626.
- The likely issue is that the model is being asked to learn one mapping across several different noise distributions, which may blur the already fragile Part1_E signal.

### Cell 22 - Noise-matched augmentation at noise = 0.05

Instead of mixing noise levels, create several independent noise realizations at the same test noise level and train on all of them.

```python
matched_noise = 0.05
train_seeds = [42, 52, 62, 72, 82, 92]

matched_features = []
matched_targets = []

for seed in train_seeds:
    X_train_level = add_fast_noise(
        X_train_spline_base,
        noise_level=matched_noise,
        seed=seed
    )

    bottom_level = bottom_pca_clean.transform(
        X_train_level[bottom_cols]
    )
    inner_level = inner_pca_clean.transform(
        X_train_level[inner_shape_cols]
    )
    outer_level = outer_pca_clean.transform(
        X_train_level[outer_shape_cols]
    )

    X_level = np.column_stack([
        bottom_level[:, :3],
        inner_level[:, :3],
        outer_level[:, :3]
    ])

    matched_features.append(X_level)
    matched_targets.append(
        y_train["Part1_E"].to_numpy()
    )

X_train_matched = np.vstack(matched_features)
y_train_matched = np.concatenate(matched_targets)

matched_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

matched_model.fit(
    X_train_matched,
    y_train_matched
)

matched_pred = matched_model.predict(
    X_model_test_9.to_numpy()
)

r2 = r2_score(
    y_test["Part1_E"],
    matched_pred
)

mae = mean_absolute_error(
    y_test["Part1_E"],
    matched_pred
)

rmse = mean_squared_error(
    y_test["Part1_E"],
    matched_pred
) ** 0.5

print("Part1_E matched-noise augmentation")
print(f"  R2:   {r2:.4f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
```

Validated matched-noise augmentation result for Part1_E:

```text
Part1_E matched-noise augmentation
  R2:   0.1004
  MAE:  0.014361
  RMSE: 0.017006
```

Interpretation:
- Matched-noise augmentation improves Part1_E slightly over the single-noise baseline (R2 = 0.0781 -> 0.1004).
- It also outperforms mixed-noise augmentation (R2 = 0.0626).
- The gain is real but modest, so augmentation alone is not enough to recover the strong clean-geometry signal at noise = 0.05.
- This suggests the next priority should be changing the representation/denoising strategy, not simply adding more noisy copies.

### Cell 23 - Test target-specific clean-PCA dimensionality at noise = 0.05

Because three PCs per group was best at lower noise, test whether asymmetric component counts help at the harder 0.05 noise level.

```python
configs = [
    (2, 1, 1),
    (2, 2, 2),
    (3, 2, 2),
    (3, 3, 3),
    (5, 3, 3),
    (5, 5, 5)
]

for bottom_n, inner_n, outer_n in configs:
    train_features = np.column_stack([
        bottom_train_scores[:, :bottom_n],
        inner_train_scores[:, :inner_n],
        outer_train_scores[:, :outer_n]
    ])

    test_features = np.column_stack([
        bottom_test_scores[:, :bottom_n],
        inner_test_scores[:, :inner_n],
        outer_test_scores[:, :outer_n]
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train_features,
        y_train["Part1_E"]
    )

    pred = model.predict(test_features)

    r2 = r2_score(
        y_test["Part1_E"],
        pred
    )

    print(
        f"Bottom {bottom_n}, Inner {inner_n}, Outer {outer_n} "
        f"-> R2: {r2:.4f}"
    )
```

Validated asymmetric PCA sweep at noise = 0.05:

```text
Bottom 2, Inner 1, Outer 1 -> R2: -0.0285
Bottom 2, Inner 2, Outer 2 -> R2: 0.0442
Bottom 3, Inner 2, Outer 2 -> R2: 0.0697
Bottom 3, Inner 3, Outer 3 -> R2: 0.0838
Bottom 5, Inner 3, Outer 3 -> R2: 0.0838
Bottom 5, Inner 5, Outer 5 -> R2: 0.0838
```

Interpretation:
- Increasing component count helps up to about 3 PCs per group.
- Adding more than 3 PCs does not improve Part1_E at noise = 0.05.
- This reinforces the earlier conclusion that the failure at high noise is not caused by insufficient PCA dimensionality.
- The best asymmetric result here (R2 = 0.0838) is still below the matched-noise augmentation result (R2 = 0.1004).

## Negative / failed experiments

These results are intentionally retained because they narrow the design space and prevent repeating unproductive approaches.

### Part1_E diagnostics that did not solve the problem

- 4-feature compact PCA representation at noise = 0.05:
  - R2 = -0.0422
- 9-feature PCA representation at noise = 0.05:
  - R2 = 0.0781
- 9 PCA features + Pressure + Inner_Radius + Outer_Radius:
  - R2 = 0.0793
  - Conclusion: these context variables did not explain the missing Part1_E signal.
- Full 54-dimensional noisy geometry:
  - R2 = 0.0768
  - Conclusion: PCA compression was not the primary cause of failure.
- Mixed-noise augmentation using training noise levels 0.0 through 0.05:
  - R2 = 0.0626
  - Conclusion: combining several noise regimes slightly hurt performance.
- Asymmetric PCA component sweep at noise = 0.05:
  - Best result R2 = 0.0838
  - Conclusion: keeping more than 3 clean PCs per group did not recover Part1_E.
- One PC per geometry group:
  - Negative R2 at noise 0.01 and 0.02.
  - Conclusion: Part1_E requires more than the dominant shape mode.
- High noise levels remain a major failure mode:
  - Raw geometry at noise 0.05: R2 = 0.0768
  - Clean-PCA projection at noise 0.05: R2 = 0.0781
  - Conclusion: at this noise level the current representation cannot recover the subtle Part1_E signal.

### What did help

- Clean geometry:
  - Part1_E R2 = 0.9896
- Clean-trained PCA projection:
  - Improved Part1_E over raw geometry at noise 0.005 to 0.03
- Matched-noise augmentation at noise = 0.05:
  - Improved Part1_E from R2 = 0.0781 to R2 = 0.1004
  - Improvement is real but modest.

Next step: preserve these negative results as part of the research record and move to a stronger denoising or shape-reconstruction method rather than spending more effort on PCA component-count tuning.


## Next phase: denoising and shape reconstruction

The PCA component-count experiments are now sufficiently explored. The next development focus is reconstructing a cleaner estimate of geometry before inverse prediction.

Recommended progression:

1. **Clean-manifold PCA reconstruction baseline**
   - Fit PCA on clean training geometry.
   - Project noisy geometry into that fixed PCA space.
   - Reconstruct geometry with `inverse_transform()`.
   - Train the inverse model on reconstructed geometry.
   - This is a cheap baseline for manifold-based denoising.

2. **Supervised denoising autoencoder**
   - Input: noisy 54-coordinate geometry.
   - Target: corresponding clean spline-base geometry.
   - Train only on training simulations and generate multiple noise realizations per clean sample.
   - Evaluate both coordinate reconstruction error and downstream `Part1_E` prediction.
   - This is the preferred next nonlinear method because paired clean/noisy data are available.

3. **Target-aware denoising (later)**
   - Only after a plain denoiser is validated.
   - Combine geometry reconstruction loss with a material-property prediction loss so that subtle geometry useful for `Part1_E` is preserved.

### Cell 24 - PCA reconstruction baseline

Use the already-fitted clean PCA models to reconstruct noisy geometry back onto the clean shape manifold.

```python
def reconstruct_from_clean_pca(
    X_noisy,
    pca_model,
    cols
):
    scores = pca_model.transform(X_noisy[cols])
    reconstructed = pca_model.inverse_transform(scores)

    return pd.DataFrame(
        reconstructed,
        columns=cols,
        index=X_noisy.index
    )

bottom_train_recon = reconstruct_from_clean_pca(
    X_train_noisy,
    bottom_pca_clean,
    bottom_cols
)

bottom_test_recon = reconstruct_from_clean_pca(
    X_test_noisy,
    bottom_pca_clean,
    bottom_cols
)

inner_train_recon = reconstruct_from_clean_pca(
    X_train_noisy,
    inner_pca_clean,
    inner_shape_cols
)

inner_test_recon = reconstruct_from_clean_pca(
    X_test_noisy,
    inner_pca_clean,
    inner_shape_cols
)

outer_train_recon = reconstruct_from_clean_pca(
    X_train_noisy,
    outer_pca_clean,
    outer_shape_cols
)

outer_test_recon = reconstruct_from_clean_pca(
    X_test_noisy,
    outer_pca_clean,
    outer_shape_cols
)

X_train_reconstructed = pd.concat([
    bottom_train_recon,
    inner_train_recon,
    outer_train_recon
], axis=1)[feature_columns]

X_test_reconstructed = pd.concat([
    bottom_test_recon,
    inner_test_recon,
    outer_test_recon
], axis=1)[feature_columns]

print("Reconstructed train:", X_train_reconstructed.shape)
print("Reconstructed test:", X_test_reconstructed.shape)
```

After this baseline is measured, the next major implementation should be a supervised denoising autoencoder trained on paired noisy -> clean geometry.


### Clarification: how this differs from Grace's InverseFEA repository

The proposed denoising/reconstruction stage is **not the same approach** used in `gracegb/InverseFEA`.

Grace's repository primarily:
- consumes already-created clean/noisy PCA datasets,
- trains Random Forest / XGBoost inverse regressors,
- experiments with chained and iterative predictions,
- compares clean-trained and noisy-tested models,
- includes OOF/CORAL/quantile experiments for Part1_E.

It does **not** train a model whose task is:
```text
noisy geometry -> reconstructed clean geometry
```

The current MultiTissue-InverseFEA direction adds that missing intermediate stage:

```text
noisy observation
      ↓
denoiser / clean-manifold reconstruction
      ↓
cleaner geometry representation
      ↓
inverse material-property expert
```

The immediate PCA reconstruction baseline is also not a learned neural denoiser. It uses PCA fitted on clean training geometry and `inverse_transform()` to project noisy observations back onto the learned clean-shape subspace. A later supervised denoising autoencoder would be a genuinely trained noisy-to-clean reconstruction model.


### Fix: recreate X_train_noisy / X_test_noisy before PCA reconstruction

If Colab reports:

```text
NameError: name 'X_train_noisy' is not defined
```

rerun the noisy-dataset creation step before the PCA reconstruction cells:

```python
X_train_noisy = add_fast_noise(
    X_train_spline_base,
    noise_level=0.05,
    seed=42
)

X_test_noisy = add_fast_noise(
    X_test_spline_base,
    noise_level=0.05,
    seed=43
)

print("Noisy train:", X_train_noisy.shape)
print("Noisy test:", X_test_noisy.shape)
```

This error usually means the Colab runtime was restarted or the relevant earlier cell was not executed in the current session.


### Fix: recreate add_fast_noise after Colab runtime reset

If Colab reports:

```text
NameError: name 'add_fast_noise' is not defined
```

the function-definition cell has not been run in the current runtime. Rerun:

```python
def add_fast_noise(X_base, noise_level=0.05, seed=42):
    rng = np.random.default_rng(seed)
    noisy = X_base.copy()

    noisy[bottom_cols] += rng.uniform(
        -0.01,
        0.01,
        size=noisy[bottom_cols].shape
    )

    shape_cols = inner_shape_cols + outer_shape_cols

    noisy[shape_cols] += rng.normal(
        0,
        noise_level,
        size=noisy[shape_cols].shape
    )

    return noisy
```

Then recreate `X_train_noisy` and `X_test_noisy`.


### Colab runtime reset recovery

If the Colab runtime resets and previously defined variables/functions disappear, the safest recovery is to rerun the notebook from the top using:

```text
Ambiente de execução -> Executar tudo
```

Before doing so, make sure the notebook contains the validated cells in order. The expensive spline-base step will rerun, but this restores all required state consistently.

Recommended later improvement: save the clean spline bases and fitted PCA models to disk so a runtime reset does not require recomputing the expensive preprocessing.


### Runtime recovery checkpoint

After the Colab reset, the notebook has been rerun through the `add_fast_noise()` function definition.

Next recovery step:

```python
X_train_noisy = add_fast_noise(
    X_train_spline_base,
    noise_level=0.05,
    seed=42
)

X_test_noisy = add_fast_noise(
    X_test_spline_base,
    noise_level=0.05,
    seed=43
)

print("Noisy train:", X_train_noisy.shape)
print("Noisy test:", X_test_noisy.shape)
```

Expected output:

```text
Noisy train: (5208, 54)
Noisy test: (1302, 54)
```


### Runtime recovery checkpoint - noisy datasets restored

Validated output:

```text
Noisy train: (5208, 54)
Noisy test: (1302, 54)
```

Next recovery step: refit the clean-training PCA models for Bottom, Inner Shape, and Outer Shape.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def fit_clean_pca(X_clean, cols, n_components=3):
    pipeline = Pipeline([
        ("center", StandardScaler(with_std=False)),
        ("pca", PCA(n_components=n_components))
    ])

    pipeline.fit(X_clean[cols])
    return pipeline

bottom_pca_clean = fit_clean_pca(
    X_train_spline_base,
    bottom_cols,
    n_components=3
)

inner_pca_clean = fit_clean_pca(
    X_train_spline_base,
    inner_shape_cols,
    n_components=3
)

outer_pca_clean = fit_clean_pca(
    X_train_spline_base,
    outer_shape_cols,
    n_components=3
)

print("PCA models fitted.")
```


### Runtime recovery checkpoint - PCA models restored

Validated output:

```text
PCA models fitted.
```

Next step: reconstruct noisy geometry back onto the clean PCA manifolds.

```python
def reconstruct_from_clean_pca(X_noisy, pca_model, cols):
    scores = pca_model.transform(X_noisy[cols])
    reconstructed = pca_model.inverse_transform(scores)

    return pd.DataFrame(
        reconstructed,
        columns=cols,
        index=X_noisy.index
    )

bottom_train_recon = reconstruct_from_clean_pca(
    X_train_noisy,
    bottom_pca_clean,
    bottom_cols
)

bottom_test_recon = reconstruct_from_clean_pca(
    X_test_noisy,
    bottom_pca_clean,
    bottom_cols
)

inner_train_recon = reconstruct_from_clean_pca(
    X_train_noisy,
    inner_pca_clean,
    inner_shape_cols
)

inner_test_recon = reconstruct_from_clean_pca(
    X_test_noisy,
    inner_pca_clean,
    inner_shape_cols
)

outer_train_recon = reconstruct_from_clean_pca(
    X_train_noisy,
    outer_pca_clean,
    outer_shape_cols
)

outer_test_recon = reconstruct_from_clean_pca(
    X_test_noisy,
    outer_pca_clean,
    outer_shape_cols
)

X_train_reconstructed = pd.concat([
    bottom_train_recon,
    inner_train_recon,
    outer_train_recon
], axis=1)[feature_columns]

X_test_reconstructed = pd.concat([
    bottom_test_recon,
    inner_test_recon,
    outer_test_recon
], axis=1)[feature_columns]

print("Reconstructed train:", X_train_reconstructed.shape)
print("Reconstructed test:", X_test_reconstructed.shape)
```

Expected output:

```text
Reconstructed train: (5208, 54)
Reconstructed test: (1302, 54)
```


### Reconstruction baseline checkpoint

Validated output:

```text
Reconstructed train: (5208, 54)
Reconstructed test: (1302, 54)
```

Next step: evaluate whether PCA reconstruction improves downstream Part1_E prediction compared with raw noisy geometry.

```python
recon_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

recon_model.fit(
    X_train_reconstructed,
    y_train["Part1_E"]
)

recon_pred = recon_model.predict(
    X_test_reconstructed
)

r2 = r2_score(
    y_test["Part1_E"],
    recon_pred
)

mae = mean_absolute_error(
    y_test["Part1_E"],
    recon_pred
)

rmse = mean_squared_error(
    y_test["Part1_E"],
    recon_pred
) ** 0.5

print("Part1_E PCA-reconstructed geometry")
print(f"  R2:   {r2:.4f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
```

Reference baseline at noise = 0.05:
- Raw noisy geometry: R2 = 0.0768
- 9-feature clean-PCA projection: R2 = 0.0781
- Matched-noise augmentation: R2 = 0.1004

This test determines whether reconstructing geometry back onto the clean PCA manifold provides any additional benefit before moving to a nonlinear denoiser.


### PCA geometry reconstruction result

Validated result:

```text
Part1_E PCA-reconstructed geometry
  R2:   0.0728
  MAE:  0.014611
  RMSE: 0.017266
```

Comparison at noise = 0.05:
- Raw noisy geometry: R2 = 0.0768
- 9-feature clean-PCA projection: R2 = 0.0781
- PCA-reconstructed 54-coordinate geometry: R2 = 0.0728
- Matched-noise augmentation: R2 = 0.1004

Interpretation:
- PCA inverse reconstruction does not improve Part1_E and slightly underperforms both raw noisy geometry and direct PCA-score prediction.
- This is an important negative result: once the noisy observation is reduced to the same low-dimensional linear PCA subspace, inverse-transforming it back to 54 coordinates cannot restore information that was discarded.
- Further linear PCA reconstruction tuning is therefore low priority.

### Next method: supervised nonlinear denoising autoencoder

The next experiment should explicitly learn the paired mapping:

```text
noisy 54-coordinate geometry -> clean 54-coordinate geometry
```

Training pairs can be generated from each clean training simulation using multiple independent noise realizations. Evaluation must remain on held-out test simulations.

The denoiser should first be evaluated on:
1. reconstruction error against clean test geometry,
2. downstream Part1_E R2 after denoising,
3. comparison against raw noisy geometry and the clean-PCA baseline.

Do not include Part1_E in the denoiser loss initially; first determine whether geometry itself can be reconstructed without target leakage.


### Experiment naming

Recommended experiment name:

**Supervised Geometry Denoising for Noise-Robust Inverse FEA**

Short internal name:

`DAE-Geometry-v1`

Suggested description:

> This experiment evaluates whether a supervised denoising autoencoder can recover clean tissue geometry from noisy coordinate observations and thereby improve inverse FEA prediction of noise-sensitive material properties. Artificially corrupted geometry is used as input while the corresponding clean simulated geometry serves as the reconstruction target. The denoiser is trained independently of the elastic-modulus targets, and its reconstructed geometry is subsequently evaluated using the inverse regression model, with particular emphasis on Part1_E.

Core hypothesis:

> Learning the nonlinear mapping from noisy observations to valid clean tissue shapes will preserve subtle geometry relevant to Part1_E better than linear PCA-based denoising.


## Recommended next steps after PCA denoising

The next phase should focus on a supervised nonlinear geometry denoiser.

Recommended order:

1. **Build paired training data**
   - Input: noisy 54-coordinate geometry.
   - Target: corresponding clean spline-base geometry.
   - Generate multiple independent noisy versions of each clean training simulation.
   - Keep test simulations completely held out.

2. **Train a small denoising autoencoder**
   - Start with a simple fully connected network rather than a complex architecture.
   - Suggested structure:
     - 54 -> 128 -> 64 -> 16 latent dimensions
     - 16 -> 64 -> 128 -> 54
   - Optimize reconstruction loss only at first.

3. **Evaluate geometry reconstruction**
   - Compare noisy-vs-clean MAE with denoised-vs-clean MAE.
   - Inspect a few reconstructed Inner/Outer shapes visually.

4. **Evaluate downstream inverse FEA**
   - Train the same Random Forest Part1_E model on denoised geometry.
   - Compare against:
     - raw noisy geometry,
     - clean-PCA features,
     - matched-noise augmentation.

5. **Only if geometry denoising helps**
   - Tune latent size and loss.
   - Try region-weighted reconstruction losses.
   - Later test a target-aware loss that preserves geometry important to Part1_E.

Do not move to the multi-expert orchestrator yet. First establish whether a denoiser can recover the subtle Part1_E geometry signal under realistic noise.


### Cell 25 - Build paired noisy/clean geometry data for denoising

Create several noisy versions of each clean training simulation and pair each noisy sample with the corresponding clean spline-base geometry.

```python
train_noise_levels = [0.005, 0.01, 0.02, 0.03, 0.05]
train_seeds = [42, 52, 62]

noisy_pairs = []
clean_pairs = []

for noise in train_noise_levels:
    for seed in train_seeds:
        X_noisy_level = add_fast_noise(
            X_train_spline_base,
            noise_level=noise,
            seed=seed
        )

        noisy_pairs.append(
            X_noisy_level[feature_columns].to_numpy()
        )

        clean_pairs.append(
            X_train_spline_base[feature_columns].to_numpy()
        )

X_denoise_train = np.vstack(noisy_pairs)
y_denoise_train = np.vstack(clean_pairs)

print("Denoiser inputs:", X_denoise_train.shape)
print("Denoiser targets:", y_denoise_train.shape)
```

Expected shape:

```text
Denoiser inputs: (78120, 54)
Denoiser targets: (78120, 54)
```

This produces 15 noisy realizations per clean training sample: 5 noise levels x 3 random seeds.


### Paired denoiser dataset checkpoint

Validated output:

```text
Denoiser inputs: (78120, 54)
Denoiser targets: (78120, 54)
```

This corresponds to 15 noisy realizations for each of the 5,208 clean training simulations.

### Cell 26 - Standardize denoiser inputs and targets

Use statistics from the clean training geometry only.

```python
from sklearn.preprocessing import StandardScaler

geometry_scaler = StandardScaler()

geometry_scaler.fit(
    X_train_spline_base[feature_columns]
)

X_denoise_train_scaled = geometry_scaler.transform(
    X_denoise_train
)

y_denoise_train_scaled = geometry_scaler.transform(
    y_denoise_train
)

print("Scaled denoiser inputs:", X_denoise_train_scaled.shape)
print("Scaled denoiser targets:", y_denoise_train_scaled.shape)
print(
    "Input mean (approx):",
    X_denoise_train_scaled.mean()
)
print(
    "Target mean (approx):",
    y_denoise_train_scaled.mean()
)
```

Next step: split the paired denoiser training data into training and validation subsets without touching the held-out inverse-FEA test simulations.


### Denoiser scaling checkpoint

Validated output:

```text
Scaled denoiser inputs: (78120, 54)
Scaled denoiser targets: (78120, 54)
Input mean: 0.025558734209917895
Target mean: approximately 0
```

Interpretation:
- The clean targets are centered essentially at zero, as expected.
- The noisy inputs are not exactly zero-centered because noise shifts the distribution slightly.
- The scikit-learn warnings about feature names are harmless and occur because the scaler was fitted on a DataFrame but transformed NumPy arrays.

### Cell 27 - Split denoiser data into train and validation sets

```python
from sklearn.model_selection import train_test_split

X_dae_train, X_dae_val, y_dae_train, y_dae_val = train_test_split(
    X_denoise_train_scaled,
    y_denoise_train_scaled,
    test_size=0.15,
    random_state=42
)

print("DAE train:", X_dae_train.shape, y_dae_train.shape)
print("DAE val:", X_dae_val.shape, y_dae_val.shape)
```

Expected:
```text
DAE train: (66402, 54) (66402, 54)
DAE val: (11718, 54) (11718, 54)
```


### Denoiser train/validation split checkpoint

Validated output:

```text
DAE train: (66402, 54) (66402, 54)
DAE val: (11718, 54) (11718, 54)
```

### Cell 28 - Build and train the first denoising autoencoder

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

dae = keras.Sequential([
    layers.Input(shape=(54,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(54)
])

dae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

history = dae.fit(
    X_dae_train,
    y_dae_train,
    validation_data=(X_dae_val, y_dae_val),
    epochs=50,
    batch_size=256,
    verbose=1
)
```

Next step: inspect training vs validation loss, then evaluate reconstruction quality on held-out test simulations at noise = 0.05.


### Denoising autoencoder training checkpoint (partial)

Training through epoch 29 shows steady improvement:

```text
Epoch 1:  loss 0.2850, val_loss 0.0768
Epoch 10: loss 0.0231, val_loss 0.0216
Epoch 20: loss 0.0180, val_loss 0.0206
Epoch 29: loss 0.0154, val_loss 0.0149
```

Interpretation:
- Validation loss is falling with training loss, so there is no obvious severe overfitting in the reported range.
- The model continues to improve through at least epoch 29.
- The next checkpoint should record the final epoch values and then test the trained denoiser on held-out noisy test geometry.


### Denoising autoencoder training complete

Final training values:

```text
Epoch 50/50
loss: 0.0115
val_loss: 0.0120
```

Training remained stable through the final epochs. Validation loss tracked training loss closely, with only small fluctuations, so there is no obvious severe overfitting.

### Cell 29 - Evaluate denoiser on held-out test geometry at noise = 0.05

```python
X_test_noisy_scaled = geometry_scaler.transform(
    X_test_noisy[feature_columns]
)

X_test_denoised_scaled = dae.predict(
    X_test_noisy_scaled,
    verbose=0
)

X_test_denoised = geometry_scaler.inverse_transform(
    X_test_denoised_scaled
)

X_test_clean = X_test_spline_base[feature_columns].to_numpy()

raw_mae = np.mean(
    np.abs(
        X_test_noisy[feature_columns].to_numpy()
        - X_test_clean
    )
)

denoised_mae = np.mean(
    np.abs(
        X_test_denoised
        - X_test_clean
    )
)

print(f"Raw noisy geometry MAE: {raw_mae:.6f}")
print(f"Denoised geometry MAE:   {denoised_mae:.6f}")
```

Next step: if denoised MAE improves, test whether Part1_E prediction improves when using the denoised geometry.


### Denoiser held-out reconstruction result

Validated result at noise = 0.05:

```text
Raw noisy geometry MAE: 0.028150
Denoised geometry MAE:   0.003754
```

Interpretation:
- The denoising autoencoder reduces average coordinate error by about 86.7%.
- This is a major improvement over the raw noisy geometry and confirms the nonlinear denoiser is learning useful structure from paired noisy/clean simulations.
- The next critical test is downstream inverse-FEA performance, especially whether Part1_E recovers when predicted from denoised geometry.

### Cell 30 - Evaluate Part1_E on denoised geometry

```python
X_train_noisy_scaled = geometry_scaler.transform(
    X_train_noisy[feature_columns]
)

X_train_denoised_scaled = dae.predict(
    X_train_noisy_scaled,
    verbose=0
)

X_train_denoised = geometry_scaler.inverse_transform(
    X_train_denoised_scaled
)

part1_denoised_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

part1_denoised_model.fit(
    X_train_denoised,
    y_train["Part1_E"]
)

part1_denoised_pred = part1_denoised_model.predict(
    X_test_denoised
)

r2 = r2_score(
    y_test["Part1_E"],
    part1_denoised_pred
)

mae = mean_absolute_error(
    y_test["Part1_E"],
    part1_denoised_pred
)

rmse = mean_squared_error(
    y_test["Part1_E"],
    part1_denoised_pred
) ** 0.5

print("Part1_E using denoised geometry")
print(f"  R2:   {r2:.4f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
```

Reference baselines at noise = 0.05:
- Raw noisy geometry R2 = 0.0768
- Clean-PCA projection R2 = 0.0781
- Matched-noise augmentation R2 = 0.1004


### Denoised geometry downstream Part1_E result

Validated result:

```text
Part1_E using denoised geometry
  R2:   0.0258
  MAE:  0.014777
  RMSE: 0.017697
```

This is a critical negative result. Although the supervised autoencoder reduced coordinate MAE from 0.028150 to 0.003754 (~86.7%), downstream Part1_E performance became worse.

Comparison at noise = 0.05:
- Raw noisy geometry: R2 = 0.0768
- Direct clean-PCA features: R2 = 0.0781
- PCA reconstructed geometry: R2 = 0.0728
- Matched-noise augmentation: R2 = 0.1004
- Denoising autoencoder geometry: R2 = 0.0258

Interpretation: minimizing average coordinate reconstruction error is not sufficient for inverse FEA. The autoencoder can produce geometrically accurate shapes while smoothing/removing small deformation patterns that carry Part1_E information.

Recommended next experiment: target-aware denoising. Preserve the geometry reconstruction objective, but add an auxiliary Part1_E prediction objective so the latent representation/reconstruction is encouraged to retain mechanically informative features. This should be done with training targets only and evaluated strictly on held-out test simulations.


### Clarification - Part1_E comparison framework

Part1_E experiments are compared against the same held-out test simulations and the same regression metric (R2), with noise level 0.05 unless otherwise stated.

The main reference points are:
- **Clean-geometry upper/reference baseline:** Random Forest trained/tested on clean spline geometry, R2 = 0.9896.
- **Raw-noisy baseline:** Random Forest using the 54 noisy geometry coordinates, R2 = 0.0768.
- **Best noise-handling baseline so far:** matched-noise augmentation, R2 = 0.1004.
- Clean-PCA projection: R2 = 0.0781.
- PCA-reconstructed geometry: R2 = 0.0728.
- Supervised geometry-denoising autoencoder followed by Random Forest: R2 = 0.0258.

For a new target-aware method, the primary practical comparison should be against the raw-noisy and matched-noise baselines. The clean-geometry result is a reference ceiling showing how much Part1_E information exists before measurement noise is introduced; it is not a directly fair noisy-input competitor.


### How the denoising autoencoder removes noise

The denoising autoencoder is trained using paired examples of the same simulation:

```text
noisy 54-coordinate geometry -> network -> clean 54-coordinate geometry
```

For each clean training geometry, artificial perturbations are generated at several noise levels and random seeds. The clean geometry remains the target. Because many different corrupted versions map back to the same underlying clean shape, the network is encouraged to learn repeatable structure shared by valid tissue geometries rather than reproduce each random perturbation.

The current network compresses 54 coordinates through 128 and 64 hidden units into a 16-dimensional latent representation, then expands 16 -> 64 -> 128 -> 54. This bottleneck limits the information that can be copied directly and encourages a compact representation of recurring shape structure.

The training loss is mean squared reconstruction error between predicted and clean standardized coordinates. Therefore, the model is not explicitly told which coordinate deviations are noise; it learns statistically that unpredictable perturbations should be removed when doing so makes its output closer to the corresponding clean examples.

Current limitation: this objective preserves features useful for minimizing average coordinate error, not necessarily features carrying material-property information. This explains why geometry MAE improved strongly while Part1_E R2 worsened.


### Cell 31 - Visualize clean vs noisy vs autoencoder-denoised geometry

This visualization overlays the same held-out test sample before noise, after noise, and after denoising.

```python
import matplotlib.pyplot as plt
import pandas as pd

# Convert denoised test output back to a DataFrame
X_test_denoised_df = pd.DataFrame(
    X_test_denoised,
    columns=feature_columns,
    index=X_test.index
)

sample = X_test.index[0]

# Inner shape
clean_x = X_test_spline_base.loc[sample, inner_shape_cols[:9]].to_numpy()
clean_y = X_test_spline_base.loc[sample, inner_shape_cols[9:]].to_numpy()

noisy_x = X_test_noisy.loc[sample, inner_shape_cols[:9]].to_numpy()
noisy_y = X_test_noisy.loc[sample, inner_shape_cols[9:]].to_numpy()

denoised_x = X_test_denoised_df.loc[sample, inner_shape_cols[:9]].to_numpy()
denoised_y = X_test_denoised_df.loc[sample, inner_shape_cols[9:]].to_numpy()

plt.figure(figsize=(7, 7))

plt.plot(clean_x, clean_y, "o-", label="Clean")
plt.plot(noisy_x, noisy_y, "o--", label="Noisy")
plt.plot(denoised_x, denoised_y, "o-", label="Denoised")

plt.axis("equal")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Inner Shape - Clean vs Noisy vs Denoised (sample {sample})")
plt.legend()
plt.grid(True)
plt.show()
```

The same pattern can later be repeated for Outer Shape and Bottom geometry.


### Visualization standards from prior researcher's Nov16 diagnostics

Previous research examples supplied:
- `nov16_noisy_from_clean: True vs Predicted (Part1_E)` and residual histogram
- `nov16_noisy_from_clean: True vs Predicted (Part3_E)` and residual histogram

Qualitative reading:
- Part1_E predicted outputs collapse toward a narrow central interval relative to the observed true values; weak prediction despite a roughly centered residual distribution.
- Part3_E predictions follow the ideal diagonal more clearly, but with noticeable dispersion, potential bias or compressed high-end predictions.
- A residual histogram alone cannot establish strong predictive performance or correlation.
- Prior Nov16 output ranges and dataset size appear different from current Colab run; do not directly compare MAE/R2 without harmonizing dataset, target units/scaling, noise level, test split, and protocol.

Recommended standard diagnostic bundle for each target and method:
1. True-vs-predicted scatter with y=x reference, matched axes, transparent points and R2/MAE/RMSE annotation.
2. Histogram of residual (True - Predicted) with zero reference.
3. Residual-vs-true scatter to diagnose bias and heteroscedasticity.
4. Cross-method R2/MAE summary computed from the same held-out test samples.
5. PCA scree/cumulative variance and clean/noisy/denoised geometry plots for the preprocessing questions.

Reusable plotting code for the current Part1_E autoencoder experiment:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def plot_prediction_diagnostics(y_true, y_pred, target, method):
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    residual = truth - pred
    r2 = r2_score(truth, pred)
    mae = mean_absolute_error(truth, pred)
    rmse = np.sqrt(mean_squared_error(truth, pred))

    low = min(truth.min(), pred.min())
    high = max(truth.max(), pred.max())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].scatter(truth, pred, s=13, alpha=0.22)
    ax[0].plot([low, high], [low, high], '--', color='black', label='Ideal')
    ax[0].set(xlabel=f'True {target}', ylabel=f'Predicted {target}', title='True vs predicted')
    ax[0].set_xlim(low, high)
    ax[0].set_ylim(low, high)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].legend()

    ax[1].hist(residual, bins=35)
    ax[1].axvline(0, color='black', linestyle='--')
    ax[1].set(xlabel='Residual (True - Predicted)', ylabel='Count', title='Residual distribution')

    ax[2].scatter(truth, residual, s=13, alpha=0.22)
    ax[2].axhline(0, color='black', linestyle='--')
    ax[2].set(xlabel=f'True {target}', ylabel='Residual (True - Predicted)', title='Residual vs true')

    fig.suptitle(f'{method} | {target} | R2={r2:.4f} | MAE={mae:.6f} | RMSE={rmse:.6f}')
    fig.tight_layout()
    plt.show()

plot_prediction_diagnostics(
    y_test['Part1_E'],
    part1_denoised_pred,
    'Part1_E',
    'Autoencoder denoising'
)
```

Note: create separate diagnostics for every target/method using its own predictions on the same held-out test set; never transfer points or numerical statistics between runs.


### Visualization plan: all tested techniques

Create True-vs-Predicted and residual diagnostics for each evaluated technique (clean baseline, raw noisy, clean PCA, PCA reconstruction, matched-noise augmentation, autoencoder), for each available target. Add an aggregate Part1_E R2 comparison and separate PCA variance and geometry reconstruction plots. Use real held-out predictions; rerun experiments if prediction arrays were not retained. Keep test split/noise conditions comparable and clearly mark the clean reference as noise-free.


### Colab rerun and prediction inventory

User reports rerunning the full Colab notebook. Before plotting per-technique True-vs-Predicted and residual diagnostics, inventory existing prediction arrays in the active runtime. Reuse retained arrays; rerun only missing models. Suggested cell:

```python
import numpy as np
for name, value in list(globals().items()):
    if isinstance(value, (np.ndarray, list)) and ('pred' in name.lower() or 'prediction' in name.lower()):
        print(name, np.shape(value))
```

Note: this inventory does not capture pandas Series/DataFrames or predictions only held inside dictionaries, so absence does not prove the outputs were never computed.


### Prediction arrays recovered after full Colab rerun

All eight arrays have 1,302 entries: `predictions`, `raw_pred`, `pca_pred`, `part1_test_pred`, `matched_pred`, `pred`, `recon_pred`, `part1_denoised_pred`. These appear to correspond to held-out test samples, but generic variable names may have been overwritten. Next verify each array against `y_test['Part1_E']` using R2 and MAE and compare with documented experimental results before labeling graphs. Preserve any ambiguity rather than assuming all arrays are distinct experiments.


### Prediction inventory validation after notebook rerun

Metrics computed against `y_test["Part1_E"]` (1,302 held-out samples):

| Variable | Likely experiment | R2 | MAE |
| --- | --- | ---: | ---: |
| `predictions` | Unidentified, must inspect cell origin | -11.4317 | 0.051468 |
| `raw_pred` | Raw noisy geometry | 0.0768 | 0.014744 |
| `pca_pred` | Clean-PCA projection | 0.0781 | 0.014600 |
| `part1_test_pred` | Mixed-noise augmentation | 0.0626 | 0.014469 |
| `matched_pred` | Matched-noise augmentation | 0.1004 | 0.014361 |
| `pred` | Asymmetric PCA experiment, likely final 0.0838 run | 0.0838 | 0.014555 |
| `recon_pred` | PCA-reconstructed geometry | 0.0728 | 0.014611 |
| `part1_denoised_pred` | Autoencoder denoising, current rerun | 0.0041 | 0.014997 |

Earlier autoencoder run: R2 0.0258, MAE 0.014777, RMSE 0.017697. Current run differs; keep both runs documented and investigate training randomness/overwritten arrays. The unlabeled `predictions` array should not be used for graphs without provenance. Confirm same row alignment before comparison.

Next plot the six clearly identifiable Part1_E methods in a 2x3 true-vs-predicted grid using the shared y_test array, with ideal y=x diagonal and identical axis ranges.


### Visualization refinement: readable Part1_E scatter comparisons

Revised the six-panel True-vs-Predicted plot to use 21x13-inch layout, larger blue points (s=30, alpha=.65) with dark outlines, a prominent red dashed y=x reference, clearer labels, shared axis limits, and 300-DPI PNG export as `Part1_E_Method_Comparison.png`. Note that overplotting remains possible; consider density plots later.


### Interpretation of six Part1_E True-vs-Predicted plots

Each plot compares the held-out true Part1_E target (x-axis) to the corresponding model prediction (y-axis). Each dot is one held-out simulation, the y=x dashed line is perfect prediction, and R2 summarizes predictive fit. Results: raw noisy 0.0768, clean-PCA projection 0.0781, mixed-noise augmentation 0.0626, matched-noise augmentation 0.1004, PCA reconstruction 0.0728, autoencoder-denoised geometry 0.0041. All have weak predictive power for Part1_E at noise 0.05; horizontal clustering suggests predictions regress toward central values. The matched-noise approach is best of the six but remains weak. These are prediction plots, not geometry reconstruction accuracy or PCA explained-variance plots.


### Clean vs noisy Part1_E diagnostic figure

Added reproducible Colab plotting instructions to train a 100-tree RandomForestRegressor (random_state=42, n_jobs=-1) on clean spline geometry (`X_train_spline_base[feature_columns]`, `y_train['Part1_E']`), predict held-out clean spline geometry, and plot true-vs-predicted alongside existing `raw_pred` for noisy geometry. Both panels use identical axes, y=x reference, R2 and MAE labels, and save `Part1_E_Clean_vs_Noisy.png` at 300 DPI. Previously observed clean R2 ~0.9896 versus raw noisy R2 ~0.0768; verify new run rather than assume exact equality. Caveat: this compares clean-train/clean-test against noisy-train/noisy-test, not only test-time noise.


### Pending method change: smooth angular/radial spline noise

The user previously proposed replacing independent Gaussian x/y coordinate perturbations of the inner/outer spline boundaries with center-origin radial rays parameterized by angle. Proposed algorithm: estimate a center for each shape, represent its boundary as radius versus angle (or intersect angular rays with a spline for suitable star-shaped contours), perturb radius with a smooth angularly correlated noise field, reconstruct/resample the boundary, and compare on matched held-out simulations against current Gaussian-coordinate noise. Important caveats: validate ray intersections and handling of non-star-shaped contours; distinguish segmentation noise from actual physical geometry changes; preserve consistent target labels for measurement-noise experiments. This was discussed but has NOT yet been implemented or validated. Existing benchmark graphs use current Gaussian-coordinate perturbations plus separate bottom jitter.


### Implemented: pluggable Gaussian vs radial/angular spline noise

Created `noise_models.py` and `noise_comparison.py` with a smoke test in `test_noise_models.py`.
- `add_geometry_noise(X_base, method="gaussian"|"radial", noise_level, seed, bottom_jitter, harmonics)` accepts existing **spline-resampled clean geometry** and preserves the 54 feature columns and row order.
- Gaussian mode reproduces the earlier `add_fast_noise` behavior: bottom coordinate uniform jitter +/-0.01 followed by iid Gaussian noise on 36 shape coordinates.
- Radial mode uses sample contour centroid, polar angle of each existing nine spline points, and a 3-harmonic smooth periodic radial displacement to define noise. This is a *ray displacement of existing spline samples*, not an algorithm for solving exact angle-ray/spline intersections. It is best suited to approximately star-shaped contours.
- Per-sample radial field normalization targets shape-coordinate RMS `noise_level`, the expectation of Gaussian coordinate noise, and limits inward radial movement to 90% of original radius. Report measured shape-coordinate RMS and MAE because clipping can reduce the actual perturbation.
- Both retain identical bottom jitter on identical RNG seeds; if `noise_level=0`, specify `bottom_jitter=0` for truly clean geometry.
- `compare_noise_methods` trains two 100-tree Random Forests separately under otherwise matched settings with the same original train/test split, target, feature columns, and train/test seeds. It reports held-out Part1_E R2/MAE plus shape coordinate MAE/RMS. `plot_noise_example` overlays clean, Gaussian, and radial contours.
- A smoke test validated deterministic behavior, equality to original Gaussian generator, preserved shapes/index, zero-noise behavior, radial displacement direction, and the comparison runner on synthetic contour samples.
- No real-dataset metrics for radial noise have yet been collected. The hypothesis is that smooth boundary errors better approximate some segmentation errors. A fixed true label remains appropriate when noise represents *measurement corruption* only; physical deformation would require regenerated ground truth.
- Subsequent studies should sweep levels, multiple independent seeds, noise amplitude matched by realized RMS, and geometry validity checks; training a model on one noise distribution and testing on the other is a separate robustness test.


### First measured Gaussian versus angular-radial noise comparison (user Colab rerun)

User executed `compare_noise_methods` on held-out test split for Part1_E at `noise_level=0.05`, same 100-tree Random Forest hyperparameters and original train/test splits. Observed metrics:

| noise method | R2 | prediction MAE | shape-coordinate MAE | shape-coordinate RMS |
| --- | ---: | ---: | ---: | ---: |
| Gaussian iid X/Y coordinates | 0.076806 | 0.014744 | 0.039729 | 0.049675 |
| smooth radial angular perturbations of existing spline samples | 0.325831 | 0.012524 | 0.038628 | 0.050000 |

The radial result improves R2 by 0.249025 absolute (4.24x versus Gaussian) and lowers Part1_E prediction MAE by approximately 15.1%, while realized shape-coordinate RMS is nearly matched. An overlay of clean, Gaussian, and radial contours (sample 0) confirms the two noise patterns but individual displacements are visually small at plotted scale.

**Interpretation:** Under this *particular* noise model and fixed seed, smoothly correlated radial corruption is less harmful to material-property prediction than independent coordinate corruption. This does not establish that radial noise is a more faithful physical measurement model or that it universally outperforms Gaussian noise. Noise directionality, angular correlation, and amplitude distribution vary simultaneously. Additional validation: repeat with several independent train/test seeds, sweep amplitudes, inspect many shapes and shape validity, and optionally test cross-noise generalization. Retain original clean labels when perturbations are observational noise; if representing real physical geometry changes, recompute FEM labels.


### Extend Gaussian-versus-radial comparison to all three variable targets

Requested to compare Part1_E, Part3_E, and Part11_E (not Part2_E/Part3_E; the actual three variable dataset target names are Part1_E, Part3_E, Part11_E). Reuse `compare_noise_methods(X_train_spline_base, X_test_spline_base, y_train, y_test, target=target, noise_level=0.05)` for each target and concatenate results. This trains a separate matched Random Forest for each method and target, holding the train/test split and noise seeds fixed. Report R2 and prediction MAE for each target; compare MAE within targets rather than directly across differently scaled targets. No Part3_E/Part11_E radial results are known yet; await Colab output.


### First all-target Gaussian versus radial results (single seed, noise level 0.05)

| Target | Method | R2 | prediction MAE | shape-coordinate RMS |
| --- | --- | ---: | ---: | ---: |
| Part1_E | Gaussian | 0.07681 | 0.01474 | 0.04967 |
| Part1_E | Radial | 0.32583 | 0.01252 | 0.05000 |
| Part3_E | Gaussian | 0.99481 | 0.00453 | 0.04967 |
| Part3_E | Radial | 0.99527 | 0.00428 | 0.05000 |
| Part11_E | Gaussian | 0.94112 | 0.00805 | 0.04967 |
| Part11_E | Radial | 0.94435 | 0.00783 | 0.05000 |

Radial noise yields a higher R2 and lower prediction MAE for all three targets in this single-seed experiment. The most pronounced gain is for Part1_E; Part3_E/Part11_E changes are small and require repeated-seed confirmation. Geometry perturbation RMS is approximately matched. For visualization, pivot the existing `comparison` DataFrame by target and method, plot grouped R2 bars across all three targets and a separate zoomed view for Part3_E and Part11_E; save a 300-DPI PNG. No real-data scatter predictions were retained by the prior multi-target loop because its `predictions` dictionary was overwritten each iteration; collect them per target if needed.


### Interpretation clarification: what 'radial performs better' means

In the first matched-amplitude, single-seed comparison, models trained and tested on radially perturbed geometry achieved higher held-out material-modulus R2 and lower prediction MAE than models trained and tested on independently Gaussian-perturbed geometry. For Part1_E, R2 was 0.32583 versus 0.07681 and prediction MAE 0.01252 versus 0.01474, while shape-coordinate RMS was approximately 0.05 for both. This is a statement about downstream prediction under two corruption models, NOT proof that radial noise is more realistic or intrinsically superior. Both train and test noise distributions change, and the comparison does not isolate noise structure from all other distributional properties. Follow up with multi-seed validation and, ideally, evidence from real measurement/segmentation errors.


### Explanation of radial spline noise implementation

For each of the inner and outer nine-point spline-resampled contours, compute the centroid (mean x/y), radial vectors, radii, and polar angles. Sample random Fourier coefficients for three sine/cosine angular harmonics, weighted by 1/k^1.5, and evaluate a smooth periodic displacement field at each point's angle. Normalize the field to target shape-coordinate RMS near the chosen noise_level, clip inward displacement at 90% of the original radius, then update coordinates along the original center-to-point ray. The original bottom-coordinate uniform jitter remains unchanged. This method displaces *existing spline samples* and does not solve exact ray/spline intersections or re-fit the spline after displacement. Because the perturbations are smooth and correlated in angle, they may preserve different structural information than independent Gaussian X/Y coordinate noise; observed downstream R2 differences do not by themselves establish physical realism.


### Next experiment: radial-noise denoising benchmark

User chose radial spline noise as the primary synthetic noise model for upcoming experiments, pending realism validation. Reevaluate prior techniques individually at noise_level=0.05 on Part1_E, Part3_E, Part11_E: raw radial baseline, clean-trained PCA projection, PCA reconstruction, autoencoder geometry denoising, and radial-noise data augmentation. Preserve same train/test split and random seeds. Refit/retrain denoisers for radial corruption (do not present Gaussian-trained model results as matched radial experiments); keep clean geometry upper reference. Report downstream R2 and prediction MAE, plus geometry reconstruction MAE for methods that output geometry. Prevent train/validation leakage by splitting original sample IDs before augmenting. Begin with a minimal reproducible raw radial baseline for all targets and then add methods one at a time. No new radial denoising results exist yet.


### PCA reconstruction under radial noise: first Colab experiment prepared

Single-cell test prescribed: generate radial-noisy spline geometry for train/test (noise_level 0.05, seeds 42/43), fit 9-component PCA on clean training geometry exclusively, project/inverse-transform both noisy splits, measure held-out geometry MAE before/after PCA, train independent 100-tree RandomForestRegressor for Part1_E, Part3_E, Part11_E on reconstructed noisy training inputs, and report R2/prediction MAE on reconstructed noisy held-out test inputs against earlier radial baseline R2 (0.32583 / 0.99527 / 0.94435). No result available yet; next action is to inspect Colab output. Caveat: nine PCA dimensions are a test setting, not optimized, and geometric reconstruction loss alone does not establish downstream prediction improvement.


### Radial-noise PCA reconstruction results (9 components)

User Colab output, radial noise_level=0.05, train/test seeds 42/43, PCA fit to clean training geometry (54 features), 100-tree RF independently for each target:

- Geometry MAE before PCA: **0.027416**; after 9-PC inverse reconstruction: **0.014076** (approx. 48.7% reduction).
- Part1_E: raw radial R2 **0.32583**, reconstructed R2 **0.08541**, reconstructed prediction MAE **0.01459** (raw MAE was ~0.01252). Substantial regression despite coordinate MAE reduction.
- Part3_E: raw radial R2 **0.99527**, reconstructed R2 **0.99575**, reconstructed prediction MAE **0.00419** (raw ~0.00428). Very small gain.
- Part11_E: raw radial R2 **0.94435**, reconstructed R2 **0.93757**, reconstructed prediction MAE **0.00843** (raw ~0.00783). Regression.

Interpretation: geometric coordinate accuracy is not a sufficient surrogate for downstream inverse FEA prediction accuracy. A possible hypothesis is that aggressive PCA compression removes subtle Part1_E-informative geometry, even while reducing average reconstruction error. No causal mechanism established.

Next single-cell experiment provided: sweep PCA component count [3,6,9,12,18,27,36,45,54] for Part1_E; fit PCA on clean training geometry for each setting, reconstruct noisy radial train and test sets, train 100-tree RF, report held-out prediction R2/MAE and reconstructed geometry MAE. 54-PC reconstruction is a control expected to reproduce raw radial geometry/predictions to numerical precision. Avoid tuning final choices on test-set results without later separate validation.


### Radial PCA component sweep: Part1_E

Single-seed radial-noise experiment (noise_level 0.05), PCA fit on clean train, reconstruct noisy train/test, RF100 Part1_E:

| PCA components | Test R2 | Prediction MAE | Geometry MAE |
|---:|---:|---:|---:|
|3|0.02189|0.01504|0.00772|
|6|0.08027|0.01461|0.01232|
|9|0.08541|0.01459|0.01408|
|12|0.09273|0.01457|0.01476|
|18|0.10954|0.01445|0.01852|
|27|0.11753|0.01444|0.02096|
|36|0.15931|0.01406|0.02433|
|45|0.22367|0.01347|0.02735|
|54|0.32583|0.01252|0.02742|

Key finding: greater PCA dimensionality monotonically improves held-out Part1_E prediction R2 and prediction MAE, but worsens geometry coordinate MAE against the clean reference. Full 54-PC reconstruction reproduces the un-denoised radial baseline. Thus minimum coordinate reconstruction error does not imply best inverse FEA prediction. Hypothesis: low-variance geometric details discarded by PCA are predictive for Part1_E; not yet proven. This sweep used test data for exploratory comparison; use train-only validation for model selection and final untouched test for confirmatory evaluation. Next benchmark could be radial-noise-trained autoencoder or radial data augmentation, maintaining all-target evaluation and clean/raw baselines.


### Radial-trained denoising autoencoder: first training cell prepared (awaiting results)

Train a nonlinear 54->128->64->32->64->128->54 autoencoder using Gaussian-like standardization fit on CLEAN subtraining geometry and MSE loss between corrupted radial inputs and clean targets. Use an 80/20 split of the 5208 clean training geometries before augmenting, three independent radial noise realizations (seeds 101/102/103) at level 0.05 on subtrain, a disjoint clean validation subset with radial noise seed 404, Adam(0.001), early stopping by validation loss patience=12, max 100 epochs, batch 128. Evaluate raw-noisy versus AE-denoised geometry MAE on validation only; hold test out until model pipeline evaluation. Model variables: `autoencoder_radial`, `scaler_ae`, `history_radial`. Revisit training on full train after validation-based tuning before final held-out benchmark if appropriate. No results yet.


### Radial-noise autoencoder: first geometry-validation result

User reports early stopping after **36 epochs** for radial-noise-trained, 54-coordinate denoising autoencoder. On held-out *validation subset of training geometries* (not the final test set), noisy geometry MAE **0.027417** and autoencoder-reconstructed geometry MAE **0.003168**, a reported **88.44% reduction**. This is a validation-only geometric result, **not evidence of improved elastic modulus predictions**. Nine-component PCA earlier reduced test geometry MAE ~48.7% but worsened Part1_E prediction; evaluation populations differed, so do not treat numeric contrast as controlled head-to-head.

Next provided single Colab evaluation cell: regenerate radial-noisy train/test geometries with level=0.05 seeds 42/43; apply trained autoencoder and scaler to both (without refitting); report held-out test geometry MAE before/after; train RF100 per target Part1_E / Part3_E / Part11_E using denoised training geometries and evaluate denoised test geometries against prior radial baselines (R2 0.32583 / 0.99527 / 0.94435; MAE approximately 0.01252 / 0.00428 / 0.00783). Await test metrics. Caveat: RF uses all original training rows, including the AE validation subset used in early stopping; these remain inside the designated training split, not held-out test leakage. Treat experiment as exploratory; compare repeated seeds and preserve untouched final test in subsequent redesign.


### Radial autoencoder held-out test results

At radial shape noise level 0.05, train seed 42 and test seed 43, AE trained with train-internal clean/noisy split:

- Held-out test geometry MAE before AE: **0.027416**; after AE: **0.003177** (about **88.4% lower**).
- **Part1_E:** raw radial R2 **0.32583**, MAE **0.01252**; AE-denoised R2 **0.08265**, MAE **0.01442** (worse).
- **Part3_E:** raw radial R2 **0.99527**, MAE **0.00428**; AE-denoised R2 **0.99727**, MAE **0.00332** (better; ~22.4% lower prediction MAE).
- **Part11_E:** raw radial R2 **0.94435**, MAE **0.00783**; AE-denoised R2 **0.84649**, MAE **0.01347** (worse).

Central finding: even dramatic coordinate reconstruction gains do not guarantee improvements to the inverse FEA prediction objective. Loss of low-amplitude predictive geometric details is a plausible hypothesis but not established. Proposed next investigation: validation-selected blending of raw radial and AE-reconstructed inputs or a supervised/task-aware denoising objective; avoid using the held-out test set to pick blend weights. Caveat: these are exploratory, single-seed comparisons.


### Visualization convention for future experiments

User requested graphs **for every benchmark**, not only numeric output. Each experiment should save (300-DPI PNG, Colab downloadable) paired per-target R2 and prediction MAE comparisons against the raw radial baseline, plus geometry MAE before/after denoising when applicable. When per-sample predictions are available, also save true-versus-predicted scatter and residual plots, per target; do not label aggregate results as predictions or fabricate arrays. Chart code provided for current radial AE experiment uses existing `ae_results` and geometry MAE 0.027416 -> 0.003177 to save `Radial_Autoencoder_Results.png`, with no retraining. Build automatic figure generation into subsequent experiment cells and preserve charts as research artifacts.


### Reaffirmed end-goal: multi-agent InverseFEA with learned orchestrator/encoder

User explicitly clarified the ultimate objective: build a modular system of specialized inverse-FEA agents/skills, controlled by a main orchestrator/encoder that selects and invokes specialists depending on input needs. A candidate architecture includes geometry processing, noise characterization/denoising, tissue-count-specific inverse-FEA prediction experts (>3 tissues eventually), and validation/uncertainty assessment. These are proposed architectural roles, not yet implemented agents. Current Gaussian-versus-radial, PCA and autoencoder studies are skill discovery and benchmarking, not the final deliverable. Particular radial AE results show downstream prediction effects differ by target; routing should therefore be conditional and validation-trained, not always denoise. Next partial-AE blending study should be framed as finding a tunable skill and collecting evidence to later train/evaluate a gate. Keep per-experiment plots and logged metrics. Distinguish noise generation used for robustness benchmarking from denoising used at inference; never assume ground-truth noise type is known to orchestrator.


### Partial radial-autoencoder blending experiment prepared

Next Colab test sweeps alpha in [0, .25, .5, .75, 1] in X_blend=(1-alpha)*X_radial+alpha*X_AE, for all targets Part1_E, Part3_E, Part11_E. Uses prior X_subtrain/X_val split and existing trained AE, samples radial corruption at 0.05 with seeds 42 and 43, trains RF100 independently for every alpha-target pair. Reports R2, modulus prediction MAE, and geometry MAE, and saves/downloads three 300-DPI line plots: Radial_Blending_R2.png, Radial_Blending_prediction_MAE.png, Radial_Blending_geometry_MAE.png. IMPORTANT methodological caveat: AE used X_val for early stopping, so this is exploratory rather than independent hyperparameter tuning; do not compare these metrics numerically as though they were previous held-out test scores. Later use a three-way split with separate AE early stopping and gating/blend tuning subsets, and reserve test for final evaluation. No blend results yet.


### Radial AE blending sweep: reported results (exploratory train-internal validation)

Alpha blend X=(1-alpha)*radial_noisy+alpha*AE_reconstructed. Noise level 0.05. User provided MAE graph and table for 15 alpha/target combinations:

| alpha | Geometry MAE | Part1 R2 / prediction MAE | Part3 R2 / prediction MAE | Part11 R2 / prediction MAE |
|---:|---:|---|---|---|
|0.00|0.02740|0.28426 / 0.01237|0.99560 / 0.00424|0.94998 / 0.00785|
|0.25|0.02065|0.22448 / 0.01290|0.99677 / 0.00364|0.94662 / 0.00816|
|0.50|0.01408|0.17316 / 0.01340|0.99731 / 0.00341|0.92025 / 0.01021|
|0.75|0.00777|0.12028 / 0.01375|0.99752 / 0.00328|0.85022 / 0.01371|
|1.00|0.00312|0.08561 / 0.01371|0.99730 / 0.00335|0.87701 / 0.01226|

Observed best among tested alphas by R2: Part1 alpha=0, Part3 alpha=0.75, Part11 alpha=0. By MAE, Part1 alpha=0, Part3 alpha=0.75, Part11 alpha=0. Part3 MAE improvement 0.00424 to 0.00328 (~22.6%) with 0.75 blending. Geometry coordinate MAE decreases as alpha increases but Part1/Part11 predictions worsen. This supports investigating target-specific preprocessing routes rather than universal denoising.

CRITICAL limitation: X_val was used in AE early stopping and then reused for alpha sweep, so observed best alpha is not independently validated. Need fresh independent AE-early-stopping vs gate-selection vs final-test split; avoid claiming orchestrator implemented or validated. Preserve current figures and repeat over multiple noise seeds and tissue configurations.


### Plan: clean orchestration notebook and independent data request (2026-10-12)

User plans to start a new Colab notebook, provisionally `MultiTissue_InverseFEA_Orchestrator_Experiment_01.ipynb`, to begin an independent comparison of static target-dependent expert routing versus fixed raw/noisy and full-autoencoder baselines. Recommended notebook structure: (1) environment/imports from existing repository, (2) load and verify 54 geometry features and Part1_E/Part3_E/Part11_E targets, (3) split by original simulation/geometry group before any noisy augmentation, (4) radial noise generation, (5) train AE and create candidate raw/blended/denoised representations, (6) per-target inverse FEA experts, (7) separate routing selection split, (8) evaluate and export automatic plots/results/models. Keep separate subsets for AE early stopping, routing selection, and test; the existing 1302-sample test set has been inspected repeatedly during research and should be treated as exploratory, not a fresh final unbiased test.

Recommended requesting a NEW independent simulation dataset from instructor for final performance validation, with original clean geometry, material-property labels, simulation/geometry IDs, diverse tissue counts (>3), and if possible realistic measurement or segmentation error information. More independent variation is preferable to repeated near-duplicate cases. Existing data suffice to start notebook development immediately; don't block work waiting for new data. A simple target-conditional routing policy is only a prototype, not a learned orchestrator; later extend to tissue-count-specific experts and a trained routing encoder.


### Orchestrator Experiment 01 — environment verified and dataset inspection cell provided

User executed fresh Colab environment successfully: downloaded `noise_models.py` and `noise_comparison.py`; NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, TensorFlow 2.21.0; `GEOMETRY_COLS` has 54 features. Next incremental cell loads the original `2025_4_14_intermediate.csv` from `SomeoneJN/BME_Muscle_Strength_Prediction_FA25/main/Spring_25`, checks expected 54 feature and three target columns (`Part1_E`, `Part3_E`, `Part11_E`), missing values, duplicate full rows, unique geometry configurations, target distributions and first rows. Await dataset output before selecting split strategy; unique geometries alone do not rule out closely related simulations.


### Dataset source identifier investigation

The full 99-column list includes a leading `File Name` column, which was missed by the keyword-based ID search (keywords did not include `file` or `name`). The rest includes modulus labels, Pressure, Inner_Radius, Outer_Radius, coordinate families, and 54 selected geometry features. A fresh Colab cell was provided to inspect filename uniqueness, missingness, examples from first/last rows, repeated filename frequencies, and repeated rows. Await output before selecting independent split groups; even unique filenames do not establish independent simulations, and filename patterns may encode shared cases.


### Filename metadata finding and physical-configuration check

All 6510 `File Name` strings are unique, but names encode `Part1_E(...)Part3_E(...)Part11_E(...)` material parameters rather than an independent simulation family identifier. Examples include filename `Part1_E(0.016)Part3_E(0.201)Part11_E(0.1807)` despite tabular Part1_E shown as 0.16; filename-vs-target mismatch needs checking, not guessing. Next Colab cell extracts three filename-encoded values via regex and compares their ranges/match counts with target columns; counts unique values of Pressure, Inner_Radius, Outer_Radius, Part7_E and Part10_E, plus unique combinations. Shared physical settings alone do not prove related cases or establish a valid grouping key. Await results before choosing evaluation splitting protocol.


### Orchestrator Experiment 01 — filename scaling and fixed simulation conditions

Filename-encoded Part1_E ranges 0.01–0.016 while target-column Part1_E ranges 0.10–0.16; filename Part3_E and Part11_E each exactly match their target columns for all 6510 rows. Pressure=0.019, Inner_Radius=1.25, Outer_Radius=1.75, Part7_E=1, Part10_E=1 are constant throughout; exactly one combination of these physical settings. This dataset supports an initial fixed-configuration inverse prediction/routing experiment but cannot establish generalization to new pressures, radii, or tissue counts. User received a one-cell check of per-row filename/column ratios and counts per Part1_E value to verify a uniform factor-of-ten discrepancy. Clarify physical units and correct Part1_E scale with instructor before scientific reporting; request independent data with varied physical configurations and tissue counts. Avoid grouping by the five constant parameters (only one group) or by filename target labels (leakage). Await ratio/count output before selecting split.


### Orchestrator Experiment 01 — verified modulus scaling; proposed four-way split

Colab confirmed filename Part1_E to target-column Part1_E ratio is exactly 10 for all 6510 samples; Part3_E and Part11_E ratios are exactly 1. Part1_E has 61 discrete levels, 0.100–0.160. Retain original column labels for current experiments, flag factor-of-ten ambiguity for instructor/units clarification. A single Colab cell was provided to create 60% model training (3906 rows), 10% AE early-stopping validation (651), 10% routing selection validation (651), and 20% evaluation (1302), with random seeds 42/43/44 and stratification by Part1_E. Validate disjoint original row indices. Caveats: no trustworthy simulation-family ID, constant physical configuration, and previous extensive use of source dataset mean results remain exploratory and do not establish cross-configuration generalization; reserve this new split's test rows from any new tuning. Await execution output before spline preprocessing and radial noise generation.


### Orchestrator Experiment 01 — geometry verified; periodic spline prototype

User's Colab Step 5A succeeded: train geometry (3906,54), AE validation (651,54), routing validation (651,54), test (1302,54); corresponding labels have 3 targets; all coordinates finite and feature ordering matches canonical GEOMETRY_COLS. Uploaded original-geometry plot depicts nested 9-point inner/outer polygonal contours without obvious intersections. Provided next single Colab cell: periodic cubic spline on each closed innerShape/outerShape contour, chord-length parameterization, resample nine points per contour, preserve bottom columns, and plot/download a 300-DPI comparison of original polygons versus dense spline curves. This spline procedure is newly written and not yet proven equivalent to earlier research notebook; check output/geometry integrity before noise injection or cross-experiment comparisons.


### Orchestrator Experiment 01 — spline plot inspected; coordinate validation next

User confirmed periodic cubic-spline preprocessing completed for all four splits: train (3906,54), AE validation (651,54), routing validation (651,54), test (1302,54). Uploaded example plot shows smooth nested inner and outer contours and nine resampled points; the single example does not guarantee all shapes remain nonintersecting. Next Colab cell validates finite inner/outer resampled coordinates and summarizes minimum Euclidean distance between sampled inner/outer point sets by split. Caution: sampled-point gap is not a true continuous-curve clearance or contour intersection test, and positive values do not establish physical validity. Await validation table before radial noise generation; stronger geometric checks may be needed.


### Orchestrator Experiment 01 — spline validation passed; radial-noise generation step

Step 5C results: all four spline geometry subsets have zero invalid coordinates. Minimum sampled-point gaps: train 0.79653, AE validation 0.79985, routing validation 0.80629, test 0.80033; mean sampled-point gaps about 0.886–0.889. These sampled distances do not prove nonintersection of continuous contours. Step 6A Colab cell provided: use repository add_geometry_noise with method radial, noise_level=0.05, harmonics=3, independent random seeds train 101, AE validation 202, routing validation 303, test 404; retain clean geometry_spline references and construct geometry_noisy dictionary, print 54-feature coordinate MAE by split, plot a clean-vs-noisy training contour and download 300 DPI PNG. Note 54-feature MAE includes bottom coordinate jitter; next step is AE training preparation after reviewing results.


### Orchestrator Experiment 01 — radial noise validated; AE scaling preparation

User ran Step 6A: 54-feature coordinate MAE between radial-noisy and clean spline geometry is train 0.027467, AE validation 0.027469, routing validation 0.027524, test 0.027483. Uploaded example plot shows subtle smooth angular/radial contour deviations with nested shapes preserved in the example; this is not proof of physical realism or global nonintersection. Next one-cell Colab step fits StandardScaler on clean training spline geometry only, transforms noisy training/AE-validation inputs and their corresponding clean targets, checks shapes (3906,54) and (651,54), and verifies finite values. Routing validation and test remain excluded from AE training and early stopping. Await user output before training AE and generating learning-curve plots.


### Orchestrator Experiment 01 — autoencoder denoising confirmed on AE validation

User reports AE validation geometry MAE before denoising 0.027469 and after denoising 0.003375, a relative reduction of 87.72% on the 651-row AE validation subset. This is similar to earlier radial AE geometry results (~88.4%), though preprocessing, split and training augmentation differ. The training run stopped after 58 epochs and achieved best scaled validation MSE 0.02274567 at epoch 46. Geometry denoising quality alone does not establish downstream inverse FEA predictive gains. Next Colab cell plots one AE-validation sample's inner and outer clean/noisy/reconstructed contours and exports 300-DPI PNG, then will proceed to per-target prediction experts. Do not tune on routing validation or test prematurely.


### Orchestrator Experiment 01 - Step 8A completed; Part1 routing comparison next

User confirmed training representations all (3906,54). Training geometry MAE: Raw 0.027467, 50-percent blend 0.014091, fully denoised 0.003170 (approximately 88.5 percent reduction vs raw). These are training-set reconstruction diagnostics, not independent predictive scores. Next Colab cell constructs corresponding routing-validation raw, 50-percent blended, and AE-denoised representations (651 rows), trains separate 100-tree RandomForestRegressor models on the training set for Part1_E and each representation, computes routing-validation R2 and prediction MAE, and plots/downloads 300 DPI side-by-side comparison. Routing validation is used for strategy selection and must not be treated as unbiased final test; do not touch reserved test set during model selection. Await user table/plot before evaluating Part3_E and Part11_E.


### Orchestrator Experiment 01 — Part1_E routing-validation comparison

User evaluated separate 100-tree RandomForestRegressor experts for Part1_E on the 651-row routing-validation split using training/routing geometry representations. Raw radial geometry: R2=0.337622, prediction MAE=0.012341. 50-percent raw/AE blend: R2=0.218590, MAE=0.013305. Full AE denoising: R2=0.121549, MAE=0.013861. Raw is best by both metrics, despite AE reducing training geometry MAE. Full AE R2 is about 64 percent lower relative to raw; this does not prove a mechanism, but is consistent with losing task-relevant geometric details. Candidate Part1 route is raw. Next single Colab cell trains and compares raw, 50-percent blend and full AE strategies for Part3_E and Part11_E on routing validation, displays R2/MAE table, and exports 300-DPI line plots. Routing-validation comparisons are model-selection evidence, not independent final evaluation; keep test untouched.


### Orchestrator Experiment 01 — all-target routing-validation results and Part3 blend sweep

On routing validation, Part1_E: raw R2 0.337622, MAE 0.012341; 50% blended R2 0.218590, MAE 0.013305; full AE R2 0.121549, MAE 0.013861. Part3_E: raw R2 0.994570, MAE 0.004675; 50% blended R2 0.996668, MAE 0.003857; full AE R2 0.995847, MAE 0.004074. Part11_E: raw R2 0.946273, MAE 0.008308; 50% blended R2 0.911034, MAE 0.010947; full AE R2 0.862285, MAE 0.013185. Initial target-conditioned routing: Part1 raw (alpha=0), Part3 50% blend (alpha=.5), Part11 raw (alpha=0). Part3 MAE improves ~17.5% versus raw on routing-validation data; results are selection evidence, not final independent estimates. Next one-cell Colab sweep trains 100-tree Random Forest for Part3_E at alpha [0,.25,.5,.75,1] on the existing training subset and scores routing-validation R2/MAE, exports a 300 DPI plot; do not use reserved test set to choose alpha. Await sweep before final routing policy and test comparison. This remains fixed per-target routing, not a learned intelligent orchestrator.


### Orchestrator Experiment 01 - fixed-policy evaluation initiated

Routing-validation alpha sweep for Part3_E: alpha 0: R2 0.994570, MAE 0.004675; 0.25: 0.995822, 0.004170; 0.50: 0.996668, 0.003857; 0.75: 0.996040, 0.004081; 1.00: 0.995847, 0.004074. Best observed alpha is 0.50 on both metrics. Fixed target-specific routing policy before test: Part1_E alpha 0, Part3_E alpha 0.50, Part11_E alpha 0. User received a single Colab evaluation cell for reserved 1302-row test split comparing Raw-only (alpha 0 for all), Fully denoised (alpha 1 for all), and fixed Orchestrator (target-specific alpha). Each 100-tree RF is fit only on the 3906 training rows using matched training and test geometry representations, and results R2 and prediction MAE are tabulated and plotted to downloadable 300-DPI PNG. The two raw-routed targets should match raw-only exactly. Caveat: this is a held-out split within the same previously explored simulation dataset, not an entirely independent new dataset. Await test results before drawing conclusions; do not adjust policy based on test outcomes.


### Orchestrator Experiment 01 — held-out split evaluation of fixed routing policy

User executed Step 9A on 1302 reserved test rows after selecting routing policy using routing-validation subset: Part1_E raw alpha 0, Part3_E 50-percent AE blend alpha 0.5, Part11_E raw alpha 0. Test results (R2, prediction MAE): Part1 raw 0.319413 / 0.012378; full AE 0.062743 / 0.014305; routed 0.319413 / 0.012378. Part3 raw 0.994418 / 0.004647; full AE 0.995861 / 0.003984; routed 0.996628 / 0.003735. Part11 raw 0.942098 / 0.008323; full AE 0.855677 / 0.012952; routed 0.942098 / 0.008323. For Part3 the routed policy reduces prediction MAE by about 19.6 percent vs raw. Raw and routed are exactly equivalent for Part1 and Part11 by design. The routed policy is no worse than raw on these measured metrics for this split, and better on Part3; no learned gating model has been trained. Important limitations: one held-out random split from same previously explored fixed-configuration dataset, no independent external simulation cases, no multiseed uncertainty estimates, no validation of transfer to different tissue counts. Do not retune policy on test results. Next recommended experiment: quantify robustness across independent seeds/replicates or collect truly external simulations before claims of generalization; export prediction scatter/residuals and research figures.


### Orchestrator Experiment 01 — robustness experiment initiated (2026-10-13)

Following the fixed-policy test evaluation, user requested the recommended robustness experiment. Supplied one Colab cell for ten fresh radial-noise realizations (seeds 500 through 509) applied to the same 1302 clean test geometries. The existing trained denoising autoencoder and Random Forest experts are reused without retraining, with fixed routes Part1_E alpha 0, Part3_E alpha 0.5, Part11_E alpha 0. The cell computes R2 and prediction MAE for Raw-only, Fully denoised, and Orchestrator on each seed; summarizes means and sample standard deviations across noise seeds and exports a 300 DPI MAE error-bar chart. This isolates noise-realization sensitivity for fixed experts; the repeated measurements share test geometries and are not independent datasets, and the previously inspected test split makes results exploratory. Await results before calculating paired per-seed deltas and producing prediction/residual plots.


### Orchestrator Experiment 01 — ten-seed robustness results

On fixed 1302 test geometries, using fixed experts and radial noise seeds 500–509, user reported mean/std R2 and MAE: Part1 raw and orchestrator R2 0.311022±0.008713, MAE 0.012393±0.000076; full AE R2 0.069899±0.018416, MAE 0.014228±0.000156. Part3 raw R2 0.994845±0.000296, MAE 0.004520±0.000104; orchestrator alpha=.5 R2 0.996974±0.000152, MAE 0.003614±0.000090; full AE R2 0.996170±0.000199, MAE 0.003897±0.000077. Part11 raw and orchestrator R2 0.943302±0.001781, MAE 0.008215±0.000144; full AE R2 0.856508±0.004038, MAE 0.013036±0.000231. Part3 mean MAE reduction approximately 20.0% vs raw. These are repeated noise realizations on the same test cases, not ten independent datasets. Next: calculate paired per-seed Part3 deltas, count how many seeds routed MAE beats raw, and export a per-seed plot; then produce true-vs-predicted and residual plots. Do not adjust fixed routing policy based on these test results.


### Orchestrator Experiment 01 — diagnostic plots interpreted; Part1 clean/noisy ablation proposed

User uploaded true-vs-predicted and residual plots for raw-only vs fixed orchestrator on reserved test noise seed 404. Part1_E predictions visibly compressed toward mean (~0.13), with pronounced positive residuals at low true values and negative residuals at high values; raw and orchestrator overlap by design. Part3_E predictions cluster near identity line; orchestrator has tighter residuals overall, but dispersion increases toward high target values. Part11_E has reasonably strong identity alignment but some mean-regression and residual bias at extremes; raw and orchestrator overlap by design. Proposed next diagnostic on train/routing-validation only, with no test refitting or retuning: fit separate 100-tree RandomForestRegressor for Part1_E using clean spline train/routing geometry and radial-noisy train/routing geometry; compare R2 and MAE, plot scatter and residuals at 300 DPI. Purpose: distinguish radial noise sensitivity from limitations in clean-geometry predictability. Routing-validation is already used for selection, so these are exploratory diagnostics. Prior ten-seed Part3 paired improvement: 10/10 wins, mean relative MAE reduction 20.04%, range 18.53%–23.12%.


### Step 11A — Part1_E clean-vs-noisy prediction diagnosis

Following diagnostic plots showing pronounced Part1_E regression-to-the-mean under radial noise (test R2 around 0.32), proposed a controlled two-arm ablation using existing disjoint 3906-row training and 651-row routing-validation splits: train separate RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) on clean spline versus matched radial-noisy geometry, evaluate Part1_E R2 and prediction MAE on matching clean or noisy routing-validation geometry, and export 300-DPI scatter and residual figures. Hold the test split untouched. If clean performs near-perfect but noisy performs poorly, prioritize noise-robust representations; if clean also struggles, inspect predictive model/features; intermediate results warrant both. Routing-validation is previously used for selection so diagnostic remains exploratory. Await user results; do not pre-judge outcomes.


### Architecture clarification: orchestrator as learned preprocessing controller

User clarified intended architecture: the orchestrator controls preprocessing of noisy geometry, routing to specialized learned denoising models trained on many noisy/clean geometry pairs so they discover corruption patterns. The downstream InverseFEA elastic-modulus predictors are separate from preprocessing. Candidate denoising experts specialize in Gaussian independent coordinate noise, smooth/radial noise, and mixed/complex noise. Orchestrator should estimate noise characteristics from incoming geometry rather than assume known noise labels, and may choose raw-preservation, one expert, or combine outputs. Evaluation must account for both coordinate reconstruction accuracy and downstream modulus prediction; previous findings show that minimizing geometry error alone may harm Part1_E and Part11_E. Existing radial autoencoder is a first noise-specialist prototype. The earlier fixed per-target alpha policy is a baseline, not the desired final learned preprocessing orchestrator. Next research direction: clean-vs-noisy Part1 diagnostic then train experts on multiple noise seeds/strengths using training-only data, compare on validation with clean-reference and prediction metrics, and learn or design a routing mechanism without leaking test information.


### Architectural correction — hierarchy of skill modules and tissue-configuration inverse experts (2026-10-13)

User corrected a drift in the design: the end goal is a modular multi-tissue InverseFEA system in which a main orchestrator chooses reusable skills (geometry inspection/validation, noise handling, feature extraction, uncertainty) and delegates the actual inverse elastic-modulus prediction to specialized experts trained for DIFFERENT tissue configurations/counts (initially 1–3 if data available, eventually >3). The recent fixed per-target alpha policy (raw for Part1/Part11, blend for Part3) is merely an empirical preprocessing subroutine/baseline, NOT the overarching intended orchestrator. Likewise noise-specific denoisers can be subordinate skills but are not the primary multi-tissue experts. Hierarchical controller design: identify tissue configuration and validate input, invoke necessary preprocessing skill conditionally, route to compatible inverse model, evaluate plausibility/uncertainty, return prediction and route provenance. Critical caveat: 3 varying output columns Part1_E, Part3_E, Part11_E do not prove dataset contains separately labeled 1-, 2-, and 3-tissue cases; fixed Part7_E and Part10_E and other constants do not resolve mesh anatomy or true count. Verify FEM ground truth and obtain datasets for each tissue-count regime before implementing count specialists or claiming count generalization. Next development: define clear model/skill interfaces and build orchestration controller around current fixed-configuration expert, without pretending count classification is learned from the existing data.


### Confirmed dataset regime: one tissue; start expert development

User confirmed that current simulation dataset represents ONE tissue, despite three variable label columns (`Part1_E`, `Part3_E`, `Part11_E`), whose interpretation as distinct tissue counts was explicitly rejected. Treat current model as the initial one-tissue InverseFEA expert; retain three output labels without inferring separate tissue datasets. User agreed to staged architecture: improve one-tissue inverse predictor, package geometry/noise/uncertainty reusable skills, build orchestration controller that calls skills and one-tissue expert, expand with trained two-/three-/more-tissue experts only after obtaining corresponding datasets. Next single Colab Step 11A compares RF100 Part1_E on clean train/routing geometries against same architecture on radial-noisy train/routing geometries, reports R2 and MAE and exports 300-DPI bar plot. Keep test out of development. Await experiment results before choosing noise-specialist development versus stronger prediction models.


### Step 11A result: clean geometry versus radial noise for the one-tissue Part1 expert (2026-10-13)

The 100-tree Random Forest on clean spline geometry (training on 3906 samples, routing validation on 651) attained Part1_E R2=0.988615 and MAE=0.001391. The same estimator family trained/evaluated with radial noise level 0.05 attained R2=0.337622 and MAE=0.012341 (approximately 8.9 times the clean MAE). This points to severe noise sensitivity under the tested corruption; the existing RF is capable of strong clean-geometry prediction, so prioritize noise robustness rather than replacing the estimator without evidence. Prior AE denoising improved geometry MAE but degraded Part1 prediction, so geometry-only reconstruction is not a sufficient preprocessing objective.

Next Step 11B: train a new RF100 using four independent radial-noisy realizations (seeds 111, 222, 333, 444, same noise level 0.05) of each of the 3906 training geometries, making 15624 augmented observations, with labels repeated accordingly. Evaluate only on previously fixed 651-sample routing-validation noisy geometry, compare R2 and MAE against single-noise RF baseline 0.337622/0.012341, and export a 300 DPI comparison PNG. All augmented copies remain confined to training; no test access. Four copies are not four times as many independent simulations; treat outcomes as exploratory and avoid claiming learned denoising unless explicitly tested.


### Step 11B outcome and Step 11C proposed — mixed clean/noisy one-tissue expert

User's Part1_E routing-validation results: single-noise RF100 R2 0.337622, MAE 0.012341; four-realization multi-noise training RF100 on 15624 augmented rows R2 0.460584, MAE 0.011028, corresponding to ~10.6% MAE reduction. Clean-geometry RF100 reference R2 0.988615 and MAE 0.001391. Multi-noise augmentation helps but large noise sensitivity remains. Step 11C Colab cell proposed to train a single mixed expert on the 3906 clean training geometries plus four noisy versions (total 19530 examples), using the existing robust_model as comparison. Evaluate both fixed models on the same 651 clean and radial-noisy routing-validation geometries, report R2 and MAE, export 300-DPI grouped MAE plot. No access to test split; the mixed input samples are augmented copies, not independent simulations. Continue in the current Colab until a clear module packaging milestone. Avoid presenting an unsolicited infographic.


### Step 11C runtime failure: empty Random Forest estimators

User got IndexError: list index out of range in sklearn RandomForestRegressor.predict during mixed training comparison, at self.estimators_[0]._support_missing_values(X). This means an estimator object has an empty estimators_ list; root cause is not yet known. Before suggesting a full retrain, provided a single Colab diagnostic cell printing type, estimators_ length (or no fit), and configured n_estimators for both robust_model and mixed_model. Await output; if either is zero/missing, refit only the affected model(s) from existing X_augmented/y_augmented and X_mixed/y_mixed respectively; avoid resetting dataset or restarting session unnecessarily.


### Step 11C diagnostic: broken multi-noise model object

User inspected RandomForestRegressor objects after IndexError in predict: robust_model reports 0 fitted trees despite n_estimators=100, while mixed_model reports all 100 fitted trees. The cause of robust_model's empty estimators_ is not established; previously reported Part1 multi-noise results are unchanged. Supplied a single Colab repair cell rebuilding and fitting robust_model only from existing X_augmented/y_augmented, asserting both models contain 100 trees, then evaluating robust_model and intact mixed_model on clean and noisy routing-validation geometry. Exports a 300-DPI MAE comparison plot. Stay in same Colab and await results before choosing a final one-tissue noise strategy.


### Step 11C mixed clean/noisy results — specialization trade-off

User's Step 11C on Part1_E routing-validation split: multi-noise-only RF100 evaluated clean R2=0.521001, MAE=0.010159; evaluated radial-noisy R2=0.460584, MAE=0.011028. Clean + noisy RF100 evaluated clean R2=0.982509, MAE=0.001688; evaluated radial-noisy R2=0.357392, MAE=0.012167. Thus including clean rows substantially restores performance for clean inputs but raises noisy-input prediction MAE by about 10.33% relative to multi-noise-only. Results suggest a trade-off in training distributions, not proof an orchestrator can infer input corruption or that one universal routing rule works. Next proposed exploratory experiment: measure both existing frozen models on the routing-validation geometries across a noise severity grid plus truly clean inputs. All tests remain on routing validation; use distinct seeds by level and plot both R2 and prediction MAE to identify crossover or regime-dependent behavior. Keep current Colab until skills modularization milestone.


### Step 11D: noise severity reveals bottom-jitter confound (2026-10-13)

User's routing-validation Part1_E severity sweep for frozen RF experts showed a severe discontinuity between truly clean geometry and nominal radial noise 0.000: clean+noisy model clean R2=0.982509, MAE=0.001688; at radial 0.000 R2=0.349490, MAE=0.011761. The cause may be that add_geometry_noise applies bottom-coordinate uniform jitter +/-0.01 even at radial noise_level=0. Part1 mixed model at radial 0.05 R2=0.364565, MAE=0.011916, similar to radial 0.000; at radial 0.075 R2=0.143126, MAE=0.013783. Multi-noise expert clean R2=0.521001, MAE=0.010159; radial 0.000 R2=0.474971, MAE=0.010689; radial 0.05 R2=0.480062, MAE=0.010704; radial 0.075 R2=0.082574, MAE=0.013979. The actual causal contributor needs isolation: propose controlled Step 11E ablation with exactly four conditions on routing validation: clean; bottom-only jitter with radial level 0 and bottom_jitter 0.01; radial contour-only with radial level 0.05 and bottom_jitter 0; combined with radial level 0.05 and bottom_jitter 0.01. Reuse existing fitted multi-noise and clean+noisy experts and same evaluation rows and random seed; compare target R2/MAE and coordinate MAE for bottom vs shape features. Export figure at 300 DPI. Keep test untouched and current Colab active; decide subsequent specialized skill only after condition-specific evidence.


### Step 11E bottom versus contour noise result; Step 11F ablation

User reports fixed routing-validation Part1_E noise-source ablation. Multi-noise RF: Clean R2=0.521001 MAE=0.010159; bottom-only (bottom MAE 0.005003, contour MAE 0) R2=0.474971 MAE=0.010689; contour-only (contour MAE 0.038661, bottom MAE 0) R2=0.515567 MAE=0.010286; combined R2=0.480062 MAE=0.010704. Clean+noisy RF: clean R2=0.982509 MAE=0.001688; bottom only R2=0.349490 MAE=0.011761; contour only R2=0.308950 MAE=0.012121; combined R2=0.364565 MAE=0.011916. Findings: multi-noise expert is relatively robust to isolated contour noise but more susceptible to bottom noise; mixed model breaks under both isolated noise sources, so neither source alone fully explains its degradation. Noise magnitude comparisons between coordinate groups must account for differing physical scales; bottom +/-0.01 jitter might disproportionately affect some inner_z coordinates and should be checked for physical plausibility.

Next Colab Step 11F isolates identical-seed perturbations into Bottom Y only (inner_y1..9), Bottom Z only (inner_z1..9), full bottom, contours only, combined, and clean. Reuses frozen robust_model and mixed_model; calculates routing-validation R2/MAE, generates a downloadable 300 DPI figure, and compares typical absolute coordinate magnitude with group noise MAE. New skills or routing must not be inferred without replication; this is a single-seed diagnostic on validation data. Remain in current Colab pending diagnosis.


### Step 11F: Bottom-Y versus Bottom-Z ablation and next region-specific AE diagnostic (2026-10-13)

For Part1_E on routing validation, user supplied complete fixed-model noise-group ablation. Multi-noise RF100: clean R2=0.521001 MAE=0.010159; bottom Y-only R2=0.489343 MAE=0.010517; bottom Z-only R2=0.507573 MAE=0.010301; both bottom R2=0.474971 MAE=0.010689; contours only R2=0.515567 MAE=0.010286; combined R2=0.480062 MAE=0.010704. Clean+noisy RF100: clean R2=0.982509 MAE=0.001688; bottom Y-only R2=0.980330 MAE=0.001784; bottom Z-only R2=0.357948 MAE=0.011664; both bottom R2=0.349490 MAE=0.011761; contours only R2=0.308950 MAE=0.012121; combined R2=0.364565 MAE=0.011916. Typical absolute coordinate magnitudes: bottom Y median=1.548405, bottom Z median=2.002772; bottom Y noise MAE=0.004972, bottom Z noise MAE=0.005033. Consequently the earlier speculation that bottom Z typically has tiny magnitude is unsupported. The clean+noisy expert is nearly invariant to Y-only corruption but brittle to Z-only or contour-only noise; the multi-noise expert exhibits modest loss under isolated corruption, with greatest loss from Y-only. Relative sensitivities depend on the fitted model, and one seed is only exploratory.

Next: experiment with existing AE as an optional targeted region correction module: compare raw, AE Z-only, AE contour-only, AE Z+contours, and full AE reconstruction on routing validation using current pre-trained experts. Measure downstream Part1_E R2 and MAE, and report geometric reconstruction MAE per feature group. Frozen models are not trained for hybrid representations, so results assess diagnostic compatibility only; retrain matched experts before making routing decisions. Continue current notebook through this diagnostic, then consider moving skill implementation into a new modular Colab.


### Project pivot: implement the one-tissue Mixture of Experts now (2026-10-13)

User correctly challenged the repeated ablation cycle. Freeze exploratory noise optimization. The existing clean-trained RF, multi-noise RF, and denoising autoencoder are candidate components, but the previous target-specific fixed preprocessing selection is NOT a learned MoE. Next deliverable: implement a real learned input-conditioned gating network combining clean, noise-robust, and optionally AE-preprocessed Part1_E prediction experts. Use train for fitting experts, ae_val for AE stopping, routing_val for gate fitting/selection, and test for final evaluation; avoid using routing_val for both gate fitting and an unbiased gate metric. Include raw/robust/expert-vs-MoE baselines, routing weights, 300-DPI plots. Start with a minimal two-expert gate (clean and robust), add AE third expert only after functioning prototype. Note identifiability: noisy-only input cannot necessarily reveal noise level, so learned gating may fail; report honestly. Move to a clean dedicated MoE Colab with reproducible artifacts and versioned code. Stop new ablations until a working gate exists.


### Final selective denoising experiment (Step 11G); experimental phase closed

User-provided routing-validation Part1_E results (R2, prediction MAE): raw multi-noise 0.460584,0.011028; raw clean+noisy 0.357392,0.012167; AE bottom Y multi-noise 0.344201,0.012173 and mixed 0.297734,0.012692; AE bottom Z multi-noise 0.475118,0.010797 and mixed 0.270747,0.012525; AE contours multi-noise 0.136184,0.013763 and mixed 0.109719,0.013930; AE Z+contours multi-noise 0.125735,0.013825 and mixed 0.087178,0.014096; full AE multi-noise 0.006165,0.014849 and mixed 0.051298,0.014421. Geometry MAE: raw 0.027524, AE bottom Y 0.027854, AE bottom Z 0.026722, AE contours 0.003972, AE Z+contours 0.003170, full AE 0.003501. Only AE bottom Z yields a modest predictive improvement for multi-noise RF; otherwise AE reconstructions harm Part1_E prediction despite substantially improving geometric coordinate error. Frozen prediction models were not retrained on reconstructed representations; interpret as exploratory compatibility diagnostic, not a universal indictment of AE. End ablation cycle now; user explicitly requests tangible learned MoE implementation and no image-only answers. Next notebook: 02_One_Tissue_MoE.ipynb; start with 2 experts clean-trained and noise-robust Part1_E regressors plus input-conditioned learned gate, optional denoising skill later. Keep clean disjoint train/ae_val/routing_val/test partition, plots and comparisons, avoid test tuning.


### Notebook 02 — One-tissue Mixture of Experts initialized (2026-10-13)

User created new Colab `02_One_Tissue_MoE.ipynb`. Initial cell provided to download original `2025_4_14_intermediate.csv` (6510 rows, 99 columns), define 54 geometry features and modulus targets Part1_E, Part3_E, Part11_E, and reproduce Notebook 01 stratified random train/AE-val/routing-val/test splits with seeds 42,43,44, expected sizes 3906/651/651/1302, checking disjoint original indices. This notebook's deliverable is a learned input-dependent gate combining a clean-trained expert and a multi-noise-trained expert for Part1_E, followed by objective model comparisons and extension to remaining targets. Keep geometric preprocessing identical to prior notebook for fair comparison; no extra exploratory noise-optimization detours. The previous test set has already been inspected extensively, so final evaluation here is exploratory until genuinely new simulation cases are available. Await user's Step 1 output before next cell.


### Notebook 02 Step 1 confirmed; Step 2 spline preprocessing

User confirmed the 6510x99 dataset and 54 geometry features; stratified four-way split is disjoint with train=3906, ae_val=651, routing_val=651, test=1302. Next single Colab cell defines periodic cubic-spline contour resampling for innerShape and outerShape, leaving bottom coordinates intact, producing geometry_spline and labels dictionaries for all four splits. Expected arrays are (3906,54), (651,54), (651,54), (1302,54), all finite. Caution: current periodic spline parameterization uniformly resamples original point order and must be checked against Notebook 01 before claiming exact preprocessing equivalence. Next: fit clean and multi-noise Part1_E experts and a learned input-conditioned gate. No further noise ablations.


### Notebook 02 Step 2 validated and Step 3 expert fitting

User confirmed spline preprocessing completed successfully for train 3906x54, ae_val 651x54, routing_val 651x54, and test 1302x54; all finite. Step 3 Colab cell provided to download existing noise_models.py, fit a clean Part1_E RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) on 3906 clean train rows, and fit a robust RandomForestRegressor with same hyperparameters on 4 radial-noise variants of the same train rows (levels 0.0,0.01,0.03,0.05 with bottom_jitter=0.01 and seeds 101..104), 15624 total augmented training rows. Noise level 0 retains bottom jitter, so robust expert has no exactly clean cases. Fit only on train, not routing_val or test. Next cell must train an input-conditioned gate on routing_val without leakage, using the two frozen expert predictions, then evaluate with caution on previously viewed test. Potential limitation: the routing gate should be trained across clean and noisy validation variants rather than on only one condition, and validation predictions across variants of the same underlying samples are correlated; do not split those variants randomly.


### Notebook 02: experts fitted; first learned gating model implemented

User confirmed Step 3 one-tissue Part1_E clean RandomForestRegressor (3906 training rows) and noise-robust RandomForestRegressor (15624 noisy augmented rows, four levels) each contain 100 trained trees. Provided Step 4 Colab cell to train a genuine learned gate: HistGradientBoostingClassifier (max_iter 100, max_leaf_nodes 15, L2 regularization 3) on ae_val geometries under true clean and radial-noise levels 0,0.01,0.03,0.05. Gate input features comprise all 54 input geometry coordinates, frozen clean expert prediction, frozen robust expert prediction, and absolute prediction disagreement. Training labels indicate which expert has lower absolute Part1_E error for each sample. Soft output combines expert predictions using gate's predicted clean-win probability. Evaluate on separate routing_val simulation rows under same five conditions, output R2/MAE for each expert and learned MoE and mean gate weight, save/download 300-DPI plot. No expert retraining and no test data used. Caveats: probability of expert winning is not necessarily optimal regression mixture weight; clean/noisy variants of same underlying sample are correlated within gate training but no source geometry overlaps routing validation. If gate fails, report honestly rather than tuning on test. Note Step 2's spline implementation used equal original point-parameter spacing, not Notebook 01's chord-length parametrization, so old accuracy results cannot be compared directly without aligning preprocessing. Await user's run results.


### Notebook 02 Step 4: first learned gating results (2026-10-13)

Frozen Part1_E experts plus HistGradientBoostingClassifier learned gate evaluated on separate routing_val geometries: Clean: clean expert R2 0.984247 MAE 0.001635, robust expert 0.871188 /0.005138, learned MoE 0.976777 /0.001932, average clean gate weight 0.826966. Noise 0.000 (bottom jitter still on): clean expert R2 -0.505869 MAE 0.017078, robust 0.825644 /0.006084, MoE 0.816074 /0.006155, weight 0.247982. Noise 0.010: clean -0.554882 /0.017278, robust 0.650875 /0.008751, MoE 0.650244 /0.008558, weight 0.271921. Noise 0.030: clean -0.801460 /0.018666, robust 0.381339 /0.011810, MoE 0.392623 /0.011236, weight 0.303522. Noise 0.050: clean -1.250592 /0.021659, robust 0.264724 /0.012823, MoE 0.178852 /0.013168, weight 0.327797. Thus gate meaningfully distinguishes pristine from corrupted geometry but does not consistently outperform best fixed expert. Current gate classifier predicts winner of absolute error and treats winner probability as optimal blend, which is not mathematically guaranteed. Next objective should directly minimize regression error for gate weights while frozen experts stay unchanged, using ae_val gate-fitting and routing_val comparison only; avoid touching test. Existing Notebook 02 spline preprocessing differs from Notebook 01 so cross-notebook performance should not be directly equated.


### Notebook 02 Step 5: regression-optimized learned gate validation

User trained a Keras soft two-expert gate minimizing squared error of combined Part1_E modulus estimates while clean and noise-robust Random Forest experts remained frozen. Evaluation on routing_val (651 unique simulations, five conditions): clean original-classification-gate MAE 0.001932 versus regression-gate MAE 0.001948 (new R2 0.975599); radial 0.000 original 0.006155 versus new 0.006014 (R2 0.826492); radial 0.010 original 0.008558 versus new 0.008274 (R2 0.672926); radial 0.030 original 0.011236 versus new 0.010686 (R2 0.454811); radial 0.050 original 0.013168 versus new 0.011960 (R2 0.324690). Improvement relative to original ~2.3%, 3.3%, 4.9%, and 9.2% as noise increases; slight loss 0.8% on clean input. At 0.05 regression gate also beats standalone robust expert routing-validation MAE 0.012823. A genuine learned input-conditioned MoE now works; cease architectural tuning using routing validation and proceed to frozen evaluation on the previously inspected 1302-case test split, with caveat that this is exploratory because Notebook 01 used the same cases. Save models, scaler, spline implementation and noise model version for reproducibility after evaluation; note spline parameterization in Notebook 02 differs from Notebook 01.


### Notebook 02 Step 6 frozen MoE test and Step 7 artifact export

The frozen two-expert Part1_E MoE was evaluated across five input conditions on the existing 1302-case test split. Clean geometry: clean expert R2 0.981290 MAE 0.001717, robust expert 0.876724/0.004943, learned regression gate MoE 0.974410/0.001957. Noise 0.000 (with bottom-coordinate jitter): clean -0.473785/0.016897, robust 0.827494/0.005999, MoE 0.831651/0.005823. Noise 0.010: clean -0.561871/0.017557, robust 0.612596/0.009057, MoE 0.645296/0.008490. Noise 0.030: clean -0.954360/0.019851, robust 0.338767/0.012171, MoE 0.411878/0.011117. Noise 0.050: clean -1.174222/0.021055, robust 0.228883/0.013137, MoE 0.288745/0.012313. MoE improves relative to standalone robust expert at all noisy conditions, whereas standalone clean expert remains best on perfectly clean geometry. Source cases already examined in Notebook 01; these results remain exploratory, not external holdout. User now moves to engineering rather than further ablations. Step 7 Colab export cell saves joblib clean/robust expert models, Keras regression gate weights (not full model, because the Lambda layer requires controlled architecture reconstruction), sklearn gate scaler, 54-feature configuration and radial noise implementation into one_tissue_moe_v1.zip. Must still package exact Notebook 02 spline preprocessing plus gate architecture and inference entrypoint before claiming loadable standalone deployment. Await export confirmation, then make reproducible loader and push code to GitHub.


### Notebook 02 Step 7 confirmed; Step 8 artifact round-trip verification

User confirmed the export directory `/content/one_tissue_moe_v1` contains `config.json` (1517 bytes), `expert_clean.joblib` (27,994,465 bytes), `expert_robust.joblib` (117,492,769 bytes), `gate_scaler.joblib` (1967 bytes), `noise_models.py` (4700 bytes), and `regression_gate.weights.h5` (48,144 bytes); ZIP at `/content/one_tissue_moe_v1.zip`. The user downloaded it and requested to continue. Next Colab cell verifies that both reloaded Random Forests reproduce original predictions for five routing_val samples, and that gate weights can be loaded into a cloned Keras gate with matching predictions, asserting equality. If cloning the Lambda architecture fails under Keras serialization, rebuild the exact architecture explicitly instead; do not disable safe mode. This is a verification step, not further hyperparameter tuning. On success, implement a standalone inference interface with exact Notebook 02 spline preprocessing and Keras gate construction, and commit code to GitHub. Do not commit the 145MB+ model artifacts directly to standard Git; consider Git LFS or release storage.


### Step 8 verification and Step 9 inference module

User confirmed exported artifact round-trip: clean and robust RF predictions identical after joblib reload; both have 100 trees; reconstructed Keras gate predictions match saved weights. Added `one_tissue_moe.py` with `OneTissueMoE.load(directory)` and `.predict(preprocessed_geometry)`; model expects exact Notebook 02 spline-preprocessed geometry with 54 ordered features. Next run a Colab smoke test comparing module predictions against in-memory regression gate. Important: exact spline preprocessing has NOT yet been packaged; do not claim raw-geometry end-to-end inference until that is done. Joblib files must be trusted and large model binaries should not be committed to normal Git.


### Notebook 02 Step 9 passed; Step 10 spline preprocessing modularized

The user's reloaded `OneTissueMoE` module reproduced original regression gate predictions on ten routing-validation geometries (`np.allclose(...,atol=1e-6)` passed). Both frozen experts and the gate are now verified after saving/reloading. Added `geometry_preprocessing.py` which exactly reproduces Notebook 02 Step 2's uniform-index periodic cubic spline transform on innerShape and outerShape, preserving 18 bottom coordinates and DataFrame row indices. This is intentionally different from Notebook 01's chord-length spline. Next Colab cell downloads this file and confirms preprocessing on raw `splits['routing_val']` equals `geometry_spline['routing_val']` numerically; then checks packaged MoE output from freshly preprocessed raw input equals output using existing preprocessed input. Do not switch notebook until preprocessing verification passes. Module prediction still explicitly accepts preprocessed geometry; the next API enhancement may supply a separate raw-geometry convenience entrypoint.
