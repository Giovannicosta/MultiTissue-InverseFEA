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

Next step: use this asymmetric sweep to see whether Part1_E benefits from preserving more information in one geometry region than the others at noise = 0.05.
