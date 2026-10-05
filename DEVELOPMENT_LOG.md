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

Next step: compare the 9-feature results against the 4-feature baseline, especially for Part1_E.
