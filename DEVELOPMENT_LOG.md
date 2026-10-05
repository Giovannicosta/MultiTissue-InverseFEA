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

Expected output:

```text
Train spline base: (5208, 54)
Test spline base: (1302, 54)
```

Next step: define the fast noise function so noise can be added without recomputing splines.
