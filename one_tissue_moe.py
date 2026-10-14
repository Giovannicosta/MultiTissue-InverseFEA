"""Inference interface for the frozen, one-tissue Part1_E MoE.

Input must already have undergone the exact spline preprocessing used in
Notebook 02. This module does NOT perform spline preprocessing.
Only load joblib artifacts from a trusted source.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


def build_gate(n_features):
    """Reconstruct Step 5 architecture for loading saved Keras weights."""
    feature_input = tf.keras.Input(shape=(n_features,))
    prediction_input = tf.keras.Input(shape=(2,))
    h = tf.keras.layers.Dense(24, activation="relu")(feature_input)
    h = tf.keras.layers.Dense(12, activation="relu")(h)
    weight = tf.keras.layers.Dense(1, activation="sigmoid")(h)
    output = tf.keras.layers.Lambda(
        lambda z: z[0] * z[1][:, 0:1] + (1 - z[0]) * z[1][:, 1:2]
    )([weight, prediction_input])
    return tf.keras.Model([feature_input, prediction_input], output)


class OneTissueMoE:
    def __init__(self, clean, robust, gate, scaler, feature_columns):
        self.clean = clean
        self.robust = robust
        self.gate = gate
        self.scaler = scaler
        self.feature_columns = list(feature_columns)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        columns = config["geometry_columns"]
        clean = joblib.load(directory / "expert_clean.joblib")
        robust = joblib.load(directory / "expert_robust.joblib")
        scaler = joblib.load(directory / "gate_scaler.joblib")
        gate = build_gate(config["gate_input_features"])
        gate.load_weights(str(directory / "regression_gate.weights.h5"))
        if len(columns) != 54 or scaler.n_features_in_ != len(columns) + 3:
            raise ValueError("Unexpected geometry or gate feature dimensions")
        return cls(clean, robust, gate, scaler, columns)

    def predict(self, geometry):
        """Predict Part1_E from preprocessed geometry (DataFrame or ndarray).

        DataFrames must contain all configured features; ndarray columns must
        be in exactly the same order as config['geometry_columns'].
        """
        if isinstance(geometry, pd.DataFrame):
            missing = set(self.feature_columns) - set(geometry.columns)
            if missing:
                raise ValueError(f"Missing geometry columns: {sorted(missing)}")
            X = geometry.loc[:, self.feature_columns]
        else:
            arr = np.asarray(geometry, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != len(self.feature_columns):
                raise ValueError("Expected (n_samples, 54) geometry")
            X = pd.DataFrame(arr, columns=self.feature_columns)
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Geometry contains non-finite coordinates")
        a = self.clean.predict(X)
        b = self.robust.predict(X)
        features = np.column_stack([values, a, b, np.abs(a - b)])
        scaled = self.scaler.transform(features).astype("float32")
        expert_preds = np.column_stack([a, b]).astype("float32")
        return self.gate.predict([scaled, expert_preds], verbose=0).ravel()
