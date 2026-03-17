import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)


def fit_polynomial(df: pd.DataFrame, x_cols: List[str], y_col: str, degree: int = 2):
    X = df[x_cols].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out([f"X{i+1}" for i in range(X.shape[1])]).tolist()

    lr = LinearRegression(fit_intercept=False)
    lr.fit(Phi, y)
    y_pred = lr.predict(Phi)

    report = {
        "y_name": y_col,
        "x_cols": x_cols,
        "degree": int(degree),
        "feature_names": feature_names,
        "coef": lr.coef_.tolist(),
        "r2": float(r2_score(y, y_pred)),
        "mape": float(mape_percent(y, y_pred)),
        "n_rows": int(len(df)),
    }
    return report, y, y_pred


def export_json_bytes(report: dict) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")


def export_py_bytes(report: dict) -> bytes:
    y_name = report["y_name"]
    x_cols = report["x_cols"]
    degree = report["degree"]
    feature_names = report["feature_names"]
    coef = report["coef"]

    py_text = f'''"""
Автосгенерированная модель (полином степени {degree}) для выхода: {y_name}
Входы X (в порядке): {x_cols}
"""

import numpy as np

Y_NAME = {repr(y_name)}
X_COLS = {repr(x_cols)}
DEGREE = {int(degree)}
FEATURE_NAMES = {repr(feature_names)}
COEF = np.array({repr(coef)}, dtype=float)

def _poly_features(X):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    feats = [np.ones((n, 1))]
    feats.append(X)

    if DEGREE >= 2:
        feats.append(X**2)
        cross = []
        for i in range(p):
            for j in range(i+1, p):
                cross.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if cross:
            feats.append(np.hstack(cross))
    return np.hstack(feats)

def predict(X):
    Phi = _poly_features(X)
    return Phi @ COEF
'''
    return py_text.encode("utf-8")
