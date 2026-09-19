"""Modelo de satisfacción entrenable (sklearn) + calibración con satisfacción real.

Se entrena con features derivadas de las emociones/prosodia/texto de cada llamada (ver features.call_features)
y una etiqueta 0-100 (etiquetado manual o CSAT/encuesta). Es independiente del modelo emocional.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class SatisfactionRegressor:
    def __init__(self, pipeline=None, meta: dict | None = None):
        self.pipeline = pipeline
        self.meta = meta or {}

    def predict(self, features: np.ndarray) -> float:
        return float(self.pipeline.predict(np.asarray(features, dtype=float).reshape(1, -1))[0])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "meta": self.meta}, path)

    @staticmethod
    def load(path: str | Path) -> "SatisfactionRegressor":
        d = joblib.load(path)
        return SatisfactionRegressor(d["pipeline"], d.get("meta"))


def make_pipeline(kind: str = "ridge"):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline as mk
    from sklearn.preprocessing import StandardScaler
    if kind == "gbr":
        return mk(StandardScaler(), GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0))
    return mk(StandardScaler(), Ridge(alpha=3.0))


def train_regressor(X: np.ndarray, y: np.ndarray, groups: list | None = None, kind: str = "ridge",
                    folds: int = 5) -> tuple[SatisfactionRegressor, dict]:
    """Entrena y evalúa por validación cruzada agrupada (sin fuga por llamada/persona)."""
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    if len(y) < 10:
        raise ValueError("Se necesitan al menos 10 muestras etiquetadas para entrenar el modelo de satisfacción.")
    pipe = make_pipeline(kind)
    if groups is not None and len(set(groups)) >= 3:
        cv = GroupKFold(n_splits=min(folds, len(set(groups))))
        pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
    else:
        pred = cross_val_predict(pipe, X, y, cv=KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=0))
    err = pred - y
    metrics = {
        "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)),
        "pearson": float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else 0.0,
        "n": int(len(y)), "cv": "grouped" if groups is not None else "kfold",
    }
    pipe.fit(X, y)
    return SatisfactionRegressor(pipe, {"kind": kind, "metrics": metrics, "dim": int(X.shape[1])}), metrics


def fit_linear_calibration(predicted: list[float], real: list[float]) -> dict:
    """Calibración lineal real ≈ a·pred + b (mínimos cuadrados). Devuelve coeficientes y error antes/después."""
    p, r = np.asarray(predicted, float), np.asarray(real, float)
    if len(p) < 3 or np.std(p) < 1e-6:
        return {"a": 1.0, "b": 0.0, "n": int(len(p)), "mae_before": None, "mae_after": None}
    a, b = np.polyfit(p, r, 1)
    a = float(np.clip(a, 0.2, 3.0))
    b = float(r.mean() - a * p.mean())
    after = np.clip(a * p + b, 0, 100)
    return {"a": a, "b": b, "n": int(len(p)), "mae_before": float(np.mean(np.abs(p - r))),
            "mae_after": float(np.mean(np.abs(after - r)))}
