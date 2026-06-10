from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import build_preprocessor

DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def build_pipeline(model_params: dict[str, Any] | None = None) -> Pipeline:
    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(**params)),
        ]
    )
