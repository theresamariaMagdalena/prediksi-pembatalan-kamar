from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import Config

logger = logging.getLogger(__name__)


def load_artifacts(config: Config | None = None) -> tuple[Pipeline, dict[str, Any]]:
    config = config or Config.load()
    if not config.pipeline_path.exists() or not config.metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {config.model_dir}. "
            "Run `python -m hotel_cancel.train` first."
        )
    pipeline = joblib.load(config.pipeline_path)
    metadata = joblib.load(config.metadata_path)
    return pipeline, metadata


@lru_cache(maxsize=1)
def _cached_artifacts() -> tuple[Pipeline, dict[str, Any]]:
    return load_artifacts()


def to_frame(inputs: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([inputs], columns=metadata["features"])


def predict_proba(
    inputs: dict[str, Any],
    pipeline: Pipeline | None = None,
    metadata: dict[str, Any] | None = None,
) -> float:
    if pipeline is None or metadata is None:
        pipeline, metadata = _cached_artifacts()
    frame = to_frame(inputs, metadata)
    return float(pipeline.predict_proba(frame)[0, 1])
