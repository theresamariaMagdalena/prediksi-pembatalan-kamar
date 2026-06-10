from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import clean, load_raw
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)


def _synthesize(metadata: dict[str, Any], n: int, rng: np.random.Generator) -> pd.DataFrame:
    options = metadata["categorical_options"]
    summary = metadata.get("numeric_summary", {})
    cols: dict[str, Any] = {}
    for col in NUMERIC_FEATURES:
        s = summary.get(col, {"min": 0, "max": 1, "is_float": col == "adr"})
        lo, hi = float(s["min"]), float(s["max"])
        if s.get("is_float"):
            cols[col] = rng.uniform(lo, hi, n).round(2)
        else:
            cols[col] = rng.integers(int(lo), int(hi) + 1, n)
    for col in CATEGORICAL_FEATURES:
        cols[col] = rng.choice(options[col], n)
    return pd.DataFrame(cols)[FEATURES]


def generate_random_bookings(
    metadata: dict[str, Any],
    n: int = 10,
    seed: int | None = None,
    source_path: str | Path | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if source_path is not None and Path(source_path).exists():
        df = clean(load_raw(source_path))
        size = min(n, len(df))
        idx = rng.choice(len(df), size=size, replace=False)
        logger.info("Sampled %d real rows for the demo", size)
        return df[FEATURES].iloc[idx].reset_index(drop=True)
    logger.info("Synthesised %d demo rows from metadata", n)
    return _synthesize(metadata, n, rng)


def score_bookings(pipeline, bookings: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    X = bookings[metadata["features"]]
    proba = pipeline.predict_proba(X)[:, 1]
    out = bookings.copy().reset_index(drop=True)
    out.insert(0, "cancel_probability", proba)
    out.insert(1, "prediction", np.where(proba >= 0.5, "Cancel", "Keep"))
    return out
