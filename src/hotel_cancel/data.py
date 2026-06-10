from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .features import CATEGORICAL_FEATURES, FEATURES, TARGET

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    pass


def load_raw(path: str | Path) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> None:
    required = set(FEATURES) | {TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing values in used columns", dropped)
    return df


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = load_raw(path)
    validate(df)
    df = clean(df)
    X = df[FEATURES]
    y = df[TARGET]
    logger.info("Prepared dataset: %d rows, %d features", len(X), X.shape[1])
    return X, y


def categorical_options(X: pd.DataFrame) -> dict[str, list[str]]:
    return {col: sorted(X[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}


def numeric_summary(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    from .features import NUMERIC_FEATURES

    summary: dict[str, dict[str, float]] = {}
    for col in NUMERIC_FEATURES:
        s = X[col]
        summary[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "is_float": bool(s.dtype.kind == "f"),
        }
    return summary
