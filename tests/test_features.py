from __future__ import annotations

from hotel_cancel.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
)


def test_feature_lists_are_disjoint_and_combined():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_target_and_leakage_excluded_from_features():
    assert TARGET not in FEATURES
    assert set(LEAKAGE_COLUMNS).isdisjoint(FEATURES)


def test_preprocessor_transforms_categoricals(synthetic_df):
    pre = build_preprocessor()
    transformed = pre.fit_transform(synthetic_df[FEATURES])
    assert transformed.shape[0] == len(synthetic_df)
    assert transformed.shape[1] >= len(FEATURES)


def test_preprocessor_handles_unknown_categories(synthetic_df):
    pre = build_preprocessor()
    pre.fit(synthetic_df[FEATURES])
    row = synthetic_df[FEATURES].iloc[[0]].copy()
    row["market_segment"] = "Aviation"
    pre.transform(row)
