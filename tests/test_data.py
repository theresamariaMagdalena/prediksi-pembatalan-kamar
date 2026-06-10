from __future__ import annotations

import pandas as pd
import pytest

from hotel_cancel.data import (
    DataValidationError,
    categorical_options,
    clean,
    validate,
)
from hotel_cancel.features import CATEGORICAL_FEATURES, FEATURES, TARGET


def test_clean_fills_children_and_drops_nulls(synthetic_df):
    assert synthetic_df["children"].isnull().any()
    cleaned = clean(synthetic_df)
    assert not cleaned["children"].isnull().any()
    assert not cleaned[FEATURES + [TARGET]].isnull().any().any()


def test_validate_passes_on_good_schema(synthetic_df):
    validate(synthetic_df)


def test_validate_raises_on_missing_column(synthetic_df):
    broken = synthetic_df.drop(columns=["lead_time"])
    with pytest.raises(DataValidationError):
        validate(broken)


def test_categorical_options_are_sorted_and_complete(synthetic_df):
    opts = categorical_options(synthetic_df)
    assert set(opts) == set(CATEGORICAL_FEATURES)
    for values in opts.values():
        assert values == sorted(values)


def test_clean_does_not_mutate_input(synthetic_df):
    original = synthetic_df.copy()
    clean(synthetic_df)
    pd.testing.assert_frame_equal(synthetic_df, original)
