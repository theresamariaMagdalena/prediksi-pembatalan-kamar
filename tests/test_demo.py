from __future__ import annotations

from hotel_cancel.data import clean, numeric_summary
from hotel_cancel.demo import generate_random_bookings, score_bookings
from hotel_cancel.features import CATEGORICAL_FEATURES, FEATURES, TARGET
from hotel_cancel.model import build_pipeline


def _metadata(df):
    return {
        "features": FEATURES,
        "categorical_options": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        "numeric_summary": numeric_summary(df),
    }


def test_synthesized_bookings_have_correct_schema(synthetic_df):
    meta = _metadata(synthetic_df)
    bookings = generate_random_bookings(meta, n=12, seed=1)
    assert list(bookings.columns) == FEATURES
    assert len(bookings) == 12
    for col in CATEGORICAL_FEATURES:
        assert set(bookings[col]).issubset(set(meta["categorical_options"][col]))


def test_generation_is_reproducible_with_seed(synthetic_df):
    meta = _metadata(synthetic_df)
    a = generate_random_bookings(meta, n=8, seed=7)
    b = generate_random_bookings(meta, n=8, seed=7)
    assert a.equals(b)


def test_score_bookings_adds_probability_and_label(synthetic_df):
    df = clean(synthetic_df)
    pipeline = build_pipeline({"n_estimators": 25})
    pipeline.fit(df[FEATURES], df[TARGET])

    meta = _metadata(df)
    bookings = generate_random_bookings(meta, n=10, seed=3)
    scored = score_bookings(pipeline, bookings, meta)

    assert len(scored) == 10
    assert {"cancel_probability", "prediction"} <= set(scored.columns)
    assert scored["cancel_probability"].between(0, 1).all()
    assert set(scored["prediction"]).issubset({"Cancel", "Keep"})
