from __future__ import annotations

from hotel_cancel.data import clean
from hotel_cancel.features import FEATURES, TARGET
from hotel_cancel.model import build_pipeline
from hotel_cancel.predict import predict_proba, to_frame


def _fit(df):
    df = clean(df)
    pipeline = build_pipeline({"n_estimators": 25, "n_jobs": 1})
    pipeline.fit(df[FEATURES], df[TARGET])
    return pipeline


def test_pipeline_fits_and_predicts_probability(synthetic_df):
    pipeline = _fit(synthetic_df)
    proba = pipeline.predict_proba(synthetic_df[FEATURES])[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_single_row_prediction_has_no_shape_mismatch(synthetic_df, sample_booking):
    pipeline = _fit(synthetic_df)
    metadata = {"features": FEATURES}
    proba = predict_proba(sample_booking, pipeline, metadata)
    assert 0.0 <= proba <= 1.0


def test_to_frame_preserves_training_column_order(sample_booking):
    metadata = {"features": FEATURES}
    frame = to_frame(sample_booking, metadata)
    assert list(frame.columns) == FEATURES
    assert len(frame) == 1


def test_prediction_is_deterministic(synthetic_df, sample_booking):
    pipeline = _fit(synthetic_df)
    metadata = {"features": FEATURES}
    first = predict_proba(sample_booking, pipeline, metadata)
    second = predict_proba(sample_booking, pipeline, metadata)
    assert first == second


def test_unknown_category_does_not_crash_prediction(synthetic_df, sample_booking):
    pipeline = _fit(synthetic_df)
    metadata = {"features": FEATURES}
    booking = dict(sample_booking, market_segment="Complementary")
    proba = predict_proba(booking, pipeline, metadata)
    assert 0.0 <= proba <= 1.0
