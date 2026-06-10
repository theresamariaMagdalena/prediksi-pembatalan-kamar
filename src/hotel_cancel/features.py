from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

TARGET = "is_canceled"

NUMERIC_FEATURES: list[str] = [
    "lead_time",
    "adults",
    "children",
    "babies",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "booking_changes",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_FEATURES: list[str] = [
    "hotel",
    "meal",
    "market_segment",
    "deposit_type",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKAGE_COLUMNS: list[str] = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )
