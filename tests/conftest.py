from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotel_cancel.features import FEATURES, TARGET


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    lead_time = rng.integers(0, 400, n)
    data = {
        "lead_time": lead_time,
        "adults": rng.integers(1, 4, n),
        "children": rng.choice([0.0, 1.0, 2.0, np.nan], n, p=[0.7, 0.15, 0.1, 0.05]),
        "babies": rng.integers(0, 2, n),
        "stays_in_weekend_nights": rng.integers(0, 4, n),
        "stays_in_week_nights": rng.integers(0, 6, n),
        "previous_cancellations": rng.integers(0, 3, n),
        "booking_changes": rng.integers(0, 3, n),
        "adr": rng.uniform(40, 300, n).round(2),
        "required_car_parking_spaces": rng.integers(0, 2, n),
        "total_of_special_requests": rng.integers(0, 3, n),
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
        "meal": rng.choice(["BB", "HB", "FB", "SC", "Undefined"], n),
        "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Direct", "Groups"], n),
        "deposit_type": rng.choice(["No Deposit", "Non Refund", "Refundable"], n),
    }
    df = pd.DataFrame(data)
    score = (df["lead_time"] / 400) + df["previous_cancellations"] * 0.3
    df[TARGET] = (score + rng.normal(0, 0.2, n) > 0.6).astype(int)
    assert set(FEATURES) <= set(df.columns)
    return df


@pytest.fixture
def sample_booking() -> dict:
    return {
        "lead_time": 120,
        "adults": 2,
        "children": 1,
        "babies": 0,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 3,
        "previous_cancellations": 1,
        "booking_changes": 0,
        "adr": 95.5,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
        "hotel": "City Hotel",
        "meal": "BB",
        "market_segment": "Online TA",
        "deposit_type": "No Deposit",
    }
