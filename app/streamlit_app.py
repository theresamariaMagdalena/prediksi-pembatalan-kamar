from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hotel_cancel.config import Config
from hotel_cancel.demo import generate_random_bookings, score_bookings
from hotel_cancel.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from hotel_cancel.predict import load_artifacts, predict_proba

NUMERIC_LABELS = {
    "lead_time": "Lead Time (hari)",
    "adults": "Jumlah Dewasa",
    "children": "Jumlah Anak",
    "babies": "Jumlah Bayi",
    "stays_in_weekend_nights": "Malam Weekend",
    "stays_in_week_nights": "Malam Weekday",
    "previous_cancellations": "Riwayat Pembatalan",
    "booking_changes": "Perubahan Booking",
    "adr": "Average Daily Rate",
    "required_car_parking_spaces": "Parkir Mobil",
    "total_of_special_requests": "Permintaan Khusus",
}
CATEGORICAL_LABELS = {
    "hotel": "Tipe Hotel",
    "meal": "Paket Makan",
    "market_segment": "Market Segment",
    "deposit_type": "Tipe Deposit",
}
MIN_VALUES = {"adults": 1}

CANCEL_COLOR = "#E4572E"
KEEP_COLOR = "#1B9E77"

st.set_page_config(page_title="Prediksi Pembatalan Kamar", page_icon="🏨", layout="wide")


@st.cache_resource
def get_model():
    config = Config.load()
    try:
        return load_artifacts(config)
    except FileNotFoundError:
        from hotel_cancel.train import run_training

        with st.spinner("Melatih model untuk pertama kali (±20–40 detik)..."):
            run_training(config, write_figures=False)
        return load_artifacts(config)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.2rem; }
        .hero {
            background: linear-gradient(120deg, #005BAC 0%, #0096C7 100%);
            padding: 1.6rem 2rem; border-radius: 16px; color: #fff;
            box-shadow: 0 6px 18px rgba(0,91,172,0.25); margin-bottom: 1.4rem;
        }
        .hero h1 { color: #fff; margin: 0; font-size: 2rem; }
        .hero p  { color: #eaf6ff; margin: .35rem 0 0; font-size: 1rem; }
        div[data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e8edf3; border-radius: 14px;
            padding: 1rem 1.1rem; box-shadow: 0 2px 8px rgba(16,42,67,0.05);
        }
        div[data-testid="stMetric"] * { color: #0d2233 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] * { color: #5b6b7b !important; }
        .stButton > button, .stDownloadButton > button {
            background: #005BAC; color: #fff; border: none; height: 48px;
            width: 100%; font-size: 1rem; font-weight: 600; border-radius: 10px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #00427D; color: #fff;
        }
        .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🏨 Prediksi Pembatalan Kamar Hotel</h1>
            <p>Model Machine Learning untuk memperkirakan kemungkinan customer
            membatalkan booking kamar hotel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(proba: float) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Probabilitas Pembatalan", f"{proba:.1%}")
    with c2:
        st.progress(min(max(proba, 0.0), 1.0))
        if proba >= 0.5:
            st.error("❌ Customer **berpotensi membatalkan** kamar.")
        else:
            st.success("✅ Customer **tidak membatalkan** kamar.")


def manual_tab(pipeline, metadata) -> None:
    options = metadata["categorical_options"]
    summary = metadata.get("numeric_summary", {})
    inputs: dict[str, object] = {}

    st.subheader("Detail Booking")
    cols = st.columns(3)
    for i, feature in enumerate(NUMERIC_FEATURES):
        with cols[i % 3]:
            is_float = summary.get(feature, {}).get("is_float", feature == "adr")
            default = MIN_VALUES.get(feature, 0)
            inputs[feature] = st.number_input(
                NUMERIC_LABELS.get(feature, feature),
                min_value=(
                    float(MIN_VALUES.get(feature, 0)) if is_float else MIN_VALUES.get(feature, 0)
                ),
                value=float(default) if is_float else default,
                step=1.0 if is_float else 1,
            )

    st.subheader("Kategori")
    cat_cols = st.columns(len(CATEGORICAL_FEATURES))
    for col, feature in zip(cat_cols, CATEGORICAL_FEATURES, strict=True):
        with col:
            label = CATEGORICAL_LABELS.get(feature, feature)
            inputs[feature] = st.selectbox(label, options[feature])

    if st.button("🔮 Prediksi", key="manual_predict"):
        result_card(predict_proba(inputs, pipeline, metadata))


def demo_tab(pipeline, metadata) -> None:
    st.subheader("Demo Acak — untuk Presentasi")
    st.caption(
        "Hasilkan beberapa booking acak (diambil dari data nyata), lalu lihat "
        "prediksi pembatalannya dalam bentuk grafik."
    )

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        n = st.slider("Jumlah booking", min_value=5, max_value=25, value=10, step=1)
    with c2:
        st.write("")
        st.write("")
        generate = st.button("🎲 Generate Data Acak", key="demo_generate")

    if generate:
        config = Config.load()
        source = config.raw_data_path if config.raw_data_path.exists() else None
        bookings = generate_random_bookings(metadata, n=n, source_path=source)
        scored = score_bookings(pipeline, bookings, metadata)
        scored.insert(0, "Booking", [f"#{i + 1}" for i in range(len(scored))])
        st.session_state["demo_scored"] = scored

    scored = st.session_state.get("demo_scored")
    if scored is None:
        st.info("Klik **Generate Data Acak** untuk memulai.")
        return

    n_cancel = int((scored["prediction"] == "Cancel").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Booking", len(scored))
    m2.metric("Diprediksi Batal", n_cancel)
    m3.metric("Rata-rata Probabilitas", f"{scored['cancel_probability'].mean():.1%}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Probabilitas Pembatalan per Booking**")
        bar = (
            alt.Chart(scored)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("cancel_probability:Q", title="P(batal)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Booking:N", sort="-x", title=None),
                color=alt.Color(
                    "prediction:N",
                    scale=alt.Scale(domain=["Cancel", "Keep"], range=[CANCEL_COLOR, KEEP_COLOR]),
                    legend=alt.Legend(title="Prediksi"),
                ),
                tooltip=[
                    "Booking:N",
                    alt.Tooltip("cancel_probability:Q", format=".1%", title="P(batal)"),
                    "prediction:N",
                    "hotel:N",
                    "market_segment:N",
                    "lead_time:Q",
                ],
            )
            .properties(height=max(220, 24 * len(scored)))
        )
        st.altair_chart(bar, use_container_width=True)

    with right:
        st.markdown("**Distribusi Prediksi**")
        donut_df = (
            scored["prediction"].value_counts().rename_axis("prediction").reset_index(name="count")
        )
        donut = (
            alt.Chart(donut_df)
            .mark_arc(innerRadius=60)
            .encode(
                theta="count:Q",
                color=alt.Color(
                    "prediction:N",
                    scale=alt.Scale(domain=["Cancel", "Keep"], range=[CANCEL_COLOR, KEEP_COLOR]),
                    legend=alt.Legend(title="Prediksi"),
                ),
                tooltip=["prediction:N", "count:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(donut, use_container_width=True)

    st.markdown("**Detail Data**")
    display = scored.copy()
    display["cancel_probability"] = (display["cancel_probability"] * 100).round(1)
    st.dataframe(
        display.rename(columns={"cancel_probability": "P(batal) %"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Unduh CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="demo_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    inject_css()
    hero()
    try:
        pipeline, metadata = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("ℹ️ Tentang Model")
        auc = metadata.get("metrics", {}).get("roc_auc")
        if auc is not None:
            st.metric("ROC-AUC (test)", f"{auc:.3f}")
        st.write(
            f"**Fitur:** {len(metadata['features'])} "
            f"({len(NUMERIC_FEATURES)} numerik · {len(CATEGORICAL_FEATURES)} kategori)"
        )
        st.write(f"**Versi:** {metadata.get('version', 'n/a')}")
        st.caption("RandomForest dalam satu Pipeline (tanpa train/serve skew).")

    tab1, tab2 = st.tabs(["🔮 Prediksi Manual", "🎲 Demo Acak"])
    with tab1:
        manual_tab(pipeline, metadata)
    with tab2:
        demo_tab(pipeline, metadata)


main()
