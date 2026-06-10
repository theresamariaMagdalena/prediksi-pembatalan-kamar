# 🏨 Prediksi Pembatalan Kamar Hotel

[![CI](https://github.com/theresamariaMagdalena/prediksi-pembatalan-kamar/actions/workflows/ci.yml/badge.svg)](https://github.com/theresamariaMagdalena/prediksi-pembatalan-kamar/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theresamariaMagdalena/prediksi-pembatalan-kamar/blob/main/notebooks/colab_quickstart.ipynb)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

A leakage-free machine-learning project that predicts whether a hotel booking
will be **cancelled**, served through a Streamlit app. Built on the
[hotel bookings dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191)
(`is_canceled` target, ~119k rows).

The whole project trains a **single scikit-learn `Pipeline`** (preprocessing +
model in one fitted object), so the exact transformations used in training are
reused at prediction time — there is no train/serve skew.

---

## Project structure

```
prediksi-pembatalan-kamar/
├── app/
│   └── streamlit_app.py        # UI; dropdowns + column order come from model metadata
├── config/
│   └── config.yaml             # paths + hyperparameters
├── data/
│   ├── raw/hotel_booking.csv   # source dataset (tracked for reproducibility)
│   └── processed/              # derived data (gitignored)
├── models/                     # pipeline.pkl + metadata.pkl (generated, gitignored)
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis
│   └── colab_quickstart.ipynb  # one-click Colab run
├── reports/figures/            # confusion matrix, ROC, importances (generated)
├── src/hotel_cancel/
│   ├── config.py               # path/param resolution (root-aware, Colab-safe)
│   ├── data.py                 # load, clean, validate
│   ├── features.py             # feature lists + ColumnTransformer (source of truth)
│   ├── model.py                # build_pipeline()
│   ├── train.py                # CLI training entrypoint
│   ├── evaluate.py             # metrics + diagnostic figures
│   └── predict.py              # load artifacts + score a booking
├── tests/                      # pytest suite (synthetic data, fast)
├── pyproject.toml              # packaging + ruff/black/pytest config
├── Makefile / tasks.ps1        # task runners (Unix / Windows)
└── requirements.txt
```

---

## Quickstart (local)

```bash
# 1. Install (editable) — use a virtualenv if you like
pip install -e .

# 2. Train the model -> writes models/pipeline.pkl + models/metadata.pkl
python -m hotel_cancel.train          # or: make train

# 3. Launch the app
streamlit run app/streamlit_app.py    # or: make app
```

The app opens at <http://localhost:8501>. Fill in the booking details, click
**Prediksi**, and it shows the cancellation **probability**.

On Windows you can use the PowerShell task runner instead of `make`:

```powershell
.\tasks.ps1 install
.\tasks.ps1 train
.\tasks.ps1 app
```

---

## Run in Google Colab

Click the **Open in Colab** badge above (or open
`notebooks/colab_quickstart.ipynb` in Colab) and choose **Runtime → Run all**.
The notebook clones the repo, installs the package, trains the model, scores a
sample booking, and can expose the Streamlit app through a public tunnel.

---

## Configuration

All tunable settings live in [`config/config.yaml`](config/config.yaml) — data
path, train/test split, and RandomForest hyperparameters. Paths are resolved
relative to the project root automatically, so commands work from any working
directory (and inside Colab). Override the root with the `HOTEL_CANCEL_ROOT`
environment variable if needed.

---

## Modelling notes

- **Features.** 11 numeric + 4 categorical columns
  (`hotel`, `meal`, `market_segment`, `deposit_type`). Categoricals are
  one-hot encoded with `handle_unknown="ignore"`; numerics pass through
  unscaled (correct for tree ensembles).
- **No leakage.** `reservation_status`, `reservation_status_date`, and
  `assigned_room_type` encode the outcome and are deliberately excluded.
- **Model.** `RandomForestClassifier(n_estimators=300, class_weight="balanced")`
  inside a `Pipeline`, trained with a stratified 80/20 split.
- **Metadata.** Training also saves `metadata.pkl` with the feature list and the
  categorical options observed in the data — the app builds its inputs from
  this, so the UI can never offer a value the model never saw.
- **Reported metrics** (held-out test set): ROC-AUC ≈ 0.91, accuracy ≈ 0.84.
  Re-run `python -m hotel_cancel.train` to reproduce; figures land in
  `reports/figures/`.

---

## Development

```bash
pip install -e ".[dev]"   # adds pytest, ruff, black, pre-commit
pre-commit install        # optional: lint/format on commit

make test                 # run the test suite      (.\tasks.ps1 test)
make lint                 # ruff                     (.\tasks.ps1 lint)
make format               # black + ruff --fix       (.\tasks.ps1 format)
```

CI (GitHub Actions) runs lint, format-check, and tests on Python 3.10–3.12.

---

## License

[MIT](LICENSE).
