from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def compute_metrics(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "report_text": classification_report(y_test, y_pred, digits=3),
        "report_dict": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def feature_importances(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return pd.DataFrame({"feature": names[order], "importance": importances[order]}).reset_index(
        drop=True
    )


def save_figures(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: str | Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    path = figures_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    path = figures_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    importances = feature_importances(pipeline)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importances["feature"][::-1], importances["importance"][::-1])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    path = figures_dir / "feature_importances.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    for p in saved:
        logger.info("Saved figure %s", p)
    return saved
