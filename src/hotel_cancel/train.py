from __future__ import annotations

import argparse
import logging

import joblib
from sklearn.model_selection import train_test_split

from . import __version__
from .config import Config
from .data import categorical_options, load_dataset, numeric_summary
from .evaluate import compute_metrics, save_figures
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .model import build_pipeline

logger = logging.getLogger(__name__)


def run_training(config: Config, data_path: str | None = None, write_figures: bool = True):
    config.ensure_dirs()
    path = data_path or config.raw_data_path
    logger.info("hotel_cancel v%s — training from %s", __version__, path)

    X, y = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    logger.info("Train/test split: %d / %d rows", len(X_train), len(X_test))

    pipeline = build_pipeline(config.model_params)
    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    metrics = compute_metrics(pipeline, X_test, y_test)
    if write_figures:
        try:
            save_figures(pipeline, X_test, y_test, config.figures_dir)
        except Exception as exc:
            logger.warning("Skipped figures (%s)", exc)

    metadata = {
        "version": __version__,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_options": categorical_options(X),
        "numeric_summary": numeric_summary(X),
        "target": config.target,
        "metrics": {"roc_auc": metrics["roc_auc"], "report": metrics["report_dict"]},
    }

    joblib.dump(pipeline, config.pipeline_path, compress=3)
    joblib.dump(metadata, config.metadata_path, compress=3)
    logger.info("Saved %s and %s", config.pipeline_path, config.metadata_path)
    return pipeline, metadata, metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hotel cancellation model.")
    parser.add_argument("--data", default=None, help="Path to the bookings CSV (overrides config).")
    parser.add_argument("--config", default=None, help="Path to a config.yaml (optional).")
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip writing diagnostic figures."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    config = Config.load(args.config)
    _, _, metrics = run_training(config, args.data, write_figures=not args.no_figures)
    print(metrics["report_text"])
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
