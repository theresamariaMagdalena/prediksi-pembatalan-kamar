from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ROOT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start: Path | None = None) -> Path:
    env_root = os.environ.get("HOTEL_CANCEL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = (start or Path(__file__)).resolve()
    for parent in (here, *here.parents):
        if any((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
    return here.parents[2]


PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:

    raw_data_path: Path
    target: str
    test_size: float
    random_state: int
    model_params: dict[str, Any]
    model_dir: Path
    pipeline_path: Path
    metadata_path: Path
    reports_dir: Path
    figures_dir: Path
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(config_path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        data = raw.get("data", {})
        split = raw.get("split", {})
        model = raw.get("model", {})
        artifacts = raw.get("artifacts", {})

        model_dir = _abs(artifacts.get("model_dir", "models"))
        reports_dir = _abs(artifacts.get("reports_dir", "reports"))
        figures_dir = _abs(artifacts.get("figures_dir", "reports/figures"))

        return cls(
            raw_data_path=_abs(data.get("raw_path", "data/raw/hotel_booking.csv")),
            target=data.get("target", "is_canceled"),
            test_size=float(split.get("test_size", 0.2)),
            random_state=int(split.get("random_state", 42)),
            model_params=dict(model),
            model_dir=model_dir,
            pipeline_path=model_dir / artifacts.get("pipeline_file", "pipeline.pkl"),
            metadata_path=model_dir / artifacts.get("metadata_file", "metadata.pkl"),
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            extra={
                k: v for k, v in raw.items() if k not in {"data", "split", "model", "artifacts"}
            },
        )

    def ensure_dirs(self) -> None:
        for directory in (self.model_dir, self.reports_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _abs(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
