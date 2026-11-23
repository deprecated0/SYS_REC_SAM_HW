"""Пути к каталогам и загрузка параметров пайплайна."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TABLES_DIR = ARTIFACTS_DIR / "tables"
SPLIT_DIR = ARTIFACTS_DIR / "split"
MODELS_DIR = ARTIFACTS_DIR / "models"


def load_params(path=None):
    """Читает параметры из params.yaml."""
    params_path = Path(path) if path is not None else ROOT / "params.yaml"
    with open(params_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs():
    """Создаёт каталоги для артефактов."""
    for directory in (FIGURES_DIR, TABLES_DIR, SPLIT_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
