"""Single source of truth for paths and the global random seed.

Every script imports from here so all three members read/write the same locations
and use the same seed. Change things here, not by hardcoding paths in scripts.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# Repo root = two levels above this file (src/supplyiq/config.py -> repo/)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
M5_RAW_DIR = RAW_DIR / "m5"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"  # small, committed lookup tables
NLP_DIR = DATA_DIR / "nlp"
NLP_SPLITS_DIR = NLP_DIR / "splits"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CONFIGS_DIR = ROOT / "configs"

SEED = 42

# The M5 files we expect after download + unzip
M5_FILES = [
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_evaluation.csv",  # d_1..d_1941 (use this one, it has the most history)
]


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
