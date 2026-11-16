"""Smoke tests so CI has something real to run from day one."""

import random

import numpy as np

from supplyiq import config


def test_root_is_repo_root():
    assert (config.ROOT / "pyproject.toml").exists()


def test_expected_dirs_exist():
    for d in [config.DATA_DIR, config.RAW_DIR, config.PROCESSED_DIR, config.NLP_SPLITS_DIR,
              config.MODELS_DIR, config.RESULTS_DIR, config.CONFIGS_DIR]:
        assert d.is_dir(), f"missing directory: {d}"


def test_set_seed_is_reproducible():
    config.set_seed(123)
    a = (random.random(), np.random.rand())
    config.set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b
