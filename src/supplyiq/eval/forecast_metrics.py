"""Shared forecasting metrics (CONTRIBUTING rule 3: everyone reports numbers with these functions)."""
from __future__ import annotations

import numpy as np


def _prepare(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty input")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("inputs contain NaN or inf")
    return yt, yp


def mae(y_true, y_pred) -> float:
    yt, yp = _prepare(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _prepare(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error = sum|error| / sum|actual|. NaN if all actuals are zero."""
    yt, yp = _prepare(y_true, y_pred)
    denom = float(np.abs(yt).sum())
    if denom == 0:
        return float("nan")
    return float(np.abs(yt - yp).sum() / denom)


def evaluate(y_true, y_pred) -> dict[str, float]:
    yt, _ = _prepare(y_true, y_pred)
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "wape": wape(y_true, y_pred), "n": int(yt.size)}
