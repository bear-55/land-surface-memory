"""Train-only input statistics; static missing flags remain unscaled 0/1."""
from __future__ import annotations

import numpy as np


def transform_raw_static(values, names, log1p_fields):
    result = np.asarray(values, dtype=np.float64).copy()
    if np.isinf(result).any():
        raise ValueError("Infinite static input")
    for field in log1p_fields:
        idx = names.index(field)
        if np.any(result[:, idx] < 0):
            raise ValueError(f"Cannot log1p negative {field}")
        result[:, idx] = np.log1p(result[:, idx])
    return result


def fit_static(values, names, train_rows, indicator_fields, log1p_fields):
    values = transform_raw_static(values, names, log1p_fields)
    unknown_missing = [name for j, name in enumerate(names) if np.isnan(values[:, j]).any() and name not in indicator_fields]
    if unknown_missing:
        raise ValueError(f"Unexpected missing fields need protocol review: {unknown_missing}")
    train = values[np.asarray(train_rows)]
    median = np.nanmedian(train, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("A static column has no finite training values")
    train_filled = np.where(np.isnan(train), median, train)
    mean = train_filled.mean(axis=0)
    std = train_filled.std(axis=0, ddof=0)
    scale = np.where(std < 1e-12, 1.0, std)
    return {"names": list(names), "median": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "constant_columns": [name for name, s in zip(names, std) if s < 1e-12],
            "indicator_fields": list(indicator_fields), "log1p_fields": list(log1p_fields)}


def apply_static(values, scaler):
    values = transform_raw_static(values, scaler["names"], scaler["log1p_fields"])
    indicators = np.column_stack([np.isnan(values[:, scaler["names"].index(name)]) for name in scaler["indicator_fields"]]) if scaler["indicator_fields"] else np.empty((len(values), 0))
    filled = np.where(np.isnan(values), scaler["median"], values)
    scaled = (filled - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    result = np.concatenate([scaled, indicators], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite transformed static inputs")
    return result


def fit_dynamic(values, valid, dates, train_rows, start, end):
    period = (dates >= np.datetime64(str(start))) & (dates <= np.datetime64(str(end)))
    train = values[np.asarray(train_rows)][:, period]
    mask = valid[np.asarray(train_rows)][:, period]
    selected = train[mask]
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("No finite source-training forcing to fit")
    std = selected.std(axis=0, ddof=0)
    return {"mean": selected.mean(axis=0).tolist(), "scale": np.where(std < 1e-12, 1, std).tolist(),
            "n_training_days": len(selected), "period": [str(start), str(end)], "ddof": 0}


def apply_dynamic(values, valid, scaler):
    result = ((values - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])).astype(np.float32)
    result[~valid] = np.nan
    return result
