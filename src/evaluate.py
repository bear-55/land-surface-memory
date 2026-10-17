"""Per-basin metrics and validation predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from project import ROOT, save_json


def basin_metrics(observed, predicted, *, min_days=365, variance_floor=1e-12, quantile=0.95):
    observed, predicted = np.asarray(observed, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if observed.ndim != 1 or observed.shape != predicted.shape or len(observed) == 0:
        raise ValueError("Metric inputs must be nonempty paired 1-D arrays")
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Observations must already have the common valid-date mask")
    if not np.isfinite(predicted).all():
        raise FloatingPointError("Nonfinite prediction; changing the evaluation mask is forbidden")
    if (predicted < 0).any():
        raise ValueError("Softplus runoff cannot be negative")
    residual = predicted - observed
    variance = float(np.var(observed))
    reason = "too_few_days" if len(observed) < min_days else "near_zero_variance" if variance <= variance_floor else None
    threshold = float(np.quantile(observed, quantile))
    high = observed >= threshold
    return {"n_days": len(observed), "nse": float(1 - np.mean(residual ** 2) / variance) if reason is None else None,
            "nse_invalid_reason": reason, "mae_mm_day": float(np.mean(np.abs(residual))),
            "rmse_mm_day": float(np.sqrt(np.mean(residual ** 2))),
            "volume_bias_percent": float(100 * residual.sum() / observed.sum()) if observed.sum() > 0 else None,
            "q95_observed_mm_day": threshold, "high_flow_days": int(high.sum()),
            "high_flow_rmse_mm_day": float(np.sqrt(np.mean(residual[high] ** 2)))}


def summarize(rows):
    valid = [row["nse"] for row in rows if row["nse"] is not None]
    return {"basins": len(rows), "valid_nse_basins": len(valid), "invalid_nse_basins": len(rows) - len(valid),
            "median_nse": float(np.median(valid)) if valid else None,
            "mean_nse": float(np.mean(valid)) if valid else None,
            "negative_nse_basins": sum(value < 0 for value in valid),
            "negative_nse_fraction": float(np.mean(np.array(valid) < 0)) if valid else None}


@torch.no_grad()
def predict_validation(model, dataset, batches, cfg, q_ref):
    from training import predict_scaled
    if dataset.role != "val":
        raise PermissionError("Only development validation is supported")
    model.eval()
    prediction = np.empty(len(dataset), dtype=np.float32)
    size = cfg["training"]["batch_size"]
    for start in range(0, len(dataset), size):
        indices = np.arange(start, min(start + size, len(dataset)))
        batch = batches(indices)
        values = predict_scaled(model, batch) * q_ref
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite validation prediction; run failed")
        prediction[indices] = values[:, 0].cpu().numpy()
    observed = dataset.q[batches.basins, batches.ends]
    rows = []
    for j, basin_id in enumerate(dataset.ids):
        mask = batches.basins == j
        rows.append({"basin_id": basin_id, **basin_metrics(observed[mask], prediction[mask],
                     min_days=cfg["evaluation"]["min_valid_days"], quantile=cfg["evaluation"]["high_flow_quantile"])})
    records = {"basin_ids": np.asarray(dataset.ids)[batches.basins], "dates": dataset.dates[batches.ends],
               "q_observed_mm_day": observed, "q_predicted_mm_day": prediction}
    return rows, summarize(rows), records


def write_csv(path, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_validation(directory, rows, summary, records):
    write_csv(directory / "metrics_per_basin.csv", rows)
    np.savez_compressed(directory / "predictions_val.npz", **records)
    save_json(directory / "validation.json", summary)

