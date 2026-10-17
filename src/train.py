"""Validation-selected training with epoch-atomic resume."""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from evaluate import predict_validation, save_validation, write_csv
from model import parameter_count
from project import ROOT, save_json
from training import (load_protocol, setup_device, environment, provenance, make_model, datasets,
                      optimizer_for, train_step, synchronize, sampling_hash, rng_state, restore_rng,
                      cpu_weights, atomic_checkpoint)


def identity_for(name, seed, hashes, smoke, backend):
    spec = {"model": name, "seed": seed, "provenance": hashes, "smoke": smoke, "backend": backend}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def fit(cfg, meta, model, optimizer, train, val, train_batches, val_batches, directory, identity, seed,
        device, *, resume=False, smoke=False, pause_after_epoch=None):
    """Epoch-atomic resume. Any interrupted partial epoch is replayed from its start."""
    max_epochs = 2 if smoke else cfg["training"]["max_epochs"]
    count = min(3 * cfg["training"]["batch_size"], len(train)) if smoke else len(train)
    history, best_score, best_epoch, bad_epochs, best_weights, start_epoch = [], -math.inf, None, 0, None, 0
    if resume:
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise RuntimeError("Checkpoint identity mismatch")
        if saved["torch_version"] != str(torch.__version__) or saved["device_type"] != device.type:
            raise RuntimeError("Resume requires the same PyTorch version and CPU/CUDA device type")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, best_score, best_epoch, bad_epochs, best_weights = (saved[key] for key in
            ("history", "best_score", "best_epoch", "bad_epochs", "best_model"))
        start_epoch = saved["completed_epochs"]
        # last.pt is authoritative if power failed between checkpoint and CSV writes.
        write_csv(directory / "history.csv", history)
    size = cfg["training"]["batch_size"]
    for epoch in range(start_epoch, max_epochs):
        if bad_epochs >= cfg["training"]["patience"]:
            break
        indices = train.paired_epoch_indices(seed, epoch, count)
        synchronize(device)
        started, weighted_loss = time.perf_counter(), 0.0
        n_steps = math.ceil(count / size)
        for step, offset in enumerate(range(0, count, size), 1):
            current = indices[offset:offset + size]
            loss = train_step(model, optimizer, train_batches(current), cfg, meta["q_ref_mm_day"])
            weighted_loss += loss * len(current)
            if step == 1 or step % 100 == 0 or step == n_steps:
                print(f"epoch {epoch+1}/{max_epochs} step {step}/{n_steps} loss={loss:.6f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        synchronize(device)
        train_seconds = time.perf_counter() - started
        val_started = time.perf_counter()
        rows, summary, records = predict_validation(model, val, val_batches, cfg, meta["q_ref_mm_day"])
        score = summary["median_nse"]
        if score is None:
            raise RuntimeError("No valid validation NSE; early stopping cannot select a model")
        improved = score > best_score  # Strict improvement; earliest tie is retained.
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch + 1, 0
            best_weights = cpu_weights(model)
        else:
            bad_epochs += 1
        history.append({"epoch": epoch + 1, "train_loss": weighted_loss / count, "validation_median_nse": score,
                        "validation_mean_nse": summary["mean_nse"], "invalid_nse_basins": summary["invalid_nse_basins"],
                        "best_epoch": best_epoch, "bad_epochs": bad_epochs, "sampled_windows": count,
                        "optimizer_steps": n_steps, "sampling_sha256": sampling_hash(indices),
                        "training_seconds": train_seconds, "validation_seconds": time.perf_counter() - val_started})
        state = {"identity": identity, "torch_version": str(torch.__version__), "device_type": device.type,
                 "model": cpu_weights(model), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                 "completed_epochs": epoch + 1, "history": history, "best_score": best_score,
                 "best_epoch": best_epoch, "bad_epochs": bad_epochs, "best_model": best_weights}
        atomic_checkpoint(directory / "last.pt", state)
        write_csv(directory / "history.csv", history)
        print(f"validation median NSE={score:.6f}; best epoch={best_epoch}; patience={bad_epochs}/{cfg['training']['patience']}", flush=True)
        if pause_after_epoch is not None and epoch + 1 == pause_after_epoch:
            return None  # Used by integration tests to simulate an epoch-boundary stop.
    if best_weights is None:
        raise RuntimeError("Training produced no selectable epoch")
    model.load_state_dict(best_weights)
    rows, summary, records = predict_validation(model, val, val_batches, cfg, meta["q_ref_mm_day"])
    if not np.isclose(summary["median_nse"], best_score, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Reloaded best checkpoint does not reproduce its selection metric")
    atomic_checkpoint(directory / "best.pt", {"identity": identity, "model": best_weights, "epoch": best_epoch})
    save_validation(directory, rows, summary, records)
    result = {"status": "engineering_smoke_only" if smoke else "completed", "completed_epochs": len(history),
              "best_epoch": best_epoch, "stopped_by_patience": bad_epochs >= cfg["training"]["patience"],
              "validation": summary, "test_q_read": False, "identity": identity,
              "finished_at": str(datetime.now(timezone.utc))}
    save_json(directory / "result.json", result)
    return result

