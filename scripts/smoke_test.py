"""Synthetic CPU checks; no CAMELS download or fitted checkpoint is required."""
import copy
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import CamelsWindows, DevelopmentStreamflow, valid_window_ends
from evaluate import basin_metrics
from evaluation import EvaluationWindows, predict
from model import copy_base_weights
from project import ROOT
from train import fit
from training import (WindowBatches, cpu_weights, make_model, optimizer_for,
                      predict_scaled, setup_device, train_step)


def toy_data(directory):
    rng = np.random.default_rng(32)
    dates = np.arange("2000-01-01", "2000-01-17", dtype="datetime64[D]")
    dynamic = rng.normal(size=(4, 16, 2)).astype(np.float32)
    static = rng.normal(size=(4, 3)).astype(np.float32)
    q = (np.arange(64).reshape(4, 16) % 7 + 1).astype(np.float32)
    cache = dict(basin_ids=np.array(["00000001", "00000002", "00000003", "00000004"]),
                 dates=dates, x_dynamic=dynamic, x_static=static,
                 dynamic_valid=np.ones((4, 16), dtype=bool), q_train=q[:2], q_val=q[2:])
    path = directory / "toy.npz"
    np.savez_compressed(path, **cache)
    meta = {"splits": {"train": cache["basin_ids"][:2].tolist(), "val": cache["basin_ids"][2:].tolist(),
                       "test": ["00000005"]}, "sequence_length": 3,
            "land_indices": [1, 2], "sigma_train_mm": {"00000001": .5, "00000002": 1.5},
            "q_ref_mm_day": 2., "label_periods": {role: ["2000-01-03", "2000-01-16"] for role in ("train", "val")},
            "dynamic_names": ["x", "y"], "static_names": ["a", "b", "c"], "land_names": ["b", "c"]}
    cfg = {"model": {"models": {"B0": "none", "A1": "shared", "M1": "conditional"},
                     "hidden_size": 4, "rank": 2, "seq_length": 3, "output_dropout": .2},
           "training": {"batch_size": 4, "max_epochs": 2, "patience": 8, "learning_rate": .001,
                        "weight_decay": 1e-5, "epsilon_mm_day": .1, "grad_clip_norm": 1.},
           "evaluation": {"min_valid_days": 2, "high_flow_quantile": .95}}
    return cfg, meta, cache, CamelsWindows(path, meta, "train"), CamelsWindows(path, meta, "val")


def compare_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            compare_nested(left, right)
    else:
        assert a == b


def run_smoke():
    device = setup_device("cpu")
    temporary_root = ROOT / "runs"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke_", dir=temporary_root) as temporary:
        directory = Path(temporary)
        cfg, meta, cache, train, val = toy_data(directory)
        tb, vb = WindowBatches(train, device), WindowBatches(val, device)
        batch = tb([0, 1, 15, 16])
        models = [make_model(cfg, meta, name, 11, device).eval() for name in ("B0", "A1", "M1")]
        for model in models[1:]:
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(predict_scaled(models[0], batch), predict_scaled(model, batch), rtol=0, atol=0)
        torch.testing.assert_close(models[0].fused_baseline(batch["x_dynamic"], batch["x_static"]),
                                   predict_scaled(models[0], batch), rtol=1e-5, atol=1e-6)
        shared, equivalent = models[1], make_model(cfg, meta, "B0", 11, device).eval()
        with torch.no_grad():
            shared.up_projection.weight.normal_(0, .1)
            copy_base_weights(shared, equivalent)
            update = shared.up_projection.weight @ shared.context_projection.weight
            h = shared.hidden_size
            equivalent.core.weight_ih_l0[h:2*h, :shared.dynamic_size] += update[:, :shared.dynamic_size]
            equivalent.core.weight_hh_l0[h:2*h] += update[:, shared.dynamic_size:]
        torch.testing.assert_close(predict_scaled(shared, batch), predict_scaled(equivalent, batch), rtol=1e-5, atol=1e-6)
        indices = train.paired_epoch_indices(11, 0, 100)
        assert np.array_equal(indices, train.paired_epoch_indices(11, 0, 100))
        assert not np.array_equal(indices, train.paired_epoch_indices(29, 0, 100))
        for name in ("B0", "A1", "M1"):
            run = directory / name
            run.mkdir()
            model = make_model(cfg, meta, name, 11, device)
            result = fit(cfg, meta, model, optimizer_for(model, cfg), train, val, tb, vb,
                         run, name, 11, device)
            assert result["status"] == "completed" and not result["test_q_read"]
            assert (run / "best.pt").is_file() and (run / "predictions_val.npz").is_file()
        resumed = directory / "resumed"
        resumed.mkdir()
        model = make_model(cfg, meta, "M1", 11, device)
        fit(cfg, meta, model, optimizer_for(model, cfg), train, val, tb, vb,
            resumed, "M1", 11, device, pause_after_epoch=1)
        model = make_model(cfg, meta, "M1", 29, device)
        fit(cfg, meta, model, optimizer_for(model, cfg), train, val, tb, vb,
            resumed, "M1", 11, device, resume=True)
        full = torch.load(directory / "M1/last.pt", weights_only=True)
        continued = torch.load(resumed / "last.pt", weights_only=True)
        for key in ("model", "optimizer", "rng", "best_model", "best_epoch", "best_score"):
            compare_nested(full[key], continued[key])
        dataset = EvaluationWindows(cache, meta, meta["splits"]["val"],
                                    meta["label_periods"]["val"], cache["q_val"])
        before = cpu_weights(model)
        output = predict(model, dataset, cfg, meta, device, use_graph=False)
        compare_nested(before, cpu_weights(model))
        assert np.isfinite(output).all() and (output >= 0).all()
        assert len(output) == len(val)
        bad_valid = np.ones(16, dtype=bool)
        bad_valid[4] = False
        ends = valid_window_ends(bad_valid, cache["dates"], "2000-01-03", "2000-01-16", 3)
        assert not set((4, 5, 6)).intersection(ends)
        reader = DevelopmentStreamflow({}, meta["splits"], meta["label_periods"])
        try:
            reader.read("00000005", "test", 1e6)
        except PermissionError:
            pass
        else:
            raise AssertionError("Held-out labels were not protected")
        assert basin_metrics([1, 2, 3], [2, 2, 2], min_days=3)["nse"] == 0
        try:
            basin_metrics([1, 2, 3], [1, np.nan, 3], min_days=3)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Nonfinite predictions were silently masked")
    print("PASS: paired initialization, A1 equivalence, sampling, three-model training, exact resume, fixed weights, masks, metrics, and label isolation.")


if __name__ == "__main__":
    run_smoke()
