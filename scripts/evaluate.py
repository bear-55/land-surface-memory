"""Evaluate fixed validation-selected weights on held-out basins in either period."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import summarize, write_csv
from evaluation import (EvaluationWindows, compare_validation, load_completed_model,
                        metric_rows, parse_streamflow, predict, require)
from project import ROOT, digest, read_json, save_json, selected_roster
from training import environment, load_protocol, provenance, setup_device


def release(model, device):
    if hasattr(model, "_graph_backend"):
        model.forward = model._graph_backend.raw
        del model._graph_backend
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def target_dataset(cache, meta, period, output):
    ids = meta["splits"]["test"]
    q = np.full((len(ids), len(cache["dates"])), np.nan, dtype=np.float32)
    access = []
    for j, basin in enumerate(ids):
        source = meta["source_files"][basin]
        path = Path(source["streamflow"])
        checksum = digest(path)
        expected = source.get("streamflow_sha256")
        if expected is not None:
            require(checksum == expected, "Raw streamflow file changed: " + basin)
        dates, values = parse_streamflow(path.read_text(encoding="utf-8-sig"), basin,
                                        period, source["area_m2"])
        positions = np.searchsorted(cache["dates"], dates)
        require(bool((positions < len(cache["dates"])).all()) and
                np.array_equal(cache["dates"][positions], dates), "Evaluation dates missing from forcing cache")
        q[j, positions] = values
        access.append({"basin_id": basin, "period": period, "source_sha256": checksum,
                       "valid_q_days": int(np.isfinite(values).sum()), "area_m2": source["area_m2"]})
    save_json(output / "test_label_access.json", {
        "time_utc": datetime.now(timezone.utc).isoformat(), "purpose": "fixed_weight_evaluation",
        "labels_used_for_training_or_selection": False, "basins": access})
    return EvaluationWindows(cache, meta, ids, period, q)


def run(args):
    cfg, meta = load_protocol()
    names, seeds = selected_roster(args.models, args.seeds)
    device = setup_device(args.device)
    hashes = provenance()
    period = cfg["periods"]["final_labels" if args.period == "test" else "frozen_same_period_diagnostic"]
    output = ROOT / f"runs/evaluation_{args.period}"
    output.mkdir(parents=True, exist_ok=True)
    roster = [(name, seed, ROOT / f"runs/formal_pub_527/{name}/seed_{seed}")
              for name in names for seed in seeds]
    context = {"period": period, "models": names, "seeds": seeds, "provenance": hashes,
               "checkpoints": {f"{n}/seed_{s}": digest(d / "best.pt") for n, s, d in roster},
               "environment": environment(device), "use_cuda_graphs": not args.no_cuda_graph,
               "weights_fixed": True, "validation_selection_repeated": False}
    identity = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    protocol = output / "protocol.json"
    if protocol.exists():
        require(read_json(protocol)["identity"] == identity, "Existing evaluation protocol differs")
    else:
        save_json(protocol, {"identity": identity, "context": context})
    with np.load(ROOT / "data/formal/formal.npz", allow_pickle=False) as saved:
        cache = {k: saved[k] for k in ("basin_ids", "dates", "x_dynamic", "x_static", "dynamic_valid", "q_val")}
    validation = EvaluationWindows(cache, meta, meta["splits"]["val"],
                                   cfg["periods"]["development_labels"], cache["q_val"])
    checks = []
    for name, seed, directory in roster:
        model, spec, result = load_completed_model(directory, cfg, meta, device, hashes)
        require(spec["model"] == name and spec["seed"] == seed, "Run folder does not match its model/seed")
        prediction = predict(model, validation, cfg, meta, device,
                             f"Validation {name}/{seed}", use_graph=not args.no_cuda_graph)
        with np.load(directory / "predictions_val.npz", allow_pickle=False) as saved:
            check = compare_validation(validation.records(prediction), saved,
                                       metric_rows(validation, prediction, cfg), result["validation"])
        checks.append({"model": name, "seed": seed, **check})
        release(model, device)
    save_json(output / "validation_recheck.json", {"identity": identity, "status": "passed",
                                                   "test_labels_parsed": False, "runs": checks})
    if args.check_only:
        print("Validation recheck passed; no held-out labels opened.")
        return
    dataset = target_dataset(cache, meta, period, output)
    write_csv(output / "common_mask_summary.csv", [
        {"basin_id": basin, "n_days": int((dataset.basins == j).sum())}
        for j, basin in enumerate(dataset.ids)])
    summaries = []
    for name, seed, directory in roster:
        target = output / name / f"seed_{seed}"
        target.mkdir(parents=True, exist_ok=True)
        model, _, result = load_completed_model(directory, cfg, meta, device, hashes)
        prediction = predict(model, dataset, cfg, meta, device, f"{args.period} {name}/{seed}",
                             use_graph=not args.no_cuda_graph)
        require(digest(directory / "best.pt") == context["checkpoints"][f"{name}/seed_{seed}"],
                "Selected checkpoint changed during evaluation")
        rows = metric_rows(dataset, prediction, cfg)
        records = dataset.records(prediction)
        filename = f"predictions_{args.period}.npz"
        np.savez_compressed(target / filename, **records)
        write_csv(target / "metrics_per_basin.csv", rows)
        summary = summarize(rows)
        save_json(target / "result.json", {"status": "completed", "identity": identity,
            "model": name, "seed": seed, "best_epoch": result["best_epoch"],
            "checkpoint_sha256": context["checkpoints"][f"{name}/seed_{seed}"],
            "period": period, "weights_fixed": True, "test": summary,
            "artifacts": {f: digest(target / f) for f in (filename, "metrics_per_basin.csv")}})
        summaries.append({"model": name, "seed": seed, "best_epoch": result["best_epoch"], **summary})
        release(model, device)
    write_csv(output / "run_summary.csv", summaries)
    save_json(output / "completed.json", {"status": "completed", "identity": identity,
        "runs": len(roster), "test_basins": len(dataset.ids), "period": period,
        "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Evaluation completed: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=["test", "historical"], default="test")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", default=["all"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
