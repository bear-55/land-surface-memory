"""Train the paired B0/A1/M1 experiments using validation-only selection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_backend import attach_graphs
from model import parameter_count
from project import ROOT, read_json, save_json, selected_roster
from train import fit, identity_for
from training import (datasets, environment, load_protocol, make_model,
                      optimizer_for, provenance, setup_device)


def run(args):
    cfg, meta = load_protocol()
    names, seeds = selected_roster(args.models, args.seeds)
    device = setup_device(args.device)
    if args.backend == "cuda_graph" and device.type != "cuda":
        raise ValueError("CUDA graphs require --device cuda")
    hashes = provenance()
    train, val, train_batches, val_batches = datasets(meta, device)
    for name in names:
        backend = ("reference_manual_loop" if args.backend == "reference" else
                   "native_fused" if name == "B0" else
                   "cuda_graph" if device.type == "cuda" else "reference_manual_loop")
        for seed in seeds:
            directory = ROOT / f"runs/formal_pub_527/{name}/seed_{seed}"
            identity = identity_for(name, seed, hashes, False, backend)
            if directory.exists():
                if not args.resume:
                    raise FileExistsError(f"Run exists; use --resume: {directory}")
                spec = read_json(directory / "run.json")
                if spec["identity"] != identity or spec["provenance"] != hashes:
                    raise RuntimeError("Resume requires unchanged code, data, and backend")
                if (directory / "result.json").exists():
                    print(f"Completed run retained: {directory}", flush=True)
                    continue
            else:
                directory.mkdir(parents=True)
                save_json(directory / "run.json", {
                    "identity": identity, "model": name, "seed": seed,
                    "backend": backend, "protocol": cfg, "provenance": hashes,
                    "environment": environment(device), "smoke": False,
                    "created_at_utc": datetime.now(timezone.utc).isoformat()})
            model = make_model(cfg, meta, name, seed, device,
                               "native_fused" if backend == "native_fused" else "reference_manual_loop")
            if backend == "cuda_graph":
                attach_graphs(model, train_batches, val_batches, len(train), len(val),
                              cfg["training"]["batch_size"])
            print(f"{name} seed={seed} backend={backend} parameters={parameter_count(model)}", flush=True)
            fit(cfg, meta, model, optimizer_for(model, cfg), train, val,
                train_batches, val_batches, directory, identity, seed, device,
                resume=(directory / "last.pt").exists())
            if hasattr(model, "_graph_backend"):
                model.forward = model._graph_backend.raw
                del model._graph_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", default=["all"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--backend", choices=["auto", "reference", "cuda_graph"], default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
