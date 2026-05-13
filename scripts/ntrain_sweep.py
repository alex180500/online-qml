#!/usr/bin/env python3
"""Generate data points for the OnlineQELM n_train sweep.

Run from the package root:

    uv run python scripts/ntrain_sweep.py

Outputs:

    ./data/<run-name>/
        metadata.json
        raw/
        layers/
        metrics/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from online_qml.core import (
    SimulationContext,
    SimulationData,
    get_complex_dtype,
    get_real_dtype,
    get_torch_device,
    load_simulation_data,
    logspace_int,
    save_data,
    save_simulation_data,
)
from online_qml.experiments import evaluate_layer_result_haar, make_layers_ntrain_grid
from online_qml.quantum import sample_dm, sample_povm, sample_traceless_operator, shots_outcome


def check_folder(folder: Path, pattern: str, expected_count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    found = list(folder.glob(pattern))
    print(f"{folder}: found {len(found)}/{expected_count} files matching {pattern}")


parser = argparse.ArgumentParser(description="Run the n_train sweep experiment.")
parser.add_argument("--data-root", type=Path, default=Path("./data"))
parser.add_argument("--run-name", type=str, default="ntrain_sweep")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--n-out", type=int, default=16)
parser.add_argument("--max-shots", type=int, default=100)
parser.add_argument("--max-ntrain", type=int, default=50_000)
parser.add_argument("--grid-start", type=int, default=10)
parser.add_argument("--grid-stop", type=int, default=None)
parser.add_argument("--grid-num", type=int, default=30)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--shots", nargs="+", type=int, default=[1, 10, 100])
parser.add_argument(
    "--methods",
    nargs="+",
    default=["ost", "aost", "prior_ost", "prior_aost", "pinv", "ridge"],
)
parser.add_argument("--traceless", action="store_true")
parser.add_argument("--observable-seed", type=int, default=10_000)
parser.add_argument("--pinv-tol", type=float, default=1e-10)
parser.add_argument("--ridge-alpha", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
parser.add_argument("--torch-threads", type=int, default=None)
parser.add_argument("--force-data", action="store_true")
parser.add_argument("--force-layers", action="store_true")
parser.add_argument("--force-metrics", action="store_true")
args = parser.parse_args()

if args.torch_threads is not None:
    torch.set_num_threads(args.torch_threads)

args.grid_stop = args.grid_stop or args.max_ntrain
if args.grid_stop > args.max_ntrain:
    raise ValueError("--grid-stop cannot be larger than --max-ntrain")
if max(args.shots) > args.max_shots:
    raise ValueError("max(--shots) cannot be larger than --max-shots")

out_dir = args.data_root / args.run_name
raw_dir = out_dir / "raw"
layer_dir = out_dir / "layers"
metric_dir = out_dir / "metrics"
for folder in (raw_dir, layer_dir, metric_dir):
    folder.mkdir(parents=True, exist_ok=True)

check_folder(raw_dir, "seed_*.pt", len(args.seeds))
check_folder(layer_dir, "seed_*_shots_*.pt", len(args.seeds) * len(args.shots))
check_folder(metric_dir, "seed_*_shots_*.pt", len(args.seeds) * len(args.shots))

real_dtype = get_real_dtype(args.precision)
complex_dtype = get_complex_dtype(real_dtype)
device = get_torch_device(args.device)
train_grid = logspace_int(args.grid_start, args.grid_stop, args.grid_num)
shadow_methods = [m for m in args.methods if m in {"ost", "aost", "prior_ost", "prior_aost"}]
linear_methods = [m for m in args.methods if m in {"pinv", "ridge"}]

metadata = {
    "experiment": "ntrain_sweep",
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "dim": args.dim,
    "n_out": args.n_out,
    "max_shots": args.max_shots,
    "max_ntrain": args.max_ntrain,
    "train_grid": train_grid.tolist(),
    "seeds": args.seeds,
    "shots": args.shots,
    "methods": args.methods,
    "shadow_methods": shadow_methods,
    "linear_methods": linear_methods,
    "traceless": args.traceless,
    "observable_seed": args.observable_seed,
    "pinv_tol": args.pinv_tol,
    "ridge_alpha": args.ridge_alpha,
    "device": str(device),
    "precision": args.precision,
    "torch_threads": torch.get_num_threads(),
}
with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
    f.write("\n")

print(f"\nWriting to {out_dir}")
print(f"train_grid = {train_grid.tolist()}")
print(f"seeds = {args.seeds}")
print(f"shots = {args.shots}")
print(f"methods = {args.methods}\n")

for seed in args.seeds:
    raw_path = raw_dir / f"seed_{seed}.pt"

    if raw_path.exists() and not args.force_data:
        print(f"[seed {seed}] reuse raw: {raw_path}")
        data = load_simulation_data(str(raw_path), device=device, dtype=complex_dtype)
    else:
        print(f"[seed {seed}] generate raw: {raw_path}")
        with SimulationContext(device, seed):
            states = sample_dm(args.max_ntrain, d=args.dim, device=device, dtype=complex_dtype)
            povm = sample_povm(args.n_out, d=args.dim, device=device, dtype=complex_dtype)
            outcomes = shots_outcome(povm, states, args.max_shots).to(torch.int16)
        data = SimulationData(
            states=states,
            povm=povm,
            outcomes=outcomes,
            seed=seed,
            d=args.dim,
            n_out=args.n_out,
            metadata={
                "num_states": args.max_ntrain,
                "max_shots": args.max_shots,
                "precision": args.precision,
            },
        )
        save_simulation_data(data, str(raw_path))

    torch.manual_seed(args.observable_seed + seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.observable_seed + seed)
    observable = (
        sample_traceless_operator(1, d=args.dim, device=device, dtype=complex_dtype).T
        if args.traceless
        else sample_dm(1, d=args.dim, device=device, dtype=complex_dtype).T
    )

    for n_shots in args.shots:
        layer_path = layer_dir / f"seed_{seed}_shots_{n_shots}.pt"
        metric_path = metric_dir / f"seed_{seed}_shots_{n_shots}.pt"

        layers = None
        if layer_path.exists() and not args.force_layers:
            print(f"[seed {seed}, shots {n_shots}] reuse layers: {layer_path}")
        else:
            print(f"[seed {seed}, shots {n_shots}] make layers: {layer_path}")
            layers = make_layers_ntrain_grid(
                data,
                observable,
                train_grid=train_grid,
                n_shots=n_shots,
                shadow_methods=shadow_methods,
                linear_methods=linear_methods,
                pinv_tol=args.pinv_tol,
                ridge_alpha=args.ridge_alpha,
                dtype=real_dtype,
            )
            layers.metadata.update({"n_shots": n_shots, "sweep": "ntrain"})
            save_data(layers, str(layer_path))

        if metric_path.exists() and not args.force_metrics:
            print(f"[seed {seed}, shots {n_shots}] reuse metrics: {metric_path}")
        else:
            print(f"[seed {seed}, shots {n_shots}] evaluate metrics: {metric_path}")
            if layers is None:
                # Keep the script simple: recompute layers instead of adding loader code.
                layers = make_layers_ntrain_grid(
                    data,
                    observable,
                    train_grid=train_grid,
                    n_shots=n_shots,
                    shadow_methods=shadow_methods,
                    linear_methods=linear_methods,
                    pinv_tol=args.pinv_tol,
                    ridge_alpha=args.ridge_alpha,
                    dtype=real_dtype,
                )
                layers.metadata.update({"n_shots": n_shots, "sweep": "ntrain"})
            metrics = evaluate_layer_result_haar(layers, data.povm)
            metrics.metadata.update(layers.metadata)
            save_data(metrics, str(metric_path))

print("\nDone.")
print(f"metadata: {out_dir / 'metadata.json'}")
print(f"raw:      {raw_dir}")
print(f"layers:   {layer_dir}")
print(f"metrics:  {metric_dir}")
