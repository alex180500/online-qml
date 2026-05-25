import argparse
from pathlib import Path
import torch
from online_qml import *

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, default=Path("my-tests/data/dim_sweep"))
parser.add_argument("--d-grid", nargs="+", type=int, default=[2, 3, 4])
parser.add_argument("-t", "--ntrain", type=int, default=100_000)
parser.add_argument("-s", "--shots", type=int, default=1)
parser.add_argument("--nseeds", type=int, default=3)
parser.add_argument("--methods", nargs="+", default=training_methods)
parser.add_argument("--pinv-tol", type=float, default=1e-10)
parser.add_argument("--ridge-alpha", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--precision", type=str, default="float64")
parser.add_argument("--torch-threads", type=int, default=None)
args = parser.parse_args()


device, rdtype, cdtype = torch_setup(
    device=args.device,
    precision=args.precision,
    threads=args.torch_threads,
    verbose=True,
)

out_dir = args.folder / f"ntrain_{args.ntrain}" / f"shots_{args.shots}"
out_dir.mkdir(parents=True, exist_ok=True)

dimension_configs = []
for d in args.d_grid:
    n_out = d**3
    dimension_configs.append(
        {
            "d": d,
            "n_out": n_out,
            "n_train": args.ntrain,
            "alpha": n_out / (d * d),
        }
    )

run_metadata = {
    "script": "dim_sweep.py",
    "experiment": "dim_sweep",
    "shots": args.shots,
    "d_grid": args.d_grid,
    "n_out_rule": "d**3",
    "n_train": args.ntrain,
    "nseeds": args.nseeds,
    "methods": args.methods,
    "precision": args.precision,
    "dimension_configs": dimension_configs,
    "seeds": [random_seed() for _ in range(args.nseeds)],
}
save_json(run_metadata, out_dir / "metadata.json")

metric_names = ("bias2", "variance")
metric_files: dict[int, Path] = {}
d_grid = torch.tensor(args.d_grid, device=device, dtype=torch.int64)

for seed_id in range(args.nseeds):
    seed = run_metadata["seeds"][seed_id]
    seed_dir = seed_run(out_dir, seed_id, seed)
    metric_files[seed_id] = seed_dir / "metrics.pt"
    seed_metrics: dict[str, list[torch.Tensor]] = {}

    for config in dimension_configs:
        d = int(config["d"])
        n_out = int(config["n_out"])

        data, sample_time = timed(
            sample_data,
            args.ntrain,
            d=d,
            n_out=n_out,
            shots=args.shots,
            seed=seed,
            device=device,
            dtype=cdtype,
        )

        observable = sample_dm(1, d=d, device=device, dtype=cdtype).T
        train_grid = torch.tensor([args.ntrain], device=device, dtype=torch.int64)
        result, layer_time = timed(
            ntrain_layers,
            data,
            observable,
            train_grid=train_grid,
            n_shots=args.shots,
            methods=args.methods,
            pinv_tol=args.pinv_tol,
            ridge_alpha=args.ridge_alpha,
            dtype=rdtype,
        )
        save_pt(result, seed_dir / f"d_{d}_layers.pt")

        metrics, metric_time = timed(haar_metrics, result, data.povm)
        for metric_name in metric_names:
            for method in args.methods:
                key = f"{method}_{metric_name}"
                if key not in metrics.metrics:
                    continue
                value = metrics.metrics[key].detach().cpu().reshape(-1)[0]
                seed_metrics.setdefault(key, []).append(value)

        print(
            f"[seed {seed_id} / {args.nseeds}] "
            f"d={d} n_out={n_out} ntrain={args.ntrain} shots={args.shots} "
            f"sample={sample_time:.2f}s layers={layer_time:.2f}s metrics={metric_time:.2f}s"
        )

    seed_metric_result = MetricResult(
        metrics={
            key: torch.stack(values).to(device=device)
            for key, values in seed_metrics.items()
        },
        train_grid=d_grid,
        shot_grid=torch.tensor([args.shots], device=device, dtype=torch.int64),
        seed=seed,
        metadata={
            "sweep": "dim",
            "n_train": args.ntrain,
            "n_out_rule": "d**3",
            "methods": args.methods,
        },
    )
    save_pt(seed_metric_result, seed_dir / "metrics.pt")

save_metrics(
    metric_files,
    out_dir / "metrics",
    methods=args.methods,
    metric_names=metric_names,
    grid_key="train_grid",
    grid_column="d",
    fixed_columns={"shots": args.shots, "n_train": args.ntrain},
)
