import argparse
from pathlib import Path
import torch
from online_qml import *

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, default=Path("my-tests/data/dim_sweep"))
parser.add_argument("-f", "--folder-name", type=str, default="")
parser.add_argument("-d","--d-grid", nargs="+", type=int, default=[2, 3, 4])
parser.add_argument("--n-out-rule", type=str, default="d**3")
parser.add_argument("-t", "--ntrain", type=int, default=10_000)
parser.add_argument("-s", "--shots", type=int, default=100)
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

out_dir = args.folder / args.folder_name
out_dir.mkdir(parents=True, exist_ok=True)

n_out_grid = [eval(args.n_out_rule, {"d": d}) for d in args.d_grid]
print(
    f"Using n_out_grid={n_out_grid} from rule '{args.n_out_rule}' and d_grid={args.d_grid}"
)

run_metadata = {
    "script": "dim_sweep.py",
    "experiment": "dim_sweep",
    "shots": args.shots,
    "d_grid": args.d_grid,
    "n_out_grid": n_out_grid,
    "n_train": args.ntrain,
    "nseeds": args.nseeds,
    "methods": args.methods,
    "precision": args.precision,
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
    metric_results: list[MetricResult] = []

    tot_sample_time = 0.0
    tot_layer_time = 0.0
    tot_metric_time = 0.0

    for d, n_out in zip(args.d_grid, n_out_grid, strict=True):
        data, sample_time = timed(
            sample_data,
            args.ntrain,
            d=d,
            n_out=n_out,
            shots=args.shots,
            seed=seed,
            device=device,
            dtype=cdtype,
            previous_time=tot_sample_time,
        )
        tot_sample_time = sample_time

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
            previous_time=tot_layer_time,
        )
        tot_layer_time = layer_time
        save_pt(result, seed_dir / f"d_{d}_layers.pt")

        metrics, metric_time = timed(
            haar_metrics,
            result,
            data.povm,
            previous_time=tot_metric_time,
        )
        tot_metric_time = metric_time
        metric_results.append(metrics)

    print(
        f"[seed {seed_id} / {args.nseeds}] "
        f"sample={tot_sample_time:.2f}s "
        f"layers={tot_layer_time:.2f}s "
        f"metrics={tot_metric_time:.2f}s"
    )

    seed_metric_result = stack_metric_results(
        metric_results,
        grid_name="d",
        grid_values=d_grid,
        extra_coords={
            "n_out": n_out_grid,
            "n_train": args.ntrain,
            "shots": args.shots,
        },
    )
    save_pt(seed_metric_result, seed_dir / "metrics.pt")

save_metrics(
    metric_files,
    out_dir / "metrics",
    methods=args.methods,
    metric_names=metric_names,
    grid_column="d",
)
