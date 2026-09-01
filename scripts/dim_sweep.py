import argparse
from pathlib import Path
from online_qml import *
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="my-data/d_sweep")
parser.add_argument("-d", "--dims", nargs="+", type=int, default=range(2, 11))
parser.add_argument("-g", "--gamma", type=int, default=200)
parser.add_argument("-s", "--shots", type=int, default=1_000_000)
parser.add_argument("-a", "--alpha", type=int, default=16)
parser.add_argument("--nseeds", type=int, default=20)
parser.add_argument("--methods", nargs="+", default=study_methods)
parser.add_argument("--obs", choices=("proj", "center", "center_norm"), default="proj")
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

dim_grid = torch.tensor(sorted(set(args.dims)), dtype=torch.int64)

n_train_grid = args.gamma * dim_grid.square()
n_out_grid = args.alpha * dim_grid.square()

out_dir = Path(args.folder).expanduser()
out_dir = out_dir / f"a_{args.alpha}_g_{args.gamma}_s_{args.shots}"
out_dir.mkdir(parents=True, exist_ok=True)

run_metadata = {
    "script": "dim_sweep.py",
    "dims": dim_grid.tolist(),
    "gamma": args.gamma,
    "n_train_rule": "gamma * d**2",
    "n_train_grid": n_train_grid.tolist(),
    "shots": args.shots,
    "alpha": args.alpha,
    "n_out_rule": "alpha * d**2",
    "n_out_grid": n_out_grid.tolist(),
    "nseeds": args.nseeds,
    "methods": args.methods,
    "observable": args.obs,
    "pinv_tol": args.pinv_tol,
    "ridge_alpha": args.ridge_alpha,
    "precision": args.precision,
    "seeds": [random_seed() for _ in range(args.nseeds)],
}
save_json(run_metadata, out_dir / "metadata.json")

metric_files = {}

for seed_id in range(args.nseeds):
    seed = run_metadata["seeds"][seed_id]
    rng = np.random.default_rng(seed=seed)
    seed_dir = seed_run(out_dir, seed_id, seed)
    metric_files[seed_id] = seed_dir / "metrics.pt"

    observables = [
        sample_observable(
            1,
            d=int(d),
            kind=args.obs,
            device=device,
            dtype=cdtype,
        )
        for d in dim_grid
    ]

    metrics, metric_time = timed(
        dim_metrics,
        dim_grid,
        observables,
        args.gamma,
        args.alpha,
        args.shots,
        args.methods,
        pinv_tol=args.pinv_tol,
        ridge_alpha=args.ridge_alpha,
        dtype=rdtype,
        seed=seed,
    )
    save_pt(metrics, seed_dir / "metrics.pt")

    print(f"[seed {seed_id} / {args.nseeds}] " f"metrics={metric_time:.2f}s")

save_metrics(
    metric_files,
    out_dir / "metrics",
    methods=args.methods,
    metric_names=("bias2", "variance"),
    grid_column="d",
)
