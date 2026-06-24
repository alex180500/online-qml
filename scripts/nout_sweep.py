import argparse
from pathlib import Path
from online_qml import *

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="my-data/nout_sweep")
parser.add_argument("-d", "--dim", type=int, default=2)
parser.add_argument("-g", "--gamma", type=int, default=500)
parser.add_argument("-s", "--shots", type=int, default=1000)
parser.add_argument("-a", "--alpha-max", type=int, default=512)
parser.add_argument("--alpha-start", type=int, default=1)
parser.add_argument("--alpha-step", type=int, default=30)
parser.add_argument("--nseeds", type=int, default=5)
parser.add_argument("--methods", nargs="+", default=shadow_methods)
parser.add_argument("--obs", choices=("proj", "center", "center_norm"), default="proj")
parser.add_argument("--pinv-tol", type=float, default=1e-10)
parser.add_argument("--ridge-alpha", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--precision", type=str, default="float64")
parser.add_argument("--torch-threads", type=int, default=None)
args = parser.parse_args()

if args.shots <= 0:
    raise ValueError("--shots must be positive.")

device, rdtype, cdtype = torch_setup(
    device=args.device,
    precision=args.precision,
    threads=args.torch_threads,
    verbose=True,
)

ntrain = args.gamma * args.dim * args.dim

alpha_grid = logspace_int(args.alpha_start, args.alpha_max, args.alpha_step)
n_out_grid = alpha_grid * args.dim * args.dim
shot_label = f"shots_{args.shots}"
out_dir = (
    Path(args.folder).expanduser() / shot_label / f"d_{args.dim}" / f"ntrain_{ntrain}"
)
out_dir.mkdir(parents=True, exist_ok=True)

run_metadata = {
    "script": "nout_sweep.py",
    "dim": args.dim,
    "n_train": ntrain,
    "gamma": ntrain / (args.dim * args.dim),
    "shots": args.shots,
    "alpha_start": args.alpha_start,
    "alpha_max": args.alpha_max,
    "alpha_step": args.alpha_step,
    "alpha_grid": alpha_grid.tolist(),
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
    seed_dir = seed_run(out_dir, seed_id, seed)
    metric_files[seed_id] = seed_dir / "metrics.pt"

    observable = sample_observable(
        1,
        d=args.dim,
        kind=args.obs,
        device=device,
        dtype=cdtype,
    )

    metrics, metric_time = timed(
        nout_metrics,
        ntrain,
        args.dim,
        observable,
        alpha_grid,
        n_out_grid,
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
    grid_column="alpha",
)
