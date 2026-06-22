import argparse
from pathlib import Path
from online_qml import *

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="my-data/shot_sweep")
parser.add_argument("-d", "--dim", type=int, default=2)
parser.add_argument("-o", "--n-out", type=int, default=16)
parser.add_argument("-t", "--ntrain", type=int, default=1000)
parser.add_argument("-s", "--shot-max", type=int, default=10_000)
parser.add_argument("--shot-start", type=int, default=1)
parser.add_argument("--shot-step", type=int, default=40)
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

out_dir = Path(args.folder).expanduser() / f"ntrain_{args.ntrain}"
out_dir.mkdir(parents=True, exist_ok=True)

run_metadata = {
    "script": "shot_sweep.py",
    "dim": args.dim,
    "n_out": args.n_out,
    "n_train": args.ntrain,
    "shot_start": args.shot_start,
    "shot_max": args.shot_max,
    "shot_step": args.shot_step,
    "nseeds": args.nseeds,
    "methods": args.methods,
    "precision": args.precision,
    "seeds": [random_seed() for _ in range(args.nseeds)],
}
save_json(run_metadata, out_dir / "metadata.json")

shot_grid = logspace_int(args.shot_start, args.shot_max, args.shot_step)
metric_files: dict[int, Path] = {}

for seed_id in range(args.nseeds):
    seed = run_metadata["seeds"][seed_id]
    seed_dir = seed_run(out_dir, seed_id, seed)
    metric_files[seed_id] = seed_dir / "metrics.pt"

    data, sample_time = timed(
        sample_data,
        args.ntrain,
        d=args.dim,
        n_out=args.n_out,
        shots=args.shot_max,
        seed=seed,
        device=device,
        dtype=cdtype,
    )

    observable = sample_dm(1, d=args.dim, device=device, dtype=cdtype).adjoint()
    result, layer_time = timed(
        shot_layers,
        data,
        observable,
        shot_grid=shot_grid,
        n_train=args.ntrain,
        methods=args.methods,
        pinv_tol=args.pinv_tol,
        ridge_alpha=args.ridge_alpha,
        dtype=rdtype,
    )
    save_pt(result, seed_dir / "layers.pt")

    metrics, metric_time = timed(haar_metrics, result, data.povm)
    save_pt(metrics, seed_dir / "metrics.pt")

    print(
        f"[seed {seed_id} / {args.nseeds}] "
        f"sample={sample_time:.2f}s layers={layer_time:.2f}s metrics={metric_time:.2f}s"
    )

save_metrics(
    metric_files,
    out_dir / "metrics",
    methods=args.methods,
    metric_names=("bias2", "variance"),
    grid_column="shots",
)
