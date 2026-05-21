import argparse
from pathlib import Path

from online_qml import (
    haar_metrics,
    logspace_int,
    ntrain_layers,
    random_seed,
    sample_data,
    sample_dm,
    save_json,
    save_metrics,
    save_pt,
    seed_run,
    timed,
    torch_setup,
)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, default=Path("my-tests/data/ntrain_sweep"))
parser.add_argument("-d", "--dim", type=int, default=2)
parser.add_argument("-o", "--n-out", type=int, default=16)
parser.add_argument("-s", "--shots", type=int, default=1)
parser.add_argument("-t", "--max-ntrain", type=int, default=50_000)
parser.add_argument("--train-start", type=int, default=1)
parser.add_argument("--train-step", type=int, default=30)
parser.add_argument("--nseeds", type=int, default=3)
parser.add_argument(
    "--methods",
    nargs="+",
    default=["ost", "state_prior_ost", "povm_prior_ost", "prior_ost", "pinv", "ridge"],
)
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

out_dir = args.folder / f"shots_{args.shots}"
out_dir.mkdir(parents=True, exist_ok=True)

run_metadata = {
    "script": "ntrain_sweep.py",
    "dim": args.dim,
    "n_out": args.n_out,
    "shots": args.shots,
    "max_ntrain": args.max_ntrain,
    "train_start": args.train_start,
    "train_step": args.train_step,
    "nseeds": args.nseeds,
    "methods": args.methods,
    "precision": args.precision,
    "seeds": [random_seed() for _ in range(args.nseeds)],
}
save_json(run_metadata, out_dir / "metadata.json")

train_grid = logspace_int(args.train_start, args.max_ntrain, args.train_step)
metric_files: dict[int, Path] = {}

for seed_id in range(args.nseeds):
    seed = run_metadata["seeds"][seed_id]
    _, layer_path, metric_path = seed_run(out_dir, seed_id, seed)
    metric_files[seed_id] = metric_path

    data, sample_time = timed(
        sample_data,
        args.max_ntrain,
        d=args.dim,
        n_out=args.n_out,
        shots=args.shots,
        seed=seed,
        device=device,
        dtype=cdtype,
    )

    observable = sample_dm(1, d=args.dim, device=device, dtype=cdtype).T
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
    save_pt(result, layer_path)

    metrics, metric_time = timed(haar_metrics, result, data.povm)
    save_pt(metrics, metric_path)

    print(
        f"[seed {seed_id} / {args.nseeds}] "
        f"sample={sample_time:.2f}s layers={layer_time:.2f}s metrics={metric_time:.2f}s"
    )

save_metrics(
    metric_files,
    out_dir / "metrics",
    methods=args.methods,
    metric_names=("bias2", "variance"),
    grid_key="train_grid",
    grid_column="n_train",
    fixed_columns={"shots": args.shots},
)
