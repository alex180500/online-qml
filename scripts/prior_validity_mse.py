import argparse
from pathlib import Path
from online_qml import *

STATE_METHODS = ("ost", "state_prior_ost")
POVM_METHODS = ("ost", "povm_prior_ost", "prior_ost")

STATE_METRICS = (
    "mse_ost",
    "mse_state_prior_ost",
    "excess_state_prior",
    "ratio_state_prior",
)
POVM_METRICS = (
    "mse_ost",
    "mse_povm_prior_ost",
    "mse_prior_ost",
    "excess_povm_prior",
    "excess_prior",
    "ratio_povm_prior",
    "ratio_prior",
)


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="my-data/prior_validity_mse")
parser.add_argument("--mode", choices=("state", "povm"), required=True)
parser.add_argument("-d", "--d-grid", nargs="+", type=int, default=[2, 4, 8])
parser.add_argument("-g", "--gamma-max", type=int, default=10_000)
parser.add_argument("--gamma-start", type=int, default=1)
parser.add_argument("--gamma-step", type=int, default=30)
parser.add_argument("-a", "--alpha-max", type=int, default=128)
parser.add_argument("--alpha-start", type=int, default=1)
parser.add_argument("--alpha-step", type=int, default=30)
parser.add_argument("--alpha", type=int, default=16)
parser.add_argument("--gamma", type=int, default=1000)
parser.add_argument("--shots", type=int, default=1_000_000)
parser.add_argument("--nseeds", type=int, default=30)
parser.add_argument("--base-seed", type=int, default=None)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--precision", type=str, default="float64")
parser.add_argument("--torch-threads", type=int, default=None)
args = parser.parse_args()

if any(d <= 1 for d in args.d_grid):
    raise ValueError("Unit-Haar-variance projector normalization requires d >= 2.")

device, rdtype, cdtype = torch_setup(
    device=args.device,
    precision=args.precision,
    threads=args.torch_threads,
    verbose=True,
)

out_dir = Path(args.folder).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

gamma_grid = logspace_int(args.gamma_start, args.gamma_max, args.gamma_step)
alpha_grid = logspace_int(args.alpha_start, args.alpha_max, args.alpha_step)
methods = STATE_METHODS if args.mode == "state" else POVM_METHODS
metric_names = STATE_METRICS if args.mode == "state" else POVM_METRICS
grid_column = "gamma" if args.mode == "state" else "alpha"
seeds = (
    [random_seed() for _ in range(args.nseeds)]
    if args.base_seed is None
    else [args.base_seed + seed_id for seed_id in range(args.nseeds)]
)

run_metadata = {
    "script": "prior_validity_mse.py",
    "mode": args.mode,
    "d_grid": args.d_grid,
    "gamma_start": args.gamma_start,
    "gamma_max": args.gamma_max,
    "gamma_step": args.gamma_step,
    "gamma_grid": gamma_grid.tolist(),
    "alpha_start": args.alpha_start,
    "alpha_max": args.alpha_max,
    "alpha_step": args.alpha_step,
    "alpha_grid": alpha_grid.tolist(),
    "alpha": args.alpha,
    "gamma": args.gamma,
    "fixed_alpha": args.alpha if args.mode == "state" else None,
    "fixed_gamma": args.gamma if args.mode == "povm" else None,
    "n_train_rule": "gamma * d**2",
    "n_out_rule": "alpha * d**2",
    "shots": args.shots,
    "nseeds": args.nseeds,
    "base_seed": args.base_seed,
    "precision": args.precision,
    "device": str(device),
    "seeds": seeds,
    "observable": "centered unit-Haar-variance projector",
    "training_probabilities": "infinite_stats",
    "mse_rule": "bias2 + variance / shots",
    "methods": methods,
    "metric_names": metric_names,
}
save_json(run_metadata, out_dir / "metadata.json")

metric_files: dict[int, dict[int, Path]] = {d: {} for d in args.d_grid}

for seed_id in range(args.nseeds):
    seed = seeds[seed_id]
    seed_dir = seed_run(out_dir, seed_id, seed)

    for d in args.d_grid:
        metric_path = seed_dir / f"{args.mode}_d_{d}.pt"

        if args.mode == "state":
            metrics, run_time = timed(
                state_prior_mse_grid,
                d=d,
                gamma_grid=gamma_grid,
                alpha=args.alpha,
                shots=args.shots,
                seed=seed,
                device=device,
                dtype=cdtype,
                accumulator_dtype=rdtype,
            )
        else:
            metrics, run_time = timed(
                povm_prior_mse_grid,
                d=d,
                alpha_grid=alpha_grid,
                gamma=args.gamma,
                shots=args.shots,
                seed=seed,
                device=device,
                dtype=cdtype,
                accumulator_dtype=rdtype,
            )

        save_pt(metrics, metric_path)
        metric_files[d][seed_id] = metric_path
        print(
            f"[seed {seed_id} / {args.nseeds}] "
            f"mode={args.mode} d={d} time={run_time:.2f}s"
        )

for d in args.d_grid:
    for metric_name in metric_names:
        save_metrics(
            metric_files[d],
            out_dir / "metrics" / args.mode / f"d_{d}" / f"{metric_name}.csv",
            methods=(metric_name,),
            metric_names=("value",),
            grid_column=grid_column,
        )
