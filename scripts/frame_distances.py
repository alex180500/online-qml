import argparse
from pathlib import Path
from online_qml import *

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, default=Path("my-tests/data/frame_distances"))
parser.add_argument("-d", "--d-grid", nargs="+", type=int, default=[2, 4, 8])
parser.add_argument("-g", "--gamma-max", type=int, default=10_000)
parser.add_argument("--gamma-start", type=int, default=1)
parser.add_argument("--gamma-step", type=int, default=20)
parser.add_argument("-a", "--alpha-max", type=int, default=64)
parser.add_argument("--alpha-start", type=int, default=1)
parser.add_argument("--alpha-step", type=int, default=20)
parser.add_argument("--nseeds", type=int, default=20)
parser.add_argument("--state-only", action="store_true")
parser.add_argument("--povm-only", action="store_true")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--precision", type=str, default="float64")
parser.add_argument("--torch-threads", type=int, default=None)
args = parser.parse_args()

if args.state_only and args.povm_only:
    raise ValueError("Choose at most one of --state-only and --povm-only.")

run_state = not args.povm_only
run_povm = not args.state_only

device, rdtype, cdtype = torch_setup(
    device=args.device,
    precision=args.precision,
    threads=args.torch_threads,
    verbose=True,
)

out_dir = args.folder
out_dir.mkdir(parents=True, exist_ok=True)

gamma_grid = logspace_int(args.gamma_start, args.gamma_max, args.gamma_step)
alpha_grid = logspace_int(args.alpha_start, args.alpha_max, args.alpha_step)

metric_names = (
    "rel_op",
    "rel_fro",
    "lambda_min",
    "lambda_max",
    "condition",
)

run_metadata = {
    "script": "frame_distances.py",
    "d_grid": args.d_grid,
    "gamma_start": args.gamma_start,
    "gamma_max": args.gamma_max,
    "gamma_step": args.gamma_step,
    "gamma_grid": gamma_grid.tolist(),
    "n_train_rule": "gamma * d**2",
    "alpha_start": args.alpha_start,
    "alpha_max": args.alpha_max,
    "alpha_step": args.alpha_step,
    "alpha_grid": alpha_grid.tolist(),
    "n_out_rule": "alpha * d**2",
    "nseeds": args.nseeds,
    "precision": args.precision,
    "run_state": run_state,
    "run_povm": run_povm,
    "metric_names": metric_names,
    "seeds": [random_seed() for _ in range(args.nseeds)],
}
save_json(run_metadata, out_dir / "metadata.json")

state_metric_files: dict[int, dict[int, Path]] = {d: {} for d in args.d_grid}
povm_metric_files: dict[int, dict[int, Path]] = {d: {} for d in args.d_grid}

for seed_id in range(args.nseeds):
    seed = run_metadata["seeds"][seed_id]
    seed_dir = seed_run(out_dir, seed_id, seed)

    for d in args.d_grid:
        train_grid = gamma_grid * d * d
        n_out_grid = alpha_grid * d * d

        state_path = seed_dir / f"state_d_{d}.pt"
        povm_path = seed_dir / f"povm_d_{d}.pt"

        state_time = 0.0
        povm_time = 0.0

        if run_state:
            state_metrics, state_time = timed(
                state_frame_distance_grid,
                d=d,
                train_grid=train_grid,
                gamma_grid=gamma_grid,
                seed=seed,
                device=device,
                dtype=cdtype,
            )
            save_pt(state_metrics, state_path)
            state_metric_files[d][seed_id] = state_path

        if run_povm:
            povm_metrics, povm_time = timed(
                povm_frame_distance_grid,
                d=d,
                n_out_grid=n_out_grid,
                alpha_grid=alpha_grid,
                seed=seed,
                device=device,
                dtype=cdtype,
            )
            save_pt(povm_metrics, povm_path)
            povm_metric_files[d][seed_id] = povm_path

        print(
            f"[seed {seed_id} / {args.nseeds}] "
            f"d={d} state={state_time:.2f}s povm={povm_time:.2f}s"
        )

if run_state:
    for d in args.d_grid:
        save_metrics(
            state_metric_files[d],
            out_dir / "metrics" / "state" / f"d_{d}",
            methods=("state",),
            metric_names=metric_names,
            grid_column="gamma",
        )

if run_povm:
    for d in args.d_grid:
        save_metrics(
            povm_metric_files[d],
            out_dir / "metrics" / "povm" / f"d_{d}",
            methods=("povm",),
            metric_names=metric_names,
            grid_column="alpha",
        )
