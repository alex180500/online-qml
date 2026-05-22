import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from online_qml import (
    haar_metrics,
    ntrain_layers,
    random_seed,
    sample_data,
    sample_dm,
    save_json,
    save_pt,
    seed_all,
    timed,
    torch_setup,
)


MAX_SEED = 2**32 - 1


def parse_n_out_rule(rule: str, d: int) -> int:
    """Evaluate a simple n_out rule for one Hilbert-space dimension."""
    key = rule.strip().lower().replace(" ", "")
    if key in {"d2", "d**2", "d^2"}:
        return d * d
    if key in {"d3", "d**3", "d^3"}:
        return d * d * d
    if key.startswith("alpha_d2:"):
        alpha = float(key.split(":", 1)[1])
        return int(round(alpha * d * d))
    if key.endswith("*d2"):
        alpha = float(key[:-3])
        return int(round(alpha * d * d))
    if key.endswith("*d**2"):
        alpha = float(key[:-5])
        return int(round(alpha * d * d))
    if key.isdigit():
        return int(key)
    raise ValueError(
        "Unknown n_out rule. Use d2, d3, alpha_d2:<alpha>, <alpha>*d2, or a fixed integer."
    )


def seed_for_dimension(base_seed: int, d_index: int) -> int:
    """Return a deterministic seed for one dimension inside one seed run."""
    return int((base_seed + 1_000_003 * (d_index + 1)) % MAX_SEED)


def append_metric_records(
    records: list[dict[str, Any]],
    metrics: Any,
    *,
    seed_index: int,
    simulation_seed: int,
    d: int,
    n_out: int,
    n_train: int,
    shots: int,
    methods: list[str],
    metric_names: tuple[str, ...],
) -> None:
    """Append one-point dimension metrics to a long-form records list."""
    for metric_name in metric_names:
        for method in methods:
            key = f"{method}_{metric_name}"
            if key not in metrics.metrics:
                continue
            values = metrics.metrics[key].detach().cpu().reshape(-1)
            for value in values:
                records.append(
                    {
                        "seed_index": seed_index,
                        "simulation_seed": simulation_seed,
                        "d": d,
                        "n_out": n_out,
                        "n_train": n_train,
                        "shots": shots,
                        "method": method,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )


def save_dimension_metric_csvs(
    records: list[dict[str, Any]],
    out_dir: Path,
    methods: list[str],
    metric_names: tuple[str, ...],
) -> None:
    """Save long-form metrics and plot-ready summary CSVs."""
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No metric records were produced.")

    data = pd.DataFrame.from_records(records)
    data = data.sort_values(["d", "seed_index", "method", "metric"])
    data.to_csv(metrics_dir / "metrics_long.csv", index=False)

    index_columns = ["d", "n_out", "n_train", "shots"]
    base = data[index_columns].drop_duplicates().sort_values("d")

    for metric_name in metric_names:
        metric_data = data[data["metric"] == metric_name]
        summary = (
            metric_data.groupby([*index_columns, "method"], sort=False)["value"]
            .agg(
                median="median",
                q30=lambda values: values.quantile(0.3),
                q70=lambda values: values.quantile(0.7),
            )
            .reset_index()
        )

        output = base.copy()
        for method in methods:
            method_summary = summary[summary["method"] == method].drop(
                columns="method"
            )
            method_summary = method_summary.rename(
                columns={
                    "median": method,
                    "q30": f"{method}_q30",
                    "q70": f"{method}_q70",
                }
            )
            output = output.merge(method_summary, on=index_columns, how="left")

        columns = [*index_columns]
        for method in methods:
            columns.extend([method, f"{method}_q30", f"{method}_q70"])
        for column in columns:
            if column not in output:
                output[column] = pd.NA
        output[columns].to_csv(metrics_dir / f"{metric_name}.csv", index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, default=Path("my-tests/data/dim_sweep"))
parser.add_argument("--d-grid", nargs="+", type=int, default=[2, 3, 4])
parser.add_argument("--n-out-rule", type=str, default="d3")
parser.add_argument("--n-train-factor", type=int, default=100)
parser.add_argument("-s", "--shots", type=int, default=1)
parser.add_argument("--nseeds", type=int, default=3)
parser.add_argument(
    "--methods",
    nargs="+",
    default=["ost", "state_prior_ost", "pinv"],
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

base_seeds = [random_seed() for _ in range(args.nseeds)]
dimension_configs = []
for d in args.d_grid:
    n_out = parse_n_out_rule(args.n_out_rule, d)
    n_train = args.n_train_factor * d * d
    dimension_configs.append(
        {
            "d": d,
            "n_out": n_out,
            "n_train": n_train,
            "alpha": n_out / (d * d),
        }
    )

run_metadata = {
    "script": "dim_sweep.py",
    "experiment": "dim_sweep",
    "shots": args.shots,
    "d_grid": args.d_grid,
    "n_out_rule": args.n_out_rule,
    "n_train_rule": f"{args.n_train_factor}*d**2",
    "n_train_factor": args.n_train_factor,
    "nseeds": args.nseeds,
    "methods": args.methods,
    "precision": args.precision,
    "dimension_configs": dimension_configs,
    "seeds": [
        {
            "index": seed_index,
            "base_seed": base_seed,
            "dimension_seeds": {
                str(config["d"]): seed_for_dimension(base_seed, d_index)
                for d_index, config in enumerate(dimension_configs)
            },
        }
        for seed_index, base_seed in enumerate(base_seeds)
    ],
}
save_json(run_metadata, out_dir / "metadata.json")

metric_names = ("bias2", "variance")
records: list[dict[str, Any]] = []

for seed_id, base_seed in enumerate(base_seeds):
    seed_dir = out_dir / f"seed_{seed_id}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for d_index, config in enumerate(dimension_configs):
        d = int(config["d"])
        n_out = int(config["n_out"])
        n_train = int(config["n_train"])
        simulation_seed = seed_for_dimension(base_seed, d_index)
        seed_all(simulation_seed)

        data, sample_time = timed(
            sample_data,
            n_train,
            d=d,
            n_out=n_out,
            shots=args.shots,
            seed=simulation_seed,
            device=device,
            dtype=cdtype,
        )

        observable = sample_dm(1, d=d, device=device, dtype=cdtype).T
        train_grid = torch.tensor([n_train], device=device, dtype=torch.int64)
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
        save_pt(metrics, seed_dir / f"d_{d}_metrics.pt")
        append_metric_records(
            records,
            metrics,
            seed_index=seed_id,
            simulation_seed=simulation_seed,
            d=d,
            n_out=n_out,
            n_train=n_train,
            shots=args.shots,
            methods=args.methods,
            metric_names=metric_names,
        )

        print(
            f"[seed {seed_id} / {args.nseeds}] "
            f"d={d} n_out={n_out} n_train={n_train} shots={args.shots} "
            f"sample={sample_time:.2f}s layers={layer_time:.2f}s metrics={metric_time:.2f}s"
        )

save_dimension_metric_csvs(records, out_dir, args.methods, metric_names)
