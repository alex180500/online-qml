#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from online_qml import (
    SimulationContext,
    SimulationData,
    evaluate_layer_result_haar,
    get_complex_dtype,
    get_real_dtype,
    get_torch_device,
    load_simulation_data,
    logspace_int,
    make_layers_ntrain_grid,
    sample_dm,
    sample_povm,
    sample_traceless_operator,
    save_data,
    save_simulation_data,
    shots_outcome,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the local n_train sweep: generate data, train readouts, "
            "and evaluate Haar bias/MSE."
        )
    )
    parser.add_argument("--data-root", type=Path, default=repo_root / "data")
    parser.add_argument("--run-name", type=str, default="ntrain_sweep")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--povm-size", type=int, default=16)
    parser.add_argument("--max-shots", type=int, default=100)
    parser.add_argument("--max-ntrain", type=int, default=50000)
    parser.add_argument("--grid-start", type=int, default=1)
    parser.add_argument("--grid-stop", type=int)
    parser.add_argument("--grid-num", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ost", "aost", "prior_ost", "prior_aost", "pinv", "ridge"],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--precision", type=str, default="float64", choices=["float32", "float64"]
    )
    parser.add_argument("--traceless", action="store_true")
    parser.add_argument("--pinv-tol", type=float, default=1e-10)
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate raw seed data even if the raw .pt file already exists.",
    )
    return parser.parse_args()


def generate_simulation_data(
    *,
    dim: int,
    povm_size: int,
    max_shots: int,
    num_states: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SimulationData:
    with SimulationContext(device, seed):
        states = sample_dm(num_states, d=dim, device=device, dtype=dtype)
        povm = sample_povm(povm_size, d=dim, device=device, dtype=dtype)
        outcomes = shots_outcome(povm, states, max_shots).to(torch.int16)
    return SimulationData(
        states=states,
        povm=povm,
        outcomes=outcomes,
        seed=seed,
        d=dim,
        n_out=povm_size,
        metadata={"max_shots": max_shots, "num_states": num_states},
    )


def sample_observable(
    *, dim: int, seed: int, traceless: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # Fixed offset keeps the observable deterministic while independent from data sampling.
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    torch.manual_seed(seed + 10_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 10_000)
    try:
        if traceless:
            return sample_traceless_operator(1, d=dim, device=device, dtype=dtype).T
        return sample_dm(1, d=dim, device=device, dtype=dtype).T
    finally:
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def main() -> None:
    args = parse_args()
    grid_stop = args.grid_stop or args.max_ntrain
    if max(args.shots) > args.max_shots:
        raise ValueError(
            f"--max-shots ({args.max_shots}) must be >= the largest --shots value "
            f"({max(args.shots)})."
        )
    if grid_stop > args.max_ntrain:
        raise ValueError(
            f"--grid-stop ({grid_stop}) must be <= --max-ntrain ({args.max_ntrain})."
        )
    out_dir = args.data_root / args.run_name
    real_dtype = get_real_dtype(args.precision)
    complex_dtype = get_complex_dtype(real_dtype)
    device = get_torch_device(args.device)
    train_grid = logspace_int(args.grid_start, grid_stop, args.grid_num)
    shadow_methods = [
        method
        for method in args.methods
        if method in {"ost", "aost", "prior_ost", "prior_aost"}
    ]
    linear_methods = [
        method for method in args.methods if method in {"pinv", "ridge"}
    ]
    unknown_methods = sorted(set(args.methods) - set(shadow_methods) - set(linear_methods))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")

    raw_dir = out_dir / "raw"
    layer_dir = out_dir / "layers"
    metric_dir = out_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    layer_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing outputs to: {out_dir}")
    print(
        f"Config: d={args.dim}, n_out={args.povm_size}, "
        f"max_shots={args.max_shots}, max_ntrain={args.max_ntrain}"
    )
    print(
        f"Grid: {args.grid_num} log-spaced values "
        f"from {args.grid_start} to {grid_stop}"
    )
    print("Seeds:", " ".join(map(str, args.seeds)))
    print("Shots:", " ".join(map(str, args.shots)))
    print("Methods:", " ".join(args.methods))

    for seed in args.seeds:
        data_path = raw_dir / f"seed_{seed}.pt"

        if args.regenerate_data or not data_path.exists():
            print(f"\n[seed {seed}] generating data")
            data = generate_simulation_data(
                dim=args.dim,
                povm_size=args.povm_size,
                max_shots=args.max_shots,
                num_states=args.max_ntrain,
                seed=seed,
                device=device,
                dtype=complex_dtype,
            )
            save_simulation_data(data, str(data_path))
            print(f"Saved {data_path}")
        else:
            print(f"\n[seed {seed}] reusing {data_path}")
            data = load_simulation_data(data_path, device=device, dtype=complex_dtype)

        observable = sample_observable(
            dim=data.d,
            seed=seed,
            traceless=args.traceless,
            device=device,
            dtype=complex_dtype,
        )
        for shots in args.shots:
            layer_path = layer_dir / f"seed_{seed}_shots_{shots}.pt"
            metric_path = metric_dir / f"seed_{seed}_shots_{shots}.pt"

            print(f"[seed {seed}, shots {shots}] training readouts")
            layer_result = make_layers_ntrain_grid(
                data,
                observable,
                train_grid=train_grid,
                n_shots=shots,
                shadow_methods=shadow_methods,
                linear_methods=linear_methods,
                pinv_tol=args.pinv_tol,
                ridge_alpha=args.ridge_alpha,
                dtype=real_dtype,
            )
            save_data(layer_result, str(layer_path))
            print(f"Saved {layer_path}")

            print(f"[seed {seed}, shots {shots}] evaluating Haar bias/MSE")
            metric_result = evaluate_layer_result_haar(layer_result, data.povm)
            save_data(metric_result, str(metric_path))
            print(f"Saved {metric_path}")

    print(f"\nDone. Metrics are in: {metric_dir}")


if __name__ == "__main__":
    main()
