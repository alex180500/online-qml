import argparse

import torch

from online_qml import (
    get_complex_dtype,
    get_real_dtype,
    load_simulation_data,
    logspace_int,
    make_layers_ntrain_grid,
    make_layers_shot_grid,
    sample_dm,
    sample_traceless_operator,
    save_data,
)


def _grid(
    values: list[int] | None, start: int | None, stop: int | None, num: int | None
):
    if values:
        return torch.tensor(sorted(values), dtype=torch.int64)
    if start is None or stop is None or num is None:
        raise ValueError("Provide explicit values or start/stop/num.")
    return logspace_int(start, stop, num)


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--sweep", type=str, choices=["ntrain", "shots"], required=True)
parser.add_argument("--shots", nargs="+", type=int)
parser.add_argument("--ntrains", nargs="+", type=int)
parser.add_argument("--grid-start", type=int)
parser.add_argument("--grid-stop", type=int)
parser.add_argument("--grid-num", type=int)
parser.add_argument("--fixed-shots", type=int)
parser.add_argument("--fixed-ntrain", type=int)
parser.add_argument(
    "--methods", nargs="+", default=["ost", "aost", "prior_ost", "prior_aost", "pinv"]
)
parser.add_argument("--traceless", action="store_true")
parser.add_argument("--pinv-tol", type=float, default=1e-10)
parser.add_argument("--ridge-alpha", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--precision", type=str, default="float64", choices=["float32", "float64"]
)
args = parser.parse_args()

real_dtype = get_real_dtype(args.precision)
complex_dtype = get_complex_dtype(real_dtype)
data = load_simulation_data(args.input, device=args.device, dtype=complex_dtype)
if args.traceless:
    observable = sample_traceless_operator(
        1, d=data.d, device=data.states.device, dtype=complex_dtype
    ).T
else:
    observable = sample_dm(
        1, d=data.d, device=data.states.device, dtype=complex_dtype
    ).T
shadow_methods = [
    m for m in args.methods if m in {"ost", "aost", "prior_ost", "prior_aost"}
]
linear_methods = [m for m in args.methods if m in {"pinv", "ridge"}]
if args.sweep == "ntrain":
    train_grid = _grid(args.ntrains, args.grid_start, args.grid_stop, args.grid_num)
    result = make_layers_ntrain_grid(
        data,
        observable,
        train_grid=train_grid,
        n_shots=args.fixed_shots or 1,
        shadow_methods=shadow_methods,
        linear_methods=linear_methods,
        pinv_tol=args.pinv_tol,
        ridge_alpha=args.ridge_alpha,
        dtype=real_dtype,
    )
else:
    shot_grid = _grid(args.shots, args.grid_start, args.grid_stop, args.grid_num)
    result = make_layers_shot_grid(
        data,
        observable,
        shot_grid=shot_grid,
        n_train=args.fixed_ntrain or 1000,
        shadow_methods=shadow_methods,
        linear_methods=linear_methods,
        pinv_tol=args.pinv_tol,
        ridge_alpha=args.ridge_alpha,
        dtype=real_dtype,
    )
save_data(result, args.output)
print(f"Saved {args.output}")
