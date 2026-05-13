import argparse

import torch

from online_qml import (
    get_complex_dtype,
    get_real_dtype,
    logspace_int,
    measurement_frame_distance_grid,
    save_data,
    state_frame_distance_grid,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--ntrains", nargs="+", type=int)
parser.add_argument("--nouts", nargs="+", type=int)
parser.add_argument("--train-start", type=int, default=10)
parser.add_argument("--train-stop", type=int, default=1000000)
parser.add_argument("--train-num", type=int, default=100)
parser.add_argument("--alphas", nargs="+", type=int)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--precision", type=str, default="float64", choices=["float32", "float64"]
)
args = parser.parse_args()

dtype = get_complex_dtype(get_real_dtype(args.precision))
train_grid = (
    torch.tensor(sorted(args.ntrains), dtype=torch.int64)
    if args.ntrains
    else logspace_int(args.train_start, args.train_stop, args.train_num)
)
if args.nouts:
    n_out_grid = torch.tensor(sorted(args.nouts), dtype=torch.int64)
else:
    alphas = args.alphas or [1, 2, 4, 8, 16, 32, 64]
    n_out_grid = torch.tensor(
        [a * args.dim * args.dim for a in alphas], dtype=torch.int64
    )
result = {
    "d": args.dim,
    "train_grid": train_grid,
    "n_out_grid": n_out_grid,
}
result.update(
    state_frame_distance_grid(args.dim, train_grid, device=args.device, dtype=dtype)
)
result.update(
    measurement_frame_distance_grid(
        args.dim, n_out_grid, device=args.device, dtype=dtype
    )
)
save_data(result, args.output)
print(f"Saved {args.output}")
