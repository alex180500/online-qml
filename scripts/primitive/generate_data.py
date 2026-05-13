import argparse
from random import randint

import torch

from online_qml import (
    SimulationContext,
    SimulationData,
    get_complex_dtype,
    get_real_dtype,
    get_torch_device,
    sample_dm,
    sample_povm,
    save_simulation_data,
    shots_outcome,
)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dim", type=int, default=2)
parser.add_argument("-p", "--povm-size", type=int, required=True)
parser.add_argument("-s", "--max-shots", type=int, default=100)
parser.add_argument("-n", "--num-states", type=int, default=100000)
parser.add_argument("--seed", type=int)
parser.add_argument("-r", "--repeat", type=int, default=1)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--precision", type=str, default="float64", choices=["float32", "float64"]
)
args = parser.parse_args()

device = get_torch_device(args.device)
dtype = get_complex_dtype(get_real_dtype(args.precision))
for rep in range(args.repeat):
    seed = args.seed if args.seed is not None else randint(0, 10**9)
    path = (
        args.output
        if args.repeat == 1
        else args.output.replace(".pt", f"_{rep}.pt").replace(".npz", f"_{rep}.npz")
    )
    with SimulationContext(device, seed):
        states = sample_dm(args.num_states, d=args.dim, device=device, dtype=dtype)
        povm = sample_povm(args.povm_size, d=args.dim, device=device, dtype=dtype)
        outcomes = shots_outcome(povm, states, args.max_shots).to(torch.int16)
        data = SimulationData(
            states=states,
            povm=povm,
            outcomes=outcomes,
            seed=seed,
            d=args.dim,
            n_out=args.povm_size,
            metadata={"max_shots": args.max_shots, "num_states": args.num_states},
        )
        save_simulation_data(data, path)
        print(f"Saved {path}")
