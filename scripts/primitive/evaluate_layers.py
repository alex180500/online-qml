import argparse

from online_qml import (
    LayerResult,
    evaluate_layer_result_haar,
    load_data,
    load_simulation_data,
    save_data,
)

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

layer_dict = load_data(args.layers, device=args.device)
data = load_simulation_data(args.data, device=args.device)
result = LayerResult(
    layers=layer_dict["layers"],
    d=int(layer_dict["d"]),
    n_out=int(layer_dict["n_out"]),
    shot_grid=layer_dict["shot_grid"],
    train_grid=layer_dict["train_grid"],
    seed=layer_dict.get("seed"),
    observable=layer_dict["observable"],
    metadata=layer_dict.get("metadata", {}),
)
metrics = evaluate_layer_result_haar(result, data.povm)
save_data(metrics, args.output)
print(f"Saved {args.output}")
