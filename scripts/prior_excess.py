"""Add metrics/mean-bias2.csv and .pt to every saved sweep below a folder.

Example:
    python prior_excess.py ~/Documents/datasets/online-qml/state_prior_proj

Reads only seed_*/metrics.pt; never overwrites those files or metrics/bias2.csv.
Means and sample standard deviations are computed in float64 with PyTorch.
The plotting scripts are independent: they only read the resulting CSV files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import warnings
from collections.abc import Mapping
from typing import Any

import torch

SEED_PATTERN = re.compile(r"seed_(\d+)$")
PRIOR_METHODS = ("state_prior_ost", "povm_prior_ost")
SUMMARY_NAME = "mean-bias2"


def _vector(value: Any, name: str) -> torch.Tensor:
    """Accept one sweep axis, possibly with singleton axes; do not average it."""
    value = torch.as_tensor(value).detach().cpu()
    if value.is_complex():
        raise ValueError(f"{name}: expected real values.")
    value = value.to(torch.float64).squeeze()
    if value.ndim == 0:
        value = value.reshape(1)
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name}: expected a nonempty 1-D sweep, got {tuple(value.shape)}.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name}: contains NaN/Inf; inspect the run before averaging.")
    return value


def _fixed(data: dict, metadata: dict, key: str) -> Any:
    """Read a scalar setting and check any repeated copies for consistency."""
    sources = (data.get("coords") or {}, data, data.get("metadata") or {}, metadata)
    candidates = [source[key] for source in sources if source.get(key) is not None]
    if key == "d" and metadata.get("dim") is not None:
        candidates.append(metadata["dim"])
    if key == "shots" and data.get("shot_grid") is not None:
        candidates.append(data["shot_grid"])
    values = []
    for value in candidates:
        if isinstance(value, str):
            values.append(value)
        else:
            tensor = torch.as_tensor(value)
            if tensor.numel() != 1:
                raise ValueError(f"{key}: expected a fixed scalar, got {tuple(tensor.shape)}.")
            values.append(tensor.item())
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Conflicting {key} settings: {values}.")
    return values[0]


def _load(path: Path, trusted_pickle: bool) -> dict:
    try:
        data = torch.load(path, map_location="cpu", weights_only=not trusted_pickle)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read {path}. Expected the dict saved by online_qml.save_pt. "
            "Use --trusted-pickle only for legacy files you trust."
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), Mapping):
        raise ValueError(f"{path}: expected a dictionary containing 'metrics'.")
    return data


def find_runs(folder: str | Path) -> list[Path]:
    """Find each directory with direct seed_N children, without pooling runs."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    runs = []
    for parent, directories, _ in os.walk(folder):
        seed_dirs = [name for name in directories if SEED_PATTERN.fullmatch(name)]
        if seed_dirs:
            runs.append(Path(parent))
        # Do not descend into raw seed payloads or existing summary folders.
        directories[:] = [
            name for name in directories
            if name not in seed_dirs
            and name not in {"metrics", "metrics_mean_std", "__pycache__"}
            and not name.startswith(".")
        ]
    if not runs:
        raise FileNotFoundError(f"No seed_N run directories found below {folder}.")
    return sorted(runs)


def _sweep(data: dict, metadata: dict) -> str:
    """Identify the ntrain/nout sweep from the supplied package's metadata."""
    script = Path(str(metadata.get("script", ""))).name
    hint = (data.get("metadata") or {}).get("sweep")
    if script == "ntrain_sweep.py" or hint == "ntrain":
        return "ntrain"
    if script == "nout_sweep.py" or hint == "nout":
        return "nout"
    coords = data.get("coords") or {}
    if coords.get("alpha") is not None:
        return "nout"
    if coords.get("n_out") is not None and torch.as_tensor(coords["n_out"]).numel() > 1:
        return "nout"
    if coords.get("n_train") is not None or data.get("train_grid") is not None:
        return "ntrain"
    raise ValueError("Cannot identify an ntrain or nout sweep from the saved coordinates.")


def _grid(data: dict, sweep: str, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords = data.get("coords") or {}
    alternatives = []
    if sweep == "ntrain":
        alternatives = [coords.get("n_train"), data.get("train_grid")]
    else:
        alternatives = [coords.get("n_out")]
        if coords.get("alpha") is not None:
            alternatives.append(_vector(coords["alpha"], "alpha") * d**2)
    grids = [_vector(value, "grid") for value in alternatives if value is not None]
    if not grids:
        raise ValueError(f"Missing {sweep} grid.")
    grid = grids[0]
    for other in grids[1:]:
        if grid.shape != other.shape or not torch.allclose(grid, other, atol=1e-8, rtol=0):
            raise ValueError("Conflicting grid coordinates in the saved file.")
    if bool((grid <= 0).any()) or not torch.allclose(grid, grid.round(), atol=1e-8, rtol=0):
        raise ValueError("Training/outcome counts must be positive integers.")
    grid = grid.round().to(torch.int64)
    order = torch.argsort(grid)
    grid = grid[order]
    if grid.unique().numel() != grid.numel():
        raise ValueError("Duplicate grid points in a seed file.")
    return grid, order


def _stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """values has shape (n_seeds, n_grid); keep the grid axis intact."""
    std, mean = torch.std_mean(values, dim=0, correction=1)
    sem = std / math.sqrt(values.shape[0])
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("Mean/standard-deviation overflow; inspect large numerical outliers.")
    return mean, std, sem


def summarize_run(run_dir: str | Path, *, trusted_pickle: bool = False) -> Path:
    """Read all seed files in ONE run and refresh its mean-bias2 summaries."""
    run_dir = Path(run_dir).expanduser().resolve()
    seed_dirs = sorted(
        (path for path in run_dir.iterdir() if path.is_dir() and SEED_PATTERN.fullmatch(path.name)),
        key=lambda path: int(SEED_PATTERN.fullmatch(path.name).group(1)),
    )
    if len(seed_dirs) < 2:
        raise ValueError(f"{run_dir}: at least two seeds are needed for sample standard deviation.")
    files = [path / "metrics.pt" for path in seed_dirs]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing seed metrics (not silently excluded):\n" + "\n".join(missing))
    metadata_file = run_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_file}: expected a JSON object.")
    if metadata.get("nseeds") not in (None, len(files)):
        warnings.warn(f"{run_dir}: metadata specifies {metadata['nseeds']} seeds; using {len(files)} saved seeds.")

    rows: dict[str, list[torch.Tensor]] = {}
    reference_grid = reference_settings = None
    random_seeds = []
    for path in files:
        data = _load(path, trusted_pickle)
        sweep = _sweep(data, metadata)
        dimension = _fixed(data, metadata, "d")
        if dimension is None or dimension <= 1 or dimension != int(dimension):
            raise ValueError(f"{path}: missing or invalid Hilbert-space dimension.")
        d = int(dimension)
        fixed_key = "n_out" if sweep == "ntrain" else "n_train"
        shots = _fixed(data, metadata, "shots")
        fixed_count = _fixed(data, metadata, fixed_key)
        for name, value in (("shots", shots), (fixed_key, fixed_count)):
            if value is not None and (value <= 0 or value != int(value)):
                raise ValueError(f"{path}: {name} must be a positive integer.")
        settings = (sweep, d, shots, fixed_count, _fixed(data, metadata, "observable"))
        grid, order = _grid(data, sweep, d)
        method_names = sorted(key[:-6] for key in data["metrics"] if key.endswith("_bias2"))
        if "ost" not in method_names or not any(name in method_names for name in PRIOR_METHODS):
            raise ValueError(f"{path}: need ost_bias2 and at least one prior method's bias2.")
        if reference_grid is None:
            reference_grid, reference_settings = grid, settings
            rows = {name: [] for name in ["ost", *(m for m in method_names if m != "ost")]}
        elif settings != reference_settings or not torch.equal(grid, reference_grid):
            raise ValueError(f"{path}: grid or fixed settings differ between seeds; refusing to pool.")
        if set(method_names) != set(rows):
            raise ValueError(f"{path}: bias2 methods differ between seeds.")
        for method in rows:
            values = _vector(data["metrics"][f"{method}_bias2"], f"{path}: {method}_bias2")
            if values.numel() != grid.numel():
                raise ValueError(f"{path}: {method} length differs from the sweep grid.")
            rows[method].append(values[order])
        seed = data.get("seed")
        random_seeds.append(None if seed is None else int(torch.as_tensor(seed).item()))

    known = [seed for seed in random_seeds if seed is not None]
    if len(set(known)) != len(known):
        raise ValueError(f"{run_dir}: duplicate random seeds; copied runs must not be counted twice.")

    sweep, d, shots, fixed_count, observable = reference_settings
    size = reference_grid.numel()
    columns = {"d": torch.full((size,), d, dtype=torch.int64)}
    raw_key = "n_train" if sweep == "ntrain" else "n_out"
    fixed_key = "n_out" if sweep == "ntrain" else "n_train"
    columns[raw_key] = reference_grid
    if fixed_count is not None:
        columns[fixed_key] = torch.full((size,), int(fixed_count), dtype=torch.int64)
    for count, ratio in (("n_train", "gamma"), ("n_out", "alpha")):
        if count in columns:
            columns[ratio] = columns[count].to(torch.float64) / d**2
    if shots is not None:
        columns["shots"] = torch.full((size,), int(shots), dtype=torch.int64)
    columns["n_seeds"] = torch.full((size,), len(files), dtype=torch.int64)

    by_seed = {method: torch.stack(values) for method, values in rows.items()}
    excess_by_seed = {}
    for method, values in by_seed.items():
        columns[method], columns[f"{method}_std"], columns[f"{method}_sem"] = _stats(values)
    for method in PRIOR_METHODS:
        if method not in by_seed:
            continue
        # Pair BEFORE averaging, so SD includes covariance between the methods.
        excess = by_seed[method] - by_seed["ost"]
        excess_by_seed[method] = excess
        name = f"{method}_excess"
        columns[name], columns[f"{name}_std"], columns[f"{name}_sem"] = _stats(excess)
        columns[f"{name}_negative_count"] = (excess < 0).sum(dim=0)

    folder = run_dir / "metrics"
    folder.mkdir(exist_ok=True)
    csv_path, pt_path = folder / f"{SUMMARY_NAME}.csv", folder / f"{SUMMARY_NAME}.pt"
    summary = {
        "format": "online_qml.mean_bias2.v1", "sweep": sweep, "metric": "bias2",
        "d": d, "observable": observable, "n_seeds": len(files), "std_correction": 1,
        "columns": columns, "metrics_by_seed": by_seed, "excess_by_seed": excess_by_seed,
        "seed_ids": [int(SEED_PATTERN.fullmatch(path.name).group(1)) for path in seed_dirs],
        "random_seeds": random_seeds, "source_files": [str(path) for path in files],
        "source_mtime_ns": [path.stat().st_mtime_ns for path in files],
        "mean_definition": "arithmetic mean over seeds at each grid point; excess paired before averaging",
    }
    # Replace only our own generated summaries, atomically per output file.
    csv_tmp, pt_tmp = csv_path.with_suffix(".csv.tmp"), pt_path.with_suffix(".pt.tmp")
    try:
        torch.save(summary, pt_tmp)
        with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for index in range(size):
                writer.writerow([value[index].item() for value in columns.values()])
        pt_tmp.replace(pt_path)
        csv_tmp.replace(csv_path)
    finally:
        csv_tmp.unlink(missing_ok=True)
        pt_tmp.unlink(missing_ok=True)
    print(f"{run_dir}: {len(files)} seeds, {size} points -> {csv_path}")
    return csv_path


def process_folder(folder: str | Path, *, trusted_pickle: bool = False) -> list[Path]:
    """Process all runs recursively; each dimension/shot run stays separate."""
    outputs, errors = [], []
    for run in find_runs(folder):
        try:
            outputs.append(summarize_run(run, trusted_pickle=trusted_pickle))
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            print(f"ERROR: {run}: {exc}", file=sys.stderr)
            errors.append(run)
    if errors:
        raise RuntimeError(
            f"{len(errors)} run(s) failed; {len(outputs)} succeeded. "
            "Failed runs were not refreshed: do not use older summaries for them."
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Dataset root, dimension folder, or individual shots_N run.")
    parser.add_argument("--trusted-pickle", action="store_true", help="Allow unrestricted loading ONLY for trusted legacy files.")
    args = parser.parse_args()
    try:
        outputs = process_folder(args.folder, trusted_pickle=args.trusted_pickle)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Done: refreshed {len(outputs)} mean-bias2.csv files (and matching .pt files).")


if __name__ == "__main__":
    main()
