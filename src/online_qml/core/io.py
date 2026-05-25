import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .containers import container_to_dict


def save_pt(obj: Any, path: str | Path) -> None:
    """Save a dataclass or dictionary with torch.save."""
    save_obj = container_to_dict(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_obj, path)


def load_pt(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a dictionary with torch.load."""
    return torch.load(Path(path), map_location=device, weights_only=False)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Save JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON metadata."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_coord_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _coord_at(value: Any, index: int, size: int) -> Any:
    values = _as_coord_list(value)
    if len(values) == size:
        return values[index]
    if len(values) == 1:
        return values[0]
    raise ValueError(f"Coordinate has {len(values)} values, expected 1 or {size}.")


def save_metrics(
    metric_files: Mapping[Any, str | Path],
    path: str | Path,
    methods: Iterable[str],
    metric_names: Iterable[str],
    *,
    grid_key: str = "train_grid",
    grid_column: str = "n_train",
    fixed_columns: Mapping[str, Any] | None = None,
) -> None:
    """Write plot-ready metric summary CSV files."""
    fixed_columns = dict(fixed_columns or {})
    methods = tuple(methods)
    metric_names = tuple(metric_names)
    path = Path(path)
    if path.suffix:
        if len(metric_names) != 1:
            raise ValueError("A CSV path can be used only with one metric name.")
        output_paths = {metric_names[0]: path}
    else:
        path.mkdir(parents=True, exist_ok=True)
        output_paths = {
            metric_name: path / f"{metric_name}.csv" for metric_name in metric_names
        }

    grid: list[Any] | None = None
    coord_columns: list[str] | None = None
    records: list[dict[str, Any]] = []

    for metric_file in metric_files.values():
        metric_file = Path(metric_file)
        if not metric_file.exists():
            continue

        metric_data = load_pt(metric_file, device="cpu")
        coords = dict(metric_data.get("coords") or {})
        if grid_column in coords:
            grid_source = coords[grid_column]
        else:
            grid_source = metric_data[grid_key]
            coords.setdefault(grid_column, grid_source)

        file_grid = _as_coord_list(grid_source)
        if grid is None:
            grid = file_grid
            coord_columns = [
                name
                for name, value in coords.items()
                if name != grid_column
                and len(_as_coord_list(value)) in {1, len(grid)}
            ]
            for name in fixed_columns:
                if name not in coord_columns:
                    coord_columns.append(name)
        elif file_grid != grid:
            raise ValueError(
                f"Metric grid in {metric_file} does not match previous files."
            )

        metrics = metric_data["metrics"]
        extra_coords = {
            name: coords[name]
            for name in (coord_columns or [])
            if name in coords and name not in fixed_columns
        }
        for metric_name in metric_names:
            for method in methods:
                key = f"{method}_{metric_name}"
                if key not in metrics:
                    continue
                series = metrics[key].detach().cpu().numpy().reshape(-1)
                if len(series) != len(file_grid):
                    raise ValueError(
                        f"Metric '{key}' in {metric_file} has {len(series)} values, "
                        f"but '{grid_key}' has {len(file_grid)} values."
                    )
                for index, (grid_value, value) in enumerate(
                    zip(file_grid, series, strict=True)
                ):
                    records.append(
                        {
                            "_metric": metric_name,
                            "_method": method,
                            "_value": float(value),
                            grid_column: grid_value,
                            **{
                                name: _coord_at(coord, index, len(file_grid))
                                for name, coord in extra_coords.items()
                            },
                            **fixed_columns,
                        }
                    )

    if grid is None:
        raise ValueError("No metric files were found.")

    data = pd.DataFrame.from_records(records)
    index_columns = [grid_column, *(coord_columns or [])]

    for metric_name, output_path in output_paths.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [grid_column, *(coord_columns or [])]
        for method in methods:
            columns.extend([method, f"{method}_q30", f"{method}_q70"])

        output = data[index_columns].drop_duplicates().sort_values(grid_column)

        if not data.empty:
            metric_data = data[data["_metric"] == metric_name]
            summary = (
                metric_data.groupby([*index_columns, "_method"], sort=False)["_value"]
                .agg(
                    median="median",
                    q30=lambda values: values.quantile(0.3),
                    q70=lambda values: values.quantile(0.7),
                )
                .reset_index()
            )

            for method in methods:
                method_summary = summary[summary["_method"] == method].drop(
                    columns="_method"
                )
                method_summary = method_summary.rename(
                    columns={
                        "median": method,
                        "q30": f"{method}_q30",
                        "q70": f"{method}_q70",
                    }
                )
                output = output.merge(method_summary, on=index_columns, how="left")

        for column in columns:
            if column not in output:
                output[column] = pd.NA
        output[columns].to_csv(output_path, index=False)
