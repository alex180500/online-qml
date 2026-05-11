from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import time
from typing import Any

import numpy as np
import torch

# =====================================
# DATA CLASSES
# =====================================


@dataclass
class SimulationData:
    """Container for simulated quantum reservoir data.

    Args:
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        outcomes (torch.Tensor | None): Outcome matrix with shape (n_states, n_shots).
        seed (int | None): Random seed used for the simulation.
        d (int): Hilbert-space dimension.
        n_out (int): Number of POVM outcomes.
        metadata (dict): Extra run information.
    """

    states: torch.Tensor
    povm: torch.Tensor
    outcomes: torch.Tensor | None
    seed: int | None
    d: int
    n_out: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerResult:
    """Container for trained readout layers.

    Args:
        layers (dict[str, torch.Tensor]): Readout layers with shape (..., n_obs, n_out).
        d (int): Hilbert-space dimension.
        n_out (int): Number of POVM outcomes.
        shot_grid (torch.Tensor): Shot values with shape (n_shots_grid,).
        train_grid (torch.Tensor): Training-state values with shape (n_train_grid,).
        seed (int | None): Random seed used for the simulation.
        observable (torch.Tensor): Observables with shape (n_obs, d^2).
        metadata (dict): Extra run information.
    """

    layers: dict[str, torch.Tensor]
    d: int
    n_out: int
    shot_grid: torch.Tensor
    train_grid: torch.Tensor
    seed: int | None
    observable: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Container for layer metrics.

    Args:
        metrics (dict[str, torch.Tensor]): Metric tensors with shape (...,).
        d (int): Hilbert-space dimension.
        n_out (int): Number of POVM outcomes.
        shot_grid (torch.Tensor): Shot values with shape (n_shots_grid,).
        train_grid (torch.Tensor): Training-state values with shape (n_train_grid,).
        seed (int | None): Random seed used for the simulation.
        metadata (dict): Extra run information.
    """

    metrics: dict[str, torch.Tensor]
    d: int
    n_out: int
    shot_grid: torch.Tensor
    train_grid: torch.Tensor
    seed: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


# =====================================
# DTYPES AND DEVICES
# =====================================


def get_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype associated with a real dtype.

    Args:
        real_dtype (torch.dtype): Real dtype, either torch.float32 or torch.float64.

    Returns:
        torch.dtype: Complex dtype, either torch.complex64 or torch.complex128.
    """
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise ValueError("real_dtype must be torch.float32 or torch.float64.")


def get_real_dtype(precision: str | torch.dtype) -> torch.dtype:
    """Return a real torch dtype from a precision value.

    Args:
        precision (str | torch.dtype): Precision name or torch dtype.

    Returns:
        torch.dtype: Real dtype.
    """
    if isinstance(precision, torch.dtype):
        if precision in (torch.float32, torch.float64):
            return precision
        raise ValueError("precision dtype must be torch.float32 or torch.float64.")
    if precision == "float32":
        return torch.float32
    if precision == "float64":
        return torch.float64
    raise ValueError("precision must be 'float32' or 'float64'.")


def get_torch_device(device_str: str, silent: bool = False) -> torch.device:
    """Return a torch device with CUDA fallback.

    Args:
        device_str (str): Device string, e.g. 'cpu' or 'cuda:0'.
        silent (bool): If True, do not print device information.

    Returns:
        torch.device: Selected torch device.
    """
    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
        properties = torch.cuda.get_device_properties(device)
    elif device_str == "cpu":
        device = torch.device("cpu")
        properties = None
    else:
        print(f"Warning: device '{device_str}' is not available. Using CPU.")
        device = torch.device("cpu")
        properties = None

    if not silent:
        if device.type == "cuda" and properties is not None:
            print(
                f"- GPU - name: {properties.name}, "
                f"CUDA version: {torch.version.cuda}, "
                f"memory: {properties.total_memory // (1024**3)} GB"
            )
        print(
            f"- CPU - OS cores: {os.cpu_count()}, torch threads: {torch.get_num_threads()}"
        )
        print(f"Using '{device.type}' as computation device.")
    return device


# =====================================
# RANGES AND PARSERS
# =====================================


def logspace_int(start: int, stop: int, num: int) -> torch.Tensor:
    """Generate unique integer values on a logarithmic grid.

    Args:
        start (int): First value, must be positive.
        stop (int): Last value.
        num (int): Number of values.

    Returns:
        torch.Tensor: Integer grid with shape (num,).
    """
    if start <= 0 or stop < start:
        raise ValueError("Require 0 < start <= stop.")
    if num > (stop - start + 1):
        raise ValueError(
            f"Cannot generate {num} unique integers between {start} and {stop}."
        )
    vals = np.logspace(np.log10(start), np.log10(stop), num)
    res = torch.zeros(num, dtype=torch.int64)
    current = int(start)
    res[0] = current
    for idx in range(1, num):
        candidate = int(np.round(vals[idx]))
        current = max(current + 1, candidate)
        res[idx] = current
    return res


def parse_tol(value: str) -> float | int:
    """Parse a pseudoinverse tolerance or truncation rank.

    Args:
        value (str): Float tolerance or integer rank.

    Returns:
        float | int: Parsed value.
    """
    try:
        return int(value)
    except ValueError:
        return float(value)


# =====================================
# IO
# =====================================


def _to_save_dict(data: Any) -> dict[str, Any]:
    if hasattr(data, "__dataclass_fields__"):
        return asdict(data)
    if isinstance(data, dict):
        return data
    raise TypeError("data must be a dataclass or dict.")


def save_data(data: Any, filepath: str) -> None:
    """Save a dictionary or dataclass to .pt or .npz.

    Args:
        data: Dictionary or dataclass to save.
        filepath (str): Output path ending in .pt or .npz.
    """
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    save_dict = _to_save_dict(data)
    if filepath.endswith(".pt"):
        torch.save(save_dict, filepath)
    elif filepath.endswith(".npz"):
        np_dict = {}
        for key, val in save_dict.items():
            if isinstance(val, torch.Tensor):
                np_dict[key] = val.detach().cpu().numpy()
            elif isinstance(val, (str, int, float, bool, type(None))):
                np_dict[key] = val
            else:
                np_dict[key] = np.array(val, dtype=object)
        np.savez_compressed(filepath, **np_dict)
    else:
        raise ValueError("filepath must end in .pt or .npz.")


def load_data(filepath: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a dictionary from .pt or .npz.

    Args:
        filepath (str): Input path ending in .pt or .npz.
        device (torch.device | str): Device for tensor outputs.

    Returns:
        dict: Loaded data.
    """
    if filepath.endswith(".pt"):
        return torch.load(filepath, map_location=device, weights_only=False)
    if filepath.endswith(".npz"):
        np_data = np.load(filepath, allow_pickle=True)
        out: dict[str, Any] = {}
        for key in np_data.files:
            val = np_data[key]
            if val.shape == () and val.dtype == object:
                out[key] = val.item()
            elif np.issubdtype(val.dtype, np.number):
                out[key] = torch.from_numpy(val).to(device)
            else:
                out[key] = val
        return out
    raise ValueError("filepath must end in .pt or .npz.")


def save_simulation_data(data: SimulationData, filepath: str) -> None:
    """Save simulation data to .pt or .npz.

    Args:
        data (SimulationData): Simulation data with states (d^2, n_states), povm (n_out, d^2), outcomes (n_states, n_shots) or None.
        filepath (str): Output path ending in .pt or .npz.
    """
    save_data(data, filepath)


def load_simulation_data(
    filepath: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> SimulationData:
    """Load simulation data from .pt or .npz.

    Args:
        filepath (str): Input path ending in .pt or .npz.
        device (torch.device | str): Device for tensor outputs.
        dtype (torch.dtype | None): Complex dtype for states and POVM.

    Returns:
        SimulationData: Simulation data with states (d^2, n_states), povm (n_out, d^2), outcomes (n_states, n_shots) or None.
    """
    data = load_data(filepath, device=device)
    states = data["states"]
    povm = data["povm"]
    if dtype is not None:
        states = states.to(dtype=dtype)
        povm = povm.to(dtype=dtype)
    outcomes = data.get("outcomes")
    d = int(data.get("d", int(states.shape[0] ** 0.5)))
    n_out = int(data.get("n_out", povm.shape[0]))
    return SimulationData(
        states=states,
        povm=povm,
        outcomes=outcomes,
        seed=data.get("seed"),
        d=d,
        n_out=n_out,
        metadata=data.get("metadata", {}),
    )


# =====================================
# CONTEXT
# =====================================


class SimulationContext:
    """Context manager for seeding and timing simulations.

    Args:
        device (torch.device | str): Device to synchronize if CUDA.
        seed (int | None): Random seed.
    """

    def __init__(self, device: torch.device | str = "cpu", seed: int | None = None):
        self.device = torch.device(device)
        self.seed = seed
        self.duration = 0.0
        self.start = 0.0

    def __enter__(self):
        if self.seed is not None:
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)
            else:
                torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.duration = time.perf_counter() - self.start
        print(f"Seed: {self.seed} - completed in {self.duration:.2f} seconds.")
