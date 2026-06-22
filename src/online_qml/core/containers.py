from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import torch


@dataclass
class SimulationData:
    """Raw simulated reservoir data.

    Args:
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        outcomes (torch.Tensor | None): Outcome matrix with shape (n_states, n_shots).
        seed (int | None): Random seed used to generate the data.
        metadata (dict): Extra simulation metadata.
    """

    states: torch.Tensor
    povm: torch.Tensor
    outcomes: torch.Tensor | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> int:
        return int(round(self.states.shape[0] ** 0.5))

    @property
    def n_out(self) -> int:
        return int(self.povm.shape[0])


@dataclass
class LayerResult:
    """Trained readout layers on one sweep grid.

    Args:
        layers (dict[str, torch.Tensor]): Readout layers with shape (..., n_obs, n_out).
        observable (torch.Tensor): Hermitian-conjugated flattened observable rows
            with shape (n_obs, d^2). With flattened matrices stored as columns,
            ``obs @ mat`` is the linear product.
        train_grid (torch.Tensor): Training-state grid with shape (n_train_grid,).
        shot_grid (torch.Tensor | None): Shot grid with shape (n_shot_grid,) or None.
        seed (int | None): Random seed used to generate the data.
        d (int | None): Hilbert-space dimension.
        n_out (int | None): Number of POVM outcomes.
        metadata (dict): Extra layer metadata.
    """

    layers: dict[str, torch.Tensor]
    observable: torch.Tensor
    train_grid: torch.Tensor
    shot_grid: torch.Tensor | None = None
    seed: int | None = None
    d: int | None = None
    n_out: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Metrics evaluated on a layer result.

    Args:
        metrics (dict[str, torch.Tensor]): Metric tensors with shape matching the layer grid.
        train_grid (torch.Tensor | None): Training-state grid with shape (n_train_grid,).
        shot_grid (torch.Tensor | None): Shot grid with shape (n_shot_grid,) or None.
        coords (dict): Named scalar or vector coordinates for metric values.
        seed (int | None): Random seed used to generate the data.
        d (int | None): Hilbert-space dimension.
        n_out (int | None): Number of POVM outcomes.
        metadata (dict): Extra metric metadata.
    """

    metrics: dict[str, torch.Tensor]
    train_grid: torch.Tensor | None = None
    shot_grid: torch.Tensor | None = None
    coords: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    d: int | None = None
    n_out: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def container_to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj) and not isinstance(obj, type):
        save_obj = asdict(obj)
    elif isinstance(obj, dict):
        save_obj = obj
    else:
        raise TypeError("save_pt expects a dataclass or dict.")
    return save_obj
