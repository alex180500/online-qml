from collections.abc import Iterable, Mapping, Sequence
from typing import Any
import torch

from .core.containers import LayerResult, MetricResult, SimulationData
from .core.methods import split_methods as _split_methods
from .estimators import (
    LinearReadoutEstimator,
    RunningOutcomeStats,
    ShadowReadoutEstimator,
)
from .evaluation import evaluate_layers_haar, fit_beta_coefficients
from .quantum import (
    as_observable_matrix,
    frame_distance_summary,
    get_observables,
    haar_state_frame,
    measurement_frame,
    naimark_measurement_frame_prior,
    sample_dm,
    sample_povm,
    shots_outcome,
    state_frame,
)

# ----- SIMULATION DATA -----


def sample_data(
    n_states: int,
    d: int,
    n_out: int,
    shots: int,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> SimulationData:
    """Sample states, a random POVM, finite-shot outcomes, and package them as simulation data.

    Args:
        n_states (int): Number of states to sample.
        d (int): Hilbert-space dimension.
        n_out (int): Number of POVM outcomes.
        shots (int): Number of shots per state.
        seed (int | None): Seed stored in the returned data.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex dtype.

    Returns:
        SimulationData: Sampled simulation data.
    """
    states = sample_dm(n_states, d=d, device=device, dtype=dtype)
    povm = sample_povm(n_out, d=d, device=device, dtype=dtype)
    outcomes = shots_outcome(povm, states, shots)
    return SimulationData(
        states=states,
        povm=povm,
        outcomes=outcomes,
        seed=seed,
        metadata={"shots": shots},
    )


# ----- LAYER GENERATION -----


def ntrain_layers(
    data: SimulationData,
    observable: torch.Tensor,
    train_grid: torch.Tensor,
    n_shots: int,
    methods: Iterable[str],
    state_prior_frame: torch.Tensor | None = None,
    pinv_tol: float | int = 1e-10,
    ridge_alpha: float = 1e-4,
    dtype: torch.dtype = torch.float64,
) -> LayerResult:
    """Generate readout layers along an increasing n_train grid.

    Args:
        data (SimulationData): Simulation data with states (d^2, n_states), povm (n_out, d^2), and outcomes (n_states, n_shots_max).
        observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
        train_grid (torch.Tensor): Increasing training sizes with shape (n_train_grid,).
        n_shots (int): Number of training shots per state.
        methods (Iterable[str]): Method names.
        state_prior_frame (torch.Tensor | None): Optional prior state frame with shape (d^2, d^2).
        pinv_tol (float | int): Pseudoinverse tolerance or truncation rank.
        ridge_alpha (float): Ridge regularization parameter.
        dtype (torch.dtype): Real accumulator dtype.

    Returns:
        LayerResult: Layers with shape (n_train_grid, n_obs, n_out).
    """
    if data.outcomes is None:
        raise ValueError("ntrain_layers requires stored outcomes.")

    device = data.states.device
    obs = as_observable_matrix(observable, data.d * data.d).to(device=device)
    shadow_methods, linear_methods = _split_methods(methods)

    shadow = None
    if shadow_methods:
        shadow = ShadowReadoutEstimator(
            data.n_out,
            data.d,
            state_prior_frame=state_prior_frame,
            device=device,
            dtype=dtype,
            methods=shadow_methods,
        )
    linear = None
    if linear_methods:
        linear = LinearReadoutEstimator(
            data.n_out, obs.shape[0], device=device, dtype=dtype
        )

    layers: dict[str, list[torch.Tensor]] = {
        method: [] for method in [*shadow_methods, *linear_methods]
    }
    prev = 0
    for n_tensor in train_grid:
        n_train = int(n_tensor.item())
        states = data.states[:, prev:n_train]
        outcomes = data.outcomes[prev:n_train, :n_shots]

        if shadow is not None:
            shadow.update(outcomes, states)
            for method, layer in shadow.layers(obs).items():
                layers[method].append(layer.detach().clone())

        if linear is not None:
            linear.update_outcomes(outcomes, states, obs, data.n_out)
            if "pinv" in linear_methods:
                layers["pinv"].append(linear.layer_pinv(tol=pinv_tol).detach().clone())
            if "ridge" in linear_methods:
                layers["ridge"].append(
                    linear.layer_ridge(alpha=ridge_alpha).detach().clone()
                )

        prev = n_train

    return LayerResult(
        layers={
            method: torch.stack(values, dim=0) for method, values in layers.items()
        },
        observable=obs,
        train_grid=train_grid.to(device=device),
        shot_grid=torch.tensor([n_shots], device=device, dtype=torch.int64),
        seed=data.seed,
        d=data.d,
        n_out=data.n_out,
        metadata={"sweep": "ntrain", "shots": n_shots, "methods": list(layers)},
    )


def shot_layers(
    data: SimulationData,
    observable: torch.Tensor,
    shot_grid: torch.Tensor,
    n_train: int,
    methods: Iterable[str],
    state_prior_frame: torch.Tensor | None = None,
    pinv_tol: float | int = 1e-10,
    ridge_alpha: float = 1e-4,
    dtype: torch.dtype = torch.float64,
) -> LayerResult:
    """Generate readout layers along an increasing shot grid.

    Args:
        data (SimulationData): Simulation data with states (d^2, n_states), povm (n_out, d^2), and outcomes (n_states, n_shots_max).
        observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
        shot_grid (torch.Tensor): Increasing shot values with shape (n_shot_grid,).
        n_train (int): Number of training states.
        methods (Iterable[str]): Method names.
        state_prior_frame (torch.Tensor | None): Optional prior state frame with shape (d^2, d^2).
        pinv_tol (float | int): Pseudoinverse tolerance or truncation rank.
        ridge_alpha (float): Ridge regularization parameter.
        dtype (torch.dtype): Real accumulator dtype.

    Returns:
        LayerResult: Layers with shape (n_shot_grid, n_obs, n_out).
    """
    if data.outcomes is None:
        raise ValueError("shot_layers requires stored outcomes.")

    device = data.states.device
    obs = as_observable_matrix(observable, data.d * data.d).to(device=device)
    shadow_methods, linear_methods = _split_methods(methods)

    states = data.states[:, :n_train]
    outcomes = data.outcomes[:n_train]
    targets = get_observables(obs, states).to(device=device, dtype=dtype)
    stats = RunningOutcomeStats(data.n_out, n_train, device=device, dtype=dtype)
    layers: dict[str, list[torch.Tensor]] = {
        method: [] for method in [*shadow_methods, *linear_methods]
    }
    prev = 0

    for s_tensor in shot_grid:
        shots = int(s_tensor.item())
        stats.update(outcomes[:, prev:shots])
        probs = stats.probabilities()

        if shadow_methods:
            shadow = ShadowReadoutEstimator(
                data.n_out,
                data.d,
                state_prior_frame=state_prior_frame,
                device=device,
                dtype=dtype,
                methods=shadow_methods,
            )
            shadow.update_probs(probs, states)
            for method, layer in shadow.layers(obs).items():
                layers[method].append(layer.detach().clone())

        if linear_methods:
            linear = LinearReadoutEstimator(
                data.n_out, obs.shape[0], device=device, dtype=dtype
            )
            linear.update_probs(probs, targets)
            if "pinv" in linear_methods:
                layers["pinv"].append(linear.layer_pinv(tol=pinv_tol).detach().clone())
            if "ridge" in linear_methods:
                layers["ridge"].append(
                    linear.layer_ridge(alpha=ridge_alpha).detach().clone()
                )

        prev = shots

    return LayerResult(
        layers={
            method: torch.stack(values, dim=0) for method, values in layers.items()
        },
        observable=obs,
        train_grid=torch.tensor([n_train], device=device, dtype=torch.int64),
        shot_grid=shot_grid.to(device=device),
        seed=data.seed,
        d=data.d,
        n_out=data.n_out,
        metadata={"sweep": "shots", "n_train": n_train, "methods": list(layers)},
    )


# ----- METRICS -----


def haar_metrics(result: LayerResult, povm: torch.Tensor) -> MetricResult:
    """Evaluate a layer result by Haar bias and variance.

    Args:
        result (LayerResult): Layer result with layers (..., n_obs, n_out).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).

    Returns:
        MetricResult: Haar metrics with shapes matching the layer grid.
    """
    metrics = evaluate_layers_haar(result.layers, povm, result.observable)
    coords: dict[str, Any] = {"n_train": result.train_grid}
    if result.shot_grid is not None:
        coords["shots"] = result.shot_grid
    return MetricResult(
        metrics=metrics,
        train_grid=result.train_grid,
        shot_grid=result.shot_grid,
        coords=coords,
        seed=result.seed,
        d=result.d,
        n_out=result.n_out,
        metadata={"metric": "haar_bias_variance"},
    )


def stack_metric_results(
    results: Sequence[MetricResult],
    grid_name: str,
    grid_values: torch.Tensor,
    extra_coords: Mapping[str, Any] | None = None,
) -> MetricResult:
    """Stack single-point metric results into one metric grid."""
    if len(results) != int(grid_values.numel()):
        raise ValueError("Number of metric results must match grid_values.")

    device = grid_values.device
    metric_keys = tuple(results[0].metrics)
    metrics: dict[str, torch.Tensor] = {}
    for key in metric_keys:
        values = []
        for result in results:
            if key not in result.metrics:
                raise ValueError(f"Metric result is missing '{key}'.")
            value = result.metrics[key].detach().reshape(-1)[0].to(device=device)
            values.append(value)
        metrics[key] = torch.stack(values)

    coords: dict[str, Any] = {grid_name: grid_values}
    for name, value in dict(extra_coords or {}).items():
        if isinstance(value, torch.Tensor):
            coords[name] = value.to(device=device)
        elif isinstance(value, list | tuple):
            coords[name] = torch.tensor(value, device=device)
        else:
            coords[name] = value

    train_grid = coords.get("n_train")
    if not isinstance(train_grid, torch.Tensor):
        train_grid = None
    shot_grid = coords.get("shots")
    if not isinstance(shot_grid, torch.Tensor):
        shot_grid = None

    return MetricResult(
        metrics=metrics,
        train_grid=train_grid,
        shot_grid=shot_grid,
        coords=coords,
        seed=results[0].seed,
        metadata={"metric": "stacked", "grid": grid_name},
    )


# ----- FRAME DISTANCES -----


def state_frame_distances(
    train_grid: torch.Tensor,
    states: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
    d: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> dict[str, torch.Tensor]:
    """Compute state-frame distances over n_train.

    Args:
        train_grid (torch.Tensor): Training sizes with shape (n_train_grid,).
        states (torch.Tensor | None): States with shape (d^2, n_max), or None to sample global-Haar states.
        reference (torch.Tensor | None): Reference frame with shape (d^2, d^2), or None for global Haar.
        d (int | None): Hilbert-space dimension, required when states and reference are None.
        device (torch.device | str): Computation device.
        dtype (torch.dtype): Complex dtype.

    Returns:
        dict[str, torch.Tensor]: Distance metrics with shape (n_train_grid,).
    """
    n_max = int(train_grid.max().item())
    if states is None:
        if d is None:
            raise ValueError("d is required when states is None.")
        states = sample_dm(n_max, d=d, device=device, dtype=dtype)
    if reference is None:
        d_state = int(round(states.shape[0] ** 0.5))
        reference = haar_state_frame(d_state, device=states.device, dtype=states.dtype)
    out: dict[str, list[torch.Tensor]] = {
        k: [] for k in ["rel_op", "rel_fro", "lambda_min", "lambda_max", "condition"]
    }
    for n_tensor in train_grid:
        emp = state_frame(states[:, : int(n_tensor.item())], normalize=True)
        summary = frame_distance_summary(reference, emp)
        for key in out:
            out[key].append(summary[key])
    return {f"state_{key}": torch.stack(vals) for key, vals in out.items()}


def measurement_frame_distances(
    d: int,
    n_out_grid: torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> dict[str, torch.Tensor]:
    """Compute measurement-frame distances over n_out.

    Args:
        d (int): Hilbert-space dimension.
        n_out_grid (torch.Tensor): POVM sizes with shape (n_out_grid,).
        device (torch.device | str): Computation device.
        dtype (torch.dtype): Complex dtype.

    Returns:
        dict[str, torch.Tensor]: Distance metrics with shape (n_out_grid,).
    """
    out: dict[str, list[torch.Tensor]] = {
        k: [] for k in ["rel_op", "rel_fro", "lambda_min", "lambda_max", "condition"]
    }
    for n_tensor in n_out_grid:
        n_out = int(n_tensor.item())
        povm = sample_povm(n_out, d=d, device=device, dtype=dtype)
        ref = naimark_measurement_frame_prior(d, n_out, device=device, dtype=dtype)
        emp = measurement_frame(povm)
        summary = frame_distance_summary(ref, emp)
        for key in out:
            out[key].append(summary[key])
    return {f"measurement_{key}": torch.stack(vals) for key, vals in out.items()}


def _int_grid(
    values: torch.Tensor | Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.int64)
    return torch.tensor(list(values), device=device, dtype=torch.int64)


def state_frame_distance_grid(
    d: int,
    train_grid: torch.Tensor | Sequence[int],
    gamma_grid: torch.Tensor | Sequence[int] | None = None,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> MetricResult:
    """Compute state-frame distances and package them for metric saving."""
    train_grid = _int_grid(train_grid, device)
    metrics = state_frame_distances(
        train_grid,
        d=d,
        device=device,
        dtype=dtype,
    )
    return MetricResult(
        metrics=metrics,
        train_grid=train_grid,
        coords={
            "n_train": train_grid,
            "d": d,
            **(
                {"gamma": _int_grid(gamma_grid, device)}
                if gamma_grid is not None
                else {}
            ),
        },
        seed=seed,
        d=d,
        metadata={"metric": "frame_distance", "frame": "state"},
    )


def measurement_frame_distance_grid(
    d: int,
    n_out_grid: torch.Tensor | Sequence[int],
    alpha_grid: torch.Tensor | Sequence[int] | None = None,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> MetricResult:
    """Compute measurement-frame distances and package them for metric saving."""
    n_out_grid = _int_grid(n_out_grid, device)
    metrics = measurement_frame_distances(
        d,
        n_out_grid,
        device=device,
        dtype=dtype,
    )
    coords: dict[str, Any] = {"n_out": n_out_grid, "d": d}
    if alpha_grid is not None:
        coords["alpha"] = _int_grid(alpha_grid, device)
    return MetricResult(
        metrics=metrics,
        coords=coords,
        seed=seed,
        d=d,
        metadata={"metric": "frame_distance", "frame": "measurement"},
    )


# ----- BETA FITS -----


def fit_betas(
    metrics: dict[str, torch.Tensor],
    shot_grid: torch.Tensor,
    train_grid: torch.Tensor,
    methods: Iterable[str],
    metric_name: str = "bias2",
) -> dict[str, torch.Tensor]:
    """Fit beta coefficients for several methods.

    Args:
        metrics (dict[str, torch.Tensor]): Metric tensors keyed as method_metric.
        shot_grid (torch.Tensor): Shot values with shape (n_shot_grid,).
        train_grid (torch.Tensor): Training-state values with shape (n_train_grid,).
        methods (Iterable[str]): Method names.
        metric_name (str): Metric suffix to fit.

    Returns:
        dict[str, torch.Tensor]: Beta coefficients keyed as method_beta.
    """
    out: dict[str, torch.Tensor] = {}
    for method in methods:
        key = f"{method}_{metric_name}"
        if key not in metrics:
            continue
        fit = fit_beta_coefficients(metrics[key], shot_grid, train_grid)
        for beta_key, value in fit.items():
            out[f"{method}_{beta_key}"] = value
    return out
