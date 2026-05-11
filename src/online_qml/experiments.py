from __future__ import annotations

from typing import Iterable

import torch

from .core import LayerResult, MetricResult, SimulationData
from .estimators import LinearReadoutEstimator, RunningOutcomeStats, ShadowReadoutEstimator
from .evaluation import evaluate_layers_haar, fit_beta_coefficients
from .quantum import (
    frame_distance_summary,
    get_observables,
    haar_state_frame,
    measurement_frame,
    naimark_measurement_frame_prior,
    sample_dm,
    sample_povm,
    shots_to_statistics,
    state_frame,
)


# =====================================
# LAYER GENERATION
# =====================================


def make_shadow_layers(
    estimator: ShadowReadoutEstimator,
    observable: torch.Tensor,
    methods: Iterable[str] = ("ost", "aost", "prior_ost", "prior_aost"),
    rcond_state: float = 1e-10,
    rcond_frame: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Compute shadow readout layers from an estimator.

    Args:
        estimator (ShadowReadoutEstimator): Accumulated shadow estimator.
        observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
        methods (Iterable[str]): Shadow method names.
        rcond_state (float): Pseudoinverse cutoff for empirical S.
        rcond_frame (float): Pseudoinverse cutoff for empirical F.

    Returns:
        dict[str, torch.Tensor]: Layers with shape (n_obs, n_out).
    """
    return estimator.layers(
        observable,
        methods=methods,
        rcond_state=rcond_state,
        rcond_frame=rcond_frame,
    )


def make_layers_ntrain_grid(
    data: SimulationData,
    observable: torch.Tensor,
    train_grid: torch.Tensor,
    n_shots: int,
    shadow_methods: Iterable[str] = ("ost", "aost", "prior_ost", "prior_aost"),
    linear_methods: Iterable[str] = (),
    pinv_tol: float | int = 1e-10,
    ridge_alpha: float = 1e-4,
    dtype: torch.dtype = torch.float64,
) -> LayerResult:
    """Generate layers over an increasing n_train grid.

    Args:
        data (SimulationData): Simulation data with states (d^2, n_states), povm (n_out, d^2), outcomes (n_states, n_shots_max).
        observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
        train_grid (torch.Tensor): Increasing training sizes with shape (n_train_grid,).
        n_shots (int): Number of training shots per state.
        shadow_methods (Iterable[str]): Shadow method names.
        linear_methods (Iterable[str]): Linear method names, pinv and/or ridge.
        pinv_tol (float | int): Pseudoinverse tolerance or rank.
        ridge_alpha (float): Ridge regularization parameter.
        dtype (torch.dtype): Real accumulator dtype.

    Returns:
        LayerResult: Layers with shape (n_train_grid, n_obs, n_out).
    """
    if data.outcomes is None:
        raise ValueError("make_layers_ntrain_grid requires stored outcomes.")
    device = data.states.device
    obs = observable.to(device=device)
    n_obs = 1 if obs.ndim == 1 else (obs.shape[0] if obs.shape[-1] == data.d * data.d else obs.shape[1])
    shadow_est = ShadowReadoutEstimator(data.n_out, data.d, device=device, dtype=dtype)
    linear_est = LinearReadoutEstimator(data.n_out, n_obs, device=device, dtype=dtype) if linear_methods else None
    layers: dict[str, list[torch.Tensor]] = {method: [] for method in shadow_methods}
    for method in linear_methods:
        layers[method] = []
    prev_n = 0
    for n_train_tensor in train_grid:
        n_train = int(n_train_tensor.item())
        states_slice = data.states[:, prev_n:n_train]
        outcomes_slice = data.outcomes[prev_n:n_train, :n_shots]
        if n_shots == 1:
            shadow_est.update_single_shot(outcomes_slice[:, 0], states_slice)
        else:
            shadow_est.update_outcomes(outcomes_slice, states_slice)
        for method, layer in shadow_est.layers(obs, methods=shadow_methods).items():
            layers[method].append(layer.detach().clone())
        if linear_est is not None:
            probs_slice = shots_to_statistics(outcomes_slice, data.n_out).to(device=device, dtype=dtype)
            targets = get_observables(obs, states_slice).to(dtype=dtype)
            linear_est.update_probs(probs_slice, targets)
            for method in linear_methods:
                if method == "pinv":
                    layers[method].append(linear_est.layer_pinv(tol=pinv_tol).detach().clone())
                elif method == "ridge":
                    layers[method].append(linear_est.layer_ridge(alpha=ridge_alpha).detach().clone())
                else:
                    raise ValueError(f"Unknown linear method '{method}'.")
        prev_n = n_train
    stacked = {method: torch.stack(seq, dim=0) for method, seq in layers.items()}
    return LayerResult(
        layers=stacked,
        d=data.d,
        n_out=data.n_out,
        shot_grid=torch.tensor([n_shots], device=device, dtype=torch.int64),
        train_grid=train_grid.to(device=device),
        seed=data.seed,
        observable=obs,
        metadata={"sweep": "n_train"},
    )


def make_layers_shot_grid(
    data: SimulationData,
    observable: torch.Tensor,
    shot_grid: torch.Tensor,
    n_train: int,
    shadow_methods: Iterable[str] = ("ost", "aost", "prior_ost", "prior_aost"),
    linear_methods: Iterable[str] = (),
    pinv_tol: float | int = 1e-10,
    ridge_alpha: float = 1e-4,
    dtype: torch.dtype = torch.float64,
) -> LayerResult:
    """Generate layers over an increasing shot grid.

    Args:
        data (SimulationData): Simulation data with states (d^2, n_states), povm (n_out, d^2), outcomes (n_states, n_shots_max).
        observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
        shot_grid (torch.Tensor): Increasing shot values with shape (n_shots_grid,).
        n_train (int): Number of training states.
        shadow_methods (Iterable[str]): Shadow method names.
        linear_methods (Iterable[str]): Linear method names, pinv and/or ridge.
        pinv_tol (float | int): Pseudoinverse tolerance or rank.
        ridge_alpha (float): Ridge regularization parameter.
        dtype (torch.dtype): Real accumulator dtype.

    Returns:
        LayerResult: Layers with shape (n_shots_grid, n_obs, n_out).
    """
    if data.outcomes is None:
        raise ValueError("make_layers_shot_grid requires stored outcomes.")
    device = data.states.device
    obs = observable.to(device=device)
    states = data.states[:, :n_train]
    outcomes = data.outcomes[:n_train]
    n_obs = 1 if obs.ndim == 1 else (obs.shape[0] if obs.shape[-1] == data.d * data.d else obs.shape[1])
    stats = RunningOutcomeStats(data.n_out, n_train, device=device, dtype=dtype)
    layers: dict[str, list[torch.Tensor]] = {method: [] for method in shadow_methods}
    for method in linear_methods:
        layers[method] = []
    prev_s = 0
    targets = get_observables(obs, states).to(dtype=dtype)
    for n_shots_tensor in shot_grid:
        n_shots = int(n_shots_tensor.item())
        stats.update(outcomes[:, prev_s:n_shots])
        probs = stats.probabilities()
        shadow_est = ShadowReadoutEstimator(data.n_out, data.d, device=device, dtype=dtype)
        shadow_est.update_probs(probs, states)
        for method, layer in shadow_est.layers(obs, methods=shadow_methods).items():
            layers[method].append(layer.detach().clone())
        if linear_methods:
            linear_est = LinearReadoutEstimator(data.n_out, n_obs, device=device, dtype=dtype)
            linear_est.update_probs(probs, targets)
            for method in linear_methods:
                if method == "pinv":
                    layers[method].append(linear_est.layer_pinv(tol=pinv_tol).detach().clone())
                elif method == "ridge":
                    layers[method].append(linear_est.layer_ridge(alpha=ridge_alpha).detach().clone())
                else:
                    raise ValueError(f"Unknown linear method '{method}'.")
        prev_s = n_shots
    stacked = {method: torch.stack(seq, dim=0) for method, seq in layers.items()}
    return LayerResult(
        layers=stacked,
        d=data.d,
        n_out=data.n_out,
        shot_grid=shot_grid.to(device=device),
        train_grid=torch.tensor([n_train], device=device, dtype=torch.int64),
        seed=data.seed,
        observable=obs,
        metadata={"sweep": "shots"},
    )


# =====================================
# METRICS
# =====================================


def evaluate_layer_result_haar(result: LayerResult, povm: torch.Tensor) -> MetricResult:
    """Evaluate a layer result by Haar bias and variance.

    Args:
        result (LayerResult): Layer result with layers (..., n_obs, n_out).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).

    Returns:
        MetricResult: Haar metrics with shapes matching the layer grid.
    """
    metrics = evaluate_layers_haar(result.layers, povm, result.observable)
    return MetricResult(
        metrics=metrics,
        d=result.d,
        n_out=result.n_out,
        shot_grid=result.shot_grid,
        train_grid=result.train_grid,
        seed=result.seed,
        metadata={"metric": "haar_bias_variance"},
    )


# =====================================
# FRAME DISTANCES
# =====================================


def state_frame_distance_grid(
    d: int,
    train_grid: torch.Tensor,
    states: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> dict[str, torch.Tensor]:
    """Compute state-frame distances over n_train.

    Args:
        d (int): Hilbert-space dimension.
        train_grid (torch.Tensor): Training sizes with shape (n_train_grid,).
        states (torch.Tensor | None): States with shape (d^2, n_max), or None to sample.
        device (torch.device | str): Computation device.
        dtype (torch.dtype): Complex dtype.

    Returns:
        dict[str, torch.Tensor]: Distance metrics with shape (n_train_grid,).
    """
    n_max = int(train_grid.max().item())
    if states is None:
        states = sample_dm(n_max, d=d, device=device, dtype=dtype)
    ref = haar_state_frame(d, device=states.device, dtype=states.dtype)
    out: dict[str, list[torch.Tensor]] = {"rel_op": [], "rel_fro": [], "lambda_min": [], "lambda_max": [], "condition": []}
    for n_train_tensor in train_grid:
        n_train = int(n_train_tensor.item())
        emp = state_frame(states[:, :n_train], normalize=True)
        summary = frame_distance_summary(ref, emp)
        for key in out:
            out[key].append(summary[key])
    return {f"state_{key}": torch.stack(vals) for key, vals in out.items()}


def measurement_frame_distance_grid(
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
    out: dict[str, list[torch.Tensor]] = {"rel_op": [], "rel_fro": [], "lambda_min": [], "lambda_max": [], "condition": []}
    for n_out_tensor in n_out_grid:
        n_out = int(n_out_tensor.item())
        povm = sample_povm(n_out, d=d, device=device, dtype=dtype)
        ref = naimark_measurement_frame_prior(d, n_out, device=device, dtype=dtype)
        emp = measurement_frame(povm)
        summary = frame_distance_summary(ref, emp)
        for key in out:
            out[key].append(summary[key])
    return {f"measurement_{key}": torch.stack(vals) for key, vals in out.items()}


# =====================================
# BETA FITS
# =====================================


def fit_betas_from_metrics(
    metrics: dict[str, torch.Tensor],
    shot_grid: torch.Tensor,
    train_grid: torch.Tensor,
    methods: Iterable[str],
    metric_name: str = "bias2",
) -> dict[str, torch.Tensor]:
    """Fit beta coefficients for several methods.

    Args:
        metrics (dict[str, torch.Tensor]): Metric tensors keyed as method_metric.
        shot_grid (torch.Tensor): Shot values with shape (n_shots_grid,).
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
