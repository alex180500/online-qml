from collections.abc import Iterable
import torch

from .core.containers import LayerResult, MetricResult, SimulationData
from .core.methods import split_methods as _split_methods
from .estimators import (
    LinearReadoutEstimator,
    RunningOutcomeStats,
    ShadowReadoutEstimator,
)
from .evaluation import evaluate_layers_haar
from .quantum import (
    as_observable_matrix,
    get_observables,
    infinite_stats,
    sample_dm,
    sample_povm,
    shots_outcome,
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
        observable (torch.Tensor): Hermitian-conjugated flattened observable rows
            with shape (n_obs, d^2), their transpose with shape (d^2, n_obs),
            or one row with shape (d^2,). With flattened matrices stored as
            columns, ``obs @ mat`` is the linear product.
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
        observable (torch.Tensor): Hermitian-conjugated flattened observable rows
            with shape (n_obs, d^2), their transpose with shape (d^2, n_obs),
            or one row with shape (d^2,). With flattened matrices stored as
            columns, ``obs @ mat`` is the linear product.
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
    targets = get_observables(obs, states, device=device, dtype=dtype)
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


def nout_metrics(
    states: torch.Tensor,
    observable: torch.Tensor,
    alpha_grid: torch.Tensor,
    n_out_grid: torch.Tensor,
    shots: int,
    methods: Iterable[str],
    batch_size: int,
    rcond_state: float = 1e-10,
    rcond_frame: float = 1e-10,
    dtype: torch.dtype = torch.float64,
    *,
    seed: int | None = None,
    pinv_tol: float | int = 1e-10,
    ridge_alpha: float = 1e-4,
) -> MetricResult:
    """Evaluate Haar metrics along a changing n_out grid.

    The readout layers cannot be stacked into a LayerResult because their
    final dimension changes with n_out, so this helper evaluates each layer
    immediately and returns only the metric grid.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    d = int(round(states.shape[0] ** 0.5))
    device = states.device
    alpha_grid = torch.as_tensor(alpha_grid, device=device, dtype=torch.int64)
    n_out_grid = torch.as_tensor(n_out_grid, device=device, dtype=torch.int64)
    if alpha_grid.numel() != n_out_grid.numel():
        raise ValueError("alpha_grid and n_out_grid must have the same length.")

    obs = as_observable_matrix(observable, d * d).to(device=device)
    methods = list(methods)
    shadow_methods, linear_methods = _split_methods(methods)
    out = {
        f"{method}_{metric}": []
        for method in [*shadow_methods, *linear_methods]
        for metric in ("bias2", "variance")
    }

    for alpha_tensor, n_out_tensor in zip(alpha_grid, n_out_grid, strict=True):
        alpha = int(alpha_tensor.item())
        n_out = int(n_out_tensor.item())
        povm = sample_povm(n_out, d=d, device=device, dtype=states.dtype)
        shadow = None
        if shadow_methods:
            shadow = ShadowReadoutEstimator(
                n_out,
                d,
                device=device,
                dtype=dtype,
                methods=shadow_methods,
            )
        linear = None
        if linear_methods:
            linear = LinearReadoutEstimator(
                n_out, obs.shape[0], device=device, dtype=dtype
            )

        for start in range(0, states.shape[1], batch_size):
            stop = min(start + batch_size, states.shape[1])
            state_batch = states[:, start:stop]
            if shots <= 0:
                probs = infinite_stats(povm, state_batch)
                if shadow is not None:
                    shadow.update_probs(probs, state_batch)
                if linear is not None:
                    targets = get_observables(
                        obs, state_batch, device=device, dtype=dtype
                    )
                    linear.update_probs(probs, targets)
            else:
                outcomes = shots_outcome(povm, state_batch, shots)
                if shadow is not None:
                    shadow.update(outcomes, state_batch)
                if linear is not None:
                    linear.update_outcomes(outcomes, state_batch, obs, n_out)

        layers: dict[str, torch.Tensor] = {}
        if shadow is not None:
            layers.update(
                shadow.layers(
                    obs,
                    rcond_state=rcond_state,
                    rcond_frame=rcond_frame,
                )
            )
        if linear is not None:
            if "pinv" in linear_methods:
                layers["pinv"] = linear.layer_pinv(tol=pinv_tol).detach().clone()
            if "ridge" in linear_methods:
                layers["ridge"] = linear.layer_ridge(alpha=ridge_alpha).detach().clone()
        metrics = evaluate_layers_haar(layers, povm, obs)
        for key in out:
            out[key].append(metrics[key].detach().clone())

        print(f"    alpha={alpha} n_out={n_out} done")

    return MetricResult(
        metrics={key: torch.stack(values) for key, values in out.items()},
        coords={
            "alpha": alpha_grid,
            "n_out": n_out_grid,
            "n_train": torch.tensor(
                [states.shape[1]], device=device, dtype=torch.int64
            ),
            "shots": torch.tensor([shots], device=device, dtype=torch.int64),
            "d": d,
        },
        seed=seed,
        d=d,
        n_out=None,
        metadata={
            "metric": "haar_bias_variance",
            "sweep": "nout",
            "shots": "infinite" if shots <= 0 else shots,
            "methods": methods,
        },
    )
