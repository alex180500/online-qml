from collections.abc import Sequence
from math import sqrt
from typing import Any

import torch

from .core.containers import MetricResult
from .quantum import sample_dm, sample_povm


def vec_identity(
    d: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.cfloat
) -> torch.Tensor:
    """Return the flattened identity matrix.

    Args:
        d (int): Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Flattened identity with shape (d^2,).
    """
    return torch.eye(d, device=device, dtype=dtype).reshape(-1)


def trace_superoperator(
    d: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Return the superoperator A -> Tr(A) I.

    Args:
        d (int): Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Superoperator matrix with shape (d^2, d^2).
    """
    vec_i = vec_identity(d, device=device, dtype=dtype)
    return torch.outer(vec_i, vec_i.conj())


def haar_state_frame(
    d: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Return the analytic global-Haar state frame.

    Args:
        d (int): Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: State frame with shape (d^2, d^2).
    """
    d2 = d * d
    eye = torch.eye(d2, device=device, dtype=dtype)
    return (trace_superoperator(d, device=device, dtype=dtype) + eye) / (d * (d + 1))


def naimark_measurement_frame_prior(
    d: int,
    n_out: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Return the analytic Naimark measurement-frame prior.

    Args:
        d (int): Hilbert-space dimension.
        n_out (int): Number of POVM outcomes.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Measurement-frame prior with shape (d^2, d^2).
    """
    d2 = d * d
    eye = torch.eye(d2, device=device, dtype=dtype)
    return (trace_superoperator(d, device=device, dtype=dtype) + eye) / (n_out + 1)


def state_frame(states: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute the empirical state frame.

    Args:
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).
        normalize (bool): If True, divide by n_states.

    Returns:
        torch.Tensor: State frame with shape (d^2, d^2).
    """
    frame = torch.matmul(states, states.adjoint())
    if normalize:
        frame = frame / states.shape[1]
    return frame


def measurement_frame(povm: torch.Tensor) -> torch.Tensor:
    """Compute the empirical measurement frame.

    Args:
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).

    Returns:
        torch.Tensor: Measurement frame with shape (d^2, d^2).
    """
    return torch.matmul(povm.T, povm.conj())


def frame_relative_spectrum(
    reference: torch.Tensor,
    empirical: torch.Tensor,
    rcond: float = 1e-12,
) -> torch.Tensor:
    """Compute eigenvalues of reference^{-1/2} empirical reference^{-1/2}.

    Args:
        reference (torch.Tensor): Reference frame with shape (d^2, d^2).
        empirical (torch.Tensor): Empirical frame with shape (d^2, d^2).
        rcond (float): Relative cutoff for reference eigenvalues.

    Returns:
        torch.Tensor: Relative eigenvalues with shape (rank,).
    """
    ref = 0.5 * (reference + reference.adjoint())
    emp = 0.5 * (empirical + empirical.adjoint())
    evals, evecs = torch.linalg.eigh(ref)
    keep = evals > rcond * evals.max()
    if not torch.any(keep):
        raise ValueError("reference frame has no eigenvalues above cutoff.")
    inv_sqrt = (evecs[:, keep] / torch.sqrt(evals[keep]).unsqueeze(0)) @ evecs[
        :, keep
    ].adjoint()
    whitened = inv_sqrt @ emp @ inv_sqrt
    whitened = 0.5 * (whitened + whitened.adjoint())
    return torch.linalg.eigvalsh(whitened).real


def frame_distance_summary(
    reference: torch.Tensor,
    empirical: torch.Tensor,
    rcond: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute relative frame-distance diagnostics.

    Args:
        reference (torch.Tensor): Reference frame with shape (d^2, d^2).
        empirical (torch.Tensor): Empirical frame with shape (d^2, d^2).
        rcond (float): Relative cutoff for reference eigenvalues.

    Returns:
        dict[str, torch.Tensor]: Relative operator norm, Frobenius norm and spectrum statistics.
    """
    lambdas = frame_relative_spectrum(reference, empirical, rcond=rcond)
    diff = lambdas - 1.0
    lambda_min = lambdas.min()
    lambda_max = lambdas.max()
    return {
        "rel_op": diff.abs().max(),
        "rel_fro": torch.linalg.vector_norm(diff) / sqrt(diff.numel()),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition": lambda_max / lambda_min,
    }


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


def povm_frame_distances(
    d: int,
    n_out_grid: torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> dict[str, torch.Tensor]:
    """Compute POVM-frame distances over n_out.

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
    return {f"povm_{key}": torch.stack(vals) for key, vals in out.items()}


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


def povm_frame_distance_grid(
    d: int,
    n_out_grid: torch.Tensor | Sequence[int],
    alpha_grid: torch.Tensor | Sequence[int] | None = None,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> MetricResult:
    """Compute POVM-frame distances and package them for metric saving."""
    n_out_grid = _int_grid(n_out_grid, device)
    metrics = povm_frame_distances(
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
        metadata={"metric": "frame_distance", "frame": "povm"},
    )
