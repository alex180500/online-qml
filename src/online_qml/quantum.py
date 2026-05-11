from __future__ import annotations

from math import sqrt

import torch

# =====================================
# STATES AND OBSERVABLES
# =====================================


def sample_states(
    n_states: int,
    d: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample normalized Haar-random pure states.

    Args:
        n_states (int): Number of states.
        d (int): Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: State vectors with shape (d, n_states).
    """
    states = torch.randn(d, n_states, dtype=dtype, device=device)
    return states / torch.linalg.vector_norm(states, dim=0, keepdim=True)


def sample_dm(
    n_states: int,
    d: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample flattened pure-state density matrices.

    Args:
        n_states (int): Number of density matrices.
        d (int): Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Flattened density matrices with shape (d^2, n_states).
    """
    states = sample_states(n_states, d=d, device=device, dtype=dtype)
    density = states.unsqueeze(1) * states.conj().unsqueeze(0)
    return density.reshape(d * d, n_states)


def sample_traceless_operator(
    n_operators: int,
    d: int = 2,
    ord: str = "fro",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample Hermitian traceless operators with unit norm.

    Args:
        n_operators (int): Number of operators.
        d (int): Hilbert-space dimension.
        ord (str): Matrix norm used for normalization.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Flattened operators with shape (d^2, n_operators).
    """
    x = torch.randn(n_operators, d, d, device=device, dtype=dtype)
    h = 0.5 * (x + x.conj().transpose(-1, -2))
    eye = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    tr = torch.einsum("nii->n", h).view(-1, 1, 1)
    h = h - (tr / d) * eye
    norms = torch.linalg.matrix_norm(h, ord=ord, dim=(-2, -1))
    eps = torch.finfo(norms.dtype).eps
    h = h / torch.clamp(norms, min=eps).view(-1, 1, 1)
    return h.reshape(n_operators, d * d).T


def get_observables(obs_matrix: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Compute observable expectation values.

    Args:
        obs_matrix (torch.Tensor): Flattened observables with shape (n_obs, d^2).
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).

    Returns:
        torch.Tensor: Expectation matrix with shape (n_obs, n_states).
    """
    return torch.matmul(obs_matrix.conj(), states).real


# =====================================
# POVMS AND STATISTICS
# =====================================


def sample_unitary(
    d: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample a Haar-random unitary by QR decomposition.

    Args:
        d (int): Unitary dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Unitary matrix with shape (d, d).
    """
    x = torch.randn(d, d, dtype=dtype, device=device)
    q, r = torch.linalg.qr(x)
    r_diag = torch.diagonal(r)
    phase = r_diag / r_diag.abs()
    return q * phase


def sample_povm(
    n_outcomes: int,
    d: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample a rank-one Naimark POVM.

    Args:
        n_outcomes (int): Number of POVM outcomes.
        d (int): Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Flattened POVM elements with shape (n_outcomes, d^2).
    """
    unitary = sample_unitary(n_outcomes, device=device, dtype=dtype)
    v = unitary[:, :d]
    ket = v.conj().unsqueeze(2)
    bra = v.unsqueeze(1)
    povm = torch.bmm(ket, bra)
    return povm.reshape(n_outcomes, d * d)


def infinite_stats(povm: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Compute exact POVM probabilities.

    Args:
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).

    Returns:
        torch.Tensor: Probability matrix with shape (n_out, n_states).
    """
    return torch.matmul(povm.conj(), states).real


def finite_stats(
    povm: torch.Tensor,
    states: torch.Tensor,
    shots: int | None,
) -> torch.Tensor:
    """Compute finite-shot empirical probabilities.

    Args:
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).
        shots (int | None): Number of shots, or None for exact probabilities.

    Returns:
        torch.Tensor: Probability matrix with shape (n_out, n_states).
    """
    if shots is None:
        return infinite_stats(povm, states)
    probs_t = infinite_stats(povm, states).T
    probs_t = torch.clamp(probs_t, min=0.0)
    probs_t = probs_t / probs_t.sum(dim=1, keepdim=True)
    multinomial = torch.distributions.Multinomial(total_count=shots, probs=probs_t)
    return multinomial.sample().T / shots


def shots_outcome(povm: torch.Tensor, states: torch.Tensor, shots: int) -> torch.Tensor:
    """Sample POVM outcomes for each state.

    Args:
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).
        shots (int): Number of shots per state.

    Returns:
        torch.Tensor: Outcome matrix with shape (n_states, shots).
    """
    probs = infinite_stats(povm, states).T
    probs.clamp_(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return torch.multinomial(probs, shots, replacement=True)


def shots_to_statistics(outcomes: torch.Tensor, n_out: int) -> torch.Tensor:
    """Convert outcome shots to probabilities.

    Args:
        outcomes (torch.Tensor): Outcome matrix with shape (n_states, n_shots).
        n_out (int): Number of POVM outcomes.

    Returns:
        torch.Tensor: Probability matrix with shape (n_out, n_states).
    """
    n_states, n_shots = outcomes.shape
    increments = torch.arange(n_states, device=outcomes.device).unsqueeze(1) * n_out
    linear_indices = (outcomes.to(torch.long) + increments).flatten()
    counts = torch.bincount(linear_indices, minlength=n_states * n_out)
    return counts.view(n_states, n_out).to(torch.float32).T / n_shots


# =====================================
# FRAMES
# =====================================


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
    """Return the analytic Haar state frame.

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
    evals, evecs = torch.linalg.eigh(0.5 * (reference + reference.adjoint()))
    keep = evals > rcond * evals.max()
    if not torch.any(keep):
        raise ValueError("reference frame has no eigenvalues above cutoff.")
    inv_sqrt = (evecs[:, keep] / torch.sqrt(evals[keep]).unsqueeze(0)) @ evecs[
        :, keep
    ].adjoint()
    whitened = inv_sqrt @ empirical @ inv_sqrt
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


# =====================================
# TESTING HELPERS
# =====================================


def get_test_input(
    test_n_or_states: int | torch.Tensor,
    povm: torch.Tensor,
    observables: torch.Tensor,
    stat: int | None = None,
    d: int = 2,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate test probabilities and observable values.

    Args:
        test_n_or_states (int | torch.Tensor): Number of states or states with shape (d^2, n_test).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        observables (torch.Tensor): Observables with shape (n_obs, d^2).
        stat (int | None): Test shots, or None for exact probabilities.
        d (int): Hilbert-space dimension.
        device (torch.device | str): Output device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Probabilities (n_out, n_test) and observables (n_obs, n_test).
    """
    dtype = povm.dtype
    if isinstance(test_n_or_states, int):
        states = sample_dm(test_n_or_states, d=d, device=device, dtype=dtype)
    else:
        states = test_n_or_states.to(device=device, dtype=dtype)
    if stat is None:
        probs = infinite_stats(povm, states)
    else:
        outcomes = shots_outcome(povm, states, stat)
        probs = shots_to_statistics(outcomes, povm.shape[0])
    return probs, get_observables(observables, states)


def get_test_mse(
    layer: torch.Tensor, test_obs: torch.Tensor, test_probs: torch.Tensor
) -> torch.Tensor:
    """Compute empirical test MSE.

    Args:
        layer (torch.Tensor): Readout layer with shape (n_obs, n_out).
        test_obs (torch.Tensor): Observable values with shape (n_obs, n_test).
        test_probs (torch.Tensor): Test probabilities with shape (n_out, n_test).

    Returns:
        torch.Tensor: Scalar MSE.
    """
    prediction = torch.matmul(layer, test_probs)
    return torch.mean(torch.abs(prediction - test_obs) ** 2)
