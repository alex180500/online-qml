from math import sqrt
import torch

# ----- STATES AND OBSERVABLES -----


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
    """Sample flattened Haar-random pure-state density matrices.

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


def haar_projector_variance(d: int) -> float:
    """Return the Haar variance of a centered rank-one projector."""
    return (d - 1) / (d * d * (d + 1))


def sample_norm_proj(
    d: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample one centered Haar projector with unit Haar variance.

    Returns:
        torch.Tensor: Flattened observable with shape (d^2,).
    """
    projector = sample_dm(1, d=d, device=device, dtype=dtype).reshape(-1)
    identity = torch.eye(d, device=device, dtype=dtype).reshape(-1)
    observable = (projector - identity / d) / (haar_projector_variance(d) ** 0.5)
    if observable.shape != (d * d,):
        raise ValueError(f"Bad observable shape: {tuple(observable.shape)}")
    return observable


def incomplete_povm_floor(d: int, n_out: int) -> float:
    """Return the normalized projection-error floor for a centered target."""
    centered_dim = d * d - 1
    accessible_dim = min(n_out - 1, centered_dim)
    return 1.0 - accessible_dim / centered_dim


def sample_product_dm(
    n_states: int,
    n_sites: int,
    local_dim: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample flattened product-Haar pure-state density matrices.

    Args:
        n_states (int): Number of density matrices.
        n_sites (int): Number of tensor-product sites.
        local_dim (int): Local Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Flattened density matrices with shape (local_dim^(2 n_sites), n_states).
    """
    state = torch.ones(1, n_states, device=device, dtype=dtype)
    for _ in range(n_sites):
        local = sample_states(n_states, d=local_dim, device=device, dtype=dtype)
        state = torch.einsum("an,bn->abn", state, local).reshape(
            state.shape[0] * local_dim, n_states
        )
    d = local_dim**n_sites
    density = state.unsqueeze(1) * state.conj().unsqueeze(0)
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


def as_observable_matrix(observable: torch.Tensor, d2: int) -> torch.Tensor:
    """Convert observables to row format.

    Args:
        observable (torch.Tensor): Observable with shape (d^2,), (n_obs, d^2), or (d^2, n_obs).
        d2 (int): Squared Hilbert-space dimension.

    Returns:
        torch.Tensor: Observable matrix with shape (n_obs, d^2).
    """
    if observable.ndim == 1:
        return observable.reshape(1, d2)
    if observable.ndim == 2 and observable.shape[0] == d2 and observable.shape[1] != d2:
        return observable.T
    if observable.ndim == 2 and observable.shape[1] == d2:
        return observable
    raise ValueError(
        f"observable must have shape (d^2,), (n_obs, d^2), or (d^2, n_obs); got {tuple(observable.shape)}."
    )


def get_observables(obs_matrix: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Compute observable expectation values.

    Args:
        obs_matrix (torch.Tensor): Flattened observables with shape (n_obs, d^2).
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).

    Returns:
        torch.Tensor: Expectation matrix with shape (n_obs, n_states).
    """
    return torch.matmul(obs_matrix.conj(), states).real


# ----- POVMS AND STATISTICS -----


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
    if n_outcomes < d:
        raise ValueError("n_outcomes must be at least d.")
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
    """Convert outcome shots to empirical probabilities.

    Args:
        outcomes (torch.Tensor): Outcome matrix with shape (n_states, n_shots).
        n_out (int): Number of POVM outcomes.

    Returns:
        torch.Tensor: Probability matrix with shape (n_out, n_states).
    """
    if outcomes.ndim == 1:
        outcomes = outcomes.reshape(-1, 1)
    n_states, n_shots = outcomes.shape
    increments = torch.arange(n_states, device=outcomes.device).unsqueeze(1) * n_out
    linear_indices = (outcomes.to(torch.long) + increments).flatten()
    counts = torch.bincount(linear_indices, minlength=n_states * n_out)
    return counts.view(n_states, n_out).to(torch.float32).T / n_shots


# ----- FRAMES -----


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


def _local_to_global_inverse_permutation(
    n_sites: int,
    local_dim: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    size = local_dim ** (2 * n_sites)
    perm = torch.empty(size, device=device, dtype=torch.long)
    for local_idx in range(size):
        x = local_idx
        digits = [0] * (2 * n_sites)
        for pos in range(2 * n_sites - 1, -1, -1):
            digits[pos] = x % local_dim
            x //= local_dim
        global_digits = digits[0::2] + digits[1::2]
        global_idx = 0
        for digit in global_digits:
            global_idx = global_idx * local_dim + digit
        perm[local_idx] = global_idx
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(size, device=device, dtype=torch.long)
    return inv


def product_haar_state_frame(
    n_sites: int,
    local_dim: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Return the product-Haar state frame.

    Args:
        n_sites (int): Number of tensor-product sites.
        local_dim (int): Local Hilbert-space dimension.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Product state frame with shape (local_dim^(2 n_sites), local_dim^(2 n_sites)).
    """
    local = haar_state_frame(local_dim, device=device, dtype=dtype)
    frame = local
    for _ in range(n_sites - 1):
        frame = torch.kron(frame, local)
    inv_perm = _local_to_global_inverse_permutation(n_sites, local_dim, device=device)
    return frame.index_select(0, inv_perm).index_select(1, inv_perm)


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
