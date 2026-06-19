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
