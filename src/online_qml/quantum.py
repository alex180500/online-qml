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


def sample_observable(
    n_obs: int = 1,
    d: int = 2,
    kind: str = "proj",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> torch.Tensor:
    """Sample Hermitian-conjugated flattened observable rows.

    Args:
        n_obs (int): Number of observables.
        d (int): Hilbert-space dimension.
        kind (str): Observable type. ``"proj"`` samples ``|phi><phi|``;
            ``"center"`` samples the centered projector ``|phi><phi| - I / d``;
            ``"center_norm"`` additionally divides by
            ``sqrt((d - 1) / (d^2 * (d + 1)))``.
        device (torch.device | str): Output device.
        dtype (torch.dtype): Complex output dtype.

    Returns:
        torch.Tensor: Hermitian-conjugated flattened observable rows with shape
            (n_obs, d^2). ``obs @ mat`` computes the linear product.
    """
    obs = sample_dm(n_obs, d=d, device=device, dtype=dtype).adjoint()
    if kind == "proj":
        return obs
    if kind not in {"center", "center_norm"}:
        raise ValueError(
            "observable kind must be 'proj', 'center', or 'center_norm'; "
            f"got {kind!r}."
        )

    identity = torch.eye(d, device=device, dtype=dtype).reshape(1, d * d)
    obs = obs - identity / d
    if kind == "center":
        return obs
    if d <= 1:
        raise ValueError("center_norm observable requires d > 1.")
    haar_std = ((d - 1) / (d * d * (d + 1))) ** 0.5
    return obs / haar_std


def as_observable_matrix(observable: torch.Tensor, d2: int) -> torch.Tensor:
    """Convert observables to Hermitian-conjugated row format.

    Args:
        observable (torch.Tensor): Hermitian-conjugated flattened observable rows
            with shape (d^2,) or (n_obs, d^2), or the transpose of those rows
            with shape (d^2, n_obs). These rows are linear functionals:
            ``obs @ mat`` computes the product.
        d2 (int): Squared Hilbert-space dimension.

    Returns:
        torch.Tensor: Hermitian-conjugated observable matrix with shape (n_obs, d^2).
    """
    if observable.ndim == 1:
        return observable.reshape(1, d2)
    if observable.ndim == 2 and observable.shape[0] == d2 and observable.shape[1] != d2:
        return observable.T
    if observable.ndim == 2 and observable.shape[1] == d2:
        return observable
    raise ValueError(
        "observable must contain Hermitian-conjugated rows with shape "
        f"(d^2,), (n_obs, d^2), or (d^2, n_obs); got {tuple(observable.shape)}."
    )


def get_observables(
    obs_matrix: torch.Tensor,
    states: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute observable expectation values as ``obs @ states``.

    Args:
        obs_matrix (torch.Tensor): Hermitian-conjugated flattened observable rows
            with shape (n_obs, d^2). No conjugation is applied here.
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).
        device (torch.device | str | None): Optional output device.
        dtype (torch.dtype | None): Optional output real dtype.

    Returns:
        torch.Tensor: Expectation matrix with shape (n_obs, n_states).
    """
    values = torch.matmul(obs_matrix, states).real
    if device is None and dtype is None:
        return values
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    return values.to(**to_kwargs)


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
