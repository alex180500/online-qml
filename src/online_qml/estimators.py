from __future__ import annotations

from math import sqrt
from typing import Iterable

import torch

from .core import get_complex_dtype
from .quantum import get_observables


# =====================================
# HELPERS
# =====================================


def _as_observable_matrix(observable: torch.Tensor, d2: int) -> torch.Tensor:
    if observable.ndim == 1:
        return observable.reshape(1, d2)
    if observable.ndim == 2 and observable.shape[0] == d2 and observable.shape[1] != d2:
        return observable.T
    if observable.ndim == 2 and observable.shape[1] == d2:
        return observable
    raise ValueError(f"observable must have shape (d^2,), (n_obs, d^2), or (d^2, n_obs); got {tuple(observable.shape)}.")


def pinv_truncated(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Compute a truncated Moore-Penrose pseudoinverse.

    Args:
        matrix (torch.Tensor): Matrix with shape (m, n).
        rank (int): Number of singular values to keep.

    Returns:
        torch.Tensor: Pseudoinverse with shape (n, m).
    """
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.shape[0])
    uk = u[:, :rank]
    sk = s[:rank]
    vk = vh[:rank, :].T
    return (vk / sk.unsqueeze(0)) @ uk.T


def method_flags(method: str) -> tuple[bool, bool]:
    """Return adaptive_state and prior_frame flags for an OST method name.

    Args:
        method (str): Method name: ost, aost, prior_ost, or prior_aost.

    Returns:
        tuple[bool, bool]: adaptive_state and prior_frame flags.
    """
    aliases = {
        "si_fe": "ost",
        "se_fe": "aost",
        "si_fi": "prior_ost",
        "se_fi": "prior_aost",
    }
    method = aliases.get(method.lower(), method.lower())
    if method == "ost":
        return False, False
    if method == "aost":
        return True, False
    if method == "prior_ost":
        return False, True
    if method == "prior_aost":
        return True, True
    raise ValueError(f"Unknown shadow method '{method}'.")


# =====================================
# SHADOW READOUT ESTIMATOR
# =====================================


class ShadowReadoutEstimator:
    """Online shadow estimator for QELM readout layers.

    Args:
        n_out (int): Number of POVM outcomes.
        d (int): Hilbert-space dimension.
        accumulate_state_frame (bool): Whether to accumulate empirical S with shape (d^2, d^2).
        device (torch.device | str): Accumulator device.
        dtype (torch.dtype): Real accumulator dtype.
    """

    def __init__(
        self,
        n_out: int,
        d: int,
        accumulate_state_frame: bool = True,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.n_out = n_out
        self.d = d
        self.d2 = d * d
        self.dtype = dtype
        self.cdtype = get_complex_dtype(dtype)
        self.device = torch.device(device)
        self.accumulate_state_frame = accumulate_state_frame
        self.vec_i = torch.eye(d, device=self.device, dtype=self.cdtype).reshape(1, -1)
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators.

        Args:
            None.
        """
        self.total_samples = 0
        self.raw_mu_acc = torch.zeros((self.n_out, self.d2), device=self.device, dtype=self.cdtype)
        self.prob_sum_acc = torch.zeros(self.n_out, device=self.device, dtype=self.dtype)
        if self.accumulate_state_frame:
            self.S_acc = torch.zeros((self.d2, self.d2), device=self.device, dtype=self.cdtype)

    def update_probs(self, probs: torch.Tensor, states: torch.Tensor) -> None:
        """Update from dense probabilities.

        Args:
            probs (torch.Tensor): Probability matrix with shape (n_out, n_batch).
            states (torch.Tensor): Flattened density matrices with shape (d^2, n_batch).
        """
        if probs.shape[0] != self.n_out or states.shape[0] != self.d2:
            raise ValueError("Expected probs (n_out, n_batch) and states (d^2, n_batch).")
        if probs.shape[1] != states.shape[1]:
            raise ValueError("probs and states must have the same batch size.")
        p = probs.to(device=self.device, dtype=self.dtype)
        rho = states.to(device=self.device, dtype=self.cdtype)
        self.total_samples += p.shape[1]
        self.raw_mu_acc.addmm_(p.to(dtype=self.cdtype), rho.T)
        self.prob_sum_acc += p.sum(dim=1)
        if self.accumulate_state_frame:
            self.S_acc.addmm_(rho, rho.adjoint())

    def update_single_shot(self, outcomes: torch.Tensor, states: torch.Tensor) -> None:
        """Update from one outcome per state.

        Args:
            outcomes (torch.Tensor): Outcome vector with shape (n_batch,).
            states (torch.Tensor): Flattened density matrices with shape (d^2, n_batch).
        """
        if states.shape[0] != self.d2:
            raise ValueError("Expected states with shape (d^2, n_batch).")
        idx = outcomes.to(device=self.device, dtype=torch.long).reshape(-1)
        if idx.numel() != states.shape[1]:
            raise ValueError("outcomes and states must have the same batch size.")
        rho_cols = states.to(device=self.device, dtype=self.cdtype)
        rho_rows = rho_cols.T.contiguous()
        self.raw_mu_acc.index_add_(0, idx, rho_rows)
        ones = torch.ones(idx.numel(), device=self.device, dtype=self.dtype)
        self.prob_sum_acc.index_add_(0, idx, ones)
        if self.accumulate_state_frame:
            self.S_acc.addmm_(rho_cols, rho_cols.adjoint())
        self.total_samples += idx.numel()

    def update_outcomes(self, outcomes: torch.Tensor, states: torch.Tensor) -> None:
        """Update from raw finite-shot outcomes.

        Args:
            outcomes (torch.Tensor): Outcome matrix with shape (n_batch, n_shots).
            states (torch.Tensor): Flattened density matrices with shape (d^2, n_batch).
        """
        if outcomes.ndim == 1:
            self.update_single_shot(outcomes, states)
            return
        if states.shape[0] != self.d2:
            raise ValueError("Expected states with shape (d^2, n_batch).")
        n_batch, n_shots = outcomes.shape
        if n_batch != states.shape[1]:
            raise ValueError("outcomes and states must have the same batch size.")
        if n_shots == 1:
            self.update_single_shot(outcomes[:, 0], states)
            return
        idx = outcomes.to(device=self.device, dtype=torch.long).reshape(-1)
        rho_cols = states.to(device=self.device, dtype=self.cdtype)
        rho_rows = rho_cols.T.contiguous().repeat_interleave(n_shots, dim=0) / n_shots
        self.raw_mu_acc.index_add_(0, idx, rho_rows)
        weights = torch.full((idx.numel(),), 1.0 / n_shots, device=self.device, dtype=self.dtype)
        self.prob_sum_acc.index_add_(0, idx, weights)
        if self.accumulate_state_frame:
            self.S_acc.addmm_(rho_cols, rho_cols.adjoint())
        self.total_samples += n_batch

    def estimated_povm(self, adaptive_state: bool = False, rcond: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute the estimated POVM after state-frame inversion.

        Args:
            adaptive_state (bool): If True, use empirical S^+; otherwise use analytic Haar S^{-1}.
            rcond (float): Pseudoinverse cutoff for empirical S.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: POVM (n_out, d^2) and traces (n_out, 1) or None.
        """
        if self.total_samples == 0:
            raise RuntimeError("No samples have been accumulated.")
        averaged_raw_mu = self.raw_mu_acc / self.total_samples
        if adaptive_state:
            if not self.accumulate_state_frame:
                raise RuntimeError("adaptive_state=True requires accumulate_state_frame=True.")
            s_inv = torch.linalg.pinv(self.S_acc / self.total_samples, rcond=rcond)
            return torch.matmul(averaged_raw_mu, s_inv.T), None
        avg_probs = (self.prob_sum_acc / self.total_samples).view(-1, 1).to(dtype=self.cdtype)
        est_povm = self.d * (self.d + 1) * averaged_raw_mu
        est_povm = est_povm + (-self.d) * torch.matmul(avg_probs, self.vec_i)
        return est_povm, (self.d * avg_probs).to(dtype=self.cdtype)

    def dual_frame(
        self,
        estimated_povm: torch.Tensor,
        prior_frame: bool = False,
        traces: torch.Tensor | None = None,
        rcond: float = 1e-10,
    ) -> torch.Tensor:
        """Compute the measurement dual frame.

        Args:
            estimated_povm (torch.Tensor): Estimated POVM with shape (n_out, d^2).
            prior_frame (bool): If True, use analytic Naimark prior; otherwise use empirical F^+.
            traces (torch.Tensor | None): POVM traces with shape (n_out, 1), or None.
            rcond (float): Pseudoinverse cutoff for empirical F.

        Returns:
            torch.Tensor: Dual frame with shape (d^2, n_out).
        """
        if prior_frame:
            if traces is None:
                traces = torch.sum(estimated_povm * self.vec_i, dim=1, keepdim=True)
            dual_t = (self.n_out + 1) * estimated_povm
            dual_t = dual_t - ((self.n_out + 1) / (self.d + 1)) * torch.matmul(traces, self.vec_i)
            return dual_t.T
        frame = torch.matmul(estimated_povm.T, estimated_povm.conj()).to(dtype=self.cdtype)
        frame_inv = torch.linalg.pinv(frame, rcond=rcond)
        return torch.matmul(frame_inv, estimated_povm.T)

    def layer(
        self,
        observable: torch.Tensor,
        adaptive_state: bool = False,
        prior_frame: bool = False,
        rcond_state: float = 1e-10,
        rcond_frame: float = 1e-10,
    ) -> torch.Tensor:
        """Compute a readout layer.

        Args:
            observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
            adaptive_state (bool): If True, use empirical S^+.
            prior_frame (bool): If True, use analytic measurement-frame prior.
            rcond_state (float): Pseudoinverse cutoff for empirical S.
            rcond_frame (float): Pseudoinverse cutoff for empirical F.

        Returns:
            torch.Tensor: Readout layer with shape (n_obs, n_out).
        """
        obs = _as_observable_matrix(observable, self.d2).to(device=self.device, dtype=self.cdtype)
        povm, traces = self.estimated_povm(adaptive_state=adaptive_state, rcond=rcond_state)
        dual = self.dual_frame(povm, prior_frame=prior_frame, traces=traces, rcond=rcond_frame)
        return torch.matmul(obs.conj(), dual).real.to(dtype=self.dtype)

    def layers(
        self,
        observable: torch.Tensor,
        methods: Iterable[str] = ("ost", "aost", "prior_ost", "prior_aost"),
        rcond_state: float = 1e-10,
        rcond_frame: float = 1e-10,
    ) -> dict[str, torch.Tensor]:
        """Compute several shadow readout layers.

        Args:
            observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
            methods (Iterable[str]): Method names.
            rcond_state (float): Pseudoinverse cutoff for empirical S.
            rcond_frame (float): Pseudoinverse cutoff for empirical F.

        Returns:
            dict[str, torch.Tensor]: Layers with shape (n_obs, n_out).
        """
        out: dict[str, torch.Tensor] = {}
        for method in methods:
            adaptive_state, prior_frame = method_flags(method)
            out[method] = self.layer(
                observable,
                adaptive_state=adaptive_state,
                prior_frame=prior_frame,
                rcond_state=rcond_state,
                rcond_frame=rcond_frame,
            )
        return out


# =====================================
# LINEAR READOUT ESTIMATOR
# =====================================


class LinearReadoutEstimator:
    """Dense probability-matrix baseline estimator.

    Args:
        n_out (int): Number of POVM outcomes.
        n_obs (int): Number of observables.
        device (torch.device | str): Accumulator device.
        dtype (torch.dtype): Real accumulator dtype.
    """

    def __init__(
        self,
        n_out: int,
        n_obs: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.n_out = n_out
        self.n_obs = n_obs
        self.device = torch.device(device)
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        """Reset the normal-equation accumulators.

        Args:
            None.
        """
        self.G = torch.zeros((self.n_out, self.n_out), device=self.device, dtype=self.dtype)
        self.C = torch.zeros((self.n_obs, self.n_out), device=self.device, dtype=self.dtype)

    def update_probs(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update dense normal-equation accumulators.

        Args:
            probs (torch.Tensor): Probability matrix with shape (n_out, n_batch).
            targets (torch.Tensor): Target matrix with shape (n_obs, n_batch).
        """
        if probs.shape[0] != self.n_out or targets.shape[0] != self.n_obs:
            raise ValueError("Expected probs (n_out, n_batch) and targets (n_obs, n_batch).")
        if probs.shape[1] != targets.shape[1]:
            raise ValueError("probs and targets must have the same batch size.")
        p = probs.to(device=self.device, dtype=self.dtype)
        y = targets.to(device=self.device, dtype=self.dtype)
        self.G.addmm_(p, p.T)
        self.C.addmm_(y, p.T)

    def layer_pinv(self, tol: float | int = 1e-10) -> torch.Tensor:
        """Compute the Moore-Penrose pseudoinverse readout layer.

        Args:
            tol (float | int): Relative tolerance or truncation rank.

        Returns:
            torch.Tensor: Readout layer with shape (n_obs, n_out).
        """
        if isinstance(tol, float):
            g_inv = torch.linalg.pinv(self.G, rtol=tol)
        elif isinstance(tol, int):
            g_inv = pinv_truncated(self.G, tol)
        else:
            raise ValueError("tol must be float or int.")
        return torch.matmul(self.C, g_inv).to(dtype=self.dtype)

    def layer_ridge(self, alpha: float = 1e-4) -> torch.Tensor:
        """Compute the ridge-regression readout layer.

        Args:
            alpha (float): Ridge regularization parameter.

        Returns:
            torch.Tensor: Readout layer with shape (n_obs, n_out).
        """
        eye = torch.eye(self.n_out, device=self.device, dtype=self.dtype)
        return torch.linalg.solve(self.G + alpha * eye, self.C, left=False).to(dtype=self.dtype)


# =====================================
# RUNNING OUTCOME STATISTICS
# =====================================


class RunningOutcomeStats:
    """Running counter for finite-shot outcome statistics.

    Args:
        n_out (int): Number of POVM outcomes.
        n_states (int): Number of states.
        device (torch.device | str): Accumulator device.
        dtype (torch.dtype): Real accumulator dtype.
    """

    def __init__(
        self,
        n_out: int,
        n_states: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.n_out = n_out
        self.n_states = n_states
        self.device = torch.device(device)
        self.dtype = dtype
        self.increments = (torch.arange(n_states, device=self.device) * n_out).unsqueeze(1)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated counts.

        Args:
            None.
        """
        self.count_acc = torch.zeros((self.n_out, self.n_states), device=self.device, dtype=self.dtype)
        self.total_shots = 0

    def update(self, outcomes_chunk: torch.Tensor) -> None:
        """Update counts from a new shot block.

        Args:
            outcomes_chunk (torch.Tensor): Outcome matrix with shape (n_states, n_new_shots).
        """
        if outcomes_chunk.ndim == 1:
            outcomes_chunk = outcomes_chunk.reshape(-1, 1)
        if outcomes_chunk.shape[0] != self.n_states:
            raise ValueError("Expected outcomes_chunk with shape (n_states, n_new_shots).")
        n_new = outcomes_chunk.shape[1]
        if n_new == 0:
            return
        linear_indices = (outcomes_chunk.to(device=self.device, dtype=torch.long) + self.increments[:, :n_new]).flatten()
        counts = torch.bincount(linear_indices, minlength=self.n_states * self.n_out)
        self.count_acc += counts.view(self.n_states, self.n_out).T.to(dtype=self.dtype)
        self.total_shots += n_new

    def counts(self) -> torch.Tensor:
        """Return accumulated counts.

        Returns:
            torch.Tensor: Count matrix with shape (n_out, n_states).
        """
        return self.count_acc

    def probabilities(self) -> torch.Tensor:
        """Return accumulated probabilities.

        Returns:
            torch.Tensor: Probability matrix with shape (n_out, n_states).
        """
        if self.total_shots == 0:
            return self.count_acc
        return (self.count_acc / self.total_shots).contiguous()


# =====================================
# CONVENIENCE
# =====================================


def train_linear_from_probs(
    probs: torch.Tensor,
    observable: torch.Tensor,
    states: torch.Tensor,
    tol: float | int = 1e-10,
) -> torch.Tensor:
    """Train a dense pseudoinverse readout from probabilities.

    Args:
        probs (torch.Tensor): Probability matrix with shape (n_out, n_states).
        observable (torch.Tensor): Observables with shape (n_obs, d^2).
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).
        tol (float | int): Relative tolerance or truncation rank.

    Returns:
        torch.Tensor: Readout layer with shape (n_obs, n_out).
    """
    obs = _as_observable_matrix(observable, states.shape[0])
    targets = get_observables(obs, states)
    est = LinearReadoutEstimator(probs.shape[0], obs.shape[0], device=probs.device, dtype=probs.dtype)
    est.update_probs(probs, targets)
    return est.layer_pinv(tol=tol)
