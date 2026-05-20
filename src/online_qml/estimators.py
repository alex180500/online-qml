from typing import Iterable
import torch

from .core.internals import complex_dtype
from .core.methods import shadow_method_flags
from .quantum import (
    as_observable_matrix,
    get_observables,
    shots_to_statistics,
    vec_identity,
)

# ----- HELPERS -----


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


# ----- SHADOW READOUT ESTIMATOR -----


class ShadowReadoutEstimator:
    """Online shadow estimator for QELM readout layers.

    Args:
        n_out (int): Number of POVM outcomes.
        d (int): Hilbert-space dimension.
        state_prior_frame (torch.Tensor | None): Optional prior state frame with shape (d^2, d^2).
        accumulate_state_frame (bool | None): Whether to accumulate empirical F_rho with shape (d^2, d^2). If None, infer from methods.
        device (torch.device | str): Accumulator device.
        dtype (torch.dtype): Real accumulator dtype.
        methods (Iterable[str]): Shadow methods to compute.
    """

    def __init__(
        self,
        n_out: int,
        d: int,
        state_prior_frame: torch.Tensor | None = None,
        accumulate_state_frame: bool | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        *,
        methods: Iterable[str],
    ):
        self.n_out = n_out
        self.d = d
        self.d2 = d * d
        self.dtype = dtype
        self.cdtype = complex_dtype(dtype)
        self.device = torch.device(device)
        self.methods = tuple(methods)
        self._method_flags = {
            method: shadow_method_flags(method) for method in self.methods
        }
        if accumulate_state_frame is None:
            accumulate_state_frame = any(
                not use_state_prior
                for use_state_prior, _ in self._method_flags.values()
            )
        self.accumulate_state_frame = accumulate_state_frame
        self.vec_i = vec_identity(d, device=self.device, dtype=self.cdtype).reshape(
            1, -1
        )
        self.state_prior_frame = None
        if state_prior_frame is not None:
            self.state_prior_frame = state_prior_frame.to(
                device=self.device, dtype=self.cdtype
            )
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators.

        Args:
            None.
        """
        self.total_samples = 0
        self.raw_mu_acc = torch.zeros(
            (self.n_out, self.d2), device=self.device, dtype=self.cdtype
        )
        self.prob_sum_acc = torch.zeros(
            self.n_out, device=self.device, dtype=self.dtype
        )
        if self.accumulate_state_frame:
            self.state_frame_acc = torch.zeros(
                (self.d2, self.d2), device=self.device, dtype=self.cdtype
            )

    def update(self, outcomes: torch.Tensor, states: torch.Tensor) -> None:
        """Update from raw outcomes.

        Args:
            outcomes (torch.Tensor): Outcome vector (n_batch,) or matrix (n_batch, n_shots).
            states (torch.Tensor): Flattened density matrices with shape (d^2, n_batch).
        """
        self.update_outcomes(outcomes, states)

    def update_probs(self, probs: torch.Tensor, states: torch.Tensor) -> None:
        """Update from dense probabilities.

        Args:
            probs (torch.Tensor): Probability matrix with shape (n_out, n_batch).
            states (torch.Tensor): Flattened density matrices with shape (d^2, n_batch).
        """
        if probs.shape[0] != self.n_out or states.shape[0] != self.d2:
            raise ValueError(
                "Expected probs (n_out, n_batch) and states (d^2, n_batch)."
            )
        if probs.shape[1] != states.shape[1]:
            raise ValueError("probs and states must have the same batch size.")
        p = probs.to(device=self.device, dtype=self.dtype)
        rho = states.to(device=self.device, dtype=self.cdtype)
        self.total_samples += p.shape[1]
        self.raw_mu_acc.addmm_(p.to(dtype=self.cdtype), rho.T)
        self.prob_sum_acc += p.sum(dim=1)
        if self.accumulate_state_frame:
            self.state_frame_acc.addmm_(rho, rho.adjoint())

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
            self.state_frame_acc.addmm_(rho_cols, rho_cols.adjoint())
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
        weights = torch.full(
            (idx.numel(),), 1.0 / n_shots, device=self.device, dtype=self.dtype
        )
        self.prob_sum_acc.index_add_(0, idx, weights)
        if self.accumulate_state_frame:
            self.state_frame_acc.addmm_(rho_cols, rho_cols.adjoint())
        self.total_samples += n_batch

    def estimated_povm(
        self, use_state_prior: bool = False, rcond: float = 1e-10
    ) -> torch.Tensor:
        """Compute the estimated POVM after state-frame inversion.

        Args:
            use_state_prior (bool): If True, use the state-frame prior. If False, use empirical F_rho^+.
            rcond (float): Pseudoinverse cutoff for empirical or explicit prior frames.

        Returns:
            torch.Tensor: Estimated POVM with shape (n_out, d^2).
        """
        if self.total_samples == 0:
            raise RuntimeError("No samples have been accumulated.")
        raw_mean = self.raw_mu_acc / self.total_samples
        if use_state_prior:
            if self.state_prior_frame is not None:
                prior_inv = torch.linalg.pinv(self.state_prior_frame, rcond=rcond)
                return torch.matmul(raw_mean, prior_inv.T)
            prob_mean = (self.prob_sum_acc / self.total_samples).view(-1, 1)
            prob_mean = prob_mean.to(dtype=self.cdtype)
            return self.d * (self.d + 1) * raw_mean - self.d * torch.matmul(
                prob_mean, self.vec_i
            )
        if not self.accumulate_state_frame:
            raise RuntimeError("Empirical state-frame inversion was not accumulated.")
        frame = self.state_frame_acc / self.total_samples
        frame_inv = torch.linalg.pinv(frame, rcond=rcond)
        return torch.matmul(raw_mean, frame_inv.T)

    def dual_frame(
        self,
        estimated_povm: torch.Tensor,
        use_povm_prior: bool = False,
        rcond: float = 1e-10,
    ) -> torch.Tensor:
        """Compute the measurement dual frame.

        Args:
            estimated_povm (torch.Tensor): Estimated POVM with shape (n_out, d^2).
            use_povm_prior (bool): If True, use the analytic Naimark prior. If False, use empirical F_mu^+.
            rcond (float): Pseudoinverse cutoff for empirical F_mu.

        Returns:
            torch.Tensor: Dual frame with shape (d^2, n_out).
        """
        if use_povm_prior:
            traces = torch.sum(estimated_povm * self.vec_i, dim=1, keepdim=True)
            dual_t = (self.n_out + 1) * estimated_povm
            dual_t = dual_t - ((self.n_out + 1) / (self.d + 1)) * torch.matmul(
                traces, self.vec_i
            )
            return dual_t.T
        frame = torch.matmul(estimated_povm.T, estimated_povm.conj()).to(
            dtype=self.cdtype
        )
        frame_inv = torch.linalg.pinv(frame, rcond=rcond)
        return torch.matmul(frame_inv, estimated_povm.T)

    def layer(
        self,
        observable: torch.Tensor,
        use_state_prior: bool = False,
        use_povm_prior: bool = False,
        rcond_state: float = 1e-10,
        rcond_frame: float = 1e-10,
    ) -> torch.Tensor:
        """Compute one readout layer.

        Args:
            observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
            use_state_prior (bool): If True, use prior F_rho.
            use_povm_prior (bool): If True, use prior F_mu.
            rcond_state (float): Pseudoinverse cutoff for F_rho.
            rcond_frame (float): Pseudoinverse cutoff for F_mu.

        Returns:
            torch.Tensor: Readout layer with shape (n_obs, n_out).
        """
        obs = as_observable_matrix(observable, self.d2).to(
            device=self.device, dtype=self.cdtype
        )
        povm = self.estimated_povm(use_state_prior=use_state_prior, rcond=rcond_state)
        dual = self.dual_frame(povm, use_povm_prior=use_povm_prior, rcond=rcond_frame)
        return torch.matmul(obs.conj(), dual).real.to(dtype=self.dtype)

    def layers(
        self,
        observable: torch.Tensor,
        methods: Iterable[str] | None = None,
        rcond_state: float = 1e-10,
        rcond_frame: float = 1e-10,
    ) -> dict[str, torch.Tensor]:
        """Compute several shadow readout layers.

        Args:
            observable (torch.Tensor): Observables with shape (n_obs, d^2), (d^2, n_obs), or (d^2,).
            methods (Iterable[str] | None): Shadow method names. If None, use the methods configured on the estimator.
            rcond_state (float): Pseudoinverse cutoff for F_rho.
            rcond_frame (float): Pseudoinverse cutoff for F_mu.

        Returns:
            dict[str, torch.Tensor]: Layers with shape (n_obs, n_out).
        """
        selected_methods = self.methods if methods is None else tuple(methods)
        selected_flags: dict[str, tuple[bool, bool]] = {}
        for method in selected_methods:
            if method in self._method_flags:
                selected_flags[method] = self._method_flags[method]
            else:
                selected_flags[method] = shadow_method_flags(method)

        obs = as_observable_matrix(observable, self.d2).to(
            device=self.device, dtype=self.cdtype
        )
        povms: dict[bool, torch.Tensor] = {}
        out: dict[str, torch.Tensor] = {}
        for method, (use_state_prior, use_povm_prior) in selected_flags.items():
            if use_state_prior not in povms:
                povms[use_state_prior] = self.estimated_povm(
                    use_state_prior=use_state_prior, rcond=rcond_state
                )
            dual = self.dual_frame(
                povms[use_state_prior],
                use_povm_prior=use_povm_prior,
                rcond=rcond_frame,
            )
            out[method] = torch.matmul(obs.conj(), dual).real.to(dtype=self.dtype)
        return out


# ----- LINEAR READOUT ESTIMATOR -----


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
        self.G = torch.zeros(
            (self.n_out, self.n_out), device=self.device, dtype=self.dtype
        )
        self.C = torch.zeros(
            (self.n_obs, self.n_out), device=self.device, dtype=self.dtype
        )

    def update_probs(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update dense normal-equation accumulators.

        Args:
            probs (torch.Tensor): Probability matrix with shape (n_out, n_batch).
            targets (torch.Tensor): Target matrix with shape (n_obs, n_batch).
        """
        if probs.shape[0] != self.n_out or targets.shape[0] != self.n_obs:
            raise ValueError(
                "Expected probs (n_out, n_batch) and targets (n_obs, n_batch)."
            )
        if probs.shape[1] != targets.shape[1]:
            raise ValueError("probs and targets must have the same batch size.")
        p = probs.to(device=self.device, dtype=self.dtype)
        y = targets.to(device=self.device, dtype=self.dtype)
        self.G.addmm_(p, p.T)
        self.C.addmm_(y, p.T)

    def update_outcomes(
        self,
        outcomes: torch.Tensor,
        states: torch.Tensor,
        observable: torch.Tensor,
        n_out: int | None = None,
    ) -> None:
        """Update from raw outcomes through empirical probabilities.

        Args:
            outcomes (torch.Tensor): Outcome matrix with shape (n_batch, n_shots).
            states (torch.Tensor): Flattened density matrices with shape (d^2, n_batch).
            observable (torch.Tensor): Observables with shape (n_obs, d^2).
            n_out (int | None): Number of POVM outcomes.
        """
        if n_out is None:
            n_out = self.n_out
        probs = shots_to_statistics(outcomes, n_out).to(
            device=self.device, dtype=self.dtype
        )
        obs = as_observable_matrix(observable, states.shape[0])
        targets = get_observables(obs, states).to(device=self.device, dtype=self.dtype)
        self.update_probs(probs, targets)

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
        return torch.linalg.solve(self.G + alpha * eye, self.C, left=False).to(
            dtype=self.dtype
        )


# ----- RUNNING OUTCOME STATISTICS -----


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
        self.increments = (torch.arange(n_states, device=self.device) * n_out).view(
            -1, 1
        )
        self.reset()

    def reset(self) -> None:
        """Reset accumulated counts.

        Args:
            None.
        """
        self.count_acc = torch.zeros(
            (self.n_out, self.n_states), device=self.device, dtype=self.dtype
        )
        self.total_shots = 0

    def update(self, outcomes_chunk: torch.Tensor) -> None:
        """Update counts from a new shot block.

        Args:
            outcomes_chunk (torch.Tensor): Outcome matrix with shape (n_states, n_new_shots).
        """
        if outcomes_chunk.ndim == 1:
            outcomes_chunk = outcomes_chunk.reshape(-1, 1)
        if outcomes_chunk.shape[0] != self.n_states:
            raise ValueError(
                "Expected outcomes_chunk with shape (n_states, n_new_shots)."
            )
        n_new = outcomes_chunk.shape[1]
        if n_new == 0:
            return
        linear_indices = (
            outcomes_chunk.to(device=self.device, dtype=torch.long) + self.increments
        ).flatten()
        counts = torch.bincount(linear_indices, minlength=self.n_states * self.n_out)
        self.count_acc += counts.view(self.n_states, self.n_out).T.to(dtype=self.dtype)
        self.total_shots += n_new

    def probabilities(self) -> torch.Tensor:
        """Return accumulated probabilities.

        Returns:
            torch.Tensor: Probability matrix with shape (n_out, n_states).
        """
        if self.total_shots == 0:
            return self.count_acc
        return (self.count_acc / self.total_shots).contiguous()
