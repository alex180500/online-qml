from __future__ import annotations

from math import sqrt
from typing import Iterable

import torch

from .quantum import get_test_mse, infinite_stats, sample_dm

# =====================================
# HAAR BIAS AND VARIANCE
# =====================================


class HaarBiasVariance:
    """Haar-averaged bias and variance evaluator for one observable.

    Args:
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        observable (torch.Tensor): Flattened observable with shape (1, d^2) or (d^2,).
    """

    def __init__(self, povm: torch.Tensor, observable: torch.Tensor):
        self.povm = povm
        if observable.ndim == 1:
            observable = observable.reshape(1, -1)
        self.observable = observable
        if self.observable.shape[0] != 1:
            raise ValueError(
                "HaarBiasVariance currently supports one observable with shape (1, d^2)."
            )
        self.precompute()

    def precompute(self) -> None:
        """Precompute POVM and observable contractions.

        Args:
            None.
        """
        n_out, d2 = self.povm.shape
        d = int(sqrt(d2))
        self.coeff = 1.0 / (d * (d + 1))
        mu = self.povm.reshape(n_out, d, d)
        obs = self.observable.reshape(d, d)
        tr_mu = torch.einsum("aii->a", mu)
        tr_mu_mu = torch.einsum("aij,bji->ab", mu, mu)
        self.tr_mu_over_d = tr_mu.real / d
        self.pair_term = (torch.outer(tr_mu, tr_mu) + tr_mu_mu).real
        tr_obs = torch.trace(obs)
        tr_mu_obs = torch.einsum("aij,ji->a", mu, obs)
        self.cross = (tr_obs * tr_mu + tr_mu_obs).real
        self.bias_const = self.coeff * (tr_obs * tr_obs + torch.trace(obs @ obs)).real

    def evaluate(self, layer: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate a readout layer.

        Args:
            layer (torch.Tensor): Readout layer with shape (1, n_out).

        Returns:
            dict[str, torch.Tensor]: variance, bias2 and mse_exact_probs scalars.
        """
        e = layer.reshape(-1)
        shared = self.coeff * torch.einsum("a,ab,b->", e, self.pair_term, e)
        variance = torch.sum(e**2 * self.tr_mu_over_d) - shared
        bias2 = shared - 2.0 * self.coeff * torch.sum(e * self.cross) + self.bias_const
        return {
            "variance": variance.real,
            "bias2": bias2.real,
            "mse_exact_probs": bias2.real,
        }

    def evaluate_tuple(self, layer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a readout layer with the old tuple convention.

        Args:
            layer (torch.Tensor): Readout layer with shape (1, n_out).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: variance and bias2 scalars.
        """
        res = self.evaluate(layer)
        return res["variance"], res["bias2"]


# =====================================
# LAYER EVALUATION
# =====================================


def evaluate_layers_haar(
    layers: dict[str, torch.Tensor],
    povm: torch.Tensor,
    observable: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate layers by Haar bias and variance.

    Args:
        layers (dict[str, torch.Tensor]): Layers with shape (1, n_out) or (..., 1, n_out).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        observable (torch.Tensor): Observable with shape (1, d^2) or (d^2,).

    Returns:
        dict[str, torch.Tensor]: Metric tensors keyed as method_metric.
    """
    evaluator = HaarBiasVariance(povm, observable)
    out: dict[str, torch.Tensor] = {}
    for method, layer in layers.items():
        if layer.ndim == 2:
            metrics = evaluator.evaluate(layer)
            for key, val in metrics.items():
                out[f"{method}_{key}"] = val
        else:
            flat = layer.reshape(-1, layer.shape[-2], layer.shape[-1])
            vals: dict[str, list[torch.Tensor]] = {
                "variance": [],
                "bias2": [],
                "mse_exact_probs": [],
            }
            for item in flat:
                metrics = evaluator.evaluate(item)
                for key in vals:
                    vals[key].append(metrics[key])
            for key, seq in vals.items():
                out[f"{method}_{key}"] = torch.stack(seq).reshape(layer.shape[:-2])
    return out


def evaluate_layers_empirical_mse(
    layers: dict[str, torch.Tensor],
    povm: torch.Tensor,
    observable: torch.Tensor,
    test_states: int | torch.Tensor = 10000,
    d: int | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate layers by empirical exact-probability test MSE.

    Args:
        layers (dict[str, torch.Tensor]): Layers with shape (n_obs, n_out).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        observable (torch.Tensor): Observables with shape (n_obs, d^2).
        test_states (int | torch.Tensor): Number of test states or states with shape (d^2, n_test).
        d (int | None): Hilbert-space dimension.

    Returns:
        dict[str, torch.Tensor]: MSE scalars keyed as method_mse_test.
    """
    if d is None:
        d = int(round(povm.shape[1] ** 0.5))
    if isinstance(test_states, int):
        states = sample_dm(test_states, d=d, device=povm.device, dtype=povm.dtype)
    else:
        states = test_states.to(device=povm.device, dtype=povm.dtype)
    probs = infinite_stats(povm, states)
    obs_vals = torch.matmul(observable.to(povm.dtype).conj(), states).real
    return {
        f"{method}_mse_test": get_test_mse(layer, obs_vals, probs)
        for method, layer in layers.items()
    }


# =====================================
# BETA FITS
# =====================================


def fit_beta_coefficients(
    mse: torch.Tensor,
    shot_grid: torch.Tensor,
    train_grid: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Fit beta0 + beta1/(N n) + beta2/n to MSE values.

    Args:
        mse (torch.Tensor): MSE matrix with shape (n_shots_grid, n_train_grid).
        shot_grid (torch.Tensor): Shot values with shape (n_shots_grid,).
        train_grid (torch.Tensor): Training-state values with shape (n_train_grid,).
        mask (torch.Tensor | None): Boolean fit mask with shape (n_shots_grid, n_train_grid).

    Returns:
        dict[str, torch.Tensor]: beta0, beta1, beta2 and residual scalars.
    """
    shots = shot_grid.to(dtype=mse.dtype, device=mse.device)
    trains = train_grid.to(dtype=mse.dtype, device=mse.device)
    s_grid, n_grid = torch.meshgrid(shots, trains, indexing="ij")
    x = torch.stack(
        [
            torch.ones_like(s_grid),
            1.0 / (s_grid * n_grid),
            1.0 / n_grid,
        ],
        dim=-1,
    )
    y = mse
    if mask is not None:
        x = x[mask]
        y = y[mask]
    else:
        x = x.reshape(-1, 3)
        y = y.reshape(-1)
    sol = torch.linalg.lstsq(x, y).solution
    residual = torch.mean((x @ sol - y) ** 2)
    return {
        "beta0": sol[0],
        "beta1": sol[1],
        "beta2": sol[2],
        "fit_residual": residual,
    }


# =====================================
# PREDICTION GEOMETRY
# =====================================


def prediction_geometry(
    layer: torch.Tensor,
    povm: torch.Tensor,
    observable: torch.Tensor,
    states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute true-vs-predicted calibration diagnostics.

    Args:
        layer (torch.Tensor): Readout layer with shape (1, n_out).
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        observable (torch.Tensor): Observable with shape (1, d^2) or (d^2,).
        states (torch.Tensor): Flattened density matrices with shape (d^2, n_states).

    Returns:
        dict[str, torch.Tensor]: True values, predictions, slope, intercept, Pearson r and MSE.
    """
    if observable.ndim == 1:
        observable = observable.reshape(1, -1)
    probs = infinite_stats(povm, states)
    true = torch.matmul(observable.to(states.dtype).conj(), states).real.reshape(-1)
    pred = torch.matmul(layer, probs).real.reshape(-1)
    x = true - true.mean()
    y = pred - pred.mean()
    slope = torch.sum(x * y) / torch.sum(x * x)
    intercept = pred.mean() - slope * true.mean()
    pearson = torch.sum(x * y) / torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    mse = torch.mean((pred - true) ** 2)
    return {
        "true": true,
        "pred": pred,
        "slope": slope,
        "intercept": intercept,
        "pearson": pearson,
        "mse": mse,
    }
