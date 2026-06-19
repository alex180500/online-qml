from math import sqrt
import torch

from .quantum import as_observable_matrix

# ----- HAAR BIAS AND VARIANCE -----


class HaarBiasVariance:
    """Haar-averaged bias and variance evaluator for one observable.

    Args:
        povm (torch.Tensor): Flattened POVM elements with shape (n_out, d^2).
        observable (torch.Tensor): Flattened observable with shape (1, d^2) or (d^2,).
    """

    def __init__(self, povm: torch.Tensor, observable: torch.Tensor):
        self.povm = povm
        self.observable = as_observable_matrix(observable, povm.shape[1]).to(
            device=povm.device, dtype=povm.dtype
        )
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
        """Evaluate one readout layer.

        Args:
            layer (torch.Tensor): Readout layer with shape (1, n_out).

        Returns:
            dict[str, torch.Tensor]: variance and bias2 scalars.
        """
        e = layer.reshape(-1)
        shared = self.coeff * torch.einsum("a,ab,b->", e, self.pair_term, e)
        variance = torch.sum(e**2 * self.tr_mu_over_d) - shared
        bias2 = shared - 2.0 * self.coeff * torch.sum(e * self.cross) + self.bias_const
        return {
            "variance": variance.real,
            "bias2": bias2.real,
        }


# ----- LAYER EVALUATION -----


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
            }
            for item in flat:
                metrics = evaluator.evaluate(item)
                for key in vals:
                    vals[key].append(metrics[key])
            for key, seq in vals.items():
                out[f"{method}_{key}"] = torch.stack(seq).reshape(layer.shape[:-2])
    return out
