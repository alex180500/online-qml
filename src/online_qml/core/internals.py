import os
import sys
import torch


def real_dtype(precision: str | torch.dtype) -> torch.dtype:
    """Return a real torch dtype."""
    if isinstance(precision, torch.dtype):
        if precision in (torch.float32, torch.float64):
            return precision
        raise ValueError("precision dtype must be torch.float32 or torch.float64.")
    if precision == "float32":
        return torch.float32
    if precision == "float64":
        return torch.float64
    raise ValueError("precision must be 'float32' or 'float64'.")


def complex_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Return the complex dtype associated with a real dtype."""
    rdtype = real_dtype(dtype) if isinstance(dtype, str) else dtype
    if rdtype == torch.float32:
        return torch.complex64
    if rdtype == torch.float64:
        return torch.complex128
    raise ValueError("dtype must be torch.float32 or torch.float64.")


def seed_all(seed: int) -> None:
    """Seed Torch RNGs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_setup(
    device: str | torch.device = "cpu",
    precision: str | torch.dtype = "float64",
    threads: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[torch.device, torch.dtype, torch.dtype]:
    """Set Torch runtime options used by scripts."""
    if threads is not None:
        torch.set_num_threads(threads)

    if seed is not None:
        seed_all(seed)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{device}' is unavailable. Falling back to CPU.")
        device = torch.device("cpu")

    rdtype = real_dtype(precision)
    cdtype = complex_dtype(rdtype)

    if verbose:
        print(f"device: {device}")
        print(f"precision: {precision}")
        print(f"torch threads: {torch.get_num_threads()} / os cores: {os.cpu_count()}")
        print(f"working folder: {os.getcwd()}")
        print(f"command string: {' '.join(sys.argv)}")

    return device, rdtype, cdtype
