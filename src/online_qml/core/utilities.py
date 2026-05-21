import time
from collections.abc import Callable
from pathlib import Path
from random import randint
from typing import Any

import torch

from .internals import seed_all

MAX_SEED = 2**32 - 1


def seed_run(root: str | Path, seed_id: int, seed: int) -> Path:
    """Prepare a per-seed run folder and seed Torch RNGs."""
    seed_dir = Path(root) / f"seed_{seed_id}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    return seed_dir


def random_seed() -> int:
    """Return a random seed for independent simulation runs."""
    return randint(0, MAX_SEED)


def timed(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return its result with elapsed wall time in seconds."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def logspace_int(start: int, stop: int, num: int) -> torch.Tensor:
    """Generate a unique increasing integer logarithmic grid."""
    if start <= 0 or stop < start:
        raise ValueError("Require 0 < start <= stop.")
    if num > stop - start + 1:
        raise ValueError(
            f"Cannot generate {num} unique integers between {start} and {stop}."
        )

    vals = torch.logspace(
        torch.log10(torch.tensor(start, dtype=torch.float64)),
        torch.log10(torch.tensor(stop, dtype=torch.float64)),
        num,
        dtype=torch.float64,
    )
    out = torch.empty(num, dtype=torch.int64)
    current = int(start)
    out[0] = current
    for i in range(1, num):
        current = max(current + 1, int(round(float(vals[i]))))
        out[i] = current
    return out


def parse_tol(value: str) -> float | int:
    """Parse a pseudoinverse tolerance or truncation rank."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def check_folder(folder: str | Path, pattern: str, expected: int) -> bool:
    """Check whether a folder contains the expected number of files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    count = len(list(folder.glob(pattern)))
    ok = count == expected
    status = "ok" if ok else "incomplete"
    print(f"{folder}: {count}/{expected} {pattern} ({status})")
    return ok
