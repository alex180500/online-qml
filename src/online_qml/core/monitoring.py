import os
import sys
import threading
import time
from dataclasses import dataclass

import torch

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only.
    resource = None

try:
    PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
except (AttributeError, OSError, ValueError):
    PAGE_SIZE = 4096


@dataclass(frozen=True)
class ResourceUsage:
    """Summary of process resource usage collected by ResourceMonitor."""

    wall_time_s: float
    memory_avg_bytes: int
    memory_max_bytes: int
    cpu_avg_percent: float
    cpu_max_percent: float
    cuda_allocated_max_bytes: int | None = None
    cuda_reserved_max_bytes: int | None = None

    def format(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            "Resource usage:",
            (
                "  memory rss "
                f"avg={format_bytes(self.memory_avg_bytes)} "
                f"max={format_bytes(self.memory_max_bytes)}"
            ),
            (
                "  cpu process "
                f"avg={self.cpu_avg_percent:.1f}% "
                f"max={self.cpu_max_percent:.1f}%"
            ),
        ]
        if self.cuda_allocated_max_bytes is not None:
            lines.append(
                "  cuda memory "
                f"allocated_max={format_bytes(self.cuda_allocated_max_bytes)} "
                f"reserved_max={format_bytes(self.cuda_reserved_max_bytes or 0)}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


class ResourceMonitor:
    """Sample process resources and PyTorch CUDA peak memory."""

    def __init__(
        self,
        interval: float = 0.5,
        device: str | torch.device | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive.")
        self.interval = interval
        self.device = torch.device(device) if device is not None else None
        self._cuda_device: torch.device | None = None
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._memory_samples: list[int] = []
        self._cpu_percent_samples: list[float] = []
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._end_wall: float | None = None
        self._end_cpu: float | None = None

    def __enter__(self) -> "ResourceMonitor":
        if self._thread is not None:
            raise RuntimeError("ResourceMonitor cannot be reused.")

        self._cuda_device = self._resolve_cuda_device()
        if self._cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(self._cuda_device)

        self._stop = threading.Event()
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._memory_samples.append(current_rss_bytes())
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def stop(self) -> ResourceUsage:
        """Stop sampling and return the collected resource usage."""
        if self._end_wall is None:
            if self._stop is None:
                raise RuntimeError("ResourceMonitor has not been started.")
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
            self._end_wall = time.perf_counter()
            self._end_cpu = time.process_time()
            self._memory_samples.append(current_rss_bytes())
        return self.summary()

    def summary(self) -> ResourceUsage:
        """Return the current resource usage summary."""
        if self._end_wall is None or self._end_cpu is None:
            return self.stop()

        wall_time = max(self._end_wall - self._start_wall, 1e-12)
        cpu_avg = 100.0 * (self._end_cpu - self._start_cpu) / wall_time
        cpu_max = max(self._cpu_percent_samples, default=cpu_avg)
        memory_samples = self._memory_samples or [current_rss_bytes()]
        memory_max = max(max(memory_samples), max_rss_bytes())
        cuda_allocated = None
        cuda_reserved = None
        if self._cuda_device is not None:
            cuda_allocated = torch.cuda.max_memory_allocated(self._cuda_device)
            cuda_reserved = torch.cuda.max_memory_reserved(self._cuda_device)

        return ResourceUsage(
            wall_time_s=wall_time,
            memory_avg_bytes=int(sum(memory_samples) / len(memory_samples)),
            memory_max_bytes=memory_max,
            cpu_avg_percent=cpu_avg,
            cpu_max_percent=cpu_max,
            cuda_allocated_max_bytes=cuda_allocated,
            cuda_reserved_max_bytes=cuda_reserved,
        )

    def _sample_loop(self) -> None:
        if self._stop is None:
            return

        last_wall = self._start_wall
        last_cpu = self._start_cpu
        while not self._stop.wait(self.interval):
            now_wall = time.perf_counter()
            now_cpu = time.process_time()
            elapsed = now_wall - last_wall
            if elapsed > 0:
                self._cpu_percent_samples.append(100.0 * (now_cpu - last_cpu) / elapsed)
            self._memory_samples.append(current_rss_bytes())
            last_wall = now_wall
            last_cpu = now_cpu

    def _resolve_cuda_device(self) -> torch.device | None:
        if not torch.cuda.is_available():
            return None
        if self.device is None:
            return torch.device("cuda")
        if self.device.type == "cuda":
            return self.device
        return None


def current_rss_bytes() -> int:
    """Return current process resident memory in bytes when available."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            resident_pages = int(f.readline().split()[1])
    except (OSError, IndexError, ValueError):
        return max_rss_bytes()
    return resident_pages * PAGE_SIZE


def max_rss_bytes() -> int:
    """Return maximum process resident memory in bytes when available."""
    if resource is None:
        return 0
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(max_rss)
    return int(max_rss) * 1024


def format_bytes(value: int | float) -> str:
    """Format bytes using binary units."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"
