"""Online QML tools for online shadow training simulations."""

from .core import (
    LayerResult,
    MetricResult,
    SimulationData,
    get_torch_device,
    logspace_int,
)
from .estimators import (
    LinearReadoutEstimator,
    RunningOutcomeStats,
    ShadowReadoutEstimator,
)
from .evaluation import HaarBiasVariance
from importlib.metadata import version

__all__ = [
    "HaarBiasVariance",
    "LayerResult",
    "LinearReadoutEstimator",
    "MetricResult",
    "RunningOutcomeStats",
    "ShadowReadoutEstimator",
    "SimulationData",
    "get_torch_device",
    "logspace_int",
]

__version__ = version("online-qml")
