"""Online QML tools for online shadow training simulations."""

from .core import (
    LayerResult,
    MetricResult,
    SimulationContext,
    SimulationData,
    get_complex_dtype,
    get_real_dtype,
    get_torch_device,
    load_data,
    load_simulation_data,
    logspace_int,
    save_data,
    save_simulation_data,
)
from .estimators import (
    LinearReadoutEstimator,
    RunningOutcomeStats,
    ShadowReadoutEstimator,
)
from .evaluation import HaarBiasVariance
from .experiments import (
    evaluate_layer_result_haar,
    make_layers_ntrain_grid,
    make_layers_shot_grid,
    measurement_frame_distance_grid,
    state_frame_distance_grid,
)
from .quantum import (
    sample_dm,
    sample_povm,
    sample_traceless_operator,
    shots_outcome,
)
from importlib.metadata import version

__all__ = [
    "HaarBiasVariance",
    "LayerResult",
    "LinearReadoutEstimator",
    "MetricResult",
    "RunningOutcomeStats",
    "SimulationContext",
    "ShadowReadoutEstimator",
    "SimulationData",
    "evaluate_layer_result_haar",
    "get_complex_dtype",
    "get_real_dtype",
    "get_torch_device",
    "load_data",
    "load_simulation_data",
    "logspace_int",
    "make_layers_ntrain_grid",
    "make_layers_shot_grid",
    "measurement_frame_distance_grid",
    "sample_dm",
    "sample_povm",
    "sample_traceless_operator",
    "save_data",
    "save_simulation_data",
    "shots_outcome",
    "state_frame_distance_grid",
]

__version__ = version("online-qml")
