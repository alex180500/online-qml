"""Online QML tools for online shadow training simulations."""

from importlib.metadata import PackageNotFoundError, version as _version

from .core import (
    LayerResult,
    MetricResult,
    SimulationData,
    load_pt,
    logspace_int,
    random_seed,
    save_metrics,
    save_json,
    save_pt,
    seed_all,
    seed_run,
    timed,
    torch_setup,
    training_methods,
)
from .estimators import (
    LinearReadoutEstimator,
    RunningOutcomeStats,
    ShadowReadoutEstimator,
    pinv_truncated,
)
from .evaluation import (
    HaarBiasVariance,
    evaluate_layers_haar,
    haar_metrics,
)
from .experiments import (
    ntrain_layers,
    sample_data,
    shot_layers,
)
from .frames import (
    frame_distance_summary,
    frame_relative_spectrum,
    haar_state_frame,
    measurement_frame,
    naimark_measurement_frame_prior,
    povm_frame_distance_grid,
    povm_frame_distances,
    state_frame,
    state_frame_distance_grid,
    state_frame_distances,
    trace_superoperator,
    vec_identity,
)
from .quantum import (
    as_observable_matrix,
    get_observables,
    infinite_stats,
    sample_dm,
    sample_povm,
    sample_states,
    sample_unitary,
    shots_outcome,
    shots_to_statistics,
)

try:
    __version__ = _version("online-qml")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # Version.
    "__version__",
    # Core containers and methods.
    "SimulationData",
    "LayerResult",
    "MetricResult",
    "training_methods",
    # Runtime, IO and small utilities.
    "torch_setup",
    "random_seed",
    "timed",
    "seed_all",
    "seed_run",
    "logspace_int",
    "save_pt",
    "load_pt",
    "save_json",
    "save_metrics",
    # Quantum state, observable and POVM utilities.
    "sample_states",
    "sample_dm",
    "as_observable_matrix",
    "get_observables",
    "sample_unitary",
    "sample_povm",
    "infinite_stats",
    "shots_outcome",
    "shots_to_statistics",
    # Frame utilities.
    "vec_identity",
    "trace_superoperator",
    "haar_state_frame",
    "naimark_measurement_frame_prior",
    "state_frame",
    "measurement_frame",
    "frame_relative_spectrum",
    "frame_distance_summary",
    # Evaluation helpers.
    "HaarBiasVariance",
    "evaluate_layers_haar",
    "haar_metrics",
    # Estimators and helpers.
    "pinv_truncated",
    "ShadowReadoutEstimator",
    "LinearReadoutEstimator",
    "RunningOutcomeStats",
    # Experiment helpers.
    "sample_data",
    "ntrain_layers",
    "shot_layers",
    "state_frame_distance_grid",
    "state_frame_distances",
    "povm_frame_distance_grid",
    "povm_frame_distances",
]
