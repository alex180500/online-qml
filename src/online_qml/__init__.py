"""Online QML tools for online shadow training simulations."""

from importlib.metadata import PackageNotFoundError, version as _version

from .core import (
    LayerResult,
    MetricResult,
    SimulationData,
    check_folder,
    load_json,
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
    fit_beta_coefficients,
    prediction_geometry,
)
from .experiments import (
    fit_betas,
    haar_metrics,
    measurement_frame_distances,
    ntrain_layers,
    sample_data,
    shot_layers,
    state_frame_distances,
)
from .quantum import (
    as_observable_matrix,
    frame_distance_summary,
    frame_relative_spectrum,
    get_observables,
    get_test_mse,
    haar_state_frame,
    infinite_stats,
    measurement_frame,
    naimark_measurement_frame_prior,
    product_haar_state_frame,
    sample_dm,
    sample_product_dm,
    sample_povm,
    sample_states,
    sample_traceless_operator,
    sample_unitary,
    shots_outcome,
    shots_to_statistics,
    state_frame,
    trace_superoperator,
    vec_identity,
)

try:
    __version__ = _version("online-qml")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # Version.
    "__version__",
    # Core containers.
    "SimulationData",
    "LayerResult",
    "MetricResult",
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
    "load_json",
    "save_metrics",
    "check_folder",
    # Quantum state, observable and POVM utilities.
    "sample_states",
    "sample_dm",
    "sample_product_dm",
    "sample_traceless_operator",
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
    "product_haar_state_frame",
    "naimark_measurement_frame_prior",
    "state_frame",
    "measurement_frame",
    "frame_relative_spectrum",
    "frame_distance_summary",
    # Test/evaluation helpers.
    "get_test_mse",
    "HaarBiasVariance",
    "evaluate_layers_haar",
    "fit_beta_coefficients",
    "prediction_geometry",
    # Estimators and helpers.
    "pinv_truncated",
    "ShadowReadoutEstimator",
    "LinearReadoutEstimator",
    "RunningOutcomeStats",
    # Experiment helpers.
    "sample_data",
    "ntrain_layers",
    "shot_layers",
    "haar_metrics",
    "state_frame_distances",
    "measurement_frame_distances",
    "fit_betas",
]
