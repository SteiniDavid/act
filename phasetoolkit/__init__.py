"""
PhaseTookit - A library for phase detection and action subspace learning.

This package provides tools for:
- Phase detection using Hidden Markov Models
- Per-phase action statistics computation
- Online phase tracking
- Action subspace learning via PCA

Main API functions:
- detect_phases(): High-level interface for phase detection
- PhaseDetector: Class-based interface for more control
"""

from .core.hmm import fit_left_to_right_hmm, HMMBundle
from .core.phase_stats import compute_phase_stats, PhaseStats
from .core.tracker import OnlinePhaseTracker
from .core.reconstruction import (
    PhaseReconstructionStats, 
    compute_phase_action_stats_for_env,
    attach_phase_canonical_to_dataset,
    convert_base_stats_to_action_stats
)
from .config import PhaseDetectionConfig
from .api import (
    detect_phases,
    PhaseDetector,
    PhaseDetectionResult,
    load_global_phase_result,
)
from .reconstruction import PhaseProjector, OnlinePhaseDecoder, PhaseWrappedPolicy, create_phase_wrapped_policy
from .gaussianization import (
    BasePhaseGaussianizer,
    PhaseGaussianizerError,
    ICAPhaseGaussianizer,
    ICAGaussianizerConfig,
    RealNVPPhaseGaussianizer,
    RealNVPConfig,
    GaussianizerSpec,
    build_gaussianizer,
    build_gaussianizer_from_spec,
)

__version__ = "0.1.0"

__all__ = [
    "detect_phases",
    "PhaseDetector", 
    "PhaseDetectionResult",
    "load_global_phase_result",
    "PhaseDetectionConfig",
    "fit_left_to_right_hmm",
    "HMMBundle",
    "compute_phase_stats",
    "PhaseStats", 
    "OnlinePhaseTracker",
    "PhaseReconstructionStats",
    "compute_phase_action_stats_for_env",
    "attach_phase_canonical_to_dataset", 
    "convert_base_stats_to_action_stats",
    "PhaseProjector",
    "OnlinePhaseDecoder",
    "PhaseWrappedPolicy",
    "create_phase_wrapped_policy",
    "BasePhaseGaussianizer",
    "PhaseGaussianizerError",
    "ICAPhaseGaussianizer",
    "ICAGaussianizerConfig",
    "RealNVPPhaseGaussianizer",
    "RealNVPConfig",
    "build_gaussianizer",
    "GaussianizerSpec",
    "build_gaussianizer_from_spec",
]
