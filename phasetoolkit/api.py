"""
High-level API for phase detection.

This module provides the main entry points for using the phase detection
library, including both functional and object-oriented interfaces.
"""

from typing import List, Dict, Any, Union, Optional
import json
import numpy as np
import pickle
import re
from pathlib import Path

from .data.loaders import load_trajectory_data, TrajectoryData
from .config import PhaseDetectionConfig, DEFAULT_CONFIG
from .core.hmm import fit_left_to_right_hmm, viterbi_labels_per_demo
from .core.phase_stats import compute_phase_stats, create_padded_buffers, PhaseStats
from .core.tracker import OnlinePhaseTracker
from .features import FEATURE_TRANSFORMS


class FeatureTransformingTracker:
    """Wrapper that applies feature transform before forwarding to tracker."""

    def __init__(self, tracker: OnlinePhaseTracker, transform_name: str, transform_kwargs: Optional[Dict[str, Any]] = None):
        self._tracker = tracker
        self._transform_fn = FEATURE_TRANSFORMS[transform_name]
        self._transform_kwargs = transform_kwargs or {}

    def reset(self):
        self._tracker.reset()

    def step(self, observation: np.ndarray) -> int:
        features = self._transform_fn(observation, **self._transform_kwargs).astype(np.float64)
        return self._tracker.step(features)

    def get_phase_probabilities(self) -> Optional[np.ndarray]:
        return self._tracker.get_phase_probabilities()

    def get_confidence(self) -> float:
        return self._tracker.get_confidence()

    def predict_future_phase_sequence(self, future_steps: int):
        return self._tracker.predict_future_phase_sequence(future_steps)


class PhaseDetectionResult:
    """
    Container for phase detection results.
    
    Provides easy access to all computed results including phase labels,
    statistics, and trained models.
    """
    
    def __init__(
        self,
        labels: List[np.ndarray],
        stats,
        hmm_model,
        hmm_bundle,
        padded_buffers: Dict[str, np.ndarray],
        config: PhaseDetectionConfig,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.labels = labels
        self.stats = stats
        self.hmm_model = hmm_model
        self.hmm_bundle = hmm_bundle
        self.padded_buffers = padded_buffers
        self.config = config
        self.feature_metadata = feature_metadata or {}
    
    def create_tracker(self) -> OnlinePhaseTracker:
        """
        Create an online phase tracker for real-time phase prediction during policy execution.

        The tracker uses the trained HMM parameters to predict the most likely current phase
        given a sequence of recent observations. This enables phase-conditioned policy execution
        without requiring complete trajectory knowledge.

        Returns:
            OnlinePhaseTracker: Configured tracker with trained HMM parameters

        Mathematical Approach:
            Uses forward algorithm to maintain belief state over phases:
            α_t(k) = P(phase_t = k | obs_1:t)

        Note:
            See OnlinePhaseTracker documentation for specific API methods.
        """
        if self.hmm_bundle is None:
            raise ValueError("Phase detection result has no HMM bundle for tracking")

        tracker = OnlinePhaseTracker(self.hmm_bundle)

        transform_name = self.feature_metadata.get('feature_transform')

        # Only wrap with feature transform if not using identity transform
        if transform_name and transform_name != 'identity' and transform_name in FEATURE_TRANSFORMS:
            # For backward compatibility, check if world_size exists in metadata
            transform_kwargs = {}
            if 'world_size' in self.feature_metadata:
                transform_kwargs['world_size'] = self.feature_metadata['world_size']
            return FeatureTransformingTracker(tracker, transform_name, transform_kwargs)

        return tracker
    
    def get_phase_means(self) -> np.ndarray:
        """Get action means for each phase. Shape: (K, A)"""
        return self.padded_buffers['mu_table']
    
    def get_subspace_bases(self) -> np.ndarray:
        """Get padded subspace bases for each phase. Shape: (K, A, r_max)"""
        return self.padded_buffers['U_padded']
    
    def get_subspace_masks(self) -> np.ndarray:
        """Get masks indicating valid subspace dimensions. Shape: (K, r_max)"""
        return self.padded_buffers['mask']
    
    def get_subspace_dims(self) -> np.ndarray:
        """Get actual subspace dimensions for each phase. Shape: (K,)"""
        return self.padded_buffers['r_vec']
    
    def save(self, filepath: str) -> None:
        """Save phase detection results to file."""
        # Save padded buffers as npz
        np.savez(filepath, **self.padded_buffers)

        # Save config separately
        config_path = filepath.replace('.npz', '_config.json')
        self.config.save(config_path)

        # Save labels separately as pickle since they have variable lengths
        import pickle
        labels_path = filepath.replace('.npz', '_labels.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(self.labels, f)

        # Save complete PhaseStats object
        stats_path = filepath.replace('.npz', '_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(self.stats, f)

        # Save HMM bundle for complete phase detection capability
        if self.hmm_bundle is not None:
            hmm_bundle_path = filepath.replace('.npz', '_hmm_bundle.pkl')
            with open(hmm_bundle_path, 'wb') as f:
                pickle.dump(self.hmm_bundle, f)

        # Save feature metadata for reproducibility
        feature_path = filepath.replace('.npz', '_features.json')
        with open(feature_path, 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PhaseDetectionResult':
        """Load phase detection results from file."""

        # Load main data
        data = np.load(filepath)
        padded_buffers = dict(data)

        # Load config
        config_path = filepath.replace('.npz', '_config.json')
        config = PhaseDetectionConfig.load(config_path)

        # Load labels
        labels_path = filepath.replace('.npz', '_labels.pkl')
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)

        # Load complete PhaseStats object
        stats_path = filepath.replace('.npz', '_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        # Load HMM bundle
        hmm_bundle_path = filepath.replace('.npz', '_hmm_bundle.pkl')
        hmm_bundle = None
        if Path(hmm_bundle_path).exists():
            with open(hmm_bundle_path, 'rb') as f:
                hmm_bundle = pickle.load(f)

        # Create dummy HMM model (not needed for phase tracking)
        hmm_model = None

        # Load feature metadata if available
        feature_metadata = {}
        feature_path = filepath.replace('.npz', '_features.json')
        if Path(feature_path).exists():
            with open(feature_path, 'r') as f:
                feature_metadata = json.load(f)

        return cls(
            labels=labels,
            stats=stats,
            hmm_model=hmm_model,
            hmm_bundle=hmm_bundle,
            padded_buffers=padded_buffers,
            config=config,
            feature_metadata=feature_metadata,
        )


def detect_phases(
    trajectory_data: Union[str, TrajectoryData, List[np.ndarray]],
    actions: Optional[List[np.ndarray]] = None,
    config: Optional[PhaseDetectionConfig] = None,
    *,
    use_actions_for_clustering: bool = True,
    **config_overrides,
) -> PhaseDetectionResult:
    """
    Detect behavioral phases in demonstration trajectories using left-to-right Hidden Markov Models.

    This is the primary entry point for the phase detection pipeline. It implements a sophisticated
    approach to decompose complex behaviors into distinct phases, each characterized by gaussian
    action distributions with optional PCA-based dimensionality reduction.

    Mathematical Framework:
        1. **HMM Topology**: Uses left-to-right constraint (no backward transitions)
           - Transition matrix: A[i,j] = 0 for j < i (enforces phase progression)
           - Diagonal persistence: A[i,i] = p_stay (configurable phase stability)

        2. **Phase Identification**: Fits HMM on observation features using Baum-Welch EM
           - Emission model: P(obs_t | phase_k) = N(μ_k^obs, Σ_k^obs)
           - Phase labels via Viterbi: argmax_z P(z | obs_1:T)

        3. **Action Statistics**: Computes per-phase gaussian distributions over actions
           - Phase means: μ_k = E[action | phase = k]
           - Phase covariances: Σ_k = Cov[action | phase = k]
           - PCA reduction: Retain var_keep fraction of variance per phase

    Args:
        trajectory_data: Trajectory data in multiple supported formats:
            - str: Path to pickle file with TrajectoryData object
            - TrajectoryData: Structured data with .features and .actions
            - List[np.ndarray]: Feature sequences (requires actions parameter)
        actions: Action sequences, required if trajectory_data is feature list.
                Each array shape (T_i, action_dim)
        config: Configuration object (uses sensible defaults if None):
            - K: Number of phases (default 3)
            - n_iter: EM iterations (default 25)
            - p_stay: Phase persistence probability (default 0.75)
            - var_keep: PCA variance retention (default 0.95)
        **config_overrides: Override specific config parameters

    Returns:
        PhaseDetectionResult: Comprehensive results container with:
            - labels: Phase assignments per trajectory
            - stats: Per-phase gaussian parameters and PCA bases
            - hmm_model: Trained hmmlearn model for transitions
            - hmm_bundle: Complete training state for online tracking
            - padded_buffers: Preprocessed arrays for efficient computation

    Raises:
        ValueError: If trajectory data format is invalid or inconsistent
        FileNotFoundError: If trajectory file path doesn't exist

    Implementation Notes:
        - **Numerical Stability**: Uses regularization for covariance estimation
        - **Memory Efficiency**: Processes trajectories in batches if needed
        - **Reproducibility**: Supports random_state for deterministic results

    Research Context:
        This function enables the core innovation of phase-conditioned diffusion policies
        by decomposing complex behaviors into simpler, more focused action distributions.
        Each identified phase represents a distinct behavioral mode that can be learned
        more effectively by conditioning the diffusion model on the predicted phase.
    """
    # Load configuration
    if config is None:
        config = DEFAULT_CONFIG
    if config_overrides:
        config = config.copy(**config_overrides)
    config.validate()

    # Load trajectory data
    data = load_trajectory_data(trajectory_data, actions=actions)

    # Determine what to use for HMM clustering
    # By default, cluster on actions directly (most appropriate for robotics tasks)
    # If use_actions_for_clustering=False, cluster on observations with feature transform
    if use_actions_for_clustering:
        # Use actions directly for phase detection - most appropriate for robotics
        demos_features = [act.astype(np.float64) for act in data.actions]
        feature_transform_name = "identity"
    else:
        # Apply feature transform to observations
        feature_transform_name = config.feature_transform
        transform_fn = FEATURE_TRANSFORMS[feature_transform_name]
        demos_features = [
            transform_fn(feat).astype(np.float64)
            for feat in data.features
        ]

    demos_actions = data.actions
    
    # Run phase detection pipeline
    # 1. Train HMM on features
    model, bundle = fit_left_to_right_hmm(
        demos_features=demos_features,
        K=config.K,
        n_iter=config.n_iter,
        p_stay=config.p_stay,
        tol=config.tol,
        random_state=config.random_state
    )
    
    # 2. Decode phase labels
    labels = viterbi_labels_per_demo(model, demos_features)
    
    # 3. Compute phase statistics
    stats = compute_phase_stats(
        demos_actions=demos_actions,
        demos_labels=labels,
        K=config.K,
        var_keep=config.var_keep,
        rmax=config.rmax
    )
    
    # 4. Create padded buffers for easy use
    padded_buffers = create_padded_buffers(
        stats=stats,
        K=config.K,
        A=demos_actions[0].shape[1]
    )
    
    feature_metadata = {
        'feature_transform': feature_transform_name,
        'feature_dim': demos_features[0].shape[1] if demos_features else None,
        'use_actions_for_clustering': use_actions_for_clustering,
    }

    return PhaseDetectionResult(
        labels=labels,
        stats=stats,
        hmm_model=model,
        hmm_bundle=bundle,
        padded_buffers=padded_buffers,
        config=config,
        feature_metadata=feature_metadata
    )


class PhaseDetector:
    """
    Object-oriented interface for phase detection with fine-grained control.

    Provides step-by-step execution of the phase detection pipeline, allowing
    inspection of intermediate results and custom processing between stages.
    Useful for research, debugging, and when you need to modify the standard pipeline.

    Pipeline Stages:
        1. **Data Loading**: Convert various input formats to standardized TrajectoryData
        2. **HMM Fitting**: Train left-to-right HMM on observation features using Baum-Welch EM
        3. **Phase Decoding**: Apply Viterbi algorithm to assign phase labels to trajectories
        4. **Statistics Computation**: Calculate per-phase gaussian parameters and PCA subspaces

    Mathematical Framework:
        Same as detect_phases() function but with explicit control over each stage:
        - Left-to-right HMM topology with configurable phase persistence
        - Gaussian emission models for observation features
        - Per-phase action statistics with optional PCA dimensionality reduction

    Advanced Usage:
        - Inspect HMM convergence during training
        - Modify phase labels before computing statistics
        - Apply custom preprocessing between pipeline stages
        - Debug phase detection quality on complex datasets

    Attributes:
        config (PhaseDetectionConfig): Configuration parameters for all stages
        trajectory_data (TrajectoryData): Loaded and preprocessed trajectory data
        hmm_model: Trained hmmlearn GaussianHMM model for phase transitions
        hmm_bundle: Complete training state with preprocessed features for online tracking
        labels (List[np.ndarray]): Phase assignments per trajectory (post-Viterbi)
        stats (PhaseStats): Per-phase gaussian statistics and PCA bases
        padded_buffers (Dict): Preprocessed arrays optimized for downstream computation
    """
    
    def __init__(
        self,
        config: Optional[PhaseDetectionConfig] = None,
        *,
        use_actions_for_clustering: bool = True,
    ):
        """
        Initialize phase detector with configuration.

        Args:
            config: PhaseDetectionConfig object (uses default if None)
            use_actions_for_clustering: If True, cluster on actions directly (default).
                                        If False, cluster on observations with feature transform.
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.config.validate()

        # State variables
        self.use_actions_for_clustering = use_actions_for_clustering
        self.trajectory_data: Optional[TrajectoryData] = None
        self.hmm_model = None
        self.hmm_bundle = None
        self.labels: Optional[List[np.ndarray]] = None
        self.stats = None
        self.padded_buffers: Optional[Dict[str, np.ndarray]] = None
    
    def load_data(
        self,
        trajectory_data: Union[str, TrajectoryData, List[np.ndarray]],
        actions: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Load trajectory data."""
        self.trajectory_data = load_trajectory_data(trajectory_data, actions=actions)
    
    def fit_hmm(self) -> None:
        """Fit HMM on feature data."""
        if self.trajectory_data is None:
            raise ValueError("Must load data first")

        # Determine what to use for clustering
        if self.use_actions_for_clustering:
            # Use actions directly for phase detection
            transformed_features = [
                act.astype(np.float64) for act in self.trajectory_data.actions
            ]
        else:
            # Apply feature transform to observations
            feature_transform = FEATURE_TRANSFORMS[self.config.feature_transform]
            transformed_features = [
                feature_transform(feat).astype(np.float64)
                for feat in self.trajectory_data.features
            ]

        self.hmm_model, self.hmm_bundle = fit_left_to_right_hmm(
            demos_features=transformed_features,
            K=self.config.K,
            n_iter=self.config.n_iter,
            p_stay=self.config.p_stay,
            tol=self.config.tol,
            random_state=self.config.random_state
        )
        # Replace stored features with transformed version for downstream steps
        self.trajectory_data.features = transformed_features
    
    def decode_phases(self) -> None:
        """Decode phase labels using fitted HMM."""
        if self.hmm_model is None:
            raise ValueError("Must fit HMM first")
        
        self.labels = viterbi_labels_per_demo(self.hmm_model, self.trajectory_data.features)
    
    def compute_stats(self) -> None:
        """Compute per-phase action statistics."""
        if self.labels is None:
            raise ValueError("Must decode phases first")
        
        self.stats = compute_phase_stats(
            demos_actions=self.trajectory_data.actions,
            demos_labels=self.labels,
            K=self.config.K,
            var_keep=self.config.var_keep,
            rmax=self.config.rmax
        )
        
        # Create padded buffers
        self.padded_buffers = create_padded_buffers(
            stats=self.stats,
            K=self.config.K,
            A=self.trajectory_data.actions[0].shape[1]
        )
    
    def fit(self, trajectory_data: Union[str, TrajectoryData, List[np.ndarray]], 
            actions: Optional[List[np.ndarray]] = None) -> PhaseDetectionResult:
        """
        Run complete phase detection pipeline.
        
        Args:
            trajectory_data: Trajectory data in various formats
            actions: Action arrays (if needed)
            
        Returns:
            PhaseDetectionResult object
        """
        self.load_data(trajectory_data, actions)
        self.fit_hmm()
        self.decode_phases()
        self.compute_stats()

        feature_metadata = {
            'feature_transform': self.config.feature_transform,
            'use_actions_for_clustering': self.use_actions_for_clustering,
        }

        return PhaseDetectionResult(
            labels=self.labels,
            stats=self.stats,
            hmm_model=self.hmm_model,
            hmm_bundle=self.hmm_bundle,
            padded_buffers=self.padded_buffers,
            config=self.config,
            feature_metadata=feature_metadata,
        )
    
    def create_tracker(self) -> OnlinePhaseTracker:
        """Create online phase tracker with fitted parameters."""
        if self.hmm_bundle is None:
            raise ValueError("Must fit HMM first")
        return OnlinePhaseTracker(self.hmm_bundle)


def load_phase_result(filepath):
    """
    Load phase detection results from file.

    Args:
        filepath: Path to phase detection results

    Returns:
        PhaseDetectionResult or fallback object with hmm_bundle=None
    """
    filepath = Path(filepath)
    if filepath.exists():
        try:
            return PhaseDetectionResult.load(str(filepath))
        except Exception:
            pass

    # Return None when no saved results available
    return None


def load_global_phase_result(results_dir: Union[str, Path]) -> Optional[PhaseDetectionResult]:
    """Load a single global phase detection result from a directory.

    Looks for a ``global_phases.npz`` file first, then falls back to the first
    ``env_*_phases.npz`` result. Returns ``None`` if no results are present.
    """

    results_path = Path(results_dir)

    # Preferred global file
    global_path = results_path / "global_phases.npz"
    if global_path.exists():
        return PhaseDetectionResult.load(str(global_path))

    # Fallback: first environment-specific file
    for phase_file in sorted(results_path.glob("env_*_phases.npz")):
        result = load_phase_result(phase_file)
        if result is not None:
            return result

    return None


def load_all_phase_results(results_dir):
    """
    Load all phase detection results from a directory.

    Args:
        results_dir: Directory containing phase result files (env_*_phases.npz)

    Returns:
        Dictionary mapping env_idx -> PhaseDetectionResult (only environments with actual results)
    """
    results_dir = Path(results_dir)
    phase_results = {}

    if not results_dir.exists():
        return phase_results

    # Find all env_*_phases.npz files
    for phase_file in results_dir.glob("env_*_phases.npz"):
        # Extract environment index from filename
        env_match = re.match(r"env_(\d+)_phases\.npz", phase_file.name)
        if env_match:
            env_idx = int(env_match.group(1))
            result = load_phase_result(phase_file)
            # Only include environments that actually have phase detection results
            if result is not None:
                phase_results[env_idx] = result

    return phase_results
