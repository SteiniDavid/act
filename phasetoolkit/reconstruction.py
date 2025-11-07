"""
Online phase reconstruction and policy wrappers.

This module provides tools for online phase prediction and action reconstruction
from canonical actions predicted by diffusion policies.
"""

from typing import Dict, Optional
import numpy as np
import torch

from .core.reconstruction import PhaseReconstructionStats
from .gaussianization import BasePhaseGaussianizer, PhaseGaussianizerError


class PhaseProjector:
    """
    Transforms canonical actions to environment action space using per-phase gaussian statistics.

    This class performs the critical transformation that enables phase-conditioned diffusion policies
    to operate in canonical space during training but produce valid environment actions during execution.
    It uses precomputed phase statistics (means and Cholesky factors) to map from the normalized
    canonical space back to the original action distribution for each behavioral phase.

    Mathematical Framework:
        **Canonical to Environment Transformation**:
        - Environment action: a = μ_k + L_k @ c
        - Where: μ_k = phase mean, L_k = Cholesky factor, c = canonical action

        **Phase-Specific Reconstruction**:
        - Each phase k has gaussian distribution: N(μ_k, Σ_k)
        - Cholesky decomposition: Σ_k = L_k @ L_k^T
        - Canonical space: c ~ N(0, I) (normalized/standardized)

    Key Benefits:
        - **Numerical Stability**: Cholesky factors ensure positive definite covariances
        - **Phase-Specific Actions**: Reconstructs actions appropriate for current behavioral phase
        - **Gaussian Structure**: Preserves distributional properties from training data
        - **Efficient Computation**: Linear transformation with precomputed factors

    Attributes:
        stats (Dict[int, PhaseReconstructionStats]): Phase-specific gaussian parameters
        A (int): Action dimensionality (inferred from first phase statistics)
    """
    
    def __init__(
        self,
        phase_stats: Dict[int, PhaseReconstructionStats],
        gaussianizer: Optional[BasePhaseGaussianizer] = None,
        transform_mode: str = "cholesky",
        pooled_phase_id: int = 0,
    ):
        """
        Initialize phase projector with phase-specific statistics.
        
        Args:
            phase_stats: Dictionary mapping phase_id to PhaseReconstructionStats
            gaussianizer: Optional fitted gaussianizer for nonlinear reconstruction
            transform_mode: Canonicalization mode (e.g., 'cholesky', 'phase_gaussianizer')
            pooled_phase_id: Phase id to use when transform_mode is pooled
        """
        self.stats = phase_stats
        self.A = next(iter(phase_stats.values())).mu.shape[0] if phase_stats else None
        self.gaussianizer = gaussianizer
        self.transform_mode = transform_mode
        self.pooled_phase_id = pooled_phase_id

    def to_action(self, c: np.ndarray, phase_id: int) -> np.ndarray:
        """
        Convert canonical action to regular action space.
        
        Args:
            c: Canonical action vector
            phase_id: Phase identifier
            
        Returns:
            Regular action vector: a = μ + L @ c
        """
        if self.gaussianizer is not None:
            transform_phase = int(phase_id)
            available = set(self.gaussianizer.available_phases())
            if self.transform_mode == "pooled_gaussianizer" or transform_phase not in available:
                transform_phase = self.pooled_phase_id
            if transform_phase in available:
                try:
                    canonical = np.atleast_2d(c)
                    recon = self.gaussianizer.inverse_transform(transform_phase, canonical)
                    return recon.squeeze(0)
                except PhaseGaussianizerError:
                    pass

        st = self.stats.get(int(phase_id))
        if st is None:
            # Fallback: identity transformation
            return c
        # a = mu + L @ c
        return st.mu + st.L @ c


class OnlinePhaseDecoder:
    """
    Real-time phase prediction for phase-conditioned diffusion policy execution.

    This decoder uses the forward algorithm to maintain belief states over behavioral phases,
    enabling online phase prediction during policy execution. It integrates with trained HMM
    parameters to provide confident phase estimates that condition the diffusion model's
    action generation process.

    Mathematical Framework:
        **Forward Algorithm**: Maintains belief state α_t(k) = P(phase_t = k | obs_1:t)
        - Initialization: α_1(k) = π_k * P(obs_1 | phase_k)
        - Recursion: α_{t+1}(j) = P(obs_{t+1} | phase_j) * Σ_k α_t(k) * A_{k,j}
        - Prediction: phase_t = argmax_k α_t(k)

        **Confidence Estimation**: Uses entropy of belief distribution
        - High confidence: Belief concentrated on single phase
        - Low confidence: Belief spread across multiple phases
        - Threshold-based decision making for robust predictions

    Key Features:
        - **Sequence-based Prediction**: Uses observation history for better accuracy
        - **Confidence Thresholding**: Filters low-confidence predictions
        - **Stateful Tracking**: Maintains HMM forward probabilities across timesteps
        - **Fallback Handling**: Graceful degradation when HMM bundle unavailable

    Integration with Phase-Conditioned Policies:
        The predicted phases condition the diffusion model to generate actions from
        the appropriate gaussian subspace, improving action quality by narrowing
        the selection scope based on the current behavioral context.

    Attributes:
        hmm_bundle: Complete HMM training state with preprocessed parameters
        hmm_model: Trained hmmlearn GaussianHMM model (fallback)
        tracker (OnlinePhaseTracker): Stateful forward algorithm implementation
        confidence_threshold (float): Minimum confidence for reliable predictions
        use_sequence_prediction (bool): Whether to use full observation sequences
        max_sequence_length (int): Maximum sequence length for prediction
    """

    def __init__(self, phase_result):
        """
        Initialize phase decoder with phase detection results.

        Args:
            phase_result: PhaseDetectionResult from phase detection
        """
        # Get the HMM bundle for online tracking
        self.hmm_bundle = getattr(phase_result, "hmm_bundle", None)
        self.hmm_model = getattr(phase_result, "hmm_model", None)

        if self.hmm_bundle is not None:
            self.tracker = phase_result.create_tracker()
        else:
            self.tracker = None
            print("⚠️ Warning: No HMM bundle found, phase prediction will use fallback")

        # Configuration for phase prediction
        self.confidence_threshold = 0.6  # Minimum confidence for phase prediction
        self.use_sequence_prediction = True  # Use full sequence vs last observation only
        self.max_sequence_length = 10  # Maximum sequence length to consider

    def reset(self):
        """Reset the internal tracker state."""
        if self.tracker is not None:
            self.tracker.reset()

    def predict_phase(self, obs_seq: np.ndarray) -> int:
        """
        Predict phase for given observation sequence.

        Args:
            obs_seq: Observation sequence of shape (seq_len, obs_dim)

        Returns:
            Predicted phase ID
        """
        if self.tracker is None:
            # Fallback: return phase 0
            return 0

        if self.use_sequence_prediction and obs_seq.shape[0] > 1:
            # Use sequence-based prediction
            return self._predict_from_sequence(obs_seq)
        else:
            # Use single-step prediction
            return self._predict_single_step(obs_seq[-1])

    def predict_phase_with_confidence(self, obs_seq: np.ndarray) -> tuple[int, float]:
        """
        Predict phase with confidence estimate.

        Args:
            obs_seq: Observation sequence of shape (seq_len, obs_dim)

        Returns:
            Tuple of (predicted_phase_id, confidence)
        """
        if self.tracker is None:
            return 0, 0.0

        # Reset tracker and process sequence
        self.tracker.reset()

        # Process observations sequentially
        predicted_phase = 0
        for obs in obs_seq:
            predicted_phase = self.tracker.step(obs)

        # Get confidence from probability distribution
        confidence = self.tracker.get_confidence()

        return predicted_phase, confidence

    def _predict_from_sequence(self, obs_seq: np.ndarray) -> int:
        """Predict phase using full observation sequence."""
        # Limit sequence length to avoid computational issues
        if obs_seq.shape[0] > self.max_sequence_length:
            obs_seq = obs_seq[-self.max_sequence_length:]

        # Reset and process full sequence
        self.tracker.reset()

        predicted_phase = 0
        for obs in obs_seq:
            predicted_phase = self.tracker.step(obs)

        # Check confidence and use fallback if too low
        confidence = self.tracker.get_confidence()
        if confidence < self.confidence_threshold:
            # Try batch prediction with HMM model if available
            if self.hmm_model is not None:
                try:
                    # Use HMM model for batch prediction
                    phase_sequence = self.hmm_model.predict(obs_seq.astype(np.float64))
                    return int(phase_sequence[-1])
                except:
                    pass

        return predicted_phase

    def _predict_single_step(self, obs: np.ndarray) -> int:
        """Predict phase using single observation (stateless)."""
        # For single step, we can't maintain state, so use HMM model directly
        if self.hmm_model is not None:
            try:
                phase = self.hmm_model.predict(obs.reshape(1, -1).astype(np.float64))
                return int(phase[0])
            except:
                pass

        return 0

    def get_phase_probabilities(self) -> Optional[np.ndarray]:
        """Get current phase probability distribution."""
        if self.tracker is not None:
            return self.tracker.get_phase_probabilities()
        return None

    def get_confidence(self) -> float:
        """Get confidence in current phase prediction."""
        if self.tracker is not None:
            return self.tracker.get_confidence()
        return 0.0


class PhaseWrappedPolicy:
    """
    Wrapper that combines canonical action prediction with phase-based reconstruction.
    
    This wrapper takes a diffusion policy trained on canonical actions and converts
    its outputs back to regular action space using phase information.
    """
    
    def __init__(self, base_policy, projector: PhaseProjector, decoder: OnlinePhaseDecoder):
        """
        Initialize phase-wrapped policy.
        
        Args:
            base_policy: Diffusion policy trained on canonical actions
            projector: PhaseProjector for C → action conversion
            decoder: OnlinePhaseDecoder for phase prediction
        """
        self.base = base_policy
        self.proj = projector
        self.decoder = decoder
        # The base policy is trained to output canonical C

    @torch.no_grad()
    def predict_action(self, observations: torch.Tensor, target_return=None, num_inference_steps=20):
        """
        Predict actions using canonical policy and phase reconstruction.
        
        Args:
            observations: Observation tensor of shape (batch_size, obs_horizon, obs_dim)
            target_return: Target return for conditioning (optional)
            num_inference_steps: Number of diffusion inference steps
            
        Returns:
            Action sequence tensor of shape (batch_size, pred_horizon, action_dim)
        """
        # 1) Predict canonical action sequence C
        C_seq = self.base.predict_action(
            observations=observations,
            target_return=target_return,
            num_inference_steps=num_inference_steps
        )  # shape (B, pred_horizon, A), numpy or torch
        if isinstance(C_seq, torch.Tensor):
            C_seq = C_seq.cpu().numpy()

        # 2) Decode current phase from observations (use last obs window)
        obs_np = observations.cpu().numpy()[0]  # (obs_horizon, obs_dim)
        phase_id = self.decoder.predict_phase(obs_np)

        # 3) Invert C → a for each step using that phase (or per-step phases if you decode a sequence)
        A_seq = np.stack([self.proj.to_action(C_seq[0, t], phase_id) for t in range(C_seq.shape[1])], axis=0)
        # Return same shape/type as base for compatibility
        return torch.from_numpy(A_seq[None]).float().to(observations.device)

    def eval(self):
        """Set policy to evaluation mode."""
        if hasattr(self.base, 'eval'):
            self.base.eval()

    def train(self):
        """Set policy to training mode.""" 
        if hasattr(self.base, 'train'):
            self.base.train()

    @property
    def obs_horizon(self):
        """Get observation horizon from base policy."""
        return getattr(self.base, 'obs_horizon', 4)

    @property
    def pred_horizon(self):
        """Get prediction horizon from base policy."""
        return getattr(self.base, 'pred_horizon', 16)


def create_phase_wrapped_policy(
    canonical_policy,
    phase_stats: Dict[int, PhaseReconstructionStats],
    phase_result,
    gaussianizer: Optional[BasePhaseGaussianizer] = None,
    transform_mode: str = "cholesky",
    pooled_phase_id: int = 0,
) -> PhaseWrappedPolicy:
    """
    Convenience function to create a phase-wrapped policy.
    
    Args:
        canonical_policy: Diffusion policy trained on canonical actions
        phase_stats: Phase-specific action statistics
        phase_result: Results from phase detection
        gaussianizer: Optional gaussianizer for canonical-to-env reconstruction
        transform_mode: Canonicalization mode used during training
        pooled_phase_id: Phase id to use when transform_mode is pooled
        
    Returns:
        PhaseWrappedPolicy ready for inference
    """
    projector = PhaseProjector(
        phase_stats=phase_stats,
        gaussianizer=gaussianizer,
        transform_mode=transform_mode,
        pooled_phase_id=pooled_phase_id,
    )
    decoder = OnlinePhaseDecoder(phase_result)
    return PhaseWrappedPolicy(canonical_policy, projector, decoder)
