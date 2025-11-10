"""
Phase-wrapped ACT policy for inference with canonical action reconstruction.

This module provides a wrapper around ACT policies trained on canonical actions,
enabling real-time phase prediction and canonical-to-environment action conversion
during policy execution.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from phasetoolkit import PhaseDetectionResult
from phasetoolkit.reconstruction import PhaseProjector, OnlinePhaseDecoder


class PhaseWrappedACT(nn.Module):
    """
    Wraps an ACT policy trained on canonical actions for inference.

    This wrapper:
    1. Predicts the current phase from observations using OnlinePhaseDecoder
    2. Calls the base policy with phase conditioning to get canonical actions
    3. Converts canonical actions back to environment actions using PhaseProjector

    The base policy must have been trained with phase conditioning (use_canonical=True).
    """

    def __init__(
        self,
        base_policy: nn.Module,
        phase_result: PhaseDetectionResult,
        stats: dict,
    ):
        """
        Initialize the phase-wrapped ACT policy.

        Args:
            base_policy: ACT policy trained on canonical actions
            phase_result: PhaseDetectionResult containing phase statistics and HMM
            stats: Dataset statistics dict with action_mean and action_std
        """
        super().__init__()
        self.base_policy = base_policy
        self.stats = stats

        # Create phase projector for canonical -> environment action conversion
        phase_stats_dict = {}
        if phase_result.stats is not None:
            for k in range(phase_result.config.K):
                # Create reconstruction stats from phase statistics
                mu_k = phase_result.stats.mu_list[k]
                Sigma_k = phase_result.stats.Sigma_list[k]

                # Compute Cholesky decomposition with regularization
                eps = 1e-6
                Sigma_reg = Sigma_k + eps * np.eye(Sigma_k.shape[0])
                L_k = np.linalg.cholesky(Sigma_reg)
                Linv_k = np.linalg.inv(L_k)

                # Store in format expected by PhaseProjector
                from phasetoolkit.core.reconstruction import PhaseReconstructionStats
                phase_stats_dict[k] = PhaseReconstructionStats(
                    mu=mu_k,
                    Sigma=Sigma_k,
                    L=L_k,
                    Linv=Linv_k
                )

        self.phase_projector = PhaseProjector(phase_stats_dict)

        # Create online phase decoder for real-time phase prediction
        self.phase_decoder = OnlinePhaseDecoder(phase_result)

        # Track current phase prediction
        self.current_phase = 0

        # Observation history for improved phase prediction
        self.obs_history = []
        self.max_history_len = 10  # Keep last 10 observations

    def reset(self):
        """Reset the phase decoder for a new episode."""
        self.phase_decoder.reset()
        self.current_phase = 0
        self.obs_history = []  # Clear observation history

    def predict_phase(self, qpos: np.ndarray) -> int:
        """
        Predict the current phase from proprioceptive state using observation history.

        Args:
            qpos: Proprioceptive state (obs_dim,)

        Returns:
            Predicted phase ID
        """
        # Add current observation to history
        self.obs_history.append(qpos)

        # Keep only the most recent observations
        if len(self.obs_history) > self.max_history_len:
            self.obs_history = self.obs_history[-self.max_history_len:]

        # Use observation history for phase prediction
        # OnlinePhaseDecoder.predict_phase expects a sequence of shape (seq_len, obs_dim)
        obs_seq = np.array(self.obs_history)  # (history_len, obs_dim)

        predicted_phase = self.phase_decoder.predict_phase(obs_seq)
        self.current_phase = predicted_phase
        return predicted_phase

    def canonical_to_env_action(self, canonical_action: np.ndarray, phase_id: int) -> np.ndarray:
        """
        Convert canonical action to environment action.

        Args:
            canonical_action: Canonical action (action_dim,) - already in correct space (mean≈0, std≈1)
            phase_id: Phase identifier

        Returns:
            Environment action (action_dim,)
        """
        # Canonical actions are NOT normalized with environment stats anymore (fixed in utils.py)
        # They are already in the correct standardized space from L_k^{-1} @ (action - μ_k)
        # So we can directly apply the phase transformation: env_action = μ_k + L_k @ canonical
        env_action = self.phase_projector.to_action(canonical_action, phase_id)

        return env_action

    def __call__(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with phase prediction and action reconstruction.

        Args:
            qpos: Proprioceptive state (B, state_dim)
            image: Image observations (B, num_cam, C, H, W)
            actions: Actions for training (optional)
            is_pad: Padding mask for training (optional)
            phase_ids: Phase IDs for training (optional, overrides prediction)

        Returns:
            If training: loss dict
            If inference: environment actions (B, chunk_size, action_dim)
        """
        is_training = actions is not None

        if is_training:
            # Training mode: use provided phase_ids and actions
            return self.base_policy(qpos, image, actions, is_pad, phase_ids)
        else:
            # Inference mode: predict phase and convert canonical actions
            batch_size = qpos.shape[0]
            assert batch_size == 1, "Phase-wrapped policy only supports batch_size=1 during inference"

            # Predict current phase from observations
            qpos_np = qpos.cpu().numpy()[0]  # (state_dim,)
            predicted_phase = self.predict_phase(qpos_np)

            # Get canonical actions from base policy
            phase_ids_tensor = torch.tensor([predicted_phase], dtype=torch.long).cuda()
            canonical_actions = self.base_policy(qpos, image, phase_ids=phase_ids_tensor)  # (1, chunk_size, action_dim)

            # Convert each timestep's canonical action to environment action
            canonical_actions_np = canonical_actions.cpu().numpy()[0]  # (chunk_size, action_dim)
            env_actions = np.zeros_like(canonical_actions_np)

            for t in range(canonical_actions_np.shape[0]):
                env_actions[t] = self.canonical_to_env_action(
                    canonical_actions_np[t],
                    predicted_phase
                )

            return torch.from_numpy(env_actions).float().cuda().unsqueeze(0)  # (1, chunk_size, action_dim)

    def configure_optimizers(self):
        """Return the base policy's optimizer."""
        return self.base_policy.configure_optimizers()

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into the base policy."""
        return self.base_policy.load_state_dict(state_dict, strict=strict)

    def state_dict(self):
        """Return the base policy's state dict."""
        return self.base_policy.state_dict()

    def cuda(self):
        """Move to CUDA."""
        self.base_policy.cuda()
        return self

    def eval(self):
        """Set to evaluation mode."""
        self.base_policy.eval()
        return self

    def train(self):
        """Set to training mode."""
        self.base_policy.train()
        return self


def create_phase_wrapped_act(
    base_policy: nn.Module,
    phase_result_path: str,
    stats: dict,
) -> PhaseWrappedACT:
    """
    Convenience function to create a phase-wrapped ACT policy.

    Args:
        base_policy: ACT policy trained on canonical actions
        phase_result_path: Path to phase detection results (.npz file)
        stats: Dataset statistics dict

    Returns:
        PhaseWrappedACT instance
    """
    # Load phase detection results
    phase_result = PhaseDetectionResult.load(phase_result_path)

    # Create wrapped policy
    wrapped_policy = PhaseWrappedACT(
        base_policy=base_policy,
        phase_result=phase_result,
        stats=stats,
    )

    return wrapped_policy
