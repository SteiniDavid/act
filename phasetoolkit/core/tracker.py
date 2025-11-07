"""
Online phase tracking using forward algorithm.

This module provides an online phase tracker that can process streaming
observations and predict the current phase in real-time using learned HMM parameters.
"""

from typing import Optional, Tuple, List
import numpy as np
from .hmm import HMMBundle


class OnlinePhaseTracker:
    """
    Real-time phase tracker using forward algorithm.
    
    This class maintains the forward probabilities (alpha) and updates them
    incrementally as new observations arrive. It provides the most likely
    current phase at each timestep.
    
    The tracker is stateful - it remembers the probability distribution
    over phases from previous timesteps to make better predictions.
    """

    def __init__(self, bundle: HMMBundle):
        """
        Initialize tracker with learned HMM parameters.
        
        Args:
            bundle: HMMBundle containing learned HMM parameters
        """
        self.bundle = bundle
        self._log_alpha: Optional[np.ndarray] = None

    @staticmethod
    def _log_emission_diag_gauss(x: np.ndarray, means: np.ndarray, covars_diag: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities under diagonal Gaussians.
        
        Computes log p(x | z=k) for all states k under diagonal Gaussian
        emission models. Handles both diagonal covariance format (K, D) 
        and full covariance format (K, D, D) by extracting diagonal.
        
        Args:
            x: Observation vector of shape (D,)
            means: Emission means of shape (K, D)
            covars_diag: Covariances of shape (K, D) or (K, D, D)
            
        Returns:
            Log emission probabilities of shape (K,)
        """
        # Handle both diagonal (K,D) and full (K,D,D) covariance matrices
        if covars_diag.ndim == 3:  # full covariance (K,D,D)
            covars_diag = np.diagonal(covars_diag, axis1=1, axis2=2)  # extract diagonal -> (K,D)
        
        diff2 = (x[None, :] - means) ** 2
        # Add small epsilon to prevent log(0) and numerical instability
        covars_safe = np.maximum(covars_diag, 1e-12)
        log_det = 0.5 * np.sum(np.log(2.0 * np.pi * covars_safe), axis=1)   # (K,)
        quad = 0.5 * np.sum(diff2 / covars_safe, axis=1)                     # (K,)
        return -(log_det + quad)

    @staticmethod
    def _logsumexp(v: np.ndarray) -> float:
        """
        Numerically stable log-sum-exp computation.
        
        Computes log(sum(exp(v))) in a numerically stable way by
        factoring out the maximum value.
        
        Args:
            v: Array of log values
            
        Returns:
            log(sum(exp(v)))
        """
        m = np.max(v)
        return float(m + np.log(np.sum(np.exp(v - m))))

    def reset(self):
        """
        Reset the tracker state.
        
        Clears the internal forward probabilities, causing the next
        observation to be processed as if it's the first timestep.
        """
        self._log_alpha = None

    def step(self, x_t: np.ndarray) -> int:
        """
        Process a single observation and return predicted phase.
        
        Updates the internal forward probabilities with the new observation
        and returns the most likely current phase (argmax of forward probabilities).
        
        Args:
            x_t: Feature vector of shape (D,)
            
        Returns:
            Predicted phase index (0 to K-1)
        """
        K = self.bundle.K
        means = self.bundle.means_
        covars = self.bundle.covars_
        trans = self.bundle.transmat_
        start = self.bundle.startprob_

        # Compute log emission probabilities for all states
        le = self._log_emission_diag_gauss(x_t.astype(np.float64), means, covars)  # (K,)

        if self._log_alpha is None:
            # First timestep: initialize with start probabilities
            self._log_alpha = np.log(start + 1e-12) + le
        else:
            # Forward step: alpha_t(j) = sum_i alpha_{t-1}(i) * A[i,j] * b_j(x_t)
            # In log-space: log_alpha_t(j) = logsumexp(log_alpha_{t-1} + log A[:,j]) + le[j]
            la_new = np.full(K, -np.inf, dtype=np.float64)
            logA = np.log(trans + 1e-12)
            for j in range(K):
                la_new[j] = self._logsumexp(self._log_alpha + logA[:, j]) + le[j]
            self._log_alpha = la_new

        # Return most likely current state
        return int(np.argmax(self._log_alpha))

    def get_phase_probabilities(self) -> Optional[np.ndarray]:
        """
        Get current phase probability distribution.
        
        Returns the normalized forward probabilities, representing
        the current belief state over phases.
        
        Returns:
            Array of phase probabilities of shape (K,), or None if no observations processed
        """
        if self._log_alpha is None:
            return None
        
        # Convert log probabilities to normal space and normalize
        max_log = np.max(self._log_alpha)
        exp_alpha = np.exp(self._log_alpha - max_log)
        return exp_alpha / np.sum(exp_alpha)

    def get_confidence(self) -> float:
        """
        Get confidence in current phase prediction.

        Returns the probability of the most likely phase, which can be
        interpreted as confidence in the prediction.

        Returns:
            Confidence score between 0 and 1, or 0 if no observations processed
        """
        probs = self.get_phase_probabilities()
        if probs is None:
            return 0.0
        return float(np.max(probs))

    def predict_future_phase_sequence(self, future_steps: int) -> Tuple[List[int], List[float]]:
        """
        Predict future phase sequence using HMM transition dynamics.

        Uses the current belief state (forward probabilities) and simulates
        forward steps using only the transition matrix to predict how phases
        will evolve over time.

        Args:
            future_steps: Number of future timesteps to predict

        Returns:
            Tuple of (predicted_phases, confidences) for each future timestep
        """
        if self._log_alpha is None:
            return [0] * future_steps, [0.0] * future_steps

        # Get current belief state
        current_log_alpha = self._log_alpha.copy()

        predicted_phases = []
        predicted_confidences = []

        # Simulate forward steps using transition matrix only
        trans_matrix = self.bundle.transmat_
        log_trans = np.log(trans_matrix + 1e-12)

        for t in range(future_steps):
            # Predict most likely phase at this future timestep
            phase_t = int(np.argmax(current_log_alpha))

            # Calculate confidence (normalized probability of most likely phase)
            max_log_alpha = np.max(current_log_alpha)
            normalized_probs = np.exp(current_log_alpha - max_log_alpha)
            normalized_probs = normalized_probs / np.sum(normalized_probs)
            confidence_t = float(np.max(normalized_probs))

            predicted_phases.append(phase_t)
            predicted_confidences.append(confidence_t)

            # Forward one step using only transition matrix
            # log_alpha_{t+1}(j) = logsumexp_i(log_alpha_t(i) + log_trans(i,j))
            new_log_alpha = np.full(len(current_log_alpha), -np.inf)

            for j in range(len(current_log_alpha)):
                # Sum over all previous states i
                log_values = current_log_alpha + log_trans[:, j]
                max_val = np.max(log_values)
                if max_val > -np.inf:
                    new_log_alpha[j] = max_val + np.log(np.sum(np.exp(log_values - max_val)))

            current_log_alpha = new_log_alpha

            # Normalize to prevent numerical drift
            current_log_alpha = current_log_alpha - np.max(current_log_alpha)

        return predicted_phases, predicted_confidences

    def step_with_future_prediction(self, x_t: np.ndarray, future_steps: int = 0) -> Tuple[int, List[int], List[float]]:
        """
        Process observation and optionally predict future phase sequence.

        Args:
            x_t: Feature vector of shape (D,)
            future_steps: Number of future timesteps to predict (0 = no prediction)

        Returns:
            Tuple of (current_phase, future_phases, future_confidences)
        """
        # Process current observation
        current_phase = self.step(x_t)

        if future_steps <= 0:
            return current_phase, [], []

        # Predict future phase sequence
        future_phases, future_confidences = self.predict_future_phase_sequence(future_steps)

        return current_phase, future_phases, future_confidences