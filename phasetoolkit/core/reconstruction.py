"""
Phase-based action reconstruction utilities.

This module provides tools for converting between canonical actions and 
regular actions using phase-specific statistics.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from .phase_stats import PhaseStats as BasePhaseStats


@dataclass
class PhaseReconstructionStats:
    """
    Phase-specific action statistics for canonical action transformation.
    
    This is an extension of the base PhaseStats that includes Cholesky decomposition
    for efficient canonical action computation.
    
    Attributes:
        mu: Action mean for the phase (action_dim,)
        Sigma: Covariance matrix (action_dim, action_dim) 
        L: Lower-triangular Cholesky decomposition of Sigma
        Linv: Inverse of L for fast canonical action computation
    """
    mu: np.ndarray       # (action_dim,)
    Sigma: np.ndarray    # (action_dim, action_dim)
    L: np.ndarray        # lower-tri Cholesky, (action_dim, action_dim)
    Linv: np.ndarray     # inverse of L (for fast solves)
    transform_kind: str = "cholesky"
    gaussianizer_label: Optional[str] = None
    gaussianizer_state: Optional[dict] = None


def _safe_cholesky(S: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust Cholesky decomposition with positive definite repair.
    
    Args:
        S: Covariance matrix to decompose
        eps: Regularization parameter for numerical stability
        
    Returns:
        Tuple of (L, Linv) where L is Cholesky factor and Linv is its inverse
    """
    S = 0.5 * (S + S.T)  # symmetrize
    try:
        L = np.linalg.cholesky(S + eps * np.eye(S.shape[0]))
    except np.linalg.LinAlgError:
        # Eigen repair if not positive definite
        w, V = np.linalg.eigh(S)
        w = np.clip(w, eps, None)
        S = (V * w) @ V.T
        L = np.linalg.cholesky(S)
    
    Linv = np.linalg.inv(L)
    return L, Linv


def compute_phase_action_stats_for_env(
    dataset,
    labels_per_traj: List[np.ndarray],
    action_dim: int,
) -> Tuple[Dict[int, PhaseReconstructionStats], Dict[int, np.ndarray]]:
    """
    Compute phase-specific action statistics for canonical action transformation.
    
    Args:
        dataset: TrajectoryDataset containing trajectories
        labels_per_traj: List of phase label arrays, one per trajectory
        action_dim: Dimension of action space
        
    Returns:
        Tuple of (phase_stats, phase_actions) where phase_stats maps phase_id to
        PhaseReconstructionStats and phase_actions stores the raw action samples per phase
    """
    # Find number of phases K from labels
    K = int(max([lbls.max() for lbls in labels_per_traj if len(lbls) > 0]) + 1)
    phase_stats: Dict[int, PhaseReconstructionStats] = {}
    phase_actions: Dict[int, np.ndarray] = {}

    # Collect actions per phase
    per_phase_actions = {k: [] for k in range(K)}
    for traj_idx, traj in enumerate(dataset.trajectories):
        acts = np.asarray(traj['actions'])    # (T, action_dim)
        lbls = np.asarray(labels_per_traj[traj_idx])  # (T,)
        T = min(len(acts), len(lbls))
        if T == 0: 
            continue
        for k in range(K):
            mask = (lbls[:T] == k)
            if mask.any():
                per_phase_actions[k].append(acts[:T][mask])

    # Fit Gaussians for each phase
    for k in range(K):
        if len(per_phase_actions[k]) == 0:
            continue
        X = np.concatenate(per_phase_actions[k], axis=0)  # (Nk, action_dim)
        mu = X.mean(axis=0)
        Xc = X - mu
        # Sample covariance with tiny ridge for stability
        Sigma = (Xc.T @ Xc) / max(1, (X.shape[0] - 1))
        lam = 1e-5 * np.trace(Sigma) / max(1, action_dim)
        Sigma = Sigma + lam * np.eye(action_dim)
        L, Linv = _safe_cholesky(Sigma)
        phase_stats[k] = PhaseReconstructionStats(mu=mu, Sigma=Sigma, L=L, Linv=Linv)
        phase_actions[k] = X

    return phase_stats, phase_actions


def attach_phase_canonical_to_dataset(
    env_idx: int,
    dataset,
    labels_per_traj: List[np.ndarray],
    phase_stats: Dict[int, PhaseReconstructionStats],
    action_dim: int,
    phase_gaussianizer=None,
    transform_mode: str = "cholesky",
    pooled_phase_id: int = 0,
) -> None:
    """
    Augments trajectories in-place with phase IDs and canonical actions.
    
    Args:
        env_idx: Environment index (for debugging/logging)
        dataset: TrajectoryDataset to augment
        labels_per_traj: Phase labels for each trajectory 
        phase_stats: Phase-specific statistics
        action_dim: Dimension of action space
    """
    assert len(dataset.trajectories) == len(labels_per_traj), \
        "Trajectory order must match labels produced in phase detection."

    for traj_idx, traj in enumerate(dataset.trajectories):
        acts = np.asarray(traj['actions'])       # (T, A)
        lbls = np.asarray(labels_per_traj[traj_idx])  # (T,)
        T = min(len(acts), len(lbls))
        if T == 0:
            traj['phase_ids'] = np.array([], dtype=int)
            traj['actions_canonical'] = np.zeros_like(acts)
            continue

        C = np.zeros_like(acts)                       # canonical actions
        mu_seq = np.zeros_like(acts)                  # per-step mu
        L_seq = np.zeros((acts.shape[0], action_dim, action_dim))  # per-step L

        for t in range(T):
            k = int(lbls[t])
            st = phase_stats.get(k, None)
            if st is None:
                mu = np.zeros(action_dim)
                L = np.eye(action_dim)
                Linv = np.eye(action_dim)
            else:
                mu, L, Linv = st.mu, st.L, st.Linv

            mu_seq[t] = mu
            L_seq[t] = L

        if transform_mode == "cholesky" or phase_gaussianizer is None:
            for t in range(T):
                k = int(lbls[t])
                st = phase_stats.get(k, None)
                if st is None:
                    mu = np.zeros(action_dim)
                    Linv = np.eye(action_dim)
                else:
                    mu, Linv = st.mu, st.Linv
                C[t] = Linv @ (acts[t] - mu)
        else:
            # Group actions per phase and apply gaussianizer transform in batch
            per_phase_indices: Dict[int, List[int]] = {}
            for idx_t in range(T):
                per_phase_indices.setdefault(int(lbls[idx_t]), []).append(idx_t)

            canonical_buffer = np.zeros_like(acts)
            for phase_id, indices in per_phase_indices.items():
                phase_actions = acts[indices]
                transform_phase_id = int(phase_id)
                if transform_mode == "pooled_gaussianizer" and transform_phase_id not in phase_gaussianizer.available_phases():
                    transform_phase_id = pooled_phase_id
                result = phase_gaussianizer.transform(transform_phase_id, phase_actions)
                canonical_buffer[indices] = result.transformed
            C = canonical_buffer

        # Persist new fields
        traj['phase_ids'] = lbls[:T]
        traj['actions_canonical'] = C
        traj['phase_mu'] = mu_seq
        traj['phase_L'] = L_seq


def convert_base_stats_to_action_stats(
    base_stats: BasePhaseStats, 
    action_dim: int
) -> Dict[int, PhaseReconstructionStats]:
    """
    Convert base PhaseStats to PhaseReconstructionStats with Cholesky decomposition.
    
    Args:
        base_stats: Base PhaseStats object from phase detection
        action_dim: Dimension of action space
        
    Returns:
        Dictionary mapping phase indices to PhaseReconstructionStats
    """
    phase_action_stats = {}
    
    for k, (mu, Sigma) in enumerate(zip(base_stats.mu_list, base_stats.Sigma_list)):
        if mu is not None and Sigma is not None:
            L, Linv = _safe_cholesky(Sigma)
            phase_action_stats[k] = PhaseReconstructionStats(
                mu=mu.copy(),
                Sigma=Sigma.copy(), 
                L=L,
                Linv=Linv
            )
    
    return phase_action_stats
