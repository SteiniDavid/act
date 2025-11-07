"""
Phase statistics computation using scikit-learn.

This module computes per-phase action statistics including means, covariances,
and principal subspaces using robust statistical estimators.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


@dataclass
class PhaseStats:
    """
    Container for per-phase action statistics.
    
    Attributes:
        mu_list: List of action means for each phase, each of shape (A,)
        Sigma_list: List of covariance matrices for each phase, each of shape (A, A)
        U_list: List of principal component bases for each phase, each of shape (A, r_k)
        lambdas: List of eigenvalues for each phase, each of shape (r_k,)
        r_list: List of subspace dimensions for each phase
    """
    mu_list: List[np.ndarray]      # Action means: (A,) for each phase
    Sigma_list: List[np.ndarray]   # Covariances: (A, A) for each phase
    U_list: List[np.ndarray]       # Subspace bases: (A, r_k) for each phase
    lambdas: List[np.ndarray]      # Eigenvalues: (r_k,) for each phase
    r_list: List[int]              # Subspace dimensions


def compute_phase_stats(
    demos_actions: List[np.ndarray],
    demos_labels: List[np.ndarray],
    K: int,
    var_keep: float = 0.95,
    rmax: int = 16,
) -> PhaseStats:
    """
    Compute per-phase action statistics using robust estimators.
    
    For each phase k:
    1. Collect all actions assigned to phase k across all demonstrations
    2. Compute action mean μ_k
    3. Estimate covariance matrix Σ_k using Ledoit-Wolf shrinkage
    4. Perform PCA and select subspace dimension to preserve var_keep variance
    5. Extract principal components U_k and eigenvalues λ_k
    
    Args:
        demos_actions: List of action arrays, each of shape (T_i, A)  
        demos_labels: List of phase label arrays, each of shape (T_i,)
        K: Number of phases
        var_keep: Fraction of variance to preserve in subspace selection
        rmax: Maximum subspace dimension
        
    Returns:
        PhaseStats object containing all computed statistics
    """
    A = demos_actions[0].shape[1]
    mu_list, Sigma_list, U_list, lambdas, r_list = [], [], [], [], []

    for k in range(K):
        # Collect all actions for this phase
        Xk_chunks = []
        for a, z in zip(demos_actions, demos_labels):
            msk = (z == k)
            if msk.any():
                Xk_chunks.append(a[msk])
        
        if not Xk_chunks:
            # Fallback if a phase has no assigned actions
            mu_k = np.zeros(A, dtype=np.float64)
            Sigma_k = np.eye(A, dtype=np.float64) * 1e-3
            U_k = np.eye(A, dtype=np.float64)[:, :1]
            lam_k = np.array([1e-3], dtype=np.float64)
            r_k = 1
        else:
            # Compute statistics from collected actions
            Xk = np.vstack(Xk_chunks).astype(np.float64)
            mu_k = Xk.mean(axis=0)
            Xc = Xk - mu_k

            # Ledoit-Wolf shrinkage for robust covariance estimation
            lw = LedoitWolf().fit(Xc)  # centered data
            Sigma_k = lw.covariance_

            # PCA for subspace identification
            pca_full = PCA(n_components=None, svd_solver="full")
            pca_full.fit(Xc)
            
            # Select subspace dimension based on variance preservation
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            r_k = int(np.searchsorted(cumsum, var_keep) + 1)
            r_k = max(1, min(r_k, min(rmax, A)))

            # Extract principal components and eigenvalues
            # sklearn's components_ has shape (n_components, A), rows are principal axes
            U_k = pca_full.components_[:r_k].T.copy()            # (A, r_k)
            lam_k = pca_full.explained_variance_[:r_k].copy()    # (r_k,)

        # Store results
        mu_list.append(mu_k.astype(np.float64))
        Sigma_list.append(Sigma_k.astype(np.float64))
        U_list.append(U_k.astype(np.float64))
        lambdas.append(lam_k.astype(np.float64))
        r_list.append(int(r_k))

    return PhaseStats(mu_list, Sigma_list, U_list, lambdas, r_list)


def create_padded_buffers(stats: PhaseStats, K: int, A: int) -> dict:
    """
    Create fixed-size padded buffers for policy consumption.
    
    This creates arrays with consistent shapes that can be easily used
    in neural network policies or other downstream applications.
    
    Args:
        stats: PhaseStats object with computed statistics
        K: Number of phases
        A: Action dimension
        
    Returns:
        Dictionary containing:
        - mu_table: (K, A) array of action means
        - U_padded: (K, A, r_max) array of padded subspace bases
        - mask: (K, r_max) binary mask indicating valid subspace dimensions
        - r_vec: (K,) array of actual subspace dimensions
    """
    r_max = max(stats.r_list) if stats.r_list else 1
    
    # Create padded arrays
    U_padded = np.zeros((K, A, r_max), dtype=np.float32)
    mask = np.zeros((K, r_max), dtype=np.float32)
    r_vec = np.zeros((K,), dtype=np.int64)
    mu_table = np.stack(stats.mu_list, axis=0).astype(np.float32)

    # Fill padded arrays
    for k, U in enumerate(stats.U_list):
        rk = U.shape[1]
        U_padded[k, :, :rk] = U.astype(np.float32)
        mask[k, :rk] = 1.0
        r_vec[k] = rk

    return {
        'mu_table': mu_table,     # (K, A)
        'U_padded': U_padded,     # (K, A, r_max)
        'mask': mask,             # (K, r_max)
        'r_vec': r_vec,           # (K,)
    }