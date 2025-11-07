"""
Hidden Markov Model functionality for phase detection.

This module provides HMM training and inference capabilities using hmmlearn,
with specialized support for left-to-right (Bakis) topology for phase detection.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from hmmlearn.hmm import GaussianHMM


def left_to_right_transmat(K: int, p_stay: float = 0.7) -> np.ndarray:
    """
    Build a KxK left-to-right transition matrix where state k can stay (k->k)
    with prob p_stay, or advance (k->k+1) with prob 1-p_stay. Last state is absorbing.
    
    Args:
        K: Number of states
        p_stay: Probability of staying in current state
        
    Returns:
        Transition matrix of shape (K, K)
    """
    A = np.zeros((K, K), dtype=np.float64)
    for k in range(K - 1):
        A[k, k] = p_stay
        A[k, k + 1] = 1.0 - p_stay
    A[K - 1, K - 1] = 1.0
    return A


def time_slice_init_means_vars(demos_features: List[np.ndarray], K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize per-state (mu, var) by time-slicing each demo into K chunks and
    aggregating statistics per chunk. Variances are diagonal.
    
    Args:
        demos_features: List of feature arrays, each of shape (T_i, D)
        K: Number of states
        
    Returns:
        means: Array of shape (K, D) - mean for each state
        vars: Array of shape (K, D) - diagonal variances for each state
    """
    D = demos_features[0].shape[1]
    means = np.zeros((K, D), dtype=np.float64)
    vars_ = np.zeros((K, D), dtype=np.float64)

    for k in range(K):
        xs = []
        for x in demos_features:
            T = len(x)
            a = int(np.floor(k     * T / K))
            b = int(np.floor((k+1) * T / K))
            xs.append(x[a:b if b > a else min(a + 1, T)])
        Xk = np.concatenate(xs, axis=0).astype(np.float64)
        mk = Xk.mean(axis=0)
        vk = np.maximum(Xk.var(axis=0), 1e-6)  # keep strictly positive
        means[k] = mk
        vars_[k] = vk
    return means, vars_


@dataclass
class HMMBundle:
    """
    Container for learned HMM parameters.
    
    This bundle contains all the parameters needed for online phase tracking
    and can be serialized/deserialized for later use.
    """
    K: int                          # Number of states
    D: int                          # Feature dimension
    means_: np.ndarray             # (K, D) - emission means
    covars_: np.ndarray            # (K, D) for diagonal covariance
    transmat_: np.ndarray          # (K, K) - transition matrix
    startprob_: np.ndarray         # (K,) - initial state probabilities


def fit_left_to_right_hmm(
    demos_features: List[np.ndarray],
    K: int,
    n_iter: int = 25,
    p_stay: float = 0.7,
    tol: float = 1e-4,
    random_state: int = 0
) -> Tuple[GaussianHMM, HMMBundle]:
    """
    Fit a diagonal-Gaussian left-to-right HMM on feature sequences.
    
    The HMM uses a left-to-right (Bakis) topology where states can only 
    stay or advance to the next state. This is ideal for sequential phases
    that don't repeat or go backward.
    
    Args:
        demos_features: List of feature arrays, each of shape (T_i, D)
        K: Number of phases/states
        n_iter: Maximum EM iterations
        p_stay: Probability of staying in current state
        tol: Convergence tolerance
        random_state: Random seed for reproducibility
        
    Returns:
        model: Trained GaussianHMM model
        bundle: HMMBundle with parameters for online tracking
    """
    # Concatenate sequences for hmmlearn; supply lengths to respect boundaries
    lengths = [len(x) for x in demos_features]
    X = np.concatenate(demos_features, axis=0).astype(np.float64)

    D = X.shape[1]
    trans = left_to_right_transmat(K, p_stay=p_stay)
    start = np.zeros(K, dtype=np.float64)
    start[0] = 1.0

    # Initialize emissions with time-slicing
    means, vars_diag = time_slice_init_means_vars(demos_features, K)

    # Create and configure HMM model
    model = GaussianHMM(
        n_components=K,
        covariance_type="diag",
        n_iter=n_iter,
        tol=tol,
        algorithm="viterbi",     # used for .predict; training uses EM internally
        init_params="",          # do NOT overwrite our init params
        params="mc",             # update only means (m) and covars (c), fix start/trans
        random_state=random_state,
        verbose=False,
    )

    # Set initial parameters (freeze start/trans by leaving 's'/'t' out of params)
    model.startprob_ = start
    model.transmat_  = trans
    model.means_     = means
    model.covars_    = vars_diag

    # Train model
    model.fit(X, lengths)  # EM over means/covars only

    # Create parameter bundle for external use
    bundle = HMMBundle(
        K=K, D=D,
        means_=model.means_.copy(),
        covars_=model.covars_.copy(),
        transmat_=model.transmat_.copy(),
        startprob_=model.startprob_.copy()
    )
    return model, bundle


def viterbi_labels_per_demo(model: GaussianHMM, demos_features: List[np.ndarray]) -> List[np.ndarray]:
    """
    Decode most likely state sequence (Viterbi) for each demo independently.
    
    Args:
        model: Trained GaussianHMM model
        demos_features: List of feature arrays, each of shape (T_i, D)
        
    Returns:
        List of label arrays, each of shape (T_i,) with integer state labels
    """
    labels = []
    for x in demos_features:
        z = model.predict(x.astype(np.float64))  # shape (T,)
        labels.append(z.astype(np.int64))
    return labels