"""
Preprocess demonstration data by adding canonical actions and phase labels to HDF5 files.

This script loads phase detection results, computes canonical (phase-normalized) actions
for each timestep, and saves them back to the HDF5 demonstration files. This allows
phase-conditioned ACT training to use pre-computed canonical actions instead of computing
them on-the-fly during training.

Usage:
    python preprocess_canonical_actions.py \
        --dataset_dir data/sim_transfer_cube_scripted \
        --phase_result_path phase_results/transfer_cube_K3.npz \
        --num_episodes 50
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from tqdm import tqdm

from phasetoolkit import PhaseDetectionResult


def compute_canonical_action(action: np.ndarray, phase_id: int, phase_stats) -> np.ndarray:
    """
    Convert environment action to canonical (phase-normalized) action.

    This is the inverse transformation of what PhaseProjector.to_action() does:
    - Environment action: a
    - Canonical action: c = L_k^{-1} @ (a - μ_k)

    Where:
    - μ_k: Mean action for phase k
    - L_k: Cholesky factor of covariance for phase k

    Args:
        action: Environment action vector (A,)
        phase_id: Phase identifier
        phase_stats: PhaseStats object containing per-phase statistics

    Returns:
        Canonical action vector (A,)
    """
    if phase_stats is None:
        # No phase stats available, return action as-is
        return action

    # Get phase statistics
    mu_k = phase_stats.mu_list[phase_id]
    Sigma_k = phase_stats.Sigma_list[phase_id]

    # Compute Cholesky factor L such that Sigma = L @ L^T
    L_k = np.linalg.cholesky(Sigma_k + 1e-6 * np.eye(Sigma_k.shape[0]))

    # Invert: c = L^{-1} @ (a - mu)
    canonical = np.linalg.solve(L_k, action - mu_k)

    return canonical


def process_episode(
    dataset_path: str,
    episode_labels: np.ndarray,
    phase_stats,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Process a single episode: compute canonical actions and add to HDF5 file.

    Args:
        dataset_path: Path to episode HDF5 file
        episode_labels: Phase labels for each timestep in episode (T,)
        phase_stats: PhaseStats object
        overwrite: Whether to overwrite existing canonical_actions dataset

    Returns:
        Dictionary with processing statistics
    """
    stats = {'timesteps': 0, 'skipped': False}

    with h5py.File(dataset_path, 'r+') as f:
        # Check if canonical actions already exist
        if '/canonical_actions' in f and not overwrite:
            stats['skipped'] = True
            return stats

        # Load actions
        actions = f['/action'][:]  # Shape: (T, A)
        episode_len = actions.shape[0]
        stats['timesteps'] = episode_len

        # Verify labels match episode length
        if len(episode_labels) != episode_len:
            raise ValueError(
                f"Phase labels length {len(episode_labels)} does not match "
                f"episode length {episode_len} for {dataset_path}"
            )

        # Compute canonical actions for each timestep
        canonical_actions = np.zeros_like(actions)
        for t in range(episode_len):
            phase_id = int(episode_labels[t])
            canonical_actions[t] = compute_canonical_action(
                actions[t], phase_id, phase_stats
            )

        # Save canonical actions to HDF5
        if '/canonical_actions' in f:
            del f['/canonical_actions']
        f.create_dataset('/canonical_actions', data=canonical_actions, dtype=np.float32)

        # Save phase labels to HDF5
        if '/phase_labels' in f:
            del f['/phase_labels']
        f.create_dataset('/phase_labels', data=episode_labels, dtype=np.int32)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess demonstrations with canonical actions and phase labels"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing episode HDF5 files (episode_0.hdf5, episode_1.hdf5, ...)",
    )
    parser.add_argument(
        "--phase_result_path",
        type=str,
        required=True,
        help="Path to phase detection results (.npz file)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        required=True,
        help="Number of episodes to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing canonical_actions if present",
    )
    args = parser.parse_args()

    # Validate paths
    dataset_dir = Path(args.dataset_dir).expanduser()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    phase_result_path = Path(args.phase_result_path).expanduser()
    if not phase_result_path.exists():
        raise FileNotFoundError(f"Phase result file not found: {phase_result_path}")

    print("=" * 70)
    print("CANONICAL ACTION PREPROCESSING")
    print("=" * 70)
    print(f"\nDataset directory: {dataset_dir}")
    print(f"Phase result path: {phase_result_path}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Overwrite existing: {args.overwrite}")

    # Load phase detection results
    print(f"\nLoading phase detection results...")
    phase_result = PhaseDetectionResult.load(str(phase_result_path))

    print(f"✓ Loaded phase detection results")
    print(f"  - Number of phases: {phase_result.config.K}")
    print(f"  - Number of trajectories with labels: {len(phase_result.labels)}")

    # Validate number of episodes matches
    if len(phase_result.labels) < args.num_episodes:
        raise ValueError(
            f"Phase result has {len(phase_result.labels)} trajectory labels but "
            f"requested to process {args.num_episodes} episodes. Please ensure phase "
            f"detection was run on all episodes."
        )

    # Get phase statistics
    phase_stats = phase_result.stats
    if phase_stats is None:
        print("WARNING: No phase statistics found in phase result. Using identity transform.")

    # Process each episode
    print(f"\nProcessing episodes...")
    total_timesteps = 0
    skipped_count = 0

    for episode_idx in tqdm(range(args.num_episodes), desc="Processing episodes"):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')

        if not os.path.exists(dataset_path):
            print(f"\nWARNING: Episode file not found: {dataset_path}")
            continue

        # Get phase labels for this episode
        episode_labels = phase_result.labels[episode_idx]

        # Process episode
        stats = process_episode(
            dataset_path,
            episode_labels,
            phase_stats,
            overwrite=args.overwrite,
        )

        total_timesteps += stats['timesteps']
        if stats['skipped']:
            skipped_count += 1

    # Print summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nProcessed episodes: {args.num_episodes}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Total timesteps processed: {total_timesteps}")

    print("\nAdded to each HDF5 file:")
    print("  • /canonical_actions - Phase-normalized actions (T, A)")
    print("  • /phase_labels - Phase labels for each timestep (T,)")

    print("\nYou can now train with:")
    print(f"  python imitate_episodes.py \\")
    print(f"    --task_name <task> \\")
    print(f"    --use_canonical \\")
    print(f"    --num_phases {phase_result.config.K} \\")
    print(f"    --phase_result_path {phase_result_path}")
    print()


if __name__ == "__main__":
    main()
