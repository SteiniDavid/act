"""
Preprocess demonstration data by adding canonical actions and phase labels to HDF5 files.

This script loads phase detection results, computes canonical (phase-normalized) actions
for each timestep using RealNVP normalizing flows, and saves them back to the HDF5
demonstration files. This allows phase-conditioned ACT training to use pre-computed
canonical actions instead of computing them on-the-fly during training.

Usage:
    python preprocess_canonical_actions.py \
        --dataset_dir data/sim_transfer_cube_scripted \
        --phase_result_path phase_results/transfer_cube_K3.npz \
        --num_episodes 50 \
        --flow_preset medium
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Literal

import h5py
import numpy as np
import torch
from tqdm import tqdm

from phasetoolkit import PhaseDetectionResult
from phasetoolkit.gaussianization import (
    RealNVPPhaseGaussianizer,
    RealNVPConfig,
    save_gaussianizer,
    GaussianizerSpec,
)


def collect_phase_actions(
    dataset_dir: Path,
    phase_labels: Dict[int, np.ndarray],
    num_episodes: int,
    num_phases: int,
) -> Dict[int, np.ndarray]:
    """
    Collect all actions grouped by phase across all episodes.

    Args:
        dataset_dir: Directory containing episode HDF5 files
        phase_labels: Dictionary mapping episode_idx -> phase labels array
        num_episodes: Number of episodes to process
        num_phases: Number of phases

    Returns:
        Dictionary mapping phase_id -> array of actions (N_k, action_dim)
    """
    phase_actions = {k: [] for k in range(num_phases)}

    print("Collecting actions by phase...")
    for episode_idx in tqdm(range(num_episodes), desc="Collecting actions"):
        dataset_path = dataset_dir / f'episode_{episode_idx}.hdf5'
        if not dataset_path.exists():
            print(f"\nWARNING: Episode file not found: {dataset_path}")
            continue

        with h5py.File(dataset_path, 'r') as f:
            actions = f['/action'][:]  # (T, action_dim)

        episode_labels = phase_labels[episode_idx]
        if len(episode_labels) != len(actions):
            raise ValueError(
                f"Phase labels length {len(episode_labels)} != episode length {len(actions)} "
                f"for episode {episode_idx}"
            )

        # Group actions by phase
        for t in range(len(actions)):
            phase_id = int(episode_labels[t])
            phase_actions[phase_id].append(actions[t])

    # Convert lists to arrays
    for phase_id in range(num_phases):
        if len(phase_actions[phase_id]) > 0:
            phase_actions[phase_id] = np.array(phase_actions[phase_id])
            print(f"  Phase {phase_id}: {phase_actions[phase_id].shape[0]} samples")
        else:
            raise ValueError(f"Phase {phase_id} has no samples!")

    return phase_actions


def process_episode(
    dataset_path: str,
    episode_labels: np.ndarray,
    gaussianizer: RealNVPPhaseGaussianizer,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Process a single episode: compute canonical actions using RealNVP and add to HDF5 file.

    Args:
        dataset_path: Path to episode HDF5 file
        episode_labels: Phase labels for each timestep in episode (T,)
        gaussianizer: Fitted RealNVPPhaseGaussianizer
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

        # Compute canonical actions for each timestep using RealNVP
        canonical_actions = np.zeros_like(actions)
        for t in range(episode_len):
            phase_id = int(episode_labels[t])
            # Transform single action using the flow for this phase
            action_batch = actions[t:t+1]  # (1, action_dim)
            result = gaussianizer.transform(phase_id, action_batch)
            canonical_actions[t] = result.transformed[0]  # Extract from batch

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
        description="Preprocess demonstrations with canonical actions using RealNVP flows"
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
    parser.add_argument(
        "--flow_preset",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="RealNVP configuration preset (default: medium)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training flows (default: cuda if available)",
    )
    parser.add_argument(
        "--gaussianizer_save_path",
        type=str,
        default=None,
        help="Path to save trained gaussianizer (default: <phase_result_dir>/<task_name>_gaussianizer.pkl)",
    )
    args = parser.parse_args()

    # Validate paths
    dataset_dir = Path(args.dataset_dir).expanduser()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    phase_result_path = Path(args.phase_result_path).expanduser()
    if not phase_result_path.exists():
        raise FileNotFoundError(f"Phase result file not found: {phase_result_path}")

    # Set up gaussianizer save path
    if args.gaussianizer_save_path is None:
        task_name = dataset_dir.name
        gaussianizer_save_path = phase_result_path.parent / f"{task_name}_gaussianizer.pkl"
    else:
        gaussianizer_save_path = Path(args.gaussianizer_save_path).expanduser()

    print("=" * 70)
    print("CANONICAL ACTION PREPROCESSING WITH REALNVP FLOWS")
    print("=" * 70)
    print(f"\nDataset directory: {dataset_dir}")
    print(f"Phase result path: {phase_result_path}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Flow preset: {args.flow_preset}")
    print(f"Device: {args.device}")
    print(f"Overwrite existing: {args.overwrite}")
    print(f"Gaussianizer save path: {gaussianizer_save_path}")

    # Load phase detection results
    print(f"\n[1/4] Loading phase detection results...")
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

    # Collect actions by phase
    print(f"\n[2/4] Collecting actions by phase...")
    phase_actions = collect_phase_actions(
        dataset_dir=dataset_dir,
        phase_labels=phase_result.labels,
        num_episodes=args.num_episodes,
        num_phases=phase_result.config.K,
    )

    # Determine action dimension
    action_dim = next(iter(phase_actions.values())).shape[1]
    print(f"\n✓ Collected actions for {phase_result.config.K} phases")
    print(f"  - Action dimension: {action_dim}")

    # Train RealNVP gaussianizer
    print(f"\n[3/4] Training RealNVP flows (one per phase)...")
    device = torch.device(args.device)
    config = RealNVPConfig.preset(args.flow_preset)

    print(f"  - Configuration: {args.flow_preset}")
    print(f"    • Coupling layers: {config.num_coupling_layers}")
    print(f"    • Hidden dim: {config.hidden_dim}")
    print(f"    • Max epochs: {config.max_epochs}")
    print(f"    • Batch size: {config.batch_size}")
    print(f"    • Learning rate: {config.learning_rate}")

    gaussianizer = RealNVPPhaseGaussianizer(
        config=config,
        action_dim=action_dim,
        device=device,
    )

    print(f"\nFitting gaussianizer...")
    gaussianizer.fit(phase_actions)
    print(f"✓ Trained {phase_result.config.K} RealNVP flows")

    # Print training diagnostics
    for phase_id in range(phase_result.config.K):
        diag = gaussianizer.diagnostics(phase_id)
        print(f"  Phase {phase_id}: best_loss={diag['best_loss']:.4f}, epochs={diag['epochs_trained']}")

    # Save gaussianizer
    print(f"\nSaving gaussianizer to {gaussianizer_save_path}...")
    spec = GaussianizerSpec(
        label=f"{dataset_dir.name}_{args.flow_preset}",
        kind="realnvp",
        config=config,
        extra_kwargs={"device": device},
    )
    save_gaussianizer(gaussianizer_save_path, spec, gaussianizer)
    print(f"✓ Saved gaussianizer")

    # Process each episode with trained gaussianizer
    print(f"\n[4/4] Processing episodes with trained flows...")
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
            gaussianizer,
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
    print("  • /canonical_actions - RealNVP flow-normalized actions (T, A)")
    print("  • /phase_labels - Phase labels for each timestep (T,)")

    print(f"\nSaved trained gaussianizer:")
    print(f"  • {gaussianizer_save_path}")

    print("\nYou can now train with:")
    print(f"  python imitate_episodes.py \\")
    print(f"    --task_name <task> \\")
    print(f"    --use_canonical \\")
    print(f"    --num_phases {phase_result.config.K} \\")
    print(f"    --phase_result_path {phase_result_path} \\")
    print(f"    --gaussianizer_path {gaussianizer_save_path}")
    print()


if __name__ == "__main__":
    main()
