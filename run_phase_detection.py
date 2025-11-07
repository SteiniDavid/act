"""
Run phase detection on demonstration trajectories and save results.

This script loads action sequences from an HDF5 demonstration file, runs phase detection
using the phasetoolkit API, and saves the results for use in phase-conditioned training.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from ai_syringe_injection.baselines.act_phase.phasetoolkit import (
    detect_phases,
    PhaseDetectionConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Run phase detection on demonstration trajectories")
    parser.add_argument(
        "--demo-path",
        type=str,
        required=True,
        help="Path to H5 demonstration file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save phase detection results (.pkl file)",
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=3,
        help="Number of phases to detect (default: 3)",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help="Number of demonstrations to use (default: all)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Number of HMM training iterations (default: 25)",
    )
    parser.add_argument(
        "--p-stay",
        type=float,
        default=0.75,
        help="Phase persistence probability for left-to-right HMM (default: 0.75)",
    )
    parser.add_argument(
        "--var-keep",
        type=float,
        default=0.95,
        help="Fraction of variance to retain in PCA (default: 0.95)",
    )
    parser.add_argument(
        "--use-actions",
        action="store_true",
        default=True,
        help="Use actions for clustering (default: True, recommended for robotics)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Validate paths
    demo_path = Path(args.demo_path).expanduser()
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure the output path has .npz extension for proper save/load
    if output_path.suffix == '.pkl':
        output_path = output_path.with_suffix('.npz')

    print(f"Loading demonstrations from: {demo_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Configuration:")
    print(f"  - Number of phases (K): {args.num_phases}")
    print(f"  - HMM iterations: {args.n_iter}")
    print(f"  - Phase persistence (p_stay): {args.p_stay}")
    print(f"  - PCA variance retention: {args.var_keep}")
    print(f"  - Use actions for clustering: {args.use_actions}")
    print(f"  - Random state: {args.random_state}")

    # Load action sequences from H5 file
    print("\nExtracting action sequences from trajectories...")
    actions_list = []

    with h5py.File(demo_path, "r") as f:
        # Sort trajectory keys by numeric suffix (e.g., traj_0, traj_1, ...)
        traj_keys = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))

        # Limit number of trajectories if specified
        if args.num_demos is not None:
            traj_keys = traj_keys[:args.num_demos]

        print(f"Found {len(traj_keys)} trajectories to process")

        for i, traj_key in enumerate(traj_keys):
            actions = f[traj_key]["actions"][:]
            actions_list.append(actions)
            if (i + 1) % 10 == 0 or (i + 1) == len(traj_keys):
                print(f"  Loaded {i + 1}/{len(traj_keys)} trajectories", end="\r")

        print()  # New line after progress

        # Print statistics
        action_dim = actions_list[0].shape[1]
        traj_lengths = [act.shape[0] for act in actions_list]
        print(f"\nDataset statistics:")
        print(f"  - Number of trajectories: {len(actions_list)}")
        print(f"  - Action dimension: {action_dim}")
        print(f"  - Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")

    # Configure phase detection
    config = PhaseDetectionConfig(
        K=args.num_phases,
        n_iter=args.n_iter,
        p_stay=args.p_stay,
        var_keep=args.var_keep,
        random_state=args.random_state,
    )

    # Run phase detection
    print(f"\nRunning phase detection with K={args.num_phases} phases...")
    print("This may take a few minutes depending on dataset size...")

    phase_result = detect_phases(
        trajectory_data=actions_list,
        actions=actions_list,
        config=config,
        use_actions_for_clustering=args.use_actions,
    )

    # Print phase statistics
    print("\nPhase detection completed!")
    print(f"\nPhase label statistics:")
    for traj_idx, labels in enumerate(phase_result.labels):
        unique_phases = np.unique(labels)
        phase_counts = {phase: np.sum(labels == phase) for phase in unique_phases}
        if traj_idx < 3:  # Show first 3 trajectories
            print(f"  Traj {traj_idx}: phases {unique_phases.tolist()}, counts {phase_counts}")
        elif traj_idx == 3:
            print(f"  ... ({len(phase_result.labels) - 3} more trajectories)")
            break

    # Compute overall phase statistics
    all_labels = np.concatenate(phase_result.labels)
    for phase_id in range(args.num_phases):
        count = np.sum(all_labels == phase_id)
        percentage = 100 * count / len(all_labels)
        print(f"  Phase {phase_id}: {count} timesteps ({percentage:.1f}%)")

    # Save results
    print(f"\nSaving phase detection results to: {output_path}")
    phase_result.save(str(output_path))

    print("\nPhase detection complete! Results saved successfully.")
    print(f"\nYou can now run training with:")
    print(f"  --use-canonical")
    print(f"  --num-phases {args.num_phases}")
    print(f"  --phase-result-path {output_path}")


if __name__ == "__main__":
    main()
