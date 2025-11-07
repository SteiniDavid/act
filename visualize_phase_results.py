# %% [markdown]
"""
# Phase Detection Results Visualization

This script loads and visualizes the phase detection results from the ACT phase baseline.
It analyzes the saved `.npz` file containing phase detection outputs and creates
comprehensive visualizations of the detected phases.

Usage:
    python visualize_phase_results.py

Or run interactively in IPython/Jupyter with the %% cell delimiters.
"""

# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

# Configure matplotlib for better notebook display
# Try to use inline backend for notebooks, fall back to default otherwise
try:
    # Check if running in IPython/Jupyter
    get_ipython()  # type: ignore
    # If we get here, we're in IPython/Jupyter
    matplotlib.use('module://matplotlib_inline.backend_inline')
    print("📊 Using inline matplotlib backend for notebook display")
except (NameError, ImportError):
    # Not in IPython/Jupyter, use default backend
    pass

# Import PhaseDetectionResult for proper data loading
from phasetoolkit import PhaseDetectionResult

# %% [markdown]
"""
## Configuration and Setup
"""

# %%
# Setup paths - handle both script and interactive execution
try:
    # When running as a script
    project_root = Path(__file__).parent.parent.parent.parent
except NameError:
    # When running in interactive notebook/IPython
    project_root = Path.cwd()
    # Navigate up to find project root (contains phase_results)
    while not (project_root / "phase_results").exists() and project_root != project_root.parent:
        project_root = project_root.parent

phase_results_dir = project_root / "phase_results"
results_file = phase_results_dir / "SyringeInjection-v1-phase-detection.npz"

# Output directory for visualizations
output_dir = phase_results_dir
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Loading phase detection results from: {results_file}")
print(f"📊 Output directory: {output_dir}")

# %% [markdown]
"""
## Step 1: Load Phase Detection Results
"""

# %%
if not results_file.exists():
    raise FileNotFoundError(
        f"Phase detection results not found at {results_file}\n"
        f"Please run phase detection first using run_phase_detection.sh"
    )

# Load phase detection results using the proper API
# This loads the NPZ file plus associated pickle and JSON files
phase_result = PhaseDetectionResult.load(str(results_file))

print("✅ Loaded phase detection results!")
print(f"\n📋 Phase Detection Result attributes:")
print(f"  • labels: {len(phase_result.labels)} trajectories")
print(f"  • config: K={phase_result.config.K} phases, p_stay={phase_result.config.p_stay}")
print(f"  • padded_buffers keys: {list(phase_result.padded_buffers.keys())}")
if hasattr(phase_result, 'stats') and phase_result.stats is not None:
    print(f"  • stats: Available")
if hasattr(phase_result, 'hmm_bundle') and phase_result.hmm_bundle is not None:
    print(f"  • hmm_bundle: Available for online tracking")

# %% [markdown]
"""
## Step 2: Extract and Analyze Phase Data
"""

# %%
# Extract data from PhaseDetectionResult object
# The PhaseDetectionResult contains:
# - labels: List of numpy arrays, one per trajectory with phase labels
# - config: PhaseDetectionConfig object with detection parameters
# - padded_buffers: Dict with 'mu_table', 'U_padded', 'mask', 'r_vec'
# - stats: Optional PhaseStats object with covariances and other statistics
# - hmm_bundle: Optional HMM parameters for online tracking

# Extract phase labels
phase_labels = phase_result.labels
print(f"\n🎯 Phase Labels:")
print(f"  Number of trajectories: {len(phase_labels)}")

# Analyze phase labels
all_phases = []
trajectory_lengths = []

for traj_idx, labels in enumerate(phase_labels):
    trajectory_lengths.append(len(labels))
    all_phases.extend(labels)

all_phases = np.array(all_phases)
trajectory_lengths = np.array(trajectory_lengths)

n_phases = len(np.unique(all_phases))
print(f"  Number of detected phases: {n_phases}")
print(f"  Average trajectory length: {np.mean(trajectory_lengths):.1f} ± {np.std(trajectory_lengths):.1f} steps")
print(f"  Total timesteps: {len(all_phases)}")

# Extract transition matrix from HMM bundle if available
transition_matrix = None
if hasattr(phase_result, 'hmm_bundle') and phase_result.hmm_bundle is not None:
    if hasattr(phase_result.hmm_bundle, 'A'):
        transition_matrix = phase_result.hmm_bundle.A
        print(f"\n🔄 Transition Matrix: {transition_matrix.shape}")

# Extract phase means from padded_buffers
phase_means = phase_result.padded_buffers.get('mu_table', None)
if phase_means is not None:
    print(f"\n📊 Phase Means (mu_table): {phase_means.shape}")

# Extract covariances from stats if available
phase_covs = None
if hasattr(phase_result, 'stats') and phase_result.stats is not None:
    if hasattr(phase_result.stats, 'Sigma_list'):
        phase_covs = phase_result.stats.Sigma_list
        print(f"📊 Phase Covariances: {len(phase_covs)} matrices")

# Get configuration
config = phase_result.config
print(f"\n⚙️ Configuration:")
print(f"  • K (num phases): {config.K}")
print(f"  • n_iter: {config.n_iter}")
print(f"  • p_stay: {config.p_stay}")
print(f"  • var_keep: {config.var_keep}")
print(f"  • random_state: {config.random_state}")

# %% [markdown]
"""
## Step 3: Phase Distribution Analysis
"""

# %%
if phase_labels is not None:
    print("\n📈 Computing phase distribution statistics...")

    # Overall phase distribution
    phase_counts = np.bincount(all_phases)
    phase_proportions = phase_counts / len(all_phases)

    print(f"\n📊 Overall Phase Distribution:")
    for phase_idx, (count, prop) in enumerate(zip(phase_counts, phase_proportions)):
        print(f"  Phase {phase_idx}: {count:6d} steps ({prop*100:5.1f}%)")

    # Phase duration statistics
    phase_durations = {i: [] for i in range(n_phases)}

    for traj_labels in phase_labels:
        if len(traj_labels) == 0:
            continue

        current_phase = traj_labels[0]
        current_duration = 1

        for step_idx in range(1, len(traj_labels)):
            if traj_labels[step_idx] == current_phase:
                current_duration += 1
            else:
                phase_durations[current_phase].append(current_duration)
                current_phase = traj_labels[step_idx]
                current_duration = 1

        # Add final phase duration
        phase_durations[current_phase].append(current_duration)

    print(f"\n⏱️ Phase Duration Statistics:")
    for phase_idx in range(n_phases):
        durations = np.array(phase_durations[phase_idx])
        if len(durations) > 0:
            print(f"  Phase {phase_idx}: {np.mean(durations):5.1f} ± {np.std(durations):5.1f} steps "
                  f"(median: {np.median(durations):.1f}, range: [{np.min(durations)}, {np.max(durations)}])")

    # Transition statistics
    print(f"\n🔄 Phase Transition Statistics:")
    n_transitions_per_traj = []

    for traj_labels in phase_labels:
        if len(traj_labels) <= 1:
            n_transitions_per_traj.append(0)
            continue

        transitions = np.sum(traj_labels[1:] != traj_labels[:-1])
        n_transitions_per_traj.append(transitions)

    n_transitions_per_traj = np.array(n_transitions_per_traj)
    print(f"  Average transitions per trajectory: {np.mean(n_transitions_per_traj):.2f} ± {np.std(n_transitions_per_traj):.2f}")
    print(f"  Median transitions: {np.median(n_transitions_per_traj):.0f}")
    print(f"  Range: [{np.min(n_transitions_per_traj)}, {np.max(n_transitions_per_traj)}]")

# %% [markdown]
"""
## Step 4: Visualization 1 - Phase Distribution
"""

# %%
if phase_labels is not None:
    print("\n🎨 Creating phase distribution visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase Detection Analysis - SyringeInjection-v1',
                 fontsize=16, fontweight='bold')

    # Plot 1: Overall phase distribution (pie chart)
    ax = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, n_phases))
    ax.pie(phase_counts, labels=[f'Phase {i}' for i in range(n_phases)],
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Overall Phase Distribution')

    # Plot 2: Phase distribution (bar chart)
    ax = axes[0, 1]
    bars = ax.bar(range(n_phases), phase_counts, color=colors, edgecolor='black')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Number of Steps')
    ax.set_title('Phase Frequency')
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels([f'Phase {i}' for i in range(n_phases)])
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    # Plot 3: Phase duration distribution (box plot)
    ax = axes[1, 0]
    duration_data = [phase_durations[i] for i in range(n_phases)]
    bp = ax.boxplot(duration_data, labels=[f'P{i}' for i in range(n_phases)],
                    patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Phase')
    ax.set_ylabel('Duration (steps)')
    ax.set_title('Phase Duration Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Transitions per trajectory (histogram)
    ax = axes[1, 1]
    ax.hist(n_transitions_per_traj, bins=range(0, int(np.max(n_transitions_per_traj))+2),
            color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Transitions')
    ax.set_ylabel('Number of Trajectories')
    ax.set_title('Transitions per Trajectory')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean line
    mean_transitions = np.mean(n_transitions_per_traj)
    ax.axvline(mean_transitions, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_transitions:.1f}')
    ax.legend()

    plt.tight_layout()
    save_path = output_dir / "phase_distribution_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()

# %% [markdown]
"""
## Step 5: Visualization 2 - Transition Matrix Heatmap
"""

# %%
if transition_matrix is not None:
    print("\n🎨 Creating transition matrix visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase Transition Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Transition matrix heatmap
    ax = axes[0]
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('To Phase')
    ax.set_ylabel('From Phase')
    ax.set_title('Transition Probability Matrix')
    ax.set_xticks(range(n_phases))
    ax.set_yticks(range(n_phases))
    ax.set_xticklabels([f'P{i}' for i in range(n_phases)])
    ax.set_yticklabels([f'P{i}' for i in range(n_phases)])

    # Add text annotations
    for i in range(n_phases):
        for j in range(n_phases):
            text = ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax, label='Probability')

    # Plot 2: Stay probabilities
    ax = axes[1]
    stay_probs = np.diag(transition_matrix)
    colors = plt.cm.Set3(np.linspace(0, 1, n_phases))
    bars = ax.bar(range(n_phases), stay_probs, color=colors, edgecolor='black')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Stay Probability')
    ax.set_title('Phase Persistence (Diagonal Elements)')
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels([f'Phase {i}' for i in range(n_phases)])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "transition_matrix_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()

# %% [markdown]
"""
## Step 6: Visualization 3 - Individual Trajectory Phase Sequences
"""

# %%
if phase_labels is not None:
    print("\n🎨 Creating trajectory phase sequence visualization...")

    # Select a subset of trajectories to visualize (e.g., first 20)
    n_trajs_to_plot = min(20, len(phase_labels))

    fig, ax = plt.subplots(figsize=(16, 8))

    # Create phase color map
    colors = plt.cm.Set3(np.linspace(0, 1, n_phases))
    phase_colormap = {i: colors[i] for i in range(n_phases)}

    # Plot each trajectory as a horizontal bar
    for traj_idx in range(n_trajs_to_plot):
        labels = phase_labels[traj_idx]

        # Create segments for continuous phases
        current_phase = labels[0]
        start_idx = 0

        for step_idx in range(1, len(labels) + 1):
            # Check if phase changes or we're at the end
            if step_idx == len(labels) or labels[step_idx] != current_phase:
                # Draw the segment
                color = phase_colormap[current_phase]
                ax.barh(traj_idx, step_idx - start_idx, left=start_idx,
                       height=0.8, color=color, edgecolor='black', linewidth=0.5)

                if step_idx < len(labels):
                    current_phase = labels[step_idx]
                    start_idx = step_idx

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Trajectory Index')
    ax.set_title(f'Phase Sequences for {n_trajs_to_plot} Trajectories')
    ax.set_ylim(-0.5, n_trajs_to_plot - 0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=phase_colormap[i], edgecolor='black',
                            label=f'Phase {i}')
                      for i in range(n_phases)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=n_phases)

    plt.tight_layout()
    save_path = output_dir / "trajectory_phase_sequences.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()

# %% [markdown]
"""
## Step 7: Summary Statistics Report
"""

# %%
print("\n" + "="*70)
print("📊 PHASE DETECTION RESULTS SUMMARY")
print("="*70)

print(f"\n📁 Results File: {results_file.name}")
print(f"📂 Output Directory: {output_dir}")

if phase_labels is not None:
    print(f"\n🎯 Detection Summary:")
    print(f"  • Number of trajectories: {len(phase_labels)}")
    print(f"  • Number of detected phases: {n_phases}")
    print(f"  • Total timesteps analyzed: {len(all_phases)}")
    print(f"  • Average trajectory length: {np.mean(trajectory_lengths):.1f} ± {np.std(trajectory_lengths):.1f}")

    print(f"\n📈 Phase Distribution:")
    for phase_idx, (count, prop) in enumerate(zip(phase_counts, phase_proportions)):
        print(f"  Phase {phase_idx}: {count:6d} steps ({prop*100:5.1f}%)")

    print(f"\n⏱️ Average Phase Durations:")
    for phase_idx in range(n_phases):
        durations = np.array(phase_durations[phase_idx])
        if len(durations) > 0:
            print(f"  Phase {phase_idx}: {np.mean(durations):5.1f} ± {np.std(durations):5.1f} steps")

    print(f"\n🔄 Transition Statistics:")
    print(f"  • Average transitions per trajectory: {np.mean(n_transitions_per_traj):.2f} ± {np.std(n_transitions_per_traj):.2f}")
    print(f"  • Median transitions: {np.median(n_transitions_per_traj):.0f}")

    if transition_matrix is not None:
        print(f"\n🔁 Phase Persistence (Stay Probabilities):")
        for phase_idx in range(n_phases):
            print(f"  Phase {phase_idx}: {transition_matrix[phase_idx, phase_idx]:.3f}")

print(f"\n🎨 Generated Visualizations:")
print(f"  • phase_distribution_analysis.png")
if transition_matrix is not None:
    print(f"  • transition_matrix_analysis.png")
print(f"  • trajectory_phase_sequences.png")

print("\n" + "="*70)

# %%
