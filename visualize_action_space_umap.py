# %% [markdown]
# # UMAP Visualization of Demonstration Action Space
#
# This notebook visualizes the raw environment action space using UMAP dimensionality reduction.
# We compare two coloring schemes:
# 1. Phase-based coloring (from detected phases)
# 2. Temporal progression coloring (episode progression in 5 bins)

# %% [markdown]
## Setup and Imports

# %%
import h5py
import numpy as np
from pathlib import Path
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Tuple, List

print("Libraries imported successfully")

# %% [markdown]
## Configuration

# %%
# Dataset configuration
DATASET_DIR = Path('/home/steini/Documents/git/act/dataset_dir')
NUM_EPISODES = 50
EPISODE_LEN = 400  # Expected episode length

# UMAP parameters (defaults)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'euclidean'
RANDOM_STATE = 42

# Timestep binning (5 bins for temporal progression)
NUM_TEMPORAL_BINS = 5
TEMPORAL_BIN_LABELS = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

print(f"Dataset directory: {DATASET_DIR}")
print(f"Number of episodes: {NUM_EPISODES}")
print(f"UMAP parameters: n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}")

# %% [markdown]
## Data Loading

# %%
def load_demonstration_data(dataset_dir: Path, num_episodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw actions and phase labels from all episodes.

    Returns:
        actions: (N, 14) array of environment actions
        phase_labels: (N,) array of phase assignments
        episode_ids: (N,) array indicating which episode each timestep belongs to
        timestep_ids: (N,) array indicating timestep within episode
    """
    all_actions = []
    all_phase_labels = []
    all_episode_ids = []
    all_timestep_ids = []

    print(f"Loading data from {num_episodes} episodes...")

    for ep_idx in range(num_episodes):
        episode_file = dataset_dir / f'episode_{ep_idx}.hdf5'

        if not episode_file.exists():
            print(f"Warning: {episode_file} not found, skipping...")
            continue

        with h5py.File(episode_file, 'r') as f:
            # Load actions and phase labels
            actions = f['/action'][:]
            phase_labels = f['/phase_labels'][:]

            episode_len = len(actions)

            all_actions.append(actions)
            all_phase_labels.append(phase_labels)
            all_episode_ids.append(np.full(episode_len, ep_idx))
            all_timestep_ids.append(np.arange(episode_len))

        if (ep_idx + 1) % 10 == 0:
            print(f"  Loaded {ep_idx + 1}/{num_episodes} episodes")

    # Concatenate all data
    actions = np.concatenate(all_actions, axis=0)
    phase_labels = np.concatenate(all_phase_labels, axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)
    timestep_ids = np.concatenate(all_timestep_ids, axis=0)

    print(f"\nData loaded successfully!")
    print(f"  Total timesteps: {len(actions)}")
    print(f"  Action dimensions: {actions.shape[1]}")
    print(f"  Unique phases: {np.unique(phase_labels)}")

    return actions, phase_labels, episode_ids, timestep_ids


# Load the data
actions, phase_labels, episode_ids, timestep_ids = load_demonstration_data(DATASET_DIR, NUM_EPISODES)

# %% [markdown]
## Temporal Binning

# %%
def create_temporal_bins(timestep_ids: np.ndarray, episode_len: int, num_bins: int = 5) -> np.ndarray:
    """
    Create temporal bins based on progression through episode.

    Args:
        timestep_ids: Array of timestep indices within episodes
        episode_len: Expected length of episodes
        num_bins: Number of bins to create

    Returns:
        bin_labels: Array of bin assignments (0 to num_bins-1)
    """
    # Normalize timesteps to [0, 1] range
    normalized_timesteps = timestep_ids / episode_len

    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_labels = np.digitize(normalized_timesteps, bins[1:])  # Returns 0 to num_bins-1

    return bin_labels


# Create temporal bins
temporal_bins = create_temporal_bins(timestep_ids, EPISODE_LEN, NUM_TEMPORAL_BINS)

print("Temporal binning complete!")
print(f"  Bin distribution:")
for bin_idx in range(NUM_TEMPORAL_BINS):
    count = np.sum(temporal_bins == bin_idx)
    print(f"    Bin {bin_idx} ({TEMPORAL_BIN_LABELS[bin_idx]}): {count} timesteps ({100*count/len(temporal_bins):.1f}%)")

# %% [markdown]
## Summary Statistics

# %%
print("="*60)
print("DATA SUMMARY")
print("="*60)
print(f"\nDataset Statistics:")
print(f"  Total timesteps: {len(actions)}")
print(f"  Action dimensions: {actions.shape[1]}")
print(f"  Episodes: {NUM_EPISODES}")
print(f"  Avg timesteps per episode: {len(actions) / NUM_EPISODES:.1f}")

print(f"\nPhase Distribution:")
unique_phases, phase_counts = np.unique(phase_labels, return_counts=True)
for phase, count in zip(unique_phases, phase_counts):
    print(f"  Phase {int(phase)}: {count} timesteps ({100*count/len(phase_labels):.1f}%)")

print(f"\nAction Statistics:")
print(f"  Mean: {actions.mean(axis=0)}")
print(f"  Std:  {actions.std(axis=0)}")
print(f"  Min:  {actions.min(axis=0)}")
print(f"  Max:  {actions.max(axis=0)}")

print("="*60)

# %% [markdown]
## UMAP Dimensionality Reduction

# %%
print("Running UMAP dimensionality reduction...")
print(f"  Input shape: {actions.shape}")

# 2D UMAP
print("\n  Computing 2D embedding...")
umap_2d = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    metric=UMAP_METRIC,
    random_state=RANDOM_STATE,
    verbose=True
)
embedding_2d = umap_2d.fit_transform(actions)
print(f"    2D embedding shape: {embedding_2d.shape}")

# 3D UMAP
print("\n  Computing 3D embedding...")
umap_3d = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=3,
    metric=UMAP_METRIC,
    random_state=RANDOM_STATE,
    verbose=True
)
embedding_3d = umap_3d.fit_transform(actions)
print(f"    3D embedding shape: {embedding_3d.shape}")

print("\nUMAP computation complete!")

# %% [markdown]
## 2D Visualizations

# %%
def create_2d_scatter(embedding: np.ndarray,
                     color_values: np.ndarray,
                     color_labels: np.ndarray,
                     title: str,
                     colorscale: str = 'Viridis',
                     hover_data: dict = None) -> go.Figure:
    """
    Create an interactive 2D scatter plot with Plotly.

    Args:
        embedding: (N, 2) UMAP embedding
        color_values: (N,) array of values for coloring
        color_labels: (N,) array of string labels for hover
        title: Plot title
        colorscale: Plotly colorscale name
        hover_data: Dictionary of additional hover data

    Returns:
        Plotly figure
    """
    # Create hover text
    hover_text = []
    for i in range(len(embedding)):
        text = f"Episode: {hover_data['episode'][i]}<br>"
        text += f"Timestep: {hover_data['timestep'][i]}<br>"
        text += f"Label: {color_labels[i]}<br>"
        text += f"UMAP X: {embedding[i, 0]:.3f}<br>"
        text += f"UMAP Y: {embedding[i, 1]:.3f}"
        hover_text.append(text)

    fig = go.Figure(data=go.Scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=color_values,
            colorscale=colorscale,
            showscale=True,
            opacity=0.6,
            colorbar=dict(title=title.split('by ')[-1])
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        width=800,
        height=600,
        hovermode='closest'
    )

    return fig


# Prepare hover data
hover_data = {
    'episode': episode_ids,
    'timestep': timestep_ids
}

# Create phase-colored plot
phase_labels_str = [f"Phase {int(p)}" for p in phase_labels]
fig_2d_phase = create_2d_scatter(
    embedding_2d,
    phase_labels,
    phase_labels_str,
    "2D UMAP: Environment Actions colored by Phase",
    colorscale='Turbo',  # Good discrete colorscale for phases
    hover_data=hover_data
)
fig_2d_phase.show()

# Create temporal-colored plot
temporal_labels_str = [TEMPORAL_BIN_LABELS[int(b)] for b in temporal_bins]
fig_2d_temporal = create_2d_scatter(
    embedding_2d,
    temporal_bins,
    temporal_labels_str,
    "2D UMAP: Environment Actions colored by Episode Progress",
    colorscale='Plasma',
    hover_data=hover_data
)
fig_2d_temporal.show()

print("2D visualizations created!")

# %% [markdown]
## 3D Visualizations

# %%
def create_3d_scatter(embedding: np.ndarray,
                     color_values: np.ndarray,
                     color_labels: np.ndarray,
                     title: str,
                     colorscale: str = 'Viridis',
                     hover_data: dict = None) -> go.Figure:
    """
    Create an interactive 3D scatter plot with Plotly.

    Args:
        embedding: (N, 3) UMAP embedding
        color_values: (N,) array of values for coloring
        color_labels: (N,) array of string labels for hover
        title: Plot title
        colorscale: Plotly colorscale name
        hover_data: Dictionary of additional hover data

    Returns:
        Plotly figure
    """
    # Create hover text
    hover_text = []
    for i in range(len(embedding)):
        text = f"Episode: {hover_data['episode'][i]}<br>"
        text += f"Timestep: {hover_data['timestep'][i]}<br>"
        text += f"Label: {color_labels[i]}<br>"
        text += f"UMAP X: {embedding[i, 0]:.3f}<br>"
        text += f"UMAP Y: {embedding[i, 1]:.3f}<br>"
        text += f"UMAP Z: {embedding[i, 2]:.3f}"
        hover_text.append(text)

    fig = go.Figure(data=go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=color_values,
            colorscale=colorscale,
            showscale=True,
            opacity=0.6,
            colorbar=dict(title=title.split('by ')[-1])
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        width=900,
        height=700,
        hovermode='closest'
    )

    return fig


# Create phase-colored 3D plot
fig_3d_phase = create_3d_scatter(
    embedding_3d,
    phase_labels,
    phase_labels_str,
    "3D UMAP: Environment Actions colored by Phase",
    colorscale='Turbo',  # Good discrete colorscale for phases
    hover_data=hover_data
)
fig_3d_phase.show()

# Create temporal-colored 3D plot
fig_3d_temporal = create_3d_scatter(
    embedding_3d,
    temporal_bins,
    temporal_labels_str,
    "3D UMAP: Environment Actions colored by Episode Progress",
    colorscale='Plasma',
    hover_data=hover_data
)
fig_3d_temporal.show()

print("3D visualizations created!")

# %% [markdown]
## Analysis and Interpretation

# %%
print("="*60)
print("VISUALIZATION ANALYSIS")
print("="*60)
print("\nKey Questions to Explore:")
print("1. Do phase-colored points form distinct clusters?")
print("   - This indicates whether phases correspond to distinct action patterns")
print()
print("2. Does temporal coloring show progression?")
print("   - If colors transition smoothly, actions evolve gradually through episodes")
print("   - If colors mix, similar actions occur at different episode times")
print()
print("3. Comparing phase vs temporal coloring:")
print("   - Large divergence suggests phases capture meaningful structure")
print("   - Similar patterns suggest phases align with temporal progression")
print()
print("4. Density and spread:")
print("   - Dense clusters indicate stereotyped behaviors")
print("   - Spread indicates variability in demonstration")
print("="*60)

# %% [markdown]
## Next Steps and Extensions

# %%
print("\nPotential extensions for this notebook:")
print("  - Compare environment vs canonical action spaces")
print("  - Add state (qpos) to action space visualization")
print("  - Compute cluster quality metrics (silhouette score)")
print("  - Visualize phase transition dynamics")
print("  - Per-dimension action distribution analysis")
print("  - UMAP parameter sweep (n_neighbors, min_dist)")
print("  - Interactive selection to examine specific episodes/phases")

# %%
