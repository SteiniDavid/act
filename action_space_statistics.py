# %% [markdown]
# # Statistical Analysis of Demonstration Action Space
#
# This notebook performs statistical analysis on the action space using:
# 1. Linear Discriminant Analysis (LDA) for phase classification
# 2. Maximum Mean Discrepancy (MMD) for distribution comparison
# 3. Maximum Likelihood Estimation (MLE) for intrinsic dimensionality
#
# All visualizations use Plotly for interactive exploration.

# %% [markdown]
## Setup and Imports

# %%
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully")

# %% [markdown]
## Configuration

# %%
# Dataset configuration
DATASET_DIR = Path('/home/steini/Documents/git/act/dataset_dir')
NUM_EPISODES = 50
EPISODE_LEN = 400  # Expected episode length

# Temporal binning (5 bins for temporal progression)
NUM_TEMPORAL_BINS = 5
TEMPORAL_BIN_LABELS = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

# LDA configuration
LDA_CV_FOLDS = 5
RANDOM_STATE = 42

# MMD configuration
MMD_KERNEL_BANDWIDTH = 1.0  # Standard bandwidth for RBF kernel

# Intrinsic dimensionality configuration
ID_K_NEIGHBORS = [10, 20, 30, 50]  # Different k values to try

print(f"Dataset directory: {DATASET_DIR}")
print(f"Number of episodes: {NUM_EPISODES}")
print(f"LDA CV folds: {LDA_CV_FOLDS}")
print(f"MMD kernel bandwidth: {MMD_KERNEL_BANDWIDTH}")
print(f"Intrinsic dim k-neighbors: {ID_K_NEIGHBORS}")

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


# Load the data
actions, phase_labels, episode_ids, timestep_ids = load_demonstration_data(DATASET_DIR, NUM_EPISODES)

# Create temporal bins
temporal_bins = create_temporal_bins(timestep_ids, EPISODE_LEN, NUM_TEMPORAL_BINS)

print("\nTemporal binning complete!")
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
## 1. Linear Discriminant Analysis (LDA)

# %% [markdown]
### 1.1 LDA with Cross-Validation

# %%
print("="*60)
print("LINEAR DISCRIMINANT ANALYSIS")
print("="*60)

# Create LDA model
lda = LinearDiscriminantAnalysis()

# Perform stratified k-fold cross-validation
print(f"\nPerforming {LDA_CV_FOLDS}-fold cross-validation...")
cv = StratifiedKFold(n_splits=LDA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(lda, actions, phase_labels, cv=cv, scoring='accuracy')

print(f"\nCross-Validation Results:")
print(f"  Fold accuracies: {cv_scores}")
print(f"  Mean accuracy: {cv_scores.mean():.4f}")
print(f"  Std accuracy: {cv_scores.std():.4f}")

# Fit on full dataset for visualization
lda.fit(actions, phase_labels)
predictions = lda.predict(actions)
full_accuracy = accuracy_score(phase_labels, predictions)

print(f"\nFull dataset accuracy: {full_accuracy:.4f}")

# Get class names
unique_phases_sorted = np.sort(np.unique(phase_labels))
class_names = [f"Phase {int(p)}" for p in unique_phases_sorted]

print(f"\nClassification Report:")
print(classification_report(phase_labels, predictions, target_names=class_names))

# %% [markdown]
### 1.2 Confusion Matrix Visualization

# %%
# Compute confusion matrix
cm = confusion_matrix(phase_labels, predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create confusion matrix heatmap
fig_cm = go.Figure(data=go.Heatmap(
    z=cm_normalized,
    x=class_names,
    y=class_names,
    colorscale='Blues',
    text=cm,
    texttemplate='%{text}',
    textfont={"size": 12},
    colorbar=dict(title="Normalized<br>Accuracy"),
    hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Normalized: %{z:.3f}<extra></extra>'
))

fig_cm.update_layout(
    title=f'LDA Confusion Matrix (Accuracy: {full_accuracy:.4f})',
    xaxis_title='Predicted Phase',
    yaxis_title='True Phase',
    width=700,
    height=600,
    yaxis=dict(autorange='reversed')
)

fig_cm.show()

print("Confusion matrix visualization created!")

# %% [markdown]
### 1.3 LDA Projection Visualizations

# %%
# Transform data using LDA (this gives us projections onto discriminant axes)
# LDA can produce at most min(n_features, n_classes - 1) components
n_components_lda = min(actions.shape[1], len(unique_phases_sorted) - 1)
print(f"\nLDA can produce up to {n_components_lda} discriminant components")

# 2D projection
lda_2d = LinearDiscriminantAnalysis(n_components=min(2, n_components_lda))
lda_2d.fit(actions, phase_labels)
embedding_2d = lda_2d.transform(actions)

print(f"  2D embedding shape: {embedding_2d.shape}")
print(f"  Explained variance ratio (2D): {lda_2d.explained_variance_ratio_}")

# Create 2D scatter plot
phase_labels_str = [f"Phase {int(p)}" for p in phase_labels]

# Create hover text
hover_text_2d = []
for i in range(len(embedding_2d)):
    text = f"Episode: {episode_ids[i]}<br>"
    text += f"Timestep: {timestep_ids[i]}<br>"
    text += f"True Phase: Phase {int(phase_labels[i])}<br>"
    text += f"Predicted: Phase {int(predictions[i])}<br>"
    text += f"Correct: {'Yes' if phase_labels[i] == predictions[i] else 'No'}<br>"
    if embedding_2d.shape[1] >= 2:
        text += f"LD1: {embedding_2d[i, 0]:.3f}<br>"
        text += f"LD2: {embedding_2d[i, 1]:.3f}"
    else:
        text += f"LD1: {embedding_2d[i, 0]:.3f}"
    hover_text_2d.append(text)

if embedding_2d.shape[1] >= 2:
    fig_lda_2d = go.Figure(data=go.Scatter(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=phase_labels,
            colorscale='Turbo',
            showscale=True,
            opacity=0.6,
            colorbar=dict(title="Phase")
        ),
        text=hover_text_2d,
        hovertemplate='%{text}<extra></extra>'
    ))

    fig_lda_2d.update_layout(
        title=f'2D LDA Projection (Explained variance: {lda_2d.explained_variance_ratio_.sum():.3f})',
        xaxis_title=f'LD1 ({lda_2d.explained_variance_ratio_[0]:.3f})',
        yaxis_title=f'LD2 ({lda_2d.explained_variance_ratio_[1]:.3f})' if len(lda_2d.explained_variance_ratio_) > 1 else 'LD2',
        width=900,
        height=700,
        hovermode='closest'
    )

    fig_lda_2d.show()
    print("2D LDA visualization created!")
else:
    print("Only 1 discriminant component available (2 classes), skipping 2D plot")

# %% [markdown]
### 1.4 3D LDA Projection

# %%
if n_components_lda >= 3:
    # 3D projection
    lda_3d = LinearDiscriminantAnalysis(n_components=3)
    lda_3d.fit(actions, phase_labels)
    embedding_3d = lda_3d.transform(actions)

    print(f"\n  3D embedding shape: {embedding_3d.shape}")
    print(f"  Explained variance ratio (3D): {lda_3d.explained_variance_ratio_}")

    # Create hover text
    hover_text_3d = []
    for i in range(len(embedding_3d)):
        text = f"Episode: {episode_ids[i]}<br>"
        text += f"Timestep: {timestep_ids[i]}<br>"
        text += f"True Phase: Phase {int(phase_labels[i])}<br>"
        text += f"Predicted: Phase {int(predictions[i])}<br>"
        text += f"Correct: {'Yes' if phase_labels[i] == predictions[i] else 'No'}<br>"
        text += f"LD1: {embedding_3d[i, 0]:.3f}<br>"
        text += f"LD2: {embedding_3d[i, 1]:.3f}<br>"
        text += f"LD3: {embedding_3d[i, 2]:.3f}"
        hover_text_3d.append(text)

    fig_lda_3d = go.Figure(data=go.Scatter3d(
        x=embedding_3d[:, 0],
        y=embedding_3d[:, 1],
        z=embedding_3d[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=phase_labels,
            colorscale='Turbo',
            showscale=True,
            opacity=0.6,
            colorbar=dict(title="Phase")
        ),
        text=hover_text_3d,
        hovertemplate='%{text}<extra></extra>'
    ))

    fig_lda_3d.update_layout(
        title=f'3D LDA Projection (Explained variance: {lda_3d.explained_variance_ratio_.sum():.3f})',
        scene=dict(
            xaxis_title=f'LD1 ({lda_3d.explained_variance_ratio_[0]:.3f})',
            yaxis_title=f'LD2 ({lda_3d.explained_variance_ratio_[1]:.3f})',
            zaxis_title=f'LD3 ({lda_3d.explained_variance_ratio_[2]:.3f})'
        ),
        width=900,
        height=700,
        hovermode='closest'
    )

    fig_lda_3d.show()
    print("3D LDA visualization created!")
else:
    print(f"\nOnly {n_components_lda} discriminant components available, skipping 3D projection")

# %% [markdown]
## 2. Maximum Mean Discrepancy (MMD)

# %% [markdown]
### 2.1 MMD Implementation

# %%
def rbf_kernel(X, Y, bandwidth=1.0):
    """
    Compute RBF (Gaussian) kernel between samples in X and Y.

    Args:
        X: (n, d) array
        Y: (m, d) array
        bandwidth: kernel bandwidth (sigma)

    Returns:
        K: (n, m) kernel matrix
    """
    # Compute pairwise squared Euclidean distances
    sq_dists = cdist(X, Y, metric='sqeuclidean')

    # Apply RBF kernel
    K = np.exp(-sq_dists / (2 * bandwidth ** 2))

    return K


def compute_mmd(X, Y, bandwidth=1.0):
    """
    Compute Maximum Mean Discrepancy between two samples using RBF kernel.

    Args:
        X: (n, d) array - first sample
        Y: (m, d) array - second sample
        bandwidth: kernel bandwidth

    Returns:
        mmd: Maximum Mean Discrepancy value
    """
    n = len(X)
    m = len(Y)

    # Compute kernel matrices
    K_XX = rbf_kernel(X, X, bandwidth)
    K_YY = rbf_kernel(Y, Y, bandwidth)
    K_XY = rbf_kernel(X, Y, bandwidth)

    # MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    mmd_sq = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
    mmd_sq += (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    mmd_sq -= 2 * K_XY.sum() / (n * m)

    # Return MMD (take sqrt of MMD^2)
    return np.sqrt(max(0, mmd_sq))  # max with 0 to handle numerical errors


print("="*60)
print("MAXIMUM MEAN DISCREPANCY (MMD)")
print("="*60)

# %% [markdown]
### 2.2 MMD Between Phases

# %%
print("\nComputing pairwise MMD between phases...")

unique_phases_sorted = np.sort(np.unique(phase_labels))
n_phases = len(unique_phases_sorted)

# Create MMD matrix for phases
mmd_phase_matrix = np.zeros((n_phases, n_phases))

for i, phase_i in enumerate(unique_phases_sorted):
    for j, phase_j in enumerate(unique_phases_sorted):
        if i <= j:  # Compute upper triangle (MMD is symmetric)
            mask_i = phase_labels == phase_i
            mask_j = phase_labels == phase_j

            X_i = actions[mask_i]
            X_j = actions[mask_j]

            mmd_val = compute_mmd(X_i, X_j, bandwidth=MMD_KERNEL_BANDWIDTH)
            mmd_phase_matrix[i, j] = mmd_val
            mmd_phase_matrix[j, i] = mmd_val  # Symmetric

            print(f"  MMD(Phase {int(phase_i)}, Phase {int(phase_j)}): {mmd_val:.6f}")

print("\nPhase MMD matrix computed!")

# Create heatmap for phase MMD
phase_labels_viz = [f"Phase {int(p)}" for p in unique_phases_sorted]

fig_mmd_phase = go.Figure(data=go.Heatmap(
    z=mmd_phase_matrix,
    x=phase_labels_viz,
    y=phase_labels_viz,
    colorscale='Viridis',
    text=np.round(mmd_phase_matrix, 4),
    texttemplate='%{text}',
    textfont={"size": 12},
    colorbar=dict(title="MMD"),
    hovertemplate='Phase %{y} vs Phase %{x}<br>MMD: %{z:.6f}<extra></extra>'
))

fig_mmd_phase.update_layout(
    title=f'MMD Between Phases (bandwidth={MMD_KERNEL_BANDWIDTH})',
    xaxis_title='Phase',
    yaxis_title='Phase',
    width=700,
    height=600,
    yaxis=dict(autorange='reversed')
)

fig_mmd_phase.show()

print("Phase MMD heatmap created!")

# %% [markdown]
### 2.3 MMD Between Temporal Bins

# %%
print("\nComputing pairwise MMD between temporal bins...")

n_bins = NUM_TEMPORAL_BINS

# Create MMD matrix for temporal bins
mmd_temporal_matrix = np.zeros((n_bins, n_bins))

for i in range(n_bins):
    for j in range(n_bins):
        if i <= j:  # Compute upper triangle (MMD is symmetric)
            mask_i = temporal_bins == i
            mask_j = temporal_bins == j

            X_i = actions[mask_i]
            X_j = actions[mask_j]

            mmd_val = compute_mmd(X_i, X_j, bandwidth=MMD_KERNEL_BANDWIDTH)
            mmd_temporal_matrix[i, j] = mmd_val
            mmd_temporal_matrix[j, i] = mmd_val  # Symmetric

            print(f"  MMD(Bin {i} ({TEMPORAL_BIN_LABELS[i]}), Bin {j} ({TEMPORAL_BIN_LABELS[j]})): {mmd_val:.6f}")

print("\nTemporal bin MMD matrix computed!")

# Create heatmap for temporal bin MMD
fig_mmd_temporal = go.Figure(data=go.Heatmap(
    z=mmd_temporal_matrix,
    x=TEMPORAL_BIN_LABELS,
    y=TEMPORAL_BIN_LABELS,
    colorscale='Viridis',
    text=np.round(mmd_temporal_matrix, 4),
    texttemplate='%{text}',
    textfont={"size": 12},
    colorbar=dict(title="MMD"),
    hovertemplate='%{y} vs %{x}<br>MMD: %{z:.6f}<extra></extra>'
))

fig_mmd_temporal.update_layout(
    title=f'MMD Between Temporal Bins (bandwidth={MMD_KERNEL_BANDWIDTH})',
    xaxis_title='Temporal Bin',
    yaxis_title='Temporal Bin',
    width=700,
    height=600,
    yaxis=dict(autorange='reversed')
)

fig_mmd_temporal.show()

print("Temporal bin MMD heatmap created!")

# %% [markdown]
### 2.4 MMD Analysis Summary

# %%
print("\n" + "="*60)
print("MMD ANALYSIS SUMMARY")
print("="*60)

print("\nPhase MMD Statistics:")
# Get off-diagonal elements (actual divergences between different phases)
phase_off_diag = mmd_phase_matrix[np.triu_indices(n_phases, k=1)]
print(f"  Mean MMD between different phases: {phase_off_diag.mean():.6f}")
print(f"  Max MMD between different phases: {phase_off_diag.max():.6f}")
print(f"  Min MMD between different phases: {phase_off_diag.min():.6f}")

# Find most different phases
max_idx = np.unravel_index(mmd_phase_matrix.argmax(), mmd_phase_matrix.shape)
if max_idx[0] != max_idx[1]:  # Exclude diagonal
    print(f"  Most different phases: Phase {int(unique_phases_sorted[max_idx[0]])} vs Phase {int(unique_phases_sorted[max_idx[1]])} (MMD={mmd_phase_matrix[max_idx]:.6f})")

print("\nTemporal Bin MMD Statistics:")
# Get off-diagonal elements
temporal_off_diag = mmd_temporal_matrix[np.triu_indices(n_bins, k=1)]
print(f"  Mean MMD between different bins: {temporal_off_diag.mean():.6f}")
print(f"  Max MMD between different bins: {temporal_off_diag.max():.6f}")
print(f"  Min MMD between different bins: {temporal_off_diag.min():.6f}")

# Check if MMD increases with temporal distance
print("\n  MMD vs temporal distance:")
for dist in range(1, n_bins):
    indices = [(i, i+dist) for i in range(n_bins-dist)]
    mmds_at_dist = [mmd_temporal_matrix[i, j] for i, j in indices]
    print(f"    Distance {dist}: mean MMD = {np.mean(mmds_at_dist):.6f}")

print("="*60)

# %% [markdown]
## 3. Intrinsic Dimensionality Estimation (MLE)

# %% [markdown]
### 3.1 Levina & Bickel MLE Implementation

# %%
def estimate_intrinsic_dimension_mle(X, k=20):
    """
    Estimate intrinsic dimensionality using MLE method by Levina & Bickel (2004).

    Based on: "Maximum Likelihood Estimation of Intrinsic Dimension"

    Args:
        X: (n, d) data matrix
        k: number of nearest neighbors to use

    Returns:
        dimension: estimated intrinsic dimension
    """
    n, d = X.shape

    # Compute pairwise distances
    dists = squareform(pdist(X, metric='euclidean'))

    # For each point, find k+1 nearest neighbors (includes the point itself)
    # Sort distances for each point
    sorted_dists = np.sort(dists, axis=1)

    # Take distances to k nearest neighbors (excluding distance to self which is 0)
    # sorted_dists[:, 1] is the nearest neighbor, sorted_dists[:, k] is the k-th nearest
    neighbor_dists = sorted_dists[:, 1:k+1]

    # MLE estimator: m_k(x_i) = [1/k * sum_{j=1}^{k} log(T_k(x_i) / T_j(x_i))]^{-1}
    # where T_j is the distance to j-th nearest neighbor

    # For numerical stability, we'll use the formula:
    # m_k = (k-1) / sum_{j=1}^{k-1} log(T_k / T_j)

    dimensions = []

    for i in range(n):
        # Get distances to k nearest neighbors for point i
        r = neighbor_dists[i]

        # The k-th nearest neighbor distance
        T_k = r[-1]

        # Distances to first k-1 neighbors
        T_j = r[:-1]

        # Avoid log(0) or division by very small numbers
        if T_k > 1e-10 and np.all(T_j > 1e-10):
            # Compute log ratios
            log_ratios = np.log(T_k / T_j)
            sum_log_ratios = np.sum(log_ratios)

            # MLE estimate for this point
            if sum_log_ratios > 1e-10:
                m_k = (k - 1) / sum_log_ratios
                dimensions.append(m_k)

    # Return mean estimate across all points
    if len(dimensions) > 0:
        return np.mean(dimensions)
    else:
        return np.nan


print("="*60)
print("INTRINSIC DIMENSIONALITY ESTIMATION (MLE)")
print("="*60)

# %% [markdown]
### 3.2 Global Intrinsic Dimensionality

# %%
print("\nEstimating global intrinsic dimensionality...")
print(f"Original feature dimension: {actions.shape[1]}")

global_dimensions = {}

for k in ID_K_NEIGHBORS:
    dim_estimate = estimate_intrinsic_dimension_mle(actions, k=k)
    global_dimensions[k] = dim_estimate
    print(f"  k={k:2d}: Estimated intrinsic dimension = {dim_estimate:.2f}")

# Visualize global dimensionality estimates
fig_global_id = go.Figure()

fig_global_id.add_trace(go.Scatter(
    x=list(global_dimensions.keys()),
    y=list(global_dimensions.values()),
    mode='lines+markers',
    marker=dict(size=10, color='blue'),
    line=dict(width=2),
    name='Estimated Dimension',
    hovertemplate='k=%{x}<br>Dimension: %{y:.2f}<extra></extra>'
))

# Add horizontal line for original dimension
fig_global_id.add_hline(
    y=actions.shape[1],
    line_dash="dash",
    line_color="red",
    annotation_text=f"Original dim ({actions.shape[1]})",
    annotation_position="right"
)

fig_global_id.update_layout(
    title='Global Intrinsic Dimensionality vs k-neighbors',
    xaxis_title='Number of Neighbors (k)',
    yaxis_title='Estimated Intrinsic Dimension',
    width=800,
    height=500,
    showlegend=True
)

fig_global_id.show()

print("\nGlobal intrinsic dimensionality visualization created!")

# %% [markdown]
### 3.3 Per-Phase Intrinsic Dimensionality

# %%
print("\nEstimating per-phase intrinsic dimensionality...")

unique_phases_sorted = np.sort(np.unique(phase_labels))
phase_dimensions = {int(phase): {} for phase in unique_phases_sorted}

for phase in unique_phases_sorted:
    phase_int = int(phase)
    mask = phase_labels == phase
    phase_actions = actions[mask]

    print(f"\nPhase {phase_int} (n={len(phase_actions)} samples):")

    for k in ID_K_NEIGHBORS:
        # Make sure we have enough samples for k neighbors
        if len(phase_actions) > k:
            dim_estimate = estimate_intrinsic_dimension_mle(phase_actions, k=k)
            phase_dimensions[phase_int][k] = dim_estimate
            print(f"  k={k:2d}: Estimated intrinsic dimension = {dim_estimate:.2f}")
        else:
            print(f"  k={k:2d}: Not enough samples (need > {k})")
            phase_dimensions[phase_int][k] = np.nan

# Visualize per-phase dimensionality estimates
fig_phase_id = go.Figure()

colors = px.colors.qualitative.Plotly

for idx, phase_int in enumerate(sorted(phase_dimensions.keys())):
    k_values = list(phase_dimensions[phase_int].keys())
    dim_values = list(phase_dimensions[phase_int].values())

    fig_phase_id.add_trace(go.Scatter(
        x=k_values,
        y=dim_values,
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=2),
        name=f'Phase {phase_int}',
        hovertemplate=f'Phase {phase_int}<br>k=%{{x}}<br>Dimension: %{{y:.2f}}<extra></extra>'
    ))

# Add horizontal line for original dimension
fig_phase_id.add_hline(
    y=actions.shape[1],
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Original dim ({actions.shape[1]})",
    annotation_position="right"
)

fig_phase_id.update_layout(
    title='Per-Phase Intrinsic Dimensionality vs k-neighbors',
    xaxis_title='Number of Neighbors (k)',
    yaxis_title='Estimated Intrinsic Dimension',
    width=900,
    height=600,
    showlegend=True,
    hovermode='closest'
)

fig_phase_id.show()

print("\nPer-phase intrinsic dimensionality visualization created!")

# %% [markdown]
### 3.4 Intrinsic Dimensionality Summary

# %%
print("\n" + "="*60)
print("INTRINSIC DIMENSIONALITY SUMMARY")
print("="*60)

print(f"\nOriginal feature dimension: {actions.shape[1]}")

print(f"\nGlobal estimates (across all data):")
for k, dim in global_dimensions.items():
    print(f"  k={k:2d}: {dim:.2f}")

mean_global = np.mean(list(global_dimensions.values()))
print(f"  Mean across k values: {mean_global:.2f}")

print(f"\nPer-phase estimates (mean across k values):")
for phase_int in sorted(phase_dimensions.keys()):
    valid_dims = [d for d in phase_dimensions[phase_int].values() if not np.isnan(d)]
    if valid_dims:
        mean_dim = np.mean(valid_dims)
        print(f"  Phase {phase_int}: {mean_dim:.2f}")

print("\nInterpretation:")
print(f"  - If estimated dimension << original dimension ({actions.shape[1]}),")
print(f"    the data lies on a lower-dimensional manifold")
print(f"  - If estimated dimension ≈ original dimension,")
print(f"    the data fills the full ambient space")
print(f"  - Different estimates across phases suggest")
print(f"    phases have different complexity/structure")

print("="*60)

# %% [markdown]
## Summary and Conclusions

# %%
print("\n" + "="*60)
print("STATISTICAL ANALYSIS COMPLETE")
print("="*60)

print("\nKey Findings:")

print("\n1. LINEAR DISCRIMINANT ANALYSIS:")
print(f"   - Cross-validated accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   - Full dataset accuracy: {full_accuracy:.4f}")
print(f"   - Number of discriminant components: {n_components_lda}")
print("   - Phases are" + (" well" if cv_scores.mean() > 0.8 else " moderately" if cv_scores.mean() > 0.6 else " poorly") + " separated in action space")

print("\n2. MAXIMUM MEAN DISCREPANCY:")
print(f"   - Mean MMD between phases: {phase_off_diag.mean():.6f}")
print(f"   - Mean MMD between temporal bins: {temporal_off_diag.mean():.6f}")
if phase_off_diag.mean() > temporal_off_diag.mean():
    print("   - Phases show stronger distributional differences than temporal progression")
else:
    print("   - Temporal progression shows stronger distributional differences than phases")

print("\n3. INTRINSIC DIMENSIONALITY:")
print(f"   - Original dimension: {actions.shape[1]}")
print(f"   - Global estimated dimension: {mean_global:.2f}")
compression_ratio = mean_global / actions.shape[1]
print(f"   - Compression ratio: {compression_ratio:.2%}")
print("   - Data" + (" significantly" if compression_ratio < 0.5 else " moderately" if compression_ratio < 0.8 else " minimally") + " lies on lower-dimensional manifold")

print("\n" + "="*60)

# %%
