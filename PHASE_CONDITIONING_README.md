# Phase Conditioning for ACT - Implementation Complete

This document describes the phase conditioning implementation that has been added to the ACT (Action Chunking Transformer) baseline.

## Overview

Phase conditioning enhances ACT by:
1. **Phase-Aware Training**: Conditioning the action encoder/decoder on learned phase embeddings
2. **Canonical Action Space**: Training on canonical (phase-normalized) actions for narrower distributions
3. **Online Reconstruction**: Converting canonical predictions back to environment actions during inference

## Key Benefits

- **Narrower Action Distributions**: Phase conditioning focuses the policy on phase-specific action subspaces
- **Better Behavioral Consistency**: Phase awareness helps maintain consistent behaviors within phases
- **Improved Constraint Satisfaction**: Phase information aids in satisfying spatial and temporal constraints
- **Enhanced Generalization**: Phase-based abstraction may improve transfer to unseen scenarios
- **Full Backward Compatibility**: All changes are optional - existing code works unchanged

## Files Modified/Created

### New Files
1. **[preprocess_canonical_actions.py](preprocess_canonical_actions.py)** - Preprocessing script to compute and save canonical actions to HDF5
2. **[phase_wrapped_policy.py](phase_wrapped_policy.py)** - Inference wrapper for canonical action reconstruction
3. **PHASE_CONDITIONING_README.md** (this file) - Documentation

### Modified Files
1. **[detr/models/detr_vae.py](detr/models/detr_vae.py)** - Added phase conditioning to CVAE encoder/decoder
2. **[detr/models/transformer.py](detr/models/transformer.py)** - Extended transformer to handle phase input tokens
3. **[policy.py](policy.py)** - Updated ACTPolicy to pass through phase_ids
4. **[utils.py](utils.py)** - Augmented EpisodicDataset to load canonical actions and phase labels
5. **[imitate_episodes.py](imitate_episodes.py)** - Added phase conditioning arguments and training/eval logic

## Architecture Changes

### 1. Phase Embeddings in DETR VAE

**Location**: `detr/models/detr_vae.py`

Added optional phase conditioning modules:
```python
if num_phases is not None:
    self.phase_emb = nn.Embedding(num_phases, phase_embed_dim)  # Learnable phase embeddings
    self.phase_proj = nn.Linear(phase_embed_dim, hidden_dim)   # Project to model dimension
```

**Encoder Conditioning** (Line 104-109):
- Phase embeddings are added to action embeddings before encoding
- Helps the CVAE learn phase-specific latent representations

**Decoder Conditioning** (Line 151-160):
- Phase embeddings are passed as additional input tokens to the transformer decoder
- Concatenated with latent and proprioception features
- Enables phase-aware action prediction

### 2. Dataset Augmentation

**Location**: `utils.py`

The `EpisodicDataset` now supports loading:
- **Canonical actions**: Phase-normalized actions stored in `/canonical_actions`
- **Phase labels**: Phase IDs for each timestep stored in `/phase_labels`

When `use_canonical=True`:
- Returns 6-tuple: `(image, qpos, action, is_pad, canonical_action, phase_id)`
- Canonical actions are used for training
- Phase IDs condition the encoder/decoder

### 3. Training Pipeline

**Location**: `imitate_episodes.py`

New command-line arguments:
```bash
--use_canonical          # Enable phase conditioning
--num_phases N           # Number of behavioral phases
--phase_embed_dim D      # Phase embedding dimension (default: 64)
--phase_result_path P    # Path to phase detection results
--wrap_with_phase        # Enable phase wrapping for evaluation
```

Training automatically:
- Loads canonical actions from preprocessed HDF5 files
- Trains on canonical actions with phase conditioning
- Saves checkpoints compatible with phase-wrapped evaluation

### 4. Evaluation Pipeline

**Location**: `imitate_episodes.py`, `phase_wrapped_policy.py`

When `--wrap_with_phase` is enabled:
1. Loads trained checkpoint
2. Wraps with `PhaseWrappedACT`
3. During rollouts:
   - Predicts current phase from observations
   - Gets canonical actions from base policy
   - Converts canonical → environment actions using PhaseProjector

## Usage Workflow

### Step 1: Run Phase Detection

First, detect phases in your demonstration data:

```bash
python run_phase_detection.py \
    --dataset-dir dataset_dir \
    --output-path phase_results/K3.npz \
    --num-phases 3 \
    --n-iter 25 \
    --p-stay 0.75
```

This creates:
- `phase_results/K3.npz` - Padded buffers for policies
- `phase_results/K3_labels.pkl` - Phase labels per trajectory
- `phase_results/K3_stats.pkl` - Phase statistics (means, covariances)
- `phase_results/K3_hmm_bundle.pkl` - HMM parameters for online tracking
- `phase_results/K3_config.json` - Phase detection configuration

### Step 2: Preprocess Demonstrations

Compute and save canonical actions to HDF5 files:

```bash
python preprocess_canonical_actions.py \
    --dataset_dir dataset_dir \
    --phase_result_path phase_results/K3.npz \
    --num_episodes 50
```

This adds to each `episode_N.hdf5`:
- `/canonical_actions` - Phase-normalized actions (T, 14)
- `/phase_labels` - Phase IDs for each timestep (T,)

### Step 3: Train with Phase Conditioning

Train ACT with phase conditioning:

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ckpt_phase \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --use_canonical \
    --num_phases 3 \
    --phase_result_path phase_results/K3.npz
```

### Step 4: Evaluate with Phase Wrapping

Evaluate with canonical action reconstruction:

**IMPORTANT**: You must include the same `--use_canonical` and `--num_phases` arguments used during training to properly load the checkpoint!

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ckpt_phase \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --use_canonical \
    --num_phases 3 \
    --eval \
    --wrap_with_phase \
    --phase_result_path phase_results/K3.npz
```

### Baseline ACT (No Phase Conditioning)

For comparison, train standard ACT without any changes:

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ckpt_baseline \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200
```

Evaluate baseline:

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ckpt_baseline \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --eval
```

## Implementation Details

### Canonical Action Transformation

**Forward** (Training): Environment → Canonical
```python
canonical = L_k^{-1} @ (action - μ_k)
```

**Inverse** (Inference): Canonical → Environment
```python
action = μ_k + L_k @ canonical
```

Where:
- `μ_k`: Mean action for phase k
- `L_k`: Cholesky factor of covariance Σ_k
- Ensures stable, invertible transformations

### Phase Prediction During Inference

The `OnlinePhaseDecoder` uses a forward algorithm to track phase probabilities:
1. Maintains belief state: `P(phase_t = k | obs_1:t)`
2. Updates with each new observation
3. Predicts most likely current phase
4. Used to condition the policy and reconstruct actions

### Normalization

**Important**: Canonical actions are normalized using the **same** statistics as environment actions:
```python
canonical_action_norm = (canonical_action - action_mean) / action_std
```

This ensures:
- Consistent scale across training
- Easier learning for the neural network
- Actions are denormalized before phase projection during inference

## Backward Compatibility

All modifications maintain full backward compatibility:

✅ **Default behavior unchanged**: Without `--use_canonical`, ACT works exactly as before
✅ **Optional parameters**: All new arguments have sensible defaults
✅ **Graceful degradation**: Missing phase data raises clear errors
✅ **Checkpoint compatibility**: Baseline checkpoints load without modification

## Testing & Validation

### Sanity Checks

1. **Baseline Unchanged**:
   ```bash
   # Train without phase conditioning
   python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir test_baseline ...
   # Should produce identical results to original ACT
   ```

2. **Phase Preprocessing**:
   ```bash
   # Verify canonical actions are added
   python -c "import h5py; f = h5py.File('data/.../episode_0.hdf5', 'r'); print(list(f.keys()))"
   # Should show /canonical_actions and /phase_labels
   ```

3. **Training with Phases**:
   ```bash
   # Train with --use_canonical
   # Monitor loss curves - should converge normally
   ```

4. **Evaluation**:
   ```bash
   # Evaluate with --wrap_with_phase
   # Should successfully complete rollouts without errors
   ```

### Expected Improvements

Phase conditioning may improve:
- Success rates on multi-stage tasks
- Consistency across episodes
- Generalization to unseen scenarios
- Sample efficiency during training

**Note**: Improvements depend on:
- Quality of phase detection
- Number of phases chosen
- Task structure and complexity
- Amount of training data

## Troubleshooting

### Error: "Unexpected key(s) in state_dict: model.phase_emb.weight..." or "size mismatch for model.additional_pos_embed.weight"

**Cause**: The checkpoint was trained with phase conditioning, but you're trying to load it without specifying the phase parameters.

**Solution**: Add the same phase conditioning arguments used during training:
```bash
python imitate_episodes.py ... --eval \
    --use_canonical \
    --num_phases 3 \
    --wrap_with_phase \
    --phase_result_path phase_results/K3.npz
```

The model architecture must match between training and evaluation!

### Error: "use_canonical=True but episode X does not have canonical_actions"

**Solution**: Run preprocessing script first:
```bash
python preprocess_canonical_actions.py --dataset_dir <dir> --phase_result_path <path> --num_episodes <N>
```

### Error: "--num_phases must be specified when using --use_canonical"

**Solution**: Add `--num_phases N` argument (must match phase detection K):
```bash
python imitate_episodes.py ... --use_canonical --num_phases 3
```

### Error: "--phase_result_path must be specified when using --wrap_with_phase"

**Solution**: Provide path to phase detection results during evaluation:
```bash
python imitate_episodes.py ... --eval --wrap_with_phase --phase_result_path phase_results/xxx.npz
```

### Warning: Phase mismatch during inference

Check that:
1. Phase detection was run on the same task
2. Number of phases matches training (`--num_phases`)
3. Phase result file path is correct

## Future Enhancements

Potential improvements to explore:

1. **Advanced Conditioning**:
   - Cross-attention between phase and observation features
   - FiLM-style modulation of transformer layers
   - Phase-specific action heads

2. **Dynamic Phase Models**:
   - Online phase model adaptation
   - Hierarchical phase structures
   - Continuous phase representations

3. **Multi-Task Phases**:
   - Shared phase models across tasks
   - Transfer learning via phase embeddings

4. **Phase Dropout** (Classifier-Free Guidance):
   - Randomly drop phase conditioning during training
   - Improves robustness to phase prediction errors

## References

This implementation is based on the phase conditioning approach described in your documentation, adapted for the ACT architecture.

## Questions?

For issues or questions:
1. Check this README first
2. Verify preprocessing steps were completed
3. Check terminal output for specific error messages
4. Ensure backward compatibility by testing without `--use_canonical` first

Happy experimenting with phase-conditioned ACT! 🚀
