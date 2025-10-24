# UV Quick Reference for ACT Project

This project has been successfully migrated from conda to UV!

## Installation

### First Time Setup

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and navigate to it
cd /path/to/act

# Install all dependencies (creates .venv and installs everything)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

## Common Commands

### Running Python Scripts

```bash
# Option 1: With activated environment
source .venv/bin/activate
python imitate_episodes.py --task_name sim_transfer_cube_scripted ...

# Option 2: Using uv run (no activation needed)
uv run python imitate_episodes.py --task_name sim_transfer_cube_scripted ...
```

### Managing Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name

# Update all dependencies
uv sync --upgrade

# Install/sync after pulling changes
uv sync
```

### Environment Management

```bash
# Create/update environment
uv sync

# Remove the virtual environment
rm -rf .venv

# Recreate from scratch
uv sync
```

## Verified Configuration

✅ **Python**: 3.10.17 (auto-managed by UV)
✅ **PyTorch**: 2.0.0+cu118 with CUDA support
✅ **NumPy**: 1.26.4 (pinned for compatibility)
✅ **All dependencies**: mujoco, dm_control, opencv-python, matplotlib, etc.

## Key Differences from Conda

| Aspect | Conda | UV |
|--------|-------|-----|
| Speed | Slow | 10-100x faster |
| Environment file | `conda_env.yaml` | `pyproject.toml` + `uv.lock` |
| Activation | `conda activate aloha` | `source .venv/bin/activate` |
| Install packages | `conda install` / `pip install` | `uv add` |
| Run without activation | ❌ | ✅ `uv run python ...` |
| Lockfile | ❌ | ✅ `uv.lock` (committed to git) |

## Advantages of UV

1. **Speed**: Dependency resolution and installation is dramatically faster
2. **Reproducibility**: `uv.lock` ensures exact versions across machines
3. **No separate conda installation**: Uses system Python or downloads automatically
4. **Better for CI/CD**: Faster builds, simpler configuration
5. **Modern tooling**: Built on Rust, actively maintained by Astral

## Troubleshooting

### CUDA not available
- Ensure NVIDIA drivers are installed on your system
- Check with: `uv run python -c "import torch; print(torch.cuda.is_available())"`

### Import errors
- Run `uv sync` to ensure all dependencies are installed
- Check Python version: `uv run python --version`

### Need to use conda temporarily
- The original `conda_env.yaml` is still in the repository for reference
- You can use both environments simultaneously (different names)

## Migration Notes

- **PyTorch**: Uses PyTorch's official pip index with CUDA 11.8 support
- **NumPy**: Pinned to <2.0 for compatibility with opencv-python and dm-control
- **detr package**: Configured as editable workspace member
- All dependencies from `conda_env.yaml` have been converted to pip equivalents

## Need Help?

- UV Documentation: https://docs.astral.sh/uv/
- Report issues: https://github.com/astral-sh/uv/issues
- For ACT-specific issues, see the main README.md
