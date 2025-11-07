#!/usr/bin/env bash
# Run phase detection on demonstration trajectories
# This script must be run before training with phase conditioning

set -e  # Exit on error

# Configuration
ENV_ID="SyringeInjection-v1"
DEMO_PATH="./demos/SyringeInjection-v1/motion_planning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
OUTPUT_DIR="./phase_results"
NUM_PHASES=3
NUM_DEMOS=300

# Phase detection hyperparameters
N_ITER=25
P_STAY=0.75
VAR_KEEP=0.95
RANDOM_STATE=42

# Derived paths
OUTPUT_PATH="${OUTPUT_DIR}/${ENV_ID}-phase-detection.npz"

echo "========================================"
echo "Phase Detection for ${ENV_ID}"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Demo path:      ${DEMO_PATH}"
echo "  Output path:    ${OUTPUT_PATH}"
echo "  Num phases:     ${NUM_PHASES}"
echo "  Num demos:      ${NUM_DEMOS}"
echo "  HMM iterations: ${N_ITER}"
echo "  P(stay):        ${P_STAY}"
echo "  PCA var keep:   ${VAR_KEEP}"
echo ""

# Check if demo file exists
if [ ! -f "${DEMO_PATH}" ]; then
    echo "Error: Demo file not found at ${DEMO_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run phase detection
echo "Running phase detection..."
echo ""

uv run python ai_syringe_injection/baselines/act_phase/run_phase_detection.py \
    --demo-path "${DEMO_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --num-phases ${NUM_PHASES} \
    --num-demos ${NUM_DEMOS} \
    --n-iter ${N_ITER} \
    --p-stay ${P_STAY} \
    --var-keep ${VAR_KEEP} \
    --random-state ${RANDOM_STATE} \
    --use-actions

echo ""
echo "========================================"
echo "Phase detection complete!"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh "${OUTPUT_PATH}"*
echo ""
echo "Next steps:"
echo "  1. Review the phase statistics above"
echo "  2. Run training with phase conditioning:"
echo "     ./ai_syringe_injection/baselines/act_phase/train_rgb_act.sh"
echo ""
