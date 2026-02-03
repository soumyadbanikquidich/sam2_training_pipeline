#!/bin/bash
# ============================================================================
# SAM2 Training Pipeline - Docker Launcher
# Standalone version - all paths relative to this repository
# ============================================================================

set -e  # Exit on error

# --- Configuration (Override via environment variables) ---
CONTAINER_NAME="${SAM2_CONTAINER_NAME:-sam2-training}"
IMAGE_NAME="${SAM2_IMAGE:-sam2-training}"

# Workspace: The root of the sam2_training project
WORKSPACE_DIR="${SAM2_WORKSPACE:-$(dirname $(realpath $0))}"

# Dataset: Override this to point to your converted dataset
DATASET_DIR="${SAM2_DATASET:-${WORKSPACE_DIR}/dataset}"

# Config file to use (relative to sam2/configs/)
CONFIG_FILE="${SAM2_CONFIG:-sam2.1_training/custom_finetune}"

# --- Derived Paths (all within this repo) ---
CHECKPOINTS_DIR="${WORKSPACE_DIR}/checkpoints"
TMP_DIR="${WORKSPACE_DIR}/tmp"
LOG_DIR="${WORKSPACE_DIR}/sam2_logs"

# --- Pre-flight Checks ---
if [ ! -f "${CHECKPOINTS_DIR}/sam2.1_hiera_base_plus.pt" ]; then
    echo "=============================================="
    echo "ERROR: Base checkpoint not found!"
    echo "Please download checkpoints first:"
    echo "  cd checkpoints && ./download_ckpts.sh"
    echo "=============================================="
    exit 1
fi

# --- Setup ---
echo "=============================================="
echo "SAM2 Training Pipeline (Standalone)"
echo "=============================================="
echo "Workspace:   ${WORKSPACE_DIR}"
echo "Dataset:     ${DATASET_DIR}"
echo "Checkpoints: ${CHECKPOINTS_DIR}"
echo "Config:      ${CONFIG_FILE}"
echo "Image:       ${IMAGE_NAME}"
echo "=============================================="

# Create necessary directories
mkdir -p "${TMP_DIR}"
mkdir -p "${LOG_DIR}"

# Check if image exists, if not build it
if ! docker image inspect "${IMAGE_NAME}" &> /dev/null; then
    echo "Image '${IMAGE_NAME}' not found. Building..."
    docker build -t "${IMAGE_NAME}" -f "${WORKSPACE_DIR}/Dockerfile.training" "${WORKSPACE_DIR}"
fi

# Remove old container if exists
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting container: ${CONTAINER_NAME}"

docker run --gpus all -it --rm \
    --net=host --privileged \
    -v "/etc/passwd:/etc/passwd:ro" \
    -v "/etc/group:/etc/group:ro" \
    -v "${WORKSPACE_DIR}:/workspace" \
    -v "${DATASET_DIR}:/workspace/dataset" \
    -v "${TMP_DIR}:/workspace/tmp" \
    --workdir /workspace \
    --name "${CONTAINER_NAME}" \
    --ipc=host \
    -e TMPDIR=/workspace/tmp \
    -e PIP_CACHE_DIR=/workspace/tmp/pip_cache \
    -e PYTHONPATH=/workspace \
    "${IMAGE_NAME}" \
    /bin/bash -c "
        set -e
        
        # Install SAM2 package if not already installed
        if ! python3 -c 'import sam2' 2>/dev/null; then
            echo '[Setup] Installing SAM2 from local source...'
            cd /workspace && pip install -e . --no-cache-dir --quiet
        fi
        
        echo '[Training] Starting...'
        cd /workspace
        python3 training/train.py \
            -c ${CONFIG_FILE} \
            --use-cluster 0 \
            --num-gpus 1
    "
