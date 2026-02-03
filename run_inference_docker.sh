#!/bin/bash
# ============================================================================
# SAM2 Inference - Docker Launcher
# Standalone version - all paths relative to this repository
# ============================================================================

set -e

CONTAINER_NAME="sam2-inference"
IMAGE_NAME="${SAM2_IMAGE:-sam2-training}"
WORKSPACE_DIR="$(dirname $(realpath $0))"

# Parse arguments
IMAGE_PATH="$1"
CHECKPOINT="${2:-sam2_logs/custom_finetune/checkpoints/checkpoint.pt}"
OUTPUT="${3:-results/inference_output.png}"

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: ./run_inference_docker.sh <image_path> [checkpoint] [output]"
    echo ""
    echo "Arguments:"
    echo "  image_path   Path to input image (required)"
    echo "  checkpoint   Model checkpoint (default: trained model)"
    echo "  output       Output path (default: results/inference_output.png)"
    echo ""
    echo "Examples:"
    echo "  ./run_inference_docker.sh dataset/images/video_001/00000.jpg"
    echo "  ./run_inference_docker.sh my_image.jpg checkpoints/sam2.1_hiera_base_plus.pt result.png"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "${WORKSPACE_DIR}/results"

echo "=============================================="
echo "SAM2 Inference (Standalone)"
echo "=============================================="
echo "Image:      ${IMAGE_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT}"
echo "=============================================="

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --gpus all --rm \
    -v "${WORKSPACE_DIR}:/workspace" \
    --workdir /workspace \
    --name "${CONTAINER_NAME}" \
    -e PYTHONPATH=/workspace \
    -e SAM2_CONFIGS_DIR=/workspace/sam2_configs \
    "${IMAGE_NAME}" \
    python3 tools/inference.py \
        --image "${IMAGE_PATH}" \
        --checkpoint "${CHECKPOINT}" \
        --config "sam2.1/sam2.1_hiera_b+.yaml" \
        --configs-dir "/workspace/sam2_configs" \
        --output "${OUTPUT}"

echo "Done! Result saved to: ${OUTPUT}"
