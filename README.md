# SAM2 Training Pipeline

Fine-tune Segment Anything Model 2 (SAM2) on custom video object segmentation datasets.

**This is a standalone repository** - all dependencies are included.

## Quick Start

```bash
# 1. Download model checkpoints (one-time, ~1.5GB for base model)
cd checkpoints && ./download_ckpts.sh && cd ..

# 2. Build the training image (one-time)
docker build -t sam2-training -f Dockerfile.training .

# 3. Prepare your dataset (if in YOLO format)
python3 tools/yolo_to_sam2.py --input /path/to/yolo_dataset --output ./dataset

# 4. Run training
./run_training_docker.sh

# 5. Run inference with auto-detection
docker run --gpus all -v "$(pwd):/workspace" sam2-training \
    python3 tools/inference_auto_detect.py \
        --image your_image.jpg \
        --checkpoint sam2_logs/custom_finetune/checkpoints/checkpoint.pt
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM2_DATASET` | `./dataset` | Path to training dataset |
| `SAM2_CONFIG` | `sam2.1_training/custom_finetune` | Hydra config name |

## Project Structure

```
sam2_training/
├── checkpoints/             # Model weights (download_ckpts.sh)
├── sam2/                    # SAM2 library code
├── sam2_configs/            # Hydra configs (symlink to sam2/configs)
├── training/                # Training code
├── tools/                   # Utilities (dataset converter, inference)
│   ├── inference_auto_detect.py  # Auto-detect players + segment
│   ├── inference.py              # Manual point/box prompt
│   └── yolo_to_sam2.py           # Dataset converter
├── Dockerfile.training      # Production Docker image
├── run_training_docker.sh   # Training launcher
└── run_inference_docker.sh  # Inference launcher
```

## Player Segmentation Results

Fine-tuned for just **10 epochs** with automatic player detection (Grounding DINO + SAM2):

![Player Segmentation](docs/hero_result.png)

| Frame | Players Detected | Confidence Range |
|-------|-----------------|------------------|
| 1 | 5 | 0.88 - 0.97 |
| 2 | 3 | 0.96 - 0.97 |
| 3 | 4 | 0.91 - 0.97 |
| 4 | 6 | 0.84 - 0.97 |
| 5 | 7 | 0.76 - 0.97 |

> More results in `docs/results/`

## Documentation

- [Quick Start](docs/QUICKSTART.md) - Get running in 5 minutes
- [Detailed Guide](docs/DETAILED.md) - Step-by-step walkthrough
- [Dataset Format](docs/DATASET.md) - Data preparation guide
- [Stakeholder Report](docs/STAKEHOLDER_REPORT.md) - **Why Finetune? (Comparison)**
- [Notes](docs/NOTES.md) - Troubleshooting and tips

## License

Apache 2.0 (see LICENSE)
