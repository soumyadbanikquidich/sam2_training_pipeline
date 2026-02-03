# SAM2 Training Pipeline

Fine-tune SAM2 on custom video segmentation datasets.

## Quick Start

```bash
# 1. Download checkpoints (~1.5GB)
cd checkpoints && ./download_ckpts.sh && cd ..

# 2. Build Docker image
docker build -t sam2-training -f Dockerfile.training .

# 3. Convert dataset (if YOLO format)
python3 tools/yolo_to_sam2.py --input /path/to/yolo --output ./dataset

# 4. Train
./run_training_docker.sh

# 5. Inference (auto-detect)
docker run --gpus all -v "$(pwd):/workspace" sam2-training \
    python3 tools/inference_auto_detect.py \
        --image your_image.jpg \
        --checkpoint sam2_logs/custom_finetune/checkpoints/checkpoint.pt
```

## Config

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM2_DATASET` | `./dataset` | Training data path |
| `SAM2_CONFIG` | `sam2.1_training/custom_finetune` | Hydra config |

## Structure

```
├── checkpoints/          # Weights (run download_ckpts.sh)
├── sam2/                 # SAM2 code
├── training/             # Training scripts
├── tools/
│   ├── inference_auto_detect.py  # Auto-detect + segment
│   ├── inference.py              # Manual prompt
│   └── yolo_to_sam2.py           # Dataset converter
├── Dockerfile.training
├── run_training_docker.sh
└── run_inference_docker.sh
```

## Results (10 epochs)

![Segmentation](docs/hero_result.png)

| Frame | Detected | Confidence |
|-------|----------|------------|
| 1 | 5 | 0.88-0.97 |
| 2 | 3 | 0.96-0.97 |
| 3 | 4 | 0.91-0.97 |
| 4 | 6 | 0.84-0.97 |
| 5 | 7 | 0.76-0.97 |

## Docs

- [Quick Start](docs/QUICKSTART.md)
- [Detailed Guide](docs/DETAILED.md)
- [Dataset Format](docs/DATASET.md)
- [Notes](docs/NOTES.md)
