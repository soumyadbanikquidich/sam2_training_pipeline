# Detailed Training Guide

Complete walkthrough of the SAM2 training pipeline.

---

## 1. Environment Setup

### Docker Image

The `Dockerfile.training` extends `auto-label` with training dependencies:

```dockerfile
FROM auto-label
RUN pip install submitit hydra-core iopath omegaconf fvcore tensordict tensorboard pandas
```

**Build:**
```bash
docker build -t sam2-training -f Dockerfile.training .
```

### Required Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `sam2_training/` | `/workspace` | Training code |
| `segment-anything-2/` | `/segment-anything-2` | SAM2 library |
| `checkpoints/` | `/checkpoints` | Model weights |
| Dataset | `/workspace/dataset` | Training data |

---

## 2. Configuration

### Config File Location

Configs are in `sam2/configs/sam2.1_training/`. The main config is `custom_finetune.yaml`.

### Key Parameters

```yaml
scratch:
  resolution: 1024          # Image resolution
  train_batch_size: 1       # Batch size (1 for single GPU)
  num_train_workers: 2      # DataLoader workers
  num_frames: 1             # Frames per video sample
  max_num_objects: 3        # Max objects per frame
  num_epochs: 10            # Training epochs
  base_lr: 5.0e-6           # Base learning rate

dataset:
  img_folder: /workspace/dataset/images
  gt_folder: /workspace/dataset/masks
  file_list_txt: null       # Optional: limit to subset

launcher:
  gpus_per_node: 1          # Number of GPUs
```

### Modifying Config

Edit `sam2/configs/sam2.1_training/custom_finetune.yaml` or create a new config file.

---

## 3. Running Training

### Basic Run

```bash
./run_training_docker.sh
```

### Environment Variables

```bash
# Custom dataset
SAM2_DATASET=/path/to/dataset ./run_training_docker.sh

# Custom config
SAM2_CONFIG=sam2.1_training/my_config ./run_training_docker.sh

# Custom checkpoint directory
SAM2_CHECKPOINTS=/path/to/checkpoints ./run_training_docker.sh

# All options
SAM2_DATASET=/data/vos \
SAM2_CONFIG=sam2.1_training/large_model \
SAM2_CHECKPOINTS=/models \
./run_training_docker.sh
```

---

## 4. Understanding the Training Loop

### What Happens

1. **Data Loading**: `VOSDataset` loads video frames + masks
2. **Sampling**: `RandomUniformSampler` picks `num_frames` from each video
3. **Augmentation**: Random flip, resize, color jitter, normalization
4. **Forward Pass**: SAM2 encodes image → generates mask predictions
5. **Loss**: Mask loss (BCE), Dice loss, IoU loss, Classification loss
6. **Optimization**: AdamW with cosine LR decay

### Loss Components

| Loss | Weight | Purpose |
|------|--------|---------|
| `loss_mask` | 20 | Binary cross-entropy on masks |
| `loss_dice` | 1 | Dice coefficient loss |
| `loss_iou` | 1 | IoU prediction accuracy |
| `loss_class` | 1 | Object presence classification |

---

## 5. Multi-GPU Training

Modify `launcher.gpus_per_node` in config:

```yaml
launcher:
  gpus_per_node: 4
```

Or via command line:
```bash
# In run_training_docker.sh, change --num-gpus
python3 training/train.py -c sam2.1_training/custom_finetune --use-cluster 0 --num-gpus 4
```

---

## 6. Resuming Training

Checkpoints are saved to `sam2_logs/<config>/checkpoints/`. To resume:

1. Training auto-resumes if checkpoint exists
2. Or manually specify in config:

```yaml
checkpoint:
  resume_from: /path/to/checkpoint.pt
```

---

## 7. Exporting for Inference

After training, the checkpoint is at:
```
sam2_logs/custom_finetune/checkpoints/checkpoint.pt
```

Load with SAM2's standard inference API:
```python
from sam2.build_sam import build_sam2
model = build_sam2("sam2.1_hiera_b+", "path/to/checkpoint.pt")
```
