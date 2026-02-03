# Important Notes & Troubleshooting

---

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| RAM | 16 GB | 32+ GB |
| Disk | 15 GB free | 50+ GB |
| CUDA | 12.1+ | 12.1+ |

---

## Common Errors & Fixes

### "No space left on device"
```bash
# Clear pip cache
pip cache purge

# Clear docker cache
docker system prune -a
```

### "ModuleNotFoundError: No module named 'X'"
Rebuild the Docker image:
```bash
docker build --no-cache -t sam2-training -f Dockerfile.training .
```

### "CUDA out of memory"
Reduce config parameters:
```yaml
scratch:
  train_batch_size: 1
  num_frames: 1
  resolution: 512  # Lower resolution
```

### "Cannot find primary config"
Ensure `train.py` uses `initialize_config_dir` pointing to `sam2/configs/`:
```python
config_dir = os.path.join(..., "sam2", "configs")
initialize_config_dir(config_dir=config_dir, version_base="1.2")
```

### Training stuck at data loading
- Reduce `num_train_workers`
- Check dataset paths are correct
- Verify mask file format (grayscale PNG)

---

## Performance Tips

### Speed
- Use SSD for dataset storage
- Increase `num_train_workers` (up to CPU cores)
- Enable compiled model (`compile_image_encoder: True` in config)

### Memory
- Use AMP (enabled by default: `amp.enabled: True`)
- Gradient checkpointing for larger models
- Reduce batch size / resolution

### Quality
- More epochs for small datasets
- Data augmentation (enabled by default)
- Higher resolution if VRAM allows

---

## Checkpoints

### Location
```
sam2_logs/<config_name>/
├── checkpoints/
│   └── checkpoint.pt         # Latest checkpoint
├── logs/
│   └── *.log                 # Training logs
├── tensorboard/
│   └── events.*              # TensorBoard data
└── config.yaml               # Resolved config
```

### Checkpoint Size
- SAM2 B+: ~350 MB
- SAM2 L: ~500 MB

---

## Docker Tips

### Interactive Mode
```bash
# Enter container for debugging
docker run --gpus all -it --rm \
    -v /path/to/sam2_training:/workspace \
    -w /workspace \
    sam2-training /bin/bash
```

### Persistent Pip Cache
Add to `run_training_docker.sh`:
```bash
-v ~/.cache/pip:/root/.cache/pip
```

---

## Hydra Config Override

Override any config value via command line:
```bash
# In the docker command
python3 training/train.py -c sam2.1_training/custom_finetune \
    ++scratch.num_epochs=50 \
    ++scratch.base_lr=1e-5
```

Note: Use `++` prefix for Hydra overrides.

---

## Logs

### View Training Progress
```bash
tail -f sam2_logs/custom_finetune/logs/*.log
```

### TensorBoard
```bash
tensorboard --logdir sam2_logs/custom_finetune/tensorboard --port 6006
```

---

## Known Limitations

1. **Single-frame training**: With `num_frames=1`, temporal consistency is not learned
2. **Small datasets**: May overfit quickly; use early stopping or fewer epochs
3. **No validation**: Current config trains only; add val split manually if needed

---

## Getting Help

1. Check Meta's SAM2 documentation: https://github.com/facebookresearch/segment-anything-2
2. Review config files in `sam2/configs/sam2.1_training/`
3. Enable debug mode: `HYDRA_FULL_ERROR=1 python3 training/train.py ...`
