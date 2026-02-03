# Dataset Format

SAM2 training requires video object segmentation (VOS) format data.

---

## Directory Structure

```
dataset/
├── images/
│   ├── video_001/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   ├── video_002/
│   │   └── ...
│   └── ...
└── masks/
    ├── video_001/
    │   ├── 00000.png
    │   ├── 00001.png
    │   ├── 00002.png
    │   └── ...
    ├── video_002/
    │   └── ...
    └── ...
```

---

## Image Requirements

| Property | Requirement |
|----------|-------------|
| Format | `.jpg`, `.jpeg`, `.png` |
| Naming | Sequential: `00000.jpg`, `00001.jpg`, ... |
| Resolution | Any (resized to 1024×1024 during training) |
| Color | RGB |

---

## Mask Requirements

| Property | Requirement |
|----------|-------------|
| Format | `.png` (grayscale) |
| Naming | Must match image names: `00000.png`, `00001.png`, ... |
| Pixel Values | `0` = background, `1,2,3...` = object IDs |
| Bit Depth | 8-bit or 16-bit grayscale |

### Mask Example

```
┌────────────────────┐
│  0  0  0  0  0  0  │  ← Background (0)
│  0  1  1  1  0  0  │  ← Object 1
│  0  1  1  1  0  0  │
│  0  0  0  0  2  2  │  ← Object 2
│  0  0  0  0  2  2  │
└────────────────────┘
```

---

## Converting from YOLO Format

YOLO segmentation datasets use `.txt` label files with polygon coordinates.

### YOLO Structure
```
yolo_dataset/
├── images/
│   ├── img001.jpg
│   └── img002.jpg
└── labels/
    ├── img001.txt
    └── img002.txt
```

### Conversion
```bash
python3 tools/yolo_to_sam2.py \
    --input /path/to/yolo_dataset \
    --output ./dataset
```

This creates:
- `dataset/images/video_001/` containing renamed frames
- `dataset/masks/video_001/` with rasterized polygon masks

---

## Single Image Datasets

For image-only datasets (not video), treat each image as a 1-frame "video":

```
dataset/
├── images/
│   ├── sample_0001/
│   │   └── 00000.jpg
│   ├── sample_0002/
│   │   └── 00000.jpg
│   └── ...
└── masks/
    ├── sample_0001/
    │   └── 00000.png
    ├── sample_0002/
    │   └── 00000.png
    └── ...
```

Set `num_frames: 1` in config.

---

## Multi-Object Tracking

For multiple objects that persist across frames:

1. Assign consistent IDs across frames (object 1 stays pixel value 1)
2. Objects can appear/disappear
3. Maximum 255 objects per video (8-bit masks)

---

## Validation / Test Split

Optionally use `file_list_txt` in config to specify a subset:

```
# train_list.txt
video_001
video_002
video_003
```

```yaml
dataset:
  file_list_txt: /workspace/train_list.txt
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "No videos found" | Check folder names match between images/ and masks/ |
| Black masks | Ensure masks are grayscale PNG, not RGB |
| Mismatched frames | File names must be identical (00000.jpg ↔ 00000.png) |
| Memory errors | Reduce `num_frames` or image resolution |
