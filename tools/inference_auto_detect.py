#!/usr/bin/env python3
"""
SAM2 Inference with Automatic Person Detection
Uses Grounding DINO to detect persons, then SAM2 to segment them
"""

import os
import sys
import argparse
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

logging.basicConfig(level=logging.INFO)


def show_mask(mask, ax, color=None):
    """Overlay mask on image"""
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, color='green'):
    """Draw bounding box"""
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, 
                               edgecolor=color, facecolor='none', lw=2))


def build_sam2_flexible(config_file, ckpt_path, device="cuda", configs_dir=None):
    """Build SAM2 model with flexible checkpoint loading"""
    if configs_dir is None:
        configs_dir = os.environ.get("SAM2_CONFIGS_DIR", "/workspace/sam2_configs")
    
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=configs_dir, version_base="1.2")
    
    hydra_overrides = [
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    ]
    
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    
    if ckpt_path is not None:
        logging.info(f"Loading SAM2 checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        logging.info("SAM2 checkpoint loaded")
    
    model = model.to(device)
    model.eval()
    return model


def detect_persons(image, detector, processor, device="cuda", threshold=0.3):
    """Detect persons in image using Grounding DINO"""
    text = "person."
    
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detector(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]]
    )
    
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    
    logging.info(f"Detected {len(boxes)} person(s)")
    return boxes, scores


def run_inference(
    image_path: str,
    checkpoint_path: str,
    model_cfg: str = "sam2.1/sam2.1_hiera_b+.yaml",
    output_path: str = None,
    configs_dir: str = None,
    detection_threshold: float = 0.3,
):
    """Run SAM2 inference with automatic person detection"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Grounding DINO for person detection
    logging.info("Loading Grounding DINO detector...")
    dino_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(dino_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    
    # Load SAM2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    logging.info("Loading SAM2 model...")
    sam2_model = build_sam2_flexible(model_cfg, checkpoint_path, configs_dir=configs_dir)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Load image
    logging.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Detect persons
    boxes, det_scores = detect_persons(image, detector, processor, device, detection_threshold)
    
    if len(boxes) == 0:
        logging.warning("No persons detected!")
        return None, None
    
    # Set image for SAM2
    predictor.set_image(image_np)
    
    # Get point prompts from box centers
    centers = []
    for box in boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centers.append([cx, cy])
    
    centers = np.array(centers)
    labels = np.ones(len(centers), dtype=np.int32)
    
    logging.info(f"Using {len(centers)} point prompt(s) from detected boxes")
    
    # Predict masks using boxes as prompts (better than points for full person)
    with torch.inference_mode():
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
    
    # Reshape masks
    if masks.ndim == 3:
        masks = masks[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)
    
    logging.info(f"Generated {len(masks)} mask(s)")
    logging.info(f"Mask scores: {[f'{s:.3f}' for s in scores.flatten()]}")
    
    # Visualize
    n_detections = min(len(boxes), 4)
    fig, axes = plt.subplots(1, n_detections + 1, figsize=(5 * (n_detections + 1), 5))
    
    if n_detections == 0:
        axes = [axes]
    
    # Original with all detections
    axes[0].imshow(image_np)
    for i, box in enumerate(boxes[:n_detections]):
        show_box(box, axes[0], color='lime')
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        axes[0].scatter(cx, cy, c='red', s=100, marker='*', zorder=10)
    axes[0].set_title(f"Detected {len(boxes)} Player(s)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Individual masks
    colors = [
        np.array([30/255, 144/255, 255/255, 0.6]),
        np.array([255/255, 144/255, 30/255, 0.6]),
        np.array([30/255, 255/255, 144/255, 0.6]),
        np.array([255/255, 30/255, 144/255, 0.6]),
    ]
    
    for i in range(n_detections):
        axes[i + 1].imshow(image_np)
        show_mask(masks[i], axes[i + 1], color=colors[i % len(colors)])
        show_box(boxes[i], axes[i + 1], color='lime')
        score = scores[i] if scores.ndim == 1 else scores[i, 0]
        axes[i + 1].set_title(f"Player {i + 1} (score: {score:.3f})", fontsize=12)
        axes[i + 1].axis('off')
    
    plt.suptitle("SAM2 Player Segmentation (Auto-Detected)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + "_players.png"
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logging.info(f"\n✓ Saved result to: {output_path}")
    
    plt.close()
    
    return masks, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Player Segmentation with Auto-Detection")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path")
    parser.add_argument("--config", default="sam2.1/sam2.1_hiera_b+.yaml", 
                        help="SAM2 config (from sam2_configs/)")
    parser.add_argument("--configs-dir", default=None, help="Path to sam2_configs directory")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_cfg=args.config,
        output_path=args.output,
        configs_dir=args.configs_dir,
        detection_threshold=args.threshold,
    )
