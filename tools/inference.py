#!/usr/bin/env python3
"""
SAM2 Inference Script
Works with SAM2.1 checkpoints that may have extra training keys
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

logging.basicConfig(level=logging.INFO)


def show_mask(mask, ax, color=None):
    """Overlay mask on image"""
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """Draw prompt points"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', 
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25, zorder=10)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25, zorder=10)


def build_sam2_flexible(config_file, ckpt_path, device="cuda", configs_dir=None):
    """Build SAM2 model with flexible checkpoint loading (strict=False)"""
    
    # Initialize Hydra with config directory
    if configs_dir is None:
        configs_dir = os.environ.get("SAM2_CONFIGS_DIR", "/segment-anything-2/sam2_configs")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    initialize_config_dir(config_dir=configs_dir, version_base="1.2")
    
    # Build model
    hydra_overrides = [
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    ]
    
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    
    # Load checkpoint with strict=False to handle extra keys
    if ckpt_path is not None:
        logging.info(f"Loading checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        
        # Handle both raw state_dict and wrapped in "model" key
        if "model" in sd:
            sd = sd["model"]
        
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        
        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys (ignored): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
        
        logging.info("Checkpoint loaded successfully")
    
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    image_path: str,
    checkpoint_path: str,
    model_cfg: str = "sam2_hiera_b+.yaml",
    output_path: str = None,
    point_coords: list = None,
    configs_dir: str = None,
):
    """Run SAM2 inference on a single image"""
    
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    print(f"Loading model: {model_cfg}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Build model with flexible loading
    sam2_model = build_sam2_flexible(model_cfg, checkpoint_path, configs_dir=configs_dir)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Set image
    predictor.set_image(image_np)
    
    # Use center point if not specified
    if point_coords is None:
        h, w = image_np.shape[:2]
        point_coords = [[w // 2, h // 2]]
    
    point_coords = np.array(point_coords)
    point_labels = np.ones(len(point_coords), dtype=np.int32)
    
    print(f"Running inference with point at {point_coords[0]}...")
    
    # Predict
    with torch.inference_mode():
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]
    
    print(f"Generated {len(masks)} masks")
    print(f"Scores: {[f'{s:.3f}' for s in scores]}")
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original with prompt point
    axes[0].imshow(image_np)
    show_points(point_coords, point_labels, axes[0])
    axes[0].set_title("Input + Prompt", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show each mask
    colors = [
        np.array([30/255, 144/255, 255/255, 0.6]),  # Blue
        np.array([255/255, 144/255, 30/255, 0.6]),  # Orange
        np.array([30/255, 255/255, 144/255, 0.6]),  # Green
    ]
    
    for idx, (mask, score) in enumerate(zip(masks[:3], scores[:3])):
        axes[idx + 1].imshow(image_np)
        show_mask(mask, axes[idx + 1], color=colors[idx])
        show_points(point_coords, point_labels, axes[idx + 1])
        axes[idx + 1].set_title(f"Mask {idx + 1} (score: {score:.3f})", fontsize=12)
        axes[idx + 1].axis('off')
    
    plt.suptitle("SAM2 Inference Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + "_inference.png"
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved result to: {output_path}")
    
    plt.close()
    
    return masks, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Inference")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="sam2_hiera_b+.yaml", 
                        help="Model config (from sam2_configs/)")
    parser.add_argument("--configs-dir", default=None,
                        help="Path to sam2_configs directory")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--point", type=float, nargs=2, action='append',
                        help="Point prompt as x y (can specify multiple)")
    
    args = parser.parse_args()
    
    point_coords = args.point if args.point else None
    
    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_cfg=args.config,
        output_path=args.output,
        point_coords=point_coords,
        configs_dir=args.configs_dir,
    )
