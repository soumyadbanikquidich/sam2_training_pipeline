import os
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import shutil

def yolo_to_mask(image_path, label_path, output_mask_path):
    """
    Converts a single YOLO label file to a mask image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])
            # YOLO format: class x1 y1 x2 y2 ... (normalized)
            coords = np.array([float(x) for x in parts[1:]])
            
            # Reshape to (N, 2) points
            points = coords.reshape(-1, 2)
            
            # Denormalize
            points[:, 0] *= w
            points[:, 1] *= h
            
            points = points.astype(np.int32)
            
            # Draw polygon on mask
            # We use idx + 1 as the object ID for this frame (instance segmentation logic for SAM2)
            # If tracking consistency is needed across frames, we'd need a way to link IDs.
            # For now, we assume simple conversion where each object gets a unique ID in the frame.
            cv2.fillPoly(mask, [points], color=(idx + 1))

    cv2.imwrite(output_mask_path, mask)

def convert_dataset(input_dir, output_dir):
    """
    Converts a dataset with 'images' and 'labels' subdirs to SAM2 format.
    SAM2 Structure:
    output_dir/
      images/
        video_name/
           00000.jpg
      masks/
        video_name/
           00000.png
    """
    
    # Locate images and labels
    # Verify input structure first
    img_dir_in = os.path.join(input_dir, "images")
    lbl_dir_in = os.path.join(input_dir, "labels")
    
    # Handle Train/Val/Test splits inside images/labels if they exist
    # But usually YOLO structure is images/train, images/val
    # We will search recursively
    
    image_files = sorted(glob(os.path.join(img_dir_in, "**", "*.[jJ][pP][gG]"), recursive=True))
    if not image_files:
         image_files = sorted(glob(os.path.join(img_dir_in, "**", "*.[pP][nN][gG]"), recursive=True))

    print(f"Found {len(image_files)} images in {img_dir_in}")

    # Output structure
    out_img_root = os.path.join(output_dir, "images")
    out_mask_root = os.path.join(output_dir, "masks")
    os.makedirs(out_img_root, exist_ok=True)
    os.makedirs(out_mask_root, exist_ok=True)

    # For this specific task, we treat the whole dataset as ONE video sequence 
    # OR separate videos if they are in subfolders.
    # Given the user's sample dataset 'Downloads/sample_auto_label', let's see how it looks.
    # It usually has images directly or in 'train' folders.
    
    # Group by parent folder to detect 'videos'
    video_frames = {}
    
    for img_path in image_files:
        parent_dir = os.path.basename(os.path.dirname(img_path))
        # If parent_dir is 'train' or 'images', we might want to use a generic video name
        if parent_dir.lower() in ['train', 'val', 'test', 'images']:
             video_name = "video_001" # Default single video assumption for flat datasets
        else:
            video_name = parent_dir
            
        if video_name not in video_frames:
            video_frames[video_name] = []
        video_frames[video_name].append(img_path)

    for video_name, frames in video_frames.items():
        print(f"Processing video: {video_name} with {len(frames)} frames")
        
        vid_img_out = os.path.join(out_img_root, video_name)
        vid_mask_out = os.path.join(out_mask_root, video_name)
        os.makedirs(vid_img_out, exist_ok=True)
        os.makedirs(vid_mask_out, exist_ok=True)
        
        # Sort frames to ensure order if they are numbered
        frames.sort()
        
        for i, img_path in enumerate(tqdm(frames)):
            # Determine label path
            # YOLO convention: replace images/ with labels/ and .jpg with .txt
            # It handles nested structures usually
            
            # Simple heuristic:
            rel_path = os.path.relpath(img_path, input_dir)
            # e.g. images/Train/001.jpg -> labels/Train/001.txt
            
            # We construct the label path by swapping 'images' with 'labels' in the rel path
            if "images" in rel_path:
                rel_lbl_path = rel_path.replace("images", "labels", 1)
            else:
                 # Fallback if structure is weird
                 rel_lbl_path = rel_path
                 
            rel_lbl_path = os.path.splitext(rel_lbl_path)[0] + ".txt"
            label_path = os.path.join(input_dir, rel_lbl_path)
            
            # Copy image to new location with frame index name
            # SAM2 likes 00000.jpg, 00001.jpg etc.
            new_img_name = f"{i:05d}.jpg"
            new_mask_name = f"{i:05d}.png"
            
            target_img_path = os.path.join(vid_img_out, new_img_name)
            target_mask_path = os.path.join(vid_mask_out, new_mask_name)
            
            shutil.copy2(img_path, target_img_path)
            yolo_to_mask(img_path, label_path, target_mask_path)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input YOLO dataset root")
    parser.add_argument("--output", required=True, help="Output SAM2 dataset root")
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output)
