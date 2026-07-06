import os
import json
import shutil
import random
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "Ready Dataset"
REVIEW_DIR = INPUT_DIR / "Crop_Review"
METADATA_FILE = REVIEW_DIR / "crop_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "YOLO_Final_Dataset"

def main():
    print(f"{'='*60}\nStarting Step 2: YOLO Dataset Generator\n{'='*60}")
    
    if not METADATA_FILE.exists():
        print(f"Error: {METADATA_FILE} not found. Did you run the crop script first?")
        return
        
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
        
    # 1. Collect surviving crops
    surviving_crops = []
    # We iterate through the JSON keys. If the crop image still exists on disk, it means you kept it!
    for crop_rel_path, data in metadata.items():
        crop_full_path = REVIEW_DIR / crop_rel_path
        if crop_full_path.exists():
            surviving_crops.append((crop_rel_path, data))
            
    print(f"Found {len(surviving_crops)} surviving crops after your manual review.")
    
    if len(surviving_crops) == 0:
        print("No crops survived. Did you delete everything? Exiting.")
        return
        
    # 2. Map original images to their surviving bounding boxes
    # This prevents creating duplicate images if an image has 2 characters in it.
    image_to_bboxes = {}
    class_names_set = set()
    
    for crop_rel_path, data in surviving_crops:
        orig_img_rel = data["original_image"]
        bbox = data["bbox"]
        
        # The class name is the parent folder of the original image
        # e.g., "Ben Tennyson/yt_vid_123.jpg" -> "Ben Tennyson"
        class_name = Path(orig_img_rel).parent.name
        class_names_set.add(class_name)
        
        if orig_img_rel not in image_to_bboxes:
            image_to_bboxes[orig_img_rel] = []
            
        image_to_bboxes[orig_img_rel].append({
            "class_name": class_name,
            "bbox": bbox
        })
        
    # 3. Create Class ID mapping (YOLO requires integer IDs starting from 0)
    class_names = sorted(list(class_names_set))
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Mapped {len(class_names)} distinct character classes to YOLO IDs.")
    
    # 4. Setup fresh output directories
    if OUTPUT_DIR.exists():
        print(f"Cleaning up old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
        
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # 5. Train / Val Split (85% train, 15% val is the industry standard)
    all_images = list(image_to_bboxes.keys())
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.85)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Splitting dataset: {len(train_images)} Training, {len(val_images)} Validation images.")
    
    # 6. Process and Copy Files
    def process_split(image_list, split_name):
        copied = 0
        for orig_img_rel in image_list:
            orig_img_full = INPUT_DIR / orig_img_rel
            
            if not orig_img_full.exists():
                continue # Original image got deleted somehow, safely skip
                
            # Create a safe filename (prefix with folder name) so we don't accidentally overwrite files
            safe_filename = orig_img_rel.replace("/", "_").replace("\\", "_").replace(" ", "_")
            new_img_path = OUTPUT_DIR / "images" / split_name / safe_filename
            new_lbl_path = OUTPUT_DIR / "labels" / split_name / f"{new_img_path.stem}.txt"
            
            # Copy the full original image
            shutil.copy(str(orig_img_full), str(new_img_path))
            
            # Write the YOLO .txt label containing ONLY the boxes you didn't delete
            bboxes = image_to_bboxes[orig_img_rel]
            with open(new_lbl_path, "w") as f:
                for box_data in bboxes:
                    cls_id = class_name_to_id[box_data["class_name"]]
                    x_c, y_c, w, h = box_data["bbox"]
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    
            copied += 1
        return copied
        
    train_copied = process_split(train_images, "train")
    val_copied = process_split(val_images, "val")
    
    # 7. Generate data.yaml (The instruction file YOLO needs to start training)
    yaml_data = {
        'path': str(OUTPUT_DIR.absolute()), # Absolute path is safer for YOLO training scripts
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"\n{'='*60}")
    print("YOLO DATASET GENERATION COMPLETE!")
    print(f"Successfully processed {train_copied + val_copied} full-frame images with precise bounding boxes.")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print("You are now fully ready to train your custom YOLOv8 model!")

if __name__ == "__main__":
    main()
