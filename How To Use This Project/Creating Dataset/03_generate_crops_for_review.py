import os
import json
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLOWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "Ready Dataset"
REVIEW_DIR = INPUT_DIR / "Crop_Review"
METADATA_FILE = REVIEW_DIR / "crop_metadata.json"

def main():
    print(f"{'='*60}\nStarting Step 1: Smart Crop Generator\n{'='*60}")
    
    if REVIEW_DIR.exists():
        print(f"Cleaning up old Crop_Review folder...")
        shutil.rmtree(REVIEW_DIR)
        
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading YOLO-World Generic AI...")
    try:
        model = YOLOWorld('yolov8s-worldv2.pt')
    except Exception:
        model = YOLOWorld('yolov8s-world.pt')
        
    # Instead of specific alien names, we give it incredibly generic terms.
    # This guarantees it will detect EVERY character in the scene.
    model.set_classes(["cartoon character", "person", "alien", "monster", "robot", "animal", "creature", "boy", "girl", "man", "woman"])
    
    metadata = {}
    total_crops = 0
    
    folders = sorted([f for f in INPUT_DIR.iterdir() if f.is_dir() and f.name != "Crop_Review"])
    
    for folder_path in folders:
        character_name = folder_path.name
        images = list(folder_path.glob("*.*"))
        
        print(f"Scanning [{character_name}]... ({len(images)} original images)")
        
        # Create the review folder for this character
        char_review_dir = REVIEW_DIR / character_name
        char_review_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in images:
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h_img, w_img, _ = img.shape
                
            # Use a low confidence (0.05) to ensure we don't miss any characters.
            # It's better to have false positives here because the user will easily delete them!
            results = model.predict(img, verbose=False, conf=0.05)
            
            crop_idx = 0
            for result in results:
                for box in result.boxes:
                    # Get the YOLO format normalized bounding box
                    x_c, y_c, w, h = box.xywhn[0].tolist()
                    
                    # Convert to pixel coordinates for cropping
                    x1 = int((x_c - w/2) * w_img)
                    y1 = int((y_c - h/2) * h_img)
                    x2 = int((x_c + w/2) * w_img)
                    y2 = int((y_c + h/2) * h_img)
                    
                    # Ensure boundaries are within the image size
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_img, x2)
                    y2 = min(h_img, y2)
                    
                    # Skip tiny useless boxes
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                        
                    # Crop the character out of the image
                    crop = img[y1:y2, x1:x2]
                    
                    # Create a unique filename for this crop
                    safe_original_name = img_path.name.replace(" ", "_")
                    crop_filename = f"{safe_original_name}_crop{crop_idx}.jpg"
                    crop_filepath = char_review_dir / crop_filename
                    
                    cv2.imwrite(str(crop_filepath), crop)
                    
                    # CRITICAL: We save exactly where this crop came from in the metadata file.
                    # We use relative paths so it works even if you move the folder.
                    rel_crop_path = f"{character_name}/{crop_filename}"
                    rel_orig_path = f"{character_name}/{img_path.name}"
                    
                    metadata[rel_crop_path] = {
                        "original_image": rel_orig_path,
                        "bbox": [x_c, y_c, w, h] # We store the original normalized YOLO box
                    }
                    
                    crop_idx += 1
                    total_crops += 1
                    
    # Save the mapping metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n{'='*60}")
    print(f"DONE! Generated {total_crops} character crops.")
    print(f"Review Folder: {REVIEW_DIR}")
    print(f"{'='*60}")
    print("INSTRUCTIONS FOR YOU:")
    print("1. Open the Crop_Review folder.")
    print("2. Go inside each character's folder.")
    print("3. DELETE any crop that isn't the correct character (e.g., delete Gwen if she's in the Ben folder).")
    print("4. When you are done deleting the bad crops, tell me, and I will generate the YOLO labels!")

if __name__ == "__main__":
    main()
