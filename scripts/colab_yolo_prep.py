"""
colab_yolo_prep.py

Prepares a balanced dataset for YOLO training on Colab.
1. Keeps folder names exactly as they are (no merges or renames).
2. Drops classes with < 50 images (they will be handled by k-NN instead).
3. Drops non-character classes like 'Police' if present.
4. Caps massively overrepresented classes (Ben, Gwen, Max) at 300 images
   so they don't dominate the loss function.
5. Performs a clean 85/15 train/val split.

Outputs to a new folder: 'Colab Dataset'
"""

import os
import shutil
import random
from pathlib import Path

MIN_IMAGES_PER_CLASS = 50
MAX_IMAGES_PER_CLASS = 300
VAL_RATIO = 0.15

# Folders to explicitly skip (e.g., non-characters)
IGNORE_FOLDERS = {"Police"}

def main():
    src_dir = Path("Ready Dataset")
    out_dir = Path("Colab Dataset")
    
    if not src_dir.exists():
        print(f"Error: Could not find '{src_dir}'. Make sure 'Ready Dataset' is in the current working directory.")
        return

    # Create fresh output directories
    if out_dir.exists():
        shutil.rmtree(out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    print("\n--- Prepping Colab Dataset ---")
    valid_classes = 0
    skipped_classes = 0

    for folder in sorted(src_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_") or folder.name in IGNORE_FOLDERS:
            continue
            
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        
        # Skip classes with too few images for YOLO
        if len(images) < MIN_IMAGES_PER_CLASS:
            print(f"⏭️  Skipping '{folder.name}' ({len(images)} images < {MIN_IMAGES_PER_CLASS} min threshold -> leaving for k-NN)")
            skipped_classes += 1
            continue
            
        random.shuffle(images)
        
        # Cap the massive classes
        original_count = len(images)
        if len(images) > MAX_IMAGES_PER_CLASS:
            images = images[:MAX_IMAGES_PER_CLASS]
            
        # Split train/val
        val_count = max(1, int(len(images) * VAL_RATIO))
        train_imgs = images[val_count:]
        val_imgs = images[:val_count]
        
        # Write files
        cls_name = folder.name
        (train_dir / cls_name).mkdir(parents=True, exist_ok=True)
        (val_dir / cls_name).mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            shutil.copy2(img, train_dir / cls_name / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_dir / cls_name / img.name)
            
        cap_note = f"(capped down from {original_count})" if original_count > MAX_IMAGES_PER_CLASS else ""
        print(f"✅ {cls_name:18s}: {len(train_imgs):3d} train | {len(val_imgs):2d} val  {cap_note}")
        valid_classes += 1

    print("\n🎉 Done!")
    print(f"Prepared {valid_classes} classes for YOLO training. Skipped {skipped_classes} rare classes for k-NN.")

if __name__ == "__main__":
    main()
