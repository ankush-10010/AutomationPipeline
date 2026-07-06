import os
import json
import uuid
import torch
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from ultralytics import YOLOWorld
    import open_clip
    import hdbscan
except ImportError:
    print("Missing libraries! Please run:")
    print("pip install ultralytics open_clip_torch hdbscan scikit-learn")
    exit(1)

def main():
    source_dir = Path("Dataset_Golden")
    cluster_dir = Path("Clusters")
    metadata_path = Path("cluster_metadata.json")

    if not source_dir.exists():
        print(f"Error: {source_dir} not found. Run sample_dataset.py first!")
        return

    images = list(source_dir.glob("*.jpg"))
    print(f"Found {len(images)} images in {source_dir}.")

    # 1. Load YOLO-World
    print("Loading YOLO-World...")
    yolo_model = YOLOWorld("yolov8s-world.pt")
    yolo_model.set_classes(["cartoon character", "alien creature", "person", "monster"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load CLIP
    print(f"Loading CLIP on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device).eval()

    def embed_crop(crop_img):
        img_tensor = preprocess(crop_img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(img_tensor).cpu().numpy()[0]
            return emb / np.linalg.norm(emb)

    all_embeddings = []
    crop_records = []
    
    print("Step 1 & 2: Detecting, Cropping, and Embedding...")
    for idx, img_path in enumerate(images, 1):
        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(images)} images...")
            
        results = yolo_model.predict(str(img_path), conf=0.15, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            continue
            
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        for box in results[0].boxes:
            # Get normalized xywh for YOLO training later
            nx, ny, nw, nh = box.xywhn[0].tolist()
            
            # Get unnormalized xyxy for cropping now
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop image
            crop_img = pil_image.crop((x1, y1, x2, y2))
            
            # Ignore tiny crops (noise)
            if crop_img.size[0] < 20 or crop_img.size[1] < 20:
                continue
                
            # Embed
            embedding = embed_crop(crop_img)
            
            crop_id = str(uuid.uuid4())
            crop_records.append({
                "crop_id": crop_id,
                "image_path": str(img_path),
                "box_normalized": [nx, ny, nw, nh],
                "crop_img": crop_img # We keep this in memory temporarily to save it to clusters later
            })
            all_embeddings.append(embedding)

    if not all_embeddings:
        print("No characters found!")
        return

    print(f"Extracted {len(all_embeddings)} character crops!")
    
    # 3. Cluster with KMeans
    print("Step 3: Clustering with KMeans (Forcing 100 distinct clusters)...")
    from sklearn.cluster import KMeans
    embeddings_matrix = np.array(all_embeddings)
    
    # We force exactly 100 clusters. This guarantees no "massive blobs".
    # If there are 30 characters, you will get about 3 folders per character.
    clusterer = KMeans(n_clusters=100, random_state=42, n_init="auto")
    labels = clusterer.fit_predict(embeddings_matrix)

    # 4. Dump to Folders
    print("Step 4: Dumping clusters to folders...")
    import shutil
    if cluster_dir.exists():
        shutil.rmtree(cluster_dir) # Wipe old bad clusters completely
    cluster_dir.mkdir(exist_ok=True)
    
    metadata = {}
    
    for record, cluster_id in zip(crop_records, labels):
        # -1 in HDBSCAN is 'noise' (junk). We put it in a noise folder.
        folder_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        out_folder = cluster_dir / folder_name
        out_folder.mkdir(exist_ok=True)
        
        crop_filename = f"{record['crop_id']}.jpg"
        out_path = out_folder / crop_filename
        record["crop_img"].save(out_path)
        
        # Save metadata mapping for Phase 3
        metadata[record["crop_id"]] = {
            "image_path": record["image_path"],
            "box_normalized": record["box_normalized"]
        }

    # Save metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Phase 1 Complete!")
    print(f"Crops are organized in '{cluster_dir}/'.")
    print("Next step: Manually look through the cluster folders and rename them to character names (e.g., 'cluster_0' -> 'heatblast').")
    print("You can just delete the 'noise' folder, or any folders that are just random backgrounds.")

if __name__ == "__main__":
    main()
