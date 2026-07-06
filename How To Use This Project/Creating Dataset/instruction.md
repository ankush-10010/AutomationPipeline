<div align="center">

# 🖼️ YOLO Dataset Creation Guide

### *From raw video frames to a high-quality object detection dataset*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-World-FF4500?style=for-the-badge)](https://github.com/ultralytics/ultralytics)

---

*Build a hyper-accurate computer vision dataset by extracting frames, visually clustering them, cropping characters with AI, and manually reviewing the crops to generate perfect labels.*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [The Big Picture Workflow](#-the-big-picture-workflow)
- [Step 1: Frame Extraction](#-step-1-frame-extraction)
- [Step 2: AI Clustering](#-step-2-ai-clustering)
- [Step 3: Character Cropping & Metadata](#-step-3-character-cropping--metadata)
- [Step 4: Final YOLO Label Generation](#-step-4-final-yolo-label-generation)
- [Step 5: Model Training](#-step-5-model-training)
- [Step 6: ArcFace Metric Training](#-step-6-arcface-metric-training-optional-but-highly-recommended)
- [Troubleshooting & Tips](#-troubleshooting--tips)

---

## 🔭 Overview

Training an accurate YOLO model for character recognition in animation requires a pristine dataset where characters are correctly boxed and labeled, while preserving the background.

This process handles the entire pipeline using **four scripts** located right here in the `Creating Dataset/` folder. The workflow heavily utilizes AI to do the heavy lifting (clustering and zero-shot bounding boxes) while relying on you for quick, visual manual review to ensure 100% accuracy.

---

## 🧩 The Big Picture Workflow

```mermaid
flowchart TD
    subgraph STAGE1 ["🎬 Stage 1: Extraction & Sorting"]
        V["📺 Video Files\n(Videos/Ben10_Classic/)"] --> S1["🐍 01_extract_raw_frames.py\n(1 frame per N seconds)"]
        S1 --> D["📂 Dataset/\n(~165k raw images)"]
        
        D -.->|"Manual Step:\nRename/Move to"| DG["📂 Dataset_Golden/"]
        
        DG --> S2["🐍 02_cluster_images.py\n(OpenCLIP + HDBSCAN)"]
        S2 --> C["📂 Clusters/\n(cluster_0, cluster_1...)"]
    end

    subgraph STAGE2 ["🔍 Stage 2: Manual Review & Organization"]
        C -.->|"Manual Step:\nDelete bad images\nMerge clusters\nRename to character names"| RD["📂 Ready Dataset/\n(BenTennyson/, Gwen/...)"]
    end

    subgraph STAGE3 ["✂️ Stage 3: AI Cropping & Labelling"]
        RD --> S3["🐍 03_generate_crops_for_review.py\n(YOLO-World)"]
        
        S3 -->|"Saves cropped faces"| CR["📂 Crop_Review/\n(BenTennyson/, Gwen/...)"]
        S3 -->|"Saves bbox axes"| MD["📄 crop_metadata.json"]
        
        CR -.->|"Manual Step:\nDelete incorrect crops\ne.g., delete Gwen crop\nfrom Ben's folder"| CR_CLEAN["📂 Crop_Review/\n(Cleaned)"]
        
        CR_CLEAN --> S4["🐍 04_generate_yolo_labels_from_crops.py"]
        MD --> S4
        RD --> S4
        
        S4 -->|"Generates standard\nYOLO .txt labels"| YFD["✅ YOLO_Final_Dataset/\n(images/ & labels/)"]
    end

    subgraph STAGE4 ["🚀 Stage 4: Model Training"]
        YFD --> S5["🐍 05_train_yolo_model.py"]
        S5 -->|"Trains & exports"| FM["🧠 final_model/\n(best.pt)"]
    end

    subgraph STAGE5 ["🧬 Stage 5: Advanced Metric Learning"]
        RD --> S6["🐍 06_train_arcface_model.py\n(Few-Shot Recognition)"]
        S6 -->|"Exports head & prototypes"| AF["🧠 arcface_head.pt &\nprototypes.npz"]
    end

    style V fill:#6366f1,stroke:#4f46e5,color:#fff
    style S1 fill:#f97316,stroke:#ea580c,color:#fff
    style D fill:#06b6d4,stroke:#0891b2,color:#fff
    style DG fill:#22c55e,stroke:#16a34a,color:#fff
    style S2 fill:#f97316,stroke:#ea580c,color:#fff
    style C fill:#a855f7,stroke:#9333ea,color:#fff
    
    style RD fill:#22c55e,stroke:#16a34a,color:#fff
    
    style S3 fill:#f97316,stroke:#ea580c,color:#fff
    style CR fill:#a855f7,stroke:#9333ea,color:#fff
    style MD fill:#eab308,stroke:#ca8a04,color:#fff
    style CR_CLEAN fill:#22c55e,stroke:#16a34a,color:#fff
    
    style S4 fill:#f97316,stroke:#ea580c,color:#fff
    style YFD fill:#06b6d4,stroke:#0891b2,color:#fff

    style S5 fill:#f97316,stroke:#ea580c,color:#fff
    style FM fill:#ec4899,stroke:#db2777,color:#fff

    style S6 fill:#f97316,stroke:#ea580c,color:#fff
    style AF fill:#ec4899,stroke:#db2777,color:#fff
```

---

## 🎬 Step 1: Frame Extraction

**Script:** `01_extract_raw_frames.py`

This script rips frames from your episodes to build the raw image pool.

### How it works:
1. It looks for video files in `Videos/Ben10_Classic/`.
2. It extracts 1 frame every few seconds from every episode.
3. Saves all images to the `Dataset/` folder.

### What you need to do:
1. Ensure your videos are placed correctly in `AutomationPipeline-main/Videos/Ben10_Classic/`.
2. Run the script:
   ```bash
   python 01_extract_raw_frames.py
   ```
3. **Important:** When extraction is complete, rename the `Dataset/` folder to `Dataset_Golden/` (or copy the best images there) before proceeding to Step 2.

> [!CAUTION]
> This step can generate over 150,000 images and consume gigabytes of storage space. Ensure you have enough disk space before running!

---

## 🤖 Step 2: AI Clustering

**Script:** `02_cluster_images.py`

Sorting 150,000 images manually is impossible. This script uses AI (OpenCLIP) to group visually similar images together into clusters.

### How it works:
1. Reads all images from `Dataset_Golden/`.
2. Generates visual embeddings for each image.
3. Groups them into ~100 folders named `cluster_0`, `cluster_1`, inside the `Clusters/` directory.

### What you need to do:
1. Run the script:
   ```bash
   python 02_cluster_images.py
   ```
2. **The Manual Labor:** Open the `Clusters/` folder in your file explorer.
   * Go into `cluster_0`. If it's mostly "Ben Tennyson", delete the random bad images inside it.
   * Rename the folder from `cluster_0` to `BenTennyson`.
   * If you find another cluster that is also Ben, clean it out, and drag its images into your renamed `BenTennyson` folder.
   * Delete clusters that are just random backgrounds or junk.
3. Once you have nice, clean folders named after your characters (e.g., ~32 character folders), move all those character folders into a new directory called **`Ready Dataset/`** in the project root.

> [!TIP]
> Keep your folder names identical to the character names used in your `config/show_config.yaml` to ensure maximum compatibility with the rest of the pipeline!

---

## ✂️ Step 3: Character Cropping & Metadata

**Script:** `03_generate_crops_for_review.py`

Just because an image is in the `BenTennyson` folder doesn't mean Ben is the *only* character in the image. If Gwen is standing next to him, we don't want the model learning that Gwen's face = Ben.

### How it works:
1. The script scans the `Ready Dataset/` folder.
2. It uses a generic YOLO-World model to detect *any* character in the image.
3. It crops out every detected character and saves them in `Ready Dataset/Crop_Review/`.
4. **Crucially:** It saves the exact X,Y bounding box coordinates of that crop to `crop_metadata.json`.

### What you need to do:
1. Run the script:
   ```bash
   python 03_generate_crops_for_review.py
   ```
2. **The Final Review:** Open the new `Ready Dataset/Crop_Review/` folder.
   * Open the `BenTennyson` folder inside it.
   * You will see tightly cropped faces/bodies.
   * If you see a crop of Gwen, Kevin, or a random alien in this folder—**delete that specific image**.
   * Leave *only* the genuine crops of Ben Tennyson. Do this for all folders.

> [!NOTE]
> Don't worry about the backgrounds being missing in these crops. We are NOT training the model on these cropped images. This is just for your visual review!

---

## 🏷️ Step 4: Final YOLO Label Generation

**Script:** `04_generate_yolo_labels_from_crops.py`

This script closes the loop, merging your manual review with the original full-frame images.

### How it works:
1. It looks at the `Crop_Review/` folder. 
2. If a cropped image still exists on disk, it assumes you approved it!
3. It reads the bounding box coordinates for that approved crop from `crop_metadata.json`.
4. It creates a `.txt` label file for the **original, full-background image**.
5. It exports the final images and labels to `YOLO_Final_Dataset/` in the standard YOLO format (`images/train/`, `labels/train/`, etc.).

### What you need to do:
1. Run the script:
   ```bash
   python 04_generate_yolo_labels_from_crops.py
   ```
2. You are done! Your dataset is now located in `YOLO_Final_Dataset/` and is ready for ultralytics YOLO training.

> [!IMPORTANT]
> Because of this workflow, if an original image had both Ben and Gwen, but it was stored in the `BenTennyson` folder, the final label file will *only* have a bounding box drawn around Ben. Gwen will be treated as background, which is exactly what we want for robust model training!

---

## 🚀 Step 5: Model Training

**Script:** `05_train_yolo_model.py`

Once your dataset is fully built and labeled, it's time to train your custom object detection model! This script handles the Ultralytics YOLOv8 training pipeline automatically.

### How it works:
1. It automatically detects if you have an NVIDIA GPU (CUDA) available.
2. It looks for the dataset configuration file at `YOLO_Final_Dataset/data.yaml`.
3. It kicks off the training process (fine-tuning a base YOLOv8 model) using your carefully curated images and labels.
4. When finished, it saves the compiled model weights to the `final_model/` directory.

### What you need to do:
1. Ensure your dataset was generated properly in Step 4.
2. Run the training script:
   ```bash
   python 05_train_yolo_model.py
   ```
3. **Wait!** Training can take anywhere from a few hours to a day depending on your dataset size and hardware.
4. Once complete, you will find your best trained weights at:
   `AutomationPipeline-main/final_model/weights/best.pt`

> [!WARNING]
> **Hardware Check:** This script requires a dedicated NVIDIA GPU for reasonable training times. If the script warns you that it is "Training on CPU", you should expect the training to take a *very* long time. If you do not have a strong GPU, you should zip the `YOLO_Final_Dataset` folder and upload it to Google Colab to run the training there!

---

## 🧬 Step 6: ArcFace Metric Training (Optional but Highly Recommended)

**Script:** `06_train_arcface_model.py`

While YOLO is incredible at finding main characters, what happens when you only have **6 images** of a rare alien (like Eye Guy) vs **1,600 images** of Ben? A standard classification model will simply ignore the rare alien. 

This script solves that by fine-tuning a projection head on top of a frozen CLIP ViT-B-32 model using **ArcFace loss**. ArcFace optimizes for angular separation in embedding space, pushing even rare identities away from the rest, making it perfect for **few-shot, open-set character recognition**.

### How it works:
1. It pre-computes CLIP embeddings for all images directly from your `Ready Dataset/` (it doesn't use the YOLO format).
2. It trains a small projection head with ArcFace loss to separate all 30+ character classes.
3. It computes a "prototype" embedding for each class in this new space.
4. It outputs `arcface_head.pt` (the weights) and `prototypes.npz` (the embeddings).

### What you need to do:
1. Make sure your `Ready Dataset/` is fully organized with all character folders.
2. Run the script:
   ```bash
   python 06_train_arcface_model.py --dataset "Ready Dataset" --epochs 40
   ```
3. Once training is complete, the `arcface_head.pt` and `prototypes.npz` files will be saved.

> [!CAUTION]
> **Local Training vs Colab:** Just like YOLO, training this ArcFace head locally requires an NVIDIA GPU and libraries like `sentence-transformers` and `torchvision`. 
> 
> If you are on a slow machine, it is highly recommended to **use Google Colab**:
> 1. Zip your `Ready Dataset/` folder and upload it to Google Drive.
> 2. Open Colab and run: `!pip install -q sentence-transformers torch torchvision`
> 3. Download this exact python script to Colab.
> 4. Run `!python 06_train_arcface_model.py --dataset "Ready Dataset" --epochs 40`
> 5. Download the extremely small `arcface_head.pt` (~500KB) and `prototypes.npz` back to your laptop.

---

## 🛠️ Troubleshooting & Tips

- **Missing OpenCLIP/HDBSCAN:** If step 2 fails, you need to install the clustering libraries:
  `pip install open_clip_torch hdbscan scikit-learn`
- **Low YOLO-World Confidence:** If Step 3 misses some characters, you can open `03_generate_crops_for_review.py` and tweak the generic classes on line 30 to better fit your show's art style (e.g., adding "anime character" or "superhero").
- **Accidental Deletions:** Always keep a backup of `Ready Dataset/` before running Step 3, just in case you need to start the cropping process over again.

</div>
