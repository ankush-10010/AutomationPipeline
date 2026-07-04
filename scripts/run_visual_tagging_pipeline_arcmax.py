"""
===============================================================================
ARCMAX CASCADE PIPELINE (YOLO 0.85 Threshold + ArcFace Tight Cropping)
===============================================================================

Two-stage highly optimized character tagging for clip_index.json:

  Stage 1 — YOLO Fast Pass (Confidence >= 0.85):
    YOLO scans 20 equally spaced frames per clip. If a character is detected
    with >= 85% confidence, it is immediately confirmed. This bypasses the
    heavy ArcFace math, saving massive amounts of GPU time.

  Stage 2 — ArcFace Verification & Filtering (Confidence < 0.85):
    For unsure YOLO detections, the script mathematically cuts out the exact
    YOLO bounding box (ignoring the messy background). It feeds this pristine,
    tightly-cropped image into CLIP and ArcFace to verify against the vault.
    If it's a dog or false positive, it gets deleted.

Why this architecture is mathematically superior:
  - Surgical Precision: Unlike brute-force methods that scan the whole screen,
    this only sends the precise cropped character boxes to ArcFace.
  - Performance: Checking 20 evenly spaced frames guarantees high accuracy
    without the crushing slowness of scanning every single video frame.

Usage:
  This script is integrated into the master orchestrator menu.
  Run: python scripts/run_visual_tagging_pipeline.py
  Select Option 8 (Test s1e1) or 9 (Full Run).
===============================================================================
"""

import os
import sys
import re
import cv2
import time
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from collections import Counter

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.config_loader import load_pipeline_config, get_project_path, get_active_show

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Constants
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "yolo_wt" / "best.pt"
DEFAULT_HEAD_PATH = PROJECT_ROOT / "arcface_head.pt"
DEFAULT_PROTO_PATH = PROJECT_ROOT / "prototypes.npz"

DEFAULT_TAU = 0.50
DEFAULT_MIN_CONSECUTIVE = 3
DEFAULT_N_FRAMES = 20
DEFAULT_MIN_GLOBAL_CLIPS = 8
NON_CHARACTER_CLASSES = {"Blast"}


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def extract_frames(video_path: Path, n_frames: int = 20) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    start = max(1, int(total * 0.10))
    end = max(start + 1, int(total * 0.90))
    indices = np.linspace(start, end, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def process_yolo_results(yolo_model, frame_pil: Image.Image, high_conf=0.85, low_conf=0.25):
    """Run YOLO on the frame. Return highly confident direct matches and crops for low-confidence verification."""
    results = yolo_model(frame_pil, verbose=False, conf=low_conf)
    
    direct_matches = {}
    crops_to_verify = []
    
    for result in results:
        names_dict = result.names
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_idx = int(boxes.cls[i].cpu().numpy())
            name = names_dict[cls_idx]
            
            # FAST-PATH: YOLO is very confident, bypass ArcFace
            if conf >= high_conf:
                if name not in direct_matches or conf > direct_matches[name]:
                    direct_matches[name] = conf
            # SLOW-PATH: YOLO is unsure, crop and send to ArcFace
            else:
                x1, y1, x2, y2 = map(int, box[:4])
                w, h = frame_pil.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Avoid impossible boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = frame_pil.crop((x1, y1, x2, y2))
                crops_to_verify.append(crop)
                
    return direct_matches, crops_to_verify


def match_frame_to_prototypes(frame_embedding: np.ndarray, prototypes: np.ndarray, prototype_labels: np.ndarray, class_names: list, tau: float) -> dict:
    similarities = prototypes @ frame_embedding  # (P,)
    class_best = {}
    for i, sim in enumerate(similarities):
        cls_idx = prototype_labels[i]
        name = class_names[cls_idx]
        if name not in class_best or sim > class_best[name]:
            class_best[name] = float(sim)
    return {name: sim for name, sim in class_best.items() if sim >= tau}


def aggregate_with_contiguity(per_frame_detections: list, min_consecutive: int) -> dict:
    all_chars = set()
    for frame_det in per_frame_detections:
        all_chars.update(frame_det.keys())
    results = {}
    for char in all_chars:
        sims = [frame_det.get(char, 0.0) for frame_det in per_frame_detections]
        max_sim = max(sims)
        longest_run = 0
        current_run = 0
        for s in sims:
            if s > 0:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        tagged = longest_run >= min_consecutive
        results[char] = {
            "max_similarity": round(max_sim, 4),
            "longest_run": longest_run,
            "tagged": tagged,
        }
    return results


def find_video(filename: str, clips_dir: Path) -> Path:
    direct = clips_dir / filename
    if direct.exists():
        return direct
    ep_match = re.match(r"(s\d+e\d+)", filename, re.IGNORECASE)
    if ep_match:
        sub = clips_dir / ep_match.group(1) / filename
        if sub.exists():
            return sub
    found = list(clips_dir.rglob(filename))
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser(description="Crop & Verify: YOLO Bounding Boxes + ArcFace Verification")
    parser.add_argument("--index", default="clip_index.json")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--yolo", type=str, default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--head", type=str, default=str(DEFAULT_HEAD_PATH))
    parser.add_argument("--prototypes", type=str, default=str(DEFAULT_PROTO_PATH))
    parser.add_argument("--show", default=None)
    parser.add_argument("--episode", default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--cascade-threshold", type=float, default=0.85, help="YOLO confidence to skip ArcFace")
    parser.add_argument("--min-run", type=int, default=DEFAULT_MIN_CONSECUTIVE)
    parser.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument("--min-global-clips", type=int, default=DEFAULT_MIN_GLOBAL_CLIPS, help="Prune characters with fewer than this many clips")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Load YOLO
    log.info("Loading YOLO...")
    from ultralytics import YOLO
    yolo_model = YOLO(args.weights if args.weights != str(DEFAULT_YOLO_WEIGHTS) else args.yolo)
    
    # Load CLIP
    log.info("Loading CLIP ViT-B-32...")
    from sentence_transformers import SentenceTransformer
    clip_model = SentenceTransformer("clip-ViT-B-32")

    # Load ArcFace Projection Head
    log.info("Loading ArcFace Head...")
    head_data = torch.load(args.head, map_location=device, weights_only=True)
    head = ProjectionHead(
        input_dim=head_data["input_dim"],
        hidden_dim=head_data["hidden_dim"],
        output_dim=head_data["output_dim"],
        dropout=head_data.get("dropout", 0.1),
    ).to(device)
    head.load_state_dict(head_data["state_dict"])
    head.eval()

    # Load Prototypes
    log.info("Loading Prototypes...")
    proto_data = np.load(args.prototypes, allow_pickle=True)
    prototypes = proto_data["prototypes"]
    prototype_labels = proto_data["prototype_labels"]
    class_names = list(proto_data["class_names"])

    # Load Data
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)
    clip_index_path = get_project_path("clip_index", pipeline_cfg)
    clips_dir = Path(show_config.get("clips_dir", f"./clips/{show_slug}"))
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / clips_dir).resolve()

    with open(clip_index_path, "r", encoding="utf-8") as f:
        clip_data = json.load(f)
    clips = clip_data.get("clips", clip_data) if isinstance(clip_data, dict) else clip_data

    target_clips = clips
    if args.episode:
        target_clips = [c for c in clips if c.get("filename", "").lower().startswith(args.episode.lower())]

    processed = 0
    skipped = 0
    char_counter = Counter()

    log.info(f"Starting Crop & Verify hybrid processing on {len(target_clips)} clips...")
    
    start_time = time.time()

    for i, clip in enumerate(target_clips):
        # Skip logic
        if clip.get("yolo_arcface", False) and not args.force:
            skipped += 1
            # Add existing tags to the global counter so they don't get falsely pruned at the end!
            for char in clip.get("visual_characters", []):
                char_counter[char] += 1
            continue

        video_path = find_video(clip.get("filename", ""), clips_dir)
        if not video_path:
            skipped += 1
            continue

        frames = extract_frames(video_path, n_frames=args.n_frames)
        if not frames:
            skipped += 1
            continue

        per_frame_detections = []
        clip_yolo_tags = set()
        clip_arc_tags = set()

        for frame_pil in frames:
            # Step 1: YOLO Cascade
            # direct_matches are > 0.85, crops_to_verify are < 0.85
            direct_matches, crops_to_verify = process_yolo_results(
                yolo_model, frame_pil, high_conf=args.cascade_threshold, low_conf=0.25
            )
            
            frame_matches = {}
            
            # Immediately accept highly confident YOLO predictions
            for name, conf in direct_matches.items():
                frame_matches[name] = conf
                clip_yolo_tags.add(name)

            # Step 2: CLIP + ArcFace Verify (Only on low confidence crops)
            if crops_to_verify:
                clip_embs = clip_model.encode(crops_to_verify, show_progress_bar=False)
                norms = np.linalg.norm(clip_embs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                clip_embs = clip_embs / norms

                with torch.no_grad():
                    t = torch.tensor(clip_embs, dtype=torch.float32).to(device)
                    projected = head(t).cpu().numpy()

                for emb in projected:
                    crop_matches = match_frame_to_prototypes(
                        emb, prototypes, prototype_labels, class_names, args.tau
                    )
                    for name, sim in crop_matches.items():
                        if name not in frame_matches or sim > frame_matches[name]:
                            frame_matches[name] = sim
                            clip_arc_tags.add(name)
            
            per_frame_detections.append(frame_matches)

        aggregated = aggregate_with_contiguity(per_frame_detections, args.min_run)
        tagged = sorted([name for name, info in aggregated.items() if info["tagged"]])

        existing_visual = set(clip.get("visual_characters", []))
        existing_chars = set(clip.get("characters", []))
        existing_tags = set(clip.get("visual_tags", []))
        
        new_chars = set(t for t in tagged if t not in NON_CHARACTER_CLASSES)
        new_objects = set(t for t in tagged if t in NON_CHARACTER_CLASSES)

        clip["visual_characters"] = sorted(list(existing_visual.union(new_chars)))
        clip["characters"] = sorted(list(existing_chars.union(new_chars)))
        
        if new_objects or existing_tags:
            clip["visual_tags"] = sorted(list(existing_tags.union(new_objects)))
        clip["yolo_arcface"] = True
        
        proto_det = clip.get("prototype_detections", {})
        for name, info in aggregated.items():
            if info["max_similarity"] > 0.3:
                proto_det[name] = {
                    "max_similarity": info["max_similarity"],
                    "longest_run": info["longest_run"],
                }
        clip["prototype_detections"] = proto_det

        for char in tagged:
            char_counter[char] += 1
        processed += 1

        # Build beautiful tag strings showing source
        display_tags = []
        for t in tagged:
            if t in clip_yolo_tags and t in clip_arc_tags:
                display_tags.append(f"{t} [Both]")
            elif t in clip_yolo_tags:
                display_tags.append(f"{t} [YOLO]")
            elif t in clip_arc_tags:
                display_tags.append(f"{t} [ArcFace]")
            else:
                display_tags.append(t)
                
        # Calculate ETA
        elapsed = time.time() - start_time
        if processed > 0:
            time_per_clip = elapsed / processed
            remaining = len(target_clips) - (i + 1)
            eta_sec = int(remaining * time_per_clip)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
        else:
            eta_str = "Calculating..."

        log.info("[%d/%d | ETA: %s] %s → %s", i + 1, len(target_clips), eta_str, clip.get("filename", "")[:20], display_tags)

        if processed % 50 == 0:
            log.info("Checkpoint: %d clips processed. Saving clip_index.json...", processed)
            if isinstance(clip_data, dict):
                clip_data["clips"] = clips
            with open(clip_index_path, "w", encoding="utf-8") as f:
                json.dump(clip_data, f, indent=2)

    # -- GLOBAL FREQUENCY PRUNING PASS --
    pruned_chars = set()
    for char, count in char_counter.items():
        if count < args.min_global_clips:
            pruned_chars.add(char)
            
    if pruned_chars:
        log.info("Performing Global Frequency Pruning...")
        log.info("Pruning characters with < %d clips: %s", args.min_global_clips, list(pruned_chars))
        for clip in target_clips:
            if "visual_characters" in clip:
                clip["visual_characters"] = [c for c in clip["visual_characters"] if c not in pruned_chars]
            if "characters" in clip:
                clip["characters"] = [c for c in clip["characters"] if c not in pruned_chars]
            if "prototype_detections" in clip:
                for pc in list(clip["prototype_detections"].keys()):
                    if pc in pruned_chars:
                        del clip["prototype_detections"][pc]
        
        # Adjust char_counter for final summary
        for char in pruned_chars:
            del char_counter[char]

    # Final save
    if isinstance(clip_data, dict):
        clip_data["clips"] = clips
    with open(clip_index_path, "w", encoding="utf-8") as f:
        json.dump(clip_data, f, indent=2)

    log.info("Final save complete.")

    print("\n" + "=" * 70)
    print("  ARCMAX CASCADE PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  ArcFace τ: {args.tau}")
    print(f"  Min Run:   {args.min_run}")
    
    print(f"\n  Character detections across {processed} clips:")
    print(f"  {'Character':<22s} {'Clips':>6s}  {'%':>5s}")
    print("  " + "-" * 40)
    for char, count in char_counter.most_common():
        pct = count / processed * 100 if processed else 0
        flag = " ⚠️ HIGH" if pct > 50 else ""
        print(f"  {char:<22s} {count:>6d}  {pct:>4.1f}%{flag}")
    print()

if __name__ == "__main__":
    main()