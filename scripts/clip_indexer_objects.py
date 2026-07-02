"""
clip_indexer_objects.py — General YOLO object detection for visual tags.

Runs a standard YOLOv8 model (COCO pretrained) on clip keyframes to detect
general visual elements (person, car, fire, etc.) and stores them as
`visual_tags` in clip_index.json.

NOTE: COCO-trained models may struggle with cartoon/animation content.
If detection rates are low, consider:
  - Using a YOLOv8 model fine-tuned on anime/cartoon data
  - Lowering the confidence threshold (--conf 0.15)
  - Using the existing fine-tuned character model alongside this

Dependencies:
    pip install ultralytics opencv-python-headless

Usage:
    python scripts/clip_indexer_objects.py
    python scripts/clip_indexer_objects.py --target-dir s1e1
    python scripts/clip_indexer_objects.py --weights yolov8n.pt --conf 0.25
    python scripts/clip_indexer_objects.py --weights yolo_wt/20epochs.pt  # Use your fine-tuned model
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config,
    get_active_show,
    get_project_path,
    load_json,
    save_json,
    setup_logging,
    PROJECT_ROOT,
)

log = setup_logging("clip_objects")

# Tags to exclude (too generic or noisy for matching)
EXCLUDE_TAGS = {"background", "sky", "wall", "floor", "ground"}


def detect_objects_in_clip(
    model, video_path: Path, sample_every_n: int = 5, conf_threshold: float = 0.25
) -> list:
    """Run YOLO object detection on sampled frames from a video clip.

    Returns a deduplicated list of detected object class names, sorted by
    frequency (most frequently detected first).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Sample frames evenly
    frame_indices = list(range(0, total_frames, sample_every_n))
    if not frame_indices:
        frame_indices = [total_frames // 2]

    class_counts = defaultdict(int)

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        try:
            results = model.predict(source=frame, verbose=False, conf=conf_threshold)
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, "boxes") and result.boxes is not None:
                    # Object detection model
                    for cls_id in result.boxes.cls.cpu().numpy():
                        name = result.names[int(cls_id)]
                        if name.lower() not in EXCLUDE_TAGS:
                            class_counts[name] += 1
                elif hasattr(result, "probs") and result.probs is not None:
                    # Classification model (like your fine-tuned one)
                    probs = result.probs.data.cpu().numpy()
                    for cls_id, conf in enumerate(probs):
                        if conf >= conf_threshold:
                            name = result.names[cls_id]
                            if name.lower() not in EXCLUDE_TAGS and name.lower() != "test":
                                class_counts[name] += 1
        except Exception as e:
            log.debug("Detection error on frame %d: %s", fi, e)
            continue

    cap.release()

    # Sort by frequency (most common first), return names only
    sorted_tags = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    return [name for name, count in sorted_tags]


def main():
    parser = argparse.ArgumentParser(
        description="General YOLO object detection for visual tags."
    )
    parser.add_argument("--index", default=None, help="Path to clip_index.json")
    parser.add_argument("--show", default=None, help="Show identifier")
    parser.add_argument(
        "--weights", default="yolov8n.pt",
        help="YOLO model weights path (default: yolov8n.pt for COCO, "
             "or pass your fine-tuned weights)"
    )
    parser.add_argument(
        "--target-dir", default=None,
        help="Only process clips matching this prefix (e.g. s1e1)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for detections (default: 0.25, "
             "lower for cartoons)"
    )
    parser.add_argument(
        "--sample-every", type=int, default=5,
        help="Sample every Nth frame (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Save checkpoint every N clips"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-detect even if visual_tags already exists"
    )
    args = parser.parse_args()

    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    clip_index_path = (
        Path(args.index) if args.index
        else get_project_path("clip_index", pipeline_cfg)
    )

    clips_dir = Path(
        show_config.get("clips_dir", f"./clips/{show_slug}")
    )
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / show_config.get("clips_dir", f"./clips/{show_slug}")).resolve()

    log.info("Clips dir: %s", clips_dir)
    log.info("YOLO weights: %s", args.weights)

    # Load YOLO model
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Try project-relative path
        weights_path = PROJECT_ROOT / args.weights
    if not weights_path.exists():
        # Try yolo_wt directory
        weights_path = PROJECT_ROOT / "yolo_wt" / args.weights
    if not weights_path.exists():
        log.info("Weights not found locally, ultralytics will download: %s", args.weights)
        weights_path = Path(args.weights)

    model = YOLO(str(weights_path))
    model.to(device)
    log.info("YOLO model loaded: %s", weights_path)

    # Load clip index
    clip_data = load_json(clip_index_path)
    if isinstance(clip_data, dict):
        clips = clip_data.get("clips", [])
    elif isinstance(clip_data, list):
        clips = clip_data
    else:
        log.error("Invalid clip index format")
        sys.exit(1)

    # Filter clips
    if args.target_dir:
        target = args.target_dir.lower()
        indices = [i for i, c in enumerate(clips)
                   if c.get("filename", "").lower().startswith(target)]
    else:
        indices = list(range(len(clips)))

    if not args.force:
        indices = [i for i in indices if not clips[i].get("visual_tags")]

    log.info("Clips to process: %d", len(indices))

    processed = 0
    skipped = 0

    for batch_start in range(0, len(indices), args.batch_size):
        batch = indices[batch_start:batch_start + args.batch_size]

        for idx in batch:
            clip = clips[idx]
            filename = clip.get("filename", "")

            # Locate video file
            video_path = clips_dir / filename
            if not video_path.exists():
                ep_match = re.match(r"(s\d+e\d+)", filename, re.IGNORECASE)
                if ep_match:
                    video_path = clips_dir / ep_match.group(1) / filename
                if not video_path.exists():
                    found = list(clips_dir.rglob(filename))
                    if found:
                        video_path = found[0]
                    else:
                        skipped += 1
                        continue

            tags = detect_objects_in_clip(
                model, video_path,
                sample_every_n=args.sample_every,
                conf_threshold=args.conf,
            )

            if tags:
                clip["visual_tags"] = tags
                processed += 1
            else:
                skipped += 1

        # Checkpoint
        log.info(
            "Progress: %d detected, %d skipped (batch %d/%d)",
            processed, skipped,
            batch_start // args.batch_size + 1,
            (len(indices) + args.batch_size - 1) // args.batch_size,
        )
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(clip_index_path, clip_data)

    log.info("Object detection complete: %d tagged, %d skipped", processed, skipped)


if __name__ == "__main__":
    main()
