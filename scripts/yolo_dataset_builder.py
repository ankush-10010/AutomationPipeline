"""
yolo_dataset_builder.py — Semi-automated YOLO classification dataset builder.

The fastest way to build a character classification dataset for animation:

  1. EXTRACT: Pull 1 frame every N seconds from episode clips.
  2. PRE-CLASSIFY: Run your existing YOLO model to sort frames into folders.
  3. HINT: Use subtitle speaker labels + wiki episode data to flag likely
     misclassifications (e.g., "Kevin" classified in an S1E1 frame).
  4. REVIEW: You manually fix mistakes by dragging images between folders.
  5. RETRAIN: Train on the corrected dataset.

This script handles steps 1-3. Step 4 is you in a file browser. Step 5 is
a standard ultralytics train command.

Dataset structure produced:
    dataset/
      train/
        Ben_Tennyson/
          s1e1_frame_0042.jpg
          s1e2_frame_0108.jpg
        Vilgax/
          s1e2_frame_0200.jpg
        Heatblast/
          ...
        _UNCERTAIN/        <-- Frames where model confidence was low
          s1e3_frame_0055.jpg
        _REVIEW/           <-- Frames flagged by wiki/subtitle cross-check
          s1e1_frame_0033.jpg
      val/                 <-- 15% split automatically
        Ben_Tennyson/
          ...

Usage:
    # Step 1: Extract + pre-classify (uses existing model)
    python scripts/yolo_dataset_builder.py extract --weights yolo_wt/20epochs.pt

    # Only specific episodes:
    python scripts/yolo_dataset_builder.py extract --weights yolo_wt/20epochs.pt --episodes s1e1 s1e2 s1e3

    # Step 2: After manual review, split into train/val
    python scripts/yolo_dataset_builder.py split

    # Step 3: Retrain
    python scripts/yolo_dataset_builder.py train --weights yolo_wt/20epochs.pt --epochs 30
"""

import argparse
import json
import os
import re
import shutil
import sys
import random
from pathlib import Path
from collections import defaultdict

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config,
    get_active_show,
    get_project_path,
    load_json,
    setup_logging,
    PROJECT_ROOT,
)

log = setup_logging("dataset_builder")

DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
UNCERTAIN_DIR = "_UNCERTAIN"
REVIEW_DIR = "_REVIEW"


# ── Wiki-based episode character map ─────────────────────────────────────────

def build_episode_character_map(show_config: dict, wiki_path: Path = None) -> dict:
    """Build a map of episode_key -> set of canonical character names.

    Sources:
      1. show_config characters with episode_appearances if available
      2. wiki.json parsed for character-episode mentions
      3. episode_index.json character mentions

    Returns {'s1e1': {'Ben Tennyson', 'Gwen Tennyson', ...}, ...}
    """
    ep_chars = defaultdict(set)

    # Always-present characters (main cast present in every episode)
    always_present = set()
    for char in show_config.get("characters", []):
        # Ben, Gwen, Grandpa Max are in almost every episode
        if char.get("always_present", False):
            always_present.add(char["name"])

    # Parse episode_index.json for character mentions
    ep_index_path = PROJECT_ROOT / "topics" / "episode_index.json"
    if ep_index_path.exists():
        try:
            ep_data = load_json(ep_index_path)
            char_names = {c["name"].lower(): c["name"]
                          for c in show_config.get("characters", [])}
            # Add aliases
            for c in show_config.get("characters", []):
                for alias in c.get("aliases", []):
                    char_names[alias.lower()] = c["name"]

            if isinstance(ep_data, dict):
                for title, summary in ep_data.items():
                    summary_lower = str(summary).lower()
                    # Extract episode key from summary
                    m = re.search(r"season\s+(\d+),?\s*episode\s+(\d+)",
                                  summary_lower)
                    if m:
                        ep_key = f"s{int(m.group(1))}e{int(m.group(2))}"
                        for alias_lower, canon in char_names.items():
                            if re.search(rf"\b{re.escape(alias_lower)}\b",
                                         summary_lower):
                                ep_chars[ep_key].add(canon)
        except Exception as e:
            log.warning("Could not parse episode_index: %s", e)

    # Parse wiki.json for additional character-episode associations
    if wiki_path and wiki_path.exists():
        try:
            wiki_data = load_json(wiki_path)
            # wiki.json structure varies — try common patterns
            if isinstance(wiki_data, dict):
                for key, content in wiki_data.items():
                    content_str = json.dumps(content).lower() if not isinstance(content, str) else content.lower()
                    for char in show_config.get("characters", []):
                        char_lower = char["name"].lower()
                        if char_lower in content_str:
                            # Try to find episode references near the character mention
                            for m in re.finditer(r"s(\d+)e(\d+)", content_str):
                                ep_key = f"s{int(m.group(1))}e{int(m.group(2))}"
                                ep_chars[ep_key].add(char["name"])
        except Exception as e:
            log.warning("Could not parse wiki: %s", e)

    # Add always-present characters to all episodes
    if always_present:
        all_eps = set(ep_chars.keys())
        # Also add for any episode we know about from clip_index
        clip_index_path = get_project_path("clip_index", load_pipeline_config())
        if clip_index_path.exists():
            try:
                ci = load_json(clip_index_path)
                clips = ci.get("clips", ci) if isinstance(ci, dict) else ci
                for c in clips:
                    m = re.match(r"(s\d+e\d+)", c.get("filename", ""))
                    if m:
                        all_eps.add(m.group(1).lower())
            except Exception:
                pass

        for ep in all_eps:
            ep_chars[ep] |= always_present

    log.info("Episode-character map: %d episodes", len(ep_chars))
    return dict(ep_chars)


# ── Frame Extraction ─────────────────────────────────────────────────────────

def extract_frames_from_clip(
    video_path: Path,
    output_dir: Path,
    every_n_seconds: float = 2.0,
    prefix: str = "",
) -> list:
    """Extract frames from a video at regular intervals.

    Returns list of (frame_path, timestamp_sec) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * every_n_seconds)
    if frame_interval < 1:
        frame_interval = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    for frame_idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp = frame_idx / fps
        fname = f"{prefix}_frame_{frame_idx:06d}.jpg"
        fpath = output_dir / fname
        cv2.imwrite(str(fpath), frame)
        extracted.append((fpath, timestamp))

    cap.release()
    return extracted


# ── Pre-Classification ───────────────────────────────────────────────────────

def preclassify_frames(
    model,
    frames: list,
    episode_key: str,
    ep_char_map: dict,
    output_base: Path,
    confidence_threshold: float = 0.5,
    uncertain_threshold: float = 0.3,
):
    """Classify extracted frames and sort into character folders.

    Frames below uncertain_threshold go to _UNCERTAIN.
    Frames where the predicted character isn't in the episode's canonical
    character list (from wiki) go to _REVIEW.
    """
    import torch

    canonical_chars = ep_char_map.get(episode_key, set())
    stats = defaultdict(int)

    for frame_path, timestamp in frames:
        try:
            results = model.predict(
                source=str(frame_path),
                verbose=False,
            )
            if not results or len(results) == 0:
                stats["failed"] += 1
                continue

            result = results[0]
            if result.probs is None:
                stats["no_probs"] += 1
                continue

            probs = result.probs.data.cpu().numpy()
            top_idx = probs.argmax()
            top_conf = probs[top_idx]
            top_name = result.names[top_idx]

            # Skip "test" class
            if top_name.lower() == "test":
                stats["test_class"] += 1
                continue

            # Decide destination folder
            if top_conf < uncertain_threshold:
                # Too uncertain — needs manual review
                dest_dir = output_base / UNCERTAIN_DIR
                stats["uncertain"] += 1
            elif canonical_chars and top_name not in canonical_chars:
                # Character shouldn't be in this episode according to wiki
                dest_dir = output_base / REVIEW_DIR
                stats["wiki_flagged"] += 1
            elif top_conf < confidence_threshold:
                # Medium confidence — still sort but flag
                dest_dir = output_base / top_name
                stats["medium_conf"] += 1
            else:
                # High confidence — good to go
                dest_dir = output_base / top_name
                stats["confident"] += 1

            dest_dir.mkdir(parents=True, exist_ok=True)

            # Move the frame to its destination
            dest_path = dest_dir / frame_path.name
            shutil.move(str(frame_path), str(dest_path))

        except Exception as e:
            log.debug("Classification error for %s: %s", frame_path.name, e)
            stats["error"] += 1

    return dict(stats)


# ── Train/Val Split ──────────────────────────────────────────────────────────

def split_dataset(dataset_dir: Path, val_ratio: float = 0.15):
    """Split sorted frames into train/ and val/ directories."""
    raw_dir = dataset_dir / "raw"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not raw_dir.exists():
        # If no raw dir, assume frames are already in dataset_dir root
        # Look for character folders directly
        raw_dir = dataset_dir

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("_"):
            # Skip _UNCERTAIN and _REVIEW
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)

        val_count = max(1, int(len(images) * val_ratio))
        val_images = images[:val_count]
        train_images = images[val_count:]

        # Create class dirs
        (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(str(img), str(train_dir / class_dir.name / img.name))
        for img in val_images:
            shutil.copy2(str(img), str(val_dir / class_dir.name / img.name))

        log.info("  %s: %d train, %d val", class_dir.name,
                 len(train_images), len(val_images))

    log.info("Split complete: train/ and val/ ready in %s", dataset_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def cmd_extract(args):
    """Extract frames from clips and pre-classify with existing model."""
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    clips_dir = Path(
        show_config.get("clips_dir", f"./clips/{show_slug}")
    )
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / show_config.get("clips_dir", f"./clips/{show_slug}")).resolve()

    wiki_path = PROJECT_ROOT / "topics" / "wiki.json"

    log.info("Clips directory: %s", clips_dir)
    log.info("Output: %s", DATASET_DIR)

    # Build episode-character canonical map
    ep_char_map = build_episode_character_map(show_config, wiki_path)

    # Load YOLO model
    from ultralytics import YOLO
    import torch

    weights_path = Path(args.weights)
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / args.weights
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / "yolo_wt" / args.weights

    model = YOLO(str(weights_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log.info("Model loaded: %s on %s", weights_path, device)

    # Find video clips
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    clip_files = []
    for ext in video_extensions:
        clip_files.extend(clips_dir.rglob(f"*{ext}"))

    # Filter by episodes if specified
    if args.episodes:
        ep_set = {e.lower() for e in args.episodes}
        clip_files = [f for f in clip_files
                      if any(f.name.lower().startswith(ep) for ep in ep_set)]

    clip_files.sort(key=lambda f: f.name)
    log.info("Found %d video clips to process", len(clip_files))

    raw_dir = DATASET_DIR / "raw"
    total_stats = defaultdict(int)

    for i, clip_path in enumerate(clip_files):
        # Parse episode key
        m = re.match(r"(s\d+e\d+)", clip_path.name, re.IGNORECASE)
        ep_key = m.group(1).lower() if m else "unknown"
        prefix = clip_path.stem

        # Extract frames to a temp directory
        temp_dir = DATASET_DIR / "_temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)

        frames = extract_frames_from_clip(
            clip_path, temp_dir,
            every_n_seconds=args.interval,
            prefix=prefix,
        )

        if not frames:
            continue

        # Pre-classify and sort
        stats = preclassify_frames(
            model, frames, ep_key, ep_char_map, raw_dir,
            confidence_threshold=args.conf_high,
            uncertain_threshold=args.conf_low,
        )

        for k, v in stats.items():
            total_stats[k] += v

        if (i + 1) % 50 == 0 or (i + 1) == len(clip_files):
            log.info("Processed %d/%d clips | Stats: %s",
                     i + 1, len(clip_files), dict(total_stats))

    # Clean up temp dir
    temp_dir = DATASET_DIR / "_temp_extract"
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nFrames sorted into: {raw_dir}")
    print(f"\nStats:")
    for k, v in sorted(total_stats.items()):
        print(f"  {k}: {v}")

    # Count per-class
    print(f"\nPer-class frame counts:")
    for class_dir in sorted(raw_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            flag = ""
            if class_dir.name == UNCERTAIN_DIR:
                flag = " ← NEEDS MANUAL SORTING"
            elif class_dir.name == REVIEW_DIR:
                flag = " ← WIKI FLAGGED, CHECK THESE"
            elif count < 50:
                flag = " ← LOW COUNT, ADD MORE"
            print(f"  {class_dir.name}: {count}{flag}")

    print(f"\n--- NEXT STEPS ---")
    print(f"1. Open {raw_dir} in your file browser")
    print(f"2. Check {REVIEW_DIR}/ — these are frames where the model's prediction")
    print(f"   conflicts with the wiki (character shouldn't be in that episode)")
    print(f"3. Check {UNCERTAIN_DIR}/ — these are low-confidence frames, sort them")
    print(f"   into the right character folder or delete them")
    print(f"4. Scan each character folder for obvious mistakes (wrong character)")
    print(f"5. Run: python scripts/yolo_dataset_builder.py split")
    print(f"6. Run: python scripts/yolo_dataset_builder.py train --weights {args.weights}")


def cmd_split(args):
    """Split the reviewed dataset into train/val."""
    log.info("Splitting dataset...")
    split_dataset(DATASET_DIR, val_ratio=args.val_ratio)
    print(f"\nDataset split complete!")
    print(f"  Train: {DATASET_DIR / 'train'}")
    print(f"  Val:   {DATASET_DIR / 'val'}")


def cmd_train(args):
    """Train/fine-tune the YOLO classifier on the reviewed dataset."""
    from ultralytics import YOLO

    weights_path = Path(args.weights)
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / args.weights
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / "yolo_wt" / args.weights

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists():
        log.error("No train/ directory found. Run 'split' first.")
        sys.exit(1)

    # Count classes
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    log.info("Training classes: %s", classes)

    model = YOLO(str(weights_path))

    results = model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(PROJECT_ROOT / "yolo_wt"),
        name="retrained",
        exist_ok=True,
    )

    print(f"\nTraining complete!")
    print(f"New weights: {PROJECT_ROOT / 'yolo_wt' / 'retrained'}")


def main():
    parser = argparse.ArgumentParser(
        description="Semi-automated YOLO classification dataset builder."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    p_extract = sub.add_parser("extract",
        help="Extract frames from clips and pre-classify with existing model")
    p_extract.add_argument("--weights", required=True,
        help="Path to existing YOLO weights")
    p_extract.add_argument("--show", default=None)
    p_extract.add_argument("--episodes", nargs="+", default=None,
        help="Only process specific episodes (e.g., s1e1 s1e2)")
    p_extract.add_argument("--interval", type=float, default=2.0,
        help="Extract one frame every N seconds (default: 2.0)")
    p_extract.add_argument("--conf-high", type=float, default=0.6,
        help="High confidence threshold (default: 0.6)")
    p_extract.add_argument("--conf-low", type=float, default=0.25,
        help="Below this confidence -> uncertain (default: 0.25)")

    # --- split ---
    p_split = sub.add_parser("split",
        help="Split reviewed dataset into train/val")
    p_split.add_argument("--val-ratio", type=float, default=0.15,
        help="Validation set ratio (default: 0.15)")

    # --- train ---
    p_train = sub.add_parser("train",
        help="Train YOLO on the reviewed dataset")
    p_train.add_argument("--weights", required=True,
        help="Path to YOLO weights to fine-tune from")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--imgsz", type=int, default=224)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
