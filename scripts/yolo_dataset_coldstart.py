"""
yolo_dataset_coldstart.py — Bootstrap a YOLO classification dataset from scratch.

When you have NO usable pretrained model for this show, the fastest path is:

  1. SPEAKER-LABEL EXTRACTION: Parse SRT files for ">> character:" speaker labels.
     Extract the video frame at that timestamp. The speaker is usually on screen.
     This gives you a ~60-70% accurate initial sort FOR FREE.

  2. KEYBOARD SORTER: A fast manual review tool. Shows each frame full-screen,
     you press a number key (1-9, 0) to assign it to a character, 'd' to delete,
     's' to skip. Processes hundreds of frames in minutes.

  3. TRAIN: Standard ultralytics train on the sorted dataset.

  4. ASSISTED ROUND 2: Now that you have a model, extract MORE frames and use
     the model to pre-sort them. Fix mistakes. Retrain. Each cycle is faster.

Dataset structure:
    yolo_dataset/
      raw/
        Ben/
        Vilgax/
        Gwen/
        Grandpa_Max/
        Heatblast/
        ...
        _UNSORTED/     <-- Frames that need manual sorting
        _TRASH/        <-- Frames you've rejected (blank, transitions, etc.)

Usage:
    # Step 1: Extract frames using SRT speaker labels (cold start, no model needed)
    python scripts/yolo_dataset_coldstart.py extract --show ben10

    # Step 1b: Also extract unlabeled frames for manual sorting (more diversity)
    python scripts/yolo_dataset_coldstart.py extract-unlabeled --show ben10 --episodes s1e1 s1e2

    # Step 2: Manual keyboard sorter for _UNSORTED frames
    python scripts/yolo_dataset_coldstart.py sort

    # Step 3: Split into train/val
    python scripts/yolo_dataset_coldstart.py split

    # Step 4: Train
    python scripts/yolo_dataset_coldstart.py train --epochs 30

    # Step 5: After training, use yolo_dataset_builder.py for the assisted loop
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

log = setup_logging("dataset_coldstart")

DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
RAW_DIR = DATASET_DIR / "raw"
UNSORTED_DIR = RAW_DIR / "_UNSORTED"
TRASH_DIR = RAW_DIR / "_TRASH"


# ── SRT Parsing ──────────────────────────────────────────────────────────────

def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp to seconds."""
    h, m, s_ms = time_str.strip().split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_speakers(srt_path: Path) -> list:
    """Parse SRT file and extract speaker-labeled timestamps.

    Returns list of (speaker_name, start_sec, end_sec, dialogue_text).
    """
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = content.strip().split("\n\n")
    entries = []

    # Match various speaker label formats:
    #   >> kevin: text
    #   >> Ben: text
    #   >>kevin: text
    #   > kevin: text
    speaker_re = re.compile(r">{1,2}\s*([A-Za-z_\s]+?):\s*(.*)", re.IGNORECASE)

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        times = lines[1].split(" --> ")
        if len(times) != 2:
            continue

        try:
            start = parse_srt_time(times[0])
            end = parse_srt_time(times[1])
        except Exception:
            continue

        text = " ".join(lines[2:])
        match = speaker_re.search(text)
        if match:
            speaker = match.group(1).strip().lower()
            dialogue = match.group(2).strip()
            if speaker and len(speaker) > 1:
                entries.append((speaker, start, end, dialogue))

    return entries


# ── Speaker to Character Mapping ─────────────────────────────────────────────

def build_speaker_to_char_map(show_config: dict) -> dict:
    """Map SRT speaker labels to canonical character names + folder-safe names.

    Returns {speaker_label: (canonical_name, folder_name)}
    """
    mapping = {}

    for char in show_config.get("characters", []):
        canon = char["name"]
        # Folder-safe name: replace spaces with underscore
        folder = canon.replace(" ", "_")

        # Map the canonical name and all aliases
        mapping[canon.lower()] = (canon, folder)
        for alias in char.get("aliases", []):
            mapping[alias.strip().lower()] = (canon, folder)

    return mapping


# ── Frame Extraction from Video ──────────────────────────────────────────────

def extract_frame_at_time(video_path: Path, timestamp_sec: float) -> any:
    """Extract a single frame from a video at the given timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_idx = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def extract_frames_uniform(video_path: Path, every_n_sec: float = 3.0) -> list:
    """Extract frames at uniform intervals. Returns list of (frame, timestamp)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    interval_frames = int(fps * every_n_sec)

    frames = []
    for fi in range(0, total_frames, interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((frame, fi / fps))

    cap.release()
    return frames


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_extract(args):
    """Extract frames using SRT speaker labels as weak labels."""
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    subtitles_dir = get_project_path("subtitles_dir", pipeline_cfg)
    clips_dir = Path(
        show_config.get("clips_dir", f"./clips/{show_slug}")
    )
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / show_config.get("clips_dir", f"./clips/{show_slug}")).resolve()

    # Also check for full episode videos (not just clips)
    episodes_dir = Path(
        show_config.get("episodes_dir",
                        show_config.get("video_dir", f"./videos/{show_slug}"))
    )
    if not episodes_dir.is_absolute():
        episodes_dir = (PROJECT_ROOT / episodes_dir).resolve()

    speaker_map = build_speaker_to_char_map(show_config)
    log.info("Speaker map: %d entries", len(speaker_map))
    log.info("Subtitles dir: %s", subtitles_dir)
    log.info("Clips dir: %s", clips_dir)
    log.info("Episodes dir: %s", episodes_dir)

    # Find SRT files
    srt_files = sorted(subtitles_dir.rglob("*.srt")) if subtitles_dir.exists() else []
    log.info("Found %d SRT files", len(srt_files))

    if args.episodes:
        ep_set = {e.lower() for e in args.episodes}
        srt_files = [f for f in srt_files
                     if any(ep in f.name.lower() for ep in ep_set)]

    total_extracted = 0
    total_unmapped = 0
    char_counts = defaultdict(int)

    for srt_path in srt_files:
        # Find the corresponding video file
        ep_match = re.search(r"(s\d+e\d+)", srt_path.name, re.IGNORECASE)
        if not ep_match:
            continue
        ep_key = ep_match.group(1).lower()
        log.info("Processing %s from %s", ep_key.upper(), srt_path.name)

        # Try to find the full episode video
        video_path = None
        video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".webm"]

        # Search in episodes_dir
        for ext in video_extensions:
            candidates = list(episodes_dir.rglob(f"*{ep_key}*{ext}"))
            if candidates:
                video_path = candidates[0]
                break

        # If no full episode, we'll use individual clips
        use_clips = video_path is None

        # Parse speaker labels from SRT
        speaker_entries = parse_srt_speakers(srt_path)
        log.info("  Found %d speaker-labeled lines", len(speaker_entries))

        if use_clips:
            # Need to map SRT timestamps to clip files using manifests or
            # sequential clip durations
            log.info("  No full episode video found, using clip files...")
            # Load clip_index to get clip timecodes
            clip_index_path = get_project_path("clip_index", pipeline_cfg)
            ci = load_json(clip_index_path)
            clips = ci.get("clips", ci) if isinstance(ci, dict) else ci
            ep_clips = [(c["filename"], c.get("duration_seconds", 0))
                        for c in clips
                        if c.get("filename", "").lower().startswith(ep_key)]
            ep_clips.sort()

            # Reconstruct timecodes
            clip_times = []
            t = 0.0
            for fname, dur in ep_clips:
                clip_times.append((fname, t, t + dur))
                t += dur

            for speaker, start, end, dialogue in speaker_entries:
                if speaker not in speaker_map:
                    total_unmapped += 1
                    continue

                canon_name, folder_name = speaker_map[speaker]
                mid_time = (start + end) / 2

                # Find which clip contains this timestamp
                for fname, clip_start, clip_end in clip_times:
                    if clip_start <= mid_time <= clip_end:
                        clip_path = None
                        for p in clips_dir.rglob(fname):
                            clip_path = p
                            break
                        if clip_path is None:
                            break

                        local_time = mid_time - clip_start
                        frame = extract_frame_at_time(clip_path, local_time)
                        if frame is not None:
                            out_dir = RAW_DIR / folder_name
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / f"{ep_key}_{mid_time:.1f}s.jpg"
                            cv2.imwrite(str(out_path), frame)
                            total_extracted += 1
                            char_counts[folder_name] += 1
                        break
        else:
            # Full episode video available — much simpler
            log.info("  Using full episode: %s", video_path.name)

            for speaker, start, end, dialogue in speaker_entries:
                if speaker not in speaker_map:
                    total_unmapped += 1
                    continue

                canon_name, folder_name = speaker_map[speaker]

                # Extract frame at the midpoint of the speech
                mid_time = (start + end) / 2
                frame = extract_frame_at_time(video_path, mid_time)

                if frame is not None:
                    out_dir = RAW_DIR / folder_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{ep_key}_{mid_time:.1f}s.jpg"
                    cv2.imwrite(str(out_path), frame)
                    total_extracted += 1
                    char_counts[folder_name] += 1

        log.info("  %s: extracted %d frames so far", ep_key, total_extracted)

    # Summary
    print("\n" + "=" * 60)
    print("SPEAKER-LABEL EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nTotal frames extracted: {total_extracted}")
    print(f"Unmapped speakers (not in show_config): {total_unmapped}")
    print(f"\nPer-character counts:")
    for char, count in sorted(char_counts.items(), key=lambda x: -x[1]):
        flag = " ← LOW, need more" if count < 50 else ""
        print(f"  {char}: {count}{flag}")

    print(f"\n--- IMPORTANT ---")
    print(f"These frames are sorted by WHO IS SPEAKING, not who is ON SCREEN.")
    print(f"~60-70% of the time the speaker is visible, but sometimes the")
    print(f"camera shows the listener or a wide shot instead.")
    print(f"")
    print(f"NEXT STEPS:")
    print(f"1. Quick-scroll each folder in {RAW_DIR}")
    print(f"2. Delete or move frames where the labeled character isn't visible")
    print(f"3. For more diversity, run:  python scripts/yolo_dataset_coldstart.py extract-unlabeled")
    print(f"4. Then sort those with:     python scripts/yolo_dataset_coldstart.py sort")


def cmd_extract_unlabeled(args):
    """Extract uniform-interval frames without labels for manual sorting."""
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    clips_dir = Path(
        show_config.get("clips_dir", f"./clips/{show_slug}")
    )
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / show_config.get("clips_dir", f"./clips/{show_slug}")).resolve()

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    clip_files = []
    for ext in video_extensions:
        clip_files.extend(clips_dir.rglob(f"*{ext}"))

    if args.episodes:
        ep_set = {e.lower() for e in args.episodes}
        clip_files = [f for f in clip_files
                      if any(ep in f.name.lower() for ep in ep_set)]

    clip_files.sort(key=lambda f: f.name)
    log.info("Found %d clips for unlabeled extraction", len(clip_files))

    UNSORTED_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for clip_path in clip_files:
        frames = extract_frames_uniform(clip_path, every_n_sec=args.interval)
        for frame, ts in frames:
            fname = f"{clip_path.stem}_{ts:.1f}s.jpg"
            cv2.imwrite(str(UNSORTED_DIR / fname), frame)
            total += 1

    print(f"\nExtracted {total} unlabeled frames to {UNSORTED_DIR}")
    print(f"Run 'python scripts/yolo_dataset_coldstart.py sort' to classify them.")


def cmd_sort(args):
    """Interactive keyboard-based frame sorter."""
    # Discover character classes from existing folders
    classes = []
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            classes.append(d.name)

    if not classes:
        print("No character folders found yet. Create them first or run 'extract'.")
        return

    # Get unsorted frames
    unsorted_frames = (
        list(UNSORTED_DIR.glob("*.jpg")) + list(UNSORTED_DIR.glob("*.png"))
    )
    if not unsorted_frames:
        print(f"No unsorted frames in {UNSORTED_DIR}")
        return

    random.shuffle(unsorted_frames)

    # Print key mapping
    print("\n" + "=" * 60)
    print("KEYBOARD SORTER")
    print("=" * 60)
    print(f"\nFrames to sort: {len(unsorted_frames)}")
    print(f"\nKey mapping:")
    key_map = {}
    for i, cls_name in enumerate(classes[:10]):
        key = str(i + 1) if i < 9 else "0"
        key_map[ord(key)] = cls_name
        print(f"  [{key}] {cls_name}")
    print(f"  [d] Delete / Trash")
    print(f"  [s] Skip")
    print(f"  [q] Quit")
    print(f"\nStarting... (press a key when the image window is focused)\n")

    sorted_count = 0
    trashed_count = 0
    skipped_count = 0

    TRASH_DIR.mkdir(parents=True, exist_ok=True)

    for frame_path in unsorted_frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Resize for display
        h, w = img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Add filename text
        cv2.putText(img, frame_path.name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Sort Frame (press key)", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("d"):
            shutil.move(str(frame_path), str(TRASH_DIR / frame_path.name))
            trashed_count += 1
        elif key == ord("s"):
            skipped_count += 1
        elif key in key_map:
            dest_dir = RAW_DIR / key_map[key]
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(frame_path), str(dest_dir / frame_path.name))
            sorted_count += 1
            print(f"  → {key_map[key]} ({sorted_count} sorted)")

    cv2.destroyAllWindows()
    print(f"\nDone! Sorted: {sorted_count}, Trashed: {trashed_count}, "
          f"Skipped: {skipped_count}")


def cmd_split(args):
    """Split into train/val."""
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            continue

        random.shuffle(images)
        val_count = max(1, int(len(images) * args.val_ratio))

        (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

        for img in images[val_count:]:
            shutil.copy2(str(img), str(train_dir / class_dir.name / img.name))
        for img in images[:val_count]:
            shutil.copy2(str(img), str(val_dir / class_dir.name / img.name))

        print(f"  {class_dir.name}: {len(images) - val_count} train, {val_count} val")

    print(f"\nSplit complete -> {DATASET_DIR}")


def cmd_train(args):
    """Train from scratch or fine-tune."""
    from ultralytics import YOLO

    train_dir = DATASET_DIR / "train"
    if not train_dir.exists():
        print("No train/ directory. Run 'split' first.")
        sys.exit(1)

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"Training classes ({len(classes)}): {classes}")

    # Start from a pretrained classification backbone (not your old show's model)
    base_model = args.base or "yolov8n-cls.pt"
    model = YOLO(base_model)

    model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(PROJECT_ROOT / "yolo_wt"),
        name="ben10_coldstart",
        exist_ok=True,
    )

    print(f"\nTraining complete!")
    print(f"Weights saved to: {PROJECT_ROOT / 'yolo_wt' / 'ben10_coldstart'}")
    print(f"\nNext: use yolo_dataset_builder.py with these weights for the assisted loop.")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap a YOLO classification dataset from scratch."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- extract (speaker-label based) ---
    p_ext = sub.add_parser("extract",
        help="Extract frames using SRT speaker labels as weak labels")
    p_ext.add_argument("--show", default=None)
    p_ext.add_argument("--episodes", nargs="+", default=None,
        help="Only process specific episodes")

    # --- extract-unlabeled ---
    p_unl = sub.add_parser("extract-unlabeled",
        help="Extract uniform-interval frames for manual sorting")
    p_unl.add_argument("--show", default=None)
    p_unl.add_argument("--episodes", nargs="+", default=None)
    p_unl.add_argument("--interval", type=float, default=3.0,
        help="Seconds between frame extractions (default: 3.0)")

    # --- sort ---
    sub.add_parser("sort",
        help="Interactive keyboard sorter for unsorted frames")

    # --- split ---
    p_split = sub.add_parser("split",
        help="Split reviewed dataset into train/val")
    p_split.add_argument("--val-ratio", type=float, default=0.15)

    # --- train ---
    p_train = sub.add_parser("train",
        help="Train YOLO classifier on the dataset")
    p_train.add_argument("--base", default=None,
        help="Base model to fine-tune from (default: yolov8n-cls.pt)")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--imgsz", type=int, default=224)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()
    {
        "extract": cmd_extract,
        "extract-unlabeled": cmd_extract_unlabeled,
        "sort": cmd_sort,
        "split": cmd_split,
        "train": cmd_train,
    }[args.command](args)


if __name__ == "__main__":
    main()
