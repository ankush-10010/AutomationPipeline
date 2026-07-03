"""
YOLO_hybrid_inference.py — Production hybrid inference for clip_index.json.

Combines:
  1. YOLO classifier (10 core classes, 99.2% accuracy, processes ALL frames)
  2. CLIP k-NN (20 rare classes, processes sampled frames with crops)

Architecture:
  - YOLO runs first on every frame of each clip using the proven multi-frame
    aggregation logic from YOLO_inference_noBoundingBox.py. Characters are
    detected if they appear clearly in at least one frame (Rule 1) or
    in a sustained run of consecutive frames (Rule 2).
  - k-NN runs second on 3-5 sampled frames. Only predictions for classes
    that YOLO was NOT trained on are kept. YOLO's judgment is final for
    its 10 classes.
  - Results are merged into a deduplicated set in 'visual_characters'.
  - The old text-based 'characters' field is replaced with visual detections.

Usage:
    # First time: build the k-NN reference database
    python scripts/clip_classifier_knn.py build-ref --dataset "Ready Dataset"

    # Run hybrid inference on all clips
    python scripts/YOLO_hybrid_inference.py --weights yolo_wt/best.pt

    # Run on specific episode only
    python scripts/YOLO_hybrid_inference.py --weights yolo_wt/best.pt --episode s1e1

    # Force re-process already tagged clips
    python scripts/YOLO_hybrid_inference.py --weights yolo_wt/best.pt --force

    # Skip k-NN (YOLO only, much faster)
    python scripts/YOLO_hybrid_inference.py --weights yolo_wt/best.pt --yolo-only
"""

import argparse
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config, get_active_show, get_project_path,
    load_json, save_json, setup_logging, PROJECT_ROOT,
)

log = setup_logging("hybrid_inference")

REF_DB_PATH = PROJECT_ROOT / "yolo_dataset" / "clip_reference_embeddings.npz"

# ── YOLO Aggregation Thresholds ──────────────────────────────────────────────
#
# IMPORTANT: YOLO classification assigns probabilities to ALL 10 classes for
# every single frame — even landscape shots, civilians, or title cards where
# zero trained characters are on screen. In those frames, confidence spreads
# thinly across all classes (e.g., 0.15 Four Arms, 0.12 Ditto, 0.10 Ben...).
# Over hundreds of frames this noise accumulates and triggers false positives.
#
# The fix: FRAME_SKIP_THRESHOLD. If YOLO's TOP-1 confidence in a frame is
# below this value, the entire frame is discarded as "no known character on
# screen." This is the single most important threshold in the whole pipeline.

# If YOLO's best guess for ANY class is below this, skip the frame entirely.
# This rejects backgrounds, civilians, title cards, etc.
FRAME_SKIP_THRESHOLD = 0.50

# Rule 1: Character appeared very clearly in at least one frame
CLEAR_APPEARANCE_THRESHOLD = 0.90

# Rule 2: Character appeared in a sustained consecutive run of frames
MIN_CONSECUTIVE_FRAMES = 5     # Must appear in ≥5 consecutive valid frames
MIN_MAX_CONF_FOR_RATIO = 0.60  # Peak confidence must be at least 60%

# Minimum per-frame confidence to count toward a consecutive run
FRAME_PRESENCE_THRESHOLD = 0.30

# ── k-NN Thresholds ─────────────────────────────────────────────────────────
KNN_MIN_CONFIDENCE = 0.80  # 80% of k neighbors must agree
KNN_MIN_SIMILARITY = 0.30  # Minimum cosine similarity


# ── YOLO: Process ALL frames (multi-character) ──────────────────────────────

def yolo_classify_clip(model, video_path: Path) -> tuple:
    """Run YOLO classification on every frame and aggregate.

    Key design: frames where YOLO's top-1 confidence is below
    FRAME_SKIP_THRESHOLD are discarded entirely. This prevents
    background/civilian frames from polluting character counts.

    Returns (detected_characters: list, stats: dict).
    """
    results = model.predict(source=str(video_path), stream=True, verbose=False)

    # Per-character: list of per-frame confidences aligned with valid_frame_idx
    char_confidences = defaultdict(list)
    total_frames = 0
    valid_frames = 0

    for result in results:
        total_frames += 1
        names_dict = result.names
        probs = result.probs.data.tolist()

        # Gate: Is there ANY character on screen with reasonable confidence?
        top1_conf = max(probs)
        if top1_conf < FRAME_SKIP_THRESHOLD:
            # No known character visible — skip this frame entirely
            continue

        valid_frames += 1
        for class_id, conf in enumerate(probs):
            char_name = names_dict[class_id]
            char_confidences[char_name].append(conf)

    if valid_frames == 0:
        return [], {"_total_frames": total_frames, "_valid_frames": 0}

    stats = {"_total_frames": total_frames, "_valid_frames": valid_frames}
    present_characters = []

    for char_name, confs in char_confidences.items():
        max_conf = max(confs)
        avg_conf = sum(confs) / len(confs)
        frames_above = sum(1 for c in confs if c > FRAME_PRESENCE_THRESHOLD)

        # Compute longest consecutive run of frames above presence threshold
        longest_run = 0
        current_run = 0
        for c in confs:
            if c > FRAME_PRESENCE_THRESHOLD:
                current_run += 1
                if current_run > longest_run:
                    longest_run = current_run
            else:
                current_run = 0

        stats[char_name] = {
            "max_confidence": round(max_conf, 4),
            "avg_confidence": round(avg_conf, 4),
            "frames_detected": frames_above,
            "valid_frames": valid_frames,
            "longest_consecutive_run": longest_run,
        }

        # Rule 1: Very clear appearance in at least one frame
        if max_conf >= CLEAR_APPEARANCE_THRESHOLD:
            present_characters.append(char_name)
        # Rule 2: Sustained consecutive presence (rejects scattered noise)
        elif (longest_run >= MIN_CONSECUTIVE_FRAMES
              and max_conf >= MIN_MAX_CONF_FOR_RATIO):
            present_characters.append(char_name)

    return present_characters, stats


# ── k-NN: Process sampled frames for rare classes ───────────────────────────

def extract_sample_frames(video_path: Path, n_frames: int = 5) -> list:
    """Extract N evenly-spaced frames as numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    start = max(1, int(total * 0.1))
    end = max(start + 1, int(total * 0.9))
    indices = np.linspace(start, end, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def grid_crop(frame: np.ndarray) -> list:
    """Split frame into whole + left/right halves + center crop.

    Returns list of PIL Images. Lighter than a full 3x3 grid but
    good enough to catch 2-3 characters in a scene.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crops = [Image.fromarray(rgb)]  # Whole frame

    # Left and right halves with overlap
    mid = w // 2
    overlap = w // 8
    left = frame[:, :mid + overlap]
    right = frame[:, mid - overlap:]
    crops.append(Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)))
    crops.append(Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB)))

    # Center crop (catches characters in conversation)
    cx, cy = w // 4, h // 4
    center = frame[cy:h - cy, cx:w - cx]
    if center.shape[0] > 30 and center.shape[1] > 30:
        crops.append(Image.fromarray(cv2.cvtColor(center, cv2.COLOR_BGR2RGB)))

    return crops


def knn_classify_crops(
    crops: list,
    clip_model,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    class_names: list,
    yolo_classes: set,
    k: int = 7,
) -> list:
    """Classify crops via k-NN. Only return characters NOT in yolo_classes."""
    if not crops:
        return []

    embeddings = clip_model.encode(crops, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Track votes across all crops for each class
    class_evidence = defaultdict(lambda: {"best_conf": 0.0, "best_sim": 0.0})

    for emb in embeddings:
        similarities = ref_embeddings @ emb
        top_k_idx = np.argpartition(similarities, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
        votes = ref_labels[top_k_idx]
        vote_counts = Counter(votes)

        for cls_idx, count in vote_counts.most_common():
            conf = count / k
            sim = float(np.mean(similarities[top_k_idx[votes == cls_idx]]))
            name = class_names[cls_idx]

            # Skip classes that YOLO already handles
            if name in yolo_classes:
                continue

            class_evidence[name]["best_conf"] = max(
                class_evidence[name]["best_conf"], conf
            )
            class_evidence[name]["best_sim"] = max(
                class_evidence[name]["best_sim"], sim
            )

    # Filter by thresholds
    detected = []
    for name, evidence in class_evidence.items():
        if (evidence["best_conf"] >= KNN_MIN_CONFIDENCE and
                evidence["best_sim"] >= KNN_MIN_SIMILARITY):
            detected.append(name)

    return detected


# ── Find video file ──────────────────────────────────────────────────────────

def find_video(filename: str, clips_dir: Path) -> Path:
    """Try multiple paths to locate a clip video file."""
    direct = clips_dir / filename
    if direct.exists():
        return direct

    # Try episode subfolder
    ep_match = re.match(r"(s\d+e\d+)", filename, re.IGNORECASE)
    if ep_match:
        sub = clips_dir / ep_match.group(1) / filename
        if sub.exists():
            return sub

    # Recursive search (slower, last resort)
    found = list(clips_dir.rglob(filename))
    if found:
        return found[0]

    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid YOLO + k-NN character tagging for clip_index.json"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to YOLO best.pt")
    parser.add_argument("--show", default=None)
    parser.add_argument("--episode", default=None,
                        help="Only process clips from this episode (e.g. s1e1)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Save checkpoint every N clips")
    parser.add_argument("--force", action="store_true",
                        help="Re-process clips that already have visual_characters")
    parser.add_argument("--yolo-only", action="store_true",
                        help="Skip k-NN, only use YOLO (much faster)")
    parser.add_argument("--knn-frames", type=int, default=3,
                        help="Number of frames to sample for k-NN (default: 3)")
    parser.add_argument("--k", type=int, default=7,
                        help="k for k-NN voting (default: 7)")
    args = parser.parse_args()

    # ── Load YOLO ────────────────────────────────────────────────────────
    from ultralytics import YOLO
    log.info("Loading YOLO model: %s", args.weights)
    yolo_model = YOLO(args.weights)

    # Get YOLO's trained class names as a set
    yolo_class_names = set(yolo_model.names.values())
    log.info("YOLO classes (%d): %s", len(yolo_class_names), sorted(yolo_class_names))

    # ── Load k-NN (optional) ─────────────────────────────────────────────
    clip_model = None
    ref_emb = None
    ref_lbl = None
    knn_classes = None

    if not args.yolo_only:
        if not REF_DB_PATH.exists():
            log.error(
                "k-NN reference DB not found at %s. "
                "Run: python scripts/clip_classifier_knn.py build-ref --dataset \"Ready Dataset\"",
                REF_DB_PATH,
            )
            log.info("Continuing with YOLO-only mode.")
            args.yolo_only = True
        else:
            log.info("Loading CLIP model and k-NN reference DB...")
            from sentence_transformers import SentenceTransformer
            clip_model = SentenceTransformer("clip-ViT-B-32")

            ref_data = np.load(str(REF_DB_PATH), allow_pickle=True)
            ref_emb = ref_data["embeddings"]
            ref_lbl = ref_data["labels"]
            knn_classes = list(ref_data["class_names"])

            knn_only_classes = set(knn_classes) - yolo_class_names
            log.info("k-NN rare classes (%d): %s",
                     len(knn_only_classes), sorted(knn_only_classes))

    # ── Load clip index ──────────────────────────────────────────────────
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)
    clip_index_path = get_project_path("clip_index", pipeline_cfg)

    clips_dir = Path(
        show_config.get("clips_dir", f"./clips/{show_slug}")
    )
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / clips_dir).resolve()

    clip_data = load_json(clip_index_path)
    clips = clip_data.get("clips", clip_data) if isinstance(clip_data, dict) else clip_data
    log.info("Loaded %d clips from %s", len(clips), clip_index_path)

    # ── Filter clips to process ──────────────────────────────────────────
    indices = list(range(len(clips)))

    if args.episode:
        ep = args.episode.lower()
        indices = [i for i in indices
                   if clips[i].get("filename", "").lower().startswith(ep)]

    if not args.force:
        indices = [i for i in indices
                   if "visual_characters" not in clips[i]]

    log.info("Clips to process: %d (episode=%s, force=%s, yolo_only=%s)",
             len(indices), args.episode, args.force, args.yolo_only)

    # ── Process ──────────────────────────────────────────────────────────
    processed = 0
    skipped = 0
    char_counter = Counter()

    for batch_start in range(0, len(indices), args.batch_size):
        batch = indices[batch_start:batch_start + args.batch_size]

        for idx in batch:
            clip = clips[idx]
            filename = clip.get("filename", "")

            video_path = find_video(filename, clips_dir)
            if video_path is None:
                skipped += 1
                continue

            # ── Step 1: YOLO (all frames, multi-character) ───────────
            yolo_chars, yolo_stats = yolo_classify_clip(yolo_model, video_path)

            # ── Step 2: k-NN (sampled frames, rare classes only) ─────
            knn_chars = []
            if not args.yolo_only and clip_model is not None:
                frames = extract_sample_frames(video_path, n_frames=args.knn_frames)
                all_crops = []
                for frame in frames:
                    all_crops.extend(grid_crop(frame))

                if all_crops:
                    knn_chars = knn_classify_crops(
                        all_crops, clip_model,
                        ref_emb, ref_lbl, knn_classes,
                        yolo_classes=yolo_class_names,
                        k=args.k,
                    )

            # ── Step 3: Merge & deduplicate ──────────────────────────
            all_detected = set(yolo_chars) | set(knn_chars)

            clip["visual_characters"] = sorted(list(all_detected))
            clip["yolo_detections"] = yolo_stats
            clip["characters"] = sorted(list(all_detected))

            for char in all_detected:
                char_counter[char] += 1

            processed += 1

        # ── Checkpoint save ──────────────────────────────────────────
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(clip_index_path, clip_data)
        log.info(
            "Checkpoint: %d/%d processed, %d skipped (batch %d/%d)",
            processed, skipped,
            batch_start // args.batch_size + 1,
            (len(indices) + args.batch_size - 1) // args.batch_size,
        )

    # ── Final save ───────────────────────────────────────────────────────
    if isinstance(clip_data, dict):
        clip_data["clips"] = clips
    save_json(clip_index_path, clip_data)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HYBRID INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Processed: {processed}")
    print(f"  Skipped (video not found): {skipped}")
    print(f"\nCharacter detections across all clips:")
    for char, count in char_counter.most_common():
        source = "YOLO" if char in yolo_class_names else "k-NN"
        print(f"  {char:20s}  {count:5d} clips  ({source})")


if __name__ == "__main__":
    main()
