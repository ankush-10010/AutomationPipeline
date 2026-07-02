"""
clip_classifier_knn.py — Classify clips using CLIP embeddings + k-NN.

Handles MULTI-CHARACTER scenes by detecting individual regions in each frame,
cropping them, and classifying each crop independently.

Three classification strategies (from fastest to most accurate):
  1. WHOLE-FRAME:   Classify the full frame as one embedding (fast, single-char only)
  2. GRID-CROP:     Split frame into overlapping regions, classify each (no detector needed)
  3. DETECT+CROP:   Use YOLO to find character-like regions, classify each crop (best accuracy)

How it works:
  1. Build reference DB: Compute CLIP embeddings for every frame in the labeled dataset.
  2. Classify clips: For each clip, extract frames → detect/crop character regions →
     compute CLIP embedding per crop → k-NN against reference DB → union all predictions.
  3. Each clip gets: knn_characters (list), knn_predictions (per-character confidence).

Dependencies:
    pip install sentence-transformers numpy Pillow opencv-python-headless
    pip install ultralytics  (only for detect+crop mode)

Usage:
    # Step 1: Build reference embeddings from labeled dataset (one-time)
    python scripts/clip_classifier_knn.py build-ref --dataset "Ready Dataset"

    # Step 2a: Classify using grid crops (no detector needed, handles multi-char)
    python scripts/clip_classifier_knn.py classify --mode grid --k 7

    # Step 2b: Classify using YOLO person detection + crop (best for multi-char)
    python scripts/clip_classifier_knn.py classify --mode detect --k 7

    # Step 2c: Classify whole frames only (fastest, single-char)
    python scripts/clip_classifier_knn.py classify --mode whole --k 5

    # Step 3: Check results
    python scripts/clip_classifier_knn.py stats
"""

import argparse
import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

import cv2
from PIL import Image

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

log = setup_logging("clip_knn")

REF_DB_PATH = PROJECT_ROOT / "yolo_dataset" / "clip_reference_embeddings.npz"


# ── Build Reference Database ────────────────────────────────────────────────

def cmd_build_ref(args):
    """Compute CLIP embeddings for all labeled frames and save as reference DB."""
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / args.dataset

    if not dataset_dir.exists():
        log.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    classes = []
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith("."):
            classes.append(d)

    log.info("Found %d character classes", len(classes))

    log.info("Loading CLIP model (clip-ViT-B-32)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("clip-ViT-B-32")
    log.info("CLIP model loaded.")

    all_embeddings = []
    all_labels = []
    class_names = []

    for class_dir in classes:
        images = (
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.webp"))
        )
        if not images:
            log.warning("No images in %s, skipping", class_dir.name)
            continue

        class_names.append(class_dir.name)
        class_idx = len(class_names) - 1
        log.info("  %s: %d images", class_dir.name, len(images))

        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch_paths = images[i:i + batch_size]
            batch_images = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                except Exception:
                    pass

            if batch_images:
                embeddings = model.encode(batch_images, show_progress_bar=False)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms
                for emb in embeddings:
                    all_embeddings.append(emb)
                    all_labels.append(class_idx)

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    REF_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(REF_DB_PATH),
        embeddings=all_embeddings,
        labels=all_labels,
        class_names=np.array(class_names),
    )

    log.info("Reference database saved: %s", REF_DB_PATH)
    print(f"\nReference database: {REF_DB_PATH}")
    print(f"Total embeddings: {len(all_embeddings)}\n")
    for idx, name in enumerate(class_names):
        count = np.sum(all_labels == idx)
        print(f"  {name:20s}  {count:5d} references")


# ── k-NN Core ────────────────────────────────────────────────────────────────

def knn_classify(
    query_embedding: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    class_names: list,
    k: int = 5,
) -> list:
    """Classify a single embedding via k-NN.

    Returns list of (class_name, vote_fraction, avg_similarity) sorted by votes.
    """
    similarities = ref_embeddings @ query_embedding
    top_k_idx = np.argpartition(similarities, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]

    votes = ref_labels[top_k_idx]
    vote_counts = Counter(votes)

    results = []
    for cls_idx, count in vote_counts.most_common():
        avg_sim = float(np.mean(similarities[top_k_idx[votes == cls_idx]]))
        results.append((class_names[cls_idx], count / k, avg_sim))
    return results


def knn_classify_multi(
    query_embeddings: list,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    class_names: list,
    k: int = 5,
    char_threshold: float = 0.3,
    sim_threshold: float = 0.15,
) -> dict:
    """Classify multiple crop embeddings and merge predictions.

    Each crop is classified independently. A character is included in the
    final prediction if it wins at least one crop with sufficient confidence,
    OR if it appears as a runner-up across multiple crops.

    Returns {character_name: {"confidence": float, "avg_similarity": float, "crop_votes": int}}
    """
    char_evidence = defaultdict(lambda: {"best_conf": 0.0, "best_sim": 0.0, "crop_wins": 0, "crop_appearances": 0})

    for query_emb in query_embeddings:
        results = knn_classify(query_emb, ref_embeddings, ref_labels, class_names, k)
        if not results:
            continue

        # The winner of this crop
        winner_name, winner_conf, winner_sim = results[0]
        char_evidence[winner_name]["crop_wins"] += 1
        char_evidence[winner_name]["best_conf"] = max(
            char_evidence[winner_name]["best_conf"], winner_conf
        )
        char_evidence[winner_name]["best_sim"] = max(
            char_evidence[winner_name]["best_sim"], winner_sim
        )

        # Track all appearances (even as runner-up)
        for name, conf, sim in results:
            if conf >= 0.15 and sim >= sim_threshold:
                char_evidence[name]["crop_appearances"] += 1
                char_evidence[name]["best_sim"] = max(
                    char_evidence[name]["best_sim"], sim
                )

    # Filter: keep characters that won at least one crop OR appeared in many
    final = {}
    n_crops = len(query_embeddings)
    for char_name, evidence in char_evidence.items():
        # Include if:
        # a) Won at least one crop with sufficient confidence
        # b) Appeared as runner-up in many crops (suggests presence even if not dominant)
        if (evidence["crop_wins"] >= 1 and evidence["best_conf"] >= char_threshold) or \
           (evidence["crop_appearances"] >= max(2, n_crops * 0.4)):
            final[char_name] = {
                "confidence": round(evidence["best_conf"], 3),
                "avg_similarity": round(evidence["best_sim"], 4),
                "crop_wins": evidence["crop_wins"],
                "crop_appearances": evidence["crop_appearances"],
            }

    return final


# ── Frame Extraction & Cropping ──────────────────────────────────────────────

def extract_frames(video_path: Path, n_frames: int = 5) -> list:
    """Extract N evenly-spaced frames from a video. Returns list of numpy arrays (BGR)."""
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


def grid_crop(frame: np.ndarray, grid_size: int = 3, overlap: float = 0.25) -> list:
    """Split a frame into overlapping grid regions.

    Returns list of PIL Images (crops).
    With grid_size=3 and overlap=0.25, produces 9 overlapping crops plus
    the whole frame = 10 embeddings per frame.
    """
    h, w = frame.shape[:2]
    crops = []

    # Whole frame first
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crops.append(Image.fromarray(rgb))

    # Grid crops
    cell_h = h // grid_size
    cell_w = w // grid_size
    pad_h = int(cell_h * overlap)
    pad_w = int(cell_w * overlap)

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = max(0, row * cell_h - pad_h)
            y2 = min(h, (row + 1) * cell_h + pad_h)
            x1 = max(0, col * cell_w - pad_w)
            x2 = min(w, (col + 1) * cell_w + pad_w)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] > 20 and crop.shape[1] > 20:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(Image.fromarray(rgb_crop))

    return crops


def detect_and_crop(frame: np.ndarray, detector, conf: float = 0.25) -> list:
    """Use YOLO to detect character-like regions, return crops as PIL Images.

    Falls back to grid crops if no detections are found.
    """
    crops = []

    # Always include whole frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crops.append(Image.fromarray(rgb))

    h, w = frame.shape[:2]

    try:
        results = detector.predict(source=frame, verbose=False, conf=conf)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                cls_name = results[0].names[cls_id]
                # Accept person-like detections
                # COCO: 0=person. Also accept any detection for cartoon content.
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                # Pad the crop slightly
                pad = int(max(x2 - x1, y2 - y1) * 0.1)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 30 and crop.shape[1] > 30:
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crops.append(Image.fromarray(rgb_crop))
    except Exception as e:
        log.debug("Detection error: %s", e)

    # If no detections, fall back to basic spatial splits
    if len(crops) <= 1:
        # Left half, right half, center
        mid_w = w // 2
        for (x1, x2) in [(0, mid_w + w // 6), (mid_w - w // 6, w)]:
            crop = frame[:, x1:x2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb_crop))

    return crops


# ── Classify Command ─────────────────────────────────────────────────────────

def cmd_classify(args):
    """Classify clips: extract frames, crop regions, k-NN each crop, merge."""
    if not REF_DB_PATH.exists():
        log.error("Reference database not found. Run 'build-ref' first.")
        sys.exit(1)

    # Load reference database
    log.info("Loading reference database...")
    ref_data = np.load(str(REF_DB_PATH), allow_pickle=True)
    ref_embeddings = ref_data["embeddings"]
    ref_labels = ref_data["labels"]
    class_names = list(ref_data["class_names"])
    log.info("  %d references, %d classes", len(ref_embeddings), len(class_names))

    # Load CLIP model for encoding crops
    log.info("Loading CLIP model...")
    from sentence_transformers import SentenceTransformer
    clip_model = SentenceTransformer("clip-ViT-B-32")

    # Load YOLO detector if needed
    detector = None
    if args.mode == "detect":
        from ultralytics import YOLO
        det_weights = args.detector or "yolov8n.pt"
        detector = YOLO(det_weights)
        log.info("Loaded YOLO detector: %s", det_weights)

    # Load clip index
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

    clip_data = load_json(clip_index_path)
    clips = clip_data.get("clips", clip_data) if isinstance(clip_data, dict) else clip_data
    log.info("Loaded %d clips", len(clips))

    # Filter
    indices = list(range(len(clips)))
    if args.target_dir:
        target = args.target_dir.lower()
        indices = [i for i, c in enumerate(clips)
                   if c.get("filename", "").lower().startswith(target)]
    if not args.force:
        indices = [i for i in indices if not clips[i].get("knn_characters")]
    log.info("Clips to process: %d (mode=%s)", len(indices), args.mode)

    classified = 0
    skipped = 0
    class_counts = defaultdict(int)

    for batch_start in range(0, len(indices), args.batch_size):
        batch = indices[batch_start:batch_start + args.batch_size]

        for idx in batch:
            clip = clips[idx]
            filename = clip.get("filename", "")

            # Find video file
            video_path = clips_dir / filename
            if not video_path.exists():
                ep_match = re.match(r"(s\d+e\d+)", filename, re.IGNORECASE)
                if ep_match:
                    video_path = clips_dir / ep_match.group(1) / filename
                if not video_path.exists():
                    found = list(clips_dir.rglob(filename))
                    video_path = found[0] if found else None

            if video_path is None or not video_path.exists():
                skipped += 1
                continue

            # Extract frames
            frames = extract_frames(video_path, n_frames=args.n_frames)
            if not frames:
                skipped += 1
                continue

            # Get crops from all frames
            all_crops = []
            for frame in frames:
                if args.mode == "grid":
                    crops = grid_crop(frame, grid_size=3, overlap=0.25)
                elif args.mode == "detect":
                    crops = detect_and_crop(frame, detector, conf=args.det_conf)
                else:  # whole
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    crops = [Image.fromarray(rgb)]
                all_crops.extend(crops)

            # Encode all crops through CLIP
            crop_embeddings = clip_model.encode(all_crops, show_progress_bar=False)
            norms = np.linalg.norm(crop_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            crop_embeddings = crop_embeddings / norms

            # k-NN classify all crops and merge
            predictions = knn_classify_multi(
                list(crop_embeddings),
                ref_embeddings, ref_labels, class_names,
                k=args.k,
                char_threshold=args.min_confidence,
                sim_threshold=args.min_similarity,
            )

            if predictions:
                # Sort by confidence
                sorted_preds = sorted(predictions.items(),
                                      key=lambda x: x[1]["confidence"], reverse=True)

                clip["knn_characters"] = [name for name, _ in sorted_preds]
                clip["knn_predictions"] = {
                    name: info for name, info in sorted_preds
                }

                # Update the main characters list
                existing = set(clip.get("characters", []))
                for name, info in sorted_preds:
                    if name not in existing:
                        existing.add(name)
                clip["characters"] = sorted(list(existing))

                for name in clip["knn_characters"]:
                    class_counts[name] += 1

                classified += 1
            else:
                clip["knn_characters"] = []
                clip["knn_predictions"] = {}
                skipped += 1

        # Checkpoint
        log.info(
            "Progress: %d classified, %d skipped (batch %d/%d)",
            classified, skipped,
            batch_start // args.batch_size + 1,
            (len(indices) + args.batch_size - 1) // args.batch_size,
        )
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(clip_index_path, clip_data)

    log.info("Classification complete: %d classified, %d skipped", classified, skipped)
    print(f"\nResults (mode={args.mode}, k={args.k}):")
    print(f"  Classified: {classified}")
    print(f"  Skipped: {skipped}")
    print(f"\nPer-character detections:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s}  {count:5d} clips")


# ── Stats Command ────────────────────────────────────────────────────────────

def cmd_stats(args):
    """Show classification statistics."""
    pipeline_cfg = load_pipeline_config()
    clip_index_path = (
        Path(args.index) if args.index
        else get_project_path("clip_index", pipeline_cfg)
    )
    clip_data = load_json(clip_index_path)
    clips = clip_data.get("clips", clip_data) if isinstance(clip_data, dict) else clip_data

    has_knn = [c for c in clips if c.get("knn_characters")]
    print(f"Total clips: {len(clips)}")
    print(f"With knn_characters: {len(has_knn)}")

    if not has_knn:
        return

    # Characters per clip distribution
    char_counts_per_clip = [len(c["knn_characters"]) for c in has_knn]
    print(f"\nCharacters per clip:")
    print(f"  1 character:  {sum(1 for x in char_counts_per_clip if x == 1)}")
    print(f"  2 characters: {sum(1 for x in char_counts_per_clip if x == 2)}")
    print(f"  3+ characters: {sum(1 for x in char_counts_per_clip if x >= 3)}")
    print(f"  Average: {np.mean(char_counts_per_clip):.1f}")

    # Per-class stats
    class_total = defaultdict(int)
    class_confs = defaultdict(list)
    for c in has_knn:
        preds = c.get("knn_predictions", {})
        for name, info in preds.items():
            class_total[name] += 1
            class_confs[name].append(info.get("confidence", 0))

    print(f"\nPer-class breakdown:")
    for cls in sorted(class_total, key=lambda x: -class_total[x]):
        avg_conf = np.mean(class_confs[cls])
        print(f"  {cls:20s}  {class_total[cls]:5d} clips  avg_conf={avg_conf:.3f}")

    # Multi-character scene examples
    multi = [c for c in has_knn if len(c["knn_characters"]) >= 2]
    if multi:
        print(f"\nSample multi-character scenes:")
        for c in multi[:8]:
            chars = ", ".join(c["knn_characters"])
            print(f"  {c['filename']:35s}  [{chars}]")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-character clip classification using CLIP k-NN."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-ref
    p_build = sub.add_parser("build-ref",
        help="Build reference embeddings from labeled dataset")
    p_build.add_argument("--dataset", required=True,
        help="Path to labeled dataset (e.g. 'Ready Dataset')")

    # classify
    p_cls = sub.add_parser("classify",
        help="Classify clips using k-NN with multi-character support")
    p_cls.add_argument("--index", default=None)
    p_cls.add_argument("--show", default=None)
    p_cls.add_argument("--mode", choices=["whole", "grid", "detect"], default="grid",
        help="Cropping strategy: 'whole' (single-char), 'grid' (multi-char, no detector), "
             "'detect' (multi-char, uses YOLO detector)")
    p_cls.add_argument("--detector", default=None,
        help="YOLO weights for detect mode (default: yolov8n.pt)")
    p_cls.add_argument("--det-conf", type=float, default=0.25,
        help="Detection confidence threshold")
    p_cls.add_argument("--k", type=int, default=7,
        help="k for k-NN (default: 7)")
    p_cls.add_argument("--min-confidence", type=float, default=0.3,
        help="Minimum vote fraction to include a character")
    p_cls.add_argument("--min-similarity", type=float, default=0.15,
        help="Minimum cosine similarity to count as a match")
    p_cls.add_argument("--n-frames", type=int, default=5,
        help="Frames to sample per clip")
    p_cls.add_argument("--target-dir", default=None,
        help="Only process clips matching prefix (e.g. s1e1)")
    p_cls.add_argument("--batch-size", type=int, default=50,
        help="Save checkpoint every N clips")
    p_cls.add_argument("--force", action="store_true",
        help="Re-classify even if knn_characters already exists")

    # stats
    p_stats = sub.add_parser("stats", help="Show classification statistics")
    p_stats.add_argument("--index", default=None)

    args = parser.parse_args()
    {"build-ref": cmd_build_ref, "classify": cmd_classify, "stats": cmd_stats}[args.command](args)


if __name__ == "__main__":
    main()
