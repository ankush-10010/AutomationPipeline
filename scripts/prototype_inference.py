"""
prototype_inference.py — Metric Learning Inference for clip_index.json
=====================================================================

Replaces the hybrid YOLO + k-NN approach with a single unified pipeline:
  1. Encode sampled video frames through frozen CLIP ViT-B-32.
  2. Project through the ArcFace-trained projection head.
  3. Compare each frame's embedding against all per-class prototypes.
  4. If best similarity > threshold τ, tag that character for that frame.
  5. Aggregate across frames using temporal contiguity (consecutive runs).
  6. Write deduplicated character lists to clip_index.json.

Advantages over YOLO classifier:
  - Covers ALL 30 character classes, not just 10.
  - Open-set rejection: frames far from every prototype are discarded as
    "unknown/background" — no forced softmax distribution, no noise.
  - Few-shot friendly: 6 images of Eye Guy work just as well as 1,600 of Ben,
    because prototypes are points in space, not softmax decision boundaries.
  - Multi-label natively: each character is evaluated independently via its
    own distance threshold, so Ben and Gwen in the same frame both register
    without competing for probability mass.

Requirements:
  - arcface_head.pt     (projection head weights from arcface_metric_train.py)
  - prototypes.npz      (per-class prototypes from arcface_metric_train.py)
  - sentence-transformers (for CLIP ViT-B-32)

Usage:
    python scripts/prototype_inference.py

    # Test on one episode first
    python scripts/prototype_inference.py --episode s1e1

    # Force re-process, adjust thresholds
    python scripts/prototype_inference.py --force --tau 0.55 --min-run 4

    # Process more frames per clip (slower but more accurate)
    python scripts/prototype_inference.py --n-frames 8
"""

import argparse
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config, get_active_show, get_project_path,
    load_json, save_json, setup_logging, PROJECT_ROOT,
)

log = setup_logging("prototype_inference")


# ═══════════════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_HEAD_PATH = PROJECT_ROOT / "arcface_head.pt"
DEFAULT_PROTO_PATH = PROJECT_ROOT / "prototypes.npz"

# Rejection threshold: frame must have cosine similarity > τ to the closest
# prototype to be considered a match. Below this → "no known character."
DEFAULT_TAU = 0.50

# A character is only tagged if it appears in ≥ MIN_CONSECUTIVE_FRAMES
# consecutive sampled frames. Prevents scattered single-frame noise.
DEFAULT_MIN_CONSECUTIVE = 3

# How many frames to sample per clip. More = slower but more reliable.
DEFAULT_N_FRAMES = 6


# ═══════════════════════════════════════════════════════════════════════════════
# Projection Head (must match architecture from arcface_metric_train.py)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Frame Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: Path, n_frames: int = 6) -> list:
    """Extract N evenly-spaced frames from a video clip.

    Avoids the first/last 10% (title cards, transitions).
    Returns list of PIL Images (RGB).
    """
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


def make_crops(pil_image: Image.Image) -> list:
    """Split a frame into whole + left/right halves for multi-character.

    Returns list of PIL Images. Each crop is scored independently so
    characters on different sides of the frame don't dilute each other.
    """
    w, h = pil_image.size
    crops = [pil_image]  # Whole frame

    mid = w // 2
    overlap = w // 8  # Slight overlap so characters near center aren't cut
    left = pil_image.crop((0, 0, mid + overlap, h))
    right = pil_image.crop((mid - overlap, 0, w, h))
    crops.append(left)
    crops.append(right)

    return crops


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame matching
# ═══════════════════════════════════════════════════════════════════════════════

def match_frame_to_prototypes(
    frame_embedding: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    class_names: list,
    tau: float,
) -> dict:
    """Compare a single frame/crop embedding against all prototypes.

    Returns dict of {class_name: best_similarity} for all classes above τ.
    Multi-label: every class is evaluated independently.
    """
    # Cosine similarity against all prototypes
    similarities = prototypes @ frame_embedding  # (P,)

    # Group by class (a class may have multiple sub-prototypes)
    class_best = {}
    for i, sim in enumerate(similarities):
        cls_idx = prototype_labels[i]
        name = class_names[cls_idx]
        if name not in class_best or sim > class_best[name]:
            class_best[name] = float(sim)

    # Filter by threshold
    return {name: sim for name, sim in class_best.items() if sim >= tau}


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal contiguity aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_with_contiguity(
    per_frame_detections: list,
    min_consecutive: int,
) -> dict:
    """Aggregate per-frame character detections using consecutive-run logic.

    Args:
        per_frame_detections: list of dicts, one per sampled frame.
            Each dict maps character_name → best_similarity for that frame.
        min_consecutive: minimum consecutive frames to tag a character.

    Returns:
        dict of {character_name: {"max_sim": float, "longest_run": int, "tagged": bool}}
    """
    # Collect all character names that appeared in any frame
    all_chars = set()
    for frame_det in per_frame_detections:
        all_chars.update(frame_det.keys())

    results = {}
    for char in all_chars:
        sims = [frame_det.get(char, 0.0) for frame_det in per_frame_detections]
        max_sim = max(sims)

        # Compute longest consecutive run of positive detections
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


# ═══════════════════════════════════════════════════════════════════════════════
# Video file lookup
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prototype-based character tagging for clip_index.json"
    )
    parser.add_argument("--head", type=str, default=str(DEFAULT_HEAD_PATH),
                        help="Path to arcface_head.pt")
    parser.add_argument("--prototypes", type=str, default=str(DEFAULT_PROTO_PATH),
                        help="Path to prototypes.npz")
    parser.add_argument("--show", default=None)
    parser.add_argument("--episode", default=None,
                        help="Only process clips from this episode (e.g. s1e1)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help=f"Rejection threshold (default: {DEFAULT_TAU})")
    parser.add_argument("--min-run", type=int, default=DEFAULT_MIN_CONSECUTIVE,
                        help=f"Min consecutive frames to tag (default: {DEFAULT_MIN_CONSECUTIVE})")
    parser.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES,
                        help=f"Frames to sample per clip (default: {DEFAULT_N_FRAMES})")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Checkpoint save interval")
    parser.add_argument("--force", action="store_true",
                        help="Re-process clips that already have visual_characters")
    parser.add_argument("--no-crops", action="store_true",
                        help="Skip left/right crops (faster, single-character only)")
    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # CLIP backbone (frozen)
    log.info("Loading CLIP ViT-B-32...")
    from sentence_transformers import SentenceTransformer
    clip_model = SentenceTransformer("clip-ViT-B-32")

    # Projection head
    head_data = torch.load(args.head, map_location=device, weights_only=True)
    head = ProjectionHead(
        input_dim=head_data["input_dim"],
        hidden_dim=head_data["hidden_dim"],
        output_dim=head_data["output_dim"],
        dropout=head_data.get("dropout", 0.1),
    ).to(device)
    head.load_state_dict(head_data["state_dict"])
    head.eval()
    log.info("Loaded projection head: %d → %d → %d",
             head_data["input_dim"], head_data["hidden_dim"], head_data["output_dim"])

    # Prototypes
    proto_data = np.load(args.prototypes, allow_pickle=True)
    prototypes = proto_data["prototypes"]          # (P, 128)
    prototype_labels = proto_data["prototype_labels"]  # (P,)
    class_names = list(proto_data["class_names"])   # list of str
    log.info("Loaded %d prototypes across %d classes", len(prototypes), len(class_names))

    # ── Load clip index ──────────────────────────────────────────────────
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)
    clip_index_path = get_project_path("clip_index", pipeline_cfg)

    clips_dir = Path(show_config.get("clips_dir", f"./clips/{show_slug}"))
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / clips_dir).resolve()

    clip_data = load_json(clip_index_path)
    clips = clip_data.get("clips", clip_data) if isinstance(clip_data, dict) else clip_data
    log.info("Loaded %d clips from %s", len(clips), clip_index_path)

    # ── Filter ───────────────────────────────────────────────────────────
    indices = list(range(len(clips)))
    if args.episode:
        ep = args.episode.lower()
        indices = [i for i in indices
                   if clips[i].get("filename", "").lower().startswith(ep)]
    if not args.force:
        indices = [i for i in indices if "visual_characters" not in clips[i]]

    log.info("Processing %d clips (episode=%s, force=%s, tau=%.2f, min_run=%d, n_frames=%d)",
             len(indices), args.episode, args.force, args.tau, args.min_run, args.n_frames)

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

            # Step 1: Extract frames
            frames = extract_frames(video_path, n_frames=args.n_frames)
            if not frames:
                skipped += 1
                continue

            # Step 2: Encode through CLIP + projection head
            per_frame_detections = []

            for frame_pil in frames:
                if args.no_crops:
                    crops = [frame_pil]
                else:
                    crops = make_crops(frame_pil)

                # Encode crops through CLIP
                clip_embs = clip_model.encode(crops, show_progress_bar=False)
                norms = np.linalg.norm(clip_embs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                clip_embs = clip_embs / norms

                # Project through ArcFace head
                with torch.no_grad():
                    t = torch.tensor(clip_embs, dtype=torch.float32).to(device)
                    projected = head(t).cpu().numpy()  # (n_crops, 128)

                # Match each crop against prototypes, merge best per character
                frame_matches = {}
                for emb in projected:
                    crop_matches = match_frame_to_prototypes(
                        emb, prototypes, prototype_labels, class_names, args.tau
                    )
                    for name, sim in crop_matches.items():
                        if name not in frame_matches or sim > frame_matches[name]:
                            frame_matches[name] = sim

                per_frame_detections.append(frame_matches)

            # Step 3: Aggregate with temporal contiguity
            aggregated = aggregate_with_contiguity(per_frame_detections, args.min_run)

            # Step 4: Extract tagged characters
            tagged = sorted([name for name, info in aggregated.items() if info["tagged"]])

            # Step 5: Write to clip
            clip["visual_characters"] = tagged
            clip["characters"] = tagged
            clip["prototype_detections"] = {
                name: {
                    "max_similarity": info["max_similarity"],
                    "longest_run": info["longest_run"],
                }
                for name, info in aggregated.items()
                if info["max_similarity"] > 0.3  # Only store non-trivial matches
            }

            for char in tagged:
                char_counter[char] += 1
            processed += 1

        # Checkpoint save
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(clip_index_path, clip_data)
        done = batch_start + len(batch)
        log.info("Checkpoint: %d/%d processed, %d skipped", processed, skipped, done)

    # ── Final save ───────────────────────────────────────────────────────
    if isinstance(clip_data, dict):
        clip_data["clips"] = clips
    save_json(clip_index_path, clip_data)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROTOTYPE INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Tau:       {args.tau}")
    print(f"  Min run:   {args.min_run}")
    print(f"\nCharacter detections across processed clips:")
    for char, count in char_counter.most_common():
        print(f"  {char:20s}  {count:5d} clips")
    print()

    # Flag characters with suspiciously high counts
    if processed > 0:
        median_count = sorted(char_counter.values())[len(char_counter) // 2] if char_counter else 0
        for char, count in char_counter.most_common():
            if count > processed * 0.5:
                print(f"  ⚠️  {char} detected in {count}/{processed} clips ({count/processed*100:.0f}%) — verify this is expected")


if __name__ == "__main__":
    main()
