"""
clip_indexer_clip_embed.py — Compute CLIP visual embeddings from clip keyframes.

Extracts the middle frame from each video clip, encodes it through CLIP's
vision encoder, and stores the embedding as `clip_visual_embedding` in
clip_index.json. At match time, the narration text is encoded with CLIP's
text encoder and compared against these visual embeddings — giving the
matcher actual visual understanding of what's on screen.

Dependencies:
    pip install sentence-transformers Pillow opencv-python-headless

Usage:
    python scripts/clip_indexer_clip_embed.py
    python scripts/clip_indexer_clip_embed.py --target-dir s1e1
    python scripts/clip_indexer_clip_embed.py --batch-size 200
"""

import argparse
import sys
import re
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config,
    get_active_show,
    get_project_path,
    load_json,
    save_json,
    setup_logging,
)

log = setup_logging("clip_embed")


def extract_middle_frame(video_path: Path) -> Image.Image | None:
    """Extract the middle frame from a video file as a PIL Image."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    mid = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # OpenCV BGR -> RGB -> PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def extract_multi_frames(video_path: Path, n_frames: int = 3) -> list:
    """Extract N evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Evenly spaced indices (skip first and last 10%)
    start = max(1, int(total * 0.1))
    end = max(start + 1, int(total * 0.9))
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP visual embeddings from clip keyframes."
    )
    parser.add_argument("--index", default=None, help="Path to clip_index.json")
    parser.add_argument("--show", default=None, help="Show identifier")
    parser.add_argument(
        "--target-dir", default=None,
        help="Only process clips matching this prefix (e.g. s1e1)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Save progress every N clips"
    )
    parser.add_argument(
        "--n-frames", type=int, default=3,
        help="Number of frames to sample per clip (averaged into one embedding)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-compute embeddings even if clip_visual_embedding already exists"
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
    ).resolve()
    if not clips_dir.is_absolute():
        from config_loader import PROJECT_ROOT
        clips_dir = (PROJECT_ROOT / show_config.get("clips_dir", f"./clips/{show_slug}")).resolve()

    log.info("Clips directory: %s", clips_dir)
    log.info("Loading clip index from %s", clip_index_path)

    clip_data = load_json(clip_index_path)
    if isinstance(clip_data, dict):
        clips = clip_data.get("clips", [])
    elif isinstance(clip_data, list):
        clips = clip_data
    else:
        log.error("Invalid clip index format")
        sys.exit(1)

    # Filter by target directory if specified
    if args.target_dir:
        target = args.target_dir.lower()
        indices = [i for i, c in enumerate(clips)
                   if c.get("filename", "").lower().startswith(target)]
        log.info("Filtering to %d clips matching prefix '%s'", len(indices), target)
    else:
        indices = list(range(len(clips)))

    # Skip already-embedded clips unless --force
    if not args.force:
        indices = [i for i in indices
                   if not clips[i].get("clip_visual_embedding")]
        log.info("After skipping already-embedded: %d clips to process", len(indices))

    if not indices:
        log.info("Nothing to do — all clips already have CLIP embeddings.")
        return

    # Load CLIP model via sentence-transformers (same library already installed)
    log.info("Loading CLIP model (clip-ViT-B-32) via sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    clip_model = SentenceTransformer("clip-ViT-B-32")
    log.info("CLIP model loaded.")

    processed = 0
    skipped = 0

    for batch_start in range(0, len(indices), args.batch_size):
        batch = indices[batch_start:batch_start + args.batch_size]

        for idx in batch:
            clip = clips[idx]
            filename = clip.get("filename", "")

            # Find video file — could be in clips_dir or a subdirectory
            video_path = clips_dir / filename
            if not video_path.exists():
                # Try subdirectory by episode prefix
                ep_match = re.match(r"(s\d+e\d+)", filename, re.IGNORECASE)
                if ep_match:
                    video_path = clips_dir / ep_match.group(1) / filename
                if not video_path.exists():
                    # Search recursively
                    found = list(clips_dir.rglob(filename))
                    if found:
                        video_path = found[0]
                    else:
                        skipped += 1
                        continue

            # Extract frames
            if args.n_frames == 1:
                frame = extract_middle_frame(video_path)
                if frame is None:
                    skipped += 1
                    continue
                frames = [frame]
            else:
                frames = extract_multi_frames(video_path, args.n_frames)
                if not frames:
                    skipped += 1
                    continue

            # Encode frames through CLIP vision encoder
            frame_embeddings = clip_model.encode(frames)

            # Average the frame embeddings into a single clip embedding
            if len(frame_embeddings.shape) == 1:
                avg_embedding = frame_embeddings
            else:
                avg_embedding = np.mean(frame_embeddings, axis=0)

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            clip["clip_visual_embedding"] = avg_embedding.tolist()
            processed += 1

        # Progress save
        log.info(
            "Progress: %d/%d processed, %d skipped (batch %d/%d)",
            processed, len(indices), skipped,
            batch_start // args.batch_size + 1,
            (len(indices) + args.batch_size - 1) // args.batch_size,
        )
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(clip_index_path, clip_data)
        log.info("Checkpoint saved.")

    log.info("CLIP embedding complete: %d processed, %d skipped", processed, skipped)


if __name__ == "__main__":
    main()
