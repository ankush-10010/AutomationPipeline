"""
clip_indexer_allphase.py — Combined clip processing pipeline.

Runs all three phases in sequence for a single episode:
  Phase 1: Scene Splitting  — Detects scene cuts and slices the episode into clips
  Phase 2: Subtitle Indexing — Cross-references SRT timecodes to tag each clip with dialogue
  Phase 3: Vision Indexing   — Extracts a frame from each clip and asks Ollama Vision to identify characters/location

Usage:
    python scripts/clip_indexer_allphase.py --episode episodes/s1e1.mp4 --srt episodes/s1e1.srt --show rick_and_morty --prefix s1e1

    # Skip vision (faster, subtitle-only tagging):
    python scripts/clip_indexer_allphase.py --episode episodes/s1e1.mp4 --srt episodes/s1e1.srt --show rick_and_morty --prefix s1e1 --skip-vision

    # Use existing manifest (skip scene splitting):
    python scripts/clip_indexer_allphase.py --manifest clips/s1e1_manifest.json --srt episodes/s1e1.srt --show rick_and_morty --prefix s1e1

    # Custom output directory and sensitivity:
    python scripts/clip_indexer_allphase.py --episode episodes/s2e5.mp4 --srt episodes/s2e5.srt --show rick_and_morty --prefix s2e5 --output clips/season2 --threshold 25.0
"""

import argparse
import base64
import json
import re
import sys
import time
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("clip_indexer_allphase")


# ---------------------------------------------------------------------------
# Phase 1: Scene Splitting (from scene_splitter.py)
# ---------------------------------------------------------------------------
def phase1_scene_split(
    video_path: Path,
    output_dir: Path,
    prefix: str,
    threshold: float = 27.0,
) -> Path:
    """Detect scene changes and split the episode into individual clip files.

    Returns the path to the generated manifest JSON.
    """
    log.info("=" * 60)
    log.info("PHASE 1: Scene Splitting")
    log.info("=" * 60)

    try:
        from scenedetect import detect, ContentDetector, split_video_ffmpeg
    except ImportError:
        log.error(
            "scenedetect not found. Install with: pip install scenedetect[opencv]"
        )
        sys.exit(1)

    if not video_path.exists():
        log.error("Video file not found: %s", video_path)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Analyzing %s for scene changes (threshold=%.1f)...", video_path.name, threshold)
    detector = ContentDetector(threshold=threshold)
    scene_list = detect(str(video_path), detector)
    log.info("Detected %d scenes.", len(scene_list))

    if not scene_list:
        log.error("No scenes detected. Try lowering --threshold.")
        sys.exit(1)

    # Split the video into clips
    output_template = str(output_dir / f"{prefix}_scene_$SCENE_NUMBER.mp4")
    log.info("Splitting video into %d clips...", len(scene_list))
    split_video_ffmpeg(
        input_video_path=str(video_path),
        scene_list=scene_list,
        output_file_template=output_template,
        show_progress=True,
    )

    # Build manifest mapping clip filenames → timecodes
    manifest = {}
    for i, (start_time, end_time) in enumerate(scene_list):
        clip_name = f"{prefix}_scene_{i + 1:03d}.mp4"
        manifest[clip_name] = {
            "start_sec": start_time.get_seconds(),
            "end_sec": end_time.get_seconds(),
        }

    manifest_path = output_dir / f"{prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("✓ Phase 1 complete — %d clips + manifest → %s", len(scene_list), manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Phase 2: Subtitle Indexing (from clip_indexer_subtitles.py)
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "have", "has",
    "do", "did", "will", "would", "shall", "should", "can", "could", "of", "in",
    "to", "for", "on", "with", "at", "by", "from", "and", "or", "but", "not",
    "so", "it", "he", "she", "they", "we", "you", "i", "me", "my", "your",
    "this", "that", "what", "which", "who", "how", "when", "where", "why",
    "just", "like", "get", "got", "know", "think", "right", "yeah", "oh", "well",
}


def _parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp (00:00:02,000) to seconds."""
    h, m, s_ms = time_str.strip().split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _load_srt(srt_path: Path) -> list:
    """Parse an .srt file into a list of {start, end, text} dicts."""
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    subs = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            times = lines[1].split(" --> ")
            if len(times) == 2:
                try:
                    start_sec = _parse_srt_time(times[0])
                    end_sec = _parse_srt_time(times[1])
                    text = " ".join(lines[2:]).replace("\n", " ")
                    text = re.sub(r"<[^>]+>", "", text)  # strip HTML tags
                    subs.append({"start": start_sec, "end": end_sec, "text": text})
                except Exception:
                    pass
    return subs


def _generate_keywords(text: str) -> list:
    """Extract clean keyword tags from dialogue text."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = text.split()
    return list({w for w in words if w not in STOP_WORDS and len(w) > 2})


def _parse_season_episode(prefix: str) -> tuple:
    """Try to extract season and episode numbers from the prefix (e.g. 's1e1' → 1, 1)."""
    match = re.match(r"s(\d+)e(\d+)", prefix, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1, 1  # fallback defaults


def phase2_subtitle_index(
    manifest_path: Path,
    srt_path: Path,
    show_slug: str,
    prefix: str,
    index_path: Path,
    clips_dir: Path,
) -> dict:
    """Cross-reference clip timecodes with subtitle timecodes to tag each clip.

    Returns the updated index data dict.
    """
    log.info("=" * 60)
    log.info("PHASE 2: Subtitle Indexing")
    log.info("=" * 60)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)
    if not srt_path.exists():
        log.error("SRT file not found: %s", srt_path)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    subs = _load_srt(srt_path)
    log.info("Loaded %d clips from manifest, %d subtitle blocks from SRT.", len(manifest), len(subs))

    # Load or create the master index
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {"clips": []}

    existing_filenames = {c["filename"] for c in index_data.get("clips", [])}
    season, episode = _parse_season_episode(prefix)

    new_count = 0
    for clip_name, times in manifest.items():
        if clip_name in existing_filenames:
            continue

        clip_start = times["start_sec"]
        clip_end = times["end_sec"]

        # Find overlapping subtitles
        overlapping_text = []
        for sub in subs:
            if sub["start"] < clip_end and sub["end"] > clip_start:
                overlapping_text.append(sub["text"])

        combined_text = " ".join(overlapping_text) if overlapping_text else ""
        tags = _generate_keywords(combined_text) if combined_text else []

        # Build the filepath relative to clips_dir
        clip_filepath = str(clips_dir / clip_name)

        clip_entry = {
            "filename": clip_name,
            "filepath": clip_filepath,
            "show": show_slug,
            "season": season,
            "episode": episode,
            "characters": [],
            "location": "",
            "action": combined_text,
            "mood": "",
            "tags": tags,
            "duration_seconds": round(clip_end - clip_start, 2),
        }

        index_data["clips"].append(clip_entry)
        new_count += 1

    # Save incrementally
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    log.info("✓ Phase 2 complete — %d new clips tagged with subtitles → %s", new_count, index_path)
    return index_data


# ---------------------------------------------------------------------------
# Phase 3: Vision Indexing (from clip_indexer_vision.py)
# ---------------------------------------------------------------------------
def _extract_middle_frame(video_path: Path, temp_image_path: Path) -> bool:
    """Extract the middle frame of a video clip and save as JPEG."""
    try:
        from moviepy.editor import VideoFileClip
        from PIL import Image
    except ImportError:
        log.error("moviepy or Pillow not found. Install with: pip install -r requirements.txt")
        return False

    try:
        clip = VideoFileClip(str(video_path))
        mid_time = clip.duration / 2.0
        frame_data = clip.get_frame(mid_time)
        img = Image.fromarray(frame_data)
        img.save(str(temp_image_path), "JPEG")
        clip.close()
        return True
    except Exception as e:
        log.warning("Failed to extract frame from %s: %s", video_path.name, e)
        return False


def _build_character_reference(characters: list) -> str:
    """Build a comma-separated list of character names."""
    if not characters:
        return ""
    names = [c.get("name") for c in characters if c.get("name")]
    return ", ".join(names)


def _analyze_frame_with_ollama(
    image_path: Path, show_name: str, model: str, characters: list = None,
) -> dict | None:
    """Send frame to Ollama vision model and parse the character/location/action response."""
    import requests

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Build character visual reference if available
    char_ref = _build_character_reference(characters or [])
    char_section = ""
    if char_ref:
        char_section = f"\nKnown characters in this show: {char_ref}\n"

    prompt = (
        f"You are an expert on the TV show '{show_name}'. "
        "Analyze this single frame from an episode.\n"
        f"{char_section}\n"
        "Identify ONLY the characters that are clearly visible in this specific frame. "
        "Do NOT list all characters from the show. If no characters are visible, write 'None'.\n"
        "Identify the location, and what is happening visually.\n\n"
        "You MUST reply in exactly this format with no other text:\n"
        "Characters: [comma separated names of VISIBLE characters ONLY, or None]\n"
        "Location: [brief location name, or Unknown]\n"
        "Action: [1 short sentence describing the visual action]"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        result_text = resp.json().get("response", "").strip()

        metadata = {"characters": [], "location": "", "action": ""}
        for line in result_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("characters:"):
                chars_str = line.split(":", 1)[1].strip()
                if chars_str.lower() not in ("none", "unknown", "n/a", ""):
                    metadata["characters"] = [c.strip() for c in chars_str.split(",")]
            elif line.lower().startswith("location:"):
                loc_str = line.split(":", 1)[1].strip()
                if loc_str.lower() not in ("none", "unknown", "n/a", ""):
                    metadata["location"] = loc_str
            elif line.lower().startswith("action:"):
                act_str = line.split(":", 1)[1].strip()
                if act_str.lower() not in ("none", "unknown", "n/a", ""):
                    metadata["action"] = act_str
        return metadata
    except Exception as e:
        log.error("Ollama vision request failed: %s", e)
        return None


def phase3_vision_index(
    index_path: Path,
    clips_dir: Path,
    show_name: str,
    vision_model: str = "llava",
    force: bool = False,
    characters: list = None,
) -> None:
    """Visually tag each clip using Ollama vision model.

    Reads clips from index, extracts mid-frame, asks the vision LLM,
    and merges the visual tags back into clip_index.json.
    """
    log.info("=" * 60)
    log.info("PHASE 3: Vision Indexing (model: %s)", vision_model)
    log.info("=" * 60)

    if not index_path.exists():
        log.error("Index file not found: %s — run phases 1 & 2 first.", index_path)
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        log.warning("No clips in index. Nothing to do.")
        return

    temp_img = Path("temp_vision_frame.jpg")
    updated_count = 0

    for i, clip in enumerate(clips):
        # Skip if already visually tagged (has characters), unless --force
        if clip.get("characters") and not force:
            continue

        video_path = clips_dir / clip["filename"]
        if not video_path.exists():
            log.warning("Skipping %s (file not found at %s)", clip["filename"], video_path)
            continue

        log.info("[%d/%d] Analyzing %s...", i + 1, len(clips), clip["filename"])

        if not _extract_middle_frame(video_path, temp_img):
            continue

        metadata = _analyze_frame_with_ollama(temp_img, show_name, vision_model, characters)
        if not metadata:
            continue

        # Merge vision results into the clip entry
        clip["characters"] = metadata["characters"]
        if metadata["location"]:
            clip["location"] = metadata["location"]
        if metadata["action"]:
            clean_action = re.sub(r"[^a-zA-Z0-9\s]", "", metadata["action"].lower())
            visual_tags = [w for w in clean_action.split() if len(w) > 3]
            existing_tags = clip.get("tags", [])
            clip["tags"] = list(set(existing_tags + visual_tags))

        log.info("   👁️  Characters: %s | Location: %s",
                 ", ".join(metadata["characters"]) or "None",
                 metadata["location"] or "Unknown")
        updated_count += 1

        # Save incrementally after each clip so we don't lose progress on crash
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Cleanup temp frame
    if temp_img.exists():
        temp_img.unlink()

    log.info("✓ Phase 3 complete — %d clips visually tagged.", updated_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="All-in-one clip indexer: Scene Split → Subtitle Tag → Vision Tag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from episode video:
  python scripts/clip_indexer_allphase.py --episode episodes/s1e1.mp4 --srt episodes/s1e1.srt --show rick_and_morty --prefix s1e1

  # Skip the (slow) vision phase:
  python scripts/clip_indexer_allphase.py --episode episodes/s1e1.mp4 --srt episodes/s1e1.srt --show rick_and_morty --prefix s1e1 --skip-vision

  # Already split? Use existing manifest:
  python scripts/clip_indexer_allphase.py --manifest clips/s1e1_manifest.json --srt episodes/s1e1.srt --show rick_and_morty --prefix s1e1
        """,
    )

    # Input sources (episode OR manifest — at least one required)
    input_group = parser.add_argument_group("Input (provide --episode OR --manifest)")
    input_group.add_argument("--episode", type=str, default=None,
                             help="Path to the full episode video file (e.g. episodes/s1e1.mp4)")
    input_group.add_argument("--manifest", type=str, default=None,
                             help="Path to an existing manifest JSON from a previous scene split (skips Phase 1)")

    # Required
    parser.add_argument("--srt", type=str, required=True,
                        help="Path to the episode's .srt subtitle file")
    parser.add_argument("--show", type=str, required=True,
                        help="Show slug (e.g. rick_and_morty)")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Episode prefix for clip naming (e.g. s1e1, s2e5)")

    # Optional
    parser.add_argument("--output", type=str, default="clips",
                        help="Directory to save clips (default: clips)")
    parser.add_argument("--index", type=str, default="clip_index.json",
                        help="Path to the master clip_index.json (default: clip_index.json)")
    parser.add_argument("--threshold", type=float, default=27.0,
                        help="Scene detection sensitivity (default: 27.0, lower = more cuts)")
    parser.add_argument("--vision-model", type=str, default="llava",
                        help="Ollama vision model for Phase 3 (default: llava)")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip Phase 3 (vision indexing) — useful for fast subtitle-only tagging")
    parser.add_argument("--force-vision", action="store_true",
                        help="Re-process clips that already have vision tags")

    args = parser.parse_args()

    # Validate: must have either --episode or --manifest
    if not args.episode and not args.manifest:
        parser.error("You must provide either --episode (full video) or --manifest (existing manifest JSON).")

    clips_dir = Path(args.output)
    index_path = Path(args.index)
    srt_path = Path(args.srt)

    start_time = time.time()

    # ── Phase 1: Scene Splitting ──────────────────────────────────────────
    if args.episode:
        episode_path = Path(args.episode)
        manifest_path = phase1_scene_split(episode_path, clips_dir, args.prefix, args.threshold)
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            log.error("Manifest file not found: %s", manifest_path)
            sys.exit(1)
        log.info("Skipping Phase 1 — using existing manifest: %s", manifest_path)

    # ── Phase 2: Subtitle Indexing ────────────────────────────────────────
    index_data = phase2_subtitle_index(
        manifest_path, srt_path, args.show, args.prefix, index_path, clips_dir,
    )

    # ── Load character data from show_config.yaml for vision tagging ─────
    characters = []
    try:
        import yaml
        show_config_path = Path("config/show_config.yaml")
        if show_config_path.exists():
            with open(show_config_path, "r", encoding="utf-8") as f:
                show_cfg = yaml.safe_load(f)
            show_data = show_cfg.get("shows", {}).get(args.show, {})
            characters = show_data.get("characters", [])
            if characters:
                log.info("Loaded %d characters from show_config.yaml for vision tagging.", len(characters))
    except Exception as e:
        log.warning("Could not load show_config.yaml for character hints: %s", e)

    # ── Phase 3: Vision Indexing ──────────────────────────────────────────
    if args.skip_vision:
        log.info("Skipping Phase 3 (vision indexing) as requested.")
    else:
        phase3_vision_index(
            index_path, clips_dir, args.show, args.vision_model, args.force_vision, characters,
        )

    elapsed = time.time() - start_time
    total_clips = len(index_data.get("clips", []))

    log.info("=" * 60)
    log.info("🎉 ALL PHASES COMPLETE in %.1f seconds", elapsed)
    log.info("   Total clips in index: %d", total_clips)
    log.info("   Index file: %s", index_path)
    log.info("   Clips dir:  %s", clips_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
