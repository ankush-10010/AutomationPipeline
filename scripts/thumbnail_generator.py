"""
thumbnail_generator.py — Phase 6: Generate thumbnails using Pillow.

Takes a video file and topic text, extracts candidate frames, picks the one
with the highest visual variance (most "interesting"), applies a dark overlay
for text readability, and adds bold topic text. Exports as 1080x1920 JPEG.

Usage:
    python thumbnail_generator.py --video output/final.mp4 --topic "Why Rick Is The Smartest"
    python thumbnail_generator.py --video output/final.mp4 --topic "Multiverse Theory Explained" --output thumbnails/thumb.jpg
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

# -- Local imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config,
    get_project_path,
    setup_logging,
)

log = setup_logging("thumbnail_gen")


# ============================================================================
# FFmpeg / FFprobe helpers
# ============================================================================

def _find_binary(name: str) -> str:
    """Return the path to a binary, or the name itself if on PATH."""
    path = shutil.which(name)
    return path if path else name


FFMPEG = _find_binary("ffmpeg")
FFPROBE = _find_binary("ffprobe")


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        log.warning("ffprobe failed for %s", video_path)
        return 0.0
    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


def extract_frames(video_path: str, output_dir: str,
                   num_frames: int = 10) -> list:
    """Extract evenly-spaced frames from a video.

    Returns a list of extracted frame file paths.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        log.error("Cannot determine video duration for %s", video_path)
        return []

    # Calculate timestamps for evenly-spaced frames
    # Skip the very beginning and end (often black/fade)
    margin = min(1.0, duration * 0.05)
    usable = duration - 2 * margin
    if usable <= 0:
        usable = duration
        margin = 0

    interval = usable / max(num_frames - 1, 1)
    timestamps = [margin + i * interval for i in range(num_frames)]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        out_file = os.path.join(output_dir, f"frame_{i:03d}.png")
        cmd = [
            FFMPEG, "-y",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            out_file,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0 and os.path.exists(out_file):
            frame_paths.append(out_file)
        else:
            log.debug("Failed to extract frame at t=%.2f", ts)

    log.info("Extracted %d/%d frames from %s", len(frame_paths), num_frames, video_path)
    return frame_paths


# ============================================================================
# Frame scoring — pick the most visually interesting frame
# ============================================================================

def score_frame(image_path: str) -> float:
    """Score a frame by its visual variance (higher = more interesting).

    Uses a combination of:
      - Standard deviation of pixel values (contrast/detail)
      - Edge detection magnitude (sharpness/detail)
      - Color saturation variance

    Higher scores indicate more visually interesting frames — avoids
    selecting black/dark/blurry/flat frames.
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # 1. Overall pixel standard deviation (contrast)
        stat = ImageStat.Stat(img)
        # Average stddev across R, G, B channels
        avg_stddev = sum(stat.stddev) / 3.0

        # 2. Edge detection score (sharpness)
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_score = edge_stat.mean[0]  # Mean edge magnitude

        # 3. Color saturation (avoid washed-out/gray frames)
        hsv = img.convert("HSV")
        hsv_stat = ImageStat.Stat(hsv)
        saturation_mean = hsv_stat.mean[1]  # S channel mean

        # Combined score
        score = avg_stddev * 1.0 + edge_score * 0.5 + saturation_mean * 0.3

        return score

    except Exception as e:
        log.warning("Error scoring frame %s: %s", image_path, e)
        return 0.0


def pick_best_frame(frame_paths: list) -> str:
    """Score all frames and return the path of the best one."""
    if not frame_paths:
        return ""

    scored = [(path, score_frame(path)) for path in frame_paths]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_path, best_score = scored[0]
    log.info(
        "Best frame: %s (score=%.1f) out of %d candidates",
        os.path.basename(best_path), best_score, len(scored),
    )
    return best_path


# ============================================================================
# Thumbnail composition with Pillow
# ============================================================================

def shorten_topic(topic: str, max_words: int = 4) -> str:
    """Shorten a topic to ~max_words for thumbnail text.

    Tries to keep the most meaningful words.
    """
    words = topic.strip().split()
    if len(words) <= max_words:
        return topic.strip().upper()

    # Remove common filler words at the start
    fillers = {"why", "how", "what", "the", "a", "an", "is", "are", "was", "of"}
    # Keep first word if it's a question word (adds intrigue)
    kept = []
    for i, w in enumerate(words):
        if i == 0 and w.lower() in {"why", "how", "what"}:
            kept.append(w)
        elif w.lower() not in fillers:
            kept.append(w)
        if len(kept) >= max_words:
            break

    if not kept:
        kept = words[:max_words]

    return " ".join(kept).upper()


def _try_load_font(font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Try to load a TrueType font, falling back to default if unavailable."""
    # Common font paths on different systems
    font_candidates = [
        font_name,
        f"{font_name}.ttf",
        f"{font_name}Bold.ttf",
        f"{font_name}-Bold.ttf",
    ]

    # System font directories
    font_dirs = []
    if sys.platform == "win32":
        font_dirs.append(os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"))
    elif sys.platform == "darwin":
        font_dirs.extend(["/Library/Fonts", "/System/Library/Fonts"])
    else:
        font_dirs.extend([
            "/usr/share/fonts/truetype",
            "/usr/share/fonts",
            "/usr/local/share/fonts",
        ])

    for font_dir in font_dirs:
        for candidate in font_candidates:
            path = os.path.join(font_dir, candidate)
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except OSError:
                    continue

    # Try direct load (Pillow may find system fonts)
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue

    # Final fallback: try common bold fonts
    bold_fallbacks = [
        "arialbd.ttf", "Arial Bold.ttf", "Impact.ttf",
        "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
    ]
    for font_dir in font_dirs:
        for fb in bold_fallbacks:
            path = os.path.join(font_dir, fb)
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except OSError:
                    continue

    log.warning("No TrueType font found — using Pillow default (will be small)")
    return ImageFont.load_default()


def compose_thumbnail(frame_path: str, topic: str, output_path: str,
                      thumb_cfg: dict):
    """Compose the final thumbnail image.

    Steps:
        1. Load and resize the best frame to target dimensions
        2. Apply a semi-transparent dark overlay
        3. Draw bold topic text (auto-wrapped)
        4. Save as JPEG
    """
    width = thumb_cfg.get("width", 1080)
    height = thumb_cfg.get("height", 1920)
    font_name = thumb_cfg.get("font", "Arial")
    font_size = thumb_cfg.get("font_size", 80)
    font_color = thumb_cfg.get("font_color", "white")
    outline_color = thumb_cfg.get("outline_color", "black")
    outline_width = thumb_cfg.get("outline_width", 4)
    overlay_opacity = thumb_cfg.get("overlay_opacity", 0.4)
    text_position = thumb_cfg.get("text_position", "center")

    # 1. Load and resize frame
    img = Image.open(frame_path).convert("RGB")

    # Scale to fill, then center-crop to target
    img_ratio = img.width / img.height
    target_ratio = width / height

    if img_ratio > target_ratio:
        # Image is wider — scale by height, crop width
        new_h = height
        new_w = int(img.width * (height / img.height))
    else:
        # Image is taller — scale by width, crop height
        new_w = width
        new_h = int(img.height * (width / img.width))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center-crop
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    # 2. Apply dark overlay
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, int(255 * overlay_opacity)))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # 3. Draw text
    draw = ImageDraw.Draw(img)
    font = _try_load_font(font_name, font_size)

    # Shorten and wrap topic text
    short_topic = shorten_topic(topic, max_words=4)

    # Auto-wrap text to fit within the image width with margins
    margin_x = int(width * 0.08)
    max_text_width = width - 2 * margin_x

    # Calculate characters per line based on font metrics
    # Use a test string to estimate character width
    test_bbox = draw.textbbox((0, 0), "M" * 10, font=font)
    char_width = (test_bbox[2] - test_bbox[0]) / 10
    chars_per_line = max(5, int(max_text_width / char_width))

    wrapped = textwrap.fill(short_topic, width=chars_per_line)
    lines = wrapped.split("\n")

    # Calculate total text height
    line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_spacing = int(font_size * 0.25)
    total_text_height = sum(line_heights) + line_spacing * (len(lines) - 1)

    # Position text
    if text_position == "center":
        y_start = (height - total_text_height) // 2
    elif text_position == "top":
        y_start = int(height * 0.15)
    else:  # bottom
        y_start = height - total_text_height - int(height * 0.15)

    # Draw each line with outline (stroke)
    y = y_start
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2

        # Draw outline by drawing text offset in all directions
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), line, font=font, fill=outline_color)

        # Draw main text
        draw.text((x, y), line, font=font, fill=font_color)

        y += line_heights[i] + line_spacing

    # 4. Save as JPEG
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "JPEG", quality=95, optimize=True)
    log.info("Thumbnail saved → %s (%dx%d)", output_path, width, height)


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 6: Generate thumbnails from video frames using Pillow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Input video file.",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic text for the thumbnail overlay.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output thumbnail file path (default: thumbnails/<video_stem>_thumb.jpg).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of candidate frames to extract (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load config
    pipeline_cfg = load_pipeline_config()
    thumb_cfg = pipeline_cfg.get("thumbnail", {})

    # Validate video
    video_path = Path(args.video)
    if not video_path.exists():
        log.error("Video file not found: %s", video_path)
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        thumb_dir = get_project_path("thumbnails_dir", pipeline_cfg)
        output_path = thumb_dir / f"{video_path.stem}_thumb.jpg"

    # Create temp dir for frame extraction
    tmp_dir = tempfile.mkdtemp(prefix="thumbnail_")
    log.info("Extracting frames to %s", tmp_dir)

    try:
        # Extract candidate frames
        frame_paths = extract_frames(
            str(video_path), tmp_dir, num_frames=args.num_frames
        )
        if not frame_paths:
            log.error("No frames could be extracted from %s", video_path)
            sys.exit(1)

        # Pick the best frame
        best_frame = pick_best_frame(frame_paths)
        if not best_frame:
            log.error("Could not select a best frame")
            sys.exit(1)

        # Compose thumbnail
        compose_thumbnail(best_frame, args.topic, str(output_path), thumb_cfg)

    finally:
        # Clean up
        try:
            import shutil as _shutil
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            log.warning("Could not clean up temp dir: %s", tmp_dir)


if __name__ == "__main__":
    main()
