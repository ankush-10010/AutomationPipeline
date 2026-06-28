"""
assembler.py — Phase 5: Assemble the final video using FFmpeg subprocess calls.

Reads an assembly manifest (from clip_matcher.py), processes each segment's
visual source (clip or AI-generated image), applies Ken Burns effects,
burns in TikTok-style word-by-word captions, mixes narration + optional BGM,
and exports a final H.264 MP4 at 1080x1920 @ 30fps.

All video processing is done via subprocess.run(['ffmpeg', ...]) — no moviepy
text rendering.

Usage:
    python assembler.py --manifest output/manifest.json --audio audio/narration.wav
    python assembler.py --manifest output/manifest.json --audio audio/narration.wav --bgm assets/bgm/track.mp3
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# -- Local imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config,
    get_project_path,
    load_json,
    setup_logging,
)

log = setup_logging("assembler")


# ============================================================================
# FFmpeg helpers
# ============================================================================

def _find_ffmpeg() -> str:
    """Return the path to the ffmpeg binary, or 'ffmpeg' if on PATH."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    log.warning("ffmpeg not found on PATH — assuming 'ffmpeg' is available")
    return "ffmpeg"


def _find_ffprobe() -> str:
    """Return the path to the ffprobe binary."""
    path = shutil.which("ffprobe")
    if path:
        return path
    return "ffprobe"


FFMPEG = _find_ffmpeg()
FFPROBE = _find_ffprobe()

def _check_nvenc_support() -> bool:
    """Check if the system hardware supports NVIDIA NVENC encoding."""
    try:
        cmd = [
            FFMPEG, "-v", "error", 
            "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1", 
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            log.info("\033[92m🚀 NVIDIA GPU detected! FFmpeg Hardware Acceleration (NVENC) ENABLED.\033[0m")
            return True
        return False
    except Exception:
        return False

HAS_NVENC = _check_nvenc_support()

if not HAS_NVENC:
    log.info("\033[93m⚠️ No NVIDIA GPU detected. FFmpeg falling back to CPU (libx264).\033[0m")

def get_video_encoder_args() -> list:
    """Return the optimal FFmpeg arguments based on hardware availability."""
    if HAS_NVENC:
        # p6 = high quality, cq=16 = constant quality visually lossless
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "16"]
    else:
        # Fallback to CPU
        return ["-c:v", "libx264", "-preset", "slow", "-crf", "16"]


def run_ffmpeg(args: list, desc: str = "ffmpeg"):
    """Run an ffmpeg command, raising on failure."""
    cmd = [FFMPEG] + args
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        log.error("%s failed:\n%s", desc, result.stderr[-2000:])
        raise RuntimeError(f"{desc} failed with return code {result.returncode}")
    return result


def get_media_duration(filepath: str) -> float:
    """Get the duration of a media file in seconds using ffprobe."""
    cmd = [
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(filepath),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.warning("ffprobe failed for %s", filepath)
        return 0.0
    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


# ============================================================================
# Segment visual preparation
# ============================================================================

def prepare_clip_segment(clip_path: str, duration: float, clip_start: float,
                         width: int, height: int, output_path: str,
                         fps: int = 30):
    """Extract and resize a clip segment to target dimensions.

    Uses center-crop to fit 9:16 aspect ratio.
    """
    # Build filter: Split video into background (blurred, fills screen) and foreground (fit in center)
    vf = (
        f"split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=40:40[bg_blur];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg_scaled];"
        f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2,"
        f"fps={fps},"
        f"setsar=1"
    )

    args = [
        "-y",
        "-stream_loop", "-1",
        "-ss", str(clip_start),
        "-i", str(clip_path),
        "-t", str(duration),
        "-vf", vf,
        "-an",  # Strip audio from clip
        *get_video_encoder_args(),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    run_ffmpeg(args, f"prepare clip: {Path(clip_path).name}")


def prepare_image_segment(image_path: str, duration: float,
                          width: int, height: int, output_path: str,
                          fps: int = 30, ken_burns: dict = None):
    """Convert a static image to a video segment with optional Ken Burns effect.

    Ken Burns: subtle zoom from zoom_range[0] to zoom_range[1] over the
    segment duration.
    """
    if ken_burns and ken_burns.get("enabled", False):
        zoom_start, zoom_end = ken_burns.get("zoom_range", [1.0, 1.15])
        # zoompan filter for Ken Burns
        total_frames = int(duration * fps)
        # Scale image larger to allow zoom headroom
        scale_w = int(width * zoom_end * 1.1)
        scale_h = int(height * zoom_end * 1.1)

        vf = (
            f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
            f"crop={scale_w}:{scale_h},"
            f"zoompan=z='min({zoom_end},max({zoom_start},"
            f"{zoom_start}+(in/{total_frames})*({zoom_end}-{zoom_start})))':"
            f"x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':"
            f"d={total_frames}:s={width}x{height}:fps={fps},"
            f"setsar=1"
        )
    else:
        # Simple static display
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={fps},"
            f"setsar=1"
        )

    args = [
        "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-t", str(duration),
        "-vf", vf,
        *get_video_encoder_args(),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    run_ffmpeg(args, f"prepare image: {Path(image_path).name}")


def prepare_black_segment(duration: float, width: int, height: int,
                          output_path: str, fps: int = 30):
    """Generate a black video segment as a fallback."""
    args = [
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
        *get_video_encoder_args(),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    run_ffmpeg(args, "prepare black segment")


# ============================================================================
# Caption filter generation (TikTok-style drawtext)
# ============================================================================

def _escape_drawtext(text: str) -> str:
    """Escape text for ffmpeg drawtext filter.

    FFmpeg drawtext needs colons, backslashes, single-quotes, semicolons,
    and brackets escaped.
    """
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace("'", "\u2019")  # Replace with right single quote
    text = text.replace(":", "\\:")
    text = text.replace(";", "\\;")
    text = text.replace("%", "%%")
    return text


def build_caption_drawtext(segments: list, caption_cfg: dict,
                           video_height: int, video_width: int) -> str:
    """Build a chain of drawtext filters for word-by-word caption highlighting.

    For each segment, groups words (words_per_group at a time) and displays
    them at the bottom of the screen. The currently-spoken word is highlighted
    in gold.

    Returns the full drawtext filter string for the FFmpeg filter_complex.
    """
    font = caption_cfg.get("font", "Arial")
    font_size = caption_cfg.get("font_size", 64)
    font_color = caption_cfg.get("font_color", "white")
    highlight_color = caption_cfg.get("highlight_color", "#FFD700")
    outline_color = caption_cfg.get("outline_color", "black")
    outline_width = caption_cfg.get("outline_width", 3)
    margin_bottom = caption_cfg.get("margin_bottom", 200)
    words_per_group = caption_cfg.get("words_per_group", 4)

    y_pos = video_height - margin_bottom
    filters = []

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # Fallback: show full segment text as a single block
            text = _escape_drawtext(seg.get("text", ""))
            if not text:
                continue
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            f = (
                f"drawtext=text='{text}':"
                f"fontfile='':"
                f"font='{font}':"
                f"fontsize={font_size}:"
                f"fontcolor={font_color}:"
                f"borderw={outline_width}:"
                f"bordercolor={outline_color}:"
                f"x=(w-text_w)/2:"
                f"y={y_pos}:"
                f"enable='between(t,{start:.3f},{end:.3f})'"
            )
            filters.append(f)
            continue

        # Group words for display
        for group_start in range(0, len(words), words_per_group):
            group = words[group_start:group_start + words_per_group]
            if not group:
                continue

            group_t_start = group[0].get("start", 0)
            group_t_end = group[-1].get("end", group[-1].get("start", 0))

            # For each word in the group, draw it individually so we can
            # highlight the active one. Use a two-pass approach:
            # 1) Draw all words in white (base layer)
            # 2) Overdraw the active word in gold

            # Build the full group text for the base layer
            group_text = " ".join(w.get("word", "") for w in group)
            group_text_escaped = _escape_drawtext(group_text)

            # Base layer: all words in white
            base = (
                f"drawtext=text='{group_text_escaped}':"
                f"font='{font}':"
                f"fontsize={font_size}:"
                f"fontcolor={font_color}:"
                f"borderw={outline_width}:"
                f"bordercolor={outline_color}:"
                f"x=(w-text_w)/2:"
                f"y={y_pos}:"
                f"enable='between(t,{group_t_start:.3f},{group_t_end:.3f})'"
            )
            filters.append(base)

            # Highlight layer: overdraw each word in gold during its time
            # Calculate approximate x offset for each word
            for wi, w in enumerate(group):
                word_text = w.get("word", "")
                if not word_text:
                    continue
                w_start = w.get("start", 0)
                w_end = w.get("end", w_start)
                word_escaped = _escape_drawtext(word_text)

                # Approximate the x position: we need the text width of
                # preceding words. Use a simpler approach — draw the
                # highlighted word centered and let the viewer's eye track it.
                # A more precise method would require font metrics.
                # Instead, just redraw the full group with the active word colored.
                # We overlay the active word by computing a prefix width estimate.
                prefix = " ".join(ww.get("word", "") for ww in group[:wi])
                if prefix:
                    prefix += " "
                prefix_escaped = _escape_drawtext(prefix)

                # Use text expansion to position: draw the highlighted word
                # offset by the prefix width
                highlight = (
                    f"drawtext=text='{word_escaped}':"
                    f"font='{font}':"
                    f"fontsize={font_size}:"
                    f"fontcolor={highlight_color}:"
                    f"borderw={outline_width}:"
                    f"bordercolor={outline_color}:"
                    f"x=(w-text_w('{group_text_escaped}'))/2+text_w('{prefix_escaped}'):"
                    f"y={y_pos}:"
                    f"enable='between(t,{w_start:.3f},{w_end:.3f})'"
                )
                filters.append(highlight)

    return ",".join(filters) if filters else ""


def build_simple_caption_drawtext(segments: list, caption_cfg: dict,
                                  video_height: int, video_width: int) -> str:
    """Build a simpler drawtext filter chain that's more compatible.

    Shows word groups at the bottom of the screen, with the active word
    highlighted using enable timing.
    """
    font = caption_cfg.get("font", "Arial")
    font_size = caption_cfg.get("font_size", 64)
    font_color = caption_cfg.get("font_color", "white")
    highlight_color = caption_cfg.get("highlight_color", "#FFD700")
    outline_color = caption_cfg.get("outline_color", "black")
    outline_width = caption_cfg.get("outline_width", 3)
    margin_bottom = caption_cfg.get("margin_bottom", 200)
    words_per_group = caption_cfg.get("words_per_group", 4)

    y_pos = video_height - margin_bottom
    filters = []

    for seg in segments:
        words = seg.get("words", [])
        seg_text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        if not words and not seg_text:
            continue

        if not words:
            # No word-level timestamps: show full segment text
            text_esc = _escape_drawtext(seg_text)
            # Wrap long text
            if len(seg_text) > 30:
                mid = len(seg_text) // 2
                # Find space near the middle
                space = seg_text.rfind(" ", 0, mid + 10)
                if space > 0:
                    line1 = _escape_drawtext(seg_text[:space])
                    line2 = _escape_drawtext(seg_text[space + 1:])
                    text_esc = f"{line1}\\n{line2}"

            f = (
                f"drawtext=text='{text_esc}':"
                f"font='{font}':"
                f"fontsize={font_size}:"
                f"fontcolor={font_color}:"
                f"borderw={outline_width}:"
                f"bordercolor={outline_color}:"
                f"x=(w-text_w)/2:"
                f"y={y_pos}:"
                f"enable='between(t\\,{seg_start:.3f}\\,{seg_end:.3f})'"
            )
            filters.append(f)
            continue

        # Group words and create drawtext entries
        for g_start in range(0, len(words), words_per_group):
            group = words[g_start:g_start + words_per_group]
            if not group:
                continue

            group_t_start = group[0].get("start", 0)
            group_t_end = group[-1].get("end", group[-1].get("start", 0))
            group_text = " ".join(w.get("word", "") for w in group)
            group_esc = _escape_drawtext(group_text)

            # Draw the full word group in the default color
            base_f = (
                f"drawtext=text='{group_esc}':"
                f"font='{font}':"
                f"fontsize={font_size}:"
                f"fontcolor={font_color}:"
                f"borderw={outline_width}:"
                f"bordercolor={outline_color}:"
                f"x=(w-text_w)/2:"
                f"y={y_pos}:"
                f"enable='between(t\\,{group_t_start:.3f}\\,{group_t_end:.3f})'"
            )
            filters.append(base_f)

    return ",".join(filters) if filters else ""


# ============================================================================
# Final assembly
# ============================================================================

def assemble_video(manifest: dict, audio_path: str, output_path: str,
                   video_cfg: dict, bgm_path: str = None,
                   clips_dir: str = None, images_dir: str = None):
    """Assemble the final video from the assembly manifest.

    Steps:
        1. Prepare each segment's visual (clip / image / black)
        2. Concatenate all segment videos
        3. Add captions via drawtext
        4. Mix narration audio (+ optional BGM)
        5. Export final H.264 MP4
    """
    width = video_cfg.get("width", 1080)
    height = video_cfg.get("height", 1920)
    fps = video_cfg.get("fps", 30)
    codec = video_cfg.get("codec", "libx264")
    audio_codec = video_cfg.get("audio_codec", "aac")
    pix_fmt = video_cfg.get("pixel_format", "yuv420p")
    caption_cfg = video_cfg.get("captions", {})
    ken_burns_cfg = video_cfg.get("ken_burns", {})
    bgm_cfg = video_cfg.get("bgm", {})

    segments = manifest.get("segments", [])
    if not segments:
        log.error("Manifest has no segments — nothing to assemble")
        return

    # Create temp directory for intermediate files
    tmp_dir = Path(tempfile.mkdtemp(prefix="assembler_"))
    log.info("Working directory: %s", tmp_dir)

    try:
        # -- Step 0: Ensure video length matches audio exactly --
        audio_dur = _get_video_duration(audio_path)
        if audio_dur > 0 and segments:
            last_seg = segments[-1]
            if last_seg.get("end", 0) < audio_dur:
                log.info("Extending last segment to match audio duration (%.2fs -> %.2fs)", 
                         last_seg.get("end", 0), audio_dur)
                last_seg["end"] = audio_dur

        # -- Step 1: Prepare each segment's visual -------------
        segment_files = []
        for seg in segments:
            seg_id = seg.get("id", len(segment_files))
            duration = seg.get("end", 0) - seg.get("start", 0)
            if duration <= 0:
                log.warning("Segment %d has zero/negative duration, skipping", seg_id)
                continue

            visual_type = seg.get("visual_type", "black")
            visual_source = seg.get("visual_source", "")
            clip_start = seg.get("clip_start", 0.0)
            seg_output = str(tmp_dir / f"seg_{seg_id:04d}.mp4")

            if visual_type == "clip" and visual_source:
                # Resolve clip path
                clip_file = _resolve_visual_path(visual_source, clips_dir)
                if clip_file and Path(clip_file).exists():
                    prepare_clip_segment(
                        clip_file, duration, clip_start,
                        width, height, seg_output, fps,
                    )
                else:
                    log.warning(
                        "Clip not found: %s — using black", visual_source
                    )
                    prepare_black_segment(duration, width, height, seg_output, fps)

            elif visual_type == "ai_image":
                # Look for a pre-generated image matching segment ID
                img_file = _resolve_visual_path(
                    f"seg_{seg_id:04d}.png", images_dir
                )
                if img_file and Path(img_file).exists():
                    prepare_image_segment(
                        img_file, duration, width, height,
                        seg_output, fps, ken_burns_cfg,
                    )
                else:
                    log.warning(
                        "AI image not found for segment %d — using black", seg_id
                    )
                    prepare_black_segment(duration, width, height, seg_output, fps)
            else:
                prepare_black_segment(duration, width, height, seg_output, fps)

            segment_files.append(seg_output)

        if not segment_files:
            log.error("No segment files were prepared — aborting")
            return

        # -- Step 2: Concatenate & apply captions ----------------
        concat_list = tmp_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for sf in segment_files:
                safe_path = str(sf).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        caption_filter = build_simple_caption_drawtext(
            segments, caption_cfg, height, width
        )

        captioned_video = str(tmp_dir / "captioned.mp4")
        if caption_filter:
            run_ffmpeg([
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-vf", caption_filter,
                *get_video_encoder_args(),
                "-pix_fmt", pix_fmt,
                "-an",
                captioned_video,
            ], "concatenate and apply captions")
        else:
            log.warning("No caption filter generated — concatenating visual segments")
            run_ffmpeg([
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                *get_video_encoder_args(),
                "-pix_fmt", pix_fmt,
                "-an",
                captioned_video,
            ], "concatenate visual segments")

        # -- Step 4: Mix audio ---------------------------------
        output_file = str(output_path)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        if bgm_path and Path(bgm_path).exists() and bgm_cfg.get("enabled", False):
            # Mix narration + BGM
            bgm_vol_db = bgm_cfg.get("volume_db", -20)
            fade_in = bgm_cfg.get("fade_in_seconds", 1.0)
            fade_out = bgm_cfg.get("fade_out_seconds", 2.0)

            # Get video duration for BGM fade-out timing
            vid_duration = get_media_duration(captioned_video)
            fade_out_start = max(0, vid_duration - fade_out)

            audio_filter = (
                f"[1:a]volume={bgm_vol_db}dB,"
                f"afade=t=in:d={fade_in},"
                f"afade=t=out:st={fade_out_start:.2f}:d={fade_out}[bgm];"
                f"[2:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )

            run_ffmpeg([
                "-y",
                "-i", captioned_video,
                "-i", str(bgm_path),
                "-i", str(audio_path),
                "-filter_complex", audio_filter,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", audio_codec,
                "-b:a", "192k",
                "-shortest",
                output_file,
            ], "mix audio with BGM")
        else:
            # Narration only
            run_ffmpeg([
                "-y",
                "-i", captioned_video,
                "-i", str(audio_path),
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", audio_codec,
                "-b:a", "192k",
                "-shortest",
                output_file,
            ], "add narration audio")

        log.info("Final video exported → %s", output_file)

    finally:
        # Clean up temp files
        try:
            import shutil as _shutil
            _shutil.rmtree(tmp_dir, ignore_errors=True)
            log.debug("Cleaned up temp dir: %s", tmp_dir)
        except Exception:
            log.warning("Could not clean up temp dir: %s", tmp_dir)


def _resolve_visual_path(filename: str, base_dir: str = None) -> str:
    """Resolve a visual source filename to a full path.

    Checks the base_dir first, then tries the filename as-is (absolute path).
    """
    if not filename:
        return ""

    # If already absolute and exists
    if Path(filename).is_absolute() and Path(filename).exists():
        return filename

    # Try relative to base_dir
    if base_dir:
        candidate = Path(base_dir) / filename
        if candidate.exists():
            return str(candidate)
        # Also search recursively one level
        for child in Path(base_dir).rglob(filename):
            return str(child)

    return filename


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 5: Assemble final video from assembly manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Assembly manifest JSON (from clip_matcher.py).",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Narration audio file (WAV).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path (default: output/final.mp4).",
    )
    parser.add_argument(
        "--bgm",
        default=None,
        help="Background music file (optional).",
    )
    parser.add_argument(
        "--clips-dir",
        default=None,
        help="Override clips directory (default: from pipeline config).",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Override images directory (default: from pipeline config).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load config
    pipeline_cfg = load_pipeline_config()
    video_cfg = pipeline_cfg.get("video", {})

    # Load manifest
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    if not manifest:
        log.error("Failed to load manifest: %s", manifest_path)
        sys.exit(1)
    log.info(
        "Loaded manifest with %d segments",
        len(manifest.get("segments", [])),
    )

    # Resolve paths
    audio_path = Path(args.audio)
    if not audio_path.exists():
        log.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = get_project_path("output_dir", pipeline_cfg) / "final.mp4"

    clips_dir = args.clips_dir or str(get_project_path("clips_dir", pipeline_cfg))
    images_dir = args.images_dir or str(get_project_path("images_dir", pipeline_cfg))

    bgm_path = args.bgm
    if not bgm_path:
        # Check if BGM is enabled and a tracks directory exists
        bgm_cfg = video_cfg.get("bgm", {})
        if bgm_cfg.get("enabled", False):
            bgm_dir = get_project_path("project_root", pipeline_cfg) / bgm_cfg.get(
                "tracks_dir", "assets/bgm"
            ).lstrip("./")
            if bgm_dir.exists():
                # Pick first available track
                for ext in ("*.mp3", "*.wav", "*.ogg", "*.m4a"):
                    tracks = list(bgm_dir.glob(ext))
                    if tracks:
                        bgm_path = str(tracks[0])
                        log.info("Auto-selected BGM: %s", bgm_path)
                        break

    # Assemble
    assemble_video(
        manifest=manifest,
        audio_path=str(audio_path),
        output_path=output_path,
        video_cfg=video_cfg,
        bgm_path=bgm_path,
        clips_dir=clips_dir,
        images_dir=images_dir,
    )


if __name__ == "__main__":
    main()
