"""
clip_indexer.py — Utility for scanning and tagging a clip library.

Modes:
  • Interactive — prompt the user to tag each clip manually.
  • Batch CSV  — import tags from a CSV file.
  • Auto-tag   — extract a frame, describe it via Ollama, and suggest tags.

CLI usage:
  python clip_indexer.py -d ./clips/rick_and_morty --interactive -s rick_and_morty
  python clip_indexer.py -d ./clips --auto-tag
  python clip_indexer.py -d ./clips --csv tags.csv -o clip_index.json
"""

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Ensure sibling imports work when run directly
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config_loader import (
    setup_logging,
    load_pipeline_config,
    get_project_path,
    load_json,
    save_json,
    PROJECT_ROOT,
)

log = setup_logging("clip_indexer")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# ---------------------------------------------------------------------------

def get_video_duration(filepath: str | Path) -> float:
    """
    Return the duration (in seconds) of a video file using ffprobe.

    Falls back to 0.0 if ffprobe is unavailable or the file cannot be probed.
    """
    filepath = str(filepath)
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("ffprobe returned non-zero for %s: %s", filepath, result.stderr.strip())
            return 0.0
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        return round(duration, 2)
    except FileNotFoundError:
        log.error("ffprobe not found — install FFmpeg and ensure it is on PATH")
        return 0.0
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as exc:
        log.warning("Could not probe %s: %s", filepath, exc)
        return 0.0


def _extract_frame(filepath: str | Path, timestamp: float = 2.0) -> Optional[bytes]:
    """
    Extract a single PNG frame from *filepath* at *timestamp* seconds.

    Returns raw PNG bytes or ``None`` on failure.
    """
    filepath = str(filepath)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-ss", str(timestamp),
                "-i", filepath,
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Try at 0s if the video might be shorter than 2s
            if timestamp > 0:
                return _extract_frame(filepath, 0.0)
            log.warning("ffmpeg frame extraction failed for %s", filepath)
            return None
        return result.stdout if result.stdout else None
    except FileNotFoundError:
        log.error("ffmpeg not found — install FFmpeg and ensure it is on PATH")
        return None
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timed out extracting frame from %s", filepath)
        return None


# ---------------------------------------------------------------------------
# Video scanning
# ---------------------------------------------------------------------------

def scan_video_files(directory: str | Path) -> List[Path]:
    """Recursively find all video files in *directory*."""
    directory = Path(directory)
    if not directory.is_dir():
        log.error("Directory not found: %s", directory)
        return []
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    log.info("Found %d video file(s) in %s", len(files), directory)
    return files


# ---------------------------------------------------------------------------
# Auto-tag via Ollama
# ---------------------------------------------------------------------------

_AUTOTAG_PROMPT = """\
Analyse this video frame screenshot and return ONLY valid JSON with these keys:
- "characters": list of character names visible (or ["unknown"] if unclear)
- "location": a short description of the setting/location
- "action": what is happening in the frame
- "mood": the emotional tone (e.g. "dramatic", "comedic", "dark", "calm")
- "tags": list of 5-10 descriptive keyword strings

Respond with JSON only — no markdown fences, no explanation.
"""


def _autotag_clip(
    filepath: Path,
    llm_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract a frame from *filepath*, send it to Ollama, and return parsed tags.

    Returns a dict with keys: characters, location, action, mood, tags.
    On failure returns empty-ish defaults.
    """
    empty = {
        "characters": [],
        "location": "",
        "action": "",
        "mood": "",
        "tags": [],
    }

    frame_data = _extract_frame(filepath)
    if frame_data is None:
        log.warning("No frame extracted for %s — skipping auto-tag", filepath.name)
        return empty

    b64_frame = base64.b64encode(frame_data).decode("ascii")

    base_url = llm_cfg.get("base_url", "http://localhost:11434").rstrip("/")
    payload = {
        "model": llm_cfg.get("model", "llama3.1:8b"),
        "prompt": _AUTOTAG_PROMPT,
        "images": [b64_frame],
        "stream": False,
        "options": {
            "temperature": llm_cfg.get("temperature", 0.5),
            "num_predict": llm_cfg.get("max_tokens", 512),
        },
    }

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=llm_cfg.get("timeout_seconds", 300),
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    except Exception as exc:
        log.warning("Ollama auto-tag request failed for %s: %s", filepath.name, exc)
        return empty

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        return {
            "characters": data.get("characters", []),
            "location": str(data.get("location", "")),
            "action": str(data.get("action", "")),
            "mood": str(data.get("mood", "")),
            "tags": data.get("tags", []),
        }
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("Failed to parse Ollama response for %s: %s", filepath.name, exc)
        return empty


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _interactive_tag(filepath: Path, index: int, total: int, duration: float) -> Optional[Dict[str, Any]]:
    """Prompt the user to tag a single clip. Returns tag dict or None to skip."""
    print(f"\n{'='*60}")
    print(f"  Clip {index + 1} / {total}")
    print(f"  File : {filepath.name}")
    print(f"  Path : {filepath}")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'='*60}")

    response = input("  Tag this clip? (Enter to continue / 'skip' / 'quit'): ").strip().lower()
    if response in ("quit", "q"):
        return None  # Signal to stop
    if response == "skip":
        return {"_skip": True}

    characters = input("  Characters (comma-separated): ").strip()
    location = input("  Location: ").strip()
    action = input("  Action: ").strip()
    mood = input("  Mood: ").strip()
    tags = input("  Tags (comma-separated): ").strip()

    return {
        "characters": [c.strip().lower() for c in characters.split(",") if c.strip()],
        "location": location,
        "action": action,
        "mood": mood,
        "tags": [t.strip().lower() for t in tags.split(",") if t.strip()],
    }


# ---------------------------------------------------------------------------
# Batch CSV import
# ---------------------------------------------------------------------------

def _load_csv_tags(csv_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Read a CSV file and return a dict keyed by filename.

    Expected columns: filename, characters, location, action, mood, tags
    Characters and tags columns use pipe (|) separators for multiple values.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.error("CSV file not found: %s", csv_path)
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Validate required columns
        required = {"filename", "characters", "location", "action", "mood", "tags"}
        if reader.fieldnames is None:
            log.error("CSV file is empty: %s", csv_path)
            return {}
        missing = required - set(reader.fieldnames)
        if missing:
            log.error("CSV missing required columns: %s", missing)
            return {}

        for row in reader:
            fname = row["filename"].strip()
            if not fname:
                continue
            result[fname] = {
                "characters": [c.strip().lower() for c in row["characters"].split("|") if c.strip()],
                "location": row["location"].strip(),
                "action": row["action"].strip(),
                "mood": row["mood"].strip(),
                "tags": [t.strip().lower() for t in row["tags"].split("|") if t.strip()],
            }
    log.info("Loaded %d entries from CSV %s", len(result), csv_path)
    return result


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _load_index(index_path: Path) -> Dict[str, Any]:
    """Load existing clip index, preserving the _schema_example."""
    if index_path.exists():
        data = load_json(index_path)
        if isinstance(data, dict):
            return data
        # File contains a list or something unexpected
        return {"clips": data if isinstance(data, list) else []}
    return {"clips": []}


def _existing_filenames(index_data: Dict[str, Any]) -> set:
    """Return the set of filenames already indexed."""
    clips = index_data.get("clips", [])
    return {c.get("filename", "") for c in clips if isinstance(c, dict)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def index_clips(
    directory: str | Path,
    mode: str = "interactive",
    output_path: Optional[str | Path] = None,
    show_slug: Optional[str] = None,
    csv_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """
    Scan *directory* for video files, tag them according to *mode*, and
    append new entries to the clip index.

    Parameters:
        directory   — directory to scan for video files
        mode        — "interactive", "auto-tag", or "csv"
        output_path — path to clip_index.json (default from config)
        show_slug   — show identifier to store with each clip
        csv_path    — path to CSV file (required when mode="csv")

    Returns the list of newly-added clip entries.
    """
    pipeline_cfg = load_pipeline_config()
    llm_cfg = pipeline_cfg.get("llm", {})

    if output_path is None:
        output_path = get_project_path("clip_index")
    output_path = Path(output_path)

    # Scan videos
    video_files = scan_video_files(directory)
    if not video_files:
        log.warning("No video files found in %s", directory)
        return []

    # Load existing index
    index_data = _load_index(output_path)
    existing = _existing_filenames(index_data)
    new_files = [f for f in video_files if f.name not in existing]
    log.info(
        "%d video(s) total, %d already indexed, %d new",
        len(video_files), len(existing), len(new_files),
    )

    if not new_files:
        log.info("All clips already indexed — nothing to do")
        return []

    # Load CSV data if in batch mode
    csv_tags: Dict[str, Dict[str, Any]] = {}
    if mode == "csv":
        if not csv_path:
            log.error("--csv path required for batch mode")
            return []
        csv_tags = _load_csv_tags(csv_path)

    new_entries: List[Dict[str, Any]] = []

    for idx, filepath in enumerate(new_files):
        duration = get_video_duration(filepath)

        if mode == "interactive":
            tag_info = _interactive_tag(filepath, idx, len(new_files), duration)
            if tag_info is None:
                log.info("User quit — stopping interactive tagging")
                break
            if tag_info.get("_skip"):
                log.info("Skipped %s", filepath.name)
                continue

        elif mode == "auto-tag":
            log.info("Auto-tagging [%d/%d]: %s", idx + 1, len(new_files), filepath.name)
            tag_info = _autotag_clip(filepath, llm_cfg)

        elif mode == "csv":
            tag_info = csv_tags.get(filepath.name)
            if tag_info is None:
                log.warning("No CSV entry for %s — skipping", filepath.name)
                continue
        else:
            log.error("Unknown mode: %s", mode)
            return []

        entry = {
            "filename": filepath.name,
            "filepath": str(filepath),
            "duration_seconds": duration,
            "characters": tag_info.get("characters", []),
            "location": tag_info.get("location", ""),
            "action": tag_info.get("action", ""),
            "mood": tag_info.get("mood", ""),
            "tags": tag_info.get("tags", []),
        }
        if show_slug:
            entry["show"] = show_slug

        new_entries.append(entry)
        log.info("Indexed: %s (%.1fs)", filepath.name, duration)

    # Append to index and save
    if new_entries:
        index_data.setdefault("clips", []).extend(new_entries)
        save_json(output_path, index_data)
        log.info("Saved %d new clip(s) → %s", len(new_entries), output_path)

    # Report unmatched CSV rows
    if mode == "csv" and csv_tags:
        indexed_names = {e["filename"] for e in new_entries}
        unmatched = set(csv_tags.keys()) - indexed_names - existing
        if unmatched:
            log.warning(
                "%d CSV row(s) did not match any video file: %s",
                len(unmatched), ", ".join(sorted(unmatched)),
            )

    return new_entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan and index a clip library for the AI Explainer pipeline.",
    )
    parser.add_argument(
        "-d", "--directory", required=True,
        help="Directory to scan for video files",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-i", "--interactive", action="store_true",
        help="Interactively tag each clip",
    )
    mode_group.add_argument(
        "-a", "--auto-tag", action="store_true",
        help="Auto-tag clips using Ollama vision model",
    )
    mode_group.add_argument(
        "-c", "--csv", default=None, metavar="FILE",
        help="Import tags from a CSV file",
    )

    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path for clip_index.json (default from pipeline config)",
    )
    parser.add_argument(
        "-s", "--show", default=None,
        help="Show slug to associate with clips (e.g. 'rick_and_morty')",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.interactive:
        mode = "interactive"
    elif args.auto_tag:
        mode = "auto-tag"
    elif args.csv:
        mode = "csv"
    else:
        mode = "interactive"

    entries = index_clips(
        directory=args.directory,
        mode=mode,
        output_path=args.output,
        show_slug=args.show,
        csv_path=args.csv,
    )

    print(f"\n✅ Indexed {len(entries)} new clip(s)")


if __name__ == "__main__":
    main()
