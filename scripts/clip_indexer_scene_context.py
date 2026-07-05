"""
clip_indexer_scene_context.py — LLM-based meaningful subtitle chunking + scene context.

Instead of raw overlapping dialogue fragments, this script:
  1. Reads the full SRT for each episode.
  2. Groups subtitle entries into meaningful narrative scenes using timecode
     gaps (a gap > threshold = scene break).
  3. Sends each scene chunk to an LLM to produce:
       - visual_description: What's likely on screen (1-2 sentences)
       - emotion_tone: The emotional register (e.g. tense, comedic, action, quiet)
  4. Maps scene chunks back to clips by timecode overlap.
  5. Writes scene_context, visual_description, and emotion_tone into clip_index.json.

Dependencies:
    pip install requests

Usage:
    python scripts/clip_indexer_scene_context.py --show ben10
    python scripts/clip_indexer_scene_context.py --episode s1e1
    python scripts/clip_indexer_scene_context.py --scene-gap 3.0 --skip-llm
"""

import argparse
import json
import re
import sys
import time
import requests
from pathlib import Path

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

log = setup_logging("scene_context")


# ── SRT Parsing ──────────────────────────────────────────────────────────────

def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format (00:00:02,000) to seconds."""
    h, m, s_ms = time_str.strip().split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def load_srt(srt_path: Path) -> list:
    """Parse an SRT file into a list of {start, end, text} dicts."""
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = content.strip().split("\n\n")
    subs = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            times = lines[1].split(" --> ")
            if len(times) == 2:
                try:
                    start = parse_srt_time(times[0])
                    end = parse_srt_time(times[1])
                    text = " ".join(lines[2:]).replace("\n", " ")
                    text = re.sub(r"<[^>]+>", "", text)  # Strip HTML tags
                    if text.strip():
                        subs.append({"start": start, "end": end, "text": text.strip()})
                except Exception:
                    continue

    return subs


# ── Scene Chunking ───────────────────────────────────────────────────────────

def chunk_into_scenes(subs: list, gap_threshold: float = 3.0,
                      max_chunk_duration: float = 30.0) -> list:
    """Group subtitle entries into scene-level chunks.

    A new scene starts when:
      - The gap between consecutive subtitles exceeds gap_threshold seconds
      - The accumulated chunk duration exceeds max_chunk_duration seconds

    Returns a list of scene dicts:
      {start, end, text, subtitle_count}
    """
    if not subs:
        return []

    scenes = []
    current_texts = [subs[0]["text"]]
    current_start = subs[0]["start"]
    current_end = subs[0]["end"]

    for i in range(1, len(subs)):
        gap = subs[i]["start"] - current_end
        duration = subs[i]["end"] - current_start

        if gap > gap_threshold or duration > max_chunk_duration:
            # Flush current scene
            scenes.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_texts),
                "subtitle_count": len(current_texts),
            })
            current_texts = [subs[i]["text"]]
            current_start = subs[i]["start"]
            current_end = subs[i]["end"]
        else:
            current_texts.append(subs[i]["text"])
            current_end = subs[i]["end"]

    # Flush last scene
    if current_texts:
        scenes.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_texts),
            "subtitle_count": len(current_texts),
        })

    return scenes


# ── LLM Scene Description ───────────────────────────────────────────────────

def describe_scene_llm(
    scene_text: str,
    episode_key: str,
    show_name: str,
    ollama_url: str,
    model: str,
    timeout: int = 60,
) -> dict:
    """Ask LLM to infer visual description and emotion from dialogue.

    Returns {visual_description, emotion_tone} or empty strings on failure.
    """
    prompt = (
        f"You are analyzing a scene from the animated show '{show_name}', "
        f"episode {episode_key.upper()}.\n\n"
        f"DIALOGUE FROM THIS SCENE:\n\"{scene_text[:2000]}\"\n\n"
        f"Based ONLY on this dialogue, answer in this exact JSON format:\n"
        f'{{"visual_description": "1-2 sentence description of what is most likely '
        f'visually happening on screen during this dialogue", '
        f'"emotion_tone": "one word: action, tense, comedic, dramatic, quiet, '
        f'mysterious, or emotional"}}\n\n'
        f"Reply with ONLY the JSON object, nothing else."
    )

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 256},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()

        # Parse JSON from response (handle markdown code blocks)
        answer = re.sub(r"```json\s*", "", answer)
        answer = re.sub(r"```\s*$", "", answer)

        # Find JSON object in response
        json_match = re.search(r"\{[^}]+\}", answer)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "visual_description": parsed.get("visual_description", ""),
                "emotion_tone": parsed.get("emotion_tone", ""),
            }
    except Exception as e:
        log.debug("LLM scene description failed: %s", e)

    return {"visual_description": "", "emotion_tone": ""}


# ── Timecode Reconstruction ─────────────────────────────────────────────────

def reconstruct_clip_timecodes(clips: list, episode_key: str) -> list:
    """Reconstruct approximate start/end times for clips from an episode.

    Uses sequential scene numbering and durations to estimate timecodes.
    Falls back to manifest files if available.

    Returns list of (clip_index, start_sec, end_sec) tuples.
    """
    ep_clips = []
    for i, clip in enumerate(clips):
        fname = clip.get("filename", "")
        if not fname.lower().startswith(episode_key):
            continue
        # Extract scene number for ordering
        m = re.search(r"scene_(\d+)", fname)
        scene_num = int(m.group(1)) if m else 0
        ep_clips.append((i, scene_num, clip.get("duration_seconds", 0)))

    # Sort by scene number
    ep_clips.sort(key=lambda x: x[1])

    result = []
    current_time = 0.0
    for idx, scene_num, duration in ep_clips:
        result.append((idx, current_time, current_time + duration))
        current_time += duration

    return result


def load_manifest_timecodes(clips_dir: Path, episode_key: str) -> dict:
    """Try to load timecodes from the scene_splitter manifest file.

    Returns {filename: (start_sec, end_sec)} or empty dict.
    """
    manifest_path = clips_dir / f"{episode_key}_manifest.json"
    if not manifest_path.exists():
        # Try subdirectory
        manifest_path = clips_dir / episode_key / f"{episode_key}_manifest.json"
    if not manifest_path.exists():
        # Search recursively
        found = list(clips_dir.rglob(f"{episode_key}_manifest.json"))
        if found:
            manifest_path = found[0]
        else:
            return {}

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        timecodes = {}
        for fname, times in data.items():
            timecodes[fname] = (times.get("start_sec", 0), times.get("end_sec", 0))
        return timecodes
    except Exception as e:
        log.warning("Failed to read manifest %s: %s", manifest_path, e)
        return {}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based subtitle chunking and scene context enrichment."
    )
    parser.add_argument("--index", default=None, help="Path to clip_index.json")
    parser.add_argument("--show", default=None, help="Show identifier")
    parser.add_argument(
        "--episode", default=None,
        help="Process only this episode prefix (e.g. s1e1)"
    )
    parser.add_argument(
        "--scene-gap", type=float, default=3.0,
        help="Seconds of silence that defines a scene break (default: 3.0)"
    )
    parser.add_argument(
        "--max-chunk-duration", type=float, default=30.0,
        help="Maximum scene chunk duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM descriptions (only do subtitle chunking and mapping)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process clips that already have scene_context"
    )
    args = parser.parse_args()

    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    clip_index_path = (
        Path(args.index) if args.index
        else get_project_path("clip_index", pipeline_cfg)
    )
    subtitles_dir = get_project_path("subtitles_dir", pipeline_cfg)
    clips_dir = Path(
        show_config.get("clips_dir", f"./clips/{show_slug}")
    )
    if not clips_dir.is_absolute():
        clips_dir = (PROJECT_ROOT / show_config.get("clips_dir", f"./clips/{show_slug}")).resolve()

    # LLM settings
    llm_cfg = pipeline_cfg.get("llm", {})
    ollama_url = llm_cfg.get("base_url",
                             llm_cfg.get("ollama", {}).get("base_url", "http://localhost:11434"))
    ollama_model = llm_cfg.get("model",
                               llm_cfg.get("ollama", {}).get("model", "llama3.1:8b"))
    show_name = show_config.get("display_name", "Ben 10")

    log.info("Subtitles dir: %s", subtitles_dir)
    log.info("Clips dir: %s", clips_dir)
    log.info("LLM: %s @ %s", ollama_model, ollama_url)

    # Load clip index
    clip_data = load_json(clip_index_path)
    if isinstance(clip_data, dict):
        clips = clip_data.get("clips", [])
    elif isinstance(clip_data, list):
        clips = clip_data
    else:
        log.error("Invalid clip index format")
        sys.exit(1)

    log.info("Loaded %d clips", len(clips))

    # Discover episodes to process
    ep_pattern = re.compile(r"s(\d+)e(\d+)", re.IGNORECASE)
    all_episodes = set()
    for clip in clips:
        m = ep_pattern.search(clip.get("filename", ""))
        if m:
            all_episodes.add(f"s{int(m.group(1))}e{int(m.group(2))}")

    if args.episode:
        m_arg = ep_pattern.search(args.episode)
        if m_arg:
            episodes_to_process = [f"s{int(m_arg.group(1))}e{int(m_arg.group(2))}"]
        else:
            episodes_to_process = [args.episode.lower()]
    else:
        episodes_to_process = sorted(all_episodes, key=lambda x: (int(x.split('e')[0][1:]), int(x.split('e')[1])))

    log.info("Episodes to process: %d", len(episodes_to_process))

    # Find SRT files
    srt_files = {}
    if subtitles_dir.exists():
        for srt_path in subtitles_dir.rglob("*.srt"):
            m = ep_pattern.search(srt_path.name)
            if m:
                srt_files[f"s{int(m.group(1))}e{int(m.group(2))}"] = srt_path

    log.info("Found %d SRT files", len(srt_files))

    total_scenes_processed = 0
    total_clips_enriched = 0

    for ep_key in episodes_to_process:
        if ep_key not in srt_files:
            log.warning("No SRT file found for %s, skipping", ep_key)
            continue

        srt_path = srt_files[ep_key]
        log.info("Processing %s from %s", ep_key.upper(), srt_path.name)

        # 1. Parse SRT
        subs = load_srt(srt_path)
        if not subs:
            log.warning("No subtitles parsed from %s", srt_path)
            continue

        # 2. Chunk into scenes
        scenes = chunk_into_scenes(
            subs,
            gap_threshold=args.scene_gap,
            max_chunk_duration=args.max_chunk_duration,
        )
        log.info("  %s: %d subtitles -> %d scene chunks", ep_key, len(subs), len(scenes))

        # 3. LLM descriptions for each scene
        if not args.skip_llm:
            log.info("  Generating LLM descriptions for %d scenes...", len(scenes))
            for i, scene in enumerate(scenes):
                desc = describe_scene_llm(
                    scene["text"], ep_key, show_name,
                    ollama_url, ollama_model,
                )
                scene["visual_description"] = desc.get("visual_description", "")
                scene["emotion_tone"] = desc.get("emotion_tone", "")

                if (i + 1) % 10 == 0:
                    log.info("    Described %d/%d scenes", i + 1, len(scenes))
                # Small delay to avoid hammering Ollama
                time.sleep(0.1)

        total_scenes_processed += len(scenes)

        # 4. Get clip timecodes (try manifest first, then reconstruct)
        manifest_times = load_manifest_timecodes(clips_dir, ep_key)

        if manifest_times:
            log.info("  Using manifest timecodes for %d clips", len(manifest_times))
            clip_timecodes = []
            for i, clip in enumerate(clips):
                fname = clip.get("filename", "")
                if fname in manifest_times:
                    start, end = manifest_times[fname]
                    clip_timecodes.append((i, start, end))
        else:
            log.info("  No manifest found, reconstructing timecodes from durations")
            clip_timecodes = reconstruct_clip_timecodes(clips, ep_key)

        # 5. Map scenes to clips by timecode overlap
        for idx, clip_start, clip_end in clip_timecodes:
            clip = clips[idx]

            if not args.force and clip.get("scene_context"):
                continue

            # Find all scenes that overlap with this clip's time range
            overlapping_scenes = []
            for scene in scenes:
                if scene["start"] < clip_end and scene["end"] > clip_start:
                    # Calculate overlap ratio
                    overlap_start = max(scene["start"], clip_start)
                    overlap_end = min(scene["end"], clip_end)
                    overlap_dur = overlap_end - overlap_start
                    clip_dur = clip_end - clip_start
                    if clip_dur > 0:
                        ratio = overlap_dur / clip_dur
                    else:
                        ratio = 0
                    overlapping_scenes.append((scene, ratio))

            if not overlapping_scenes:
                continue

            # Use the scene with the most overlap
            overlapping_scenes.sort(key=lambda x: x[1], reverse=True)
            best_scene = overlapping_scenes[0][0]

            # Write scene context (the meaningful dialogue chunk)
            clip["scene_context"] = best_scene["text"]

            # Write LLM-generated fields if available
            if best_scene.get("visual_description"):
                clip["visual_description"] = best_scene["visual_description"]
            if best_scene.get("emotion_tone"):
                clip["emotion_tone"] = best_scene["emotion_tone"]

            total_clips_enriched += 1

        log.info("  %s: enriched %d clips with scene context", ep_key, total_clips_enriched)

        # Checkpoint after each episode
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(clip_index_path, clip_data)
        log.info("  Checkpoint saved after %s", ep_key)

    log.info("Scene context enrichment complete:")
    log.info("  Total scene chunks: %d", total_scenes_processed)
    log.info("  Total clips enriched: %d", total_clips_enriched)


if __name__ == "__main__":
    main()
