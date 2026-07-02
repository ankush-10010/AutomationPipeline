"""
enrich_clip_metadata.py — One-shot metadata enrichment for existing clip_index.json.

Fixes four data quality gaps in a single pass over the clip index:
  1. Propagates episode_summary from episode_index.json into each clip.
  2. Strips SRT speaker labels (>> name:) from action text, stores cleaned
     dialogue and parsed speakers separately.
  3. Re-runs character alias matching on cleaned text (no speaker label noise).
  4. Detects alien transformation mentions from dialogue + show_config aliases.

After enrichment, re-computes semantic embeddings with the richer text.

Usage:
    python scripts/enrich_clip_metadata.py
    python scripts/enrich_clip_metadata.py --show ben10 --skip-embed
"""

import argparse
import json
import re
import sys
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

log = setup_logging("enrich_clip_metadata")

# -- SRT speaker label pattern ------------------------------------------------
_SPEAKER_LABEL_RE = re.compile(r">>\s*\w[\w\s]*?:", re.IGNORECASE)


def strip_speaker_labels(text: str) -> str:
    """Remove SRT speaker labels like '>> kevin:' from dialogue text."""
    return _SPEAKER_LABEL_RE.sub("", text).strip()


def extract_speakers(text: str) -> list:
    """Parse SRT speaker labels into a deduplicated list of speaker names."""
    raw = _SPEAKER_LABEL_RE.findall(text)
    seen = set()
    speakers = []
    for m in raw:
        name = m.strip().lstrip(">").strip().rstrip(":").strip().lower()
        if name and name not in seen:
            seen.add(name)
            speakers.append(name)
    return speakers


# -- Episode summary mapping ---------------------------------------------------

def build_episode_summary_map(episode_index_path: Path) -> dict:
    """Build a mapping of 's{season}e{episode}' -> summary text.

    The episode_index.json has title -> "Season X, Episode Y. summary..." format.
    We parse the season/episode from the value string and build the lookup key.
    """
    data = load_json(episode_index_path)
    if not data:
        log.warning("Episode index is empty or missing: %s", episode_index_path)
        return {}

    ep_map = {}
    pattern = re.compile(r"Season\s+(\d+),?\s*Episode\s+(\d+)", re.IGNORECASE)

    if isinstance(data, dict):
        for title, summary_text in data.items():
            m = pattern.search(str(summary_text))
            if m:
                key = f"s{int(m.group(1))}e{int(m.group(2))}"
                ep_map[key] = f"{title}: {summary_text}"
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("content", ""))
                full = f"{title}: {summary}" if title else summary
                m = pattern.search(full)
                if m:
                    key = f"s{int(m.group(1))}e{int(m.group(2))}"
                    ep_map[key] = full

    log.info("Built episode summary map with %d entries: %s",
             len(ep_map), sorted(ep_map.keys()))
    return ep_map


def parse_clip_episode_key(filename: str) -> str:
    """Extract 's1e1' prefix from clip filenames like 's1e1_scene_001.mp4'."""
    m = re.match(r"(s\d+e\d+)", filename, re.IGNORECASE)
    return m.group(1).lower() if m else ""


# -- Transformation detection --------------------------------------------------

def build_transformation_set(show_config: dict) -> dict:
    """Build a map of lowercase alien name -> canonical name from Ben's aliases."""
    transforms = {}
    for char in show_config.get("characters", []):
        if char.get("name", "") != "Ben Tennyson":
            continue
        for alias in char.get("aliases", []):
            clean = alias.strip()
            if clean.lower() not in ("ben",):  # skip "Ben" itself
                transforms[clean.lower()] = clean
    return transforms


def detect_transformations(text: str, transform_map: dict) -> list:
    """Find alien transformation names mentioned in text."""
    text_lower = text.lower()
    found = []
    for alias_lower, canonical in transform_map.items():
        # Word boundary match to avoid partial matches
        if re.search(rf"\b{re.escape(alias_lower)}\b", text_lower):
            if canonical not in found:
                found.append(canonical)
    return found


# -- Character alias matching (clean version) ----------------------------------

def build_alias_map(show_config: dict) -> dict:
    """Build regex pattern -> canonical character name map."""
    alias_map = {}
    for char_entry in show_config.get("characters", []):
        canon_name = char_entry.get("name", "").strip()
        if not canon_name:
            continue
        terms = [canon_name] + char_entry.get("aliases", [])
        for term in terms:
            clean_term = term.strip()
            if len(clean_term) >= 2:
                escaped = re.escape(clean_term.lower())
                pattern = rf"\b{escaped}\b"
                alias_map[pattern] = canon_name
    return alias_map


def match_characters(text: str, alias_map: dict) -> set:
    """Match character aliases against text, returning canonical names."""
    text_lower = text.lower()
    matched = set()
    for pattern, canon_name in alias_map.items():
        if re.search(pattern, text_lower):
            matched.add(canon_name)
    return matched


# -- Main enrichment pass ------------------------------------------------------

def enrich(
    clip_index_path: Path,
    episode_index_path: Path,
    show_config: dict,
    skip_embed: bool = False,
):
    log.info("Loading clip index from %s ...", clip_index_path)
    clip_data = load_json(clip_index_path)
    if isinstance(clip_data, dict):
        clips = clip_data.get("clips", [])
    elif isinstance(clip_data, list):
        clips = clip_data
    else:
        log.error("Invalid clip index format")
        return

    log.info("Loaded %d clips", len(clips))

    # Build lookup structures
    ep_map = build_episode_summary_map(episode_index_path)
    alias_map = build_alias_map(show_config)
    transform_map = build_transformation_set(show_config)

    log.info("Alias patterns: %d, Transformation aliases: %d",
             len(alias_map), len(transform_map))

    # Stats
    stats = {
        "episode_summary_added": 0,
        "speakers_extracted": 0,
        "characters_fixed": 0,
        "transformations_tagged": 0,
    }

    for i, clip in enumerate(clips):
        action_raw = clip.get("action", "")

        # 1. Episode summary propagation
        ep_key = parse_clip_episode_key(clip.get("filename", ""))
        if ep_key and ep_key in ep_map and not clip.get("episode_summary"):
            clip["episode_summary"] = ep_map[ep_key]
            stats["episode_summary_added"] += 1

        # 2. Speaker label extraction + cleaning
        speakers = extract_speakers(action_raw)
        if speakers:
            clip["speakers"] = speakers
            stats["speakers_extracted"] += 1

        action_clean = strip_speaker_labels(action_raw)

        # 3. Character re-matching on clean text (no speaker labels)
        tags_text = " ".join(clip.get("tags", [])) if isinstance(clip.get("tags"), list) else ""
        summary_text = clip.get("episode_summary", "")
        corpus = f"{action_clean} {tags_text} {summary_text}"

        new_chars = match_characters(corpus, alias_map)

        # Preserve YOLO-detected characters, but rebuild the list from clean matching
        # YOLO characters are the ones that were there before any enrichment
        # We keep them and add clean-matched ones
        existing_chars = set()
        for c in clip.get("characters", []):
            existing_chars.add(c)

        # Replace the character list: keep YOLO detections, add clean matches, remove false positives
        # A false positive is a character found ONLY via speaker label and not in clean text
        combined = new_chars | existing_chars
        # Re-verify existing chars against clean corpus
        verified = set()
        for char in combined:
            char_lower = char.lower()
            # Check if this character is mentioned in the clean corpus
            if re.search(rf"\b{re.escape(char_lower)}\b", corpus.lower()):
                verified.add(char)
            elif char in existing_chars:
                # Keep YOLO detections even if not in text (visual detection)
                # But only if it's not purely a speaker-label artifact
                if char_lower not in [s for s in speakers]:
                    verified.add(char)

        if verified != set(clip.get("characters", [])):
            stats["characters_fixed"] += 1

        clip["characters"] = sorted(list(verified))

        # 4. Transformation detection
        transforms = detect_transformations(f"{action_clean} {tags_text}", transform_map)
        if transforms:
            clip["transformations"] = transforms
            stats["transformations_tagged"] += 1

        if (i + 1) % 2000 == 0:
            log.info("  Processed %d/%d clips...", i + 1, len(clips))

    log.info("Enrichment pass complete:")
    log.info("  Episode summaries added: %d", stats["episode_summary_added"])
    log.info("  Speakers extracted: %d", stats["speakers_extracted"])
    log.info("  Characters fixed: %d", stats["characters_fixed"])
    log.info("  Transformations tagged: %d", stats["transformations_tagged"])

    # 5. Re-compute embeddings with enriched text
    if not skip_embed:
        log.info("Loading SentenceTransformer for re-embedding...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            log.error("sentence-transformers not installed! Saving without re-embedding.")
            skip_embed = True

        if not skip_embed:
            log.info("Re-embedding %d clips with enriched metadata...", len(clips))
            for i, clip in enumerate(clips):
                chars_str = ", ".join(clip.get("characters", []))
                action_clean = strip_speaker_labels(clip.get("action", ""))
                transforms_str = ", ".join(clip.get("transformations", []))
                ep_summary = clip.get("episode_summary", "")

                # Richer embedding text
                parts = []
                if chars_str:
                    parts.append(f"Characters: {chars_str}")
                if transforms_str:
                    parts.append(f"Alien forms: {transforms_str}")
                parts.append(f"Scene: {action_clean}")
                if ep_summary:
                    # Use first 200 chars of episode summary to not dominate
                    parts.append(f"Episode: {ep_summary[:200]}")

                text_to_embed = ". ".join(parts)
                clip["embedding"] = model.encode(text_to_embed).tolist()

                if (i + 1) % 500 == 0 or (i + 1) == len(clips):
                    log.info("  Embedded [%d/%d] clips...", i + 1, len(clips))

    # Save
    if isinstance(clip_data, dict):
        clip_data["clips"] = clips
    save_json(clip_index_path, clip_data)
    log.info("Saved enriched clip index -> %s", clip_index_path)


def main():
    parser = argparse.ArgumentParser(
        description="One-shot metadata enrichment for clip_index.json"
    )
    parser.add_argument("--show", default=None, help="Show identifier")
    parser.add_argument("--index", default=None, help="Path to clip_index.json")
    parser.add_argument("--episode-index", default=None, help="Path to episode_index.json")
    parser.add_argument(
        "--skip-embed", action="store_true",
        help="Skip re-computing embeddings (faster for testing)"
    )
    args = parser.parse_args()

    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    clip_index_path = Path(args.index) if args.index else get_project_path("clip_index", pipeline_cfg)

    if args.episode_index:
        episode_index_path = Path(args.episode_index)
    else:
        ep_cfg = pipeline_cfg.get("episode_index", {})
        episode_index_path = (PROJECT_ROOT / ep_cfg.get("path", "episode_index.json")).resolve()
        # Also try the topics path from pipeline_config
        if not episode_index_path.exists():
            episode_index_path = get_project_path("episode_index", pipeline_cfg)

    log.info("Show: %s", show_config.get("display_name", show_slug))
    log.info("Clip index: %s", clip_index_path)
    log.info("Episode index: %s", episode_index_path)

    enrich(clip_index_path, episode_index_path, show_config, skip_embed=args.skip_embed)


if __name__ == "__main__":
    main()
