"""
enrich_clip_characters.py — Enrich clip_index.json characters using dialogue & show_config aliases.

Scans subtitle transcripts (clip['action']), tags, and summaries against canonical character names
and aliases from show_config.yaml. Appends new character detections to existing YOLO tags
without duplicate entries, then automatically updates semantic vector embeddings.

Usage:
    python scripts/enrich_clip_characters.py
    python scripts/enrich_clip_characters.py --index clip_index.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config,
    get_active_show,
    get_project_path,
    load_json,
    save_json,
    setup_logging,
)

log = setup_logging("enrich_clip_characters")


def build_alias_map(show_config: dict) -> dict:
    """Build a map of lowercase regex word boundary pattern -> canonical character name."""
    alias_map = {}
    for char_entry in show_config.get("characters", []):
        canon_name = char_entry.get("name", "").strip()
        if not canon_name:
            continue
        
        # Add canonical name itself
        terms = [canon_name] + char_entry.get("aliases", [])
        for term in terms:
            clean_term = term.strip()
            if len(clean_term) >= 2:
                # Escape special regex characters in names (like Mr. Poopybutthole or C-137)
                escaped = re.escape(clean_term.lower())
                pattern = rf"\b{escaped}\b"
                alias_map[pattern] = canon_name
                
    return alias_map


def enrich_characters(index_path: Path, show_config: dict):
    if not index_path.exists():
        log.error("Clip index not found: %s", index_path)
        sys.exit(1)

    clip_data = load_json(index_path)
    if isinstance(clip_data, dict):
        clips = clip_data.get("clips", [])
    elif isinstance(clip_data, list):
        clips = clip_data
    else:
        clips = []

    if not clips:
        log.warning("No clips found in %s", index_path)
        return

    alias_map = build_alias_map(show_config)
    log.info("Built alias matching engine for %d patterns across show characters", len(alias_map))

    enriched_clips = 0
    new_chars_added = 0

    for clip in clips:
        # 1. Preserve existing YOLO detections
        existing = clip.get("characters", [])
        existing_lower = {str(c).lower().strip(): str(c).strip() for c in existing if str(c).strip()}
        
        # Combined text corpus for this clip
        action_text = clip.get("action", "")
        tags_text = " ".join(clip.get("tags", [])) if isinstance(clip.get("tags"), list) else str(clip.get("tags", ""))
        summary_text = clip.get("episode_summary", "")
        corpus = f"{action_text} {tags_text} {summary_text}".lower()

        matched_canonical = set()
        for pattern, canon_name in alias_map.items():
            if re.search(pattern, corpus):
                matched_canonical.add(canon_name)

        # Append new matches without removing YOLO tags or adding duplicates
        added_for_clip = False
        for canon in matched_canonical:
            canon_lower = canon.lower()
            if canon_lower not in existing_lower:
                existing_lower[canon_lower] = canon
                new_chars_added += 1
                added_for_clip = True

        if added_for_clip:
            enriched_clips += 1

        # Sorted unified character list
        clip["characters"] = sorted(list(existing_lower.values()))

    log.info("Character enrichment pass complete ✓")
    log.info("Updated %d clips, appended %d total new character tags", enriched_clips, new_chars_added)

    # 2. Re-compute Semantic Embeddings
    log.info("Loading SentenceTransformer model to update semantic embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        log.error("sentence-transformers not installed! Saving character tags without embeddings.")
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(index_path, clip_data)
        return

    log.info("Re-embedding %d clips with unified character & action text...", len(clips))
    for i, clip in enumerate(clips, 1):
        chars_str = ", ".join(clip.get("characters", []))
        action_str = clip.get("action", "")
        text_to_embed = f"Characters: {chars_str}. Dialogue/Action: {action_str}"
        clip["embedding"] = model.encode(text_to_embed).tolist()

        if i % 50 == 0 or i == len(clips):
            log.info("  Embedded [%d/%d] clips...", i, len(clips))

    # Save back to disk
    if isinstance(clip_data, dict):
        clip_data["clips"] = clips
    save_json(index_path, clip_data)
    log.info("Saved fully enriched clip database -> %s", index_path)


def main():
    parser = argparse.ArgumentParser(description="Enrich clip characters from subtitle dialogue")
    parser.add_argument("--index", default=None, help="Path to clip_index.json")
    parser.add_argument("--show", default=None, help="Show identifier")
    args = parser.parse_args()

    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    if args.index:
        index_path = Path(args.index)
    else:
        index_path = get_project_path("clip_index", pipeline_cfg)

    log.info("Enriching clip database for show: %s", show_config.get("display_name", show_slug))
    enrich_characters(index_path, show_config)


if __name__ == "__main__":
    main()
