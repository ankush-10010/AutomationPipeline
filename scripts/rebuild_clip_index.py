"""
===============================================================================
MASTER CLIP INDEX REBUILD — rebuild_clip_index.py
===============================================================================

PURPOSE:
    Builds a COMPLETE, FRESH clip_index.json with ALL clips on disk (~28,000+),
    including silent/action-only clips that were previously skipped.

STRATEGY:
    Phase A: Scan ALL mp4 files on disk + match with manifests for timecodes
    Phase B: Match with SRT subtitles (but NEVER skip silent clips)
    Phase C: Merge rich tags from old backup (visual_characters, prototype_detections, etc.)
    Phase D: Extract season/episode metadata from filenames

    After this script, the index will have basic fields for ALL clips.
    Then run the enrichment pipeline to fill in the rest.

USAGE:
    python scripts/rebuild_clip_index.py
    python scripts/rebuild_clip_index.py --dry-run        # Preview without writing
    python scripts/rebuild_clip_index.py --no-merge       # Skip merging from old backup

===============================================================================
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = PROJECT_ROOT / "clips" / "ben10"
SUBTITLES_DIR = PROJECT_ROOT / "ben10_subtitles"
OUTPUT_INDEX = PROJECT_ROOT / "clip_index.json"
OLD_BACKUP = PROJECT_ROOT / "clip_index_old_backup.json"
SHOW_SLUG = "ben10"

# Stop words for tag generation
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "like",
    "through", "after", "over", "between", "out", "up", "down", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "no", "than", "too", "very", "just", "because", "its", "it",
    "this", "that", "these", "those", "my", "your", "his", "her", "our",
    "their", "what", "which", "who", "whom", "when", "where", "why", "how",
    "im", "dont", "youre", "hes", "shes", "theyre", "weve", "ive",
    "he", "she", "we", "you", "they", "me", "him", "us", "them",
}


# ---------------------------------------------------------------------------
# SRT Parser
# ---------------------------------------------------------------------------
def parse_srt(srt_path: Path) -> list:
    """Parse an SRT file into a list of {start, end, text} dicts."""
    if not srt_path.exists():
        return []

    content = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Find the timecode line (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
        tc_match = None
        tc_idx = -1
        for idx, line in enumerate(lines):
            tc_match = re.match(
                r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
                line.strip()
            )
            if tc_match:
                tc_idx = idx
                break

        if not tc_match:
            continue

        g = tc_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        text = " ".join(lines[tc_idx + 1:]).strip()
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        if text:
            entries.append({"start": start, "end": end, "text": text})

    return entries


# ---------------------------------------------------------------------------
# Keyword Generator
# ---------------------------------------------------------------------------
def generate_keywords(text: str) -> list:
    """Generate keyword tags from text."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    unique = []
    seen = set()
    for w in words:
        if len(w) > 2 and w not in STOP_WORDS and w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


# ---------------------------------------------------------------------------
# Season/Episode Extractor
# ---------------------------------------------------------------------------
def extract_season_episode(filename: str) -> tuple:
    """Extract season and episode numbers from clip filename like s1e10_scene_001.mp4."""
    match = re.match(r"s(\d+)e(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1, 1


# ---------------------------------------------------------------------------
# SRT Filename Matcher
# ---------------------------------------------------------------------------
def find_srt_for_episode(season: int, episode: int, srt_dir: Path) -> Path | None:
    """Find the matching SRT file for a given season/episode."""
    # Try common naming patterns
    patterns = [
        f"Ben_10_Classic_S{season:02d}E{episode:02d}.srt",
        f"Ben_10_Classic_S{season}E{episode}.srt",
        f"s{season}e{episode}.srt",
        f"S{season:02d}E{episode:02d}.srt",
    ]
    for p in patterns:
        path = srt_dir / p
        if path.exists():
            return path

    # Fallback: search for any file containing the episode pattern
    for f in srt_dir.iterdir():
        if f.suffix.lower() == ".srt":
            if f"S{season:02d}E{episode:02d}" in f.name or f"s{season}e{episode}" in f.name.lower():
                return f

    return None


# ---------------------------------------------------------------------------
# Main Rebuild Logic
# ---------------------------------------------------------------------------
def rebuild_index(dry_run: bool = False, merge_backup: bool = True):
    print("=" * 80)
    print("  MASTER CLIP INDEX REBUILD")
    print("=" * 80)

    # ── Step 1: Scan all MP4 files on disk ─────────────────────────────
    print("\n📂 Step 1: Scanning clips directory for ALL mp4 files...")
    mp4_files = sorted(CLIPS_DIR.glob("*.mp4"))
    print(f"   Found {len(mp4_files)} mp4 files on disk")

    if not mp4_files:
        print("❌ No mp4 files found! Check CLIPS_DIR path.")
        sys.exit(1)

    # ── Step 2: Load all manifests for timecode data ───────────────────
    print("\n📋 Step 2: Loading manifest files for timecodes...")
    manifests = {}
    manifest_files = sorted(CLIPS_DIR.glob("*_manifest.json"))
    for mf in manifest_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifests.update(data)
        except Exception as e:
            print(f"   ⚠️ Failed to load {mf.name}: {e}")
    print(f"   Loaded timecodes for {len(manifests)} clips from {len(manifest_files)} manifests")

    # ── Step 3: Load and parse all SRT files by episode ────────────────
    print("\n🗣️  Step 3: Loading subtitle files...")
    srt_cache = {}  # key: (season, episode) -> list of subtitle entries
    episodes_found = set()
    for mp4 in mp4_files:
        se = extract_season_episode(mp4.name)
        episodes_found.add(se)

    for season, episode in sorted(episodes_found):
        srt_path = find_srt_for_episode(season, episode, SUBTITLES_DIR)
        if srt_path:
            srt_cache[(season, episode)] = parse_srt(srt_path)
        else:
            srt_cache[(season, episode)] = []

    srt_count = sum(1 for v in srt_cache.values() if v)
    print(f"   Matched SRT files for {srt_count}/{len(episodes_found)} episodes")

    # ── Step 4: Load old backup for merging ────────────────────────────
    old_backup_map = {}
    if merge_backup and OLD_BACKUP.exists():
        print("\n🔄 Step 4: Loading old backup for rich tag merging...")
        try:
            with open(OLD_BACKUP, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_clips = old_data.get("clips", old_data) if isinstance(old_data, dict) else old_data
            for c in old_clips:
                old_backup_map[c["filename"]] = c
            print(f"   Loaded {len(old_backup_map)} clips from old backup")
        except Exception as e:
            print(f"   ⚠️ Failed to load old backup: {e}")
    else:
        print("\n⏭️  Step 4: Skipping old backup merge")

    # ── Step 5: Build fresh index ──────────────────────────────────────
    print("\n🔨 Step 5: Building fresh clip index...")

    # Rich fields to merge from old backup
    MERGE_FIELDS = [
        "visual_characters", "prototype_detections", "scene_context",
        "visual_description", "emotion_tone", "clip_visual_embedding",
        "visual_tags", "yolo_arcface", "transformations", "speakers",
        "raw_vision", "episode_summary",
    ]

    new_clips = []
    stats = {
        "total": 0,
        "with_dialogue": 0,
        "silent": 0,
        "merged_from_backup": 0,
        "with_manifest_timecodes": 0,
    }

    for mp4 in mp4_files:
        fname = mp4.name
        season, episode = extract_season_episode(fname)
        stats["total"] += 1

        # Get timecodes from manifest
        manifest_entry = manifests.get(fname, {})
        start_sec = manifest_entry.get("start_sec", 0.0)
        end_sec = manifest_entry.get("end_sec", 0.0)
        duration = round(end_sec - start_sec, 2) if end_sec > start_sec else 0.0

        if manifest_entry:
            stats["with_manifest_timecodes"] += 1

        # If we don't have manifest data, try to get duration from ffprobe-style estimate
        if duration <= 0:
            # We'll set duration to 0 and let enrichment scripts fix it later
            duration = 0.0

        # Match subtitles
        subs = srt_cache.get((season, episode), [])
        overlapping_text = []
        if subs and end_sec > start_sec:
            for sub in subs:
                if sub["start"] < end_sec and sub["end"] > start_sec:
                    overlapping_text.append(sub["text"])

        action_text = " ".join(overlapping_text) if overlapping_text else ""
        tags = generate_keywords(action_text) if action_text else []

        if overlapping_text:
            stats["with_dialogue"] += 1
        else:
            stats["silent"] += 1

        # Build the clip entry
        clip_entry = {
            "filename": fname,
            "show": SHOW_SLUG,
            "season": season,
            "episode": episode,
            "characters": [],
            "location": "",
            "action": action_text,
            "mood": "",
            "tags": tags,
            "duration_seconds": duration,
        }

        # Merge rich fields from old backup if available
        old_clip = old_backup_map.get(fname)
        if old_clip:
            stats["merged_from_backup"] += 1

            # Merge characters (prefer old backup's richer data)
            old_chars = old_clip.get("characters", [])
            if old_chars:
                clip_entry["characters"] = old_chars

            # Merge the old embedding if present
            old_emb = old_clip.get("embedding")
            if old_emb:
                clip_entry["embedding"] = old_emb

            # Merge all rich fields
            for field in MERGE_FIELDS:
                old_val = old_clip.get(field)
                if old_val is not None and old_val != "" and old_val != [] and old_val != {}:
                    clip_entry[field] = old_val

            # Merge scene_context into action if clip was silent but old backup had context
            if not action_text and old_clip.get("scene_context"):
                clip_entry["action"] = old_clip["scene_context"]
                clip_entry["tags"] = generate_keywords(clip_entry["action"])

        new_clips.append(clip_entry)

    # ── Step 6: Report & Save ──────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  REBUILD RESULTS")
    print(f"{'=' * 80}")
    print(f"  Total clips indexed:        {stats['total']:,}")
    print(f"  With dialogue:              {stats['with_dialogue']:,}")
    print(f"  Silent (action-only):       {stats['silent']:,}")
    print(f"  With manifest timecodes:    {stats['with_manifest_timecodes']:,}")
    print(f"  Merged from old backup:     {stats['merged_from_backup']:,}")
    print(f"  Brand new (no backup data): {stats['total'] - stats['merged_from_backup']:,}")

    # Count field coverage in the new index
    print(f"\n  --- Field Coverage ---")
    all_keys = set()
    for c in new_clips:
        all_keys.update(c.keys())
    for k in sorted(all_keys):
        count = sum(1 for c in new_clips if c.get(k) not in [None, "", [], {}, False])
        pct = 100 * count / len(new_clips)
        print(f"  {k:30s}: {count:6,} / {len(new_clips):,} ({pct:5.1f}%)")

    if dry_run:
        print(f"\n🏁 DRY RUN — no file written. Run without --dry-run to save.")
        return

    # Backup current index
    if OUTPUT_INDEX.exists():
        backup_path = OUTPUT_INDEX.with_name("clip_index_pre_rebuild_backup.json")
        print(f"\n💾 Backing up current index to {backup_path.name}...")
        import shutil
        shutil.copy2(OUTPUT_INDEX, backup_path)

    # Write new index
    print(f"📝 Writing fresh clip_index.json with {len(new_clips):,} clips...")
    index_data = {"clips": new_clips}
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    file_size_mb = OUTPUT_INDEX.stat().st_size / (1024 * 1024)
    print(f"✅ Done! Wrote {file_size_mb:.1f} MB to {OUTPUT_INDEX}")

    # Print next steps
    print(f"\n{'=' * 80}")
    print("  NEXT STEPS — Run enrichment pipeline to fill remaining fields")
    print(f"{'=' * 80}")
    print("""
  1. Semantic Embeddings (for clips missing embeddings):
     python scripts/clip_indexer_embed.py

  2. YOLO + ArcFace Character Tagging (visual_characters, prototype_detections):
     python scripts/run_visual_tagging_pipeline_arcmax.py

  3. Text-Based Character Enrichment (merge dialogue-based character detection):
     python scripts/enrich_clip_characters.py

  4. Full Enrichment Pipeline (scene_context, visual_tags, clip_visual_embedding):
     python scripts/run_full_enrichment.py

  NOTE: Each script skips already-processed clips, so you can safely
        Ctrl+C and resume at any time.
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild clip_index.json from scratch with ALL clips on disk."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview results without writing the index file"
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Don't merge rich fields from the old backup"
    )
    args = parser.parse_args()
    rebuild_index(dry_run=args.dry_run, merge_backup=not args.no_merge)
