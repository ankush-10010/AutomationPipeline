"""
episode_indexer.py — Generates canonical episode summaries from subtitle files.

Reads .srt files from the configured subtitles directory, extracts dialogue,
sends it to Ollama for structured summarization, and builds an episode index.
Can also enrich the clip index with episode-level metadata.
"""

import argparse
import json
import re
import sys
import time
import requests
from pathlib import Path

from config_loader import (
    setup_logging,
    load_pipeline_config,
    get_active_show,
    get_project_path,
    load_json,
    save_json,
    load_text,
    PROJECT_ROOT,
)

log = setup_logging("episode_indexer")


# ---------------------------------------------------------------------------
# Episode Indexer
# ---------------------------------------------------------------------------
class EpisodeIndexer:
    """Processes subtitle files into structured episode summaries via LLM."""

    # Regex patterns for episode ID extraction
    _PATTERN_SXXEXX = re.compile(r"S(\d+)E(\d+)", re.IGNORECASE)
    _PATTERN_NxNN = re.compile(r"(\d+)x(\d+)", re.IGNORECASE)

    # Regex for new-format title extraction:
    #   "Rick and Morty - 8x01 - Summer of All Fears.WEB.MAX.en.srt"
    _PATTERN_NEW_TITLE = re.compile(
        r"^.+?\s*-\s*\d+x\d+\s*-\s*(.+?)\.(?:WEB|HDTV|BluRay|DVDRip)",
        re.IGNORECASE,
    )

    # Lines that are SRT timecodes or numeric counters
    _SRT_TIMECODE_RE = re.compile(r"\d+:\d+")
    _SRT_COUNTER_RE = re.compile(r"^\d+\s*$")
    # HTML-style tags sometimes present in SRTs
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    def __init__(self, pipeline_config: dict, show_config: dict):
        self.pipeline_config = pipeline_config
        self.show_config = show_config

        # LLM settings
        llm = pipeline_config.get("llm", {})
        self.base_url = llm.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = llm.get("model", "llama3.1:8b")
        self.temperature = llm.get("temperature", 0.8)
        self.max_tokens = llm.get("max_tokens", 4096)
        self.timeout = llm.get("timeout_seconds", 300)

        # Paths
        self.subtitles_dir = get_project_path("subtitles_dir", pipeline_config)
        self.prompts_dir = get_project_path("prompts_dir", pipeline_config)

        # Episode index config
        ep_cfg = pipeline_config.get("episode_index", {})
        self.index_path = (PROJECT_ROOT / ep_cfg.get("path", "episode_index.json")).resolve()
        self.auto_enrich = ep_cfg.get("auto_enrich_clips", True)

        # Show metadata
        self.show_name = show_config.get("display_name", "Unknown Show")

        log.info("EpisodeIndexer initialized — subtitles: %s", self.subtitles_dir)
        log.info("Episode index path: %s", self.index_path)

    # ----- Episode ID parsing -----

    def _parse_episode_id(self, filename: str) -> tuple | None:
        """
        Extract (season, episode) from a subtitle filename.

        Supports two formats:
          - Old: Rick.and.morty.S01E01.xxx.srt  →  (1, 1)
          - New: Rick and Morty - 8x01 - Title.srt  →  (8, 1)

        Returns None if no pattern matches.
        """
        # Try S01E01 pattern first
        m = self._PATTERN_SXXEXX.search(filename)
        if m:
            return int(m.group(1)), int(m.group(2))

        # Try 8x01 pattern
        m = self._PATTERN_NxNN.search(filename)
        if m:
            return int(m.group(1)), int(m.group(2))

        return None

    def _extract_title_from_filename(self, filename: str) -> str:
        """
        Extract episode title from new-format filenames.

        Example:
          "Rick and Morty - 8x01 - Summer of All Fears.WEB.MAX.en.srt"
          → "Summer of All Fears"

        Returns empty string for old-format filenames or if extraction fails.
        """
        m = self._PATTERN_NEW_TITLE.match(filename)
        if m:
            return m.group(1).strip()
        return ""

    # ----- SRT reading -----

    def _read_srt_as_text(self, srt_path: Path) -> str:
        """
        Read an .srt file and return clean dialogue text.

        Strips:
          - Numeric line counters (e.g., "1", "2", ...)
          - SRT timecodes (lines containing "digits:digits")
          - HTML tags (e.g., <font ...>)
          - Blank lines

        Returns all dialogue lines joined by spaces.
        """
        try:
            content = srt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error("Failed to read SRT file %s: %s", srt_path, e)
            return ""

        lines = content.splitlines()
        dialogue_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Skip numeric counters (just a number on its own line)
            if self._SRT_COUNTER_RE.match(stripped):
                continue

            # Skip timecode lines (contain digits:digits pattern)
            if self._SRT_TIMECODE_RE.search(stripped):
                continue

            # Remove HTML tags
            cleaned = self._HTML_TAG_RE.sub("", stripped)
            cleaned = cleaned.strip()

            if cleaned:
                dialogue_lines.append(cleaned)

        return " ".join(dialogue_lines)

    # ----- Prompt building -----

    def _build_extraction_prompt(
        self,
        dialogue_text: str,
        season: int,
        episode: int,
        show_name: str,
        filename_title: str,
    ) -> str:
        """
        Build the LLM prompt for episode extraction using the template file.
        """
        template_path = self.prompts_dir / "episode_extract_prompt.txt"
        template = load_text(template_path)

        if not template:
            log.error("Episode extraction prompt template missing: %s", template_path)
            sys.exit(1)

        # Truncate dialogue if extremely long to stay within context window
        max_dialogue_chars = 15000
        if len(dialogue_text) > max_dialogue_chars:
            log.warning(
                "Dialogue text truncated from %d to %d chars for S%02dE%02d",
                len(dialogue_text),
                max_dialogue_chars,
                season,
                episode,
            )
            dialogue_text = dialogue_text[:max_dialogue_chars] + "\n[... truncated ...]"

        title_line = f'Filename title hint: "{filename_title}"' if filename_title else ""

        prompt = template.format(
            show_name=show_name,
            season=season,
            episode=episode,
            filename_title=title_line,
            dialogue_text=dialogue_text,
        )

        return prompt

    # ----- Ollama API -----

    def _call_ollama(self, prompt: str) -> str:
        """
        Send a prompt to the Ollama /api/generate endpoint and return the
        full response text.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        log.info("Calling Ollama → %s (model: %s)", url, self.model)
        log.debug("Prompt length: %d chars", len(prompt))

        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.ConnectionError:
            log.error(
                "Cannot connect to Ollama at %s — is it running? "
                "Start with: ollama serve",
                self.base_url,
            )
            return ""
        except requests.Timeout:
            log.error("Ollama request timed out after %ds", self.timeout)
            return ""
        except requests.HTTPError as e:
            log.error("Ollama returned HTTP error: %s", e)
            return ""

        elapsed = time.time() - start
        result = resp.json()
        response_text = result.get("response", "")
        log.info("Ollama responded in %.1fs (%d chars)", elapsed, len(response_text))

        return response_text

    # ----- Response parsing -----

    def _parse_llm_response(self, raw: str) -> dict | None:
        """
        Extract a JSON object from the LLM response.

        Handles markdown code fences (```json ... ```) and stray text
        before/after the JSON block.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if "```" in text:
            lines = text.split("\n")
            cleaned = []
            inside_fence = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside_fence = not inside_fence
                    continue
                if inside_fence or not text.startswith("```"):
                    cleaned.append(line)
            text = "\n".join(cleaned).strip()

        # Find the JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            log.error("Could not find JSON object in LLM response")
            log.debug("Raw response:\n%s", raw[:500])
            return None

        json_str = text[start : end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.error("Failed to parse episode JSON: %s", e)
            log.debug("Attempted to parse:\n%s", json_str[:500])
            return None

        if not isinstance(data, dict):
            log.error("Expected JSON object, got %s", type(data).__name__)
            return None

        # Validate required fields
        required = ["title", "one_line", "summary", "key_events", "characters_featured", "themes"]
        missing = [f for f in required if f not in data]
        if missing:
            log.warning("LLM response missing fields: %s — keeping partial result", missing)

        return data

    # ----- Single episode processing -----

    def index_single_episode(self, srt_path: Path) -> dict | None:
        """
        Full indexing flow for a single .srt file.

        Returns a dict with episode metadata, or None on failure.
        """
        filename = srt_path.name
        log.info("Processing: %s", filename)

        # Parse episode ID
        ep_id = self._parse_episode_id(filename)
        if ep_id is None:
            log.warning("Could not extract episode ID from: %s — skipping", filename)
            return None

        season, episode = ep_id
        episode_key = f"s{season}e{episode}"
        log.info("Identified as %s (Season %d, Episode %d)", episode_key, season, episode)

        # Extract title from filename (new format only)
        filename_title = self._extract_title_from_filename(filename)
        if filename_title:
            log.info("Filename title: %s", filename_title)

        # Read and clean dialogue
        dialogue = self._read_srt_as_text(srt_path)
        if not dialogue:
            log.error("No dialogue extracted from: %s", filename)
            return None

        log.info("Extracted %d chars of dialogue", len(dialogue))

        # Build prompt and call LLM
        prompt = self._build_extraction_prompt(
            dialogue, season, episode, self.show_name, filename_title
        )
        raw_response = self._call_ollama(prompt)
        if not raw_response:
            log.error("Empty response from Ollama for %s", episode_key)
            return None

        # Parse structured response
        parsed = self._parse_llm_response(raw_response)
        if parsed is None:
            log.error("Failed to parse LLM response for %s", episode_key)
            return None

        # Add metadata
        parsed["season"] = season
        parsed["episode"] = episode
        parsed["episode_key"] = episode_key
        parsed["source_file"] = filename

        log.info("✓ Indexed %s — \"%s\"", episode_key, parsed.get("title", "?"))
        return parsed

    # ----- Batch processing -----

    def index_all_episodes(self, force: bool = False) -> dict:
        """
        Process all .srt files in the subtitles directory.

        Returns a dict keyed by episode_key (e.g., 's1e1').
        Skips episodes already in the index unless force=True.
        """
        # Load existing index
        existing = load_json(self.index_path)
        if not isinstance(existing, dict):
            existing = {}

        # Find all .srt files
        if not self.subtitles_dir.exists():
            log.error("Subtitles directory not found: %s", self.subtitles_dir)
            return existing

        srt_files = sorted(self.subtitles_dir.glob("*.srt"))
        log.info("Found %d .srt files in %s", len(srt_files), self.subtitles_dir)

        processed = 0
        skipped = 0

        for srt_path in srt_files:
            # Check if already indexed
            ep_id = self._parse_episode_id(srt_path.name)
            if ep_id is not None:
                key = f"s{ep_id[0]}e{ep_id[1]}"
                if key in existing and not force:
                    log.debug("Skipping %s — already indexed (use --force to reprocess)", key)
                    skipped += 1
                    continue

            result = self.index_single_episode(srt_path)
            if result:
                existing[result["episode_key"]] = result
                processed += 1

                # Save after each episode (in case of interruption)
                save_json(self.index_path, existing)

                # Rate-limit LLM calls
                time.sleep(1)

        log.info(
            "Indexing complete: %d processed, %d skipped, %d total in index",
            processed,
            skipped,
            len(existing),
        )
        return existing

    # ----- Clip enrichment -----

    def enrich_clip_index(self) -> None:
        """
        Enrich the clip_index.json with episode-level metadata.

        For each clip whose filename contains a season/episode pattern (e.g.,
        s1e1_scene_029.mp4), adds:
          - episode_id (e.g., "s1e1")
          - episode_title
          - episode_summary (one_line)
        """
        # Load episode index
        episode_index = load_json(self.index_path)
        if not episode_index or not isinstance(episode_index, dict):
            log.warning("Episode index is empty or missing — run indexing first")
            return

        # Load clip index
        clip_index_path = get_project_path("clip_index", self.pipeline_config)
        clip_data = load_json(clip_index_path)
        if not clip_data:
            log.warning("Clip index is empty or missing: %s", clip_index_path)
            return

        # clip_index.json has {"clips": [...], ...} structure
        clips = clip_data.get("clips", []) if isinstance(clip_data, dict) else clip_data

        enriched_count = 0
        for clip in clips:
            filename = clip.get("filename", "")
            ep_id = self._parse_episode_id(filename)
            if ep_id is None:
                continue

            key = f"s{ep_id[0]}e{ep_id[1]}"
            ep_data = episode_index.get(key)
            if ep_data is None:
                continue

            clip["episode_id"] = key
            clip["episode_title"] = ep_data.get("title", "")
            clip["episode_summary"] = ep_data.get("one_line", "")
            enriched_count += 1

        log.info("Enriched %d clips with episode metadata", enriched_count)

        # Save back
        if isinstance(clip_data, dict):
            clip_data["clips"] = clips
        save_json(clip_index_path, clip_data)
        log.info("Saved enriched clip index → %s", clip_index_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate canonical episode summaries from subtitle files",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="Show slug from show_config.yaml (default: first active show)",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        metavar="FILE",
        help="Process a single .srt file (path relative to subtitles dir or absolute)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Run clip index enrichment only (requires existing episode_index.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess episodes even if already in the index",
    )
    args = parser.parse_args()

    # Load configs
    pipeline_config = load_pipeline_config()
    slug, show = get_active_show(args.show)
    log.info("=== Episode Indexer for '%s' ===", show.get("display_name", slug))

    indexer = EpisodeIndexer(pipeline_config, show)

    # --- Enrich-only mode ---
    if args.enrich:
        log.info("Running clip enrichment only")
        indexer.enrich_clip_index()
        log.info("✓ Clip enrichment complete")
        return

    # --- Single file mode ---
    if args.single:
        srt_path = Path(args.single)
        if not srt_path.is_absolute():
            srt_path = indexer.subtitles_dir / srt_path
        if not srt_path.exists():
            log.error("SRT file not found: %s", srt_path)
            sys.exit(1)

        result = indexer.index_single_episode(srt_path)
        if result:
            # Merge into existing index
            existing = load_json(indexer.index_path)
            if not isinstance(existing, dict):
                existing = {}
            existing[result["episode_key"]] = result
            save_json(indexer.index_path, existing)
            log.info("✓ Saved to %s", indexer.index_path)
        else:
            log.error("Failed to index: %s", srt_path)
            sys.exit(1)
        return

    # --- Default: process all episodes ---
    index = indexer.index_all_episodes(force=args.force)
    log.info("✓ Episode index complete — %d episodes in %s", len(index), indexer.index_path)

    # Auto-enrich clips if configured
    if indexer.auto_enrich:
        log.info("Auto-enriching clip index with episode data...")
        indexer.enrich_clip_index()


if __name__ == "__main__":
    main()
