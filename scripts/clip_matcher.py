"""
clip_matcher.py — Phase 4: Match narration segments to video clips or AI images.

Reads caption JSON (word-level timestamps from captioner.py) and clip_index.json,
scores each segment against available clips using keyword or LLM-assisted matching,
and outputs an assembly manifest for the video assembler.

Usage:
    python clip_matcher.py --captions captions/topic_001.json --output output/manifest.json
    python clip_matcher.py --captions captions/ --strategy llm --output output/manifest.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

# -- Local imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import (
    load_pipeline_config,
    load_show_config,
    get_active_show,
    get_project_path,
    load_json,
    save_json,
    setup_logging,
)

log = setup_logging("clip_matcher")


# ============================================================================
# Keyword extraction helpers
# ============================================================================

# Common English stop-words that add no matching value
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into about between through after before above below "
    "and or but not no nor so yet both either neither each every all "
    "some any few more most other such than too very it its he him his "
    "she her they them their this that these those what which who whom "
    "how when where why i me my we us our you your just also still even "
    "then now here there if only really actually like well much many "
    "got get goes going gone".split()
)


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric chars (except spaces)."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from a narration segment."""
    words = _normalize(text).split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def extract_character_mentions(text: str, show_config: dict) -> set:
    """Extract character names/aliases mentioned in the text."""
    text_lower = text.lower()
    mentions = set()
    for char in show_config.get("characters", []):
        for name in [char["name"]] + char.get("aliases", []):
            if name.lower() in text_lower:
                mentions.add(name.lower())
    return mentions


def extract_location_mentions(text: str, show_config: dict) -> set:
    """Extract location references from the text."""
    text_lower = text.lower()
    mentions = set()
    for loc in show_config.get("locations", []):
        if loc.lower() in text_lower:
            mentions.add(loc.lower())
    return mentions


def extract_theme_mentions(text: str, show_config: dict) -> set:
    """Extract theme references from the text."""
    text_lower = text.lower()
    mentions = set()
    for theme in show_config.get("themes", []):
        if theme.lower() in text_lower:
            mentions.add(theme.lower())
    return mentions


def _is_banned_clip(clip: dict) -> bool:
    """Check if a candidate clip is an outro, intro, end credits, or vanity card."""
    fname = clip.get("filename", "").lower()
    action = clip.get("action", "").lower()
    tags = {str(t).lower() for t in clip.get("tags", [])}
    summary = clip.get("episode_summary", "").lower()

    banned_terms = {
        "credits", "outro", "ending credits", "end credits", "theme song",
        "title card", "executive producer", "adult swim", "directed by",
        "written by", "logo", "vanity card", "production company", "created by",
        "black screen"
    }

    if any(term in fname for term in ("outro", "credit", "ending", "intro")):
        return True
    if any(term in action for term in banned_terms):
        return True
    if any(term in tags for term in banned_terms):
        return True
    if any(term in summary for term in banned_terms):
        return True

    return False


# ============================================================================
# Clip scoring — keyword strategy
# ============================================================================

def score_clip_keyword(segment_text: str, clip: dict, show_config: dict) -> float:
    """Score a clip against a narration segment using keyword matching.

    Scoring weights:
        - Character match:  3 points per character
        - Location match:   2 points
        - Action keyword:   2 points per matching word
        - Tag keyword:      1 point per matching tag
        - Theme match:      1.5 points per theme
    """
    score = 0.0

    # Extract features from segment
    seg_keywords = extract_keywords(segment_text)
    seg_characters = extract_character_mentions(segment_text, show_config)
    seg_locations = extract_location_mentions(segment_text, show_config)
    seg_themes = extract_theme_mentions(segment_text, show_config)

    # Clip metadata
    clip_characters = {c.lower() for c in clip.get("characters", [])}
    clip_location = clip.get("location", "").lower()
    clip_action = _normalize(clip.get("action", ""))
    clip_tags = {t.lower() for t in clip.get("tags", [])}

    # Character overlap (highest weight)
    char_overlap = seg_characters & clip_characters
    score += len(char_overlap) * 3.0

    # Location match
    if clip_location and clip_location in seg_locations:
        score += 2.0

    # Action keyword overlap
    action_words = extract_keywords(clip_action)
    action_overlap = seg_keywords & action_words
    score += len(action_overlap) * 2.0

    # Tag overlap
    tag_overlap = seg_keywords & clip_tags
    score += len(tag_overlap) * 1.0

    # Theme overlap with tags
    theme_overlap = seg_themes & clip_tags
    score += len(theme_overlap) * 1.5

    return score


def match_keyword(segment_text: str, clips: list, show_config: dict,
                  threshold: int = 1) -> tuple:
    """Find the best clip using keyword matching."""
    best_clip = None
    best_score = 0.0

    for clip in clips:
        if _is_banned_clip(clip):
            continue
        s = score_clip_keyword(segment_text, clip, show_config)
        if s > best_score:
            best_score = s
            best_clip = clip

    if best_score >= threshold:
        return best_clip, best_score
    return None, 0.0


# ============================================================================
# Clip scoring — Semantic strategy (Vector Embeddings)
# ============================================================================

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def match_semantic(segment_text: str, clips: list, show_config: dict, embedding_model, threshold: float = 3.0) -> tuple:
    """Find the best clip using Vector Embeddings (Semantic Search)."""
    best_clip = None
    best_score = -1.0

    seg_characters = extract_character_mentions(segment_text, show_config)
    seg_keywords = extract_keywords(segment_text)
    segment_embedding = embedding_model.encode(segment_text).tolist()

    for clip in clips:
        if _is_banned_clip(clip):
            continue
        clip_emb = clip.get("embedding")
        if not clip_emb:
            continue

        sim = cosine_similarity(segment_embedding, clip_emb)
        score = sim * 10.0

        # Character matching precision boost
        clip_characters = {c.lower() for c in clip.get("characters", [])}
        char_overlap = seg_characters & clip_characters
        if seg_characters:
            if char_overlap:
                score += len(char_overlap) * 7.0
            else:
                score -= 4.0

        # Location match bonus
        seg_locations = extract_location_mentions(segment_text, show_config)
        clip_location = clip.get("location", "").lower()
        if clip_location and clip_location in seg_locations:
            score += 2.0

        # Enriched episode summary RAG overlap bonus
        ep_summary = clip.get("episode_summary", "").lower()
        if ep_summary:
            ep_keywords = extract_keywords(ep_summary)
            overlap = seg_keywords & ep_keywords
            score += len(overlap) * 1.5

        if score > best_score:
            best_score = score
            best_clip = clip

    if best_score >= threshold:
        return best_clip, best_score
    return None, 0.0


# ============================================================================
# Clip scoring — LLM strategy (Ollama)
# ============================================================================

def match_llm(segment_text: str, clips: list, llm_config: dict) -> tuple:
    """Use Ollama to pick the best clip for a narration segment.

    Sends a prompt with the segment text and a numbered list of clip
    descriptions, asks the LLM to pick the best match by number.

    Returns (best_clip, confidence) or (None, 0) on failure.
    """
    if not clips:
        return None, 0.0

    # Build clip descriptions for the prompt (limit to top 20 for context)
    clip_descs = []
    for i, clip in enumerate(clips[:20]):
        chars = ", ".join(clip.get("characters", []))
        loc = clip.get("location", "unknown")
        action = clip.get("action", "")
        tags = ", ".join(clip.get("tags", []))
        clip_descs.append(
            f"{i + 1}. [{clip.get('filename', '?')}] "
            f"Characters: {chars} | Location: {loc} | "
            f"Action: {action} | Tags: {tags}"
        )

    clips_text = "\n".join(clip_descs)

    prompt = (
        f"You are a video editor matching narration to B-roll clips.\n\n"
        f"NARRATION SEGMENT:\n\"{segment_text}\"\n\n"
        f"AVAILABLE CLIPS:\n{clips_text}\n\n"
        f"Which clip number (1-{len(clip_descs)}) best matches this narration? "
        f"Reply with ONLY the number. If none match well, reply '0'."
    )

    base_url = llm_config.get("base_url", "http://localhost:11434")
    model = llm_config.get("model", "llama3.1:8b")
    timeout = llm_config.get("timeout_seconds", 300)

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for deterministic picks
                    "num_predict": 16,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()

        # Parse the number from the response
        match = re.search(r"\b(\d+)\b", answer)
        if match:
            idx = int(match.group(1))
            if 1 <= idx <= len(clip_descs):
                return clips[idx - 1], 10.0  # High confidence for LLM pick
            elif idx == 0:
                return None, 0.0  # LLM said no match

    except requests.RequestException as e:
        log.warning("LLM request failed, falling back to keyword: %s", e)
    except (ValueError, KeyError) as e:
        log.warning("Failed to parse LLM response: %s", e)

    return None, 0.0


# ============================================================================
# Fallback: generate AI image prompt
# ============================================================================

def generate_ai_image_prompt(segment_text: str, show_config: dict) -> str:
    """Create an image-generation prompt for segments with no matching clip.

    Produces a concise visual description suitable for Stable Diffusion or
    similar generators.
    """
    show_name = show_config.get("display_name", "the show")
    # Keep it short and visual
    clean = re.sub(r"[\"']", "", segment_text)
    if len(clean) > 120:
        clean = clean[:120] + "..."

    return (
        f"Cinematic still from {show_name}, depicting: {clean}. "
        f"Dramatic lighting, animation style, 9:16 vertical composition, "
        f"high detail, vibrant colors."
    )


# ============================================================================
# Assembly manifest builder
# ============================================================================

def build_manifest(caption_data: dict, clips: list, show_config: dict,
                   strategy: str, matching_config: dict,
                   llm_config: dict) -> dict:
    """Build an assembly manifest from captions and clip index.

    Parameters
    ----------
    caption_data : dict
        Output from captioner.py — must have "audio_file" and "segments" keys.
        Each segment has "id", "text", "start", "end", "words".
    clips : list
        List of clip metadata dicts from clip_index.json.
    show_config : dict
        The active show's config section.
    strategy : str
        "keyword", "semantic", or "llm".
    matching_config : dict
        clip_matching section from pipeline_config.yaml.
    llm_config : dict
        llm section from pipeline_config.yaml.

    Returns
    -------
    dict
        Assembly manifest ready for assembler.py.
    """
    threshold = matching_config.get("keyword_match_threshold", 1)
    fallback = matching_config.get("fallback", "ai_image")
    max_clip_dur = matching_config.get("max_clip_duration_seconds", 5)

    manifest_segments = []
    stats = {"matched": 0, "fallback": 0, "total": 0}

    for seg in caption_data.get("segments", []):
        stats["total"] += 1
        seg_text = seg.get("text", "").strip()
        seg_id = seg.get("id", stats["total"] - 1)

        if not seg_text:
            log.warning("Segment %d has empty text, skipping", seg_id)
            continue

        best_clip = None
        score = 0.0

        # --- Matching ---
        if strategy == "semantic" and clips:
            # We lazy load the model here only if strategy is semantic
            if not hasattr(build_manifest, "embedding_model"):
                try:
                    from sentence_transformers import SentenceTransformer
                    log.info("Loading SentenceTransformer model for semantic matching...")
                    build_manifest.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except ImportError:
                    log.error("sentence-transformers not installed! Falling back to keyword matching.")
                    strategy = "keyword"
            
            if strategy == "semantic":
                best_clip, score = match_semantic(
                    seg_text, clips, show_config, build_manifest.embedding_model, threshold=3.0
                )

        if strategy == "llm" and clips:
            best_clip, score = match_llm(seg_text, clips, llm_config)
            # Fall back to keyword if LLM fails
            if best_clip is None:
                best_clip, score = match_keyword(
                    seg_text, clips, show_config, threshold
                )
        elif strategy == "keyword" and clips:
            best_clip, score = match_keyword(
                seg_text, clips, show_config, threshold
            )

        # --- Build manifest entry ---
        entry = {
            "id": seg_id,
            "text": seg_text,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "words": seg.get("words", []),
        }

        if best_clip is not None:
            entry["visual_type"] = "clip"
            entry["visual_source"] = best_clip.get("filename", "")
            # Clip start: use 0.0 (beginning of clip) — assembler will trim
            entry["clip_start"] = 0.0
            entry["match_score"] = round(score, 2)
            stats["matched"] += 1
            log.info(
                "Segment %d → clip '%s' (score=%.1f)",
                seg_id, best_clip.get("filename", "?"), score,
            )
        else:
            # Fallback
            if fallback == "ai_image":
                entry["visual_type"] = "ai_image"
                entry["visual_source"] = generate_ai_image_prompt(
                    seg_text, show_config
                )
            elif fallback == "generic_broll":
                entry["visual_type"] = "clip"
                entry["visual_source"] = "__generic_broll__"
            else:  # "black"
                entry["visual_type"] = "black"
                entry["visual_source"] = ""

            entry["clip_start"] = 0.0
            entry["match_score"] = 0.0
            stats["fallback"] += 1
            log.info("Segment %d → fallback (%s)", seg_id, fallback)

        manifest_segments.append(entry)

    manifest = {
        "audio_file": caption_data.get("audio_file", ""),
        "segments": manifest_segments,
        "stats": stats,
    }

    log.info(
        "Matching complete: %d/%d matched, %d fallback",
        stats["matched"], stats["total"], stats["fallback"],
    )
    return manifest


# ============================================================================
# Caption file loading
# ============================================================================

def load_caption_files(captions_path: Path) -> list:
    """Load one or more caption JSON files.

    If *captions_path* is a file, return [data].
    If it's a directory, load all .json files and return a list.
    """
    results = []

    if captions_path.is_file():
        data = load_json(captions_path)
        if data:
            results.append(data)
    elif captions_path.is_dir():
        for f in sorted(captions_path.glob("*.json")):
            data = load_json(f)
            if data:
                results.append(data)
    else:
        log.error("Captions path does not exist: %s", captions_path)

    if not results:
        log.warning("No caption data loaded from %s", captions_path)

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 4: Match narration segments to video clips or AI images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--captions",
        required=True,
        help="Caption JSON file or directory of caption JSONs.",
    )
    parser.add_argument(
        "--clip-index",
        default=None,
        help="Path to clip_index.json (default: from pipeline config).",
    )
    parser.add_argument(
        "--strategy",
        choices=["keyword", "llm", "semantic"],
        default=None,
        help="Matching strategy (default: from pipeline config).",
    )
    parser.add_argument(
        "--show",
        default=None,
        help="Show slug (default: first active show).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest file path (default: output/manifest.json).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load configs
    pipeline_cfg = load_pipeline_config()
    matching_cfg = pipeline_cfg.get("clip_matching", {})
    llm_cfg = pipeline_cfg.get("llm", {})

    # Resolve show
    show_slug, show_config = get_active_show(args.show)
    log.info("Using show: %s (%s)", show_config.get("display_name", "?"), show_slug)

    # Strategy
    strategy = args.strategy or matching_cfg.get("strategy", "semantic")
    log.info("Matching strategy: %s", strategy)

    # Load clip index
    if args.clip_index:
        clip_index_path = Path(args.clip_index)
    else:
        clip_index_path = get_project_path("clip_index", pipeline_cfg)

    clip_data = load_json(clip_index_path)
    if isinstance(clip_data, dict):
        clips = clip_data.get("clips", [])
    elif isinstance(clip_data, list):
        clips = clip_data
    else:
        clips = []
    log.info("Loaded %d clips from %s", len(clips), clip_index_path)

    # Load captions
    captions_path = Path(args.captions)
    caption_files = load_caption_files(captions_path)
    if not caption_files:
        log.error("No caption files found — exiting")
        sys.exit(1)

    # Process each caption file
    for i, caption_data in enumerate(caption_files):
        manifest = build_manifest(
            caption_data, clips, show_config,
            strategy, matching_cfg, llm_cfg,
        )

        # Output path
        if args.output:
            out_path = Path(args.output)
            if len(caption_files) > 1:
                # Append index for multiple files
                out_path = out_path.parent / f"{out_path.stem}_{i}{out_path.suffix}"
        else:
            out_dir = get_project_path("output_dir", pipeline_cfg)
            out_path = out_dir / f"manifest_{i}.json"

        save_json(out_path, manifest)
        log.info("Assembly manifest saved → %s", out_path)

    log.info("Done — processed %d caption file(s)", len(caption_files))


if __name__ == "__main__":
    main()
