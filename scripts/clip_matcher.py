"""
clip_matcher.py — Phase 4: Match narration segments to video clips or AI images.

Reads caption JSON (word-level timestamps from captioner.py) and clip_index.json,
scores each segment against available clips using keyword, semantic (vector/CLIP),
or LLM-assisted matching, and outputs an assembly manifest for the video assembler.

Upgraded in v2 to fully leverage all 21 clip metadata fields, including:
  - ArcMax visual character tags (visual_characters) & ArcFace similarity confidence (prototype_detections)
  - Three-channel semantic scoring (Character Match, Semantic Vector/CLIP, Contextual Metadata)
  - Emotion/tone matching (emotion_tone & mood)
  - Multi-frame visual narration search (raw_vision & visual_description)
  - Anti-repetition adjacency swapping & cooldown penalties

Usage:
    python clip_matcher.py --captions captions/topic_001.json --output output/manifest.json
    python clip_matcher.py --captions captions/ --strategy semantic --output output/manifest.json
"""

import argparse
import json
import re
import sys
from collections import deque
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
# Keyword & Feature Extraction Helpers
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

# Emotion and tone keyword mappings for matching segment text against clip emotion_tone
_TONE_KEYWORDS = {
    "action": {"fight", "battle", "attack", "run", "chase", "explosion", "blast", "strike", "punch", "smash", "danger", "fast", "transform", "slam", "laser", "shoot", "dodge"},
    "dramatic": {"reveal", "secret", "truth", "death", "serious", "danger", "threat", "vilgax", "destiny", "fail", "lose", "betray", "shock", "warning", "power"},
    "comedic": {"joke", "laugh", "funny", "silly", "grandpa", "food", "smoothy", "weird", "gross", "oops", "clumsy", "smirk", "smile", "teaser"},
    "suspense": {"mystery", "dark", "shadow", "creepy", "ghost", "quiet", "hide", "sneak", "waiting", "unknown", "scary", "eerie", "lurking"},
}


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


def _extract_transformation_mentions(text: str, show_config: dict) -> set:
    """Extract alien transformation names from text using show_config aliases."""
    text_lower = text.lower()
    mentions = set()
    for char in show_config.get("characters", []):
        if char.get("name", "") != "Ben Tennyson":
            continue
        for alias in char.get("aliases", []):
            clean = alias.strip()
            if clean.lower() not in ("ben",) and len(clean) > 2:
                if re.search(rf"\b{re.escape(clean.lower())}\b", text_lower):
                    mentions.add(clean.lower())
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


def parse_clip_identity(filename: str) -> tuple:
    """Parse episode and scene number from a clip filename.

    Expected format: s{season}e{episode}_scene_{number}.mp4
    Returns (episode_key, scene_number) or (None, None) if unparseable.
    """
    match = re.match(r"(s\d+e\d+)_scene_(\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower(), int(match.group(2))
    return None, None


def find_adjacent_clips(clip: dict, clips: list, used_filenames: set,
                        max_distance: int = 5) -> list:
    """Find clips from the same episode within max_distance scenes.

    Returns a list of (clip, distance) tuples sorted by distance,
    excluding clips already in the used set.
    """
    ep_key, scene_num = parse_clip_identity(clip.get("filename", ""))
    if ep_key is None:
        return []

    candidates = []
    for c in clips:
        if _is_banned_clip(c):
            continue
        if c.get("filename", "") in used_filenames:
            continue
        c_ep, c_scene = parse_clip_identity(c.get("filename", ""))
        if c_ep == ep_key and c_scene is not None and c_scene != scene_num:
            dist = abs(c_scene - scene_num)
            if dist <= max_distance:
                candidates.append((c, dist))

    candidates.sort(key=lambda x: x[1])
    return candidates


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
        "black screen", "my-subs", "wwwmysubsco"
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


def classify_intent(text: str, show_config: dict) -> dict:
    """Classify the intent of the narration segment to adjust scoring weights."""
    text_lower = text.lower()
    
    action_words = {"explosion", "attack", "destroy", "fight", "blast", "shoot", "dodge", "punch", "kick", "battle", "transform", "hero time", "run", "chase"}
    character_words = extract_character_mentions(text, show_config)
    
    action_count = sum(1 for w in action_words if w in text_lower)
    char_count = len(character_words)
    
    weights = {
        "rrf_scale": 300.0,
        "char_bonus": 10.0,
        "alien_bonus": 15.0,
        "plot_bonus": 1.5,
    }
    
    if action_count > 0 and char_count == 0:
        weights["rrf_scale"] = 450.0
        weights["plot_bonus"] = 3.0
    elif char_count > 0 and action_count == 0:
        weights["char_bonus"] = 20.0
        weights["alien_bonus"] = 25.0
        weights["rrf_scale"] = 200.0
    elif char_count > 0 and action_count > 0:
        weights["char_bonus"] = 15.0
        weights["alien_bonus"] = 20.0
        weights["rrf_scale"] = 350.0
        
    return weights


def _get_proto_sim(proto_dets: dict, char_name: str) -> float:
    """Safely extract ArcFace prototype similarity score for a character."""
    if not proto_dets or not isinstance(proto_dets, dict):
        return 0.0
    for k, v in proto_dets.items():
        if k.lower() == char_name.lower():
            if isinstance(v, dict):
                return float(v.get("max_similarity", 0.0))
            elif isinstance(v, (int, float)):
                return float(v)
    return 0.0


def _get_clip_characters(clip: dict) -> set:
    """Get lowercase character set from visual_characters (ArcMax) or fallback to characters."""
    if "visual_characters" in clip:
        return {c.lower() for c in clip.get("visual_characters", [])}
    return {c.lower() for c in clip.get("characters", [])}


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two 1D vectors."""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


_ALIEN_REFERENCE_PROMPTS = {
    "heatblast": "a photo of Heatblast, a flaming fire alien superhero made of magma and rocks",
    "four arms": "a photo of Four Arms, a giant red four-armed muscular alien superhero",
    "xlr8": "a photo of XLR8, a blue and black armored dinosaur-like speed alien with wheels on his feet",
    "diamondhead": "a photo of Diamondhead, a crystalline green shard alien made of living crystals",
    "stinkfly": "a photo of Stinkfly, a giant winged insectoid alien superhero with four eyes",
    "upgrade": "a photo of Upgrade, a biomechanical black and green techno-organic alien biomech",
    "cannonbolt": "a photo of Cannonbolt, a bulky white and yellow armored rolling sphere alien",
    "wildmutt": "a photo of Wildmutt, an orange beast-like dog alien with no eyes and sharp teeth"
}

def _get_alien_visual_similarity(clip: dict, alien_name: str) -> float:
    """Calculate CLIP visual similarity against static alien reference phrases."""
    clip_vis = clip.get("clip_visual_embedding")
    if not clip_vis or alien_name.lower() not in _ALIEN_REFERENCE_PROMPTS:
        return 0.0
    
    encoder = _get_clip_text_encoder()
    if not encoder:
        return 0.0
        
    ref_text = _ALIEN_REFERENCE_PROMPTS[alien_name.lower()]
    ref_emb = encoder.encode(ref_text).tolist()
    return cosine_similarity(ref_emb, clip_vis)


def _get_clip_text_encoder():
    """Lazily load CLIP text encoder for visual embedding matching."""
    if not hasattr(_get_clip_text_encoder, "_model"):
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading CLIP text encoder for visual matching...")
            _get_clip_text_encoder._model = SentenceTransformer("clip-ViT-B-32")
        except ImportError:
            log.warning("sentence-transformers not available for CLIP matching")
            _get_clip_text_encoder._model = None
    return _get_clip_text_encoder._model


# ============================================================================
# Clip Scoring — Keyword Strategy
# ============================================================================

def score_clip_keyword(segment_text: str, clip: dict, show_config: dict,
                       seg_characters: set = None, seg_locations: set = None) -> float:
    """Score a clip against a narration segment using keyword & visual tag matching."""
    score = 0.0

    seg_keywords = extract_keywords(segment_text)
    if seg_characters is None:
        seg_characters = extract_character_mentions(segment_text, show_config)
    if seg_locations is None:
        seg_locations = extract_location_mentions(segment_text, show_config)
    seg_themes = extract_theme_mentions(segment_text, show_config)

    clip_characters = _get_clip_characters(clip)
    clip_location = clip.get("location", "").lower()
    clip_action = _normalize(clip.get("action", ""))
    clip_tags = {str(t).lower() for t in clip.get("tags", [])}
    visual_tags = {str(t).lower() for t in clip.get("visual_tags", [])}

    # Character overlap (highest weight)
    char_overlap = seg_characters & clip_characters
    score += len(char_overlap) * 5.0

    # Location match
    if clip_location and clip_location in seg_locations:
        score += 2.0

    # Action & visual description overlap
    action_words = extract_keywords(clip_action)
    vis_desc = clip.get("visual_description", "").lower()
    raw_vis = clip.get("raw_vision", "").lower()
    vis_words = extract_keywords(f"{vis_desc} {raw_vis}")
    
    score += len(seg_keywords & action_words) * 2.0
    score += len(seg_keywords & vis_words) * 1.5
    score += len(seg_keywords & (clip_tags | visual_tags)) * 1.0
    score += len(seg_themes & clip_tags) * 1.5

    return score


def match_keyword(segment_text: str, clips: list, show_config: dict,
                  threshold: int = 1,
                  cooldown_set: set = None,
                  seg_characters: set = None,
                  seg_locations: set = None,
                  ban_cooldown: bool = False) -> tuple:
    """Find the best clip using keyword matching across all visual/text metadata."""
    if cooldown_set is None:
        cooldown_set = set()

    scored_clips = []
    seg_keywords = extract_keywords(segment_text)
    if seg_characters is None:
        seg_characters = extract_character_mentions(segment_text, show_config)
    if seg_locations is None:
        seg_locations = extract_location_mentions(segment_text, show_config)

    for clip in clips:
        if _is_banned_clip(clip):
            continue
            
        reason_parts = []
        clip_characters = _get_clip_characters(clip)
        clip_location = clip.get("location", "").lower()
        clip_action = _normalize(clip.get("action", ""))
        clip_tags = {str(t).lower() for t in clip.get("tags", [])}
        visual_tags = {str(t).lower() for t in clip.get("visual_tags", [])}
        
        s = 0.0
        
        char_overlap = seg_characters & clip_characters
        if char_overlap:
            s += len(char_overlap) * 5.0
            reason_parts.append(f"Chars: {', '.join(char_overlap)}")
            
        if clip_location and clip_location in seg_locations:
            s += 2.0
            reason_parts.append(f"Loc: {clip_location}")
            
        action_words = extract_keywords(clip_action)
        action_overlap = seg_keywords & action_words
        if action_overlap:
            s += len(action_overlap) * 2.0
            reason_parts.append(f"Action: {', '.join(action_overlap)}")
            
        vis_desc = clip.get("visual_description", "").lower()
        raw_vis = clip.get("raw_vision", "").lower()
        vis_overlap = seg_keywords & extract_keywords(f"{vis_desc} {raw_vis}")
        if vis_overlap:
            s += len(vis_overlap) * 1.5
            reason_parts.append(f"VisDesc: {len(vis_overlap)}")

        tag_overlap = seg_keywords & (clip_tags | visual_tags)
        if tag_overlap:
            s += len(tag_overlap) * 1.0
            reason_parts.append(f"Tags: {', '.join(tag_overlap)}")
            
        if not reason_parts:
            reason_parts.append("Weak Keyword Match")

        raw_s = s
        reason = " | ".join(reason_parts)

        # Apply cooldown penalty
        if clip.get("filename", "") in cooldown_set:
            if ban_cooldown:
                continue
            reason += " [PREVIOUSLY USED]"

        scored_clips.append((s, raw_s, clip, reason))

    valid_clips = [x for x in scored_clips if x[1] >= threshold]
    if not valid_clips:
        return [], 0.0, ""

    valid_clips.sort(key=lambda x: x[1], reverse=True)
    unused_clips = [x for x in valid_clips if x[2].get("filename", "") not in cooldown_set]

    if unused_clips:
        best_pool = unused_clips
    else:
        best_pool = valid_clips

    clips_only = [c for s, r, c, reason in best_pool]
    return clips_only[:10], best_pool[0][1], best_pool[0][3]


# ============================================================================
# Clip Scoring — Semantic Strategy (Three-Channel Architecture)
# ============================================================================

def match_semantic(segment_text: str, clips: list, bm25_scores: list, show_config: dict,
                   embedding_model, threshold: float = 3.0,
                   cooldown_set: set = None,
                   seg_characters: set = None,
                   seg_locations: set = None,
                   ban_cooldown: bool = False) -> tuple:
    if cooldown_set is None:
        cooldown_set = set()

    if seg_characters is None:
        seg_characters = extract_character_mentions(segment_text, show_config)
    if seg_locations is None:
        seg_locations = extract_location_mentions(segment_text, show_config)
    seg_keywords = extract_keywords(segment_text)
    seg_transforms = _extract_transformation_mentions(segment_text, show_config)
    segment_embedding = embedding_model.encode(segment_text).tolist()

    missing_embeddings = sum(1 for c in clips if not c.get("embedding"))
    if missing_embeddings > 0:
        log.warning(
            f"[Coverage Warning] {missing_embeddings}/{len(clips)} clips lack semantic embeddings. "
            "These clips will rely entirely on character, keyword, and subtitle matching."
        )

    dense_scores = []
    for clip in clips:
        if _is_banned_clip(clip):
            dense_scores.append(0.0)
            continue
        clip_emb = clip.get("embedding")
        if not clip_emb:
            dense_scores.append(0.0)
            continue
        dense_scores.append(cosine_similarity(segment_embedding, clip_emb))
        
    try:
        from scripts.bm25 import reciprocal_rank_fusion
    except ImportError:
        try:
            from bm25 import reciprocal_rank_fusion
        except ImportError:
            reciprocal_rank_fusion = lambda d, s, k: d # dummy fallback
        
    rrf_scores = reciprocal_rank_fusion(dense_scores, bm25_scores, k=60)
    intent_weights = classify_intent(segment_text, show_config)

    scored_clips = []
    for i, clip in enumerate(clips):
        if _is_banned_clip(clip):
            continue
            
        base_score = rrf_scores[i] * intent_weights["rrf_scale"]
        score = base_score
        reason_parts = [f"RRF: {base_score:.1f}"]

        clip_visual_chars = {c.lower() for c in clip.get("visual_characters", [])}
        clip_text_chars = {c.lower() for c in clip.get("characters", [])}
        clip_characters = clip_visual_chars | clip_text_chars

        if seg_characters:
            char_overlap = seg_characters & clip_characters
            has_ground_truth = bool(clip.get("prototype_detections") or clip.get("visual_characters"))
            if char_overlap:
                for char in char_overlap:
                    proto_dets = clip.get("prototype_detections", {})
                    sim = _get_proto_sim(proto_dets, char)
                    conf_bonus = 3.0 * sim if sim > 0 else 2.0
                    score += intent_weights["char_bonus"] + conf_bonus
                reason_parts.append(f"Chars: {', '.join(char_overlap)}")
            elif has_ground_truth:
                score -= 5.0
                reason_parts.append("Visual Absence Verified")
            else:
                reason_parts.append("Char Data Missing/Neutral")

        if seg_transforms:
            clip_transforms = {t.lower() for t in clip.get("transformations", [])}
            clip_all_chars = clip_characters | clip_transforms
            transform_overlap = seg_transforms & clip_all_chars
            if transform_overlap:
                score += len(transform_overlap) * intent_weights["alien_bonus"]
                reason_parts.append(f"Alien: {', '.join(transform_overlap)}")
            else:
                alien_sim_total = 0.0
                for alien_name in seg_transforms:
                    alien_sim_total += _get_alien_visual_similarity(clip, alien_name)
                
                if alien_sim_total > 0.25:
                    score += alien_sim_total * intent_weights["alien_bonus"]
                    reason_parts.append(f"CLIP Alien Sim: {alien_sim_total:.2f}")
                elif has_ground_truth:
                    score -= 10.0
                    reason_parts.append(f"Missing Alien Verified")
                else:
                    reason_parts.append("Alien Data Missing/Neutral")

        # Subtitle & Dialogue Alignment Score
        clip_subtitles = clip.get("subtitles", "").strip().lower()
        seg_text_clean = segment_text.strip().lower()
        
        if clip_subtitles and len(clip_subtitles) > 8:
            if clip_subtitles in seg_text_clean or seg_text_clean in clip_subtitles:
                score += 15.0 * (intent_weights["rrf_scale"] / 100.0)
                reason_parts.append("Literal Quote Match")
            else:
                sub_words = set(extract_keywords(clip_subtitles))
                if seg_keywords & sub_words:
                    overlap_cnt = len(seg_keywords & sub_words)
                    score += overlap_cnt * 2.5 * (intent_weights["rrf_scale"] / 100.0)
                    reason_parts.append(f"Dialogue Overlap: {overlap_cnt}")

        sub_emb = clip.get("subtitle_embedding")
        if sub_emb and segment_embedding:
            sub_sim = cosine_similarity(segment_embedding, sub_emb)
            if sub_sim > 0.25:
                score += sub_sim * 9.0 * (intent_weights["rrf_scale"] / 100.0)
                reason_parts.append(f"Sub Vector Sim: {sub_sim:.2f}")

        clip_location = clip.get("location", "").lower()
        if clip_location and clip_location in seg_locations:
            score += 2.0
            reason_parts.append(f"Loc: {clip_location}")
            
        ep_summary = clip.get("episode_summary", "").lower()
        if ep_summary:
            ep_keywords = extract_keywords(ep_summary)
            overlap = seg_keywords & ep_keywords
            score += len(overlap) * intent_weights["plot_bonus"]
            if overlap:
                reason_parts.append(f"Plot Overlap: {len(overlap)}")

        visual_tags = {t.lower() for t in clip.get("visual_tags", [])}
        if visual_tags:
            vtag_overlap = seg_keywords & visual_tags
            score += len(vtag_overlap) * 1.0
            if vtag_overlap:
                reason_parts.append(f"VisTags: {', '.join(vtag_overlap)}")

        raw_score = score
        reason = " | ".join(reason_parts)

        if clip.get("filename", "") in cooldown_set:
            if ban_cooldown:
                continue
            reason += " [PREVIOUSLY USED]"

        scored_clips.append((score, raw_score, clip, reason))

    valid_clips = [x for x in scored_clips if x[1] >= threshold]
    if not valid_clips:
        return [], 0.0, ""

    valid_clips.sort(key=lambda x: x[1], reverse=True)
    unused_clips = [x for x in valid_clips if x[2].get("filename", "") not in cooldown_set]

    if unused_clips:
        best_pool = unused_clips
    else:
        best_pool = valid_clips

    clips_only = [c for s, r, c, reason in best_pool]
    return clips_only[:10], best_pool[0][1], best_pool[0][3]


# ============================================================================
# Clip Scoring — LLM Strategy (Ollama)
# ============================================================================

def match_llm(segment_text: str, clips: list, llm_config: dict) -> tuple:
    """Use Ollama to pick the best clip for a narration segment.

    Sends a prompt with the segment text and a numbered list of clip
    descriptions, asks the LLM to pick the best match by number.

    Returns (top_clips, confidence, reason) or ([], 0.0, "") on failure.
    """
    if not clips:
        return [], 0.0, ""

    clip_descs = []
    for i, clip in enumerate(clips[:20]):
        chars = ", ".join(clip.get("visual_characters", clip.get("characters", [])))
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
                    "temperature": 0.1,
                    "num_predict": 16,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()

        match = re.search(r"\b(\d+)\b", answer)
        if match:
            try:
                chosen_idx = int(match.group(1))
                if 1 <= chosen_idx <= len(clip_descs):
                    chosen_clip = clips[chosen_idx - 1]
                    return [chosen_clip], 1.0, "LLM Pick"
            except ValueError:
                pass
    except requests.RequestException as e:
        log.warning("LLM request failed, falling back to keyword: %s", e)
    except (ValueError, KeyError) as e:
        log.warning("Failed to parse LLM response: %s", e)

    return [], 0.0, ""


# ============================================================================
# Fallback: Generate AI Image Prompt
# ============================================================================

def generate_ai_image_prompt(segment_text: str, show_config: dict) -> str:
    """Create an image-generation prompt for segments with no matching clip."""
    show_name = show_config.get("display_name", "the show")
    clean = re.sub(r"[\"']", "", segment_text)
    if len(clean) > 120:
        clean = clean[:120] + "..."

    return (
        f"Cinematic still from {show_name}, depicting: {clean}. "
        f"Dramatic lighting, animation style, 9:16 vertical composition, "
        f"high detail, vibrant colors."
    )


# ============================================================================
# Assembly Manifest Builder
# ============================================================================

def build_manifest(caption_data: dict, clips: list, show_config: dict,
                   strategy: str, matching_config: dict,
                   llm_config: dict) -> dict:
    """Build an assembly manifest from captions and clip index.

    Maintains a cooldown window to prevent the same clip from being
    selected for consecutive segments. When a top match is on cooldown,
    prefers an adjacent scene from the same episode for visual continuity.
    """
    threshold = matching_config.get("keyword_match_threshold", 1)
    fallback = matching_config.get("fallback", "ai_image")
    max_clip_dur = matching_config.get("max_clip_duration_seconds", 5)
    min_clip_dur = matching_config.get("min_clip_duration_seconds", 1.5)

    cooldown_size = matching_config.get("cooldown_window", 10)
    cooldown_penalty = matching_config.get("cooldown_penalty", -50.0)
    prefer_adjacent = matching_config.get("prefer_adjacent_episode", True)

    eligible_clips = [
        c for c in clips
        if not _is_banned_clip(c)
        and min_clip_dur <= c.get("duration_seconds", 0) <= max_clip_dur
    ]
    log.info(
        "Eligible clips after duration filter (%.1f-%.1fs): %d/%d",
        min_clip_dur, max_clip_dur, len(eligible_clips), len(clips),
    )

    if len(eligible_clips) < 10:
        log.warning("Too few clips after duration filter, using all %d clips", len(clips))
        eligible_clips = [c for c in clips if not _is_banned_clip(c)]

    # Global Episode Affinity has been deprecated in favor of dynamic intent routing and RRF.
    manifest_segments = []
    stats = {"matched": 0, "fallback": 0, "total": 0, "adjacent_used": 0}

    # --- BM25 Initialization ---
    try:
        from scripts.bm25 import SimpleBM25
    except ImportError:
        try:
            from bm25 import SimpleBM25
        except ImportError:
            SimpleBM25 = None
            
    bm25_index = None
    if SimpleBM25:
        bm25_corpus = []
        for c in eligible_clips:
            text_parts = []
            text_parts.extend([ch.lower() for ch in c.get("characters", [])])
            text_parts.append(c.get("location", "").lower())
            text_parts.append(c.get("action", "").lower())
            text_parts.append(c.get("scene_context", "").lower())
            text_parts.extend([t.lower() for t in c.get("tags", [])])
            doc_str = " ".join(text_parts)
            doc_words = re.sub(r"[^a-z0-9\s]", "", doc_str).split()
            bm25_corpus.append(doc_words)
        bm25_index = SimpleBM25(bm25_corpus)

    cooldown_set = set()

    def _push_cooldown(filename: str):
        cooldown_set.add(filename)

    active_characters = set()
    active_locations = set()

    for seg in caption_data.get("segments", []):
        stats["total"] += 1
        seg_text = seg.get("text", "").strip()
        seg_id = seg.get("id", stats["total"] - 1)

        if not seg_text:
            log.warning("Segment %d has empty text, skipping", seg_id)
            continue

        current_chars = extract_character_mentions(seg_text, show_config)
        current_locs = extract_location_mentions(seg_text, show_config)

        if current_chars:
            active_characters = current_chars
        if current_locs:
            active_locations = current_locs

        best_clips = []
        score = 0.0
        reason = ""

        # --- Matching ---
        if strategy == "semantic" and eligible_clips:
            if not hasattr(build_manifest, "embedding_model"):
                try:
                    from sentence_transformers import SentenceTransformer
                    log.info("Loading SentenceTransformer model for semantic matching...")
                    build_manifest.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except ImportError:
                    log.error("sentence-transformers not installed! Falling back to keyword.")
                    strategy = "keyword"

            if strategy == "semantic":
                bm25_scores = [0.0] * len(eligible_clips)
                if bm25_index:
                    query_words = re.sub(r"[^a-z0-9\s]", "", seg_text.lower()).split()
                    bm25_scores = bm25_index.get_scores(query_words)

                best_clips, score, reason = match_semantic(
                    seg_text, eligible_clips, bm25_scores, show_config,
                    build_manifest.embedding_model, threshold=3.0,
                    cooldown_set=cooldown_set,
                    seg_characters=active_characters,
                    seg_locations=active_locations,
                )

        if strategy == "llm" and eligible_clips:
            best_clips, score, reason = match_llm(seg_text, eligible_clips, llm_config)
            if not best_clips:
                best_clips, score, reason = match_keyword(
                    seg_text, eligible_clips, show_config, threshold,
                    cooldown_set=cooldown_set,
                    seg_characters=active_characters,
                    seg_locations=active_locations,
                )
        elif strategy == "keyword" and eligible_clips:
            best_clips, score, reason = match_keyword(
                seg_text, eligible_clips, show_config, threshold,
                cooldown_set=cooldown_set,
                seg_characters=active_characters,
                seg_locations=active_locations,
            )

        best_clip = best_clips[0] if best_clips else None

        # --- Build manifest entry ---
        entry = {
            "id": seg_id,
            "text": seg_text,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "words": seg.get("words", []),
        }

        if best_clip is not None:
            clip_filename = best_clip.get("filename", "")
            entry["visual_type"] = "clip"
            entry["visual_source"] = clip_filename
            entry["visual_sources"] = [c.get("filename", "") for c in best_clips]
            entry["clip_start"] = 0.0
            entry["match_score"] = round(score, 2)
            entry["match_reason"] = reason
            stats["matched"] += 1

            clip_chars = _get_clip_characters(best_clip)
            clip_loc = best_clip.get("location", "").lower()
            if clip_chars:
                active_characters = clip_chars
            if clip_loc:
                active_locations = {clip_loc}

            _push_cooldown(clip_filename)

            log.info(
                "Segment %d -> clip '%s' (score=%.1f, cooldown=%d/%d)",
                seg_id, clip_filename, score,
                len(cooldown_set), cooldown_size,
            )
            log.info("   -> Segment Text: \"%s\"", seg_text)
            log.info("   -> Match Reason: %s", reason)
            
            db_action = best_clip.get("action", "").strip()
            db_tags = ", ".join(best_clip.get("tags", []))
            if db_action:
                log.info("   -> DB Action Match: \"%s\"", db_action)
            if db_tags:
                log.info("   -> DB Tags: %s", db_tags)
        else:
            if fallback == "ai_image":
                entry["visual_type"] = "ai_image"
                entry["visual_source"] = generate_ai_image_prompt(
                    seg_text, show_config
                )
            elif fallback == "generic_broll":
                entry["visual_type"] = "clip"
                entry["visual_source"] = "__generic_broll__"
            else:
                entry["visual_type"] = "black"
                entry["visual_source"] = ""

            entry["clip_start"] = 0.0
            entry["match_score"] = 0.0
            entry["match_reason"] = "Fallback"
            stats["fallback"] += 1
            log.info("Segment %d -> fallback (%s)", seg_id, fallback)

        manifest_segments.append(entry)

    manifest = {
        "audio_file": caption_data.get("audio_file", ""),
        "segments": manifest_segments,
        "stats": stats,
    }

    log.info(
        "Matching complete: %d/%d matched, %d fallback, %d adjacent swaps",
        stats["matched"], stats["total"], stats["fallback"],
        stats.get("adjacent_used", 0),
    )
    return manifest


# ============================================================================
# Caption File Loading
# ============================================================================

def load_caption_files(captions_path: Path) -> list:
    """Load one or more caption JSON files."""
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

    pipeline_cfg = load_pipeline_config()
    matching_cfg = pipeline_cfg.get("clip_matching", {})
    llm_cfg = pipeline_cfg.get("llm", {})

    show_slug, show_config = get_active_show(args.show)
    log.info("Using show: %s (%s)", show_config.get("display_name", "?"), show_slug)

    strategy = args.strategy or matching_cfg.get("strategy", "semantic")
    log.info("Matching strategy: %s", strategy)

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

    captions_path = Path(args.captions)
    caption_files = load_caption_files(captions_path)
    if not caption_files:
        log.error("No caption files found — exiting")
        sys.exit(1)

    for i, caption_data in enumerate(caption_files):
        manifest = build_manifest(
            caption_data, clips, show_config,
            strategy, matching_cfg, llm_cfg,
        )

        if args.output:
            out_path = Path(args.output)
            if len(caption_files) > 1:
                out_path = out_path.parent / f"{out_path.stem}_{i}{out_path.suffix}"
        else:
            out_dir = get_project_path("output_dir", pipeline_cfg)
            out_path = out_dir / f"manifest_{i}.json"

        save_json(out_path, manifest)
        log.info("Assembly manifest saved → %s", out_path)

    log.info("Done — processed %d caption file(s)", len(caption_files))


if __name__ == "__main__":
    main()