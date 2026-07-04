# Ben 10 Clip Matcher — Full Architecture Context
### For brainstorming a better architecture with another model

---

## What This System Does (End-to-End)

The pipeline produces YouTube Shorts by:
1. Generating a narration script about a Ben 10 topic (e.g. "Why Vilgax is the greatest villain")
2. Converting it to TTS audio and getting word-level timestamps via Whisper
3. Splitting narration into short segments (3–8 words each)
4. **For each segment, finding the most relevant video clip from the library** ← this is what's broken
5. Assembling segments + clips into a vertical 1080×1920 video with animated captions

---

## clip_index.json — Structure

The file is a JSON dict with a top-level `"clips"` array. Each entry looks like this:

```json
{
  "filename": "s1e1_scene_042.mp4",
  "filepath": "/path/to/clips/ben10/s1e1_scene_042.mp4",
  "duration_seconds": 3.84,
  "show": "ben10",

  // --- TEXT METADATA (populated by auto-tagging via Ollama Vision) ---
  "characters": ["Ben Tennyson", "Gwen Tennyson"],
  "location": "forest",
  "action": "Ben transforms into Heatblast while Gwen watches",
  "mood": "dramatic",
  "tags": ["transformation", "fire", "alien", "heatblast", "forest"],

  // --- SEMANTIC EMBEDDING (populated by clip_indexer_embed.py) ---
  // Text embedding of "action + tags + characters" via all-MiniLM-L6-v2
  // Used for cosine similarity against narration segment embeddings
  "embedding": [0.023, -0.14, ...],   // 384-dimensional

  // --- VISUAL EMBEDDING (populated by clip_indexer_clip_embed.py) ---
  // CLIP ViT-B-32 embedding of a sampled frame from the clip
  // Used for cross-modal text→image matching
  "clip_visual_embedding": [0.08, -0.03, ...],   // 512-dimensional

  // --- ENRICHMENT (populated by enrich_clip_metadata.py / clip_indexer_allphase.py) ---
  "scene_context": "An outdoor forest scene at night. Ben holds his wrist as the Omnitrix glows.",
  "visual_description": "Bright orange flames with a rocky humanoid figure standing tall.",
  "visual_tags": ["fire", "outdoors", "night", "rocky"],
  "episode_summary": "In this episode Ben discovers the Omnitrix and first transforms...",
  "subtitles": "Ben: I'm going hero!",

  // --- VISUAL CHARACTER TAGS (what we've been building) ---
  // Populated by run_visual_tagging_pipeline_arcmax.py
  "visual_characters": ["Ben Tennyson", "Gwen Tennyson"],
  "detection_meta": {
    "yolo_confirmed": ["Ben Tennyson"],
    "arcface_tagged": ["Gwen Tennyson"],
    "sources": {"Ben Tennyson": "yolo", "Gwen Tennyson": "arcface"}
  }
}
```

**CRITICAL GAPS — fields that are frequently empty/missing:**
- `embedding`: Only populated if `clip_indexer_embed.py` was run. If missing, clip is SILENTLY SKIPPED by the semantic matcher (line 413: `if not clip_emb: continue`)
- `clip_visual_embedding`: Only populated if `clip_indexer_clip_embed.py` was run. Optional bonus, often absent.
- `characters`: May be empty `[]` if auto-tagging via Ollama vision failed, or may contain hallucinated/wrong names
- `visual_characters`: Only populated after running our new YOLO/ArcFace pipeline — not yet run on the other laptop
- `action`, `tags`: Quality varies wildly — Ollama vision model sometimes produces generic descriptions like "two characters talking"

---

## How clip_matcher.py Scores Clips (Semantic Strategy — the default)

For each narration segment (e.g. "Vilgax arrives on Earth to retrieve the Omnitrix"), the matcher:

### Step 1: Encode the segment
```python
segment_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(segment_text)
clip_text_emb = SentenceTransformer('clip-ViT-B-32').encode(segment_text)
```
`all-MiniLM-L6-v2` is a TEXT embedding model. `clip-ViT-B-32` is a CROSS-MODAL model that maps text and images into the same space.

### Step 2: Score every clip
```
score = 0

# Core signal (text→text cosine similarity)
sim = cosine_similarity(segment_embedding, clip["embedding"])
score += sim * 10.0

# Cross-modal bonus (text→frame image similarity)
if clip has clip_visual_embedding:
    visual_sim = cosine_similarity(clip_text_emb, clip["clip_visual_embedding"])
    score += visual_sim * 8.0

# Character match bonus/penalty
if narration mentions characters:
    if clip["characters"] matches those characters: score += len(overlap) * 7.0
    if clip["characters"] does NOT match:           score -= 4.0

# Alien transformation match
if narration mentions "heatblast" etc:
    score += len(overlap) * 5.0

# Location match
score += 2.0 if location matches

# Dominant episode bonus (the episode whose clips share most keywords with full script)
score += 2.0 if clip is from dominant episode

# Episode summary keyword overlap
score += len(overlap) * 1.5

# scene_context + visual_description keyword overlap
score += len(overlap) * 1.5

# visual_tags overlap
score += len(overlap) * 1.0
```

### Step 3: Pick top clip, apply cooldown
Recently used clips get a penalty of `score * 0.01 - 0.001` (effectively zeroed), replaced by an adjacent clip from the same episode.

---

## The Black Screen Problem

When `best_clip is None` (no clip scored above threshold=3.0 OR all clips are on cooldown with no adjacent option), the fallback triggers. Config sets `fallback: "ai_image"` — meaning it tries to generate an AI image prompt. If the AI image generation also fails or isn't hooked up, the assembler gets `visual_type: "black"` and renders a black frame.

**Root cause:** A clip scores 0 because it has no `embedding` field → silently skipped → no clips survive the threshold → fallback.

---

## Known Failure Modes (Ranked by Severity)

### 1. The Embedding Gap — CRITICAL
**Problem:** `clip_matcher.py` line 413 silently skips any clip without an `embedding` field: `if not clip_emb: continue`. If embedding generation was never run, or ran partially, the matcher has a fraction of clips to choose from. A completely untagged clip library means ZERO candidates, pure fallback, black screen.

**How to diagnose:** Count `len([c for c in clips if 'embedding' in c])` vs total clips. If this is much lower than total, that's your black screen.

### 2. Wrong Characters in clip_index — HIGH
**Problem:** The `characters` field is populated by Ollama Vision looking at ONE frame of the clip. Ollama is a generic LLM, not trained on Ben 10 characters. It often outputs `["unknown"]`, `["boy", "girl"]`, or confidently wrong names. Since character matching has the highest weight (7 points per match, -4 for mismatch), a clip with wrong characters will be actively suppressed when the narration mentions the right ones.

**Example failure:** Narration says "Ben transforms into Heatblast." Clip is actually Ben transforming. But Ollama tagged it as `characters: ["boy in green jacket"]`. The character extractor looks for "Ben Tennyson" in the characters list → no match → `-4.0` penalty. Another clip where Ollama happened to write "Ben" wins instead, even if it's a random walking scene.

**Status:** Our YOLO + ArcFace pipeline fixes this, but it hasn't been run on the other laptop yet.

### 3. The Text Embedding is Matching Tone, Not Content — HIGH
**Problem:** `all-MiniLM-L6-v2` is a general-purpose sentence similarity model. It matches semantic meaning and tone, not specific events. The narration embedding for "Vilgax destroys the ship" is very close to the clip embedding of "characters fighting dramatically" — both have combat-related vocabulary. But it's completely blind to whether Vilgax is actually in the clip, what specific attack is happening, or whether the clip even contains action.

The clip embedding is built from: `action + tags + characters` concatenated as text, then embedded. This is essentially matching vibes, not specifics.

### 4. Dominant Episode Bias — MEDIUM
**Problem:** The matcher pre-computes the single episode that shares the most keywords with the full script. That episode gets a flat +2.0 bonus on every clip selection. For a 60-second script about Vilgax, this might correctly focus on the Vilgax episodes. But for general topics, it locks the entire video to one episode's B-roll, making every clip visually monotonous and increasing cooldown conflicts.

### 5. Cooldown Forcing Adjacent Clips — MEDIUM
**Problem:** The cooldown window is 10 clips. When a clip is on cooldown, it preferentially picks an "adjacent scene" (within 30 scenes) from the same episode. This creates visual monotony — a 30-second Short can end up using 8 consecutive scene cuts from s1e1 episodes 20–50, feeling like raw episode footage rather than a curated highlight reel.

### 6. Duration Filter Eliminating Too Many Clips — LOW/MEDIUM
**Problem:** Config sets `max_clip_duration_seconds: 5` and `min_clip_duration_seconds: 1.3`. Scene splits at a threshold of 27.0 (content detector) produce highly variable clip lengths. Very action-heavy episodes with fast cuts produce many clips under 1.3 seconds. Conversational scenes produce clips over 5 seconds. Both ends get filtered, leaving a biased subset.

### 7. No Hard Character Filter — LOW
**Problem:** The system uses soft scoring for character matching rather than a hard pre-filter. If narration says "Vilgax attacks" and there are 50 clips with Vilgax in the library, those 50 clips compete on equal footing against 11,500 other clips with only a +7.0 point advantage. A random clip with many keyword hits from its `action` or `tags` field can beat a correctly tagged Vilgax clip.

---

## What Data We Actually Have (Summary)

| Field | Coverage | Quality | Used By |
|---|---|---|---|
| `filename`, `duration_seconds` | 100% | Perfect | Duration filter |
| `characters` (Ollama) | ~90% | Low — Ollama hallucinates | Character boost/penalty |
| `action` (Ollama) | ~90% | Medium — generic descriptions | Keyword + embedding text |
| `tags` (Ollama) | ~90% | Medium | Keyword + embedding text |
| `location` (Ollama) | ~90% | Medium | Location bonus |
| `embedding` (MiniLM) | Unknown — possibly partial | Good IF the text it was built from is good | Core cosine sim ×10 |
| `clip_visual_embedding` (CLIP) | Unknown — possibly sparse | Good | Visual bonus ×8 |
| `scene_context` (enrichment) | Partial | Good | Keyword overlap ×1.5 |
| `episode_summary` | Partial | Good | Keyword overlap ×1.5 |
| `subtitles` | Partial | Excellent | Not currently used in scoring! |
| `visual_characters` (YOLO/ArcFace) | NOT YET RUN on main laptop | Very Good | Not yet wired into clip_matcher |

---

## The Subtitle Blind Spot

`subtitles` is a field that contains the actual dialogue spoken in the clip (cross-referenced from SRT files). This is extremely high-quality ground truth — if Ben says "It's hero time!" in the clip, the subtitle field will contain exactly that. But **clip_matcher.py does not use the subtitles field at all in scoring**. This is probably the single highest-value unused signal in the entire system.

---

## The Cross-Modal Embedding Architecture

The `clip_visual_embedding` is a 512-dimensional CLIP image embedding of one sampled frame. The matcher encodes the narration text with CLIP's text encoder into the same 512-dimensional space, then computes cosine similarity. This is theoretically the best signal because CLIP was explicitly trained to align text and images.

**But:** The weight is `visual_sim * 8.0`, while text-embedding similarity is `sim * 10.0`. Text-to-text similarity wins even when the visual match is perfect. More critically, CLIP ViT-B-32 is trained on real photos. Animated characters from Ben 10 live in a different visual distribution — CLIP knows "fire" and "alien" but struggles to distinguish Heatblast from Swampfire from Fourarms based on color/shape alone.

---

## Ideas for a Better Architecture (Seed for Brainstorming)

1. **Hard pre-filter on characters, then rank within that pool.** If narration mentions "Vilgax", pre-filter to only clips where `visual_characters` contains "Vilgax". Then run embedding similarity only within that filtered set. Eliminates the 11,500 vs 50 unfairness.

2. **Use subtitle embeddings.** Embed the `subtitles` field separately (it's dialogue, very different from action description). Narration about "hero time" should match clips where Ben says "It's hero time!". Dialogue-to-dialogue semantic matching is far more precise than action-description matching.

3. **Multi-vector retrieval.** Give each clip 3 separate embeddings: (a) the visual description text, (b) the dialogue/subtitles, (c) the character+event tags. Match narration against each independently, score separately, fuse. This avoids one bad field dragging down the whole score.

4. **Replace MiniLM with a domain-adapted model.** Fine-tune the sentence embeddings on Ben 10 episode summaries + clip descriptions. Currently MiniLM treats "Ben transforms" and "character changes form" as equally similar, losing specificity.

5. **Episode-level routing before clip-level search.** First decide which episode(s) are most relevant to the narration (based on episode summaries in the index). Then search only those episodes' clips. Reduces search space from 11,574 clips to ~200 per episode, and improves precision.

6. **Temporal context within episode.** If a segment says "after the transformation, Vilgax grabs him," find the clip that comes AFTER a transformation clip from the same episode. Scene sequence information is available in the filename (`_scene_042`) but not currently exploited for narrative flow matching.

7. **Confidence-gated fallback.** Instead of falling back to AI image when score < 3.0, use a tiered system: score > 5 → use clip directly, score 2–5 → use clip but apply Ken Burns zoom to mask quality, score < 2 → use a "generic action montage" from pre-selected highlight clips, score 0 or black → only then AI image.
