# Script Generator v2 — Implementation Plan

> **Goal:** Fix topic relevancy by adding a **Verifier LLM** layer that fact-checks generated scripts using web search, and an **Episode Anchor** system that tags each clip/episode with a canonical summary so the pipeline always knows *what each episode was actually about*.

---

## The Core Problem

```
Current Flow (broken relevancy):

  Topic → RAG (subtitles + wiki + theories) → Script Generator LLM → Output
                                                      ↓
                                              No validation.
                                              LLM hallucinates freely.
                                              RAG keyword match is fuzzy.
                                              Script may reference wrong episodes,
                                              wrong characters, or made-up events.
```

The RAG manager uses basic keyword matching for theories/wiki and a small embedding model for subtitles. The LLM then takes this fuzzy context and generates a script with no grounding check. There's no feedback loop — if it's wrong, nobody catches it.

---

## Architecture: What We're Building

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VERIFIER-CORRECTOR LOOP                       │
│                                                                      │
│   ┌─────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│   │  Topic   │───▶│  Web Research │───▶│   Research Dossier       │   │
│   │  (known) │    │  Agent (LLM2)│    │  (ground truth facts)    │   │
│   └─────────┘    └──────────────┘    └────────────┬─────────────┘   │
│                                                    │                 │
│                                                    ▼                 │
│   ┌──────────────────┐    ┌────────────────────────────────────┐    │
│   │  Script Generator │───▶│        Verifier LLM (LLM3)        │    │
│   │  (LLM1 - Ollama)  │    │                                    │    │
│   │                    │    │  Compares script vs. dossier:      │    │
│   │  Existing code,    │    │  • Wrong episode references?       │    │
│   │  RAG-powered       │    │  • Made-up events?                 │    │
│   │                    │    │  • Incorrect character details?     │    │
│   └────────▲───────────┘    │  • Missing key facts?              │    │
│            │                │                                    │    │
│            │                │  Output: PASS or CORRECTIONS[]     │    │
│            │                └───────────────┬────────────────────┘    │
│            │                                │                        │
│            │    ┌───────────────────────┐    │                        │
│            └────│  Correction Prompt     │◀──┘                        │
│                 │  Builder               │   (if CORRECTIONS)        │
│                 │  "Fix these specific   │                            │
│                 │   issues: [...]"       │   Max 2 retry loops        │
│                 └───────────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Episode Anchor System

### Why This Comes First

Before we can verify scripts, we need ground truth. Right now clips are tagged with characters/locations/mood, but **no clip knows what the episode it came from was actually about**. A clip tagged `characters: ["rick", "morty"], location: "garage"` tells us nothing about the plot.

### 1.1 Episode Summary Index

Create a new file: `episode_index.json`

```json
{
  "s1e1": {
    "title": "Pilot",
    "one_line": "Rick takes Morty on his first adventure to get mega tree seeds.",
    "summary": "Rick drags a drunk Morty through a portal to Dimension 35-C to collect mega tree seeds. Morty has to shove them way up inside his butt. Meanwhile, Beth and Jerry argue about Rick's influence. The seeds give Morty temporary super-intelligence before wearing off.",
    "key_events": [
      "Rick builds portal gun in garage",
      "Morty falls off cliff in Dimension 35-C",
      "Mega tree seeds give temporary intelligence",
      "Rick threatens Morty at gunpoint (bluff)"
    ],
    "characters_featured": ["rick", "morty", "beth", "jerry", "mr_goldenfold"],
    "themes": ["family dysfunction", "science vs morality", "Rick's manipulation"],
    "locations": ["garage", "dimension 35-C", "harry herpson high school"]
  },
  "s1e2": { ... },
  ...
}
```

### 1.2 How to Populate It

**Option A: LLM + Subtitles (recommended — fully local)**

We already have all the subtitle files in `rick_and_morty_subtitles/Subtitles_Allinone`. Send each episode's full subtitle text to the LLM with this prompt:

```
You are given the full subtitle transcript of {show_name} Season {season} Episode {episode}.

Extract:
1. "title": The episode title
2. "one_line": A single sentence summarizing the entire episode
3. "summary": A 3-4 sentence plot summary covering the main arc
4. "key_events": Array of 4-8 specific, factual events that happened (no opinions)
5. "characters_featured": Array of character names who appeared
6. "themes": Array of 2-4 themes explored
7. "locations": Array of locations shown

Output valid JSON only.
```

**Option B: Web scrape** (faster but external dependency)
Scrape from rickandmorty.fandom.com — you already have `scrape_fandom.py`.

### 1.3 Link Clips to Episode Anchors

Update the clip index to include an `episode_anchor` field:

```json
{
  "filename": "s9e1_scene_029.mp4",
  "episode_id": "s9e1",
  "episode_one_line": "Rick and Morty deal with...",
  "characters": ["rick", "morty"],
  ...
}
```

This lets the clip matcher and script generator know **exactly what context** each clip belongs to.

### 1.4 New Script: `episode_indexer.py`

```python
"""
episode_indexer.py — Build canonical episode summaries from subtitles.

Reads subtitle files, sends to LLM, produces episode_index.json.
Run once per show, update when new episodes release.
"""

class EpisodeIndexer:
    def __init__(self, pipeline_config, show):
        self.subtitles_dir = get_project_path("subtitles_dir", pipeline_config)
        self.output_path = PROJECT_ROOT / "episode_index.json"
        self.llm_config = pipeline_config.get("llm", {})

    def index_all_episodes(self):
        """Process all subtitle files and generate summaries."""
        # 1. Glob all .srt/.txt files
        # 2. Parse season/episode from filename (s1e1, s2e3, etc.)
        # 3. Send each to LLM with extraction prompt
        # 4. Save to episode_index.json

    def index_single_episode(self, subtitle_path):
        """Process one episode's subtitles."""
        # Read subtitles → build prompt → call_ollama → parse JSON

    def enrich_clip_index(self):
        """Add episode_one_line to each clip in clip_index.json."""
        # Load episode_index.json + clip_index.json
        # For each clip, match s{X}e{Y} → add one_line and summary
```

---

## Part 2: Web Research Agent

### 2.1 Why Not Just More RAG?

RAG searches your **local** data (subtitles, wiki scrapes, theories). But local data has gaps:
- Wiki scrapes might be outdated
- Subtitle context is dialogue, not plot analysis
- Fan theories might be wrong
- You don't have every episode fully covered

A web search gives you **external ground truth** from wikis, Reddit discussions, YouTube analyses, and fan sites — things your local RAG will never have.

### 2.2 Web Research Agent Design

New script: `web_researcher.py`

```python
"""
web_researcher.py — Research a topic using web search before script generation.

Builds a "research dossier" of verified facts that the script generator
and verifier can use as ground truth.
"""

class WebResearcher:
    def __init__(self, pipeline_config):
        self.llm_config = pipeline_config.get("llm", {})
        self.search_config = pipeline_config.get("web_research", {})

    def research_topic(self, topic: str, show_name: str) -> dict:
        """
        Given a topic, perform web searches and compile a dossier.

        Returns:
        {
            "topic": "Why did Rick destroy the Citadel?",
            "search_queries": ["rick destroy citadel reason", ...],
            "facts": [
                {"fact": "Rick destroyed the Citadel in S3E7", "source": "rickandmorty.fandom.com"},
                {"fact": "Evil Morty orchestrated the final destruction in S5E10", "source": "reddit.com"},
                ...
            ],
            "relevant_episodes": ["s3e7", "s5e10", "s1e10"],
            "key_details": "Rick's hatred of the Citadel stems from...",
            "common_misconceptions": ["Many fans think Rick destroyed it to save Morty, but..."]
        }
        """

    def _generate_search_queries(self, topic: str, show_name: str) -> list[str]:
        """Use LLM to generate 3-5 targeted search queries for the topic."""
        # Prompt: "Generate 3-5 web search queries to fact-check a YouTube script about: {topic}"
        # Example output: ["rick and morty citadel destruction episode", "why did rick hate citadel of ricks", ...]

    def _execute_search(self, query: str) -> list[dict]:
        """Run a web search and return top results."""
        # Option 1: SearXNG (self-hosted, free, private)
        # Option 2: DuckDuckGo API (free, no key needed)
        # Option 3: Brave Search API (free tier: 2000/month)
        # Option 4: Google Custom Search API (100 free/day)

    def _extract_facts(self, search_results: list[dict], topic: str) -> list[dict]:
        """Use LLM to extract relevant facts from search result snippets."""
        # Send search snippets to LLM
        # Ask it to extract factual claims relevant to the topic
        # Tag each fact with its source URL

    def _compile_dossier(self, topic, queries, all_facts, episodes) -> dict:
        """Compile all research into a structured dossier."""
```

### 2.3 Search Engine Options (ranked)

| Engine | Cost | Rate Limit | Setup |
|--------|------|-----------|-------|
| **DuckDuckGo** (`duckduckgo-search` pip) | Free | ~30/min | `pip install duckduckgo-search` — **start here** |
| **SearXNG** (self-hosted) | Free | Unlimited | Docker container, more setup |
| **Brave Search API** | Free tier | 2000/month | API key required |
| **Google CSE** | Free tier | 100/day | Google Cloud setup |
| **Serper.dev** | Free tier | 2500 queries | API key, very fast |

**Recommendation:** Start with `duckduckgo-search`. Zero cost, zero setup, no API keys. Upgrade to Brave or Serper if you need more volume.

```python
# Minimal search implementation
from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results]
```

### 2.4 Research Dossier Format

The dossier is the **contract** between the researcher and the verifier. Example:

```json
{
  "topic": "Why does Rick hate time travel?",
  "researched_at": "2026-06-25T06:30:00Z",
  "search_queries_used": [
    "rick and morty time travel stance",
    "rick sanchez hates time travel reason",
    "rick and morty time travel box shelf"
  ],
  "verified_facts": [
    {
      "fact": "Rick has a box labeled 'TIME TRAVEL STUFF' on his garage shelf that is never used",
      "episodes": ["multiple — visible in background"],
      "source": "rickandmorty.fandom.com/wiki/Time_travel",
      "confidence": "high"
    },
    {
      "fact": "Dan Harmon (co-creator) has stated he considers time travel a 'cop out' in storytelling",
      "episodes": [],
      "source": "reddit.com/r/rickandmorty discussion",
      "confidence": "high"
    },
    {
      "fact": "Season 5 Episode 1 briefly references time travel with 'Mr. Nimbus' storyline but doesn't use it",
      "episodes": ["s5e1"],
      "source": "screen rant article",
      "confidence": "medium"
    }
  ],
  "relevant_episodes": ["s5e1"],
  "common_fan_theories": [
    "Rick avoids time travel because it would let him undo Diane's death, which he considers 'cheating'"
  ]
}
```

---

## Part 3: Verifier LLM (The Core Innovation)

### 3.1 Verifier Design

The verifier sits **after** the script generator and **before** the script is saved. It has access to:
1. The generated script
2. The research dossier (web search facts)
3. The episode index (canonical episode summaries)
4. The original RAG context

New script: `script_verifier.py`

```python
"""
script_verifier.py — Fact-checks generated scripts against web research
and episode anchors.

Returns PASS or a list of specific corrections for the script generator
to fix.
"""

class ScriptVerifier:
    def __init__(self, pipeline_config):
        self.llm_config = pipeline_config.get("llm", {})
        self.max_retries = pipeline_config.get("verification", {}).get("max_retries", 2)
        self.episode_index = self._load_episode_index()

    def verify(self, script: str, dossier: dict, topic: str) -> dict:
        """
        Verify a script against the research dossier.

        Returns:
        {
            "verdict": "PASS" | "NEEDS_CORRECTION",
            "score": 8,          # 1-10 accuracy score
            "corrections": [
                {
                    "type": "wrong_episode",
                    "issue": "Script claims Rick destroyed the Citadel in S3E1, but it was S3E7",
                    "fix": "Change S3E1 reference to S3E7 'The Ricklantis Mixup'"
                },
                {
                    "type": "fabricated_event",
                    "issue": "Script claims Morty used the portal gun alone in the pilot, which never happened",
                    "fix": "Remove this claim or replace with an actual pilot event"
                },
                {
                    "type": "missing_context",
                    "issue": "Script doesn't mention that Evil Morty was the one who ultimately destroyed the Citadel",
                    "fix": "Add mention of Evil Morty's role in the final Citadel destruction (S5E10)"
                }
            ],
            "factual_claims_checked": 7,
            "factual_claims_correct": 5
        }
        """

    def _build_verification_prompt(self, script, dossier, topic) -> str:
        """
        Build the prompt for the verifier LLM.
        """
        return f"""You are a ruthless fact-checker for a YouTube Shorts channel about {dossier.get('show_name', 'a TV show')}.

TOPIC: {topic}

GENERATED SCRIPT:
---
{script}
---

VERIFIED FACTS FROM WEB RESEARCH:
---
{self._format_facts(dossier)}
---

EPISODE DATABASE:
---
{self._format_relevant_episodes(dossier)}
---

YOUR JOB:
1. Read the script line by line
2. Identify every factual claim made in the script
3. Cross-reference each claim against the verified facts and episode database
4. Flag any claim that is:
   - Referencing the wrong episode or season
   - Describing an event that didn't happen in the show
   - Attributing an action to the wrong character
   - Presenting a fan theory as confirmed canon without qualifying it
   - Missing a crucial fact that would make the script misleading

OUTPUT FORMAT (JSON only):
{{
    "verdict": "PASS" or "NEEDS_CORRECTION",
    "score": <1-10 accuracy score>,
    "corrections": [
        {{
            "type": "<wrong_episode|fabricated_event|wrong_character|unqualified_theory|missing_context>",
            "issue": "<what is wrong>",
            "fix": "<specific instruction to fix it>"
        }}
    ]
}}

If the script is factually sound, return verdict "PASS" with an empty corrections array.
Be strict. A single wrong episode reference is enough for NEEDS_CORRECTION.
"""

    def build_correction_prompt(self, original_topic, original_script, corrections) -> str:
        """
        Build a targeted correction prompt for the script generator.
        Instead of regenerating from scratch, we ask it to fix specific issues.
        """
        corrections_text = "\n".join(
            f"- [{c['type']}] {c['issue']} → FIX: {c['fix']}"
            for c in corrections
        )
        return f"""You previously generated a script about: {original_topic}

YOUR PREVIOUS SCRIPT:
---
{original_script}
---

A fact-checker found these issues:
{corrections_text}

Rewrite the script fixing ONLY the issues listed above.
Keep the same tone, structure, length (120-180 words), and hook style.
Do NOT add any new unverified claims.
Output ONLY the corrected raw narration text. No markdown, no notes.
"""
```

### 3.2 The Verification Loop

```python
def generate_verified_script(topic, show, pipeline_config, rag_manager, web_researcher, verifier):
    """
    Full flow: research → generate → verify → correct → verify → save.
    Max 2 correction loops to avoid infinite retries.
    """

    # Step 1: Web research
    log.info("📡 Researching topic: %s", topic)
    dossier = web_researcher.research_topic(topic, show["display_name"])

    # Step 2: Generate initial script (existing code, RAG-powered)
    prompt = build_script_prompt(topic, show, pipeline_config, rag_manager)

    # Inject dossier facts into the prompt as additional context
    dossier_context = format_dossier_for_prompt(dossier)
    prompt += f"\n\nADDITIONAL VERIFIED FACTS:\n{dossier_context}"

    script = call_ollama(prompt, pipeline_config)

    # Step 3: Verify loop
    for attempt in range(verifier.max_retries):
        log.info("🔍 Verification attempt %d/%d", attempt + 1, verifier.max_retries)
        result = verifier.verify(script, dossier, topic)

        if result["verdict"] == "PASS":
            log.info("✅ Script passed verification (score: %d/10)", result["score"])
            break

        log.warning(
            "⚠️ Script needs correction (%d issues, score: %d/10)",
            len(result["corrections"]), result["score"]
        )

        # Build correction prompt and regenerate
        correction_prompt = verifier.build_correction_prompt(topic, script, result["corrections"])
        script = call_ollama(correction_prompt, pipeline_config)

    else:
        log.warning("🟡 Max retries reached. Using best available script.")

    return save_script(topic, script, pipeline_config)
```

---

## Part 4: Episode Anchors for Clips (Marking Episodes)

### 4.1 The Idea

Every clip in your library came from a specific episode. Right now clips have `characters`, `location`, `mood`, `tags` — but **not what the episode was about**. This means:

- The clip matcher can find a clip with Rick in the garage, but can't tell if that clip is from the pilot or from the finale
- The script generator can reference "that scene where Rick..." but can't specify which episode

### 4.2 Implementation

Extend the clip index with episode-level metadata:

```json
{
  "filename": "s9e1_scene_029.mp4",
  "filepath": "clips/s9e1_scene_029.mp4",
  "episode_id": "s9e1",
  "episode_title": "The Jerrick Trap",
  "episode_summary": "Rick and Jerry are forced to work together when they're trapped in a dimension where Jerry is the genius and Rick is the idiot.",
  "characters": ["rick", "jerry"],
  "location": "alternate dimension",
  ...
}
```

### 4.3 Enrichment Script

```python
def enrich_clips_with_episode_data(clip_index_path, episode_index_path):
    """
    One-time enrichment: add episode metadata to every clip.
    Matches clips to episodes using the sXeY prefix in filenames.
    """
    episodes = load_json(episode_index_path)
    clip_data = load_json(clip_index_path)

    for clip in clip_data.get("clips", []):
        # Extract episode ID from filename: "s9e1_scene_029.mp4" → "s9e1"
        match = re.match(r"(s\d+e\d+)", clip["filename"])
        if match and match.group(1) in episodes:
            ep = episodes[match.group(1)]
            clip["episode_id"] = match.group(1)
            clip["episode_title"] = ep.get("title", "")
            clip["episode_summary"] = ep.get("one_line", "")

    save_json(clip_index_path, clip_data)
```

### 4.4 How This Helps Everything

| Component | Before | After |
|-----------|--------|-------|
| **Script Generator** | "Rick was in the garage" (which time?) | "In S1E1, Rick builds the portal gun in the garage to take Morty to Dimension 35-C" |
| **Clip Matcher** | Finds any garage clip | Finds the specific S1E1 garage clip that matches the script's context |
| **Verifier** | Can't check episode references | Can cross-reference "S3E7" claims against episode_index.json |
| **Topic Miner** | Generates generic topics | Can generate episode-specific topics: "The darkest detail in S3E7" |

---

## Part 5: Upgraded RAG Manager

### 5.1 What Changes

The RAG manager currently has 3 sources: subtitles (ChromaDB), theories (JSON keyword match), wiki (JSON keyword match). We add two more:

```python
class RAGManagerV2(RAGManager):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.episode_index = self._load_episode_index()

    def get_combined_context(self, query: str, dossier: dict = None) -> str:
        """Enhanced context that includes episode anchors and web research."""
        canon = self.query_subtitles(query)
        wiki = self.query_wiki(query)
        theories = self.query_theories(query)
        episodes = self._query_episode_index(query)

        combined = []
        if canon:
            combined.append(canon)
        if wiki:
            combined.append(wiki)
        if episodes:
            combined.append(episodes)
        if theories:
            combined.append(theories)

        # Web research dossier gets highest priority
        if dossier:
            combined.insert(0, self._format_dossier(dossier))

        if not combined:
            return "No additional context found."

        return "\n\n".join(combined)

    def _query_episode_index(self, query: str) -> str:
        """Find relevant episodes based on the query."""
        # Semantic search or keyword match against episode summaries
        # Return matching episode summaries as context
```

---

## Part 6: Full Pipeline Integration

### 6.1 Updated Pipeline Flow

```
Phase 0 (one-time setup):
  episode_indexer.py → episode_index.json

Phase 1a: Topic Mining (existing, mostly unchanged)
  topic_miner.py → queue.json

Phase 1b: Script Generation (UPGRADED)
  ┌─────────────────────────────────────────────────────┐
  │  1. web_researcher.py researches the topic          │
  │  2. script_generator.py generates script            │
  │     (now with dossier + episode index in context)   │
  │  3. script_verifier.py fact-checks the script       │
  │  4. If NEEDS_CORRECTION → send fixes back to step 2 │
  │  5. Max 2 retries, then save best version           │
  └─────────────────────────────────────────────────────┘

Phase 2-7: (unchanged — TTS, caption, match, assemble, thumbnail, publish)
```

### 6.2 Config Additions

Add to `pipeline_config.yaml`:

```yaml
# -- Web Research (for script verification) --
web_research:
  enabled: true
  engine: "duckduckgo"           # "duckduckgo", "brave", "serper", "searxng"
  max_results_per_query: 5
  max_queries_per_topic: 3
  api_key: null                  # Required for brave/serper
  cache_results: true            # Cache research to avoid repeated searches
  cache_dir: "./research_cache"

# -- Script Verification --
verification:
  enabled: true
  max_retries: 2                 # Max correction loops
  min_score: 7                   # Scripts scoring below this get regenerated from scratch
  verifier_model: null           # null = use same model as llm.model, or specify a different one
  save_dossier: true             # Save research dossiers alongside scripts for debugging
  save_verification_log: true    # Save the verifier's output for review

# -- Episode Index --
episode_index:
  path: "./episode_index.json"
  auto_enrich_clips: true        # Automatically add episode data to clip_index on first run
```

### 6.3 Orchestrator Changes

In `orchestrator.py`, the `run_script_gen()` function changes from:

```python
# OLD
script_path = generate_script_for_topic(topic, show, pcfg, rag_manager)
```

to:

```python
# NEW
from web_researcher import WebResearcher
from script_verifier import ScriptVerifier

web_researcher = WebResearcher(pcfg)
verifier = ScriptVerifier(pcfg)

script_path = generate_verified_script(
    topic, show, pcfg, rag_manager, web_researcher, verifier
)
```

---

## Part 7: Topic Mining with the Same Pattern

You mentioned wanting to do "something like this for topics too". Here's how:

### 7.1 Topic Verifier

After the topic miner generates topics, run a verification pass:

```python
class TopicVerifier:
    def verify_topics(self, topics: list[dict], show_name: str) -> list[dict]:
        """
        For each mined topic:
        1. Quick web search to confirm the topic is about a real event/detail
        2. Check it's not based on a misconception
        3. Confirm there's enough material for a 60-second script
        4. Rate the topic's viral potential based on search interest

        Returns the filtered/ranked list with verification metadata.
        """
        verified = []
        for topic in topics:
            # Search: does this topic reference real show events?
            results = search_web(f"{show_name} {topic['topic']}")

            if len(results) < 2:
                log.warning("Topic has low web coverage, may be obscure: %s", topic["topic"])
                topic["verified"] = False
                topic["skip_reason"] = "low_coverage"
                continue

            topic["verified"] = True
            topic["web_coverage"] = len(results)
            topic["sources"] = [r["url"] for r in results[:3]]
            verified.append(topic)

        return verified
```

---

## Implementation Order (What to Build First)

### Sprint 1: Foundation (Day 1-2)

| # | Task | File | Effort |
|---|------|------|--------|
| x | Build `episode_indexer.py` | `scripts/episode_indexer.py` | DONE |
| x | Run it to generate `episode_index.json` | one-time | DONE |
| x | Build enrichment to add episode data to `clip_index.json` | inside `episode_indexer.py` | DONE |

### Sprint 2: Web Research (Day 2-3)

| # | Task | File | Effort |
|---|------|------|--------|
| x | Install `duckduckgo-search` | `requirements.txt` | DONE |
| x | Build `web_researcher.py` | `scripts/web_researcher.py` | DONE |
| x | Test with 5 topics, review dossier quality | manual | DONE |

### Sprint 3: Verifier Loop (Day 3-4)

| # | Task | File | Effort |
|---|------|------|--------|
| x | Build `script_verifier.py` | `scripts/script_verifier.py` | DONE |
| x | Create verification prompt template | `prompts/verify_prompt.txt` | DONE |
| x | Create correction prompt template | `prompts/correction_prompt.txt` | DONE |
| x | Integrate into `script_generator.py` | modify existing | DONE |
| x | Add verification config to `pipeline_config.yaml` | modify existing | DONE |

### Sprint 4: Orchestrator Integration (Day 4-5)

| # | Task | File | Effort |
|---|------|------|--------|
| x | Update `orchestrator.py` to use verified flow | modify existing | DONE |
| 13 | Add topic verification (optional) | `scripts/topic_verifier.py` | OPTIONAL |
| x | End-to-end test with topics | manual | DONE |


**Total estimated effort: 4-5 focused days**

---

## Performance Considerations

### LLM Call Count Per Script (Before vs After)

| Step | Before | After |
|------|--------|-------|
| Script generation | 1 call | 1 call |
| Web search query generation | — | 1 call |
| Fact extraction from search results | — | 1 call |
| Verification | — | 1 call |
| Correction (if needed, 0-2x) | — | 0-2 calls |
| **Total** | **1 call** | **4-6 calls** |

With Ollama on local GPU, each call is ~10-30 seconds. So the full verified flow adds ~1-3 minutes per script. For a batch of 10 scripts, that's ~10-30 extra minutes — totally acceptable for significantly better accuracy.

### Caching Strategy

```python
# Cache web research by topic (avoid re-searching the same topic)
research_cache/
  why_rick_hates_time_travel.json    # Dossier
  why_did_rick_destroy_citadel.json  # Dossier
  ...
```

Research dossiers are cached and reused if the same topic comes up again (e.g., during a retry or re-run). Cache TTL: 7 days (configurable).

---

## Alternative Approaches Considered

### ❌ Just improve the RAG (not enough)
More RAG context doesn't help if the LLM hallucinates on top of correct context. You need an independent check.

### ❌ Fine-tune the LLM on show data
Requires thousands of examples, expensive, and makes the model brittle. The verifier pattern is model-agnostic.

### ❌ Use a bigger model for generation
A 70B model hallucinates less than 8B, but still hallucinates. The verifier pattern works regardless of model size.

### ✅ Verifier + Web Research (chosen)
- Model-agnostic (works with llama3.1:8b or any model)
- Self-correcting (gets better with each retry)
- Grounded in external facts (not just regurgitating training data)
- Transparent (dossiers and verification logs are saved for debugging)
- The same pattern scales to topic mining

---

## New File Structure After Implementation

```
scripts/
  ├── episode_indexer.py       ← NEW: Build episode summaries from subtitles
  ├── web_researcher.py        ← NEW: Web search for topic research
  ├── script_verifier.py       ← NEW: Fact-check scripts against dossiers
  ├── topic_verifier.py        ← NEW: Verify mined topics are legit
  ├── script_generator.py      ← MODIFIED: Integrates verification loop
  ├── rag_manager.py           ← MODIFIED: V2 with episode index support
  ├── orchestrator.py          ← MODIFIED: Uses verified generation flow
  ├── topic_miner.py           ← MODIFIED: Optional topic verification
  └── ... (rest unchanged)

prompts/
  ├── script_prompt.txt        ← existing
  ├── topic_prompt.txt         ← existing
  ├── verify_prompt.txt        ← NEW: Verification prompt template
  ├── correction_prompt.txt    ← NEW: Correction prompt template
  └── episode_extract_prompt.txt ← NEW: Episode summary extraction prompt

config/
  └── pipeline_config.yaml     ← MODIFIED: new web_research + verification sections

episode_index.json             ← NEW: Canonical episode summaries
research_cache/                ← NEW: Cached web research dossiers
```

---

## Summary

The key insight: **generation and verification should be separate concerns handled by separate LLM calls**. Your script generator is the "writer" — creative, fast, occasionally wrong. The verifier is the "editor" — strict, fact-driven, catches mistakes. The web researcher is the "researcher" — provides ground truth neither the writer nor editor could have known on their own.

### v2.1 Upgrade: Fresh Prompt Injection & Guardrails Filter
When correcting factual errors, small models (8B) struggle to surgically patch their own broken output without outputting meta commentary (*"Let's fact check this..."*). To solve this:
1. **Fresh Prompt Injection (Solution #2)**: Instead of passing broken drafts back to the LLM, the verifier extracts the exact demanded facts and appends them to the **original generation prompt** for a completely fresh draft.
2. **Output Guardrails Sanitizer (Solution #3)**: A final regex and trigger filter (`_sanitize_script`) strips any stray meta-commentary sentences before the script is saved to disk.

This is the same pattern used in production AI systems (Google, OpenAI, etc.): generate → verify → correct. It works because catching errors is easier than preventing them.
