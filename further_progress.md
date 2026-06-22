# AI Explainer Channel — Further Progress Roadmap

> Last updated: 2026-06-22
> Current status: MVP pipeline running — topic mining → script gen (RAG-powered) → TTS → captioning → clip matching → video assembly → per-topic output folders

---

## What's Working Now ✅

- [x] Topic mining via LLM (Ollama)
- [x] Script generation with RAG (wiki + theories + subtitles via ChromaDB)
- [x] Local TTS with Piper
- [x] GPU TTS via Colab (StyleTTS2)
- [x] Word-level captioning (faster-whisper)
- [x] Clip matching (keyword-based)
- [x] Video assembly (FFmpeg, 9:16 vertical, TikTok-style captions)
- [x] Thumbnail generation (frame extraction + text overlay)
- [x] Queue-based batch processing (process_queue.py)
- [x] Per-topic output folders with dedup naming
- [x] Auto-approve mode for fully unattended runs

---

## Priority 1: Immediate Improvements (This Week)

### 1.1 Voice Quality Upgrade
**Priority:** 🔴 Critical | **Effort:** 2-3 hours

The voice is the #1 thing viewers notice. Piper is functional but robotic.

**Action items:**
- [ ] Finalize StyleTTS2 Colab notebook as the default production voice
- [ ] Create a 10-second "channel narrator voice" reference clip (record yourself or find a royalty-free deep narrator voice)
- [ ] Test voice consistency across 10+ scripts — the narrator should sound the same every time
- [ ] Add a `--voice-preset` flag to the orchestrator to switch between voice profiles
- [ ] Consider trying [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI) — newer, MIT licensed, runs well on free Colab T4, excellent quality

### 1.2 Thumbnail Quality
**Priority:** 🟡 High | **Effort:** 3-4 hours

Current thumbnails are frame extracts with text. YouTube Shorts thumbnails need to pop.

**Action items:**
- [ ] Add emoji overlays to thumbnails (⚡🔥😱) — proven to increase CTR
- [ ] Use larger, bolder font (Impact or custom font) with gradient color instead of plain white
- [ ] Add a character face/reaction crop in one corner (detect and crop character faces from clips)
- [ ] Consider generating thumbnails with AI image gen (SDXL on Colab) instead of frame extraction
- [ ] Add a "question mark" or "arrow" graphic element for "Why did..." topics
- [ ] Test different thumbnail styles and track which get more clicks

### 1.3 Script Quality Refinement
**Priority:** 🔴 Critical | **Effort:** Ongoing

**Action items:**
- [ ] Generate 20+ scripts, read them aloud, identify weak patterns
- [ ] Add a "script scorer" step that uses the LLM to rate scripts 1-10 on: hook strength, insight depth, naturalness, engagement
- [ ] Auto-reject scripts scoring below 7 and regenerate
- [ ] Add more variety to script structures (don't always start with a question)
- [ ] Create separate prompt templates for different topic types:
  - "Why did X do Y?" → motivation explainer
  - "What if X happened?" → hypothetical
  - "Most people missed this..." → hidden detail
  - "The real reason behind..." → reveal

### 1.4 Background Music
**Priority:** 🟡 High | **Effort:** 1 hour

**Action items:**
- [ ] Download 5-10 royalty-free ambient/dramatic tracks from YouTube Audio Library
- [ ] Place them in `assets/bgm/`
- [ ] The pipeline already supports BGM mixing — just needs tracks
- [ ] Consider auto-selecting BGM based on topic mood (dramatic, mysterious, funny)

---

## Priority 2: Growth & Optimization (Weeks 2-4)

### 2.1 Cross-Posting to TikTok & Instagram Reels
**Priority:** 🔴 Critical | **Effort:** 4-6 hours

Same 9:16 video works on all three platforms. Triple your reach for zero extra production cost.

**Action items:**
- [ ] Create TikTok and Instagram accounts with matching branding
- [ ] Build a `cross_poster.py` script that:
  - Uploads to TikTok via unofficial API (or manual upload workflow)
  - Uploads to Instagram Reels via unofficial API (or manual upload workflow)
  - Adjusts hashtags per platform
- [ ] For now, manual upload is fine — just copy video from the topic folder
- [ ] Platform-specific tweaks:
  - TikTok: Add trending sounds / hashtags
  - Reels: Different hashtag strategy (#explore #reels)

### 2.2 Analytics & Performance Tracking
**Priority:** 🟡 High | **Effort:** 3-4 hours

You need data to know what works.

**Action items:**
- [ ] Build an `analytics_tracker.py` that:
  - Uses YouTube Data API to fetch view counts, watch time, CTR for each video
  - Stores results in `analytics/performance.json`
  - Generates a weekly summary report
- [ ] Track which topic types perform best
- [ ] Track which thumbnail styles get higher CTR
- [ ] Feed top-performing topic patterns back into the topic miner prompt

### 2.3 Caption Style Improvements
**Priority:** 🟡 High | **Effort:** 2-3 hours

TikTok-style captions are a major retention driver.

**Action items:**
- [ ] Implement word-by-word color highlight (active word in gold, rest in white) — already configured in pipeline_config but may not be fully implemented in assembler
- [ ] Add text animation effects: fade-in, scale-up on key words
- [ ] Use a custom font (download a bold, modern font like Montserrat Black or Bebas Neue)
- [ ] Increase caption font size for mobile readability
- [ ] Add "emphasis words" detection — key words get a different style (color, size, underline)

### 2.4 Smart Clip Selection
**Priority:** 🟢 Medium | **Effort:** 4-6 hours

Current keyword matching is basic. Better clip selection = more engaging videos.

**Action items:**
- [ ] Implement LLM-assisted clip matching (send segment text + clip descriptions to Ollama)
- [ ] Use CLIP embeddings to semantically match narration segments to video frames
- [ ] Build a "clip freshness" tracker — avoid using the same clip too frequently across videos
- [ ] Auto-detect and avoid clips with burned-in subtitles or watermarks
- [ ] Add scene transition effects between clips (crossfade, zoom transition)

### 2.5 Video Intro/Outro
**Priority:** 🟢 Medium | **Effort:** 2-3 hours

**Action items:**
- [ ] Create a 2-3 second channel intro (logo reveal, channel name)
- [ ] Create a 2-3 second outro (subscribe CTA, next video tease)
- [ ] Auto-prepend/append to every video in the assembler
- [ ] Keep it short — Shorts viewers skip anything slow

---

## Priority 3: Scaling & Expansion (Month 2-3)

### 3.1 Multi-Show Expansion
**Priority:** 🟡 High | **Effort:** 4-6 hours

Architecture already supports this via `show_config.yaml`.

**Action items:**
- [ ] Pick your second show (high fan engagement, lots of clip material)
- [ ] Scrape wiki + theories for the new show
- [ ] Index clips into `clip_index.json` with show-specific tags
- [ ] Add show entry to `show_config.yaml`
- [ ] Run the pipeline with `--show new_show_slug`
- [ ] Decision: Same channel or separate channel?
  - Same channel: Broader reach, risk of diluting niche
  - Separate: Stronger niche authority, more management

### 3.2 AI Image Generation Fallback
**Priority:** 🟢 Medium | **Effort:** 4-6 hours

You deferred this from v1 — add it back when ready.

**Action items:**
- [ ] Set up Stable Diffusion XL or Flux.1-schnell on Colab
- [ ] Generate images in the show's art style for unmatched segments
- [ ] Apply Ken Burns zoom/pan to make still images feel dynamic
- [ ] Use this for abstract concepts that have no good clip match

### 3.3 Auto-Topic Trending Detection
**Priority:** 🟢 Medium | **Effort:** 6-8 hours

**Action items:**
- [ ] Monitor Reddit (r/rickandmorty etc.) for trending discussion topics
- [ ] Monitor Twitter/X for show-related trending hashtags
- [ ] Feed trending topics into the topic miner as high-priority seeds
- [ ] Build a `trend_detector.py` that runs daily and injects trending topics into queue.json

### 3.4 YouTube Publisher Integration
**Priority:** 🟢 Medium | **Effort:** 3-4 hours

You're doing manual uploads now. Automate when ready.

**Action items:**
- [ ] Set up Google Cloud project + OAuth credentials
- [ ] Test `publisher.py` with a private upload
- [ ] Add scheduled publishing (upload as private, publish at optimal times)
- [ ] Auto-generate titles, descriptions, and tags from the script
- [ ] Toggle "Altered or synthetic content" flag

---

## Priority 4: Advanced Features (Month 3+)

### 4.1 A/B Testing Framework
**Priority:** 🟢 Medium | **Effort:** 8-10 hours

**Action items:**
- [ ] Generate 2-3 versions of each script (different hooks, different angles)
- [ ] Generate 2-3 thumbnail variants per video
- [ ] Upload as unlisted, track CTR difference
- [ ] Auto-select winning variant after 48 hours of data
- [ ] Feed learnings back into prompt templates

### 4.2 Long-Form Content Pipeline
**Priority:** 🟢 Medium | **Effort:** Full project

Shorts build subscribers. Long-form videos (10-20 min) generate real ad revenue.

**Action items:**
- [ ] Create a "compilation" mode: combine 5-8 Shorts into a longer themed video
- [ ] Add chapter markers and structured sections
- [ ] Different pacing — long-form allows for deeper analysis
- [ ] Higher TTS quality bar for long-form
- [ ] Custom long-form prompt template

### 4.3 Community Engagement Automation
**Priority:** 🟢 Low | **Effort:** 4-6 hours

**Action items:**
- [ ] Auto-generate pinned comments on each video (engagement question)
- [ ] Monitor and respond to top comments using LLM
- [ ] Create community posts between videos to maintain engagement
- [ ] Run polls asking viewers what topics they want covered

### 4.4 Voice Consistency & Character Voices
**Priority:** 🟢 Low | **Effort:** 6-8 hours

**Action items:**
- [ ] Train/fine-tune a consistent narrator voice using RVC or StyleTTS2
- [ ] Optionally add character voice impressions for quotes (Rick voice, Morty voice)
- [ ] Use different voice presets for different topic moods

### 4.5 SEO Optimization Engine
**Priority:** 🟢 Low | **Effort:** 4-6 hours

**Action items:**
- [ ] Research top-performing keywords for your niche
- [ ] Auto-optimize video titles for SEO (keyword placement, emotional triggers)
- [ ] Generate descriptions with relevant keywords, timestamps, and links
- [ ] A/B test hashtag combinations

---

## Infrastructure & Code Quality

### Codebase Improvements
- [ ] Add `pyproject.toml` or `setup.py` for proper package management
- [ ] Add unit tests for critical functions (clip matching, script parsing, folder creation)
- [ ] Add a `--verbose` / `--quiet` flag to the orchestrator
- [ ] Create a `config_validator.py` that checks all paths and dependencies before running
- [ ] Add timing/profiling to identify bottleneck phases
- [ ] Consider migrating from subprocess FFmpeg calls to a Python FFmpeg wrapper for better error handling

### Monitoring & Alerting
- [ ] Add Discord/Telegram webhook notifications when:
  - A video finishes rendering
  - A pipeline run fails
  - The queue is empty
- [ ] Daily summary report of videos produced, queue status, storage usage

### Storage Management
- [ ] Auto-cleanup old intermediate files (captions, manifests) after video is published
- [ ] Compress/archive completed topic folders after 30 days
- [ ] Track total storage usage and warn when running low

---

## Quick Reference: Daily Workflow

```
# 1. Mine new topics (if queue is low)
python scripts/orchestrator_noImage_gpuVoice.py --phase topic_mine --count 15

# 2. Process entire queue automatically
python scripts/process_queue.py

# 3. Check output folders
ls output/

# Each topic folder contains:
# output/why_did_rick_destroy_the_citadel/
#   ├── script.txt          ← clean narration text
#   ├── script_raw.txt      ← original LLM output
#   ├── video.mp4           ← final assembled video
#   ├── thumbnail.jpg       ← generated thumbnail
#   └── pipeline_state.json ← full run metadata

# 4. Manually upload to YouTube, TikTok, Instagram
# (or use publisher.py when ready)
```

---

## Monetization Milestones Tracker

| Milestone | Target | Status |
|---|---|---|
| First video uploaded | 1 | ⬜ |
| 7 consecutive days of posting | 7 | ⬜ |
| 30 videos published | 30 | ⬜ |
| 100 subscribers | 100 | ⬜ |
| 1,000 subscribers (YPP threshold) | 1,000 | ⬜ |
| First viral Short (100K+ views) | 1 | ⬜ |
| 10M Shorts views (monetization) | 10,000,000 | ⬜ |
| First dollar earned | $1 | ⬜ |
| Second show added | 1 | ⬜ |
| First long-form video | 1 | ⬜ |
