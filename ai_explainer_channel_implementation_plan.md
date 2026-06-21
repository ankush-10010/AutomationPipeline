# Implementation plan: automated AI explainer video channel

A note on research before we start: I tried to pull specifics on the @NyleTrix channel directly — video titles, formats, view counts — but YouTube blocks automated scraping of channel pages, and the handle doesn't show up in any third-party indexing tools I have access to (it's likely too small to be tracked by stats sites). So this plan is built around the format you originally described (AI-narrated "why did X do Y" explainer videos over relevant clips), which is a well-established and well-documented genre regardless of that one channel. If you send me a specific video link or title later, I can adjust the prompt templates and pacing to match it more precisely.

## 1. What you're building

A pipeline that takes a topic ("why did Rick do X") and produces a finished video with zero per-video cash cost:

1. **Script generation** — a local LLM writes the narration in a consistent voice/format.
2. **Voice generation** — text-to-speech turns the script into narration audio.
3. **Captions & timing** — word-level timestamps get extracted from the audio.
4. **Clip matching** — script segments get matched to b-roll/clips from your library.
5. **Video assembly** — clips get trimmed, stitched, and synced to the narration; captions get burned in. (This is already in progress — see Section 6.)
6. **Publishing** — the finished video gets uploaded with title/description/tags via the YouTube API.

Everything runs free and local except step 2, which benefits from a GPU you don't have. Section 3 covers exactly how to borrow one for free.

## 2. Hardware reality check, given CPU-only

Two stages are GPU-hungry in theory: script generation (LLM) and voice generation (TTS). In practice:

- **Script generation works fine on CPU.** A quantized 7-8B model (Llama 3.1 8B, Mistral 7B, Qwen2.5 7B, all via Ollama) generates a ~600-word script in roughly 1-3 minutes on a modern CPU. That's a perfectly acceptable wait for something you run once per video.
- **Voice generation is the real bottleneck.** High-quality TTS (Coqui XTTS-v2, the model that sounds closest to ElevenLabs) is slow on CPU — minutes per sentence, not per video. This is the one stage worth borrowing a GPU for.
- **Captioning, clip matching, and video assembly are CPU-native.** faster-whisper (CPU-optimized Whisper) and ffmpeg are designed to run efficiently without a GPU.

So the practical split is: run everything locally except the TTS step, which you batch-render on a free cloud GPU session.

## 3. Free GPU resources

| Platform | Free GPU | Quota | Session length | Notes |
|---|---|---|---|---|
| Google Colab | T4 or P100 (16GB) | No published hard cap; commonly 15-25 hrs/week, varies by demand | Up to 12 hrs | Easiest to start with, but availability and disconnects are unpredictable under load |
| Kaggle Notebooks | T4 or P100 (16GB) | 30 hrs/week | Up to 9 hrs, supports background execution | Generally the most reliable free option — no waitlist, consistent access |
| Lightning AI Studio | T4 (free credits for new accounts) | Limited free credits, then pay-as-you-go | Persistent studio (no session resets) | Good if you want a persistent cloud IDE instead of a notebook that resets |
| AWS SageMaker Studio Lab | T4-class | 4 GPU hrs per 24 hrs | Up to 4 hrs/session | No credit card required; useful as a backup when Colab/Kaggle are busy |
| Google Cloud / Azure new-account credits | T4/A100-class | $300 (GCP) / $200 (Azure), one-time | N/A | Not recurring, but enough for 50-100+ hours of T4 time if you want to front-load testing |

**Recommended approach**: use Kaggle as your primary TTS-rendering environment (most reliable, 30 hrs/week is plenty for batch voice generation), and Colab as overflow when you need more in a given week. Develop and debug your TTS script locally first with a tiny test input (even on CPU, just to confirm there are no bugs), then upload the working script to a Kaggle notebook and run your actual batch jobs there.

## 4. Phase-by-phase build plan

### Phase 0 — Environment & repo setup
Set up a project folder with subfolders for scripts, audio output, clip library, and final renders. Install Python, ffmpeg (already confirmed available), and Ollama for local LLM serving. Time: under an hour.

### Phase 1 — Script generation (local LLM)
Install Ollama and pull a 7-8B model (`ollama pull llama3.1:8b` or `mistral`). Build a prompt template that enforces your format: hook line, setup, explanation beats, payoff — matching the "why did Rick do X" structure. Generate 5-10 test scripts and refine the prompt until the tone and length are consistent without manual editing. Output: a Python function `generate_script(topic) -> str`.

### Phase 2 — Voice generation (TTS, on free GPU)
Two paths depending on quality bar:
- **Piper** (CPU-friendly, runs locally, more robotic): good for fast iteration and testing the full pipeline before investing GPU time.
- **Coqui XTTS-v2** (near-ElevenLabs quality, needs GPU): run this on Kaggle/Colab. Upload your script text, generate the narration .wav, download it back.

Start with Piper locally to validate the whole pipeline end-to-end cheaply, then swap in XTTS on a free GPU session once everything else works.

### Phase 3 — Captioning (faster-whisper, CPU)
Run faster-whisper (`small` or `base` model, `int8` compute type for speed) on the narration audio to get word-level timestamps. This gives you both burnable captions and the exact timing data needed to sync clips to the narration in Phase 4.

### Phase 4 — Clip library & matching
Build a `clip_tags.json` mapping keywords (character names, locations, themes) to clip filenames in your library. For each caption segment, the assembly script scans the segment's text for keyword matches and picks a clip; unmatched segments round-robin through a generic b-roll pool so nothing ever fails to find a clip.

### Phase 5 — Video assembly (in progress)
This is the script we already started building — clip selection and trimming happen in Python (moviepy), and subtitle burn-in happens via ffmpeg's subtitle filter directly (faster and avoids the ImageMagick dependency MoviePy's text rendering needs). Next session, we finish wiring this to real Phase 2/3 outputs and do an end-to-end test render.

### Phase 6 — Publishing automation
Use the YouTube Data API (`google-api-python-client`) to upload the finished video along with title, description, and tags generated by the same local LLM from Phase 1. This requires a one-time OAuth setup in Google Cloud Console (free, just needs a Google account).

### Phase 7 — Orchestration
Wire Phases 1-6 into a single script (or a scheduled cron job / GitHub Actions workflow) that takes a topic as input and produces an uploaded video as output, with the GPU-dependent TTS step flagged as a manual "run this batch on Kaggle" checkpoint rather than fully automated, since free GPU sessions can't be triggered unattended.

## 5. Suggested timeline

| Week | Focus |
|---|---|
| 1 | Phase 0 + Phase 1 (script generation working and tuned) |
| 2 | Phase 2 with Piper locally, full pipeline validated end-to-end on dummy data |
| 3 | Phase 3 + Phase 4 (captions and clip matching working) |
| 4 | Phase 5 finished and tested with real audio + real clips |
| 5 | Swap in XTTS on free GPU for production-quality voice |
| 6 | Phase 6 + Phase 7 (publishing automated, first real videos go out) |

This assumes a few hours a week, not full-time work — compress it if you can dedicate more time.

## 6. Where we already are

We've started Phase 5: a project at `video_pipeline/` with `moviepy` and `faster-whisper` installed, ready to build the clip-matching + assembly + caption-burning script. Next step in our next session is finishing that script and running a synthetic end-to-end test (dummy clips + dummy audio) to validate the logic before plugging in real narration and real footage.

## 7. Platform risk, addressed honestly

Two separate things matter here, and they're easy to conflate:

**Copyright on the source clips.** Whether the show clips themselves are something you have rights to use depends on specifics I can't evaluate for you (what footage you're sourcing, how it's used, monetization status). This doesn't change based on your tooling being free — it's the same legal question whether you pay for APIs or not.

**YouTube's AI content policy.** YouTube requires creators to toggle "Altered or synthetic content" in YouTube Studio for content that could realistically mislead a viewer about something real — this mainly targets realistic AI voice clones of real people or deepfake-style footage, not stylized narration over existing show clips. Separately, YouTube's "inauthentic content" policy (renamed from "repetitious content" in 2025) specifically targets mass-produced, templated AI channels with no human creative layer — identical formats repeated dozens of times, narration with no editorial point of view, zero original commentary. Channels that survive scrutiny tend to have a consistent named persona, original commentary or analysis layered over the clips (not just plot summary), and visible format variation between videos. This is worth designing for from the start rather than retrofitting later, since it affects how you write your script prompt template in Phase 1.

## 8. Suggested next session

Pick up Phase 5: finish the clip-matching and assembly script, then run a synthetic test render so you have a working pipeline skeleton before plugging in real content. From there we move to Phase 1 (script generation) so you have an actual script to feed it.
