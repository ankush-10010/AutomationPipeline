# AI Explainer - Implementation Checklist

## Project Setup

- [x] Directory structure (`audio/`, `captions/`, `clips/`, `config/`, `images/`, `notebooks/`, `output/`, `prompts/`, `scripts/`, `thumbnails/`, `topics/`)
- [x] `requirements.txt`
- [x] Git repo initialized
- [x] `topics/queue.json` (empty, ready)
- [x] `topics/approved/` directory
- [x] `topics/completed/` directory
- [x] `clip_index.json` (empty with schema example)

## Configuration

- [x] `config/pipeline_config.yaml` - paths, LLM, TTS, captioning, video, thumbnail, publishing, clip matching settings
- [x] `config/show_config.yaml` - Rick and Morty show config (characters, locations, themes, narrator style)
- [ ] `config/youtube_credentials.json` - YouTube OAuth credentials (manual setup required)
- [ ] `config/youtube_token.json` - Generated after first OAuth flow

## Prompt Templates

- [x] `prompts/topic_prompt.txt` - LLM prompt for topic mining
- [x] `prompts/script_prompt.txt` - LLM prompt for narration script generation
- [x] `prompts/thumbnail_prompt.txt` - Image generation prompt template

## Pipeline Scripts

- [x] `scripts/__init__.py` - Package marker
- [x] `scripts/config_loader.py` - Shared config, path resolution, logging, JSON/text helpers
- [x] `scripts/topic_miner.py` - Phase 1a: Topic mining via Ollama LLM
- [x] `scripts/script_generator.py` - Phase 1b: Narration script generation via Ollama
- [x] `scripts/tts_local.py` - Phase 2: Local TTS using Piper (dev only)
- [x] `scripts/captioner.py` - Phase 3: Word-level audio captioning via faster-whisper
- [x] `scripts/clip_matcher.py` - Phase 4: Keyword/LLM clip matching + AI image fallback
- [x] `scripts/assembler.py` - Phase 5: FFmpeg video assembly with TikTok-style captions
- [x] `scripts/thumbnail_generator.py` - Phase 6: Pillow thumbnail generation with text overlay
- [x] `scripts/publisher.py` - Phase 7: YouTube API upload with LLM metadata generation
- [x] `scripts/orchestrator.py` - Phase 8: Full pipeline controller with state management
- [x] `scripts/clip_indexer.py` - Utility: Clip library indexing (interactive, CSV, auto-tag)

## Kaggle GPU Scripts

- [x] `notebooks/kaggle_gpu_batch.py` - XTTS-v2 TTS + SDXL image generation for Kaggle

## External Tools Setup

- [ ] Ollama installed and running (`ollama serve`)
- [ ] Ollama model pulled (`ollama pull llama3.1:8b`)
- [ ] FFmpeg + ffprobe on PATH
- [ ] Piper TTS installed (`pip install piper-tts`) (optional, dev only)
- [ ] Python dependencies installed (`pip install -r requirements.txt`)

## Content Prep

- [ ] Collect video clips from show (50-100 clips to start)
- [ ] Index clips using `clip_indexer.py` (interactive or auto-tag mode)
- [ ] Prepare episode data file (optional, for richer topic mining)
- [ ] Add background music tracks to `assets/bgm/` (optional)

## YouTube Setup

- [ ] Create YouTube channel
- [ ] Enable YouTube Data API v3 in Google Cloud Console
- [ ] Create OAuth 2.0 credentials (Desktop app) and download to `config/youtube_credentials.json`
- [ ] Run first OAuth flow to generate token

## First Run

- [ ] Mine topics: `python scripts/orchestrator.py --phase topic_mine --count 10`
- [ ] Generate first video: `python scripts/orchestrator.py --topic "Your Topic Here"`
- [ ] Review and approve script at checkpoint
- [ ] Verify final video in `output/`
- [ ] Upload to YouTube (or use `--auto-approve` for hands-off runs)

## Production Quality (Kaggle)

- [ ] Set up Kaggle notebook with `notebooks/kaggle_gpu_batch.py`
- [ ] Upload script text files as Kaggle dataset
- [ ] Generate XTTS-v2 audio (voice-cloned narration)
- [ ] Generate SDXL fallback images for unmatched segments
- [ ] Download outputs and resume pipeline: `python scripts/orchestrator.py --resume`

## Future Enhancements

- [ ] Add more shows to `show_config.yaml`
- [ ] Build a batch scheduler for daily uploads
- [ ] Analytics dashboard (YouTube API reporting)
- [ ] Auto-tagging pipeline for new clips
- [ ] A/B test thumbnail variations
- [ ] Multi-language support (Fish Speech TTS)
