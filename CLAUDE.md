# AI Explainer — Project Memory

## What this project is
An 8-phase autonomous AI video content pipeline. It takes topics, generates scripts via RAG + LLM, selects clips via YOLOv8 + LLaVA vision, synthesizes voiceover (Colab GPU XTTS), burns captions, and uploads to YouTube. State is persisted in `pipeline_state.json` for fault-tolerant resume.

## Language & Runtime
- Python 3.10+ (primary language for the pipeline)
- Node.js / JavaScript for graph utilities (`add-layers.js`, `add-semantic-edges.js`, `run_pipeline.js`)
- Requirements: `requirements.txt`

## Key entry points
- `run_pipeline.js` — Node.js pipeline orchestrator
- `cmpress.py` — Python video/clip compression utility
- `pipeline_state.json` — JSON state machine (8 phases); edit carefully — this is the checkpoint file

## Important directories
- `scripts/` — Core Python pipeline phase scripts
- `clips/` — Video clip assets
- `audio/` — Synthesized voiceover files
- `captions/` — Subtitle/caption files
- `vector_db/` — ChromaDB persistent vector store
- `graphify-out/` — Knowledge graph output (from graphify skill)
- `prompts/` — Prompt templates used by the pipeline LLMs

## Do not
- Do not delete `pipeline_state.json` without asking — it is the resume checkpoint
- Do not commit API keys or tokens; check `.gitignore`
- Do not run GPU-heavy XTTS synthesis steps locally without checking available VRAM first

## Testing
- No automated test suite currently exists. Verification must be done manually by checking pipeline output files.

## Conventions
- Prefer editing existing pipeline scripts over creating new files
- When adding a new pipeline phase, mirror the state-machine pattern in `pipeline_state.json`
- Keep LLM prompts in `prompts/` not hard-coded in scripts
