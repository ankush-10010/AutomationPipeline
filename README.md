<div align="center">

# 🤖 Autonomous AI Content Automation Pipeline

**An enterprise-grade, 8-phase neural content production engine featuring self-correcting RAG verification, multi-modal computer vision indexing, and fault-tolerant state orchestration.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge)](https://www.trychroma.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Subprocess_Rendering-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![YouTube API](https://img.shields.io/badge/YouTube_API-v3_OAuth-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://developers.google.com/youtube/v3)

---

</div>

## 📌 Executive Summary (Recruiter TL;DR)

This repository houses a **compound AI engineering system** designed to solve the reliability, grounding, and workflow bottlenecks of generative video production. While standard AI demos rely on fragile, single-shot `model.generate()` scripts, this engine decouples generation into an **idempotent, 8-phase state machine** capable of taking raw thematic concepts and autonomously deploying fully edited, fact-checked, caption-burned 1080p video assets to YouTube.

### 🌟 Core Engineering Differentiators

| Dimension | Standard AI Demos | This Pipeline |
| :--- | :--- | :--- |
| **Execution Architecture** | Fragile monolithic scripts | **Decoupled 8-Phase State Machine** with JSON persistence |
| **Factuality & Grounding** | Blind generation (High hallucination) | **Closed-loop Verifier LLM** cross-checking RAG vs. live web dossiers |
| **Visual Asset Retrieval** | Basic filename/keyword regex | **Multi-Modal CV Fusion** (Fine-tuned YOLOv8 + LLaVA Vision + NLP Embeddings) |
| **Compute Strategy** | Locked to local hardware or 100% API | **Hybrid Local/Cloud Offloading** (Local pipeline + Colab GPU XTTS synthesis) |
| **Disaster Recovery** | Crashes require complete restart | **Sub-second Resume-on-Interrupt** from exact failure checkpoint |

---

## 🏗️ System Architecture & Workflow

The pipeline runs sequentially across 8 distinct architectural boundaries. If interrupted by rate limits, hardware blips, or manual review gates, execution hydrates cleanly from `pipeline_state.json`.

```mermaid
graph TD
    classDef llm fill:#f9f,stroke:#333,stroke-width:2px;
    classDef ml fill:#bbf,stroke:#333,stroke-width:2px;
    classDef media fill:#bfb,stroke:#333,stroke-width:2px;
    classDef api fill:#fbb,stroke:#333,stroke-width:2px;

    A([1. topic_mine]) -->|Ollama LLM Queue| B([2. script_gen])
    B -->|RAG + Fact Dossier| C{Script Review Gate}
    C -->|Approved| D([3. tts])
    C -->|Rejected| X([Abort / Auto-Heal])
    D -->|Local Piper / Cloud XTTS| E([4. caption])
    E -->|Faster-Whisper Word-Timestamps| F([5. match])
    F -->|YOLOv8 + VLM + Embeddings| G([6. assemble])
    G -->|FFmpeg Ken Burns + Subtitles| H([7. thumbnail])
    H -->|Frame Extraction + Overlay| I([8. publish])
    I -->|YouTube Data API v3| J((Published asset))

    class A,B llm;
    class D,E,F ml;
    class G,H media;
    class I api;
```

---

## 🧠 Core Innovation: The Verifier-Corrector Loop

To prevent large language models from fabricating plot points or misattributing canonical lore, **Phase 1b (`script_gen`)** implements an autonomous agentic fact-checking loop before any media rendering begins.

```mermaid
flowchart TD
    T[Approved Topic] --> W[Web Research Agent <br/> LLM 2: DuckDuckGo Search]
    W --> RD[(Research Fact Dossier <br/> Ground Truth Facts)]
    
    T --> SG[Script Generator <br/> LLM 1: Ollama + ChromaDB RAG]
    SG --> DRAFT[Draft Script]
    
    RD --> V[Verifier LLM <br/> LLM 3: Strict Lore Auditor]
    DRAFT --> V
    EP[(Episode Anchor DB <br/> Canonical Summaries)] --> V

    V -->|Factual Score >= 8/10| PASS([✅ PASS: Save Script])
    V -->|Hallucinations Detected| CORR[Build Targeted Correction Prompt]
    CORR -->|Max 2 Iterations| SG
```

1. **Web Grounding**: A research subagent pulls live discussions, wiki updates, and community consensus into a structured `Research Dossier`.
2. **Canonical Anchoring**: Queries `ChromaDB` vector stores containing 200+ canonical episode transcripts.
3. **Lore Auditing**: An independent Verifier LLM audits every factual claim in the generated script against both the external dossier and local database. Any discrepancy triggers a targeted correction prompt.

---

## 👁️ Multi-Modal Visual Indexing Engine

Matching narration text to video B-roll at scale requires understanding video clips across multiple semantic layers. **Phase 5 (`match`)** utilizes a hybrid retrieval strategy:

* **Object & Character Level (`YOLO_finetuning.py`)**: Custom fine-tuned **YOLOv8** models detect specific character bounding boxes and screen presence across thousands of raw video frames.
* **Semantic & Action Level (`clip_indexer_vision.py`)**: Local Vision-Language Models (**LLaVA** via Ollama) analyze middle-frame extractions to index scene lighting, character actions, and physical environments.
* **Dialogue Level (`episode_indexer.py`)**: Subtitle files are parsed into 384-dimensional sentence embeddings to anchor clips to canonical episode plotlines.

When drafting the assembly manifest, the decision engine calculates a composite similarity score to pair narration chunks with the mathematically optimal clip.

---

## 🛡️ Fault-Tolerant State Machine

Generative video pipelines are inherently volatile: API sockets drop, local GPUs overheat, and user interruptions occur. 

Instead of wrapping code in generic `try/except` blocks, the orchestrator maintains a persistent ledger (`pipeline_state.json`). 

```json
{
  "run_id": "20260626_113000",
  "status": "paused_at_tts",
  "last_completed_phase": "script_gen",
  "phase_outputs": {
    "script_path": "output/why_rick_hates_time_travel/script.txt",
    "topic_folder": "output/why_rick_hates_time_travel"
  }
}
```

* **Zero Work Loss**: Running `python scripts/orchestrator.py --resume` reads the ledger and jumps execution directly to $Phase_{N+1}$.
* **Cloud Offloading**: If local hardware lacks the VRAM for high-end voice synthesis, the state machine cleanly pauses execution, prompts the operator to execute `notebooks/orchestrator_noImage_gpuVoice.ipynb` on cloud GPUs (Google Colab), ingests the resulting `.wav` artifacts, and resumes local assembly.

---

## 📂 Repository Structure

```text
├── 📁 config/                 # YAML configuration definitions (Models, API endpoints, thresholds)
├── 📁 notebooks/              # GPU Colab notebooks for cloud-offloaded XTTS voice synthesis
├── 📁 prompts/                # System prompts for Topic Miner, Script Verifier, and RAG agents
├── 📁 scripts/                # Core modular execution engine
│   ├── orchestrator.py        # Master 8-phase pipeline controller & state ledger manager
│   ├── topic_miner.py         # Phase 1a: Autonomous topic ideation queue manager
│   ├── script_generator.py    # Phase 1b: RAG-augmented script drafting engine
│   ├── web_researcher.py      # Fact dossier compiler via search APIs
│   ├── script_verifier.py     # Closed-loop fact-checking auditor loop
│   ├── tts_local.py           # Phase 2: Local neural voice synthesis (Piper TTS)
│   ├── captioner.py           # Phase 3: Faster-Whisper word-level timestamp extraction
│   ├── clip_matcher.py        # Phase 4: Multi-modal visual assembly manifest builder
│   ├── assembler.py           # Phase 5: Subprocess FFmpeg hardware video compositor
│   ├── thumbnail_generator.py # Phase 6: Computer vision frame ranker & thumbnail renderer
│   ├── publisher.py           # Phase 7: YouTube Data API v3 OAuth upload controller
│   ├── YOLO_finetuning.py     # Custom YOLOv8 training pipeline for character detection
│   └── clip_indexer_vision.py # LLaVA local VLM automated scene tagger
├── 📁 vector_db/              # Persistent ChromaDB vector collections
└── README.md                  # System documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Installation

```bash
# Clone repository
git clone https://github.com/ankush-10010/AutomationPipeline.git
cd AutomationPipeline

# Install Python dependencies
pip install -r requirements.txt

# Verify local hardware dependencies
ffmpeg -version
ollama list
```

### 2. Pipeline Execution Modes

```bash
# Run complete autonomous production pipeline from a custom concept
python scripts/orchestrator.py --topic "Why Rick's Portal Gun Changes Everything"

# Execute autonomous batch mining & run pipeline on top queued item
python scripts/orchestrator.py --phase topic_mine --count 5
python scripts/orchestrator.py --phase all --auto-approve

# Dry-run system architecture (Calculates manifest & audit trail without rendering)
python scripts/orchestrator.py --topic "Evil Morty's Grand Plan" --dry-run

# Recover from an unexpected hardware shutdown or API rate limit
python scripts/orchestrator.py --resume
```

---

<div align="center">

*Designed & Architected for High-Reliability Generative Media Workflows.*

</div>
