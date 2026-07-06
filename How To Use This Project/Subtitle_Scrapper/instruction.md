<div align="center">

# 🎬 Subtitle Acquisition Guide

### *Automatically download or generate subtitles for your entire video library*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Whisper](https://img.shields.io/badge/Whisper-GPU_Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/SYSTRAN/faster-whisper)
[![Subliminal](https://img.shields.io/badge/Subliminal-Online_Providers-FF6F61?style=for-the-badge)](https://github.com/Diaoul/subliminal)

---

*Two powerful tools. Zero manual searching. Every episode covered.*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Tool 1 — Web Scraper](#-tool-1--web-scraper-subtitles_scrapperpy)
- [Tool 2 — Smart Subtitle Manager](#-tool-2--smart-subtitle-manager-subtitle_managerpy)
- [Video Naming Convention](#-critical-video-naming-convention)
- [Path Configuration](#-path-configuration)
- [Quick Start](#-quick-start)
- [Troubleshooting](#-troubleshooting)

---

## 🔭 Overview

This project provides **two complementary tools** to ensure every video file in your library has accurate English subtitles — without you ever needing to search for them manually.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBTITLE ACQUISITION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   📁 Your Video Files                                               │
│        │                                                            │
│        ├──▶ Tool 1: Web Scraper (Subtitles_Scrapper.py)             │
│        │       Scrapes subtitle websites for exact episode matches  │
│        │       Downloads .srt files automatically                   │
│        │                                                            │
│        └──▶ Tool 2: Subtitle Manager (subtitle_manager.py)         │
│                Step 1: Search online providers (OpenSubtitles, etc) │
│                Step 2: If not found → GPU Whisper generates them    │
│                                                                     │
│   📂 Output: ./ben10_subtitles/                                     │
│        ├── Ben_10_Classic_S01E01.srt                                │
│        ├── Ben_10_Classic_S01E02.srt                                │
│        ├── ...                                                      │
│        └── generation_report.txt  ← tells you which were generated │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **Which tool should I use?** Start with **Tool 1** if a subtitle website covers your show. Use **Tool 2** as an all-in-one fallback — it searches online first, then automatically generates subtitles using AI if nothing is found.

---

## 🕷️ Tool 1 — Web Scraper (`Subtitles_Scrapper.py`)

**Location:** `How To Use This Project/Subtitle_Scrapper/Subtitles_Scrapper.py`

This script scrapes subtitle hosting websites (like [my-subs.co](https://my-subs.co)) and downloads `.srt` subtitle files for every episode it finds.

### ⚙️ Configuration

Open `Subtitles_Scrapper.py` and modify these variables at the top of the file:

```python
# ── CONFIGURE THESE ──────────────────────────────────────────────

BASE_URL  = "https://my-subs.co"                                  # Subtitle website
SHOW_URL  = f"{BASE_URL}/showlistsubtitles-2075-rick-and-morty"   # ⚠️ Change to your show's page
SAVE_DIR  = "rick_and_morty_subtitles"                            # ⚠️ Change to your output folder
LANGUAGE  = "English"                                              # Language to download
```

| Variable | What to Change | Example |
|:---------|:---------------|:--------|
| `SHOW_URL` | Replace with the URL of your show's subtitle listing page | `f"{BASE_URL}/showlistsubtitles-XXXX-your-show-name"` |
| `SAVE_DIR` | Set the folder name where downloaded `.srt` files will be saved | `"ben10_subtitles"` |
| `LANGUAGE` | Set to your target language | `"English"` |

### 📦 Dependencies

```bash
pip install requests beautifulsoup4
```

### ▶️ Run

```bash
python Subtitles_Scrapper.py
```

> [!NOTE]
> The scraper adds a **2-second delay** between requests to avoid getting IP-banned by the subtitle website. This is intentional — be patient and let it run.

---

## 🧠 Tool 2 — Smart Subtitle Manager (`subtitle_manager.py`)

**Location:** `scripts/subtitle_manager.py`

This is the **recommended all-in-one tool**. For each video file, it follows a two-step cascade:

```
For each video file:
   │
   ├─ Step 1: Search online subtitle providers (OpenSubtitles, TVSubtitles, Addic7ed...)
   │          Found? ──▶ ✅ Download & save the .srt file
   │
   └─ Step 2: Not found online?
              ──▶ ⚡ Generate subtitles from scratch using Whisper AI on your GPU
                     (Uses faster-whisper with large-v3 model for maximum accuracy)
```

### 📦 Dependencies

```bash
pip install faster-whisper subliminal babelfish
```

> [!IMPORTANT]
> **GPU Required for Whisper fallback.** The Whisper `large-v3` model runs on your GPU (CUDA). An NVIDIA GPU with at least **4 GB VRAM** is recommended. If you only have a CPU, use `--model tiny` or `--model base` instead.

### ⚙️ Configuration (Command-Line Arguments)

| Argument | Default | Description |
|:---------|:--------|:------------|
| `directory` | *(required)* | Path to the folder containing your video files |
| `--outdir` | `ben10_subtitles` | Path where `.srt` files will be saved |
| `--model` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |

### ▶️ Run

```bash
# Basic usage — point it at your videos folder
python scripts/subtitle_manager.py ./clips/ben10 --outdir ./ben10_subtitles

# With a smaller Whisper model (for lower VRAM GPUs)
python scripts/subtitle_manager.py ./clips/ben10 --outdir ./ben10_subtitles --model medium
```

### 📊 Generation Report

After running, check `ben10_subtitles/generation_report.txt` to see which subtitles were downloaded vs. generated:

```
DOWNLOADED : Ben_10_Classic_S01E01.mp4     ← Found online
DOWNLOADED : Ben_10_Classic_S01E02.mp4     ← Found online
WHISPER    : Ben_10_Classic_S01E03.mp4     ← Generated by AI (no online match)
WHISPER    : Ben_10_Classic_S01E04.mp4     ← Generated by AI (no online match)
```

> [!TIP]
> The script **automatically skips** videos that already have a matching `.srt` file in the output directory. You can safely re-run it after adding new videos without re-processing old ones.

---

## 🏷️ CRITICAL: Video Naming Convention

> [!CAUTION]
> **Your video filenames directly determine whether the pipeline can find subtitles and match clips correctly.** Incorrect naming will cause subtitle downloads to fail and clip matching to break. Follow this format exactly.

### ✅ Required Format

```
{Show_Name}_S{season:02d}E{episode:02d}.{ext}
```

| Component | Format | Example |
|:----------|:-------|:--------|
| Show Name | Words separated by underscores | `Ben_10_Classic` |
| Season | `S` + 2-digit zero-padded number | `S01`, `S02`, `S04` |
| Episode | `E` + 2-digit zero-padded number | `E01`, `E09`, `E13` |
| Extension | Standard video format | `.mp4`, `.mkv`, `.avi` |

### ✅ Correct Examples

```
Ben_10_Classic_S01E01.mp4
Ben_10_Classic_S01E02.mp4
Ben_10_Classic_S02E05.mkv
Ben_10_Classic_S04E10.mp4
Rick_And_Morty_S01E01.mp4
Rick_And_Morty_S03E07.mkv
```

### ❌ Wrong Examples — DO NOT USE

```
ben10 episode 1.mp4              ← No season/episode format
S1E1.mp4                         ← Missing show name, not zero-padded
Ben 10 - 1x01 - And Then.mp4    ← Spaces, wrong separator, extra text
episode_01.mp4                   ← No show name, no season number
Ben10_s1e1.mp4                   ← Lowercase s/e, not zero-padded
```

### 📂 Recommended Folder Structure

Organize your videos in season folders for clarity. The subtitle manager uses **recursive scanning** (`rglob`), so nested folders are fully supported:

```
clips/
└── ben10/
    ├── season1/
    │   ├── Ben_10_Classic_S01E01.mp4
    │   ├── Ben_10_Classic_S01E02.mp4
    │   └── ...
    ├── season2/
    │   ├── Ben_10_Classic_S02E01.mp4
    │   └── ...
    ├── season3/
    │   └── ...
    └── season4/
        └── ...
```

> [!NOTE]
> The output `.srt` files are **always saved flat** (not in season subfolders) in your output directory. The subtitle filename will mirror the video filename: `Ben_10_Classic_S01E01.srt`.

---

## 🛠️ Path Configuration

After downloading/generating subtitles, make sure the pipeline knows where to find them.

### Step 1: Verify `pipeline_config.yaml`

Open `config/pipeline_config.yaml` and confirm these paths match your setup:

```yaml
paths:
  clips_dir: "./clips"                    # ← Where your video files live
  subtitles_dir: "./ben10_subtitles"      # ← Where your .srt files are saved
  clip_index: "./clip_index.json"         # ← Auto-generated clip database
```

### Step 2: Verify `show_config.yaml`

Open `config/show_config.yaml` and confirm the clips directory:

```yaml
shows:
  ben10:
    clips_dir: "./clips/ben10"            # ← Must point to your video files
```

### Step 3: Verify Subtitle Filenames Match Video Filenames

The pipeline expects each `.srt` file to share the **exact base name** as its corresponding video:

```
Video:    clips/ben10/season1/Ben_10_Classic_S01E01.mp4
Subtitle: ben10_subtitles/Ben_10_Classic_S01E01.srt
                          ^^^^^^^^^^^^^^^^^^^^^^
                          Must match exactly!
```

> [!WARNING]
> If you rename your video files after generating subtitles, you must also rename the corresponding `.srt` files to match. A mismatch will cause the scene context enrichment step to silently skip those episodes.

---

## 🚀 Quick Start

### Option A: Full automated pipeline (recommended)

```bash
# 1. Install dependencies
pip install requests beautifulsoup4 faster-whisper subliminal babelfish

# 2. Run the Smart Subtitle Manager on your videos folder
python scripts/subtitle_manager.py ./clips/ben10 --outdir ./ben10_subtitles

# 3. Verify output
ls ./ben10_subtitles/
cat ./ben10_subtitles/generation_report.txt

# 4. Done! Continue with the main pipeline.
```

### Option B: Web scraper first, then fill gaps

```bash
# 1. Configure and run the web scraper
#    (Edit SHOW_URL and SAVE_DIR in the script first!)
python "How To Use This Project/Subtitle_Scrapper/Subtitles_Scrapper.py"

# 2. Run Subtitle Manager on remaining gaps
#    (It will skip videos that already have .srt files)
python scripts/subtitle_manager.py ./clips/ben10 --outdir ./ben10_subtitles
```

---

## 🔧 Troubleshooting

<details>
<summary><b>❓ "No online subtitles found" for every episode</b></summary>

This usually means:
- Your video filenames don't follow the naming convention (the `subliminal` library parses the filename to identify the show/season/episode)
- The show is too obscure for online providers
- **Solution:** The Whisper fallback will automatically kick in and generate subtitles from the audio track. No action needed.

</details>

<details>
<summary><b>❓ Whisper runs out of GPU memory (CUDA OOM)</b></summary>

The `large-v3` model needs ~4-5 GB of VRAM. If you're running out:
```bash
# Use a smaller model
python scripts/subtitle_manager.py ./clips/ben10 --model medium

# Model sizes: tiny (1GB) < base (1GB) < small (2GB) < medium (5GB) < large-v3 (6GB)
```

</details>

<details>
<summary><b>❓ Web scraper gets blocked (403 / connection errors)</b></summary>

- The subtitle website may have rate-limited your IP
- Wait 15–30 minutes and try again
- The 2-second delay between requests is already built in to minimize this

</details>

<details>
<summary><b>❓ Subtitles exist but pipeline ignores them</b></summary>

Check that:
1. `subtitles_dir` in `config/pipeline_config.yaml` points to the correct folder
2. The `.srt` filename matches the video filename exactly (case-sensitive)
3. The `.srt` file is not empty (open it and verify it has content)

</details>

---

<div align="center">

**Built with ❤️ for the Automation Pipeline**

</div>
