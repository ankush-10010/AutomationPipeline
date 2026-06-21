"""
kaggle_gpu_batch.py — Standalone Kaggle GPU Batch Script
=========================================================
Copy-paste this into a Kaggle notebook (or upload as a .py file).

This script is STANDALONE — it does NOT import config_loader or any project modules.
All configuration is defined inline below.

What it does:
  1. Installs XTTS-v2 (Coqui TTS) and diffusers
  2. Generates high-quality .wav audio from .txt script files using XTTS-v2
  3. Generates fallback images using Stable Diffusion XL for segments without clips

How to use on Kaggle:
  1. Create a new Kaggle Notebook with GPU accelerator (T4 x2 or P100)
  2. Upload your .txt script files as a Kaggle dataset, or paste them inline
  3. Copy-paste each cell section (marked with # %%) into separate notebook cells
  4. Run cells in order
  5. Download outputs from the output directories

Cell structure:
  Cell 1: Install dependencies
  Cell 2: Configuration
  Cell 3: Load XTTS-v2 model
  Cell 4: Generate audio from text files
  Cell 5: Generate word-level captions (faster-whisper)
  Cell 6: Match segments to clips
  Cell 7: Load Stable Diffusion XL
  Cell 8: Generate AI images for unmatched segments
  Cell 9: Package outputs for download
"""

# %% Cell 1: Install Dependencies
# =============================================================================
# Run this cell first. It installs all required packages.
# Restart the kernel after this cell if prompted (Kaggle usually handles this).
# =============================================================================

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# TTS (Coqui XTTS-v2)
install("TTS>=0.22.0")

# Captioning
install("faster-whisper>=1.0.0")
install("PyYAML>=6.0.1")

# Image generation
install("diffusers>=0.25.0")
install("transformers>=4.36.0")
install("accelerate>=0.25.0")
install("safetensors>=0.4.0")
install("invisible_watermark>=0.2.0")

# Utilities
install("tqdm>=4.66.0")

print("✅ All dependencies installed!")


# %% Cell 2: Configuration
# =============================================================================
# Edit these paths and settings to match your Kaggle dataset layout.
# =============================================================================

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# On Kaggle, input datasets are mounted at /kaggle/input/<dataset-name>/
# Outputs go to /kaggle/working/
INPUT_DIR = Path("/kaggle/input/explainer-scripts")  # Your uploaded .txt files
AUDIO_OUTPUT_DIR = Path("/kaggle/working/audio")
CAPTIONS_OUTPUT_DIR = Path("/kaggle/working/captions")
IMAGES_OUTPUT_DIR = Path("/kaggle/working/images")
MANIFEST_OUTPUT_DIR = Path("/kaggle/working/output")

# Create output directories
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CAPTIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- XTTS-v2 Settings -----------------------------------------------------
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE = "en"
# Optional: path to a 10-second reference .wav for voice cloning
# Set to None to use the default XTTS voice
REFERENCE_AUDIO = None  # e.g., "/kaggle/input/reference-voice/speaker.wav"

# --- Stable Diffusion XL Settings ------------------------------------------
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
USE_REFINER = True  # Set to False to save VRAM (base-only is still good)
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920  # Vertical/Shorts format
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

print(f"📁 Input dir:  {INPUT_DIR}")
print(f"🔊 Audio dir:  {AUDIO_OUTPUT_DIR}")
print(f"📝 Captions dir: {CAPTIONS_OUTPUT_DIR}")
print(f"🖼️  Images dir: {IMAGES_OUTPUT_DIR}")
print(f"📄 Output dir: {MANIFEST_OUTPUT_DIR}")
print(f"✅ Configuration ready!")


# %% Cell 3: Load XTTS-v2 Model
# =============================================================================
# This loads the XTTS-v2 model onto the GPU. Takes 1-2 minutes on first run.
# =============================================================================

import torch
from TTS.api import TTS

print("🔄 Loading XTTS-v2 model...")
print(f"   GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU device:    {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory:    {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# Initialize TTS with XTTS-v2
tts = TTS(model_name=XTTS_MODEL, progress_bar=True)
tts.to("cuda" if torch.cuda.is_available() else "cpu")

print("✅ XTTS-v2 model loaded!")


# %% Cell 4: Generate Audio from Text Files
# =============================================================================
# Reads all .txt files from INPUT_DIR and generates .wav audio for each.
# =============================================================================

import time
from tqdm import tqdm

def generate_audio(text: str, output_path: Path, tts_model, reference_wav=None):
    """Generate a .wav file from text using XTTS-v2."""
    if reference_wav and Path(reference_wav).exists():
        # Voice cloning mode: use reference audio
        tts_model.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=reference_wav,
            language=XTTS_LANGUAGE,
        )
    else:
        # Default voice mode
        # XTTS-v2 requires a speaker_wav; if none provided, use a built-in speaker
        tts_model.tts_to_file(
            text=text,
            file_path=str(output_path),
            language=XTTS_LANGUAGE,
        )


# Discover text files
txt_files = sorted(INPUT_DIR.glob("*.txt"))
print(f"📝 Found {len(txt_files)} text file(s) in {INPUT_DIR}")

if not txt_files:
    print("⚠️  No .txt files found! Make sure your dataset is mounted correctly.")
    print(f"   Expected path: {INPUT_DIR}")
    print(f"   Contents of /kaggle/input/:")
    for p in Path("/kaggle/input").iterdir():
        print(f"     {p}")
else:
    audio_results = []
    total_start = time.time()

    for txt_file in tqdm(txt_files, desc="Generating audio"):
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  ⚠️ Skipping empty file: {txt_file.name}")
            continue

        wav_path = AUDIO_OUTPUT_DIR / f"{txt_file.stem}.wav"

        try:
            start = time.time()
            generate_audio(text, wav_path, tts, REFERENCE_AUDIO)
            elapsed = time.time() - start

            size_mb = wav_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {txt_file.name} → {wav_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
            audio_results.append({"file": txt_file.name, "output": str(wav_path), "size_mb": size_mb})

        except Exception as e:
            print(f"  ❌ Failed: {txt_file.name} — {e}")

    total_elapsed = time.time() - total_start
    print(f"\n🎉 Audio generation complete! {len(audio_results)}/{len(txt_files)} files in {total_elapsed:.0f}s")


# %% Cell 5: Generate Word-Level Captions
# =============================================================================
# Generate word-level captions from audio using faster-whisper.
# =============================================================================

import json
from faster_whisper import WhisperModel

print("🔄 Loading Faster Whisper model...")
whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
print("✅ Faster Whisper loaded!")

print(f"📝 Captioning audio files in {AUDIO_OUTPUT_DIR}...")
for wav_path in sorted(AUDIO_OUTPUT_DIR.glob("*.wav")):
    print(f"  📝 Processing {wav_path.name}...")
    segments, info = whisper_model.transcribe(str(wav_path), word_timestamps=True)
    
    caption_data = {"audio_file": str(wav_path), "segments": []}
    
    for i, segment in enumerate(segments):
        words = []
        for word in segment.words:
            words.append({
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "score": word.probability
            })
        
        caption_data["segments"].append({
            "id": i,
            "text": segment.text.strip(),
            "start": segment.start,
            "end": segment.end,
            "words": words
        })
    
    out_path = CAPTIONS_OUTPUT_DIR / f"{wav_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(caption_data, f, indent=2)
    print(f"  ✅ Saved captions to {out_path.name}")


# %% Cell 6: Match Segments to Clips
# =============================================================================
# Match caption segments to clips or generate AI image prompts.
# =============================================================================

import yaml
import re

CLIP_INDEX_PATH = INPUT_DIR / "clip_index.json"
SHOW_CONFIG_PATH = INPUT_DIR / "show_config.yaml"

clips = []
if CLIP_INDEX_PATH.exists():
    try:
        with open(CLIP_INDEX_PATH, "r") as f:
            data = json.load(f)
            clips = data.get("clips", []) if isinstance(data, dict) else data
        print(f"✅ Loaded {len(clips)} clips from {CLIP_INDEX_PATH.name}")
    except Exception as e:
        print(f"⚠️ Error loading clip index: {e}")

show_config = {}
if SHOW_CONFIG_PATH.exists():
    try:
        with open(SHOW_CONFIG_PATH, "r") as f:
            show_config = yaml.safe_load(f) or {}
        print(f"✅ Loaded show config from {SHOW_CONFIG_PATH.name}")
    except Exception as e:
        print(f"⚠️ Error loading show config: {e}")

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

def _normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower())

def extract_keywords(text):
    words = _normalize(text).split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

def score_clip_keyword(segment_text, clip, show_config):
    score = 0.0
    seg_keywords = extract_keywords(segment_text)
    
    seg_characters = set()
    for char in show_config.get("characters", []):
        for name in [char["name"]] + char.get("aliases", []):
            if name.lower() in segment_text.lower():
                seg_characters.add(name.lower())
    
    seg_locations = set()
    for loc in show_config.get("locations", []):
        if loc.lower() in segment_text.lower():
            seg_locations.add(loc.lower())
            
    clip_chars = {c.lower() for c in clip.get("characters", [])}
    clip_loc = clip.get("location", "").lower()
    clip_action = _normalize(clip.get("action", ""))
    clip_tags = {t.lower() for t in clip.get("tags", [])}
    
    char_overlap = seg_characters & clip_chars
    score += len(char_overlap) * 3.0
    
    if clip_loc and clip_loc in seg_locations:
        score += 2.0
        
    action_words = extract_keywords(clip_action)
    score += len(seg_keywords & action_words) * 2.0
    
    score += len(seg_keywords & clip_tags) * 1.0
    return score

def generate_ai_image_prompt(segment_text, show_config):
    show_name = show_config.get("display_name", "the show")
    clean = re.sub(r"[\"']", "", segment_text)
    if len(clean) > 120:
        clean = clean[:120] + "..."
    return f"Cinematic still from {show_name}, depicting: {clean}. Dramatic lighting, animation style, 9:16 vertical composition, high detail, vibrant colors."

for cap_path in sorted(CAPTIONS_OUTPUT_DIR.glob("*.json")):
    print(f"  🔍 Matching segments for {cap_path.name}...")
    with open(cap_path, "r") as f:
        cap_data = json.load(f)
        
    manifest = {"audio_file": cap_data.get("audio_file", ""), "segments": [], "stats": {"matched": 0, "fallback": 0, "total": 0}}
    
    for seg in cap_data.get("segments", []):
        manifest["stats"]["total"] += 1
        seg_text = seg.get("text", "")
        best_clip = None
        best_score = 0.0
        
        for clip in clips:
            s = score_clip_keyword(seg_text, clip, show_config)
            if s > best_score:
                best_score = s
                best_clip = clip
                
        entry = {
            "id": seg.get("id"),
            "text": seg_text,
            "start": seg.get("start"),
            "end": seg.get("end"),
            "words": seg.get("words", [])
        }
        
        if best_score >= 1 and best_clip:
            entry["visual_type"] = "clip"
            entry["visual_source"] = best_clip.get("filename", "")
            entry["clip_start"] = 0.0
            entry["match_score"] = round(best_score, 2)
            manifest["stats"]["matched"] += 1
        else:
            entry["visual_type"] = "ai_image"
            entry["visual_source"] = generate_ai_image_prompt(seg_text, show_config)
            entry["clip_start"] = 0.0
            entry["match_score"] = 0.0
            manifest["stats"]["fallback"] += 1
            
        manifest["segments"].append(entry)
        
    out_manifest = MANIFEST_OUTPUT_DIR / f"manifest_{cap_path.stem}.json"
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✅ Saved manifest to {out_manifest.name} ({manifest['stats']['matched']} matched, {manifest['stats']['fallback']} fallback)")


# %% Cell 7: Load Stable Diffusion XL (for Fallback Images)
# =============================================================================
# Loads SDXL for generating images when no matching clip is available.
# Skip this cell if you don't need fallback images.
# =============================================================================

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline

print("🔄 Loading Stable Diffusion XL base model...")

# Free TTS and Whisper GPU memory first
if "tts" in dir():
    del tts
    print("   Freed TTS model from GPU memory")
if "whisper_model" in dir():
    del whisper_model
    print("   Freed Whisper model from GPU memory")
torch.cuda.empty_cache()

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
sdxl_base = DiffusionPipeline.from_pretrained(
    SDXL_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True,
    variant="fp16" if device == "cuda" else None,
)
sdxl_base.to(device)
if device == "cuda":
    sdxl_base.enable_model_cpu_offload()  # Saves VRAM

# Optionally load refiner
sdxl_refiner = None
if USE_REFINER:
    print("🔄 Loading SDXL refiner...")
    sdxl_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_REFINER,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    sdxl_refiner.to(device)
    if device == "cuda":
        sdxl_refiner.enable_model_cpu_offload()

print("✅ SDXL models loaded!")


# %% Cell 8: Generate Fallback Images
# =============================================================================
# Reads manifest.json and generates images for segments marked as "ai_image".
# =============================================================================

import json
from tqdm import tqdm

# Negative prompt to improve quality
NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, logo, signature, cropped, out of frame"
)


def generate_image(prompt: str, output_path: Path, negative_prompt: str = NEGATIVE_PROMPT):
    """Generate an image using SDXL (with optional refiner pass)."""
    # Base generation
    if USE_REFINER and sdxl_refiner is not None:
        # Generate latents with base, refine with refiner
        image = sdxl_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            denoising_end=0.8,
            output_type="latent",
        ).images[0]

        image = sdxl_refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image[None, :],
            num_inference_steps=NUM_INFERENCE_STEPS,
            denoising_start=0.8,
        ).images[0]
    else:
        image = sdxl_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images[0]

    image.save(str(output_path), quality=95)
    return image


image_results = []
total_images_to_generate = 0

# First, count how many images we need
for manifest_path in MANIFEST_OUTPUT_DIR.glob("*.json"):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    for seg in manifest.get("segments", []):
        if seg.get("visual_type") == "ai_image":
            total_images_to_generate += 1

print(f"🖼️  Generating {total_images_to_generate} fallback images from manifests...")

if total_images_to_generate > 0:
    with tqdm(total=total_images_to_generate, desc="Generating images") as pbar:
        for manifest_path in MANIFEST_OUTPUT_DIR.glob("*.json"):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            for seg in manifest.get("segments", []):
                if seg.get("visual_type") == "ai_image":
                    prompt = seg.get("visual_source", "")
                    seg_id = seg.get("id", 0)
                    name = f"seg_{seg_id:04d}"
                    output_path = IMAGES_OUTPUT_DIR / f"{name}.png"
                    
                    if not output_path.exists():
                        try:
                            start = time.time()
                            generate_image(prompt, output_path)
                            elapsed = time.time() - start
                            
                            size_mb = output_path.stat().st_size / (1024 * 1024)
                            image_results.append({"name": name, "path": str(output_path), "prompt": prompt})
                        except Exception as e:
                            print(f"  ❌ Failed: {name} — {e}")
                    else:
                        image_results.append({"name": name, "path": str(output_path), "prompt": prompt})
                        
                    pbar.update(1)

# Save metadata for downstream pipeline
metadata_path = IMAGES_OUTPUT_DIR / "generation_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(image_results, f, indent=2)

print(f"\n🎉 Image generation complete! {len(image_results)} images ready")
print(f"   Metadata saved to {metadata_path}")


# %% Cell 9: Package Outputs for Download
# =============================================================================
# Creates a zip archive of all generated files for easy download from Kaggle.
# =============================================================================

import shutil

OUTPUT_ZIP = Path("/kaggle/working/explainer_outputs")

# List generated files
print("📦 Generated files:")
print(f"\n🔊 Audio ({AUDIO_OUTPUT_DIR}):")
for f in sorted(AUDIO_OUTPUT_DIR.glob("*")):
    size = f.stat().st_size / (1024 * 1024)
    print(f"   {f.name:30s} {size:.1f} MB")

print(f"\n📝 Captions ({CAPTIONS_OUTPUT_DIR}):")
for f in sorted(CAPTIONS_OUTPUT_DIR.glob("*")):
    size = f.stat().st_size / 1024
    print(f"   {f.name:30s} {size:.1f} KB")

print(f"\n🖼️  Images ({IMAGES_OUTPUT_DIR}):")
for f in sorted(IMAGES_OUTPUT_DIR.glob("*")):
    size = f.stat().st_size / (1024 * 1024)
    print(f"   {f.name:30s} {size:.1f} MB")

print(f"\n📄 Output ({MANIFEST_OUTPUT_DIR}):")
for f in sorted(MANIFEST_OUTPUT_DIR.glob("*")):
    size = f.stat().st_size / 1024
    print(f"   {f.name:30s} {size:.1f} KB")

# Create zip for download
print(f"\n📦 Creating zip archive...")

# Zip audio
shutil.make_archive(
    str(OUTPUT_ZIP),
    "zip",
    root_dir="/kaggle/working",
    base_dir=".",
)

zip_path = Path(str(OUTPUT_ZIP) + ".zip")
zip_size = zip_path.stat().st_size / (1024 * 1024)
print(f"✅ Archive created: {zip_path} ({zip_size:.1f} MB)")
print(f"\n💡 Download from the 'Output' tab on the right side of Kaggle, or use:")
print(f"   from IPython.display import FileLink")
print(f"   FileLink('{zip_path}')")
