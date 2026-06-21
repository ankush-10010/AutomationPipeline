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
  Cell 5: Load Stable Diffusion XL
  Cell 6: Generate fallback images
  Cell 7: Package outputs for download
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
IMAGES_OUTPUT_DIR = Path("/kaggle/working/images")

# Create output directories
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"🖼️  Images dir: {IMAGES_OUTPUT_DIR}")
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


# %% Cell 5: Load Stable Diffusion XL (for Fallback Images)
# =============================================================================
# Loads SDXL for generating images when no matching clip is available.
# Skip this cell if you don't need fallback images.
# =============================================================================

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline

print("🔄 Loading Stable Diffusion XL base model...")

# Free TTS GPU memory first (if TTS is no longer needed)
if "tts" in dir():
    del tts
    torch.cuda.empty_cache()
    print("   Freed TTS model from GPU memory")

# Load base model
sdxl_base = DiffusionPipeline.from_pretrained(
    SDXL_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
sdxl_base.to("cuda")
sdxl_base.enable_model_cpu_offload()  # Saves VRAM

# Optionally load refiner
sdxl_refiner = None
if USE_REFINER:
    print("🔄 Loading SDXL refiner...")
    sdxl_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_REFINER,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    sdxl_refiner.to("cuda")
    sdxl_refiner.enable_model_cpu_offload()

print("✅ SDXL models loaded!")


# %% Cell 6: Generate Fallback Images
# =============================================================================
# Define prompts and generate images. These are used as background visuals
# for video segments that don't have matching clips.
#
# Edit the IMAGE_PROMPTS list to match your video's content.
# =============================================================================

import json
from tqdm import tqdm

# Define image prompts — one per scene/segment that needs a fallback image
# Format: {"name": "filename_stem", "prompt": "description for SDXL"}
IMAGE_PROMPTS = [
    {
        "name": "intro_bg",
        "prompt": (
            "A cinematic sci-fi scene with glowing portals and cosmic energy, "
            "dark background with vibrant neon colors, digital art style, "
            "vertical composition 9:16 aspect ratio, highly detailed"
        ),
    },
    {
        "name": "explanation_bg",
        "prompt": (
            "Abstract visualization of neural networks and data flowing, "
            "dark purple and blue tones with golden highlights, "
            "futuristic technology concept art, vertical composition, 4K detailed"
        ),
    },
    {
        "name": "conclusion_bg",
        "prompt": (
            "Epic wide shot of a futuristic city at sunset with dramatic lighting, "
            "cyberpunk aesthetic, warm orange and cool blue contrast, "
            "vertical composition 9:16, cinematic concept art"
        ),
    },
]

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


print(f"🖼️  Generating {len(IMAGE_PROMPTS)} fallback images...")
image_results = []

for item in tqdm(IMAGE_PROMPTS, desc="Generating images"):
    name = item["name"]
    prompt = item["prompt"]
    output_path = IMAGES_OUTPUT_DIR / f"{name}.png"

    try:
        start = time.time()
        generate_image(prompt, output_path)
        elapsed = time.time() - start

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {name}.png ({size_mb:.1f} MB, {elapsed:.1f}s)")
        image_results.append({"name": name, "path": str(output_path), "prompt": prompt})

    except Exception as e:
        print(f"  ❌ Failed: {name} — {e}")

# Save metadata for downstream pipeline
metadata_path = IMAGES_OUTPUT_DIR / "generation_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(image_results, f, indent=2)

print(f"\n🎉 Image generation complete! {len(image_results)}/{len(IMAGE_PROMPTS)} images generated")
print(f"   Metadata saved to {metadata_path}")


# %% Cell 7: Package Outputs for Download
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

print(f"\n🖼️  Images ({IMAGES_OUTPUT_DIR}):")
for f in sorted(IMAGES_OUTPUT_DIR.glob("*")):
    size = f.stat().st_size / (1024 * 1024)
    print(f"   {f.name:30s} {size:.1f} MB")

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
