"""
tts_local.py — Phase 2: Local TTS using Piper (development testing only)
=========================================================================
Converts script text files to .wav audio using the Piper TTS CLI.

This is for LOCAL DEVELOPMENT TESTING ONLY — production TTS runs on Kaggle
via XTTS-v2 (see notebooks/kaggle_gpu_batch.py).

Usage:
    # Single file
    python scripts/tts_local.py --input scripts_text/intro.txt

    # Entire directory of text files
    python scripts/tts_local.py --input scripts_text/ --output-dir audio/

    # Override Piper model
    python scripts/tts_local.py --input scripts_text/ --model en_US-amy-medium

Requirements for Piper:
    pip install piper-tts
    (or install Piper standalone: https://github.com/rhasspy/piper)

Requirements for Kokoro:
    pip install soundfile kokoro-onnx
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running as `python scripts/tts_local.py` from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import (
    load_pipeline_config,
    get_project_path,
    setup_logging,
    PROJECT_ROOT,
)

log = setup_logging("tts_local")


# ---------------------------------------------------------------------------
# Piper availability check
# ---------------------------------------------------------------------------
def check_piper_installed() -> bool:
    """Check whether the `piper` CLI is available on PATH."""
    if shutil.which("piper") is not None:
        return True

    # Also try the Python module entry point
    try:
        result = subprocess.run(
            [sys.executable, "-m", "piper", "--help"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def get_piper_command() -> list[str]:
    """Return the base command list to invoke Piper."""
    if shutil.which("piper") is not None:
        return ["piper"]
    return [sys.executable, "-m", "piper"]


# ---------------------------------------------------------------------------
# Core TTS function
# ---------------------------------------------------------------------------
def synthesize_file(
    text_path: Path,
    output_path: Path,
    piper_model: str,
    sample_rate: int,
) -> bool:
    """
    Synthesize a single text file to .wav using Piper.

    Returns True on success, False on failure.
    """
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        log.warning("Skipping empty file: %s", text_path.name)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *get_piper_command(),
        "--model", piper_model,
        "--output_file", str(output_path),
    ]

    log.info("Synthesizing: %s → %s", text_path.name, output_path.name)
    log.debug("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per file
        )
        if result.returncode != 0:
            log.error(
                "Piper failed for %s (exit %d):\n%s",
                text_path.name,
                result.returncode,
                result.stderr.strip(),
            )
            return False

        if output_path.exists() and output_path.stat().st_size > 0:
            size_kb = output_path.stat().st_size / 1024
            log.info("  ✓ Generated %s (%.1f KB)", output_path.name, size_kb)
            return True
        else:
            log.error("  ✗ Output file missing or empty: %s", output_path)
            return False

    except subprocess.TimeoutExpired:
        log.error("Piper timed out for %s", text_path.name)
        return False
    except FileNotFoundError:
        log.error("Piper command not found — is piper-tts installed?")
        return False


# ---------------------------------------------------------------------------
# Kokoro TTS
# ---------------------------------------------------------------------------
def synthesize_file_kokoro(
    text_path: Path,
    output_path: Path,
    model_path: str,
    voices_path: str,
    voice: str,
    speed: float,
    lang: str,
) -> bool:
    """
    Synthesize a single text file to .wav using Kokoro standalone engine.
    """
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        log.warning("Skipping empty file: %s", text_path.name)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        from kokoro_onnx import Kokoro
    except ImportError:
        log.error("kokoro-onnx is not installed. Run: pip install soundfile kokoro-onnx")
        return False

    # Check if model files exist, if not we'll download them automatically
    import urllib.request
    import os

    if not os.path.exists(model_path):
        log.info(f"Downloading Kokoro model to {model_path}...")
        try:
            urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model/kokoro-v0_19.onnx", model_path)
            log.info("Download complete.")
        except Exception as e:
            log.error(f"Failed to download model: {e}")
            return False

    if not os.path.exists(voices_path):
        log.info(f"Downloading Kokoro voices to {voices_path}...")
        try:
            urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model/voices.json", voices_path)
            log.info("Download complete.")
        except Exception as e:
            log.error(f"Failed to download voices: {e}")
            return False

    log.info("Synthesizing (Kokoro): %s → %s", text_path.name, output_path.name)

    try:
        # Initialize the standalone engine
        kokoro = Kokoro(model_path, voices_path)
        
        # Generate the audio
        samples, sample_rate = kokoro.create(
            text,
            voice=voice,
            speed=speed,
            lang=lang
        )

        # Save the output
        sf.write(str(output_path), samples, sample_rate)

        if output_path.exists() and output_path.stat().st_size > 0:
            size_kb = output_path.stat().st_size / 1024
            log.info("  ✓ Generated %s (%.1f KB)", output_path.name, size_kb)
            return True
        else:
            log.error("  ✗ Output file missing or empty: %s", output_path)
            return False

    except Exception as e:
        log.error("Kokoro synthesis failed for %s: %s", text_path.name, e)
        return False


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_input(
    input_path: Path,
    output_dir: Path,
    tts_engine: str,
    tts_params: dict,
) -> dict:
    """
    Process a single file or a directory of .txt files.

    Returns a summary dict with counts of successes and failures.
    """
    results = {"success": 0, "failed": 0, "skipped": 0, "files": []}

    if input_path.is_file():
        text_files = [input_path]
    elif input_path.is_dir():
        text_files = sorted(input_path.glob("*.txt"))
        if not text_files:
            log.warning("No .txt files found in %s", input_path)
            return results
    else:
        log.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    log.info("Processing %d text file(s)...", len(text_files))

    for text_file in text_files:
        wav_name = text_file.stem + ".wav"
        output_path = output_dir / wav_name

        if tts_engine == "kokoro":
            ok = synthesize_file_kokoro(
                text_file,
                output_path,
                model_path=tts_params.get("model", "kokoro-v0_19.onnx"),
                voices_path=tts_params.get("voices", "voices.json"),
                voice=tts_params.get("voice", "af_bella"),
                speed=tts_params.get("speed", 1.0),
                lang=tts_params.get("lang", "en-us"),
            )
        else: # default to piper
            ok = synthesize_file(
                text_file,
                output_path,
                tts_params.get("piper_model", "en_US-lessac-medium"),
                tts_params.get("sample_rate", 22050)
            )

        if ok:
            results["success"] += 1
            results["files"].append(str(output_path))
        else:
            results["failed"] += 1

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local TTS using Piper — generates .wav from text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/tts_local.py --input scripts_text/intro.txt\n"
            "  python scripts/tts_local.py --input scripts_text/ --output-dir audio/\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a .txt file or a directory containing .txt files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for .wav files (default: from pipeline_config.yaml -> audio/)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Piper model name (default: from pipeline_config.yaml)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Output sample rate in Hz (default: from pipeline_config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config -------------------------------------------------------
    config = load_pipeline_config()
    tts_cfg = config.get("tts", {})
    engine = tts_cfg.get("engine", "piper")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = get_project_path("audio_dir")

    input_path = Path(args.input).resolve()
    
    tts_params = {}
    if engine == "kokoro":
        kokoro_cfg = tts_cfg.get("kokoro", {})
        tts_params["model"] = kokoro_cfg.get("model", "kokoro-v0_19.onnx")
        tts_params["voices"] = kokoro_cfg.get("voices", "voices.json")
        tts_params["voice"] = kokoro_cfg.get("voice", "af_bella")
        tts_params["speed"] = kokoro_cfg.get("speed", 1.0)
        tts_params["lang"] = kokoro_cfg.get("lang", "en-us")
    else:
        piper_cfg = tts_cfg.get("piper", {})
        tts_params["piper_model"] = args.model or piper_cfg.get("model", "en_US-lessac-medium")
        tts_params["sample_rate"] = args.sample_rate or piper_cfg.get("sample_rate", 22050)

        # --- Check Piper availability ------------------------------------------
        if not check_piper_installed():
            log.error(
                "Piper TTS is not installed or not found on PATH.\n\n"
                "Install options:\n"
                "  1. pip install piper-tts\n"
                "  2. Download from https://github.com/rhasspy/piper/releases\n\n"
                "This script is for LOCAL DEVELOPMENT TESTING ONLY.\n"
                "For production-quality TTS, use notebooks/kaggle_gpu_batch.py "
                "with XTTS-v2 on a Kaggle GPU runtime."
            )
            sys.exit(1)

    # --- Run synthesis -----------------------------------------------------
    log.info("=" * 60)
    log.info("TTS — Local Development")
    log.info("=" * 60)
    log.info("Engine:     %s", engine)
    log.info("Input:      %s", input_path)
    log.info("Output dir: %s", output_dir)
    log.info("-" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = process_input(input_path, output_dir, engine, tts_params)

    # --- Summary -----------------------------------------------------------
    log.info("-" * 60)
    log.info(
        "Done! %d succeeded, %d failed",
        results["success"],
        results["failed"],
    )
    if results["files"]:
        log.info("Output files:")
        for f in results["files"]:
            log.info("  %s", f)

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
