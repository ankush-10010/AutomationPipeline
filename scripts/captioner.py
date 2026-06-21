"""
captioner.py — Phase 3: Audio captioning using faster-whisper
==============================================================
Transcribes audio files and extracts word-level timestamps for
TikTok-style word-by-word caption rendering in the video assembly phase.

Output format (one JSON per audio file):
{
  "source_audio": "intro.wav",
  "segments": [
    {
      "id": 0,
      "text": "Hello world this is a test",
      "start": 0.0,
      "end": 3.2,
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.3, "probability": 0.95},
        {"word": "world", "start": 0.35, "end": 0.6, "probability": 0.92},
        ...
      ]
    }
  ]
}

Usage:
    # Single file
    python scripts/captioner.py --input audio/intro.wav

    # Entire directory
    python scripts/captioner.py --input audio/ --output-dir captions/

Requirements:
    pip install faster-whisper>=1.0.0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Allow running as `python scripts/captioner.py` from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import (
    load_pipeline_config,
    get_project_path,
    setup_logging,
    save_json,
)

log = setup_logging("captioner")

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


# ---------------------------------------------------------------------------
# Whisper model loading (lazy singleton)
# ---------------------------------------------------------------------------
_whisper_model = None


def get_whisper_model(model_size: str, compute_type: str):
    """Load the faster-whisper model (cached after first call)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error(
            "faster-whisper is not installed.\n"
            "Install with: pip install faster-whisper>=1.0.0"
        )
        sys.exit(1)

    # Auto-detect GPU and optimize compute type
    device = "auto"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"  # GPU is much faster with float16
        else:
            device = "cpu"
            compute_type = "int8"     # CPU requires int8 or float32
    except ImportError:
        pass  # Fallback to the settings from pipeline_config.yaml

    log.info("Loading faster-whisper model: %s (device=%s, compute_type=%s)", model_size, device, compute_type)
    start = time.time()
    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    log.info("Model loaded in %.1f seconds", time.time() - start)
    return _whisper_model


# ---------------------------------------------------------------------------
# Core captioning function
# ---------------------------------------------------------------------------
def caption_audio_file(
    audio_path: Path,
    model_size: str,
    compute_type: str,
    language: str,
) -> Dict[str, Any]:
    """
    Transcribe an audio file and extract word-level timestamps.

    Returns a dict in the pipeline's standard caption format.
    """
    model = get_whisper_model(model_size, compute_type)

    log.info("Transcribing: %s", audio_path.name)
    start_time = time.time()

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,  # Filter out silence for cleaner segments
    )

    log.info(
        "  Language: %s (probability %.2f)",
        info.language,
        info.language_probability,
    )

    segments_out: List[Dict[str, Any]] = []

    for seg_id, segment in enumerate(segments_iter):
        words_out: List[Dict[str, Any]] = []

        if segment.words:
            for w in segment.words:
                words_out.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                })

        segments_out.append({
            "id": seg_id,
            "text": segment.text.strip(),
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "words": words_out,
        })

    elapsed = time.time() - start_time
    total_words = sum(len(s["words"]) for s in segments_out)
    log.info(
        "  ✓ %d segments, %d words in %.1f seconds",
        len(segments_out),
        total_words,
        elapsed,
    )

    return {
        "source_audio": audio_path.name,
        "segments": segments_out,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_input(
    input_path: Path,
    output_dir: Path,
    model_size: str,
    compute_type: str,
    language: str,
) -> Dict[str, Any]:
    """
    Process a single audio file or a directory of audio files.

    Returns a summary dict with counts of successes and failures.
    """
    results = {"success": 0, "failed": 0, "files": []}

    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not audio_files:
            log.warning("No audio files found in %s", input_path)
            return results
    else:
        log.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    log.info("Processing %d audio file(s)...", len(audio_files))
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in audio_files:
        try:
            caption_data = caption_audio_file(
                audio_file, model_size, compute_type, language
            )

            json_name = audio_file.stem + ".json"
            output_path = output_dir / json_name

            save_json(output_path, caption_data)
            log.info("  Saved → %s", output_path)

            results["success"] += 1
            results["files"].append(str(output_path))

        except Exception as e:
            log.error("Failed to caption %s: %s", audio_file.name, e, exc_info=True)
            results["failed"] += 1

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio captioning with word-level timestamps using faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/captioner.py --input audio/intro.wav\n"
            "  python scripts/captioner.py --input audio/ --output-dir captions/\n"
            "  python scripts/captioner.py --input audio/ --model-size medium\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to an audio file or directory of audio files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for caption JSON files (default: from pipeline_config.yaml → captions/)",
    )
    parser.add_argument(
        "--model-size",
        default=None,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: from pipeline_config.yaml)",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        choices=["int8", "float16", "float32"],
        help="Compute type for inference (default: from pipeline_config.yaml)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for transcription (default: from pipeline_config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config -------------------------------------------------------
    config = load_pipeline_config()
    cap_cfg = config.get("captioning", {})

    model_size = args.model_size or cap_cfg.get("model_size", "small")
    compute_type = args.compute_type or cap_cfg.get("compute_type", "int8")
    language = args.language or cap_cfg.get("language", "en")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = get_project_path("captions_dir")

    input_path = Path(args.input).resolve()

    # --- Run captioning ----------------------------------------------------
    log.info("=" * 60)
    log.info("Audio Captioner — faster-whisper")
    log.info("=" * 60)
    log.info("Model:        %s", model_size)
    log.info("Compute type: %s", compute_type)
    log.info("Language:     %s", language)
    log.info("Input:        %s", input_path)
    log.info("Output dir:   %s", output_dir)
    log.info("-" * 60)

    results = process_input(input_path, output_dir, model_size, compute_type, language)

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
