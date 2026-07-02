import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Please install faster-whisper: pip install faster-whisper")
    sys.exit(1)

# We use subliminal as the python-native equivalent to Bazarr for a standalone script.
# Bazarr requires Sonarr to function properly, whereas subliminal can just scan video files directly.
try:
    from babelfish import Language
    from subliminal import download_best_subtitles, region, save_subtitles, scan_video
except ImportError:
    print("Please install subliminal: pip install subliminal")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("subtitle_manager")

# Suppress noisy external library logs (even their internal ERRORs)
logging.getLogger("subliminal").setLevel(logging.CRITICAL)
logging.getLogger("dogpile").setLevel(logging.CRITICAL)
logging.getLogger("babelfish").setLevel(logging.CRITICAL)

class SubtitleManager:
    def __init__(self, whisper_model_size="large-v3", device="cuda", compute_type="float16"):
        self.whisper_model_size = whisper_model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_whisper(self):
        """Lazy load the Whisper model only when needed to save VRAM."""
        if self.model is None:
            log.info(f"Loading Whisper model '{self.whisper_model_size}' on {self.device} ({self.compute_type})...")
            # For RTX 4050 (6GB VRAM), large-v3 with float16 is the best balance of max accuracy and VRAM usage.
            self.model = WhisperModel(self.whisper_model_size, device=self.device, compute_type=self.compute_type)
            log.info("Whisper model loaded successfully!")

    def download_with_bazarr_alternative(self, video_path: Path, output_dir: Path) -> bool:
        """
        Attempts to download English subtitles using subliminal (the core engine similar to Bazarr).
        This avoids needing a full Sonarr setup just to scan a file.
        """
        log.info(f"Searching online providers for subtitles: {video_path.name}")
        
        # Configure cache for subliminal (allow replacing to avoid crashes in loops)
        region.configure('dogpile.cache.dbm', arguments={'filename': 'subliminal_cache.dbm'}, replace_existing_backend=True)
        
        video = scan_video(str(video_path))
        languages = {Language('eng')}
        
        # Search OpenSubtitles, TVSubtitles, Addic7ed, etc.
        subtitles = download_best_subtitles([video], languages)
        
        if subtitles.get(video):
            log.info("Found matching subtitles online!")
            save_subtitles(video, subtitles[video], single=True, directory=str(output_dir))
            return True
            
        log.info("No online subtitles found.")
        return False

    def generate_with_whisper(self, video_path: Path, srt_path: Path):
        """Generates highly accurate subtitles using faster-whisper on the GPU."""
        log.info(f"Starting GPU Whisper transcription for: {video_path.name}")
        self.load_whisper()
        
        # Transcribe
        segments, info = self.model.transcribe(str(video_path), beam_size=5, language="en")
        
        log.info(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        # Write to SRT
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                # Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
                start = self._format_timestamp(segment.start)
                end = self._format_timestamp(segment.end)
                text = segment.text.strip()
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
                
                # Print progress occasionally
                if i % 50 == 0:
                    log.info(f"Transcribed {i} segments (approx {int(segment.end)} seconds of audio)...")
                    
        log.info(f"Whisper generation complete: {srt_path.name}")

    def _format_timestamp(self, seconds: float) -> str:
        """Helper to format seconds into SRT timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def process_directory(self, target_dir: str, output_srt_dir: str = "ben10_subtitles"):
        target_path = Path(target_dir)
        if not target_path.exists() or not target_path.is_dir():
            log.error(f"Invalid directory: {target_dir}")
            return

        srt_dir_path = Path(output_srt_dir)
        srt_dir_path.mkdir(parents=True, exist_ok=True)

        video_extensions = {".mp4", ".mkv", ".avi", ".mov"}
        # Use rglob to recursively find videos in nested folders (like season1/)
        video_files = [f for f in target_path.rglob("*") if f.is_file() and f.suffix.lower() in video_extensions]
        
        log.info(f"Found {len(video_files)} video files in {target_dir}")

        for video_file in video_files:
            srt_file = srt_dir_path / video_file.with_suffix(".srt").name
            
            # Skip if we already have subtitles
            if srt_file.exists():
                log.info(f"⏭️  SKIPPED: {video_file.name} (Subtitles already exist)")
                continue
                
            log.info(f"\n--- Processing {video_file.name} ---")
            
            # Step 1: Try downloading (Bazarr equivalent)
            success = self.download_with_bazarr_alternative(video_file, srt_dir_path)
            
            with open(srt_dir_path / "generation_report.txt", "a") as report:
                # Step 2: Fallback to GPU generation
                if success:
                    log.info(f"✅ DOWNLOADED: {video_file.name} (Found online match)")
                    report.write(f"DOWNLOADED : {video_file.name}\n")
                else:
                    log.info(f"⚡ WHISPER GPU: {video_file.name} (Generating from scratch)")
                    self.generate_with_whisper(video_file, srt_file)
                    log.info(f"✅ WHISPER DONE: {video_file.name}")
                    report.write(f"WHISPER    : {video_file.name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-detect & generate subtitles for videos.")
    parser.add_argument("directory", help="Directory containing the video files")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (default: large-v3 for best accuracy)")
    parser.add_argument("--outdir", default="ben10_subtitles", help="Directory to save generated .srt files")
    
    args = parser.parse_args()
    
    manager = SubtitleManager(whisper_model_size=args.model)
    manager.process_directory(args.directory, output_srt_dir=args.outdir)
