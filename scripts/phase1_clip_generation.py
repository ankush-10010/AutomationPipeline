import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        # We use subprocess.run so the user can see the live output from the scripts
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {description} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Master Script: Subtitle Generation & Clip Ingestion")
    parser.add_argument("episodes_dir", help="Directory containing raw video episodes (.mp4, .mkv)")
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir).resolve()

    if not episodes_dir.exists() or not episodes_dir.is_dir():
        print(f"❌ ERROR: Directory not found: {episodes_dir}")
        sys.exit(1)

    print(f"🌟 Starting Phase 1 Clip Generation Pipeline on directory: {episodes_dir}\n")

    # Step 1: Subtitle Check/Download/Generate (using Subliminal + GPU Whisper)
    run_command(
        [sys.executable, "scripts/subtitle_manager.py", str(episodes_dir)],
        "Step 1/2: Subtitle Generation & Validation"
    )

    # Step 2: Slice scenes and build the JSON clip database
    run_command(
        [sys.executable, "scripts/clip_indexer_allphasesUpdated.py", "--batch", str(episodes_dir)],
        "Step 2/2: Scene Splitting & Clip Indexing (All Phases)"
    )

    print(f"\n{'='*60}")
    print(f"✅ PHASE 1 COMPLETE! All episodes in {episodes_dir.name} have been processed.")
    print("Clips have been cut, tagged with subtitles, and added to the clip index.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
