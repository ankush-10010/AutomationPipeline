import os
import subprocess
import time
from pathlib import Path

# Paths
BASE_DIR = Path("/home/rishav/MY_PERSONAL_WORKS/MY_ALL_PROJECTS/Github(Ankush+ME)/AutomationPipeline-main")
VIDEOS_DIR = BASE_DIR / "Videos" / "Ben10_Classic"
DATASET_DIR = BASE_DIR / "Dataset"

def main():
    if not DATASET_DIR.exists():
        DATASET_DIR.mkdir(parents=True)

    # Find all video files
    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.avi"]:
        video_files.extend(VIDEOS_DIR.rglob(ext))
    
    video_files = sorted(video_files)
    print(f"Found {len(video_files)} episodes to process.")
    
    total_extracted = 0
    start_time = time.time()
    
    for idx, video_path in enumerate(video_files, 1):
        # We use a clean prefix like S1E1
        season_folder = video_path.parent.name # e.g., Season_1
        season = ''.join(filter(str.isdigit, season_folder)) or "0"
        
        # Try to guess episode number from filename
        ep_name = "".join(c for c in video_path.stem if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')
        prefix = f"S{season}_{ep_name}"
        
        output_pattern = f"Dataset/{prefix}_%04d.jpg"
        
        print(f"[{idx}/{len(video_files)}] Extracting diverse frames from {video_path.name}...")
        
        # FFmpeg magic filter:
        # gt(scene,0.1) -> Extracts a frame if there is a camera cut / visual scene change
        # gt(t-prev_selected_t,5) -> OR if it's been 5 seconds since the last extracted frame
        # This guarantees we get every distinct camera shot, plus coverage of long continuous shots!
        filter_str = "select='gt(scene,0.1) + isnan(prev_selected_t) + gt(t-prev_selected_t,5)'"
        
        cmd = [
            "ffmpeg",
            "-y", # Overwrite if exists
            "-i", str(video_path),
            "-vf", filter_str,
            "-vsync", "vfr",
            "-q:v", "2", # High quality JPEG
            output_pattern
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"  [Error] FFmpeg failed on {video_path.name}")
            print(f"  Reason: {result.stderr.splitlines()[-1] if result.stderr else 'Unknown'}")
            
    end_time = time.time()
    
    # Count final images
    extracted_images = list(DATASET_DIR.glob("*.jpg"))
    print(f"\n✅ Finished! Extracted {len(extracted_images)} highly diverse screenshots in {round((end_time-start_time)/60, 2)} minutes.")
    print(f"They are safely saved in: {DATASET_DIR}")

if __name__ == "__main__":
    main()
