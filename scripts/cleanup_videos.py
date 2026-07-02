import os
import shutil
import re
from pathlib import Path

def clean_and_organize_videos(base_dir="Videos"):
    base_path = Path(base_dir)
    target_show_dir = base_path / "Ben10_Classic"
    
    # Regex to extract Season and Episode numbers from filenames
    # This matches S01E01, S1E1, S01_E01, etc.
    pattern = re.compile(r"[Ss](\d+)[_ ]?[Ee](\d+)")

    # Find all video files in the Videos directory
    video_extensions = {".mp4", ".mkv", ".avi"}
    all_videos = [f for f in base_path.rglob("*") if f.is_file() and f.suffix.lower() in video_extensions]

    print(f"Found {len(all_videos)} video files. Organizing...")

    moved_count = 0
    for video in all_videos:
        # Skip if already perfectly placed (in case we run this twice)
        if target_show_dir in video.parents and video.name.startswith("Ben_10_Classic_S"):
            continue

        match = pattern.search(video.name)
        if match:
            season = int(match.group(1))
            episode = int(match.group(2))
            
            # Create target Season folder
            season_dir = target_show_dir / f"Season_{season}"
            season_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean filename: Ben_10_Classic_S01E01.mp4
            clean_name = f"Ben_10_Classic_S{season:02d}E{episode:02d}{video.suffix.lower()}"
            target_path = season_dir / clean_name
            
            print(f"Moving: {video.name} -> {target_path.relative_to(base_path)}")
            shutil.move(str(video), str(target_path))
            moved_count += 1
        else:
            print(f"Warning: Could not detect Season/Episode in '{video.name}'. Skipping.")

    print(f"\nSuccessfully organized and renamed {moved_count} episodes.")

    # Cleanup empty directories
    print("Cleaning up old empty folders...")
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

if __name__ == "__main__":
    clean_and_organize_videos()
