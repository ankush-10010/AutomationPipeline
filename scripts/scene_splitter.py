"""
scene_splitter.py - Utility to break down full episodes into short scene clips.

This script uses PySceneDetect to analyze a long video file, find the exact moments
where the camera cuts to a new scene, and uses FFmpeg to slice the video into 
perfectly cut short clips.

Usage:
    python scripts/scene_splitter.py path/to/episode.mp4 --output clips/ --prefix "s1e1"
"""

import argparse
import sys
from pathlib import Path
import logging

try:
    from scenedetect import detect, ContentDetector, split_video_ffmpeg
except ImportError:
    print("Error: scenedetect not found.")
    print("Please install it by running: pip install scenedetect[opencv]")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("scene_splitter")

def split_episode(video_path: str, output_dir: str, prefix: str, threshold: float = 27.0):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        log.error(f"Video file not found: {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Analyzing {video_path.name} for scene changes...")
    log.info(f"This may take a few minutes depending on the video length.")
    
    # ContentDetector finds cuts between different scenes
    # threshold=27.0 is the default. Lower = more sensitive (more cuts), Higher = less sensitive.
    detector = ContentDetector(threshold=threshold)
    
    # Find the scenes
    scene_list = detect(str(video_path), detector)
    log.info(f"Detected {len(scene_list)} scenes.")
    
    if not scene_list:
        log.warning("No scenes detected. Try lowering the threshold.")
        return

    # Define output template: e.g., output/s1e1_scene_001.mp4
    output_template = str(output_dir / f"{prefix}_scene_$SCENE_NUMBER.mp4")
    
    log.info(f"Splitting video into {len(scene_list)} clips...")
    # split_video_ffmpeg runs FFmpeg under the hood to cut the video losslessly (fast)
    split_video_ffmpeg(
        input_video_path=str(video_path),
        scene_list=scene_list,
        output_file_template=output_template,
        show_progress=True
    )
    
    # Save a manifest of exact timecodes so the subtitle indexer can match them
    import json
    manifest = {}
    for i, (start_time, end_time) in enumerate(scene_list):
        # By default scenedetect uses 1-based indexing formatted as 3 digits: %03d
        clip_name = f"{prefix}_scene_{i+1:03d}.mp4"
        manifest[clip_name] = {
            "start_sec": start_time.get_seconds(),
            "end_sec": end_time.get_seconds()
        }
    
    manifest_path = output_dir / f"{prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    log.info(f"✅ Success! {len(scene_list)} clips saved to {output_dir}")
    log.info(f"✅ Saved timecode manifest to {manifest_path}")

if __name__ == "__main__":
    import torch
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    color = "\033[92m" if torch.cuda.is_available() else "\033[93m"
    print(f"\n{color}🚀 Scene Splitter is running on: {device_name}\033[0m\n")

    parser = argparse.ArgumentParser(description="Split long videos into scenes.")
    parser.add_argument("video", help="Path to the long video file (e.g. episode.mp4)")
    parser.add_argument("--output", default="clips", help="Directory to save the clips (default: clips)")
    parser.add_argument("--prefix", default="clip", help="Prefix for the generated clips (e.g. 's1e1')")
    parser.add_argument("--threshold", type=float, default=27.0, help="Scene change sensitivity (default: 27.0. Lower is more sensitive)")
    
    args = parser.parse_args()
    split_episode(args.video, args.output, args.prefix, args.threshold)
