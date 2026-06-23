import os
import glob
import shutil
import json
import random
from pathlib import Path

# Add the root directory to sys.path so we can import from scripts
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.clip_indexer_allphase import _extract_middle_frame, _analyze_frame_with_ollama

def main():
    root_dir = Path(os.getcwd())
    clips_dir = root_dir / "clips"
    test_dir = root_dir / "test1_clipVision"
    test_json = test_dir / "test.json"
    
    # 1. Create test directory
    test_dir.mkdir(exist_ok=True)
    print(f"Created test directory: {test_dir.name}")
    
    # 2. Find all mp4 files recursively
    all_clips = list(clips_dir.rglob("*.mp4"))
    
    if not all_clips:
        print("No clips found in clips/ directory!")
        return
        
    # 3. Pick 5 random clips
    test_clips = random.sample(all_clips, min(5, len(all_clips)))
    
    print(f"Selected {len(test_clips)} random clips for testing.")
    
    # 4. Copy them
    copied_clips = []
    for clip in test_clips:
        dest = test_dir / clip.name
        shutil.copy2(clip, dest)
        copied_clips.append(dest)
        
    # 5. Load characters from config to simulate the real pipeline
    characters = []
    show_name = "rick_and_morty"
    try:
        import yaml
        with open("config/show_config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            characters = cfg.get("shows", {}).get(show_name, {}).get("characters", [])
        print(f"Loaded {len(characters)} characters from show_config.yaml")
    except Exception as e:
        print(f"Could not load characters from config: {e}")
        
    # 6. Process the clips using the exact same logic as Phase 3
    results = []
    temp_img = test_dir / "temp_frame.jpg"
    
    print("\nWaking up Ollama LLaVA Vision Model...\n" + "-"*50)
    for clip_path in copied_clips:
        print(f"Analyzing {clip_path.name}...")
        
        if not _extract_middle_frame(clip_path, temp_img):
            print("   Failed to extract frame.")
            continue
            
        metadata = _analyze_frame_with_ollama(temp_img, show_name, "llava", characters)
        
        if metadata:
            res = {
                "clip": clip_path.name,
                "vision_results": metadata
            }
            results.append(res)
            print(f"   Characters: {', '.join(metadata['characters']) or 'None'}")
            print(f"   Location:   {metadata['location'] or 'Unknown'}")
            print(f"   Action:     {metadata['action'] or 'N/A'}\n")
        else:
            print("   Vision model failed to respond properly.\n")
            
    # 7. Cleanup temp image
    if temp_img.exists():
        temp_img.unlink()
        
    # 8. Save test results to JSON
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("-" * 50)
    print(f"Test complete! Saved full results to {test_json}")

if __name__ == "__main__":
    main()
