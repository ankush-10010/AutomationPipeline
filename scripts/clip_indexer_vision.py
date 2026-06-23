"""
clip_indexer_vision.py - Visually auto-tag video clips using Ollama Vision models.

This script extracts a single frame from the middle of each video clip
and sends it to a local Vision-Language Model (like LLaVA) via Ollama.
The AI "watches" the frame and identifies the characters, location, and action.
It then updates your clip_index.json database with these visual tags.

Usage:
    python scripts/clip_indexer_vision.py --model llava
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

try:
    from moviepy import VideoFileClip
    from PIL import Image
except ImportError:
    print("Error: moviepy or Pillow not found.")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

import requests

def extract_middle_frame(video_path: str, temp_image_path: str):
    """Extracts the middle frame of a video and saves it as a JPEG."""
    try:
        clip = VideoFileClip(str(video_path))
        # Get frame exactly in the middle of the clip
        mid_time = clip.duration / 2.0
        frame_data = clip.get_frame(mid_time)
        
        # Convert numpy array to PIL Image and save
        img = Image.fromarray(frame_data)
        img.save(temp_image_path, "JPEG")
        clip.close()
        return True
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return False

def analyze_image_with_ollama(image_path: str, show_name: str, model: str) -> dict:
    """Sends the image to Ollama LLaVA model to extract visual metadata."""
    
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    prompt = f"""You are an expert on the TV show '{show_name}'. 
Analyze this single frame from an episode.
Identify ONLY the characters that are clearly visible in this specific frame. Do NOT list characters that are not in the image. If no characters are visible, write 'None'.
Identify the location, and what is happening visually.

You MUST reply in exactly this format with no other text:
Characters: [comma separated names of VISIBLE characters ONLY, or None]
Location: [brief location name, or Unknown]
Action: [1 short sentence describing the visual action]"""

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.2 # Keep it factual
        }
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        result_text = response.json().get("response", "").strip()
        
        # Parse the 3-line response
        metadata = {"characters": [], "location": "", "action": ""}
        for line in result_text.split('\n'):
            line = line.strip()
            if line.lower().startswith("characters:"):
                chars_str = line.split(":", 1)[1].strip()
                if chars_str.lower() not in ["none", "unknown", "n/a", ""]:
                    metadata["characters"] = [c.strip() for c in chars_str.split(",")]
            elif line.lower().startswith("location:"):
                loc_str = line.split(":", 1)[1].strip()
                if loc_str.lower() not in ["none", "unknown", "n/a", ""]:
                    metadata["location"] = loc_str
            elif line.lower().startswith("action:"):
                act_str = line.split(":", 1)[1].strip()
                if act_str.lower() not in ["none", "unknown", "n/a", ""]:
                    metadata["action"] = act_str
                    
        return metadata
    except requests.exceptions.RequestException as e:
        print(f"\nError communicating with Ollama: {e}")
        print("Make sure Ollama is running ('ollama serve') and you pulled the model.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Visually auto-tag clips using Ollama Vision.")
    parser.add_argument("--index", default="clip_index.json", help="Path to clip_index.json")
    parser.add_argument("--clips-dir", default="clips", help="Directory where video clips are stored")
    parser.add_argument("--model", default="llava", help="Ollama vision model to use (default: llava)")
    parser.add_argument("--force", action="store_true", help="Re-process clips that already have characters tagged")
    args = parser.parse_args()

    index_path = Path(args.index)
    clips_dir = Path(args.clips_dir)
    temp_img = Path("temp_vision_frame.jpg")

    if not index_path.exists():
        print(f"Error: {index_path} not found. Run a regular indexer first.")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        print("No clips found in index.")
        return

    print(f"Loaded {len(clips)} clips from database.")
    print(f"Waking up Vision AI ({args.model})...")

    updated_count = 0

    for i, clip in enumerate(clips):
        # Skip if already visually tagged, unless forced
        if clip.get("characters") and not args.force:
            continue

        video_path = clips_dir / clip["filename"]
        if not video_path.exists():
            print(f"Skipping {clip['filename']} (file missing)")
            continue

        print(f"[{i+1}/{len(clips)}] Watching {clip['filename']}...")
        
        # 1. Grab frame
        if not extract_middle_frame(video_path, str(temp_img)):
            continue
            
        # 2. Ask Vision AI
        show_name = clip.get("show", "the show")
        metadata = analyze_image_with_ollama(str(temp_img), show_name, args.model)
        
        if metadata:
            # 3. Update database safely
            clip["characters"] = metadata["characters"]
            
            # If the vision model found a location, use it
            if metadata["location"]:
                clip["location"] = metadata["location"]
                
            # We don't overwrite subtitle action, we just append visual action to tags
            if metadata["action"]:
                # Remove punctuation and split into keywords
                import re
                clean_action = re.sub(r'[^a-zA-Z0-9\s]', '', metadata["action"].lower())
                visual_tags = [w for w in clean_action.split() if len(w) > 3]
                
                # Merge existing tags with new visual tags
                existing_tags = clip.get("tags", [])
                merged_tags = list(set(existing_tags + visual_tags))
                clip["tags"] = merged_tags

            print(f"   👁️  Found: {', '.join(metadata['characters']) or 'No one'}")
            print(f"   📍 Location: {metadata['location'] or 'Unknown'}")
            updated_count += 1
            
            # Save incrementally so we don't lose data if it crashes
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    # Cleanup temp image
    if temp_img.exists():
        temp_img.unlink()

    print(f"\n✅ Vision Auto-Tagging Complete!")
    print(f"Successfully visually tagged {updated_count} clips.")

if __name__ == "__main__":
    main()
