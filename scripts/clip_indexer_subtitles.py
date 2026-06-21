"""
clip_indexer_subtitles.py - Auto-tag video clips based on subtitle transcripts.

This script takes the JSON manifest created by scene_splitter.py and the
original episode's .srt subtitle file. It cross-references the timecodes
and automatically tags each clip in clip_index.json with the exact dialogue
spoken during that scene.

Usage:
    python scripts/clip_indexer_subtitles.py --manifest clips/s1e1_manifest.json --srt episodes/s1e1.srt --show rick_and_morty
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Common English stop-words to exclude from generated tags
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "have", "has", "do", "did", 
              "will", "would", "shall", "should", "can", "could", "of", "in", "to", "for", "on", "with", 
              "at", "by", "from", "and", "or", "but", "not", "so", "it", "he", "she", "they", "we", "you", 
              "i", "me", "my", "your", "this", "that", "what", "which", "who", "how", "when", "where", "why",
              "just", "like", "get", "got", "know", "think", "right", "yeah", "oh", "well"}

def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format (00:00:02,000) to seconds."""
    h, m, s_ms = time_str.strip().split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def load_srt(srt_path: str) -> list:
    """Parse an SRT file into a list of subtitle dictionaries."""
    with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Blocks are separated by double newlines
    blocks = content.strip().split('\n\n')
    subs = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Line 1 is the time range: 00:00:02,000 --> 00:00:05,000
            times = lines[1].split(' --> ')
            if len(times) == 2:
                try:
                    start_sec = parse_srt_time(times[0])
                    end_sec = parse_srt_time(times[1])
                    # Line 2 onwards is the text
                    text = " ".join(lines[2:]).replace('\n', ' ')
                    
                    # Remove HTML tags or subtitle specific formatting (like <i>)
                    text = re.sub(r'<[^>]+>', '', text)
                    
                    subs.append({"start": start_sec, "end": end_sec, "text": text})
                except Exception as e:
                    print(f"Skipping malformed SRT block: {lines[1]}")
    
    return subs

def generate_keywords(text: str) -> list:
    """Extract clean keyword tags from text."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    words = text.split()
    # Keep unique words > 2 chars that aren't stop words
    keywords = list(set([w for w in words if w not in STOP_WORDS and len(w) > 2]))
    return keywords

def auto_index_clips(manifest_path: str, srt_path: str, show_slug: str, index_path: str = "clip_index.json"):
    manifest_path = Path(manifest_path)
    srt_path = Path(srt_path)
    index_path = Path(index_path)

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        return
    if not srt_path.exists():
        print(f"Error: SRT file not found: {srt_path}")
        return

    # 1. Load data
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    subs = load_srt(srt_path)
    print(f"Loaded {len(manifest)} clips from manifest and {len(subs)} subtitle blocks.")

    # 2. Load existing index
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {"clips": []}
    
    existing_filenames = {c["filename"] for c in index_data.get("clips", [])}

    # 3. Match clips to subtitles
    new_clips_added = 0
    
    for clip_name, times in manifest.items():
        if clip_name in existing_filenames:
            continue
            
        clip_start = times["start_sec"]
        clip_end = times["end_sec"]
        
        # Find all subtitles that overlap with this clip's time range
        # We consider an overlap if the subtitle starts before the clip ends AND ends after the clip starts
        overlapping_text = []
        for sub in subs:
            if sub["start"] < clip_end and sub["end"] > clip_start:
                overlapping_text.append(sub["text"])
        
        if overlapping_text:
            combined_text = " ".join(overlapping_text)
            tags = generate_keywords(combined_text)
            
            clip_entry = {
                "filename": clip_name,
                "show": show_slug,
                "season": 1,  # Default, can be customized later
                "episode": 1, # Default
                "characters": [], # Auto-extracting characters from sub text is hard without NLP, leaving empty
                "location": "",
                "action": combined_text,  # We store the exact spoken dialogue as the 'action'
                "mood": "",
                "tags": tags,
                "duration_seconds": round(clip_end - clip_start, 2)
            }
            
            index_data["clips"].append(clip_entry)
            new_clips_added += 1

    # 4. Save updated index
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
        
    print(f"✅ Auto-indexing complete!")
    print(f"Added {new_clips_added} new clips to {index_path} with subtitle tags.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-tag video clips using SRT subtitles.")
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest created by scene_splitter.py")
    parser.add_argument("--srt", required=True, help="Path to the episode's .srt subtitle file")
    parser.add_argument("--show", required=True, help="The slug name of the show (e.g. rick_and_morty)")
    parser.add_argument("--index", default="clip_index.json", help="Path to the master clip index JSON")
    
    args = parser.parse_args()
    auto_index_clips(args.manifest, args.srt, args.show, args.index)
