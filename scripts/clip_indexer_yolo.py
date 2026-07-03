import argparse
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ── Aggregation Thresholds (Same as inference script) ──────────
CLEAR_APPEARANCE_THRESHOLD = 0.85
MIN_FRAME_RATIO = 0.15
MIN_MAX_CONF_FOR_RATIO = 0.20
FRAME_PRESENCE_THRESHOLD = 0.10


import torch

def classify_clip(model, video_path):
    """Run classification on frames and aggregate results."""
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # stream=True ensures we don't run out of RAM
        results = model.predict(
            source=str(video_path), 
            stream=True, 
            verbose=False,
            device=device,
            conf=0.65  # High confidence to prevent false positives in official pipeline
        )
    except Exception as e:
        print(f"  [Error reading video: {e}]", end="")
        return []

    char_confidences = defaultdict(list)
    total_frames = 0

    for result in results:
        total_frames += 1
        names_dict = result.names
        probs = result.probs.data.tolist()

        for class_id, conf in enumerate(probs):
            char_name = names_dict[class_id]
            if char_name.lower() == "test":
                continue
            char_confidences[char_name].append(conf)

    if total_frames == 0:
        return []

    present_characters = []
    for char_name, confs in char_confidences.items():
        max_conf = max(confs)
        frames_above = sum(1 for c in confs if c > FRAME_PRESENCE_THRESHOLD)
        frame_ratio = frames_above / total_frames

        if max_conf >= CLEAR_APPEARANCE_THRESHOLD:
            present_characters.append(char_name)
        elif frame_ratio >= MIN_FRAME_RATIO and max_conf >= MIN_MAX_CONF_FOR_RATIO:
            present_characters.append(char_name)

    return present_characters


def main():
    parser = argparse.ArgumentParser(description="Batch update clip_index.json characters using YOLO")
    parser.add_argument("--index", default="clip_index.json", help="Path to clip_index.json")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO .pt file")
    parser.add_argument("--target-dir", help="Only process clips whose filepath contains this string (e.g. 'S8/E2')")
    parser.add_argument("--force", action="store_true", help="Re-process clips even if they were already tagged by YOLO")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: {index_path} not found.")
        return

    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        print("No clips found in index.")
        return

    import torch
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    color = "\033[92m" if torch.cuda.is_available() else "\033[93m"
    print(f"\n{color}🚀 YOLO Model is running on: {device_name}\033[0m\n")

    print(f"Loading YOLO model from {args.weights}...")
    model = YOLO(args.weights)

    # ── 1. Filter clips based on target_dir and completion status ──
    clips_to_process = []
    for clip in clips:
        # Check directory target
        if args.target_dir and args.target_dir.replace("\\", "/") not in clip.get("filepath", "").replace("\\", "/"):
            continue
        
        # Skip if already processed (unless --force is used)
        if clip.get("yolo_tagged", False) and not args.force:
            continue
            
        clips_to_process.append(clip)

    print(f"Found {len(clips_to_process)} clips needing YOLO processing.")
    if len(clips_to_process) == 0:
        return

    # ── 2. Process Clips and Save Incrementally ──
    def safe_save():
        """Write to a temp file first, then replace the real file to prevent corruption."""
        temp_path = index_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        temp_path.replace(index_path)

    try:
        for i, clip in enumerate(clips_to_process, 1):
            video_path = Path(clip["filepath"])
            if not video_path.exists():
                print(f"[{i}/{len(clips_to_process)}] SKIPPING (File not found): {video_path}")
                continue

            print(f"[{i}/{len(clips_to_process)}] Analyzing {clip['filename']}... ", end="", flush=True)
            
            detected_chars = classify_clip(model, video_path)
            
            # OVERWRITE the old garbage characters with the true YOLO ones
            clip["characters"] = detected_chars
            clip["yolo_tagged"] = True  
            
            print(f"Result: {detected_chars}")
            
            # Save every 50 clips to prevent huge disk I/O bottlenecks
            if i % 50 == 0:
                safe_save()

    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user! Saving progress before exiting...")
    finally:
        # Guarantee a save when the script finishes or is killed
        safe_save()

    print("\n✅ Batch YOLO tagging complete (progress safely saved).")

if __name__ == "__main__":
    main()
