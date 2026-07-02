"""
===============================================================================
ANTIGRAVITY & USER INSTRUCTIONS FOR VISUAL TAGGING PIPELINE
===============================================================================

Hello Antigravity Assistant / User!
This script is the master wrapper for updating 'clip_index.json' with our new 
visual character detection system (YOLOv8 + CLIP k-NN Hybrid).

RECOMMENDED PIPELINE WORKFLOW:
------------------------------
Always run the steps sequentially in this order:
  [Step 1] Build k-NN Reference Database
           -> Processes 'Ready Dataset' so CLIP knows the 20 rare classes.
  [Step 2] Test Hybrid Inference on Episode 1 (s1e1)
           -> Fast test run using YOLO weights + k-NN on just the first episode.
  [Step 3] Sanity Check & Review Results
           -> Inspects clip_index.json for s1e1 to verify character tags look correct.
  [Step 4] Run Full Hybrid Inference on ALL Clips
           -> Process the entire show's library once s1e1 is verified.

ADVANCED OPTIONS:
-----------------
  [Option 5] Force Re-run on All Clips (--force)
             -> Overwrites existing 'visual_characters' tags across all clips.
  [Option 6] Fast YOLO-Only Mode (--yolo-only)
             -> Skips k-NN entirely. Only detects the 10 core trained classes. Extremely fast.

ANTIGRAVITY GUIDANCE:
---------------------
When guiding the user, recommend executing Step 1 first, then Step 2, then Step 3.
During Step 3, review the printed sanity check output with the user to ensure
characters are being correctly identified and deduplicated before triggering Step 4.
===============================================================================
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = PROJECT_ROOT / "yolo_wt" / "best.pt"
READY_DATASET = PROJECT_ROOT / "Ready Dataset"
CLIP_INDEX = PROJECT_ROOT / "clip_index.json"

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_prerequisites():
    print("Checking prerequisites...")
    if not YOLO_WEIGHTS.exists():
        print(f"⚠️  WARNING: YOLO weights not found at: {YOLO_WEIGHTS}")
        print("   Please ensure you have downloaded 'best.pt' from Colab and placed it in 'yolo_wt/best.pt'.")
    else:
        print(f"✅ Found YOLO weights: {YOLO_WEIGHTS}")
        
    if not READY_DATASET.exists():
        print(f"⚠️  WARNING: Ready Dataset not found at: {READY_DATASET}")
    else:
        print(f"✅ Found Ready Dataset: {READY_DATASET}")
        
    if not CLIP_INDEX.exists():
        print(f"❌ ERROR: clip_index.json not found at: {CLIP_INDEX}")
        return False
    else:
        print(f"✅ Found clip_index.json: {CLIP_INDEX}")
    return True

def run_step_1_build_knn():
    print_header("STEP 1: Building k-NN Reference Database")
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "clip_classifier_knn.py"), "build-ref", "--dataset", "Ready Dataset"]
    print(f"Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)

def run_step_2_test_s1e1():
    print_header("STEP 2: Running Hybrid Inference Test on Episode 1 (s1e1)")
    if not YOLO_WEIGHTS.exists():
        print("❌ Cannot run: YOLO weights missing at yolo_wt/best.pt")
        return
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "YOLO_hybrid_inference.py"), "--weights", str(YOLO_WEIGHTS), "--episode", "s1e1"]
    print(f"Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)

def run_step_3_sanity_check():
    print_header("STEP 3: Sanity Check on Episode 1 Results")
    if not CLIP_INDEX.exists():
        print("❌ Cannot inspect: clip_index.json not found.")
        return
        
    with open(CLIP_INDEX, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    clips = data.get("clips", data) if isinstance(data, dict) else data
    s1e1_clips = [c for c in clips if c.get("filename", "").lower().startswith("s1e1")]
    
    tagged_clips = [c for c in s1e1_clips if "visual_characters" in c]
    
    print(f"Total s1e1 clips found: {len(s1e1_clips)}")
    print(f"Clips successfully tagged with 'visual_characters': {len(tagged_clips)}\n")
    
    if not tagged_clips:
        print("⚠️ No tagged s1e1 clips found. Please run Step 2 first!")
        return
        
    print("Sample of tagged clips (First 15):")
    print("-" * 70)
    for c in tagged_clips[:15]:
        fname = c.get("filename", "unknown")
        chars = c.get("visual_characters", [])
        # Verify deduplication
        is_unique = len(chars) == len(set(chars))
        unique_flag = "✅ Unique" if is_unique else "❌ Duplicates Found!"
        print(f"  🎬 {fname:30s} -> {str(chars):30s} [{unique_flag}]")
    print("-" * 70)
    
    # Character frequency distribution in s1e1
    all_chars = []
    for c in tagged_clips:
        all_chars.extend(c.get("visual_characters", []))
        
    from collections import Counter
    counts = Counter(all_chars)
    print("\nCharacter Detection Frequency in Episode 1:")
    for char_name, cnt in counts.most_common():
        print(f"  👤 {char_name:20s}: {cnt} clips")
    print("\nAntigravity / User Guidance: Inspect the sample above. If tags align well with the show, proceed to Step 4!")

def run_step_4_full_inference(force=False, yolo_only=False):
    title = "STEP 4: Full Hybrid Inference on ALL Clips"
    if force: title += " [FORCE RE-RUN]"
    if yolo_only: title += " [YOLO-ONLY FAST MODE]"
    print_header(title)
    
    if not YOLO_WEIGHTS.exists():
        print("❌ Cannot run: YOLO weights missing at yolo_wt/best.pt")
        return
        
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "YOLO_hybrid_inference.py"), "--weights", str(YOLO_WEIGHTS)]
    if force: cmd.append("--force")
    if yolo_only: cmd.append("--yolo-only")
    
    print(f"Running command: {' '.join(cmd)}\n")
    subprocess.run(cmd)

def main():
    print_header("BEN 10 VISUAL CHARACTER TAGGING MASTER PIPELINE")
    check_prerequisites()
    
    while True:
        print("\n" + "-" * 70)
        print("SELECT AN ACTION (Recommended sequence: 1 -> 2 -> 3 -> 4):")
        print("  1. [Step 1] Build k-NN Reference Database (Run once first)")
        print("  2. [Step 2] Test Hybrid Inference on Episode 1 (s1e1)")
        print("  3. [Step 3] Sanity Check & Review Episode 1 Results")
        print("  4. [Step 4] Run Full Hybrid Inference on ALL Clips")
        print("  --------------------------------------------------")
        print("  5. [Advanced] Force Re-run Hybrid Inference on ALL Clips (--force)")
        print("  6. [Advanced] Run Fast YOLO-Only Mode on ALL Clips (--yolo-only)")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-6) [Recommended: 1]: ").strip()
        
        if choice == "1" or (choice == ""):
            run_step_1_build_knn()
        elif choice == "2":
            run_step_2_test_s1e1()
        elif choice == "3":
            run_step_3_sanity_check()
        elif choice == "4":
            confirm = input("Ready to process ALL clips? (y/n) [y]: ").strip().lower()
            if confirm in ["y", "yes", ""]:
                run_step_4_full_inference()
        elif choice == "5":
            confirm = input("⚠️ This will overwrite all existing tags. Continue? (y/n) [n]: ").strip().lower()
            if confirm in ["y", "yes"]:
                run_step_4_full_inference(force=True)
        elif choice == "6":
            run_step_4_full_inference(yolo_only=True)
        elif choice == "0":
            print("\nExiting pipeline. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 0 to 6.")

if __name__ == "__main__":
    main()
