"""
===============================================================================
ANTIGRAVITY & USER INSTRUCTIONS FOR VISUAL TAGGING PIPELINE
===============================================================================

Hello Antigravity Assistant / User!
This script is the master wrapper for updating 'clip_index.json' with visual
character detection.

TWO PIPELINES ARE AVAILABLE:
-----------------------------
  [A] ArcFace Metric Learning (RECOMMENDED — covers all 30 characters)
      Step A1: Train ArcFace projection head on Colab
      Step A2: Run prototype inference locally on all clips
      Step A3: Sanity check results

  [B] YOLO + k-NN Hybrid (Legacy — covers only 10 core characters via YOLO)
      Step B1: Build k-NN reference database
      Step B2: Test hybrid inference on episode 1 (s1e1)
      Step B3: Sanity check results
      Step B4: Run full hybrid inference on all clips

ANTIGRAVITY GUIDANCE:
---------------------
Recommend Pipeline A (ArcFace) for new runs. It handles all 30 characters,
has native background rejection (no forced-softmax noise), and works with
as few as 6 training images per class.

Pipeline B is retained for comparison and as a fast secondary signal on the
10 core classes.
===============================================================================
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = PROJECT_ROOT / "yolo_wt" / "best.pt"
ARCFACE_HEAD = PROJECT_ROOT / "arcface_head.pt"
PROTOTYPES = PROJECT_ROOT / "prototypes.npz"
READY_DATASET = PROJECT_ROOT / "Ready Dataset"
CLIP_INDEX = PROJECT_ROOT / "clip_index.json"


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_prerequisites():
    print("\nChecking prerequisites...")
    checks = [
        (CLIP_INDEX, "clip_index.json"),
        (READY_DATASET, "Ready Dataset"),
        (ARCFACE_HEAD, "arcface_head.pt (ArcFace projection head)"),
        (PROTOTYPES, "prototypes.npz (ArcFace prototypes)"),
        (YOLO_WEIGHTS, "yolo_wt/best.pt (YOLO weights)"),
    ]
    for path, name in checks:
        if path.exists():
            print(f"  ✅ Found: {name}")
        else:
            print(f"  ⚠️  Missing: {name}")


def sanity_check(episode="s1e1"):
    print_header(f"SANITY CHECK: {episode.upper()} Results")
    if not CLIP_INDEX.exists():
        print("❌ clip_index.json not found.")
        return

    with open(CLIP_INDEX, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", data) if isinstance(data, dict) else data
    ep_clips = [c for c in clips if c.get("filename", "").lower().startswith(episode.lower())]
    tagged = [c for c in ep_clips if "visual_characters" in c]

    print(f"  Total {episode} clips: {len(ep_clips)}")
    print(f"  Tagged with visual_characters: {len(tagged)}\n")

    if not tagged:
        print("  ⚠️  No tagged clips found. Run inference first!")
        return

    print("  Sample of tagged clips (first 15):")
    print("  " + "-" * 65)
    for c in tagged[:15]:
        fname = c.get("filename", "unknown")
        chars = c.get("visual_characters", [])
        unique_ok = "✅" if len(chars) == len(set(chars)) else "❌ DUPES"
        print(f"    🎬 {fname:30s} → {str(chars):30s} [{unique_ok}]")
    print("  " + "-" * 65)

    all_chars = []
    for c in tagged:
        all_chars.extend(c.get("visual_characters", []))

    counts = Counter(all_chars)
    print(f"\n  Character frequency in {episode}:")
    for name, cnt in counts.most_common():
        pct = cnt / len(tagged) * 100
        flag = " ⚠️ HIGH" if pct > 60 else ""
        print(f"    👤 {name:20s}: {cnt:4d} clips ({pct:4.1f}%){flag}")

    print("\n  Review the above. If tags look accurate, proceed to full inference.")


# ── Pipeline A: ArcFace Metric Learning ──────────────────────────────────────

def run_arcface_colab_instructions():
    print_header("STEP A1: Train ArcFace on Colab (INSTRUCTIONS)")
    print("""
  Run these commands in a Colab notebook:

  Cell 1 — Install & Download:
    !pip install -q sentence-transformers torch torchvision scikit-learn
    !gdown --folder "1D6blD6g_kycN3Y__KjtSGj9zP_sHZaBL"

  Cell 2 — Upload arcface_metric_train.py to Colab, then run:
    !python arcface_metric_train.py --dataset "Ready Dataset" --epochs 40

  Cell 3 — Download results:
    from google.colab import files
    files.download("arcface_head.pt")
    files.download("prototypes.npz")

  Place both files in your project root:
    {root}/arcface_head.pt
    {root}/prototypes.npz
""".format(root=PROJECT_ROOT))
    input("  Press Enter once you have arcface_head.pt and prototypes.npz ready...")


def run_arcface_test(episode="s1e1"):
    print_header(f"STEP A2: Test Prototype Inference on {episode.upper()}")
    if not ARCFACE_HEAD.exists() or not PROTOTYPES.exists():
        print("  ❌ Missing arcface_head.pt or prototypes.npz. Complete Step A1 first.")
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prototype_inference.py"),
        "--episode", episode, "--force",
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_arcface_full(force=False):
    print_header("STEP A3: Full Prototype Inference on ALL Clips")
    if not ARCFACE_HEAD.exists() or not PROTOTYPES.exists():
        print("  ❌ Missing arcface_head.pt or prototypes.npz. Complete Step A1 first.")
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prototype_inference.py"),
    ]
    if force:
        cmd.append("--force")
    print(f"  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


# ── Pipeline B: YOLO + k-NN Hybrid (Legacy) ─────────────────────────────────

def run_knn_build():
    print_header("[Legacy] Build k-NN Reference Database")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "clip_classifier_knn.py"),
        "build-ref", "--dataset", "Ready Dataset",
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_yolo_test(episode="s1e1"):
    print_header(f"[Legacy] Test YOLO Hybrid Inference on {episode.upper()}")
    if not YOLO_WEIGHTS.exists():
        print("  ❌ YOLO weights not found at yolo_wt/best.pt")
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "YOLO_hybrid_inference.py"),
        "--weights", str(YOLO_WEIGHTS),
        "--episode", episode, "--force",
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_yolo_full(force=False, yolo_only=False):
    print_header("[Legacy] Full YOLO Hybrid Inference on ALL Clips")
    if not YOLO_WEIGHTS.exists():
        print("  ❌ YOLO weights not found at yolo_wt/best.pt")
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "YOLO_hybrid_inference.py"),
        "--weights", str(YOLO_WEIGHTS),
    ]
    if force:
        cmd.append("--force")
    if yolo_only:
        cmd.append("--yolo-only")
    print(f"  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


# ── Main Menu ────────────────────────────────────────────────────────────────

def main():
    print_header("BEN 10 VISUAL CHARACTER TAGGING PIPELINE")
    check_prerequisites()

    while True:
        print("\n" + "-" * 70)
        print("PIPELINE A — ArcFace Metric Learning (RECOMMENDED):")
        print("  1. [Step A1] Show Colab Training Instructions")
        print("  2. [Step A2] Test Prototype Inference on s1e1")
        print("  3. [Step A3] Sanity Check s1e1 Results")
        print("  4. [Step A4] Run Full Prototype Inference on ALL Clips")
        print()
        print("PIPELINE B — YOLO + k-NN Hybrid (Legacy):")
        print("  5. [Step B1] Build k-NN Reference Database")
        print("  6. [Step B2] Test YOLO Hybrid on s1e1")
        print("  7. [Step B3] Run Full YOLO Hybrid on ALL Clips")
        print()
        print("ADVANCED:")
        print("  8. Force re-run Prototype Inference (--force)")
        print("  9. Force re-run YOLO Hybrid (--force)")
        print("  0. Exit")
        print("-" * 70)

        choice = input("\nEnter choice (0-9) [Recommended: 1 → 2 → 3 → 4]: ").strip()

        if choice == "1":
            run_arcface_colab_instructions()
        elif choice == "2":
            run_arcface_test()
        elif choice == "3":
            sanity_check()
        elif choice == "4":
            confirm = input("  Ready to process ALL clips? (y/n) [y]: ").strip().lower()
            if confirm in ["y", "yes", ""]:
                run_arcface_full()
        elif choice == "5":
            run_knn_build()
        elif choice == "6":
            run_yolo_test()
        elif choice == "7":
            confirm = input("  Process ALL clips with YOLO? (y/n) [y]: ").strip().lower()
            if confirm in ["y", "yes", ""]:
                run_yolo_full()
        elif choice == "8":
            run_arcface_full(force=True)
        elif choice == "9":
            run_yolo_full(force=True)
        elif choice == "0":
            print("\nExiting. Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
