"""
process_queue.py — Continuously process topics from queue.json until empty.

Runs the orchestrator for each topic in auto-approve mode, removes completed
topics from the queue, and tracks progress. Each video's output goes into its
own per-topic folder under output/.

Usage:
    python scripts/process_queue.py
    python scripts/process_queue.py --skip-thumbnail
"""

import argparse
import json
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path


def load_queue(queue_path: Path) -> list:
    """Load the topic queue from disk."""
    if not queue_path.exists():
        return []
    with open(queue_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_queue(queue_path: Path, queue: list) -> None:
    """Save the topic queue to disk."""
    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=4, ensure_ascii=False)


def read_topic_folder_from_state(state_path: Path) -> str:
    """Read the topic folder path from the pipeline state file."""
    if not state_path.exists():
        return ""
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state.get("phase_outputs", {}).get("topic_folder", "")
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Process all topics in queue.json through the full pipeline."
    )
    parser.add_argument(
        "--skip-thumbnail", action="store_true",
        help="Skip thumbnail generation phase (faster)."
    )
    args = parser.parse_args()

    queue_path = Path("topics/queue.json")
    state_path = Path("pipeline_state.json")

    queue = load_queue(queue_path)
    if not queue:
        print("Queue is empty. Use the topic_mine phase to add more topics!")
        sys.exit(0)

    total = len(queue)
    completed = 0
    failed = 0
    start_time = datetime.now()

    print(f"\n{'='*70}")
    print(f"📋 QUEUE PROCESSOR — {total} topics to process")
    print(f"   Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    while queue:
        entry = queue[0]
        topic = entry.get("topic") if isinstance(entry, dict) else str(entry)
        remaining = len(queue)

        print(f"\n{'='*70}")
        print(f"🚀 [{completed + 1}/{total}] STARTING: {topic}")
        print(f"   Remaining in queue: {remaining}")
        print(f"{'='*70}\n")

        # Delete old pipeline_state.json to ensure a fresh run
        if state_path.exists():
            state_path.unlink()

        # Build the orchestrator command
        cmd = [
            sys.executable, "scripts/orchestrator_noImage_gpuVoice.py",
            "--topic", topic,
            "--phase", "all",
            "--auto-approve",
        ]

        try:
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"\n❌ Pipeline failed for: '{topic}'")
                failed += 1
                # Remove failed topic and continue with the next one
                queue.pop(0)
                save_queue(queue_path, queue)
                print(f"   Skipping to next topic. {len(queue)} remaining.")
                continue

            completed += 1

            # Read the topic folder from the state
            topic_folder = read_topic_folder_from_state(state_path)

            print(f"\n✅ [{completed}/{total}] COMPLETED: '{topic}'")
            if topic_folder:
                print(f"   📁 Output: {topic_folder}")

            # Remove the completed topic from the queue and save
            queue.pop(0)
            save_queue(queue_path, queue)
            print(f"   Queue: {len(queue)} topics remaining.")

            if queue:
                print("   ⏳ 5-second pause before next video...")
                time.sleep(5)

        except KeyboardInterrupt:
            print(f"\n⚠️ Automation interrupted by user at topic #{completed + 1}.")
            print(f"   {len(queue)} topics still in queue.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            failed += 1
            queue.pop(0)
            save_queue(queue_path, queue)
            continue

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"🏁 QUEUE PROCESSING COMPLETE")
    print(f"   Total:     {total}")
    print(f"   Completed: {completed}")
    print(f"   Failed:    {failed}")
    print(f"   Skipped:   {total - completed - failed}")
    print(f"   Duration:  {elapsed}")
    print(f"{'='*70}\n")

    if not queue:
        print("🎉 All topics in the queue have been processed!")


if __name__ == "__main__":
    main()
