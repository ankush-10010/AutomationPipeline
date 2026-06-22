import json
import subprocess
import time
import sys
from pathlib import Path

def main():
    queue_path = Path("topics/queue.json")
    if not queue_path.exists():
        print("Queue file not found at topics/queue.json!")
        sys.exit(1)

    try:
        with open(queue_path, "r", encoding="utf-8") as f:
            queue = json.load(f)
    except Exception as e:
        print(f"Error reading queue.json: {e}")
        sys.exit(1)

    if not queue:
        print("Queue is empty. Use the topic_mine phase to add more topics!")
        sys.exit(0)

    print(f"Found {len(queue)} topics in queue. Starting continuous automation...")

    # Process topics one by one
    while queue:
        entry = queue[0]
        topic = entry.get("topic") if isinstance(entry, dict) else str(entry)
        
        print(f"\n" + "="*70)
        print(f"🚀 STARTING END-TO-END PIPELINE FOR: {topic}")
        print("="*70 + "\n")
        
        # VERY IMPORTANT: Delete old pipeline_state.json to ensure a fresh run
        state_path = Path("pipeline_state.json")
        if state_path.exists():
            state_path.unlink()
            
        # Call the orchestrator in fully automatic mode
        cmd = [
            sys.executable, "scripts/orchestrator_noImage_gpuVoice.py",
            "--topic", topic,
            "--phase", "all",
            "--auto-approve"
        ]
        
        try:
            result = subprocess.run(cmd)
            
            # Check if orchestrator exited cleanly
            if result.returncode != 0:
                print(f"\n❌ Pipeline failed or was paused for '{topic}'. Stopping automation.")
                print(f"You can resume manually later, or fix the error and restart.")
                break
                
            print(f"\n✅ Video pipeline successfully completed for: '{topic}'")
            
            # Remove the completed topic from the queue and save
            queue.pop(0)
            with open(queue_path, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=4)
                
            print(f"Queue updated. {len(queue)} topics remaining.")
            
            if queue:
                print("Taking a 5-second breather before starting the next video...")
                time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n⚠️ Automation interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error running orchestrator: {e}")
            break

    if not queue:
        print("\n🎉 All tasks in the queue have been completed!")

if __name__ == "__main__":
    main()
