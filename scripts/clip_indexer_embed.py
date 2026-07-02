import json
import argparse
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers is not installed.")
    print("Please run: pip install sentence-transformers")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate semantic embeddings for clip_index.json")
    parser.add_argument("--index", default="clip_index.json", help="Path to clip_index.json")
    parser.add_argument("--force", action="store_true", help="Force recompute embeddings")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: Could not find {index_path}")
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
    print(f"\n{color}🚀 SentenceTransformers is running on: {device_name}\033[0m\n")

    print("Loading SentenceTransformer model (this will download ~80MB the very first time)...")
    # all-MiniLM-L6-v2 is incredibly fast on CPU and highly accurate for sentence matching
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Generating semantic embeddings for {len(clips)} clips...")
    
    count = 0
    for clip in clips:
        if "embedding" in clip and not args.force:
            continue
            
        # Create a rich text representation of the clip
        chars = ", ".join(clip.get("characters", []))
        action = clip.get("action", "")
        
        text_to_embed = f"Characters: {chars}. Dialogue/Action: {action}"
        
        # Convert text into a 384-dimensional mathematical vector
        vector = model.encode(text_to_embed).tolist()
        clip["embedding"] = vector
        count += 1
        
        if count % 50 == 0:
            print(f"  Embedded {count}/{len(clips)} clips...")

    # Save the embeddings directly back into the JSON
    print("Saving updated clip_index.json...")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print("✅ Successfully added semantic embeddings!")

if __name__ == "__main__":
    main()
