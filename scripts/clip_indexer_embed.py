import json
import argparse
from pathlib import Path
from tqdm import tqdm

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
    pbar = tqdm(clips, desc="Embedding Sentences", unit="clip")
    for clip in pbar:
        if "embedding" in clip and not args.force:
            continue
            
        parts = []
        
        chars = clip.get("characters")
        if chars and isinstance(chars, list):
            parts.append(f"Characters: {', '.join(str(c) for c in chars if c)}")
            
        emotion = clip.get("emotion_tone")
        if emotion and isinstance(emotion, str):
            parts.append(f"Emotion/Tone: {emotion}")
            
        visual = clip.get("visual_description")
        if visual and isinstance(visual, str):
            parts.append(f"Visual: {visual}")
            
        context = clip.get("scene_context")
        if context and isinstance(context, str):
            parts.append(f"Context: {context}")
            
        action = clip.get("action")
        if action and isinstance(action, str):
            parts.append(f"Action/Dialogue: {action}")
            
        tags = clip.get("tags")
        if tags and isinstance(tags, list):
            parts.append(f"Tags: {', '.join(str(t) for t in tags if t)}")
            
        text_to_embed = ". ".join(parts)
        
        vector = model.encode(text_to_embed).tolist()
        clip["embedding"] = vector
        count += 1

    # Save the embeddings directly back into the JSON
    print("Saving updated clip_index.json...")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("✅ Successfully added semantic embeddings!")

if __name__ == "__main__":
    main()
