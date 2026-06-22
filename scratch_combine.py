import json
import os
from pathlib import Path

# Paths
root = Path(r"C:\CODE\automation\ai_explainer")
wiki_path = root / "topics" / "wiki.json"
theories_path = root / "topics" / "theories.json"
output_dir = root / "clips" / "rick_and_morty"
output_path = output_dir / "episode_summaries.txt"

output_dir.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as out:
    out.write("==== RICK AND MORTY LORE & SUMMARIES ====\n\n")
    
    # Write Wiki data
    if wiki_path.exists():
        out.write("--- CHARACTER LORE & GENERAL WIKI ---\n\n")
        with open(wiki_path, "r", encoding="utf-8") as f:
            try:
                wiki_data = json.load(f)
                for title, text in wiki_data.items():
                    out.write(f"[{title}]\n{text}\n\n")
            except json.JSONDecodeError:
                pass
                
    # Write Theories data
    if theories_path.exists():
        out.write("--- THEORIES & EPISODE SPECULATION ---\n\n")
        with open(theories_path, "r", encoding="utf-8") as f:
            try:
                theories_data = json.load(f)
                for title, text in theories_data.items():
                    out.write(f"[{title}]\n{text}\n\n")
            except json.JSONDecodeError:
                pass

print(f"Successfully generated {output_path}")
