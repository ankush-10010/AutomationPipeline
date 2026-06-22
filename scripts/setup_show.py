"""
setup_show.py — Automates setting up a new show in the AI Explainer pipeline.
Generates the show_config.yaml entry using Ollama and fetches episode summaries from Wikipedia.
"""

import argparse
import json
import re
import sys
import yaml
import requests
from pathlib import Path

from config_loader import load_pipeline_config, get_project_path, PROJECT_ROOT

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    return text

def fetch_episode_summaries(show_name: str) -> str:
    """Attempts to fetch the 'List of [Show Name] episodes' page from Wikipedia."""
    print(f"Fetching episode summaries from Wikipedia for '{show_name}'...")
    title = f"List_of_{show_name.replace(' ', '_')}_episodes"
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={title}&format=json"
    
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if page_id != "-1":
                return page_data.get('extract', '')
    except Exception as e:
        print(f"Error fetching from Wikipedia: {e}")
        
    print(f"Fallback: Fetching main '{show_name}' page from Wikipedia...")
    title = show_name.replace(' ', '_')
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={title}&format=json"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if page_id != "-1":
                return page_data.get('extract', '')
    except Exception as e:
        print(f"Error fetching fallback from Wikipedia: {e}")
        
    return ""

def generate_show_config(show_name: str, pipeline_config: dict) -> dict:
    """Uses Ollama to generate the characters, themes, and prompt tuning for the show."""
    llm = pipeline_config.get("llm", {})
    base_url = llm.get("base_url", "http://localhost:11434").rstrip("/")
    model = llm.get("model", "llama3.1:8b")
    
    prompt = f"""
You are configuring an AI video generator for the TV show "{show_name}".
I need you to output a raw JSON object containing the show's configuration.
Do not output any markdown code blocks, just raw JSON.

The JSON format must be EXACTLY:
{{
  "display_name": "{show_name}",
  "characters": [
    {{"name": "Character 1", "aliases": ["Alias1"], "description": "Short description"}},
    {{"name": "Character 2", "aliases": ["Alias2"], "description": "Short description"}}
  ],
  "locations": ["location 1", "location 2", "location 3"],
  "themes": ["theme 1", "theme 2", "theme 3"],
  "prompt_tuning": {{
    "narrator_style": "confident, insightful, engaging video essayist",
    "avoid_phrases": ["buckle up", "mind-blowing", "let's dive in", "without further ado", "in this video"],
    "reference_style": "Reference specific episodes by name or description rather than by season/episode number"
  }}
}}

Generate this for the show "{show_name}". Provide at least 5 major characters, 5 locations, and 5 themes. Tailor the narrator_style to fit the specific tone, humor, and vibe of "{show_name}".
"""
    
    print(f"Asking Ollama ({model}) to generate config for '{show_name}'...")
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 1024},
        "format": "json"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        
        # Clean markdown fences if any
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        return json.loads(text)
    except Exception as e:
        print(f"Failed to generate config via Ollama: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automate setting up a new show.")
    parser.add_argument("show_name", type=str, help="The full name of the show (e.g., 'Breaking Bad')")
    args = parser.parse_args()
    
    show_name = args.show_name
    slug = slugify(show_name)
    
    pipeline_config = load_pipeline_config()
    
    # 1. Fetch Wikipedia summaries
    summaries = fetch_episode_summaries(show_name)
    
    clips_dir = PROJECT_ROOT / "clips" / slug
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    episodes_file = clips_dir / "episode_summaries.txt"
    if summaries:
        episodes_file.write_text(summaries, encoding="utf-8")
        print(f"✓ Saved episode summaries to: {episodes_file}")
    else:
        print("⚠️ Could not fetch summaries from Wikipedia. Created empty file.")
        episodes_file.write_text("No data found. Please paste episode summaries here manually.", encoding="utf-8")
        
    # 2. Generate config via Ollama
    generated_data = generate_show_config(show_name, pipeline_config)
    
    # Add our local file paths to the generated data
    generated_data["active"] = True
    generated_data["episode_data_file"] = f"./clips/{slug}/episode_summaries.txt"
    generated_data["clips_dir"] = f"./clips/{slug}"
    
    # 3. Append to show_config.yaml
    config_file = PROJECT_ROOT / "config" / "show_config.yaml"
    
    # Format generated dict as YAML string
    new_yaml_str = yaml.dump({slug: generated_data}, default_flow_style=False, sort_keys=False)
    
    # Append to file
    with open(config_file, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(f"# --- Auto-generated configuration for {show_name} ---\n")
        # Add indentation for 'shows:'
        for line in new_yaml_str.strip().split('\n'):
            f.write(f"  {line}\n")
            
    print(f"✓ Appended '{show_name}' configuration to: {config_file}")
    print(f"\n🎉 You're all set! To mine topics for this show, run:")
    print(f"   python scripts/orchestrator_noImage_gpuVoice.py --show {slug} --phase topic_mine")

if __name__ == "__main__":
    main()
