"""
topic_miner.py — Phase 1a: Topic Mining for the AI Explainer pipeline.

Generates new video topic ideas using Ollama LLM, reads show context from
config, and appends results to the topics queue.
"""

import argparse
import json
import sys
import time
import requests
from pathlib import Path

from config_loader import (
    setup_logging,
    load_pipeline_config,
    get_active_show,
    get_project_path,
    load_json,
    save_json,
    load_text,
    PROJECT_ROOT,
)

log = setup_logging("topic_miner")


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------
def _format_character_details(characters: list) -> str:
    """Format the character list into a readable string for the prompt."""
    lines = []
    for c in characters:
        aliases = ", ".join(c.get("aliases", []))
        lines.append(f"- {c['name']} (aliases: {aliases}): {c.get('description', '')}")
    return "\n".join(lines)


def _load_completed_topics(completed_dir: Path) -> str:
    """
    Scan the completed/ directory for finished topic files and build an
    exclusion list string.  Each .txt or .json file name (sans extension)
    is treated as a completed topic.
    """
    if not completed_dir.exists():
        return "None yet — this is the first batch."

    topics = []
    for f in sorted(completed_dir.iterdir()):
        if f.is_file():
            # Try reading JSON files for structured topic data
            if f.suffix == ".json":
                try:
                    data = load_json(f)
                    if isinstance(data, dict) and "topic" in data:
                        topics.append(data["topic"])
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "topic" in item:
                                topics.append(item["topic"])
                    continue
                except Exception:
                    pass
            # Fall back to filename-based topic extraction
            topics.append(f.stem.replace("_", " ").replace("-", " "))

    if not topics:
        return "None yet — this is the first batch."

    return "\n".join(f"- {t}" for t in topics)


def build_topic_prompt(
    show: dict,
    num_topics: int,
    pipeline_config: dict,
) -> str:
    """
    Load the topic prompt template and fill in all placeholders from the
    show config and episode data.
    """
    prompts_dir = get_project_path("prompts_dir", pipeline_config)
    template_path = prompts_dir / "topic_prompt.txt"
    template = load_text(template_path)

    if not template:
        log.error("Topic prompt template is empty or missing: %s", template_path)
        sys.exit(1)

    # Show context from episode data file
    show_context = ""
    episode_file_rel = show.get("episode_data_file", "")
    if episode_file_rel:
        episode_path = (PROJECT_ROOT / episode_file_rel).resolve()
        show_context = load_text(episode_path)
        if show_context:
            log.info("Loaded episode data (%d chars) from %s", len(show_context), episode_path)
        else:
            log.warning("Episode data file not found or empty: %s", episode_path)
            show_context = "No episode data available — generate topics from general show knowledge."

    # Character details
    characters = show.get("characters", [])
    character_details = _format_character_details(characters)

    # Themes
    themes = ", ".join(show.get("themes", []))

    # Completed topics (exclusions)
    completed_dir = get_project_path("topics_completed", pipeline_config)
    completed_topics = _load_completed_topics(completed_dir)

    # Fill placeholders
    prompt = template.format(
        show_name=show.get("display_name", "Unknown Show"),
        show_context=show_context,
        character_details=character_details,
        themes=themes,
        num_topics=num_topics,
        completed_topics=completed_topics,
    )

    return prompt


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------
def call_ollama(prompt: str, pipeline_config: dict) -> str:
    """
    Send a prompt to the Ollama /api/generate endpoint and return the
    full response text.
    """
    llm = pipeline_config.get("llm", {})
    base_url = llm.get("base_url", "http://localhost:11434").rstrip("/")
    model = llm.get("model", "llama3.1:8b")
    timeout = llm.get("timeout_seconds", 300)
    temperature = llm.get("temperature", 0.8)

    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": llm.get("max_tokens", 1024),
        },
    }

    log.info("Calling Ollama → %s (model: %s)", url, model)
    log.debug("Prompt length: %d chars", len(prompt))

    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.ConnectionError:
        log.error(
            "Cannot connect to Ollama at %s — is it running? "
            "Start with: ollama serve",
            base_url,
        )
        sys.exit(1)
    except requests.Timeout:
        log.error("Ollama request timed out after %ds", timeout)
        sys.exit(1)
    except requests.HTTPError as e:
        log.error("Ollama returned HTTP error: %s", e)
        sys.exit(1)

    elapsed = time.time() - start
    result = resp.json()
    response_text = result.get("response", "")
    log.info("Ollama responded in %.1fs (%d chars)", elapsed, len(response_text))

    return response_text


# ---------------------------------------------------------------------------
# Parse topics from LLM response
# ---------------------------------------------------------------------------
def parse_topics(raw_response: str) -> list:
    """
    Extract a JSON array of topic objects from the LLM response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        log.error("Could not find JSON array in LLM response")
        log.debug("Raw response:\n%s", raw_response)
        return []

    json_str = text[start : end + 1]

    try:
        topics = json.loads(json_str)
    except json.JSONDecodeError as e:
        log.error("Failed to parse topics JSON: %s", e)
        log.debug("Attempted to parse:\n%s", json_str[:500])
        return []

    if not isinstance(topics, list):
        log.error("Expected JSON array, got %s", type(topics).__name__)
        return []

    # Validate each topic has required fields
    valid_topics = []
    for t in topics:
        if isinstance(t, dict) and "topic" in t:
            valid_topics.append(t)
        else:
            log.warning("Skipping malformed topic entry: %s", t)

    log.info("Parsed %d valid topics from LLM response", len(valid_topics))
    return valid_topics


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------
def append_to_queue(new_topics: list, pipeline_config: dict) -> int:
    """
    Append new topics to the queue.json file without overwriting existing
    ones.  Returns the number of actually-new topics added (deduped against
    existing queue entries).
    """
    queue_path = get_project_path("topics_queue", pipeline_config)
    existing = load_json(queue_path)

    # Deduplicate by topic text (case-insensitive)
    existing_set = {t["topic"].lower() for t in existing if isinstance(t, dict) and "topic" in t}
    added = 0

    for topic in new_topics:
        if topic["topic"].lower() not in existing_set:
            existing.append(topic)
            existing_set.add(topic["topic"].lower())
            added += 1
        else:
            log.debug("Skipping duplicate topic: %s", topic["topic"][:60])

    save_json(queue_path, existing)
    log.info(
        "Queue updated: +%d new topics (%d total in queue)",
        added,
        len(existing),
    )
    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1a — Mine video topics using Ollama LLM",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="Show slug from show_config.yaml (default: first active show)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of topics to generate (default: 10)",
    )
    args = parser.parse_args()

    pipeline_config = load_pipeline_config()
    slug, show = get_active_show(args.show)
    log.info("=== Topic Mining for '%s' (%d topics) ===", show["display_name"], args.count)

    # Build prompt
    prompt = build_topic_prompt(show, args.count, pipeline_config)

    # Call LLM
    raw = call_ollama(prompt, pipeline_config)

    # Parse response
    topics = parse_topics(raw)
    if not topics:
        log.error("No topics could be extracted. Check Ollama output above.")
        sys.exit(1)

    # Add to queue
    added = append_to_queue(topics, pipeline_config)

    log.info("✓ Done — %d new topics added to queue", added)


if __name__ == "__main__":
    main()
