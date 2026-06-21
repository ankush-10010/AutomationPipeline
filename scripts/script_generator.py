"""
script_generator.py — Phase 1b: Script Generation for the AI Explainer pipeline.

Takes topics (from queue.json or CLI) and generates narration scripts using
Ollama LLM, saving output to the approved/ directory.
"""

import argparse
import re
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

log = setup_logging("script_generator")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sanitize_filename(topic: str, max_length: int = 80) -> str:
    """
    Convert a topic string into a safe, readable filename.
    Example: "Why did Rick destroy the Citadel?" → "why_did_rick_destroy_the_citadel"
    """
    # Lowercase and strip
    name = topic.lower().strip()
    # Remove non-alphanumeric chars (keep spaces and hyphens)
    name = re.sub(r"[^\w\s-]", "", name)
    # Collapse whitespace / hyphens to underscores
    name = re.sub(r"[\s-]+", "_", name)
    # Trim to max length
    name = name[:max_length].rstrip("_")
    return name


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------
def build_script_prompt(
    topic: str,
    show: dict,
    pipeline_config: dict,
) -> str:
    """
    Load the script prompt template and fill placeholders from the show
    config and the given topic.
    """
    prompts_dir = get_project_path("prompts_dir", pipeline_config)
    template_path = prompts_dir / "script_prompt.txt"
    template = load_text(template_path)

    if not template:
        log.error("Script prompt template is empty or missing: %s", template_path)
        sys.exit(1)

    prompt_tuning = show.get("prompt_tuning", {})
    narrator_style = prompt_tuning.get("narrator_style", "engaging and informative")
    reference_style = prompt_tuning.get("reference_style", "reference specific episodes when possible")
    avoid_phrases = prompt_tuning.get("avoid_phrases", [])
    avoid_str = ", ".join(f'"{p}"' for p in avoid_phrases)

    prompt = template.format(
        show_name=show.get("display_name", "Unknown Show"),
        narrator_style=narrator_style.strip(),
        topic=topic,
        reference_style=reference_style.strip(),
        avoid_phrases=avoid_str,
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
# Script saving
# ---------------------------------------------------------------------------
def save_script(topic: str, script_text: str, pipeline_config: dict) -> Path:
    """
    Save the generated script to topics/approved/{sanitized_name}.txt.
    Returns the path to the saved file.
    """
    approved_dir = get_project_path("topics_approved", pipeline_config)
    approved_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(topic) + ".txt"
    out_path = approved_dir / filename

    # Avoid overwriting — append a counter if the file exists
    if out_path.exists():
        counter = 1
        while out_path.exists():
            out_path = approved_dir / f"{sanitize_filename(topic)}_{counter}.txt"
            counter += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script_text.strip())
        f.write("\n")

    log.info("Script saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Process a single topic
# ---------------------------------------------------------------------------
def generate_script_for_topic(
    topic: str,
    show: dict,
    pipeline_config: dict,
) -> Path:
    """Build prompt, call LLM, save script. Returns the output file path."""
    log.info("Generating script for: %s", topic)

    prompt = build_script_prompt(topic, show, pipeline_config)
    script_text = call_ollama(prompt, pipeline_config)

    if not script_text.strip():
        log.error("Ollama returned an empty response for topic: %s", topic)
        sys.exit(1)

    return save_script(topic, script_text, pipeline_config)


# ---------------------------------------------------------------------------
# Batch processing from queue
# ---------------------------------------------------------------------------
def process_batch(
    batch_size: int,
    show: dict,
    pipeline_config: dict,
) -> list:
    """
    Pop up to *batch_size* topics from queue.json, generate scripts for each,
    and remove processed topics from the queue.  Returns a list of output paths.
    """
    queue_path = get_project_path("topics_queue", pipeline_config)
    queue = load_json(queue_path)

    if not queue:
        log.warning("Queue is empty — nothing to process")
        return []

    to_process = queue[:batch_size]
    remaining = queue[batch_size:]

    log.info("Batch processing %d topics (queue has %d total)", len(to_process), len(queue))
    output_paths = []

    for i, entry in enumerate(to_process, 1):
        topic_text = entry.get("topic", "") if isinstance(entry, dict) else str(entry)
        if not topic_text:
            log.warning("Skipping entry with no topic text: %s", entry)
            continue

        log.info("--- [%d/%d] ---", i, len(to_process))
        try:
            path = generate_script_for_topic(topic_text, show, pipeline_config)
            output_paths.append(path)
        except SystemExit:
            log.error("Failed on topic: %s — skipping", topic_text[:60])
            # Put it back into remaining so we don't lose it
            remaining.insert(0, entry)

    # Update the queue with remaining topics
    save_json(queue_path, remaining)
    log.info("Queue updated: %d topics remaining", len(remaining))

    return output_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1b — Generate narration scripts using Ollama LLM",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Single topic string to generate a script for",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help="Process N topics from queue.json",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="Show slug from show_config.yaml (default: first active show)",
    )
    args = parser.parse_args()

    if args.topic is None and args.batch is None:
        parser.error("Provide either --topic 'some topic' or --batch N")

    pipeline_config = load_pipeline_config()
    slug, show = get_active_show(args.show)
    log.info("=== Script Generation for '%s' ===", show["display_name"])

    if args.topic:
        # Single topic mode
        path = generate_script_for_topic(args.topic, show, pipeline_config)
        log.info("✓ Done — script saved to %s", path)

    elif args.batch:
        # Batch mode
        paths = process_batch(args.batch, show, pipeline_config)
        log.info("✓ Done — generated %d scripts", len(paths))
        for p in paths:
            log.info("  → %s", p)


if __name__ == "__main__":
    main()
