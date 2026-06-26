"""
metadata_generator.py — Generate CTR-optimized YouTube upload metadata.
=======================================================================
Reads an approved script and uses Ollama to generate 3 viral CTR Titles,
an SEO Description, and high-traffic Hashtag stacks.
"""

import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import setup_logging
from script_generator import call_ollama

log = setup_logging("metadata_generator")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def generate_upload_metadata(
    script_path: Path,
    script_text: str,
    show: Dict[str, Any],
    pipeline_config: Dict[str, Any],
) -> Path:
    """Generate and save upload metadata (.metadata.txt) alongside the approved script."""
    show_name = show.get("display_name", "the show")
    show_slug = show.get("slug", "cartoon").replace("_", "")

    prompt_file = PROMPTS_DIR / "metadata_prompt.txt"
    if not prompt_file.exists():
        log.error("Metadata prompt file missing: %s", prompt_file)
        return script_path

    template = prompt_file.read_text(encoding="utf-8")
    prompt = template.format(
        show_name=show_name,
        show_slug=show_slug,
        script=script_text,
    )

    log.info("Generating SEO upload metadata for: %s", script_path.name)
    raw_metadata = call_ollama(prompt, pipeline_config)

    if not raw_metadata.strip():
        log.warning("Ollama returned empty metadata — skipping SEO generation")
        return script_path

    meta_path = script_path.with_suffix(".metadata.txt")
    meta_path.write_text(raw_metadata.strip() + "\n", encoding="utf-8")
    log.info("Upload metadata saved → %s", meta_path)
    return meta_path
