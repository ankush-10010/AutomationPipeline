"""
config_loader.py — Shared utility module for the AI Explainer pipeline.

Loads pipeline and show configuration, resolves paths, and sets up logging.
All other pipeline scripts import from this module.
"""

import os
import sys
import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root: one level up from the scripts/ directory
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

CONFIG_DIR = PROJECT_ROOT / "config"
PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline_config.yaml"
SHOW_CONFIG_PATH = CONFIG_DIR / "show_config.yaml"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(name: str = "ai_explainer", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format for all pipeline scripts."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------
def load_pipeline_config() -> Dict[str, Any]:
    """Load and return the pipeline configuration dictionary."""
    if not PIPELINE_CONFIG_PATH.exists():
        log.error("Pipeline config not found: %s", PIPELINE_CONFIG_PATH)
        sys.exit(1)
    with open(PIPELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    log.debug("Loaded pipeline config from %s", PIPELINE_CONFIG_PATH)
    return config


def load_show_config() -> Dict[str, Any]:
    """Load and return the show configuration dictionary."""
    if not SHOW_CONFIG_PATH.exists():
        log.error("Show config not found: %s", SHOW_CONFIG_PATH)
        sys.exit(1)
    with open(SHOW_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    log.debug("Loaded show config from %s", SHOW_CONFIG_PATH)
    return config


def get_active_show(show_slug: Optional[str] = None) -> tuple:
    """
    Return (slug, show_dict) for the requested show.

    If *show_slug* is None, returns the first show marked ``active: true``.
    Exits with an error if no matching active show is found.
    """
    show_cfg = load_show_config()
    shows: Dict[str, Any] = show_cfg.get("shows", {})

    if show_slug:
        if show_slug not in shows:
            log.error("Show '%s' not found in show_config.yaml", show_slug)
            sys.exit(1)
        show = shows[show_slug]
        if not show.get("active", False):
            log.warning("Show '%s' is marked inactive — proceeding anyway", show_slug)
        return show_slug, show

    # Default: first active show
    for slug, show in shows.items():
        if show.get("active", False):
            log.info("Auto-selected active show: %s", slug)
            return slug, show

    log.error("No active shows found in show_config.yaml")
    sys.exit(1)


def get_project_path(key: str, pipeline_config: Optional[Dict] = None) -> Path:
    """
    Resolve a path key from ``paths`` in pipeline_config.yaml against the
    project root.  Returns an absolute ``Path`` object.

    Example:
        get_project_path("topics_queue")
        → <PROJECT_ROOT>/topics/queue.json
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()
    paths: Dict[str, str] = pipeline_config.get("paths", {})
    raw = paths.get(key)
    if raw is None:
        log.error("Path key '%s' not found in pipeline_config.yaml [paths]", key)
        sys.exit(1)
    resolved = (PROJECT_ROOT / raw).resolve()
    return resolved


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Load a JSON file and return its contents. Returns [] if file is empty."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        return json.loads(text)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write *data* as formatted JSON to *path*, creating dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    log.debug("Saved JSON → %s", path)


def load_text(path: Path) -> str:
    """Read and return the full text content of a file, or '' if missing."""
    if not path.exists():
        log.warning("File not found (returning empty): %s", path)
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
