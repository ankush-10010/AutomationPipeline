"""
publisher.py — Phase 7: Upload videos to YouTube via the Data API v3.

Features:
  • OAuth 2.0 authentication (first-time interactive + token caching/refresh)
  • Resumable upload with progress logging
  • LLM-powered metadata generation (title / description / tags) via Ollama
  • Scheduled publishing (private + publishAt timestamp)

CLI usage:
  python publisher.py --video output/final.mp4 --script output/script.txt
  python publisher.py --video output/final.mp4 --title "My Title" --privacy unlisted
  python publisher.py --video output/final.mp4 --script output/script.txt --schedule 2026-07-01T14:00:00+05:30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dateutil import parser as dt_parser

# ---------------------------------------------------------------------------
# Ensure sibling imports work when run directly
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config_loader import (
    setup_logging,
    load_pipeline_config,
    load_text,
    PROJECT_ROOT,
)

log = setup_logging("publisher")

# YouTube API scopes
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

# Retry settings for resumable upload
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds, doubled each retry


# ---------------------------------------------------------------------------
# OAuth 2.0 helpers
# ---------------------------------------------------------------------------

def _get_authenticated_service(pub_cfg: Dict[str, Any]):
    """
    Build and return an authenticated ``youtube`` API resource.

    Flow:
      1. Try to load cached token from *token_file*.
      2. If valid → use it.  If expired → refresh it.
      3. If missing / refresh fails → run InstalledAppFlow interactively.
      4. Cache the token for future runs.
    """
    # Lazy imports — these are heavy and only needed here
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds_path = Path(pub_cfg["credentials_file"])
    token_path = Path(pub_cfg["token_file"])

    creds: Optional[Credentials] = None

    # 1. Try cached token
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            log.info("Loaded cached token from %s", token_path)
        except Exception as exc:
            log.warning("Failed to load cached token: %s", exc)
            creds = None

    # 2. Refresh or re-auth
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            log.info("Token refreshed successfully")
        except Exception as exc:
            log.warning("Token refresh failed (%s) — re-authenticating", exc)
            creds = None

    if not creds or not creds.valid:
        if not creds_path.exists():
            log.error(
                "YouTube credentials file not found: %s\n"
                "Download it from Google Cloud Console → APIs & Services → Credentials → "
                "OAuth 2.0 Client ID → Download JSON.",
                creds_path,
            )
            raise FileNotFoundError(f"Credentials file missing: {creds_path}")

        flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
        creds = flow.run_local_server(port=0)
        log.info("OAuth consent completed — new token acquired")

    # 3. Cache token
    token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    log.debug("Token cached → %s", token_path)

    return build("youtube", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# LLM metadata generation
# ---------------------------------------------------------------------------

_METADATA_PROMPT = """\
You are a YouTube Shorts metadata expert.

Given the following narration script for a short-form video, generate:
1. **title** — an attention-grabbing title, max 100 characters, no clickbait
2. **description** — 2-3 sentences summarising the video, include 3-5 relevant #hashtags at the end
3. **tags** — a JSON list of 8-15 keyword strings for discoverability

{show_context}

Script:
---
{script_text}
---

Respond with ONLY valid JSON (no markdown fences):
{{"title": "...", "description": "...", "tags": ["tag1", "tag2", ...]}}
"""


def generate_metadata(
    script_text: str,
    llm_cfg: Dict[str, Any],
    show_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call the local Ollama LLM to generate YouTube metadata from a script.

    Returns ``{"title": str, "description": str, "tags": list[str]}``.
    Falls back to sensible defaults if the LLM response cannot be parsed.
    """
    show_ctx = f"This video is about the show: {show_name}." if show_name else ""
    prompt = _METADATA_PROMPT.format(
        script_text=script_text,
        show_context=show_ctx,
    )

    base_url = llm_cfg.get("base_url", "http://localhost:11434").rstrip("/")
    payload = {
        "model": llm_cfg.get("model", "llama3.1:8b"),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": llm_cfg.get("temperature", 0.7),
            "num_predict": llm_cfg.get("max_tokens", 1024),
        },
    }

    log.info("Requesting metadata from Ollama (%s) …", payload["model"])

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=llm_cfg.get("timeout_seconds", 300),
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except Exception as exc:
        log.error("Ollama request failed: %s", exc)
        return _fallback_metadata(script_text)

    # Parse JSON from LLM output — it might be wrapped in markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]  # drop opening fence line
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    try:
        meta = json.loads(raw)
        # Validate expected keys
        title = str(meta.get("title", ""))[:100]
        desc = str(meta.get("description", ""))
        tags = meta.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(t) for t in tags]
        log.info("LLM metadata generated — title: %s", title)
        return {"title": title, "description": desc, "tags": tags}
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("Failed to parse LLM metadata (%s) — using fallback", exc)
        return _fallback_metadata(script_text)


def _fallback_metadata(script_text: str) -> Dict[str, Any]:
    """Generate basic metadata from the first line of the script."""
    first_line = script_text.strip().split("\n")[0][:100] if script_text.strip() else "Untitled Video"
    return {
        "title": first_line,
        "description": script_text[:500] if script_text else "",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _upload_video(
    youtube,
    video_path: str,
    title: str,
    description: str,
    tags: List[str],
    category_id: str,
    privacy: str,
    language: str,
    made_for_kids: bool,
    notify_subscribers: bool,
    schedule_time: Optional[str] = None,
) -> str:
    """
    Perform a resumable upload and return the resulting video ID.
    """
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    body: Dict[str, Any] = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": language,
            "defaultAudioLanguage": language,
        },
        "status": {
            "privacyStatus": privacy,
            "madeForKids": made_for_kids,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }

    # Scheduled publishing
    if schedule_time:
        body["status"]["privacyStatus"] = "private"
        body["status"]["publishAt"] = schedule_time
        log.info("Video scheduled for publishing at %s", schedule_time)

    # Notify subscribers only when going public immediately
    if privacy == "public" and not schedule_time:
        body["status"]["notifySubscribers"] = notify_subscribers

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=1024 * 1024,  # 1 MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    log.info("Starting upload: %s (%s)", title, video_path)
    response = None
    retries = 0

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log.info("Upload progress: %d%%", pct)
        except HttpError as exc:
            if exc.resp.status in (500, 502, 503, 504) and retries < MAX_RETRIES:
                retries += 1
                wait = RETRY_BACKOFF ** retries
                log.warning(
                    "Server error %s — retry %d/%d in %ds",
                    exc.resp.status, retries, MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                raise

    video_id = response["id"]
    log.info("Upload complete — video ID: %s", video_id)
    log.info("URL: https://youtu.be/%s", video_id)
    return video_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def publish_video(
    video_path: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    privacy: Optional[str] = None,
    category_id: Optional[str] = None,
    schedule_time: Optional[str] = None,
    script_text: Optional[str] = None,
    show_name: Optional[str] = None,
) -> str:
    """
    Upload a video to YouTube.

    If *title*, *description*, or *tags* are not supplied and *script_text* is
    provided, they are generated via the local Ollama LLM.

    Returns the YouTube video ID on success.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    pipeline_cfg = load_pipeline_config()
    pub_cfg = pipeline_cfg.get("publishing", {})
    llm_cfg = pipeline_cfg.get("llm", {})

    # ---------- Metadata resolution ----------
    if script_text and (not title or not description or not tags):
        meta = generate_metadata(script_text, llm_cfg, show_name)
        title = title or meta["title"]
        description = description or meta["description"]
        tags = tags or meta["tags"]

    title = title or video_file.stem.replace("_", " ").title()
    description = description or ""
    tags = tags or []

    privacy = privacy or pub_cfg.get("default_privacy", "private")
    category_id = category_id or pub_cfg.get("default_category_id", "24")
    language = pub_cfg.get("default_language", "en")
    made_for_kids = pub_cfg.get("made_for_kids", False)
    notify_subs = pub_cfg.get("notify_subscribers", True)

    # ---------- Schedule handling ----------
    publish_at: Optional[str] = None
    if schedule_time:
        dt = dt_parser.isoparse(schedule_time)
        # Ensure UTC ISO 8601 with 'Z'
        publish_at = dt.astimezone().isoformat()
        if not publish_at.endswith("Z"):
            # Convert to UTC string ending in Z for YouTube API
            from datetime import timezone
            dt_utc = dt.astimezone(timezone.utc)
            publish_at = dt_utc.strftime("%Y-%m-%dT%H:%M:%S.0Z")

    # ---------- Auth + Upload ----------
    youtube = _get_authenticated_service(pub_cfg)

    video_id = _upload_video(
        youtube=youtube,
        video_path=str(video_file),
        title=title,
        description=description,
        tags=tags,
        category_id=category_id,
        privacy=privacy,
        language=language,
        made_for_kids=made_for_kids,
        notify_subscribers=notify_subs,
        schedule_time=publish_at,
    )

    return video_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 7 — Upload a video to YouTube with auto-generated metadata.",
    )
    parser.add_argument(
        "--video", required=True, help="Path to the video file to upload",
    )
    parser.add_argument(
        "--script",
        default=None,
        help="Path to the narration script text file (used to generate title/description/tags via LLM)",
    )
    parser.add_argument("--title", default=None, help="Override video title")
    parser.add_argument("--description", default=None, help="Override video description")
    parser.add_argument(
        "--tags",
        default=None,
        help="Override tags (comma-separated, e.g. 'rick,morty,science')",
    )
    parser.add_argument(
        "--schedule",
        default=None,
        help="Schedule publish time (ISO 8601 datetime, e.g. 2026-07-01T14:00:00+05:30)",
    )
    parser.add_argument(
        "--privacy",
        choices=["public", "private", "unlisted"],
        default=None,
        help="Privacy status (default from config)",
    )
    parser.add_argument(
        "--category", default=None, help="YouTube category ID (default from config)",
    )
    parser.add_argument(
        "--show", default=None, help="Show name for LLM context (e.g. 'Rick and Morty')",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    script_text = None
    if args.script:
        script_text = load_text(Path(args.script))
        if not script_text:
            log.warning("Script file is empty or missing: %s", args.script)

    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None

    try:
        vid = publish_video(
            video_path=args.video,
            title=args.title,
            description=args.description,
            tags=tag_list,
            privacy=args.privacy,
            category_id=args.category,
            schedule_time=args.schedule,
            script_text=script_text,
            show_name=args.show,
        )
        print(f"\n✅ Published — Video ID: {vid}")
        print(f"   https://youtu.be/{vid}")
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        log.error("Upload failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
