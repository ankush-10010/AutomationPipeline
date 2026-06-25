"""
orchestrator.py -- Phase 8: Main pipeline controller for the AI Explainer pipeline.

Wires all phases together into a single end-to-end video production workflow:
  topic_mine -> script_gen -> (manual review) -> tts -> caption -> match -> assemble -> thumbnail -> publish

Features:
  - Full pipeline execution or individual phase selection
  - Dry-run mode (prints plan without executing)
  - Manual review checkpoint after script generation
  - Pipeline state tracking in pipeline_state.json for resume-on-interrupt
  - Post-publish housekeeping (moves topics to completed/)

Usage:
    # Full pipeline from a topic
    python orchestrator.py --topic "Why Rick's Portal Gun Changes Everything"

    # Mine topics first, then run full pipeline on the first one
    python orchestrator.py --phase topic_mine --count 5
    python orchestrator.py --phase all

    # Resume from where you left off after an interruption
    python orchestrator.py --resume

    # Dry-run to see what would happen
    python orchestrator.py --topic "Evil Morty's Grand Plan" --dry-run

    # Run a specific phase only
    python orchestrator.py --phase caption --audio audio/my_narration.wav

    # Skip the manual review checkpoint
    python orchestrator.py --topic "Multiverse Theory" --auto-approve
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Allow running as `python scripts/orchestrator.py` from project root
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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

log = setup_logging("orchestrator")


# ============================================================================
# Pipeline phases (ordered)
# ============================================================================

PHASES = [
    "topic_mine",
    "script_gen",
    "tts",
    "caption",
    "match",
    "assemble",
    "thumbnail",
    "publish",
]

PHASE_DESCRIPTIONS = {
    "topic_mine": "Generate video topic ideas via LLM",
    "script_gen": "Generate narration script for a topic",
    "tts":        "Convert script text to speech audio (.wav)",
    "caption":    "Transcribe audio to word-level captions",
    "match":      "Match narration segments to video clips / AI images",
    "assemble":   "Assemble final video from manifest + audio",
    "thumbnail":  "Generate thumbnail from final video + topic",
    "publish":    "Upload video to YouTube",
}


# ============================================================================
# Pipeline state management
# ============================================================================

STATE_FILE = PROJECT_ROOT / "pipeline_state.json"


def _default_state() -> Dict[str, Any]:
    """Return a fresh pipeline state dict."""
    return {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "started_at": datetime.now().isoformat(),
        "show_slug": None,
        "topic": None,
        "last_completed_phase": None,
        "phase_outputs": {},
        "status": "initialized",
    }


def load_state() -> Dict[str, Any]:
    """Load pipeline state from disk (or return an empty state)."""
    data = load_json(STATE_FILE)
    if isinstance(data, dict) and data:
        return data
    return _default_state()


def save_state(state: Dict[str, Any]) -> None:
    """Persist the current pipeline state to disk."""
    state["updated_at"] = datetime.now().isoformat()
    save_json(STATE_FILE, state)
    log.debug("State saved → %s", STATE_FILE)


def _next_phase(current: Optional[str]) -> Optional[str]:
    """Return the phase that follows *current*, or None if done."""
    if current is None:
        return PHASES[0]
    try:
        idx = PHASES.index(current)
        return PHASES[idx + 1] if idx + 1 < len(PHASES) else None
    except ValueError:
        return None


# ============================================================================
# Individual phase runners
# ============================================================================

def run_topic_mine(
    state: Dict,
    pipeline_cfg: Dict,
    show_slug: str,
    show: Dict,
    dry_run: bool = False,
    count: int = 10,
) -> Optional[str]:
    """Phase 1a: Mine topics and pick the first new one.

    Returns the selected topic string, or None if topics were only mined
    (no selection when running in isolation).
    """
    log.info("=" * 60)
    log.info("PHASE: topic_mine -- %s", PHASE_DESCRIPTIONS["topic_mine"])
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY RUN] Would mine %d topics for show '%s'", count, show_slug)
        return state.get("topic")

    from topic_miner import build_topic_prompt, call_ollama, parse_topics, append_to_queue

    prompt = build_topic_prompt(show, count, pipeline_cfg)
    raw = call_ollama(prompt, pipeline_cfg)
    topics = parse_topics(raw)

    if not topics:
        log.error("No topics parsed from LLM response")
        return None

    added = append_to_queue(topics, pipeline_cfg)
    log.info("Added %d new topics to queue", added)

    # If user didn't specify a topic, pick the first newly mined one
    if not state.get("topic") and topics:
        selected = topics[0].get("topic", str(topics[0]))
        log.info("Auto-selected topic: %s", selected)
        state["topic"] = selected
        return selected

    return state.get("topic")


def run_script_gen(
    state: Dict,
    pipeline_cfg: Dict,
    show_slug: str,
    show: Dict,
    dry_run: bool = False,
    auto_approve: bool = False,
) -> Optional[Path]:
    """Phase 1b: Generate the narration script.

    Returns the path to the saved script file.
    """
    log.info("=" * 60)
    log.info("PHASE: script_gen -- %s", PHASE_DESCRIPTIONS["script_gen"])
    log.info("=" * 60)

    topic = state.get("topic")
    if not topic:
        log.error("No topic set — run topic_mine first or pass --topic")
        return None

    if dry_run:
        log.info("[DRY RUN] Would generate script for: %s", topic)
        return None

    from script_verifier import generate_verified_script, ScriptVerifier

    # Optional: Initialise verifier components if enabled
    web_researcher = None
    try:
        from web_researcher import WebResearcher
        if pipeline_cfg.get("web_research", {}).get("enabled", True):
            web_researcher = WebResearcher(pipeline_cfg)
    except ImportError:
        log.debug("web_researcher not available")

    # Optional: Initialise RAG manager
    rag_manager = None
    try:
        from rag_manager import RAGManager
        rag_manager = RAGManager(pipeline_cfg)
    except ImportError:
        log.debug("rag_manager not available")

    verifier = ScriptVerifier(pipeline_cfg)

    script_path = generate_verified_script(
        topic=topic,
        show=show,
        pipeline_config=pipeline_cfg,
        rag_manager=rag_manager,
        web_researcher=web_researcher,
        verifier=verifier
    )
    log.info("Script generated → %s", script_path)
    # ── Manual review checkpoint ──────────────────────────────
    script_text = load_text(script_path)

    if not auto_approve:
        print("\n" + "=" * 60)
        print("📝  SCRIPT REVIEW CHECKPOINT")
        print("=" * 60)
        print(f"\nTopic: {topic}\n")
        print("-" * 60)
        print(script_text)
        print("-" * 60)
        print("\nOptions:")
        print("  [y] Approve and continue")
        print("  [n] Reject and abort")
        print("  [e] Open in editor (saves back on close)")
        print()

        while True:
            choice = input("Approve this script? [y/n/e]: ").strip().lower()
            if choice in ("y", "yes"):
                log.info("Script approved ✓")
                break
            elif choice in ("n", "no"):
                log.info("Script rejected — pipeline aborted")
                state["status"] = "rejected"
                save_state(state)
                return None
            elif choice in ("e", "edit"):
                import os
                editor = os.environ.get("EDITOR", "notepad" if sys.platform == "win32" else "nano")
                log.info("Opening in %s: %s", editor, script_path)
                import subprocess
                subprocess.run([editor, str(script_path)])
                # Re-read the edited file
                script_text = load_text(script_path)
                print("\n--- Updated script ---")
                print(script_text)
                print("---")
            else:
                print("Please enter y, n, or e.")
    else:
        log.info("Auto-approve enabled — skipping review checkpoint")

    state["phase_outputs"]["script_path"] = str(script_path)
    return script_path


def run_tts(
    state: Dict,
    pipeline_cfg: Dict,
    dry_run: bool = False,
    audio_file: Optional[str] = None,
) -> Optional[Path]:
    """Phase 2: Convert script to speech.

    Returns the path to the generated .wav file.
    """
    log.info("=" * 60)
    log.info("PHASE: tts -- %s", PHASE_DESCRIPTIONS["tts"])
    log.info("=" * 60)

    if dry_run:
        script_path_str = state.get("phase_outputs", {}).get("script_path", "<pending>")
        log.info("[DRY RUN] Would synthesize TTS for: %s", script_path_str)
        return None

    # If user supplied a pre-made audio file, just use it
    if audio_file:
        audio_path = Path(audio_file)
        if audio_path.exists():
            log.info("Using supplied audio: %s", audio_path)
            state["phase_outputs"]["audio_path"] = str(audio_path)
            return audio_path
        else:
            log.error("Supplied audio file not found: %s", audio_file)
            return None

    script_path_str = state.get("phase_outputs", {}).get("script_path")
    if not script_path_str:
        log.error("No script file in state -- run script_gen first")
        return None
    script_path = Path(script_path_str)

    tts_cfg = pipeline_cfg.get("tts", {})
    engine = tts_cfg.get("engine", "piper")

    if engine == "piper":
        from tts_local import synthesize_file, check_piper_installed

        if not check_piper_installed():
            log.error(
                "Piper TTS is not installed. Install with: pip install piper-tts\n"
                "Or switch to Kaggle XTTS-v2 for production quality."
            )
            return None

        piper_cfg = tts_cfg.get("piper", {})
        model = piper_cfg.get("model", "en_US-lessac-medium")
        sample_rate = piper_cfg.get("sample_rate", 22050)

        audio_dir = get_project_path("audio_dir", pipeline_cfg)
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / (script_path.stem + ".wav")

        ok = synthesize_file(script_path, audio_path, model, sample_rate)
        if not ok:
            log.error("TTS synthesis failed for: %s", script_path)
            return None

        state["phase_outputs"]["audio_path"] = str(audio_path)
        return audio_path

    else:
        log.warning(
            "TTS engine '%s' requires Kaggle GPU. Use notebooks/kaggle_gpu_batch.py "
            "to generate audio, then resume with: --resume or --phase caption --audio <path>",
            engine,
        )
        print("\n" + "=" * 60)
        print("⚠️  KAGGLE TTS REQUIRED")
        print("=" * 60)
        print(f"\n  Engine '{engine}' is configured for production quality.")
        print(f"  Script file: {script_path}")
        print("\n  Steps:")
        print("  1. Upload your script .txt files to a Kaggle dataset")
        print("  2. Run notebooks/kaggle_gpu_batch.py as a Kaggle notebook")
        print("  3. Download the generated .wav files to audio/")
        print("  4. Resume: python orchestrator.py --resume")
        print()

        # Ask for manual audio path
        audio_input = input("Enter path to audio .wav (or press Enter to pause pipeline): ").strip()
        if audio_input and Path(audio_input).exists():
            state["phase_outputs"]["audio_path"] = audio_input
            return Path(audio_input)

        state["status"] = "paused_at_tts"
        save_state(state)
        return None


def run_caption(
    state: Dict,
    pipeline_cfg: Dict,
    dry_run: bool = False,
    audio_file: Optional[str] = None,
) -> Optional[Path]:
    """Phase 3: Caption the audio with word-level timestamps.

    Returns the path to the saved caption JSON.
    """
    log.info("=" * 60)
    log.info("PHASE: caption -- %s", PHASE_DESCRIPTIONS["caption"])
    log.info("=" * 60)

    if dry_run:
        audio_path_str = audio_file or state.get("phase_outputs", {}).get("audio_path", "<pending>")
        log.info("[DRY RUN] Would caption audio: %s", audio_path_str)
        return None

    # Resolve audio path
    audio_path_str = audio_file or state.get("phase_outputs", {}).get("audio_path")
    if not audio_path_str:
        log.error("No audio file -- run tts first or pass --audio")
        return None
    audio_path = Path(audio_path_str)

    if not audio_path.exists():
        log.error("Audio file not found: %s", audio_path)
        return None

    from captioner import caption_audio_file

    cap_cfg = pipeline_cfg.get("captioning", {})
    model_size = cap_cfg.get("model_size", "small")
    compute_type = cap_cfg.get("compute_type", "int8")
    language = cap_cfg.get("language", "en")

    caption_data = caption_audio_file(audio_path, model_size, compute_type, language)

    # Save caption JSON
    captions_dir = get_project_path("captions_dir", pipeline_cfg)
    captions_dir.mkdir(parents=True, exist_ok=True)
    caption_path = captions_dir / (audio_path.stem + ".json")
    save_json(caption_path, caption_data)
    log.info("Captions saved → %s", caption_path)

    state["phase_outputs"]["caption_path"] = str(caption_path)
    state["phase_outputs"]["audio_path"] = str(audio_path)
    return caption_path


def run_match(
    state: Dict,
    pipeline_cfg: Dict,
    show_slug: str,
    show: Dict,
    dry_run: bool = False,
) -> Optional[Path]:
    """Phase 4: Match narration segments to video clips.

    Returns the path to the saved manifest JSON.
    """
    log.info("=" * 60)
    log.info("PHASE: match -- %s", PHASE_DESCRIPTIONS["match"])
    log.info("=" * 60)

    if dry_run:
        caption_path_str = state.get("phase_outputs", {}).get("caption_path", "<pending>")
        log.info("[DRY RUN] Would match segments from: %s", caption_path_str)
        return None

    caption_path_str = state.get("phase_outputs", {}).get("caption_path")
    if not caption_path_str:
        log.error("No caption file in state -- run caption first")
        return None
    caption_path = Path(caption_path_str)

    from clip_matcher import build_manifest

    caption_data = load_json(caption_path)
    if not caption_data:
        log.error("Caption file is empty: %s", caption_path)
        return None

    # Load clip index
    clip_index_path = get_project_path("clip_index", pipeline_cfg)
    clip_index = load_json(clip_index_path)
    clips = clip_index.get("clips", []) if isinstance(clip_index, dict) else clip_index
    if not clips:
        log.warning("Clip index is empty — all segments will use fallback visuals")

    matching_cfg = pipeline_cfg.get("clip_matching", {})
    llm_cfg = pipeline_cfg.get("llm", {})
    strategy = matching_cfg.get("strategy", "keyword")

    manifest = build_manifest(
        caption_data=caption_data,
        clips=clips,
        show_config=show,
        strategy=strategy,
        matching_config=matching_cfg,
        llm_config=llm_cfg,
    )

    # Save manifest
    output_dir = get_project_path("output_dir", pipeline_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a numbered manifest name to avoid collisions
    run_id = state.get("run_id", "unknown")
    manifest_path = output_dir / f"manifest_{run_id}.json"
    save_json(manifest_path, manifest)
    log.info("Manifest saved → %s", manifest_path)

    stats = manifest.get("stats", {})
    log.info(
        "Matching stats: %d/%d matched, %d fallback",
        stats.get("matched", 0),
        stats.get("total", 0),
        stats.get("fallback", 0),
    )

    state["phase_outputs"]["manifest_path"] = str(manifest_path)
    return manifest_path


def run_assemble(
    state: Dict,
    pipeline_cfg: Dict,
    dry_run: bool = False,
) -> Optional[Path]:
    """Phase 5: Assemble the final video.

    Returns the path to the final .mp4 file.
    """
    log.info("=" * 60)
    log.info("PHASE: assemble -- %s", PHASE_DESCRIPTIONS["assemble"])
    log.info("=" * 60)

    if dry_run:
        manifest_path_str = state.get("phase_outputs", {}).get("manifest_path", "<pending>")
        audio_path_str = state.get("phase_outputs", {}).get("audio_path", "<pending>")
        log.info("[DRY RUN] Would assemble video from:")
        log.info("  Manifest: %s", manifest_path_str)
        log.info("  Audio:    %s", audio_path_str)
        return None

    manifest_path_str = state.get("phase_outputs", {}).get("manifest_path")
    audio_path_str = state.get("phase_outputs", {}).get("audio_path")

    if not manifest_path_str:
        log.error("No manifest in state -- run match first")
        return None
    if not audio_path_str:
        log.error("No audio in state -- run tts first")
        return None

    manifest_path = Path(manifest_path_str)
    audio_path = Path(audio_path_str)

    from assembler import assemble_video

    manifest = load_json(manifest_path)
    video_cfg = pipeline_cfg.get("video", {})

    output_dir = get_project_path("output_dir", pipeline_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = state.get("run_id", "unknown")
    output_path = output_dir / f"final_{run_id}.mp4"

    clips_dir = str(get_project_path("clips_dir", pipeline_cfg))
    images_dir = str(get_project_path("images_dir", pipeline_cfg))

    # Check for BGM
    bgm_path = None
    bgm_cfg = video_cfg.get("bgm", {})
    if bgm_cfg.get("enabled", False):
        bgm_tracks_dir = PROJECT_ROOT / bgm_cfg.get("tracks_dir", "assets/bgm").lstrip("./")
        if bgm_tracks_dir.exists():
            for ext in ("*.mp3", "*.wav", "*.ogg", "*.m4a"):
                tracks = list(bgm_tracks_dir.glob(ext))
                if tracks:
                    bgm_path = str(tracks[0])
                    log.info("Auto-selected BGM: %s", bgm_path)
                    break

    assemble_video(
        manifest=manifest,
        audio_path=str(audio_path),
        output_path=str(output_path),
        video_cfg=video_cfg,
        bgm_path=bgm_path,
        clips_dir=clips_dir,
        images_dir=images_dir,
    )

    state["phase_outputs"]["video_path"] = str(output_path)
    return output_path


def run_thumbnail(
    state: Dict,
    pipeline_cfg: Dict,
    dry_run: bool = False,
) -> Optional[Path]:
    """Phase 6: Generate a thumbnail from the final video.

    Returns the path to the thumbnail image.
    """
    log.info("=" * 60)
    log.info("PHASE: thumbnail -- %s", PHASE_DESCRIPTIONS["thumbnail"])
    log.info("=" * 60)

    if dry_run:
        video_path_str = state.get("phase_outputs", {}).get("video_path", "<pending>")
        topic = state.get("topic", "<pending>")
        log.info("[DRY RUN] Would generate thumbnail for: %s", video_path_str)
        log.info("[DRY RUN] Topic text: %s", topic)
        return None

    video_path_str = state.get("phase_outputs", {}).get("video_path")
    topic = state.get("topic")

    if not video_path_str:
        log.error("No video file in state -- run assemble first")
        return None
    if not topic:
        log.error("No topic in state -- cannot generate thumbnail text")
        return None

    video_path = Path(video_path_str)

    import tempfile
    from thumbnail_generator import extract_frames, pick_best_frame, compose_thumbnail

    thumb_cfg = pipeline_cfg.get("thumbnail", {})
    thumb_dir = get_project_path("thumbnails_dir", pipeline_cfg)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    run_id = state.get("run_id", "unknown")
    thumb_path = thumb_dir / f"thumb_{run_id}.jpg"

    # Extract frames to a temp dir
    tmp_dir = tempfile.mkdtemp(prefix="orchestrator_thumb_")
    try:
        frame_paths = extract_frames(str(video_path), tmp_dir, num_frames=10)
        if not frame_paths:
            log.error("No frames extracted from %s", video_path)
            return None

        best_frame = pick_best_frame(frame_paths)
        if not best_frame:
            log.error("Could not select a best frame")
            return None

        compose_thumbnail(best_frame, topic, str(thumb_path), thumb_cfg)
        log.info("Thumbnail saved → %s", thumb_path)
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    state["phase_outputs"]["thumbnail_path"] = str(thumb_path)
    return thumb_path


def run_publish(
    state: Dict,
    pipeline_cfg: Dict,
    show_slug: str,
    show: Dict,
    dry_run: bool = False,
    privacy: Optional[str] = None,
    schedule_time: Optional[str] = None,
) -> Optional[str]:
    """Phase 7: Upload the video to YouTube.

    Returns the YouTube video ID.
    """
    log.info("=" * 60)
    log.info("PHASE: publish -- %s", PHASE_DESCRIPTIONS["publish"])
    log.info("=" * 60)

    if dry_run:
        video_path_str = state.get("phase_outputs", {}).get("video_path", "<pending>")
        thumbnail_path_str = state.get("phase_outputs", {}).get("thumbnail_path", "(none)")
        topic = state.get("topic", "<pending>")
        show_name = show.get("display_name", show_slug)
        log.info("[DRY RUN] Would upload to YouTube:")
        log.info("  Video:     %s", video_path_str)
        log.info("  Topic:     %s", topic)
        log.info("  Show:      %s", show_name)
        log.info("  Thumbnail: %s", thumbnail_path_str)
        log.info("  Privacy:   %s", privacy or "(from config)")
        log.info("  Schedule:  %s", schedule_time or "(immediate)")
        return None

    video_path_str = state.get("phase_outputs", {}).get("video_path")
    script_path_str = state.get("phase_outputs", {}).get("script_path")
    thumbnail_path_str = state.get("phase_outputs", {}).get("thumbnail_path")

    if not video_path_str:
        log.error("No video file in state -- run assemble first")
        return None

    video_path = Path(video_path_str)
    if not video_path.exists():
        log.error("Video file not found: %s", video_path)
        return None

    topic = state.get("topic", "")
    show_name = show.get("display_name", show_slug)

    from publisher import publish_video

    script_text = None
    if script_path_str:
        script_text = load_text(Path(script_path_str))

    video_id = publish_video(
        video_path=str(video_path),
        script_text=script_text,
        show_name=show_name,
        privacy=privacy,
        schedule_time=schedule_time,
    )

    log.info("Published — YouTube Video ID: %s", video_id)
    log.info("URL: https://youtu.be/%s", video_id)

    state["phase_outputs"]["video_id"] = video_id
    state["phase_outputs"]["video_url"] = f"https://youtu.be/{video_id}"

    # ── Post-publish: move topic to completed/ ────────────────
    _move_topic_to_completed(state, pipeline_cfg)

    return video_id


def _move_topic_to_completed(state: Dict, pipeline_cfg: Dict) -> None:
    """Move the topic script from approved/ to completed/ after publishing."""
    topic = state.get("topic", "")
    script_path_str = state.get("phase_outputs", {}).get("script_path")
    if not script_path_str:
        return

    script_path = Path(script_path_str)
    if not script_path.exists():
        return

    completed_dir = get_project_path("topics_completed", pipeline_cfg)
    completed_dir.mkdir(parents=True, exist_ok=True)

    dest = completed_dir / script_path.name
    try:
        shutil.copy2(str(script_path), str(dest))
        log.info("Topic marked completed → %s", dest)
    except Exception as exc:
        log.warning("Could not copy topic to completed/: %s", exc)

    # Also save a completion record
    record = {
        "topic": topic,
        "video_id": state.get("phase_outputs", {}).get("video_id"),
        "video_url": state.get("phase_outputs", {}).get("video_url"),
        "completed_at": datetime.now().isoformat(),
        "run_id": state.get("run_id"),
    }
    record_path = completed_dir / (script_path.stem + ".json")
    save_json(record_path, record)


# ============================================================================
# Main pipeline runner
# ============================================================================

def run_pipeline(
    topic: Optional[str] = None,
    show_slug: Optional[str] = None,
    phase: str = "all",
    dry_run: bool = False,
    resume: bool = False,
    auto_approve: bool = False,
    count: int = 10,
    audio_file: Optional[str] = None,
    privacy: Optional[str] = None,
    schedule_time: Optional[str] = None,
    use_kaggle: bool = False,
) -> None:
    """Run the AI Explainer pipeline (full or partial)."""

    # ── Load config ──────────────────────────────────────────
    pipeline_cfg = load_pipeline_config()
    slug, show = get_active_show(show_slug)

    # ── State management ─────────────────────────────────────
    if resume:
        state = load_state()
        if state.get("status") == "initialized":
            log.warning("No previous run state found — starting fresh")
            state = _default_state()
        else:
            log.info("Resuming from run %s (last completed: %s)",
                     state.get("run_id"), state.get("last_completed_phase"))
            
            if state.get("status") == "paused_for_kaggle":
                log.info("Resuming after Super-Kaggle workflow. Assuming TTS, Caption, and Match are done.")
                script_path_str = state.get("phase_outputs", {}).get("script_path", "")
                stem = Path(script_path_str).stem if script_path_str else "unknown"
                manifest_path = PROJECT_ROOT / "output" / f"manifest_{stem}.json"
                audio_path = PROJECT_ROOT / "audio" / f"{stem}.wav"
                
                state.setdefault("phase_outputs", {})
                state["phase_outputs"]["manifest_path"] = str(manifest_path)
                state["phase_outputs"]["audio_path"] = str(audio_path)
                state["last_completed_phase"] = "match"
                state["status"] = "running"
                save_state(state)
    else:
        state = _default_state()

    state["show_slug"] = slug
    if topic:
        state["topic"] = topic

    # ── Determine which phases to run ────────────────────────
    if phase == "all":
        if resume and state.get("last_completed_phase"):
            start_phase = _next_phase(state["last_completed_phase"])
            if not start_phase:
                log.info("Pipeline already completed for run %s", state.get("run_id"))
                return
            phases_to_run = PHASES[PHASES.index(start_phase):]
        else:
            phases_to_run = PHASES[:]
    else:
        if phase not in PHASES:
            log.error(
                "Unknown phase '%s'. Valid phases: %s",
                phase,
                ", ".join(PHASES),
            )
            sys.exit(1)
        phases_to_run = [phase]

    # ── Print plan ───────────────────────────────────────────
    log.info("+" + "=" * 58 + "+")
    log.info("|   AI EXPLAINER PIPELINE" + " " * 34 + "|")
    log.info("+" + "=" * 58 + "+")
    log.info("|  Run ID:  %-48s|", state.get("run_id", "?"))
    log.info("|  Show:    %-48s|", show.get("display_name", slug))
    log.info("|  Topic:   %-48s|", (state.get("topic") or "(TBD)")[:48])
    log.info("|  Mode:    %-48s|", "DRY RUN" if dry_run else "LIVE")
    log.info("+" + "=" * 58 + "+")
    for p in PHASES:
        marker = ">" if p in phases_to_run else " "
        log.info("|  %s %-12s  %s", marker, p, PHASE_DESCRIPTIONS[p].ljust(40) + "|")
    log.info("+" + "=" * 58 + "+")

    # ── Execute phases ───────────────────────────────────────
    state["status"] = "running"
    save_state(state)

    for p in phases_to_run:
        start_time = time.time()
        result = None

        try:
            if p == "topic_mine":
                result = run_topic_mine(state, pipeline_cfg, slug, show, dry_run, count)
            elif p == "script_gen":
                result = run_script_gen(state, pipeline_cfg, slug, show, dry_run, auto_approve)
            elif p == "tts":
                result = run_tts(state, pipeline_cfg, dry_run, audio_file)
            elif p == "caption":
                result = run_caption(state, pipeline_cfg, dry_run, audio_file)
            elif p == "match":
                result = run_match(state, pipeline_cfg, slug, show, dry_run)
            elif p == "assemble":
                result = run_assemble(state, pipeline_cfg, dry_run)
            elif p == "thumbnail":
                result = run_thumbnail(state, pipeline_cfg, dry_run)
            elif p == "publish":
                result = run_publish(state, pipeline_cfg, slug, show, dry_run, privacy, schedule_time)
        except KeyboardInterrupt:
            log.warning("Pipeline interrupted at phase '%s'", p)
            state["status"] = f"interrupted_at_{p}"
            save_state(state)
            print(f"\n⚠️  Pipeline paused. Resume with: python orchestrator.py --resume")
            sys.exit(130)
        except Exception as exc:
            log.error("Phase '%s' failed: %s", p, exc, exc_info=True)
            state["status"] = f"failed_at_{p}"
            state["phase_outputs"][f"{p}_error"] = str(exc)
            save_state(state)
            print(f"\n❌  Pipeline failed at phase '{p}'. Resume with: python orchestrator.py --resume")
            sys.exit(1)

        elapsed = time.time() - start_time
        log.info("Phase '%s' completed in %.1f seconds", p, elapsed)

        # If a non-dry-run phase returned None and it's not topic_mine
        # (which can legitimately return None when mining in isolation),
        # we should pause the pipeline.
        if result is None and not dry_run and p != "topic_mine":
            if state.get("status") in ("paused_at_tts", "rejected"):
                log.info("Pipeline paused — resume when ready")
                return
            # For single-phase runs, None is okay
            if phase != "all":
                break
            # For full runs, this means something went wrong
            log.error("Phase '%s' produced no output — pipeline cannot continue", p)
            state["status"] = f"stalled_at_{p}"
            save_state(state)
            return

        state["last_completed_phase"] = p
        save_state(state)

        if p == "script_gen" and use_kaggle:
            print("\n" + "=" * 60)
            print("⏸️  PAUSED FOR KAGGLE")
            print("=" * 60)
            print("Upload the approved script text, `clip_index.json`, and `config/show_config.yaml` to Kaggle.")
            print("Run the Super-Kaggle notebook.")
            print("Download the ZIP, extract it to the project root, and then run with --resume.\n")
            
            state["status"] = "paused_for_kaggle"
            save_state(state)
            return

    # ── Done ─────────────────────────────────────────────────
    if phase == "all" and not dry_run:
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        save_state(state)

        print("\n" + "=" * 60)
        print("✅  PIPELINE COMPLETE")
        print("=" * 60)
        outputs = state.get("phase_outputs", {})
        if outputs.get("video_url"):
            print(f"  YouTube: {outputs['video_url']}")
        if outputs.get("video_path"):
            print(f"  Video:   {outputs['video_path']}")
        if outputs.get("thumbnail_path"):
            print(f"  Thumb:   {outputs['thumbnail_path']}")
        print(f"  Run ID:  {state.get('run_id')}")
        print()
    elif dry_run:
        log.info("Dry run complete — no files were modified")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AI Explainer Pipeline Orchestrator — runs the full video production "
            "pipeline or individual phases."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from topic
  python orchestrator.py --topic "Why Rick's Portal Gun Changes Everything"

  # Mine 5 topics, then run everything
  python orchestrator.py --phase topic_mine --count 5
  python orchestrator.py --phase all

  # Resume after interruption
  python orchestrator.py --resume

  # Dry run
  python orchestrator.py --topic "Multiverse Theory" --dry-run

  # Single phase with custom audio
  python orchestrator.py --phase caption --audio audio/narration.wav

  # Full run, skip manual review
  python orchestrator.py --topic "Evil Morty" --auto-approve

Phases (in order):
  topic_mine  - Generate topic ideas via LLM
  script_gen  - Generate narration script
  tts         - Convert script to speech audio
  caption     - Word-level timestamps via whisper
  match       - Match segments to clips / AI images
  assemble    - Assemble final video via FFmpeg
  thumbnail   - Generate video thumbnail
  publish     - Upload to YouTube
""",
    )

    parser.add_argument(
        "--topic",
        default=None,
        help="Topic string to create a video about.",
    )
    parser.add_argument(
        "--show",
        default=None,
        help="Show slug (default: first active show from config).",
    )
    parser.add_argument(
        "--phase",
        choices=PHASES + ["all"],
        default="all",
        help="Which phase to run (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without executing anything.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved pipeline state.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip the manual script review checkpoint.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of topics to mine (default: 10, used with topic_mine phase).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to a pre-existing audio .wav file (skips TTS, used with caption/tts phases).",
    )
    parser.add_argument(
        "--privacy",
        choices=["public", "private", "unlisted"],
        default=None,
        help="YouTube privacy status (default from config).",
    )
    parser.add_argument(
        "--schedule",
        default=None,
        help="Schedule publish time (ISO 8601, e.g. 2026-07-01T14:00:00+05:30).",
    )
    parser.add_argument(
        "--use-kaggle",
        action="store_true",
        help="Use Super-Kaggle workflow. Pauses after script generation.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_pipeline(
        topic=args.topic,
        show_slug=args.show,
        phase=args.phase,
        dry_run=args.dry_run,
        resume=args.resume,
        auto_approve=args.auto_approve,
        count=args.count,
        audio_file=args.audio,
        privacy=args.privacy,
        schedule_time=args.schedule,
        use_kaggle=args.use_kaggle,
    )


if __name__ == "__main__":
    main()
