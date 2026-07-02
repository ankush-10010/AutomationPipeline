"""
clip_indexer_allphasesUpdated.py — Master Orchestration Script for Episode Ingestion & Indexing.

Orchestrates the complete 6-step ingestion workflow for single episodes or batches:
  Step 1: Run scene_splitter.py on the episode MP4 to slice it into scene clips and produce a manifest.
  Step 2: Run clip_indexer_subtitles.py using the manifest and matching SRT file to tag dialogue.
  Step 3: Run clip_indexer_embed.py to compute semantic vector embeddings.
  Step 4: Run clip_indexer_yolo.py to detect visual bounding boxes.
  Step 5: Run episode_indexer.py on the episode to extract plot summaries and canonical metadata.
  Step 6: Run enrich_clip_characters.py to unify character tags and update embeddings.

CLI Execution Examples:
    # Single episode mode:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --show rick_and_morty

    # Single episode mode with custom SRT directory:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --show rick_and_morty --srt-dir rick_and_morty_subtitles/Subtitles_Allinone

    # Batch directory mode:
    python scripts/clip_indexer_allphasesUpdated.py --batch episodes/ --show rick_and_morty
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure scripts directory is in sys.path to import shared configuration utilities
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config_loader import (
    PROJECT_ROOT,
    get_active_show,
    get_project_path,
    load_pipeline_config,
)


class Colors:
    """ANSI escape formatting codes for colored terminal output."""
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log_info(msg: str) -> None:
    """Print standard informational message in green."""
    print(f"{Colors.GREEN}[INFO]{Colors.RESET} {msg}")


def log_header(msg: str) -> None:
    """Print prominent section banner in cyan bold."""
    banner = "=" * 70
    print(f"\n{Colors.CYAN}{Colors.BOLD}{banner}\n{msg}\n{banner}{Colors.RESET}")


def log_step(step_num: int, total_steps: int, desc: str) -> None:
    """Print workflow step indicator in cyan bold."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}[Step {step_num}/{total_steps}]{Colors.RESET} {Colors.BOLD}{desc}{Colors.RESET}")


def log_success(msg: str) -> None:
    """Print success checkmark message in green."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def log_warning(msg: str) -> None:
    """Print warning indicator message in yellow."""
    print(f"{Colors.YELLOW}⚠ [WARNING] {msg}{Colors.RESET}")


def log_error(msg: str) -> None:
    """Print error indicator message in red."""
    print(f"{Colors.RED}✗ [ERROR] {msg}{Colors.RESET}", file=sys.stderr)


def parse_season_episode(filename: str) -> Optional[Tuple[int, int]]:
    """Extract season and episode numbers from filename using standard patterns."""
    match = re.search(r"[Ss](\d+)[Ee](\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\b(\d+)x(\d+)\b", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def find_matching_srt(
    episode_mp4: Path,
    prefix: str,
    srt_dir: Optional[Path],
    default_subtitles_dir: Path,
) -> Optional[Path]:
    """Locate matching SRT subtitle file across candidate directories."""
    search_dirs = []
    if srt_dir and srt_dir.exists():
        search_dirs.append(srt_dir)
    if episode_mp4.parent.exists():
        search_dirs.append(episode_mp4.parent)
    if default_subtitles_dir and default_subtitles_dir.exists():
        search_dirs.append(default_subtitles_dir)

    ep_id = parse_season_episode(episode_mp4.name)

    for directory in search_dirs:
        for srt_path in directory.glob("**/*.srt"):
            srt_id = parse_season_episode(srt_path.name)
            if ep_id and srt_id and ep_id == srt_id:
                return srt_path
            if episode_mp4.stem.lower() in srt_path.name.lower() or prefix.lower() in srt_path.name.lower():
                return srt_path
    return None


def run_command(cmd: List[str]) -> None:
    """Execute subprocess command with colored logging and error verification."""
    log_info(f"Executing command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        log_success("Step finished successfully.")
    except subprocess.CalledProcessError as err:
        log_error(f"Command failed with exit code {err.returncode}")
        raise err


def process_episode(
    episode_mp4: Path,
    show_slug: str,
    show_config: dict,
    pipeline_cfg: dict,
    srt_dir_arg: Optional[Path],
    weights_path: Path,
    episode_index: int = 1,
    total_episodes: int = 1,
) -> bool:
    """Execute complete 6-step ingestion workflow for a single episode video file."""
    if not episode_mp4.exists():
        log_error(f"Episode video file not found: {episode_mp4}")
        return False

    header_text = f"[Episode {episode_index}/{total_episodes}] Processing: {episode_mp4.name}"
    log_header(header_text)

    ep_id = parse_season_episode(episode_mp4.name)
    if ep_id:
        season, ep_num = ep_id
        prefix = f"s{season}e{ep_num}"
    else:
        prefix = episode_mp4.stem

    clips_dir = (PROJECT_ROOT / show_config.get("clips_dir", f"./clips/{show_slug}")).resolve()
    clip_index_path = get_project_path("clip_index", pipeline_cfg)
    default_subtitles_dir = get_project_path("subtitles_dir", pipeline_cfg)
    srt_path = find_matching_srt(episode_mp4, prefix, srt_dir_arg, default_subtitles_dir)

    if srt_path:
        log_info(f"Found matching subtitle file: {srt_path}")
    else:
        log_warning("No matching subtitle file found. Subtitle-dependent steps will be skipped.")

    try:
        # Step 1: Scene Splitter
        log_step(1, 6, "Running scene_splitter.py to slice episode MP4...")
        cmd1 = [
            sys.executable,
            str(SCRIPTS_DIR / "scene_splitter.py"),
            str(episode_mp4),
            "--output",
            str(clips_dir),
            "--prefix",
            prefix,
        ]
        run_command(cmd1)

        # Step 2: Subtitle Indexer
        manifest_path = clips_dir / f"{prefix}_manifest.json"
        log_step(2, 6, "Running clip_indexer_subtitles.py to tag dialogue...")
        if srt_path and manifest_path.exists():
            cmd2 = [
                sys.executable,
                str(SCRIPTS_DIR / "clip_indexer_subtitles.py"),
                "--manifest",
                str(manifest_path),
                "--srt",
                str(srt_path),
                "--show",
                show_slug,
                "--index",
                str(clip_index_path),
            ]
            run_command(cmd2)
        else:
            log_warning("Skipping Step 2 because manifest or SRT file is missing.")

        # Step 3: Embeddings
        log_step(3, 6, "Running clip_indexer_embed.py to compute semantic vector embeddings...")
        cmd3 = [
            sys.executable,
            str(SCRIPTS_DIR / "clip_indexer_embed.py"),
            "--index",
            str(clip_index_path),
        ]
        run_command(cmd3)

        # Step 4: YOLO Vision Tagging
        log_step(4, 6, "Running clip_indexer_yolo.py to detect visual bounding boxes...")
        cmd4 = [
            sys.executable,
            str(SCRIPTS_DIR / "clip_indexer_yolo.py"),
            "--index",
            str(clip_index_path),
            "--weights",
            str(weights_path),
            "--target-dir",
            prefix,
        ]
        run_command(cmd4)

        # Step 5: Whole Episode Summary Indexer
        log_step(5, 6, "Running episode_indexer.py on whole episode to extract plot summaries...")
        if srt_path:
            cmd5 = [
                sys.executable,
                str(SCRIPTS_DIR / "episode_indexer.py"),
                "--show",
                show_slug,
                "--single",
                str(srt_path),
            ]
            run_command(cmd5)
        else:
            log_warning("Skipping Step 5 because matching SRT file is missing.")

        # Step 6: Character Enrichment & Re-embedding
        log_step(6, 6, "Running enrich_clip_characters.py to unify character tags and update embeddings...")
        cmd6 = [
            sys.executable,
            str(SCRIPTS_DIR / "enrich_clip_characters.py"),
            "--index",
            str(clip_index_path),
            "--show",
            show_slug,
        ]
        run_command(cmd6)

        log_success(f"All workflow phases finished successfully for {episode_mp4.name}")
        return True

    except Exception as exc:
        log_error(f"Workflow interrupted during episode ingestion: {exc}")
        return False


def main() -> None:
    """Parse CLI arguments and coordinate single or batch episode processing."""
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for Complete Episode Ingestion and Indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CLI Execution Examples:
    # Single episode mode:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --show rick_and_morty

    # Single episode mode with custom SRT directory:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --show rick_and_morty --srt-dir rick_and_morty_subtitles/Subtitles_Allinone

    # Batch directory mode:
    python scripts/clip_indexer_allphasesUpdated.py --batch episodes/ --show rick_and_morty
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--episode",
        type=str,
        help="Absolute or relative path to a single episode MP4 video file",
    )
    input_group.add_argument(
        "--batch",
        type=str,
        help="Path to a directory containing multiple episode MP4 video files",
    )

    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="Show slug identifier (default: active show from config)",
    )
    parser.add_argument(
        "--srt-dir",
        type=str,
        default=None,
        help="Optional directory containing matching .srt subtitle files",
    )

    args = parser.parse_args()

    start_time = time.time()
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)

    srt_dir_arg = Path(args.srt_dir).resolve() if args.srt_dir else None

    weights_path = PROJECT_ROOT / "yolo_wt" / "20epochs.pt"
    if not weights_path.exists():
        candidate_weights = list((PROJECT_ROOT / "yolo_wt").glob("*.pt"))
        if candidate_weights:
            weights_path = candidate_weights[0]
        else:
            log_warning(f"Default YOLO weights file not found at {weights_path}")

    episodes_to_process: List[Path] = []
    if args.episode:
        ep_path = Path(args.episode).resolve()
        episodes_to_process.append(ep_path)
    elif args.batch:
        batch_path = Path(args.batch).resolve()
        if not batch_path.exists() or not batch_path.is_dir():
            log_error(f"Batch directory does not exist or is not a directory: {batch_path}")
            sys.exit(1)
        raw_videos = sorted([p for p in batch_path.rglob("*.*") if p.suffix.lower() in [".mp4", ".mkv"]])
        episodes_to_process = [
            p for p in raw_videos if "_scene_" not in p.name and not p.name.startswith(".")
        ]
        log_info(f"Discovered {len(episodes_to_process)} episode video files in batch directory.")

    if not episodes_to_process:
        log_error("No episode video files found to process.")
        sys.exit(1)

    succeeded_count = 0
    failed_count = 0
    total_count = len(episodes_to_process)

    for idx, ep_file in enumerate(episodes_to_process, 1):
        success = process_episode(
            episode_mp4=ep_file,
            show_slug=show_slug,
            show_config=show_config,
            pipeline_cfg=pipeline_cfg,
            srt_dir_arg=srt_dir_arg,
            weights_path=weights_path,
            episode_index=idx,
            total_episodes=total_count,
        )
        if success:
            succeeded_count += 1
        else:
            failed_count += 1

        if not success and args.episode:
            sys.exit(1)

    elapsed_seconds = time.time() - start_time
    log_header("WORKFLOW EXECUTION SUMMARY")
    log_info(f"Total execution time: {round(elapsed_seconds, 2)} seconds")
    log_info(f"Total episodes scanned: {total_count}")
    log_info(f"Succeeded ingestion runs: {succeeded_count}")
    if failed_count > 0:
        log_warning(f"Failed ingestion runs: {failed_count}")


if __name__ == "__main__":
    main()
