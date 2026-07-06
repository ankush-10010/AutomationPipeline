"""
clip_indexer_allphasesUpdated.py — Master Orchestration Script for Episode Ingestion & Indexing.

Orchestrates a 6-step ingestion + enrichment workflow for single episodes or batches.
Every step can be run, skipped, resumed from, or cherry-picked independently.

Steps:
  1 (split)        : scene_splitter.py — Slice episode MP4 into scene clips + manifest.
  2 (subtitle)     : clip_indexer_subtitles.py — Align SRT dialogue to each clip.
  3 (embed)        : clip_indexer_embed.py — Compute MiniLM-L6-v2 text embeddings.
  4 (arcmax)       : run_visual_tagging_pipeline_arcmax.py — YOLO 0.85 + ArcFace cascade.
  5 (enrich_full)  : run_full_enrichment.py — LLM scene context + CLIP visual embeddings.
  6 (enrich_chars) : enrich_clip_characters.py — Dialogue alias matching + re-embedding.

CLI Execution Examples:
    # Full pipeline (all 6 steps):
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --show ben10

    # Resume from step 4 (skip steps 1-3):
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --start arcmax

    # Run only specific steps:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --only arcmax,enrich_chars

    # Skip expensive steps:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --skip enrich_full

    # Batch mode with step range:
    python scripts/clip_indexer_allphasesUpdated.py --batch episodes/ --show ben10 --steps 3-6
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config_loader import (
    PROJECT_ROOT,
    get_active_show,
    get_project_path,
    load_pipeline_config,
)

# ── Step Registry ─────────────────────────────────────────────────────────────
# Canonical ordered list. Each entry: (number, name, description)
STEP_REGISTRY = [
    (1, "split",        "Scene Splitter — slice episode MP4 into clips"),
    (2, "subtitle",     "Subtitle Indexer — align SRT dialogue to clips"),
    (3, "embed",        "Text Embeddings — MiniLM-L6-v2 semantic vectors"),
    (4, "arcmax",       "ArcMax Cascade — YOLO 0.85 fast-path + ArcFace verification"),
    (5, "enrich_full",  "Full Enrichment — LLM scene context + CLIP visual embeddings"),
    (6, "enrich_chars", "Character Enrichment — dialogue alias matching + re-embedding"),
]

STEP_NAMES = [s[1] for s in STEP_REGISTRY]
STEP_NAME_TO_NUM = {s[1]: s[0] for s in STEP_REGISTRY}
STEP_NUM_TO_NAME = {s[0]: s[1] for s in STEP_REGISTRY}
TOTAL_STEPS = len(STEP_REGISTRY)


# ── Terminal Colors ───────────────────────────────────────────────────────────
class Colors:
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"


def log_info(msg: str) -> None:
    print(f"{Colors.GREEN}[INFO]{Colors.RESET} {msg}")

def log_header(msg: str) -> None:
    banner = "=" * 70
    print(f"\n{Colors.CYAN}{Colors.BOLD}{banner}\n{msg}\n{banner}{Colors.RESET}")

def log_step(step_num: int, total: int, desc: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}[Step {step_num}/{total}]{Colors.RESET} {Colors.BOLD}{desc}{Colors.RESET}")

def log_skip(step_num: int, desc: str) -> None:
    print(f"\n{Colors.DIM}[Step {step_num}] SKIPPED — {desc}{Colors.RESET}")

def log_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def log_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠ [WARNING] {msg}{Colors.RESET}")

def log_error(msg: str) -> None:
    print(f"{Colors.RED}✗ [ERROR] {msg}{Colors.RESET}", file=sys.stderr)


# ── Step Selection Parser ─────────────────────────────────────────────────────

def _resolve_step_token(token: str) -> int:
    """Convert a step name or number string to its integer step number."""
    token = token.strip().lower()
    if token.isdigit():
        num = int(token)
        if 1 <= num <= TOTAL_STEPS:
            return num
        raise ValueError(f"Step number {num} out of range (1-{TOTAL_STEPS})")
    if token in STEP_NAME_TO_NUM:
        return STEP_NAME_TO_NUM[token]
    raise ValueError(
        f"Unknown step '{token}'. Valid names: {', '.join(STEP_NAMES)} or numbers 1-{TOTAL_STEPS}"
    )


def parse_step_selection(
    steps_arg: Optional[str],
    only_arg: Optional[str],
    skip_arg: Optional[str],
    start_arg: Optional[str],
) -> Set[int]:
    """Resolve CLI flags into the final set of step numbers to execute.

    Priority (highest first):
      --only   : run ONLY these steps
      --steps  : run this range/list (supports '3-6', '1,4,6', 'embed,arcmax')
      --start  : run from this step to the end
      --skip   : remove steps from the default full set
      (none)   : run all steps
    """
    # --only takes absolute priority
    if only_arg:
        return {_resolve_step_token(t) for t in only_arg.split(",")}

    # --steps supports ranges and comma lists
    if steps_arg:
        result: Set[int] = set()
        for part in steps_arg.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_n, hi_n = _resolve_step_token(lo), _resolve_step_token(hi)
                result.update(range(lo_n, hi_n + 1))
            else:
                result.add(_resolve_step_token(part))
        return result

    # --start sets a floor
    active = set(range(1, TOTAL_STEPS + 1))
    if start_arg:
        floor = _resolve_step_token(start_arg)
        active = {s for s in active if s >= floor}

    # --skip removes from whatever remains
    if skip_arg:
        to_skip = {_resolve_step_token(t) for t in skip_arg.split(",")}
        active -= to_skip

    return active


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_season_episode(filename: str) -> Optional[Tuple[int, int]]:
    """Extract season and episode numbers from filename."""
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
    log_info(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        log_success("Step finished successfully.")
    except subprocess.CalledProcessError as err:
        log_error(f"Command failed with exit code {err.returncode}")
        raise err


# ── Episode Processor ─────────────────────────────────────────────────────────

def process_episode(
    episode_mp4: Path,
    show_slug: str,
    show_config: dict,
    pipeline_cfg: dict,
    srt_dir_arg: Optional[Path],
    weights_path: Path,
    active_steps: Set[int],
    episode_index: int = 1,
    total_episodes: int = 1,
) -> bool:
    """Execute the selected ingestion steps for a single episode."""
    if not episode_mp4.exists():
        log_error(f"Episode video file not found: {episode_mp4}")
        return False

    log_header(f"[Episode {episode_index}/{total_episodes}] Processing: {episode_mp4.name}")

    # Print execution plan
    log_info("Execution plan:")
    for num, name, desc in STEP_REGISTRY:
        marker = f"{Colors.GREEN}▶ RUN {Colors.RESET}" if num in active_steps else f"{Colors.DIM}○ SKIP{Colors.RESET}"
        print(f"  {marker}  Step {num} ({name}): {desc}")
    print()

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

    active_count = len(active_steps)
    run_idx = 0

    try:
        # ── Step 1: Scene Splitter ────────────────────────────────────
        if 1 in active_steps:
            run_idx += 1
            log_step(run_idx, active_count, "scene_splitter.py — Slicing episode MP4 into scene clips...")
            run_command([
                sys.executable,
                str(SCRIPTS_DIR / "scene_splitter.py"),
                str(episode_mp4),
                "--output", str(clips_dir),
                "--prefix", prefix,
            ])
        else:
            log_skip(1, "Scene Splitter")

        # ── Step 2: Subtitle Indexer ──────────────────────────────────
        if 2 in active_steps:
            run_idx += 1
            manifest_path = clips_dir / f"{prefix}_manifest.json"
            log_step(run_idx, active_count, "clip_indexer_subtitles.py — Tagging dialogue...")
            if srt_path and manifest_path.exists():
                run_command([
                    sys.executable,
                    str(SCRIPTS_DIR / "clip_indexer_subtitles.py"),
                    "--manifest", str(manifest_path),
                    "--srt", str(srt_path),
                    "--show", show_slug,
                    "--index", str(clip_index_path),
                ])
            else:
                log_warning("Skipping subtitle indexing — manifest or SRT file is missing.")
        else:
            log_skip(2, "Subtitle Indexer")

        # ── Step 3: Text Embeddings ───────────────────────────────────
        if 3 in active_steps:
            run_idx += 1
            log_step(run_idx, active_count, "clip_indexer_embed.py — MiniLM-L6-v2 text embeddings...")
            run_command([
                sys.executable,
                str(SCRIPTS_DIR / "clip_indexer_embed.py"),
                "--index", str(clip_index_path),
            ])
        else:
            log_skip(3, "Text Embeddings")

        # ── Step 4: ArcMax Cascade (YOLO + ArcFace) ───────────────────
        if 4 in active_steps:
            run_idx += 1
            log_step(run_idx, active_count, "ArcMax Cascade — YOLO 0.85 fast-path + ArcFace verification...")
            run_command([
                sys.executable,
                str(SCRIPTS_DIR / "run_visual_tagging_pipeline_arcmax.py"),
                "--force",
                "--episode", prefix,
                "--show", show_slug,
                "--weights", str(weights_path),
            ])
        else:
            log_skip(4, "ArcMax Cascade")

        # ── Step 5: Full Enrichment ───────────────────────────────────
        if 5 in active_steps:
            run_idx += 1
            log_step(run_idx, active_count, "run_full_enrichment.py — LLM scene context + CLIP visual embeddings...")
            run_command([
                sys.executable,
                str(SCRIPTS_DIR / "run_full_enrichment.py"),
                "--episode", prefix,
            ])
        else:
            log_skip(5, "Full Enrichment")

        # ── Step 6: Character Enrichment & Re-embedding ───────────────
        if 6 in active_steps:
            run_idx += 1
            log_step(run_idx, active_count, "enrich_clip_characters.py — Dialogue alias matching + re-embedding...")
            run_command([
                sys.executable,
                str(SCRIPTS_DIR / "enrich_clip_characters.py"),
                "--index", str(clip_index_path),
                "--show", show_slug,
            ])
        else:
            log_skip(6, "Character Enrichment")

        log_success(f"All selected steps finished for {episode_mp4.name}")
        return True

    except Exception as exc:
        log_error(f"Workflow interrupted: {exc}")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    step_help_lines = "\n".join(
        f"    {num}  {name:14s}  {desc}" for num, name, desc in STEP_REGISTRY
    )

    parser = argparse.ArgumentParser(
        description="Master Orchestrator for Episode Ingestion & Clip Index Enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Steps:
{step_help_lines}

Step Selection (mutually exclusive priority: --only > --steps > --start > --skip):
  --only   Run ONLY these steps.              e.g. --only arcmax,enrich_chars
  --steps  Run a range or comma list.         e.g. --steps 3-6  or  --steps embed,arcmax
  --start  Resume from this step onward.      e.g. --start arcmax  (runs steps 4,5,6)
  --skip   Skip specific steps from full run. e.g. --skip enrich_full

Steps can be specified by name or number. Ranges use dash: 2-5

Usage Examples:
    # Full pipeline (all 6 steps):
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --show ben10

    # Resume from ArcMax (steps 4-6):
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --start arcmax

    # Run ONLY the two visual tagging steps:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --only arcmax,enrich_chars

    # Skip the expensive LLM enrichment:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --skip enrich_full

    # Run steps 3 through 6 on a batch:
    python scripts/clip_indexer_allphasesUpdated.py --batch episodes/ --show ben10 --steps 3-6

    # Run just scene splitting and subtitles:
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --steps 1-2

    # Jump from splitting straight to ArcMax (skip subtitles and embed):
    python scripts/clip_indexer_allphasesUpdated.py --episode episodes/s1e1.mp4 --only split,arcmax
        """,
    )

    # ── Input selection ───────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--episode", type=str,
        help="Path to a single episode MP4 video file",
    )
    input_group.add_argument(
        "--batch", type=str,
        help="Path to a directory containing multiple episode MP4 video files",
    )

    # ── Show and subtitle config ──────────────────────────────────────
    parser.add_argument(
        "--show", type=str, default=None,
        help="Show slug identifier (default: active show from config)",
    )
    parser.add_argument(
        "--srt-dir", type=str, default=None,
        help="Optional directory containing matching .srt subtitle files",
    )

    # ── Step selection (mutually exclusive priority group) ─────────────
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--only", type=str, default=None, metavar="STEPS",
        help="Run ONLY these steps (comma-separated names or numbers)",
    )
    step_group.add_argument(
        "--steps", type=str, default=None, metavar="RANGE",
        help="Run a range or list of steps (e.g. 3-6, embed,arcmax,enrich_chars)",
    )
    step_group.add_argument(
        "--start", type=str, default=None, metavar="STEP",
        help="Resume from this step onward (name or number)",
    )
    parser.add_argument(
        "--skip", type=str, default=None, metavar="STEPS",
        help="Skip these steps (comma-separated). Combinable with --start.",
    )

    args = parser.parse_args()

    # ── Resolve step selection ────────────────────────────────────────
    try:
        active_steps = parse_step_selection(args.steps, args.only, args.skip, args.start)
    except ValueError as e:
        log_error(str(e))
        sys.exit(1)

    if not active_steps:
        log_error("No steps selected. Check your --only/--steps/--skip/--start flags.")
        sys.exit(1)

    # ── Load config ───────────────────────────────────────────────────
    start_time = time.time()
    pipeline_cfg = load_pipeline_config()
    show_slug, show_config = get_active_show(args.show)
    srt_dir_arg = Path(args.srt_dir).resolve() if args.srt_dir else None

    weights_path = PROJECT_ROOT / "yolo_wt" / "best.pt"
    if not weights_path.exists():
        candidate_weights = list((PROJECT_ROOT / "yolo_wt").glob("*.pt"))
        if candidate_weights:
            weights_path = candidate_weights[0]
        else:
            log_warning(f"YOLO weights not found at {weights_path}")

    # ── Discover episodes ─────────────────────────────────────────────
    episodes_to_process: List[Path] = []
    if args.episode:
        episodes_to_process.append(Path(args.episode).resolve())
    elif args.batch:
        batch_path = Path(args.batch).resolve()
        if not batch_path.exists() or not batch_path.is_dir():
            log_error(f"Batch directory does not exist: {batch_path}")
            sys.exit(1)
        raw_videos = sorted(
            p for p in batch_path.rglob("*.*")
            if p.suffix.lower() in [".mp4", ".mkv"]
        )
        episodes_to_process = [
            p for p in raw_videos
            if "_scene_" not in p.name and not p.name.startswith(".")
        ]
        log_info(f"Discovered {len(episodes_to_process)} episode video files in batch directory.")

    if not episodes_to_process:
        log_error("No episode video files found to process.")
        sys.exit(1)

    # ── Process ───────────────────────────────────────────────────────
    succeeded = 0
    failed = 0
    total = len(episodes_to_process)

    for idx, ep_file in enumerate(episodes_to_process, 1):
        ok = process_episode(
            episode_mp4=ep_file,
            show_slug=show_slug,
            show_config=show_config,
            pipeline_cfg=pipeline_cfg,
            srt_dir_arg=srt_dir_arg,
            weights_path=weights_path,
            active_steps=active_steps,
            episode_index=idx,
            total_episodes=total,
        )
        if ok:
            succeeded += 1
        else:
            failed += 1
            if args.episode:
                sys.exit(1)

    elapsed = time.time() - start_time
    log_header("WORKFLOW EXECUTION SUMMARY")
    log_info(f"Total execution time : {round(elapsed, 2)} seconds")
    log_info(f"Episodes processed   : {total}")
    log_info(f"Succeeded            : {succeeded}")
    if failed > 0:
        log_warning(f"Failed               : {failed}")
    steps_ran = sorted(active_steps)
    step_names_ran = [STEP_NUM_TO_NAME[s] for s in steps_ran]
    log_info(f"Steps executed       : {', '.join(step_names_ran)}")


if __name__ == "__main__":
    main()
