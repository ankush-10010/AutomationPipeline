"""
=============================================================================
CHAT CONTEXT & PIPELINE DIAGNOSIS
=============================================================================
User Request:
- The clip matcher was selecting irrelevant clips for the narration scripts.
- Data contamination: Kevin Levin was being tagged in S1E1 clips (he doesn't appear until S1E7).
- The current architecture/columns for clip matching were insufficient.
- Llava was already tried and ruled out for visual understanding.

Diagnosis & Root Causes:
1. Metadata Starvation:
   - 82% of clips had no character tags.
   - 100% of clips had empty 'location', 'mood', and 'episode_summary' fields.
   - The 'action' field was just raw subtitle dialogue (e.g., ">> kevin: Supermenace."), not visual descriptions.
   - The semantic matcher was comparing narration text to raw dialogue text, with zero visual understanding.

2. Kevin Contamination in S1E1:
   - Caused by SRT speaker labels (">> kevin:").
   - The character enrichment script regex-matched the alias "Kevin" against this raw text, resulting in false positives for characters who were merely speaking or mentioned in speaker tags.

3. Outdated Configuration:
   - show_config.yaml contained Rick & Morty locations (Citadel of Ricks) and themes (nihilism), actively hurting the location and theme overlap scoring for Ben 10.

Solutions Implemented:
1. show_config.yaml: Updated with Ben 10 canonical locations (Bellwood, Null Void) and themes (Omnitrix, alien transformation).
2. enrich_clip_characters.py: Added regex stripping for SRT speaker labels to prevent false positive character tags.
3. enrich_clip_metadata.py: A single-pass script to propagate episode summaries, clean existing speaker labels, re-match characters cleanly, detect alien transformations, and re-compute text embeddings.
4. clip_indexer_scene_context.py: Uses an LLM (Ollama) to chunk subtitles by timecode gaps into meaningful scenes, infer a 'visual_description' and 'emotion_tone', and map them back to clips.
5. clip_indexer_objects.py: Runs YOLOv8 object detection on clip keyframes to populate 'visual_tags'.
6. clip_indexer_clip_embed.py: Extracts middle frames from clips and runs a CLIP vision encoder (via sentence-transformers) to populate 'clip_visual_embedding'.
7. clip_matcher.py: Updated `match_semantic` to heavily weigh CLIP visual embeddings, transformation matches, scene context/visual descriptions, and visual tags.

This script executes the entire pipeline in the correct order.
=============================================================================
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_step(command: list, description: str):
    """Run a subprocess command and stream its output."""
    print(f"\n{'='*80}")
    print(f"🚀 STEP: {description}")
    print(f"💻 CMD:  {' '.join(command)}")
    print(f"{'='*80}\n")
    
    try:
        # Run process, streaming stdout and stderr to the console
        process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ ERROR: Step '{description}' failed with exit code {process.returncode}")
            sys.exit(process.returncode)
        
        print(f"\n✅ SUCCESS: Step '{description}' completed successfully.\n")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR executing command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full clip enrichment pipeline.")
    parser.add_argument("--episode", default=None, help="Run scene context for a specific episode only (e.g. s1e1). If omitted, runs for all episodes in the show.")
    args = parser.parse_args()

    python_exe = sys.executable

    # -------------------------------------------------------------------------
    # STEP 1: Fast metadata fix without embeddings
    # Cleans speaker labels, propagates episode summaries, and detects transformations.
    # -------------------------------------------------------------------------
    run_step(
        [python_exe, "scripts/enrich_clip_metadata.py", "--skip-embed"],
        "Fast Metadata Enrichment (Cleaning labels, propagating summaries)"
    )

    # -------------------------------------------------------------------------
    # STEP 2: LLM Scene Context (Chunking & Descriptions)
    # -------------------------------------------------------------------------
    if args.episode:
        run_step(
            [python_exe, "scripts/clip_indexer_scene_context.py", "--episode", args.episode],
            f"LLM Scene Context & Chunking (Episode: {args.episode})"
        )
    else:
        run_step(
            [python_exe, "scripts/clip_indexer_scene_context.py", "--show", "ben10"],
            "LLM Scene Context & Chunking (All episodes)"
        )

    # -------------------------------------------------------------------------
    # STEP 3: CLIP Visual Embeddings
    # Adds clip_visual_embedding for visual-semantic matching.
    # (Note: YOLO COCO object detection was removed as it generates useless tags for cartoons)
    # -------------------------------------------------------------------------
    run_step(
        [python_exe, "scripts/clip_indexer_clip_embed.py"],
        "CLIP Visual Embeddings"
    )

    # -------------------------------------------------------------------------
    # STEP 4: Final Enrichment Pass with Re-embedding
    # Re-embeds the text with all the newly added rich metadata.
    # -------------------------------------------------------------------------
    run_step(
        [python_exe, "scripts/enrich_clip_metadata.py"],
        "Final Metadata Enrichment & Text Re-embedding"
    )

    print("🎉 FULL ENRICHMENT PIPELINE COMPLETE! 🎉")
    print("Your clip_index.json is now fully enriched and ready for clip_matcher.py.")

if __name__ == "__main__":
    main()
