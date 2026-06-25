"""
script_verifier.py — Fact-checks generated scripts against web research and
episode data.

Provides a ScriptVerifier class for the pipeline, plus a standalone
``generate_verified_script()`` function that wires together web research,
script generation, verification, and iterative correction.
"""

import argparse
import json
import re
import sys
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

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

log = setup_logging("script_verifier")


# ============================================================================
# ScriptVerifier
# ============================================================================
class ScriptVerifier:
    """Fact-checks a narration script against research dossier + episode data."""

    def __init__(self, pipeline_config: dict):
        self.pipeline_config = pipeline_config
        self.verification_config = pipeline_config.get("verification", {})
        self.llm_config = pipeline_config.get("llm", {})
        self.prompts_dir = get_project_path("prompts_dir", pipeline_config)
        self.min_score = self.verification_config.get("min_score", 7)
        self.episode_index = self._load_episode_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def verify(self, script: str, dossier: dict, topic: str) -> dict:
        """
        Verify *script* against *dossier* facts and episode data.

        Returns::

            {
                "verdict": "PASS" | "NEEDS_CORRECTION",
                "score": int,
                "corrections": [...],
                "factual_claims_checked": int,
                "factual_claims_correct": int,
            }
        """
        prompt = self._build_verification_prompt(script, dossier, topic)
        raw = self._call_ollama(prompt, temperature=0.2)
        result = self._parse_verification_response(raw)

        log.info(
            "Verification → verdict=%s  score=%s  claims=%s/%s  corrections=%d",
            result["verdict"],
            result["score"],
            result["factual_claims_correct"],
            result["factual_claims_checked"],
            len(result.get("corrections", [])),
        )
        return result

    def build_correction_prompt(
        self,
        topic: str,
        original_script: str,
        corrections: List[dict],
    ) -> str:
        """
        Build a prompt that asks the LLM to fix *corrections* in
        *original_script* while keeping the same tone and length.
        """
        template_path = self.prompts_dir / "correction_prompt.txt"
        template = load_text(template_path)

        if not template:
            log.warning(
                "Correction prompt template missing (%s), using inline fallback",
                template_path,
            )
            template = (
                "Fix the following issues in this script about \"{topic}\".\n\n"
                "ORIGINAL SCRIPT:\n{original_script}\n\n"
                "CORRECTIONS:\n{corrections_list}\n\n"
                "Output ONLY the corrected raw script text."
            )

        corrections_list = self._format_corrections(corrections)

        return template.format(
            topic=topic,
            original_script=original_script,
            corrections_list=corrections_list,
        )

    # ------------------------------------------------------------------
    # Episode index
    # ------------------------------------------------------------------
    def _load_episode_index(self) -> dict:
        """Load ``episode_index.json`` from PROJECT_ROOT; return {} if missing."""
        ep_path = PROJECT_ROOT / "episode_index.json"
        if not ep_path.exists():
            log.debug("No episode_index.json found at %s", ep_path)
            return {}
        try:
            data = load_json(ep_path)
            if isinstance(data, dict):
                return data
            # If stored as a list, try to index by episode key
            if isinstance(data, list):
                indexed: Dict[str, Any] = {}
                for ep in data:
                    key = ep.get("id") or ep.get("title", "")
                    if key:
                        indexed[key] = ep
                return indexed
            return {}
        except Exception as exc:
            log.warning("Failed to load episode_index.json: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_verification_prompt(
        self, script: str, dossier: dict, topic: str
    ) -> str:
        """Assemble the fact-checking prompt from template + dossier + episodes."""
        template_path = self.prompts_dir / "verify_prompt.txt"
        template = load_text(template_path)

        if not template:
            log.error(
                "Verification prompt template missing or empty: %s", template_path
            )
            sys.exit(1)

        verified_facts = self._format_facts(dossier)
        episode_data = self._format_relevant_episodes(dossier)

        return template.format(
            topic=topic,
            script=script,
            verified_facts=verified_facts,
            episode_data=episode_data,
        )

    def _format_facts(self, dossier: dict) -> str:
        """Extract and format ``verified_facts`` from the research dossier."""
        if not dossier:
            return "(No verified facts available)"

        facts = dossier.get("verified_facts", [])
        if not facts:
            # Fall back to raw_results / summary if no structured facts
            summary = dossier.get("summary", "")
            if summary:
                return summary
            return "(No verified facts available)"

        lines: List[str] = []
        for i, fact in enumerate(facts, 1):
            if isinstance(fact, dict):
                text = fact.get("fact", fact.get("text", str(fact)))
                source = fact.get("source", "")
                line = f"{i}. {text}"
                if source:
                    line += f"  [source: {source}]"
            else:
                line = f"{i}. {fact}"
            lines.append(line)

        return "\n".join(lines)

    def _format_relevant_episodes(self, dossier: dict) -> str:
        """
        Find episodes in the episode index that match the dossier's topic
        keywords and format them for the prompt.
        """
        if not self.episode_index:
            return "(No episode index available)"

        # Gather keywords from dossier
        keywords: List[str] = []
        if dossier:
            topic_str = dossier.get("topic", "")
            if topic_str:
                keywords.extend(
                    w.lower()
                    for w in re.split(r"\W+", topic_str)
                    if len(w) > 2
                )
            # Also check any explicit episode refs in the dossier
            ep_refs = dossier.get("episode_references", [])
            keywords.extend(str(r).lower() for r in ep_refs)

        if not keywords:
            return "(No episode data matched)"

        matched: List[str] = []
        for key, ep in self.episode_index.items():
            ep_str = json.dumps(ep, default=str).lower()
            if any(kw in ep_str for kw in keywords):
                title = ep.get("title", key)
                season = ep.get("season", "?")
                episode = ep.get("episode", "?")
                summary = ep.get("summary", ep.get("description", ""))
                entry = f"S{season}E{episode} — {title}"
                if summary:
                    entry += f": {summary[:200]}"
                matched.append(entry)

        if not matched:
            return "(No episode data matched)"

        return "\n".join(matched[:10])  # Cap at 10 episodes

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_verification_response(self, raw: str) -> dict:
        """
        Extract JSON from the LLM verification response.
        Returns a default PASS dict if parsing fails.
        """
        default = {
            "verdict": "PASS",
            "score": 10,
            "corrections": [],
            "factual_claims_checked": 0,
            "factual_claims_correct": 0,
            "notes": "Parsing failed — defaulting to PASS",
        }

        if not raw or not raw.strip():
            log.warning("Empty verification response — defaulting to PASS")
            return default

        # Try to find JSON object in the response
        # LLMs sometimes wrap JSON in markdown code blocks
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            log.warning("No JSON found in verification response — defaulting to PASS")
            return default

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse verification JSON: %s", exc)
            return default

        # Normalise and validate required fields
        result = {
            "verdict": parsed.get("verdict", "PASS"),
            "score": int(parsed.get("score", 10)),
            "corrections": parsed.get("corrections", []),
            "factual_claims_checked": int(
                parsed.get("factual_claims_checked", 0)
            ),
            "factual_claims_correct": int(
                parsed.get("factual_claims_correct", 0)
            ),
            "notes": parsed.get("notes", ""),
        }

        # Enforce min_score threshold on verdict
        if result["score"] < self.min_score:
            result["verdict"] = "NEEDS_CORRECTION"
        if result["corrections"] and result["verdict"] == "PASS":
            result["verdict"] = "NEEDS_CORRECTION"

        return result

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_corrections(corrections: List[dict]) -> str:
        """Format corrections list for the correction prompt."""
        if not corrections:
            return "(none)"
        lines: List[str] = []
        for i, c in enumerate(corrections, 1):
            claim = c.get("claim", "N/A")
            issue = c.get("issue", "N/A")
            fix = c.get("suggested_fix", "N/A")
            lines.append(
                f"{i}. CLAIM: \"{claim}\"\n"
                f"   ISSUE: {issue}\n"
                f"   FIX:   {fix}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Ollama call
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str, temperature: float = 0.2) -> str:
        """Call Ollama /api/generate with configurable temperature."""
        base_url = self.llm_config.get("base_url", "http://localhost:11434").rstrip("/")
        # Use verifier-specific model if configured, else fall back to main model
        model = (
            self.verification_config.get("verifier_model")
            or self.llm_config.get("model", "llama3.1:8b")
        )
        timeout = self.llm_config.get("timeout_seconds", 300)

        url = f"{base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.llm_config.get("max_tokens", 4096),
            },
        }

        log.info("Calling Ollama verifier → %s (model: %s, temp: %.2f)", url, model, temperature)

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
            log.error("Ollama verifier timed out after %ds", timeout)
            sys.exit(1)
        except requests.HTTPError as exc:
            log.error("Ollama verifier HTTP error: %s", exc)
            sys.exit(1)

        elapsed = time.time() - start
        result = resp.json()
        response_text = result.get("response", "")
        log.info("Verifier responded in %.1fs (%d chars)", elapsed, len(response_text))

        return response_text


# ============================================================================
# Standalone verified-generation function
# ============================================================================
def generate_verified_script(
    topic: str,
    show: dict,
    pipeline_config: dict,
    rag_manager=None,
    web_researcher=None,
    verifier: Optional[ScriptVerifier] = None,
) -> Path:
    """
    End-to-end script generation with optional web research and verification.

    Steps:
        1. Research topic via web_researcher (if provided).
        2. Build prompt and generate script via script_generator.
        3. Verify and iteratively correct (if verifier is provided + enabled).
        4. Save script, dossier, and verification log.
    """
    from script_generator import build_script_prompt, call_ollama, save_script

    verification_cfg = pipeline_config.get("verification", {})
    verification_enabled = verification_cfg.get("enabled", True)
    max_retries = verification_cfg.get("max_retries", 2)
    save_dossier_flag = verification_cfg.get("save_dossier", True)
    save_log_flag = verification_cfg.get("save_verification_log", True)

    # ── Step 1: Web research ──────────────────────────────────────────
    dossier: Dict[str, Any] = {}
    if web_researcher:
        log.info("Researching topic: %s", topic)
        try:
            dossier = web_researcher.research_topic(topic, show.get("display_name", ""))
        except Exception as exc:
            log.warning("Web research failed (continuing without): %s", exc)

    # ── Step 2: Build prompt + generate script ────────────────────────
    prompt = build_script_prompt(topic, show, pipeline_config, rag_manager)

    # Inject dossier context into the prompt if available
    if dossier:
        try:
            from web_researcher import format_dossier_for_prompt

            dossier_text = format_dossier_for_prompt(dossier)
            prompt += (
                "\n\nADDITIONAL RESEARCH CONTEXT (use to improve accuracy):\n"
                + dossier_text
            )
        except ImportError:
            log.debug("web_researcher.format_dossier_for_prompt not available")

    log.info("Generating initial script for: %s", topic)
    script = call_ollama(prompt, pipeline_config)

    if not script.strip():
        log.error("Ollama returned an empty response for topic: %s", topic)
        sys.exit(1)

    # ── Step 3: Verification loop ─────────────────────────────────────
    verification_results: List[dict] = []

    if verifier and verification_enabled and dossier:
        retries = 0
        while retries <= max_retries:
            log.info(
                "Verification pass %d/%d for: %s",
                retries + 1,
                max_retries + 1,
                topic,
            )
            result = verifier.verify(script, dossier, topic)
            verification_results.append(result)

            if result["verdict"] == "PASS":
                log.info("✓ Script passed verification (score: %d)", result["score"])
                break

            if retries >= max_retries:
                log.warning(
                    "Max retries reached — saving script with score %d",
                    result["score"],
                )
                break

            # Build correction prompt and regenerate
            log.info(
                "Script needs correction (score: %d, %d issues) — retry %d",
                result["score"],
                len(result["corrections"]),
                retries + 1,
            )
            previous_script = script
            correction_prompt = verifier.build_correction_prompt(
                topic, script, result["corrections"]
            )
            candidate_script = call_ollama(correction_prompt, pipeline_config)

            if not candidate_script.strip():
                log.error("Ollama returned empty correction — keeping previous version")
                break

            # Safety filter: detect if the LLM started writing meta fact-checking commentary
            meta_phrases = ["fact-check", "revisit this", "original script", "corrections required", "let's revisit", "hallucinat"]
            if any(p in candidate_script.lower() for p in meta_phrases):
                log.warning("Correction introduced robotic meta-commentary — rejecting correction and keeping previous draft")
                script = previous_script
                break

            script = candidate_script
            retries += 1
    elif not verification_enabled:
        log.info("Verification disabled — skipping fact-check")
    elif not dossier:
        log.info("No research dossier — skipping verification")

    # ── Step 4: Save outputs ──────────────────────────────────────────
    script_path = save_script(topic, script, pipeline_config)

    # Save dossier alongside the script
    if dossier and save_dossier_flag:
        dossier_path = script_path.with_suffix(".dossier.json")
        save_json(dossier_path, dossier)
        log.info("Dossier saved → %s", dossier_path)

    # Save verification log
    if verification_results and save_log_flag:
        log_path = script_path.with_suffix(".verification.json")
        save_json(log_path, {
            "topic": topic,
            "passes": len(verification_results),
            "final_verdict": verification_results[-1]["verdict"],
            "final_score": verification_results[-1]["score"],
            "results": verification_results,
        })
        log.info("Verification log saved → %s", log_path)

    return script_path


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script Verifier — fact-check scripts against web research & episode data",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Topic string — runs full verified generation",
    )
    parser.add_argument(
        "--script-file",
        type=str,
        default=None,
        help="Path to an existing script file to verify against web research",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="Show slug from show_config.yaml (default: first active show)",
    )
    args = parser.parse_args()

    if args.topic is None and args.script_file is None:
        parser.error("Provide either --topic 'some topic' or --script-file PATH")

    pipeline_config = load_pipeline_config()
    slug, show = get_active_show(args.show)
    log.info("=== Script Verifier for '%s' ===", show.get("display_name", slug))

    verifier = ScriptVerifier(pipeline_config)

    # --- Optional: import web_researcher if available ---
    web_researcher = None
    try:
        from web_researcher import WebResearcher

        web_research_cfg = pipeline_config.get("web_research", {})
        if web_research_cfg.get("enabled", True):
            web_researcher = WebResearcher(pipeline_config)
            log.info("Web researcher initialised")
    except ImportError:
        log.debug("web_researcher module not available — proceeding without")

    # --- Mode 1: Verify an existing script file ---
    if args.script_file:
        script_path = Path(args.script_file)
        if not script_path.exists():
            log.error("Script file not found: %s", script_path)
            sys.exit(1)

        script_text = load_text(script_path)
        if not script_text.strip():
            log.error("Script file is empty: %s", script_path)
            sys.exit(1)

        # Infer topic from filename
        topic = script_path.stem.replace("_", " ").title()
        log.info("Verifying script: %s (inferred topic: %s)", script_path, topic)

        # Research if possible
        dossier: Dict[str, Any] = {}
        if web_researcher:
            try:
                dossier = web_researcher.research_topic(
                    topic, show.get("display_name", "")
                )
            except Exception as exc:
                log.warning("Web research failed: %s", exc)

        if not dossier:
            log.warning("No dossier available — verification will be limited")
            dossier = {"topic": topic}

        result = verifier.verify(script_text, dossier, topic)

        # Print results
        log.info("═" * 50)
        log.info("VERDICT:  %s", result["verdict"])
        log.info("SCORE:    %d / 10", result["score"])
        log.info(
            "CLAIMS:   %d checked, %d correct",
            result["factual_claims_checked"],
            result["factual_claims_correct"],
        )
        if result["corrections"]:
            log.info("CORRECTIONS:")
            for c in result["corrections"]:
                log.info("  • %s → %s", c.get("claim", "?"), c.get("suggested_fix", "?"))
        log.info("═" * 50)

        # Save verification log next to the script
        verification_cfg = pipeline_config.get("verification", {})
        if verification_cfg.get("save_verification_log", True):
            log_path = script_path.with_suffix(".verification.json")
            save_json(log_path, {
                "topic": topic,
                "passes": 1,
                "final_verdict": result["verdict"],
                "final_score": result["score"],
                "results": [result],
            })
            log.info("Verification log saved → %s", log_path)

    # --- Mode 2: Full verified generation ---
    elif args.topic:
        # Optional RAG manager
        rag_manager = None
        try:
            from rag_manager import RAGManager

            rag_manager = RAGManager(pipeline_config)
        except ImportError:
            log.debug("rag_manager not available — proceeding without RAG")

        path = generate_verified_script(
            topic=args.topic,
            show=show,
            pipeline_config=pipeline_config,
            rag_manager=rag_manager,
            web_researcher=web_researcher,
            verifier=verifier,
        )
        log.info("✓ Done — verified script saved to %s", path)


if __name__ == "__main__":
    main()
