"""
web_researcher.py — Web Research module for the AI Explainer pipeline.

Researches topics using DuckDuckGo web search and Ollama LLM to compile
verified fact dossiers.  Results are cached locally to avoid redundant
lookups.
"""

import argparse
import json
import re
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

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

log = setup_logging("web_researcher")


# ---------------------------------------------------------------------------
# Standalone helper — usable without instantiating the class
# ---------------------------------------------------------------------------
def format_dossier_for_prompt(dossier: dict) -> str:
    """
    Format a research dossier into readable text suitable for injection
    into an LLM prompt.
    """
    if not dossier:
        return ""

    lines = [
        f"=== Research Dossier: {dossier.get('topic', 'Unknown')} ===",
        f"Researched at: {dossier.get('researched_at', 'N/A')}",
        f"Sources consulted: {dossier.get('source_count', 0)}",
        "",
    ]

    facts = dossier.get("verified_facts", [])
    if facts:
        lines.append("Verified Facts:")
        for i, f in enumerate(facts, 1):
            confidence = f.get("confidence", "medium")
            source = f.get("source", "unknown")
            lines.append(f"  {i}. [{confidence}] {f.get('fact', '')}  (source: {source})")
        lines.append("")

    episodes = dossier.get("relevant_episodes", [])
    if episodes:
        lines.append("Relevant Episodes:")
        for ep in episodes:
            lines.append(f"  - {ep}")
        lines.append("")

    queries = dossier.get("search_queries_used", [])
    if queries:
        lines.append("Search Queries Used:")
        for q in queries:
            lines.append(f"  - {q}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WebResearcher class
# ---------------------------------------------------------------------------
class WebResearcher:
    """
    Researches topics via web search and LLM-powered fact extraction.

    Uses DuckDuckGo for search and Ollama for query generation / fact
    extraction.  Results are cached as JSON files in ``research_cache/``.
    """

    def __init__(self, pipeline_config: dict):
        self.pipeline_config = pipeline_config
        self.config = pipeline_config.get("web_research", {})

        # LLM settings
        llm_cfg = pipeline_config.get("llm", {})
        self.base_url = llm_cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = llm_cfg.get("model", "llama3.1:8b")
        self.timeout = llm_cfg.get("timeout_seconds", 300)
        self.max_tokens = llm_cfg.get("max_tokens", 1024)

        # Web-research-specific settings
        self.enabled = self.config.get("enabled", True)
        self.engine = self.config.get("engine", "duckduckgo")
        self.max_results_per_query = self.config.get("max_results_per_query", 5)
        self.max_queries = self.config.get("max_queries_per_topic", 3)
        self.cache_results = self.config.get("cache_results", True)
        self.cache_ttl_days = self.config.get("cache_ttl_days", 7)

        # Cache directory
        cache_rel = self.config.get("cache_dir", "./research_cache")
        self.cache_dir = (PROJECT_ROOT / cache_rel).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "WebResearcher initialised (engine=%s, cache=%s, ttl=%dd)",
            self.engine,
            self.cache_dir,
            self.cache_ttl_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def research_topic(self, topic: str, show_name: str) -> dict:
        """
        Main entry point.  Returns a dossier dict for the given topic.

        Checks cache first; if a valid cached dossier exists it is
        returned immediately.
        """
        if not self.enabled:
            log.warning("Web research is disabled in pipeline config")
            return self._compile_dossier(topic, show_name, [], [], [])

        # Check cache
        cached = self._load_cached_dossier(topic)
        if cached is not None:
            log.info("Using cached dossier for '%s'", topic)
            return cached

        log.info("Researching topic: %s", topic)

        # 1. Generate search queries via LLM
        queries = self._generate_search_queries(topic, show_name)

        # 2. Execute searches
        all_results = []
        for q in queries:
            results = self._execute_search(q, max_results=self.max_results_per_query)
            all_results.extend(results)
            log.info("  Search '%s' → %d results", q[:60], len(results))

        # 3. Extract facts via LLM
        facts = self._extract_facts(all_results, topic, show_name)

        # 4. Identify relevant episodes from facts
        episodes = self._identify_relevant_episodes(facts)

        # 5. Compile dossier
        dossier = self._compile_dossier(topic, show_name, queries, facts, episodes)

        # 6. Cache
        if self.cache_results:
            self._save_cached_dossier(topic, dossier)

        log.info(
            "Research complete: %d facts, %d episodes, %d sources",
            len(facts),
            len(episodes),
            dossier.get("source_count", 0),
        )
        return dossier

    # ------------------------------------------------------------------
    # Ollama helper
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str, temperature: float | None = None) -> str:
        """
        Send a prompt to Ollama and return the response text.

        Parameters
        ----------
        prompt : str
            The prompt to send.
        temperature : float, optional
            Override the default LLM temperature.  If *None*, uses the
            value from ``pipeline_config['llm']['temperature']``.
        """
        llm_cfg = self.pipeline_config.get("llm", {})
        temp = temperature if temperature is not None else llm_cfg.get("temperature", 0.8)

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": self.max_tokens,
            },
        }

        log.debug("Calling Ollama → %s (model: %s, temp: %.2f)", url, self.model, temp)

        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.ConnectionError:
            log.error(
                "Cannot connect to Ollama at %s — is it running? "
                "Start with: ollama serve",
                self.base_url,
            )
            return ""
        except requests.Timeout:
            log.error("Ollama request timed out after %ds", self.timeout)
            return ""
        except requests.HTTPError as e:
            log.error("Ollama returned HTTP error: %s", e)
            return ""

        elapsed = time.time() - start
        result = resp.json()
        response_text = result.get("response", "")
        log.debug("Ollama responded in %.1fs (%d chars)", elapsed, len(response_text))
        return response_text

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------
    def _generate_search_queries(self, topic: str, show_name: str) -> list:
        """
        Use the LLM to generate 3-5 targeted search queries for the
        given topic.  Uses low temperature (0.3) for consistency.
        """
        prompt = (
            f"You are a research assistant preparing to fact-check a video "
            f"script about the TV show '{show_name}'.\n\n"
            f"Topic: {topic}\n\n"
            f"Generate exactly {self.max_queries} concise web search queries "
            f"that would help verify facts about this topic. The queries "
            f"should cover different angles (plot details, character info, "
            f"fan theories, episode references).\n\n"
            f"Return ONLY a JSON array of strings, no other text. Example:\n"
            f'["query one", "query two", "query three"]'
        )

        response = self._call_ollama(prompt, temperature=0.3)
        if not response:
            log.warning("LLM returned empty response for query generation")
            return [f"{show_name} {topic}"]

        # Try to parse JSON array from response
        try:
            # Extract JSON array even if surrounded by extra text
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    return queries[: self.max_queries]
        except (json.JSONDecodeError, ValueError):
            pass

        log.warning("Failed to parse LLM query response — using fallback")
        return [f"{show_name} {topic}"]

    # ------------------------------------------------------------------
    # Search execution
    # ------------------------------------------------------------------
    def _execute_search(self, query: str, max_results: int = 5) -> list:
        """
        Execute a DuckDuckGo search and return a list of result dicts:
        ``[{title, snippet, url}, ...]``.

        Returns an empty list if the ``duckduckgo_search`` package is
        not installed or if the search fails.
        """
        if DDGS is None:
            log.warning(
                "duckduckgo_search is not installed — "
                "pip install duckduckgo-search to enable web research"
            )
            return []

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            log.error("DuckDuckGo search failed for '%s': %s", query, e)
            return []

        results = []
        for r in raw_results:
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------
    def _extract_facts(self, search_results: list, topic: str, show_name: str) -> list:
        """
        Use the LLM to extract verified facts from search result
        snippets.  Returns a list of ``{fact, source, confidence}``
        dicts.  Uses low temperature (0.3) for factual accuracy.
        """
        if not search_results:
            log.warning("No search results to extract facts from")
            return []

        # Build a condensed view of search results for the prompt
        snippets_text = ""
        for i, r in enumerate(search_results, 1):
            snippets_text += (
                f"[{i}] Title: {r['title']}\n"
                f"    URL: {r['url']}\n"
                f"    Snippet: {r['snippet']}\n\n"
            )

        prompt = (
            f"You are a fact-checker for a video about '{show_name}'.\n\n"
            f"Topic: {topic}\n\n"
            f"Below are search result snippets. Extract verified, specific "
            f"facts relevant to the topic. For each fact, assess confidence "
            f"(high, medium, low) based on how well-supported it is.\n\n"
            f"Search Results:\n{snippets_text}\n"
            f"Return ONLY a JSON array of objects, each with keys: "
            f'"fact", "source", "confidence". Example:\n'
            f'[{{"fact": "some fact", "source": "https://example.com", '
            f'"confidence": "high"}}]\n\n'
            f"Extract up to 10 facts. Return an empty array [] if no "
            f"relevant facts can be extracted."
        )

        response = self._call_ollama(prompt, temperature=0.3)
        if not response:
            log.warning("LLM returned empty response for fact extraction")
            return []

        # Parse JSON array from response
        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                facts = json.loads(match.group())
                if isinstance(facts, list):
                    # Validate structure
                    valid_facts = []
                    for f in facts:
                        if isinstance(f, dict) and "fact" in f:
                            valid_facts.append(
                                {
                                    "fact": f.get("fact", ""),
                                    "source": f.get("source", "unknown"),
                                    "confidence": f.get("confidence", "medium"),
                                }
                            )
                    return valid_facts
        except (json.JSONDecodeError, ValueError):
            pass

        log.warning("Failed to parse LLM fact-extraction response")
        return []

    # ------------------------------------------------------------------
    # Episode identification
    # ------------------------------------------------------------------
    def _identify_relevant_episodes(self, facts: list) -> list:
        """
        Scan extracted facts for episode references using regex
        patterns like ``S01E02``, ``s1e3``, ``Season 2 Episode 5``, etc.

        Returns a sorted, deduplicated list of episode reference strings.
        """
        episode_refs = set()

        # Patterns: S01E02, s1e3, season 2 episode 5
        patterns = [
            re.compile(r"[Ss]\d{1,2}[Ee]\d{1,2}"),
            re.compile(r"[Ss]eason\s*\d{1,2}\s*[Ee]pisode\s*\d{1,2}", re.IGNORECASE),
        ]

        for f in facts:
            text = f.get("fact", "")
            for pattern in patterns:
                matches = pattern.findall(text)
                for m in matches:
                    episode_refs.add(m.strip())

        return sorted(episode_refs)

    # ------------------------------------------------------------------
    # Dossier compilation
    # ------------------------------------------------------------------
    def _compile_dossier(
        self,
        topic: str,
        show_name: str,
        queries: list,
        facts: list,
        episodes: list,
    ) -> dict:
        """
        Assemble all research data into a single dossier dict.
        """
        # Count unique sources
        sources = set()
        for f in facts:
            src = f.get("source", "")
            if src and src != "unknown":
                sources.add(src)

        return {
            "topic": topic,
            "show_name": show_name,
            "researched_at": datetime.now(timezone.utc).isoformat(),
            "search_queries_used": queries,
            "verified_facts": facts,
            "relevant_episodes": episodes,
            "source_count": len(sources),
        }

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _load_cached_dossier(self, topic: str) -> dict | None:
        """
        Check for a cached dossier at ``research_cache/{sanitized}.json``.

        Returns the dossier dict if the cache file exists and is within
        the TTL window, otherwise ``None``.
        """
        sanitized = self._sanitize_topic(topic)
        cache_path = self.cache_dir / f"{sanitized}.json"

        if not cache_path.exists():
            return None

        try:
            dossier = load_json(cache_path)
        except Exception as e:
            log.warning("Failed to load cached dossier at %s: %s", cache_path, e)
            return None

        if not isinstance(dossier, dict):
            return None

        # Validate TTL
        researched_at = dossier.get("researched_at")
        if researched_at:
            try:
                cached_time = datetime.fromisoformat(researched_at)
                age_days = (datetime.now(timezone.utc) - cached_time).days
                if age_days > self.cache_ttl_days:
                    log.info(
                        "Cached dossier for '%s' expired (%d days old, ttl=%d)",
                        topic,
                        age_days,
                        self.cache_ttl_days,
                    )
                    return None
            except (ValueError, TypeError):
                log.warning("Invalid timestamp in cached dossier — treating as expired")
                return None

        log.debug("Cache hit for topic '%s' at %s", topic, cache_path)
        return dossier

    def _save_cached_dossier(self, topic: str, dossier: dict) -> None:
        """Save a dossier to the cache directory as JSON."""
        sanitized = self._sanitize_topic(topic)
        cache_path = self.cache_dir / f"{sanitized}.json"
        save_json(cache_path, dossier)
        log.info("Cached dossier → %s", cache_path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _sanitize_topic(self, topic: str) -> str:
        """
        Convert a topic string into a safe filename slug.

        Lowercases, replaces spaces with underscores, strips special
        characters, and truncates to 80 chars.
        """
        name = topic.lower().strip()
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s-]+", "_", name)
        name = name[:80].rstrip("_")
        return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Web Researcher — research topics using web search and LLM",
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic to research (e.g. 'Why did Rick destroy the Citadel?')",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="Show slug from show_config.yaml (default: first active show)",
    )
    args = parser.parse_args()

    pipeline_config = load_pipeline_config()
    slug, show = get_active_show(args.show)
    show_name = show.get("display_name", slug)

    log.info("=== Web Research for '%s' (show: %s) ===", args.topic, show_name)

    researcher = WebResearcher(pipeline_config)
    dossier = researcher.research_topic(args.topic, show_name)

    # Print formatted JSON to stdout
    print(json.dumps(dossier, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
