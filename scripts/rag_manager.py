"""
rag_manager.py — Advanced Hybrid RAG Engine (Vectorized Lore + HyDE + Llama Context Distillation).

Manages 4 dedicated ChromaDB vector collections:
  1. subtitles: Episode dialogue transcripts chunked by sentence boundaries.
  2. episodes: Canonical episode titles and plot summaries.
  3. wiki: Scraped character lore and science rules.
  4. theories: Scraped fan theories and trivia from topics/theories.json.

Topic Lifecycle:
  Query → HyDE Monologue Expansion → ChromaDB Multi-Vector Retrieval → Llama Context Distiller → Clean Lore Dossier.
"""

import json
import os
import re
import chromadb
import requests
from pathlib import Path
from chromadb.utils import embedding_functions

from config_loader import get_project_path, setup_logging, load_json

log = setup_logging("rag_manager")


class RAGManager:
    def __init__(self, pipeline_config: dict):
        self.cfg = pipeline_config
        self.db_dir = get_project_path("vector_db_dir", pipeline_config)
        self.subtitles_dir = get_project_path("subtitles_dir", pipeline_config)
        self.theories_path = get_project_path("theories_db", pipeline_config)
        self.wiki_path = get_project_path("wiki_db", pipeline_config)
        
        # Episode index resolution
        ep_cfg = pipeline_config.get("episode_index", {})
        if "episode_index" in pipeline_config.get("paths", {}):
            self.episode_index_path = get_project_path("episode_index", pipeline_config)
        else:
            ep_p = Path(ep_cfg.get("path", "./episode_index.json"))
            self.episode_index_path = ep_p.resolve() if ep_p.is_absolute() else self.db_dir.parent / ep_p.lstrip("./")

        # Create dirs
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)
        self.theories_path.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure JSON files exist
        for p in [self.theories_path, self.wiki_path]:
            if not p.exists():
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({}, f)

        log.info("Initializing Persistent ChromaDB engine at %s", self.db_dir)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize 4 dedicated vector collections
        self.col_subtitles = self.client.get_or_create_collection("subtitles", embedding_function=self.emb_fn)
        self.col_episodes = self.client.get_or_create_collection("episodes", embedding_function=self.emb_fn)
        self.col_wiki = self.client.get_or_create_collection("wiki", embedding_function=self.emb_fn)
        self.col_theories = self.client.get_or_create_collection("theories", embedding_function=self.emb_fn)

        self.ollama_url = pipeline_config.get("ollama", {}).get("base_url", "http://localhost:11434")
        self.ollama_model = (
            pipeline_config.get("llm", {}).get("ollama", {}).get("model")
            or "llama3.1:8b"
        )

    # ── Ingestion Engine ──────────────────────────────────────────────────────

    def ingest_subtitles(self, chunk_size=1000):
        """Indexes subtitles along sentence boundaries (~1000 chars)."""
        log.info("Ingesting subtitles from %s", self.subtitles_dir)
        files = list(self.subtitles_dir.glob("*.*"))
        if not files:
            log.warning("No subtitle files found in %s", self.subtitles_dir)
            return

        for fp in files:
            if fp.suffix not in [".txt", ".srt"]:
                continue
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                content = re.sub(r"<[^>]+>", "", f.read())

            # Smart sentence chunking
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
            chunks = []
            curr = []
            curr_len = 0
            for s in sentences:
                if curr_len + len(s) > chunk_size and curr:
                    chunks.append(" ".join(curr))
                    curr = [s]
                    curr_len = len(s)
                else:
                    curr.append(s)
                    curr_len += len(s)
            if curr:
                chunks.append(" ".join(curr))

            ids = [f"{fp.stem}_sub_{i}" for i in range(len(chunks))]
            metas = [{"source": fp.name, "type": "canon_sub"} for _ in chunks]

            log.info("Upserting %d sentence chunks for %s", len(chunks), fp.name)
            self.col_subtitles.upsert(documents=chunks, metadatas=metas, ids=ids)

    def ingest_json_database(self, collection, filepath: Path, doc_type: str):
        """Vectorizes key-value lore dictionaries or lists of JSON objects."""
        if not filepath.exists():
            return
        data = load_json(filepath)
        if not data:
            return

        # Support both {"Key": {...}} and [{"title": "...", ...}] formats
        if isinstance(data, list):
            items = []
            for item in data:
                if isinstance(item, dict):
                    key = item.get("title", "Unknown Topic")
                    items.append((key, item))
        elif isinstance(data, dict):
            items = data.items()
        else:
            return

        docs = []
        ids = []
        metas = []
        for i, (key, val) in enumerate(items):
            clean_key = str(key).strip()
            if isinstance(val, dict):
                content = val.get('content', val.get('summary', val.get('one_line', '')))
                text = f"Title: {val.get('title', clean_key)}. Summary: {content}"
            else:
                text = f"Topic: {clean_key} — {str(val).strip()}"
            
            if len(text) < 10:
                continue
            docs.append(text)
            safe_key = re.sub(r'[^a-zA-Z0-9]', '', clean_key[:20])
            ids.append(f"{doc_type}_{i}_{safe_key}")
            metas.append({"source": fp_name(filepath), "topic": clean_key[:50]})

        if docs:
            log.info("Upserting %d vector entries into Chroma collection [%s]", len(docs), collection.name)
            batch_size = 5000
            for i in range(0, len(docs), batch_size):
                collection.upsert(
                    documents=docs[i:i+batch_size], 
                    metadatas=metas[i:i+batch_size], 
                    ids=ids[i:i+batch_size]
                )

    def ingest_all(self):
        """Runs full RAG vectorization across all 4 databases."""
        log.info("Starting total multiversal RAG vector ingestion...")
        self.ingest_subtitles()
        self.ingest_json_database(self.col_episodes, self.episode_index_path, "ep")
        self.ingest_json_database(self.col_wiki, self.wiki_path, "wiki")
        self.ingest_json_database(self.col_theories, self.theories_path, "theory")
        log.info("Total RAG vectorization complete!")

    # ── Intelligence Pre-Pass (HyDE + Distillation) ───────────────────────────

    def _call_ollama(self, prompt: str, temp=0.3, timeout=12) -> str:
        """Fast local Ollama inference helper."""
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": temp}}
        try:
            res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=timeout)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
        except Exception as e:
            log.debug("Ollama pre-pass bypassed (%s)", e)
        return ""

    def _generate_hyde(self, topic: str) -> str:
        """HyDE: Generates a hypothetical show monologue answering the topic.
        
        Dynamically uses the active show's characters and display name
        so the hypothetical document is grounded in the correct universe.
        """
        # Pull active show info for a grounded HyDE prompt
        show_name = "Rick and Morty"  # fallback
        character_names = "Rick Sanchez, Morty Smith"  # fallback
        try:
            from config_loader import load_show_config, get_active_show
            show_cfg = load_show_config()
            show_slug, show_data = get_active_show(show_cfg)
            show_name = show_data.get("display_name", show_name)
            chars = show_data.get("characters", [])
            if chars:
                top_chars = [c["name"] for c in chars[:5]]
                character_names = ", ".join(top_chars)
        except Exception:
            pass

        prompt = (
            f"You are an expert analyst of the TV show '{show_name}'. "
            f"The main characters are: {character_names}.\n\n"
            f"Write a detailed 3-sentence dramatic monologue that a narrator would speak "
            f"answering or explaining this topic: '{topic}'.\n"
            f"Include specific references to episodes, scenes, character motivations, "
            f"and plot events. Do not write commentary — write ONLY the monologue itself."
        )
        return self._call_ollama(prompt, temp=0.5, timeout=12)

    def _distill(self, topic: str, raw_context: str) -> str:
        """Context Distillation: Filters raw vector hits into 4 undeniable canonical bullets."""
        if not raw_context or len(raw_context) < 50:
            return raw_context
        prompt = (
            f"You are an expert lore archivist. Given the raw scraped excerpts below, extract EXACTLY 4 "
            f"undeniable factual canonical bullet points relevant to the topic: '{topic}'.\n"
            f"Discard all random dialogue noise. If fan theories are included, state 'Theory: ...'.\n\n"
            f"RAW EXCERPTS:\n{raw_context[:3500]}"
        )
        distilled = self._call_ollama(prompt, temp=0.1, timeout=15)
        return distilled if distilled else raw_context

    # ── Query & Retrieval Engine ──────────────────────────────────────────────

    def _query_collection(self, col, query_texts: list, n=3) -> list:
        if col.count() == 0:
            return []
        try:
            res = col.query(query_texts=query_texts, n_results=min(n, col.count()))
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            return [f"[{m.get('source', 'DB')}]: {d.strip()}" for d, m in zip(docs, metas) if d]
        except Exception as e:
            log.debug("Collection %s query error: %s", col.name, e)
            return []

    def get_combined_context(self, query: str, dossier: dict = None) -> str:
        """Master RAG Retrieval Method called by script_generator.py."""
        log.info("🔍 Running Hybrid RAG retrieval for topic: '%s'", query[:60])
        
        # 1. HyDE Expansion
        hyde_text = self._generate_hyde(query)
        search_queries = [query]
        if hyde_text:
            log.debug("Generated HyDE vector query: '%s'", hyde_text[:60])
            search_queries.append(hyde_text)

        # 2. Multi-Vector Collection Pull
        hits_sub = self._query_collection(self.col_subtitles, search_queries, n=4)
        hits_ep = self._query_collection(self.col_episodes, search_queries, n=3)
        hits_wiki = self._query_collection(self.col_wiki, search_queries, n=3)
        hits_th = self._query_collection(self.col_theories, search_queries, n=3)

        raw_lore = "\n\n".join(hits_ep + hits_wiki + hits_sub + hits_th)
        if not raw_lore:
            log.warning("No RAG vector matches found. Returning baseline.")
            return "No local database entries matched. Rely on internal model knowledge."

        # 3. Llama Context Distillation
        log.info("Distilling %d raw vector excerpts into verified canon dossier...", len(hits_sub + hits_ep + hits_wiki + hits_th))
        distilled_dossier = self._distill(query, raw_lore)

        final_sections = []
        if dossier:
            try:
                from web_researcher import format_dossier_for_prompt
                final_sections.append("--- Live Web Search Dossier ---\n" + format_dossier_for_prompt(dossier))
            except Exception:
                pass

        final_sections.append("--- Verified Multiversal Canon Dossier (Distilled via RAG) ---\n" + distilled_dossier)
        return "\n\n".join(final_sections)


def fp_name(path: Path) -> str:
    return path.name if path else "JSON"


if __name__ == "__main__":
    from config_loader import load_pipeline_config
    cfg = load_pipeline_config()
    rag = RAGManager(cfg)
    rag.ingest_all()
