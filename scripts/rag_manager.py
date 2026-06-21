"""
rag_manager.py — Handles Retrieval-Augmented Generation for script accuracy.
"""

import os
import json
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

from config_loader import get_project_path, setup_logging

log = setup_logging("rag_manager")

class RAGManager:
    def __init__(self, pipeline_config: dict):
        self.db_dir = get_project_path("vector_db_dir", pipeline_config)
        self.subtitles_dir = get_project_path("subtitles_dir", pipeline_config)
        self.theories_path = get_project_path("theories_db", pipeline_config)
        self.wiki_path = get_project_path("wiki_db", pipeline_config)
        
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)
        self.theories_path.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize theories file if not exists
        if not self.theories_path.exists():
            with open(self.theories_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
                
        # Initialize wiki file if not exists
        if not self.wiki_path.exists():
            with open(self.wiki_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        
        log.info(f"Initializing ChromaDB at {self.db_dir}")
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # We use default embedding function (all-MiniLM-L6-v2) for simplicity and speed locally
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="subtitles", 
            embedding_function=self.emb_fn
        )

    def ingest_subtitles(self, chunk_size=500):
        """Reads .txt or .srt files from the subtitles_dir and indexes them."""
        log.info(f"Starting ingestion from {self.subtitles_dir}")
        files = list(self.subtitles_dir.glob("*.*"))
        if not files:
            log.warning("No subtitle files found. Please place .txt or .srt files in the subtitles directory.")
            return

        for filepath in files:
            if filepath.suffix not in ['.txt', '.srt']:
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking by character length (could be improved to chunk by sentences/timecodes)
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            ids = [f"{filepath.stem}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filepath.name} for _ in chunks]
            
            log.info(f"Adding {len(chunks)} chunks for {filepath.name}")
            self.collection.upsert(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        log.info("Ingestion complete.")

    def query_subtitles(self, query: str, n_results=3) -> str:
        """Queries the vector DB for context matching the given query."""
        if self.collection.count() == 0:
            return ""
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        docs = results['documents'][0]
        sources = results['metadatas'][0]
        
        context_str = "--- Canon Subtitle Context ---\n"
        for doc, meta in zip(docs, sources):
            context_str += f"[Source: {meta['source']}]: {doc.strip()}\n\n"
            
        return context_str

    def query_theories(self, query: str) -> str:
        """Queries the theories JSON for matching topics."""
        with open(self.theories_path, 'r', encoding='utf-8') as f:
            theories = json.load(f)
            
        # Basic keyword matching (can be upgraded to semantic search later)
        # Remove common stop words for better matching
        stop_words = {"what", "why", "who", "when", "how", "the", "a", "an", "is", "are", "do", "does", "did", "in", "on", "at", "to", "for", "of", "and", "or"}
        keywords = {w for w in query.lower().split() if w not in stop_words}
        matched_theories = []
        
        for topic, theory_text in theories.items():
            topic_words = {w for w in topic.lower().split() if w not in stop_words}
            if keywords.intersection(topic_words):
                matched_theories.append(f"Theory on {topic}: {theory_text}")
                
        if not matched_theories:
            return ""
            
        theory_str = "--- Speculative Theories & Conspiracies ---\n"
        theory_str += "IMPORTANT: The following are fan theories. When mentioning these, clearly state that they are theories or 'it is believed that...', and have not been confirmed in the actual show.\n\n"
        theory_str += "\n\n".join(matched_theories)
        return theory_str
        
    def query_wiki(self, query: str) -> str:
        """Queries the wiki JSON for basic character/topic info."""
        with open(self.wiki_path, 'r', encoding='utf-8') as f:
            wiki_data = json.load(f)
            
        stop_words = {"what", "why", "who", "when", "how", "the", "a", "an", "is", "are", "do", "does", "did", "in", "on", "at", "to", "for", "of", "and", "or"}
        keywords = {w for w in query.lower().split() if w not in stop_words}
        matched_wiki = []
        
        for topic, wiki_text in wiki_data.items():
            topic_words = {w for w in topic.lower().split() if w not in stop_words}
            if keywords.intersection(topic_words):
                matched_wiki.append(f"Basic Info about {topic}: {wiki_text}")
                
        if not matched_wiki:
            return ""
            
        wiki_str = "--- General Wiki Context (Characters/Lore) ---\n"
        wiki_str += "\n\n".join(matched_wiki)
        return wiki_str

    def get_combined_context(self, query: str) -> str:
        canon = self.query_subtitles(query)
        wiki = self.query_wiki(query)
        theories = self.query_theories(query)
        
        combined = []
        if canon:
            combined.append(canon)
        if wiki:
            combined.append(wiki)
        if theories:
            combined.append(theories)
            
        if not combined:
            return "No additional context found. Rely on your internal knowledge."
            
        return "\n\n".join(combined)

if __name__ == "__main__":
    # If run directly, run ingestion
    from config_loader import load_pipeline_config
    config = load_pipeline_config()
    rag = RAGManager(config)
    rag.ingest_subtitles()
