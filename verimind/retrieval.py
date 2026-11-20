"""Evidence retrieval layer combining local knowledge and web search."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from verimind.fts_retrieval import SQLiteFtsEvidenceRetriever
from verimind.faiss_retrieval import FaissEvidenceRetriever
from verimind.knowledge_base import KnowledgeBase
from verimind.web_search import get_search_backend

logger = logging.getLogger(__name__)
INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"
FULL_TEXT_INDEX_PATH = INDEX_DIR / "wiki_fts.sqlite3"


@lru_cache(maxsize=1)
def load_faiss_retriever() -> FaissEvidenceRetriever | None:
    if not (INDEX_DIR / "index_manifest.json").is_file():
        return None
    return FaissEvidenceRetriever.from_index_dir(INDEX_DIR)


@lru_cache(maxsize=1)
def load_full_text_retriever() -> SQLiteFtsEvidenceRetriever | None:
    if not FULL_TEXT_INDEX_PATH.is_file():
        return None
    return SQLiteFtsEvidenceRetriever(FULL_TEXT_INDEX_PATH)


class LocalEvidenceRetriever:
    """Retrieve local evidence with TF-IDF ranking."""

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self.kb = knowledge_base or KnowledgeBase()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fit = False

    def fit(self, docs: list[str]) -> None:
        self.vectorizer.fit(docs)
        self._fit = True

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        documents = [fact["text"] for fact in self.kb.facts]
        if not documents:
            return []
        if not self._fit:
            self.fit(documents)
        query_vec = self.vectorizer.transform([query])
        doc_vec = self.vectorizer.transform(documents)
        scores = cosine_similarity(query_vec, doc_vec).ravel()
        ranked = sorted(
            ((idx, score) for idx, score in enumerate(scores) if score > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            fact = self.kb.facts[idx]
            results.append({
                "text": fact["text"],
                "source": fact["source"],
                "domain": fact["domain"],
                "score": float(score),
            })
        return results


class HybridRetriever:
    """Combine local retrieval with live-web results while deduplicating sources."""

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self.local = LocalEvidenceRetriever(knowledge_base)
        self.indexed = load_faiss_retriever()
        self.full_text = load_full_text_retriever()
        self.backend = get_search_backend()

    def build_queries(self, claim: str) -> list[str]:
        """Create deterministic full, keyword, and entity-oriented query variants."""
        cleaned = re.sub(r"[^\w\s]", " ", claim).strip()
        tokens = [token for token in cleaned.split() if len(token) > 2]
        keywords = " ".join(tokens[:8])
        entities = " ".join(re.findall(r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*", claim))
        queries = [claim.strip(), keywords, entities]
        return list(dict.fromkeys(query for query in queries if query))[:3]

    @staticmethod
    def _key(item: dict[str, Any]) -> str:
        return str(item.get("source") or item.get("doc_id") or item.get("title") or item["text"])

    def _add_ranked(
        self,
        merged: dict[str, dict[str, Any]],
        items: list[dict[str, Any]],
        backend: str,
    ) -> None:
        for rank, item in enumerate(items, start=1):
            key = self._key(item)
            candidate = merged.setdefault(key, {**item, "retrieval_sources": [], "rrf_score": 0.0})
            candidate["rrf_score"] = float(candidate["rrf_score"]) + 1 / (60 + rank)
            candidate["retrieval_sources"].append({"backend": backend, "rank": rank})

    def search(self, claim: str, k: int = 5) -> list[dict[str, Any]]:
        """Return reciprocal-rank-fused evidence and attach audit diagnostics."""
        merged: dict[str, dict[str, Any]] = {}
        variants = self.build_queries(claim)
        for query in variants:
            self._add_ranked(merged, self.local.search(query, k=k), "local")
            if self.indexed is not None:
                self._add_ranked(merged, self.indexed.search(query, k=k), "faiss")
            if self.full_text is not None:
                self._add_ranked(merged, self.full_text.search(query, k=k), "fulltext")
            try:
                web_items = [
                    {
                        "text": item.text,
                        "source": item.url or item.source,
                        "domain": item.domain,
                        "score": float(item.score or 0.5),
                    }
                    for item in self.backend.search(query, k=k)
                ]
                self._add_ranked(merged, web_items, "web")
            except requests.RequestException as exc:
                logger.warning("Web evidence retrieval failed; continuing with local evidence: %s", exc)
        ranked = sorted(
            merged.values(),
            key=lambda item: (float(item["rrf_score"]), float(item.get("score", 0.0))),
            reverse=True,
        )[:k]
        for item in ranked:
            item["query_variants"] = variants
        return ranked
