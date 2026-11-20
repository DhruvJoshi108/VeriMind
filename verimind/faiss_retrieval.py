"""Validated FAISS retrieval for the local Wikipedia passage index."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HASH_EMBEDDING_BACKEND = "sha256-token-hash-v1"
INNER_PRODUCT_METRIC = "inner_product"
MANIFEST_FILENAME = "index_manifest.json"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
MIN_QUERY_TERM_COVERAGE = 0.7


def informative_terms(text: str) -> frozenset[str]:
    return frozenset(
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
        if token.casefold() not in ENGLISH_STOP_WORDS
    )


def required_term_overlap(query_terms: frozenset[str]) -> int:
    return max(1, math.ceil(len(query_terms) * MIN_QUERY_TERM_COVERAGE))


def hash_text_vector(text: str, dimension: int) -> np.ndarray:
    """Encode text using the deterministic fallback used by the index builder."""
    vector = np.zeros(dimension, dtype=np.float32)
    for token in TOKEN_PATTERN.findall(text.lower()):
        for offset in range(8):
            digest = hashlib.sha256(f"{token}:{offset}".encode()).digest()
            index = int.from_bytes(digest[:8], byteorder="little", signed=False) % dimension
            vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def has_lexical_match(query: str, passage: str) -> bool:
    """Prevent hash-vector collisions from returning passages with no shared terms."""
    query_terms = informative_terms(query)
    if not query_terms:
        return False
    passage_terms = informative_terms(passage)
    minimum_overlap = required_term_overlap(query_terms)
    return len(query_terms & passage_terms) >= minimum_overlap


class FaissEvidenceRetriever:
    """Load and query an index only when its embedding contract is explicit."""

    def __init__(
        self,
        index: Any,
        passages: list[dict[str, Any]],
        manifest: dict[str, Any],
        model: Any = None,
    ) -> None:
        self.index = index
        self.passages = passages
        self.manifest = manifest
        self.model = model
        self.passage_terms = tuple(
            informative_terms(str(passage.get("text", ""))) for passage in passages
        )
        self.lexical_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.lexical_passage_vectors = self.lexical_vectorizer.fit_transform(
            [
                f"{passage.get('title', '')} {passage.get('text', '')}"
                for passage in passages
            ]
        )

    @classmethod
    def from_index_dir(cls, index_dir: Path) -> FaissEvidenceRetriever:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("Install faiss-cpu to load the local evidence index.") from error

        manifest_path = index_dir / MANIFEST_FILENAME
        index_path = index_dir / "faiss.index"
        passages_path = index_dir / "passages_meta.jsonl"
        for path in (manifest_path, index_path, passages_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required retrieval index file is missing: {path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))
        passages = [
            json.loads(line)
            for line in passages_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if manifest.get("metric") != INNER_PRODUCT_METRIC:
            raise ValueError(f"Unsupported FAISS metric in {manifest_path}")
        if manifest.get("normalized_embeddings") is not True:
            raise ValueError(f"Index vectors are not declared normalized in {manifest_path}")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS index metric does not match its manifest")
        if index.d != manifest.get("dimension"):
            raise ValueError("FAISS index dimension does not match its manifest")
        if index.ntotal != len(passages) or index.ntotal != manifest.get("passage_count"):
            raise ValueError("FAISS index, passage metadata, and manifest counts differ")

        backend = manifest.get("embedding_backend")
        if backend == HASH_EMBEDDING_BACKEND:
            model = None
        elif backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            model_name = manifest.get("embedding_model")
            if not isinstance(model_name, str) or not model_name:
                raise ValueError("Sentence-transformer model name is missing from the index manifest")
            model = SentenceTransformer(model_name, local_files_only=True)
            if model.get_embedding_dimension() != index.d:
                raise ValueError("Query embedding dimension does not match the FAISS index")
        else:
            raise ValueError(f"Unsupported embedding backend in {manifest_path}: {backend!r}")

        return cls(index, passages, manifest, model)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be positive")
        query_terms = informative_terms(query)
        if not query_terms:
            return []
        minimum_overlap = required_term_overlap(query_terms)
        candidate_positions = [
            position
            for position, terms in enumerate(self.passage_terms)
            if len(query_terms & terms) >= minimum_overlap
        ]
        if not candidate_positions:
            return []
        lexical_scores = cosine_similarity(
            self.lexical_vectorizer.transform([query]),
            self.lexical_passage_vectors,
        ).ravel()

        if self.model is None:
            vector = hash_text_vector(query, self.index.d)
            query_vector = vector.reshape(1, -1)
        else:
            query_vector = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            if query_vector.shape != (1, self.index.d):
                raise ValueError("Query embedding dimension does not match the FAISS index")
            if not np.allclose(np.linalg.norm(query_vector, axis=1), 1.0, atol=1e-5):
                raise ValueError("Query embedding is not normalized as required by the index")

        scores, positions = self.index.search(query_vector, self.index.ntotal)
        candidate_set = set(candidate_positions)
        ranked = [
            (float(lexical_scores[int(position)]), float(score), int(position))
            for score, position in zip(scores[0], positions[0], strict=True)
            if int(position) in candidate_set and lexical_scores[int(position)] > 0.0
        ]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            {
                **self.passages[position],
                "score": semantic_score,
                "lexical_score": lexical_score,
            }
            for lexical_score, semantic_score, position in ranked[:k]
        ]
