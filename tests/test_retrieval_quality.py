import json
from unittest.mock import Mock
from zipfile import ZipFile

import numpy as np

from eval import retrieval_hit, summarize
from verimind.faiss_retrieval import FaissEvidenceRetriever, has_lexical_match
from verimind.fts_retrieval import SQLiteFtsEvidenceRetriever, build_full_text_index
from verimind.retrieval import LocalEvidenceRetriever
from verimind.web_search import WikipediaSearchBackend


class _FixedScoresIndex:
    d = 2

    def __init__(self, ntotal=3):
        self.ntotal = ntotal

    def search(self, _query, _k):
        scores = np.linspace(0.9, 0.1, self.ntotal, dtype=np.float32)
        positions = np.arange(self.ntotal, dtype=np.int64)
        return scores.reshape(1, -1), positions.reshape(1, -1)


def test_distinct_claims_retrieve_distinct_relevant_local_facts():
    retriever = LocalEvidenceRetriever()

    france = retriever.search("capital of France", k=3)
    germany = retriever.search("capital of Germany", k=3)

    assert france
    assert germany
    assert "France" in france[0]["text"]
    assert "Germany" in germany[0]["text"]
    assert france[0]["text"] != germany[0]["text"]


def test_query_variants_are_deduplicated_and_include_keywords():
    from verimind.retrieval import HybridRetriever

    retriever = HybridRetriever()
    variants = retriever.build_queries("Paris is the capital of France.")

    assert variants[0] == "Paris is the capital of France."
    assert len(variants) == len(set(variants))
    assert any("Paris" in variant and "France" in variant for variant in variants)


def test_no_lexical_match_does_not_return_unrelated_passages():
    retriever = LocalEvidenceRetriever()

    results = retriever.search("Sicily largest island Mediterranean Sea", k=5)

    assert results == []


def test_hash_retrieval_filters_unrelated_passages_and_keeps_relevant_results():
    passages = [
        {"doc_id": "paris", "text": "Paris is the capital city of France."},
        {"doc_id": "london", "text": "London is the capital of the United Kingdom."},
        {"doc_id": "france", "text": "France is a country in Europe."},
    ]
    retriever = FaissEvidenceRetriever(
        _FixedScoresIndex(),
        passages,
        {
            "embedding_backend": "sha256-token-hash-v1",
            "dimension": 2,
            "normalized_embeddings": True,
            "metric": "inner_product",
            "passage_count": len(passages),
        },
    )

    paris = retriever.search("capital of France")
    london = retriever.search("London is the capital of the United Kingdom")
    unrelated = retriever.search("Sicily Mediterranean island")

    assert paris[0]["doc_id"] == "paris"
    assert london[0]["doc_id"] == "london"
    assert paris[0]["doc_id"] != london[0]["doc_id"]
    assert [passage["doc_id"] for passage in london] == ["london"]
    assert unrelated == []
    assert has_lexical_match(
        "The Godfather Part III only stars male actors",
        passages[0]["text"],
    ) is False


def test_distinct_claims_return_distinct_topically_relevant_index_passages():
    passages = [
        {
            "doc_id": "Paris",
            "title": "Paris",
            "text": "Paris is the capital city of France and home to the Eiffel Tower.",
        },
        {
            "doc_id": "Berlin",
            "title": "Berlin",
            "text": "Berlin is the capital city of Germany and lies on the Spree.",
        },
        {
            "doc_id": "Sicily",
            "title": "Sicily",
            "text": "Sicily is the largest island in the Mediterranean Sea.",
        },
    ]
    retriever = FaissEvidenceRetriever(
        _FixedScoresIndex(),
        passages,
        {
            "embedding_backend": "sha256-token-hash-v1",
            "dimension": 2,
            "normalized_embeddings": True,
            "metric": "inner_product",
            "passage_count": len(passages),
        },
    )

    paris = retriever.search("Paris capital France")
    berlin = retriever.search("Berlin capital Germany")
    sicily = retriever.search("Sicily Mediterranean island")

    assert paris[0]["doc_id"] == "Paris"
    assert berlin[0]["doc_id"] == "Berlin"
    assert sicily[0]["doc_id"] == "Sicily"
    assert len({paris[0]["doc_id"], berlin[0]["doc_id"], sicily[0]["doc_id"]}) == 3


def test_lexical_reranking_prefers_specific_evidence_to_related_distractor():
    passages = [
        {
            "doc_id": "Paris_attacks",
            "title": "Paris attacks",
            "text": "Terror attacks occurred in Paris, France.",
        },
        {
            "doc_id": "Paris",
            "title": "Paris",
            "text": "Paris is the capital city of France.",
        },
        {
            "doc_id": "France",
            "title": "France",
            "text": "France is a country in Europe.",
        },
    ]
    retriever = FaissEvidenceRetriever(
        _FixedScoresIndex(len(passages)),
        passages,
        {
            "embedding_backend": "sha256-token-hash-v1",
            "dimension": 2,
            "normalized_embeddings": True,
            "metric": "inner_product",
            "passage_count": len(passages),
        },
    )

    result = retriever.search("Paris is the capital of France", k=3)

    assert result[0]["doc_id"] == "Paris"
    assert [passage["doc_id"] for passage in result] == ["Paris"]


def test_retrieval_hit_is_unknown_when_fever_has_no_gold_page():
    assert retrieval_hit([{"doc_id": "paris", "text": "Paris"}], []) is None

    summary = summarize(
        [
            {
                "true_label": "refuted",
                "predicted_label": "unverifiable",
                "retrieval_hit": None,
                "retrieval_evaluable": False,
                "error": None,
                "latency_seconds": 0.1,
            }
        ]
    )

    assert summary["retrieval_evaluable_count"] == 0
    assert summary["retrieval_cases_without_gold_evidence"] == 1
    assert summary["retrieval_hit_rate_verifiable_claims"] is None


def test_full_text_index_retrieves_from_the_complete_archive(tmp_path):
    archive_path = tmp_path / "wiki-pages.zip"
    index_path = tmp_path / "wiki_fts.sqlite3"
    pages = [
        {
            "id": "Paris",
            "text": "Paris is the capital city of France and is located on the Seine River.",
        },
        {
            "id": "Berlin",
            "text": "Berlin is the capital city of Germany and lies on the Spree River.",
        },
    ]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "wiki-pages/wiki-001.jsonl",
            "\n".join(json.dumps(page) for page in pages),
        )

    manifest = build_full_text_index(archive_path, index_path)
    retriever = SQLiteFtsEvidenceRetriever(index_path)

    results = retriever.search("Paris capital of France", k=2)

    assert manifest["page_count"] == 2
    assert manifest["passage_count"] == 2
    assert results[0]["doc_id"] == "Paris"
    assert results[0]["source"] == "https://en.wikipedia.org/wiki/Paris"
    assert retriever.search("!!!", k=2) == []


def test_hybrid_retriever_includes_full_text_candidates():
    from verimind.retrieval import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.local = Mock()
    retriever.local.search.return_value = []
    retriever.indexed = None
    retriever.full_text = Mock()
    retriever.full_text.search.return_value = [
        {
            "doc_id": "Paris",
            "title": "Paris",
            "text": "Paris is the capital of France.",
            "source": "https://en.wikipedia.org/wiki/Paris",
            "score": 0.8,
        }
    ]
    retriever.backend = Mock()
    retriever.backend.search.return_value = []

    results = retriever.search("Paris is the capital of France", k=1)

    assert results[0]["doc_id"] == "Paris"
    assert any(
        source["backend"] == "fulltext"
        for source in results[0]["retrieval_sources"]
    )


def test_wikipedia_backend_searches_only_the_full_claim_variant():
    from verimind.retrieval import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.local = Mock()
    retriever.local.search.return_value = []
    retriever.indexed = None
    retriever.full_text = None
    retriever.backend = WikipediaSearchBackend()
    retriever.backend.search = Mock(return_value=[])

    retriever.search("Paris is the capital of France", k=1)

    retriever.backend.search.assert_called_once_with("Paris is the capital of France", k=1)
