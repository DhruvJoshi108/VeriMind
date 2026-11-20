from eval import normalize_wiki_title, retrieval_hit, summarize, wilson_interval


def test_wilson_interval_covers_sample_accuracy():
    lower, upper = wilson_interval(8, 10)

    assert lower < 0.8 < upper


def test_wilson_interval_has_no_value_for_empty_sample():
    assert wilson_interval(0, 0) is None


def test_retrieval_hit_matches_wikipedia_slug_and_title():
    passages = [{"doc_id": "https://en.wikipedia.org/wiki/Paris", "text": "Paris"}]

    assert retrieval_hit(passages, ["Paris"])
    assert normalize_wiki_title("https://en.wikipedia.org/wiki/New_York_City") == "new york city"


def test_failures_are_counted_without_becoming_label_predictions():
    records = [
        {
            "true_label": "refuted",
            "predicted_label": None,
            "error": {"category": "pipeline_error"},
            "retrieval_hit": False,
            "latency_seconds": 0.1,
        }
    ]

    summary = summarize(records)

    assert summary["n_errors"] == 1
    assert summary["accuracy"] is None
    assert summary["class_metrics"]["refuted"]["support"] == 0
    assert summary["error_categories"] == {"pipeline_error": 1}
