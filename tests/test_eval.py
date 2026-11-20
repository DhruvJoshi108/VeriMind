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


def test_summarize_reports_retrieval_recall_at_k_without_using_unverifiable_cases():
    records = [
        {
            "true_label": "refuted",
            "predicted_label": "unverifiable",
            "retrieval_hit": True,
            "retrieval_evaluable": True,
            "retrieval_hits_at_k": {
                "1": False,
                "3": True,
                "5": True,
                "10": True,
            },
            "error": None,
            "latency_seconds": 0.1,
        },
        {
            "true_label": "unverifiable",
            "predicted_label": "unverifiable",
            "retrieval_hit": None,
            "retrieval_evaluable": False,
            "retrieval_hits_at_k": {"1": None, "3": None, "5": None, "10": None},
            "error": None,
            "latency_seconds": 0.1,
        },
    ]

    summary = summarize(records)

    assert summary["retrieval_recall_at_k"] == {
        "1": 0.0,
        "3": 1.0,
        "5": 1.0,
        "10": 1.0,
    }
