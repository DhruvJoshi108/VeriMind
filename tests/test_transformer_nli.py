from dataclasses import replace

import pytest

from verimind import pipeline as pipeline_module
from verimind.config import Settings
from verimind.transformer_nli import TransformerNLIModel


def test_transformer_nli_preserves_model_label_order():
    labels = TransformerNLIModel._ordered_labels(
        {"0": "entailment", "1": "neutral", "2": "contradiction"}
    )

    assert labels == ["entailment", "neutral", "contradiction"]


def test_transformer_nli_rejects_non_nli_label_sets():
    with pytest.raises(ValueError, match="must expose"):
        TransformerNLIModel._ordered_labels(
            {"0": "positive", "1": "neutral", "2": "negative"}
        )


def test_settings_accepts_transformer_nli_backend(monkeypatch):
    monkeypatch.setenv("NLI_BACKEND", "transformer")
    monkeypatch.setenv("NLI_DEVICE", "cuda")
    monkeypatch.setenv(
        "NLI_MODEL_NAME",
        "MoritzLaurer/DeBERTa-v3-base-mnli",
    )

    settings = Settings.from_env()

    assert settings.nli_backend == "transformer"
    assert settings.nli_device == "cuda"
    assert settings.nli_model_name == "MoritzLaurer/DeBERTa-v3-base-mnli"


def test_settings_accepts_local_model_path(monkeypatch):
    monkeypatch.setenv("NLI_MODEL_PATH", "artifacts/nli_model_fever.pkl")

    settings = Settings.from_env()

    assert settings.nli_model_path == "artifacts/nli_model_fever.pkl"


def test_settings_rejects_unknown_nli_backend(monkeypatch):
    monkeypatch.setenv("NLI_BACKEND", "ensemble-everything")

    with pytest.raises(ValueError, match="NLI_BACKEND"):
        Settings.from_env()


def test_pipeline_loads_selected_transformer_backend(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        pipeline_module,
        "settings",
        replace(pipeline_module.settings, nli_backend="transformer"),
    )
    monkeypatch.setattr(
        pipeline_module.TransformerNLIModel,
        "load",
        lambda model: expected,
    )

    loaded = pipeline_module.VerificationPipeline.__new__(
        pipeline_module.VerificationPipeline
    )._load_nli_model()

    assert loaded is expected
