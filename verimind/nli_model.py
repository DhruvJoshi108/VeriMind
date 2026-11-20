"""Trainable hybrid NLI classifier for claim-evidence verification."""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from verimind.features import (
    cosine_similarity,
    entity_overlap,
    negation_mismatch,
    numeric_mismatch,
    token_overlap,
)

_LABELS = ("entailment", "contradiction", "neutral")


class NLIModel:
    """Offline NLI-style model with calibrated lexical consistency safeguards."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.classes_: list[str] = []

    def _make_pipeline(self) -> Pipeline:
        """Build a stable model suitable for small and medium labeled datasets."""
        return Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(
                        transformer_list=[
                            (
                                "word",
                                TfidfVectorizer(
                                    ngram_range=(1, 2),
                                    sublinear_tf=True,
                                    strip_accents="unicode",
                                ),
                            ),
                            (
                                "character",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    sublinear_tf=True,
                                    max_features=20_000,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(96, 48),
                        alpha=0.001,
                        learning_rate_init=0.003,
                        max_iter=1_000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    @staticmethod
    def _combine(pairs: Sequence[tuple[str, str]]) -> list[str]:
        return [f"[CLAIM] {claim} [EVIDENCE] {evidence}" for claim, evidence in pairs]

    def fit(self, pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> NLIModel:
        """Fit the classifier on labeled claim-evidence pairs."""
        if len(pairs) != len(labels):
            raise ValueError("Pairs and labels length mismatch")
        if len(pairs) < 3 or len(set(labels)) < 2:
            raise ValueError("Training requires at least three pairs and two label classes")
        unknown_labels = set(labels) - set(_LABELS)
        if unknown_labels:
            raise ValueError(f"Unsupported NLI labels: {sorted(unknown_labels)}")

        self.pipeline = self._make_pipeline()
        self.pipeline.fit(self._combine(pairs), labels)
        classifier = self.pipeline.named_steps["classifier"]
        self.classes_ = [str(label) for label in classifier.classes_]
        return self

    @staticmethod
    def _rule_probabilities(claim: str, evidence: str) -> dict[str, float]:
        """Produce conservative factual-consistency probabilities for one pair."""
        normalized_claim = claim.strip().lower()
        normalized_evidence = evidence.strip().lower()
        if not normalized_claim or not normalized_evidence:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        if normalized_claim == normalized_evidence:
            return {"entailment": 0.995, "contradiction": 0.002, "neutral": 0.003}

        overlap = token_overlap(claim, evidence)
        cosine = cosine_similarity(claim, evidence)
        entities = entity_overlap(claim, evidence)
        containment = float(normalized_claim in normalized_evidence)
        relatedness = max(overlap, cosine, entities)
        contradiction = max(numeric_mismatch(claim, evidence), negation_mismatch(claim, evidence))
        contradiction *= 0.35 + 0.65 * relatedness

        claim_entities = set(re.findall(r"\b[A-Z][a-z]+\b", claim))
        evidence_entities = set(re.findall(r"\b[A-Z][a-z]+\b", evidence))
        if overlap >= 0.45 and claim_entities & evidence_entities and claim_entities != evidence_entities:
            contradiction = max(contradiction, 0.72)

        entailment = max(containment, 0.55 * cosine + 0.25 * overlap + 0.20 * entities)
        if contradiction >= 0.65:
            entailment *= 0.15
        neutral = max(0.02, 1.0 - max(entailment, contradiction))
        total = entailment + contradiction + neutral
        return {
            "entailment": entailment / total,
            "contradiction": contradiction / total,
            "neutral": neutral / total,
        }

    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return class probabilities in ``classes_`` order for each input pair."""
        if self.pipeline is None:
            raise ValueError("Model is not trained yet")
        if not pairs:
            return np.empty((0, len(self.classes_)), dtype=float)

        learned = self.pipeline.predict_proba(self._combine(pairs))
        rows: list[list[float]] = []
        for pair, learned_row in zip(pairs, learned, strict=True):
            learned_scores = dict(zip(self.classes_, learned_row, strict=True))
            rule_scores = self._rule_probabilities(*pair)
            rule_weight = 0.80 if max(rule_scores.values()) >= 0.70 else 0.35
            scores = {
                label: (1.0 - rule_weight) * float(learned_scores.get(label, 0.0))
                + rule_weight * rule_scores[label]
                for label in _LABELS
            }
            normalizer = sum(scores.values())
            rows.append([scores[label] / normalizer for label in self.classes_])
        return np.asarray(rows, dtype=float)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[str]:
        """Return the highest-probability label for each pair."""
        probabilities = self.predict_proba(pairs)
        return [self.classes_[int(index)] for index in np.argmax(probabilities, axis=1)]

    def save(self, path: str | Path) -> None:
        """Persist the trained model and its label metadata."""
        if self.pipeline is None:
            raise ValueError("No model to save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump({"pipeline": self.pipeline, "classes": self.classes_}, handle)

    def load(self, path: str | Path) -> NLIModel:
        """Load a trained model, including legacy pipeline-only artifacts."""
        with Path(path).open("rb") as handle:
            stored = pickle.load(handle)
        self.pipeline = stored["pipeline"] if isinstance(stored, dict) else stored
        classifier = self.pipeline.named_steps["classifier"]
        self.classes_ = [str(label) for label in classifier.classes_]
        return self
