"""Optional pretrained transformer model for natural-language inference."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_LABELS = {"entailment", "contradiction", "neutral"}
DEFAULT_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli"
logger = logging.getLogger(__name__)


class TransformerNLIModel:
    """Run a pretrained three-class NLI transformer behind the local model interface."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        batch_size: int = 4,
        max_length: int = 384,
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.classes_: list[str] = []
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    @staticmethod
    def _ordered_labels(id2label: Mapping[int | str, str]) -> list[str]:
        labels: list[str] = []
        for index in range(3):
            label = id2label.get(index, id2label.get(str(index)))
            if not isinstance(label, str):
                raise ValueError("NLI model must define three named output labels")
            labels.append(label.strip().casefold())
        if len(set(labels)) != 3 or set(labels) != _LABELS:
            raise ValueError(
                "NLI transformer must expose entailment, contradiction, and neutral labels"
            )
        return labels

    def load(self) -> TransformerNLIModel:
        """Load tokenizer and weights, downloading them to the Hugging Face cache if needed."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "The transformer NLI backend requires PyTorch, Transformers, and SentencePiece."
            ) from error

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NLI_DEVICE=cuda was requested, but CUDA is unavailable.")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except ImportError as error:
            raise RuntimeError(
                "The DeBERTa tokenizer needs SentencePiece. Install project requirements."
            ) from error
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )
        self._model.to(self.device)
        self._model.eval()
        self.classes_ = self._ordered_labels(self._model.config.id2label)
        logger.info("Loaded NLI transformer %s on %s", self.model_name, self.device)
        return self

    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return NLI probabilities in the class order advertised by the model config."""
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise ValueError("Transformer model is not loaded")
        if not pairs:
            return np.empty((0, len(self.classes_)), dtype=float)

        batches: list[np.ndarray] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [claim for claim, _ in batch],
                [evidence for _, evidence in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1)
            batches.append(probabilities.cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[str]:
        """Return the highest-probability NLI label for each pair."""
        probabilities = self.predict_proba(pairs)
        return [self.classes_[int(index)] for index in np.argmax(probabilities, axis=1)]
