"""Validated environment configuration for VERIMIND."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings. This stdlib loader avoids a required settings dependency."""

    environment: str = "development"
    api_key: str | None = None
    database_url: str | None = None
    search_api_key: str | None = None
    search_backend: str = "null"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    judge_backend: str = "local"
    nli_backend: str = "local"
    nli_model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli"
    nli_device: str = "auto"
    nli_model_path: str | None = None
    cors_origins: tuple[str, ...] = ()
    log_level: str = "INFO"
    max_claims_per_doc: int = 50
    max_evidence_per_claim: int = 5
    risk_threshold: float = 60.0
    request_timeout_seconds: float = 20.0
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    redis_url: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        environment = os.getenv("ENVIRONMENT", "development").lower()
        api_key = os.getenv("API_KEY")
        origins = tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "").split(",")
            if origin.strip()
        )
        if environment == "production" and not api_key:
            raise ValueError("API_KEY is required when ENVIRONMENT=production")
        if "*" in origins and environment == "production":
            raise ValueError("CORS_ORIGINS must not contain * in production")
        nli_backend = os.getenv("NLI_BACKEND", "local").lower()
        if nli_backend not in {"local", "transformer"}:
            raise ValueError("NLI_BACKEND must be 'local' or 'transformer'")
        nli_device = os.getenv("NLI_DEVICE", "auto").lower()
        if nli_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("NLI_DEVICE must be 'auto', 'cpu', or 'cuda'")
        search_backend = os.getenv("SEARCH_BACKEND", "null").lower()
        search_api_key = os.getenv("SEARCH_API_KEY") or None
        if search_backend not in {"null", "tavily", "wikipedia"}:
            raise ValueError("SEARCH_BACKEND must be 'null', 'tavily', or 'wikipedia'")
        if search_backend == "tavily" and not search_api_key:
            raise ValueError("SEARCH_API_KEY is required when SEARCH_BACKEND=tavily")
        return cls(
            environment=environment,
            api_key=api_key,
            database_url=os.getenv("DATABASE_URL"),
            search_api_key=search_api_key,
            search_backend=search_backend,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            judge_backend=os.getenv("JUDGE_BACKEND", "local").lower(),
            nli_backend=nli_backend,
            nli_model_name=os.getenv(
                "NLI_MODEL_NAME",
                "MoritzLaurer/DeBERTa-v3-base-mnli",
            ),
            nli_device=nli_device,
            nli_model_path=os.getenv("NLI_MODEL_PATH") or None,
            cors_origins=origins,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            max_claims_per_doc=_integer("MAX_CLAIMS_PER_DOC", 50),
            max_evidence_per_claim=_integer("MAX_EVIDENCE_PER_CLAIM", 5),
            risk_threshold=float(os.getenv("RISK_THRESHOLD", "60.0")),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20")),
            rate_limit_requests=_integer("RATE_LIMIT_REQUESTS", 60),
            rate_limit_window_seconds=_integer("RATE_LIMIT_WINDOW_SECONDS", 60),
            redis_url=os.getenv("REDIS_URL"),
        )


settings = Settings.from_env()
