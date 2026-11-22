"""FastAPI app exposing the VERIMIND verification endpoint."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response

from api.deps import get_pipeline, require_api_key
from api.schemas import VerifyRequest, VerifyResponse
from verimind.config import settings
from verimind.metrics import metrics
from verimind.pipeline import VerificationPipeline
from verimind.storage import get_report, save_report


def create_app() -> FastAPI:
    """Build an application without constructing models during import or test collection."""
    application = FastAPI(title="VERIMIND API", version="0.2.0")
    application.state.pipeline = None
    ui_path = Path(__file__).with_name("static") / "index.html"
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "x-api-key", "x-request-id"],
        )

    @application.middleware("http")
    async def observe_request(request: Request, call_next: Callable[..., Any]) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        metrics.observe_request(
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @application.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(ui_path, media_type="text/html")

    @application.get("/health")
    @application.get("/healthz")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "search_backend": settings.search_backend,
            "judge_backend": settings.judge_backend,
        }

    @application.get("/readyz")
    def ready() -> dict[str, str]:
        return {"status": "ready"}

    @application.post("/verify", response_model=VerifyResponse)
    def verify_document(
        payload: VerifyRequest,
        _: str = Depends(require_api_key),
        pipeline: VerificationPipeline = Depends(get_pipeline),  # noqa: B008
    ) -> dict[str, Any]:
        report = pipeline.verify_text(payload.text)
        report_id = save_report(payload.text, report)
        report["report_id"] = report_id
        report.setdefault("degraded", False)
        metrics.observe_report(report)
        return report

    @application.get("/report/{report_id}")
    def fetch_report(report_id: int, _: str = Depends(require_api_key)) -> dict[str, Any]:
        report = get_report(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        return report

    @application.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics.render_prometheus()

    return application


app = create_app()
