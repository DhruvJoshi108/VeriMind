"""FastAPI app exposing the VERIMIND verification endpoint."""

from __future__ import annotations

import logging
import fastapi

import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from sqlalchemy.exc import SQLAlchemyError

from api.deps import get_pipeline, require_api_key
from api.schemas import VerifyRequest, VerifyResponse
from verimind.config import settings
from verimind.metrics import metrics
from verimind.pipeline import VerificationPipeline
from verimind.storage import check_database, get_report, save_report


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
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "search_backend": settings.search_backend,
            "judge_backend": settings.judge_backend,
            "pipeline_loaded": getattr(request.app.state, "pipeline", None) is not None,
        }

    @application.get("/readyz")
    def ready() -> dict[str, str]:
        try:
            check_database()
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc
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
    def metrics_endpoint(_: str = Depends(require_api_key)) -> str:
        return metrics.render_prometheus()

    return application

import logging
logger = logging.getLogger("uvicorn.error")

@app.get("/readyz")
async def readyz():
    try:
        # tumhare existing checks yahan
        ...
        return {"status": "ready"}
    except Exception as e:
        logger.exception("readyz failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

app = create_app()
