"""Pydantic schemas for the verification API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    """Request body for verification."""

    text: str = Field(..., min_length=1, max_length=20000)


class VerifyResponse(BaseModel):
    """Response payload for verification."""

    document_summary: dict
    claims: list[dict]
    report_id: int | None = None
    degraded: bool = False
