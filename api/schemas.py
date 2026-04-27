"""Pydantic modeller — FastAPI request/response şemaları."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CatalogRecord(BaseModel):
    formula: str
    ic: float
    rank_ic: float
    adj_ic: float
    source: str = "evolution"
    discovered_at: Optional[str] = None
    updated_at: Optional[str] = None
    split_date: Optional[str] = None
    complexity: Optional[int] = None
    wf: Optional[dict[str, Any]] = None
    overfit: Optional[dict[str, Any]] = None
    backtests: Optional[dict[str, Any]] = None
    ast: Optional[dict[str, Any]] = None
    ast_schema: Optional[int] = None

    class Config:
        extra = "allow"  # eski şemalardaki ekstra alanlar drop olmasın


class ParseRequest(BaseModel):
    formula: str


class ParseResponse(BaseModel):
    ok: bool
    ast: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    complexity: Optional[int] = None


class EvaluateRequest(BaseModel):
    formula: str
    window: str = Field(default="test", pattern="^(test|train|all)$")


class EquityPoint(BaseModel):
    date: str
    equity: float
    benchmark: Optional[float] = None


class EvaluateResponse(BaseModel):
    formula: str
    window: str
    ic: Optional[float] = None
    rank_ic: Optional[float] = None
    sharpe: Optional[float] = None
    annual: Optional[float] = None
    mdd: Optional[float] = None
    net_return: Optional[float] = None
    alpha_ir: Optional[float] = None
    beta: Optional[float] = None
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    error: Optional[str] = None


class JobStatus(BaseModel):
    id: str
    status: str        # "pending" | "running" | "done" | "error"
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[Any] = None


class BacktestRequest(BaseModel):
    formula: str
    window: str = "test"
    top_k: int = 50
    n_drop: int = 5
    buy_fee: float = 0.0005
    sell_fee: float = 0.0015
