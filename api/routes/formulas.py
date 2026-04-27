"""POST /api/formulas/parse + /evaluate."""
from __future__ import annotations

from fastapi import APIRouter

from api.deps import (
    get_benchmark,
    get_cfg,
    get_market_db,
    get_split_date,
)
from api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    ParseRequest,
    ParseResponse,
)
from engine.core.api_helpers import parse_or_raise, run_full_evaluate
from engine.ml.replay_buffer import _tree_to_dict

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest) -> ParseResponse:
    """Formül string'i AST'a parse et. Hata mesajı human-readable."""
    cfg = get_cfg()
    try:
        tree = parse_or_raise(req.formula, cfg)
    except ValueError as e:
        return ParseResponse(ok=False, error=str(e))
    return ParseResponse(
        ok=True,
        ast=_tree_to_dict(tree),
        complexity=tree.size(),
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """
    Formülü parse et, IC hesapla, backtest et — tek çağrıda tam pipeline.

    Workbench Col C (KPI'lar + equity chart) bu endpoint'in çıktısıyla beslenir.
    """
    cfg = get_cfg()
    db = get_market_db()
    split_ts = get_split_date()
    benchmark = get_benchmark()

    result = run_full_evaluate(
        formula=req.formula,
        db=db,
        split_ts=split_ts,
        cfg=cfg,
        window=req.window,
        benchmark=benchmark,
    )

    if "error" in result and "ic" not in result:
        return EvaluateResponse(
            formula=req.formula,
            window=req.window,
            error=result["error"],
        )

    return EvaluateResponse(
        formula=req.formula,
        window=req.window,
        ic=result.get("ic"),
        rank_ic=result.get("rank_ic"),
        sharpe=result.get("sharpe"),
        annual=result.get("annual"),
        mdd=result.get("mdd"),
        net_return=result.get("net_return"),
        alpha_ir=result.get("alpha_ir"),
        beta=result.get("beta"),
        equity_curve=result.get("equity_curve", []),
        error=result.get("error"),
    )
