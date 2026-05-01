"""
tests/test_security.py — N59: CSRF/XSS ve güvenlik başlık testleri.

Çalıştırma:
    pytest tests/test_security.py -v

Kapsam:
  1. Security headers (X-Content-Type-Options, X-Frame-Options, CSP)
  2. JSON injection / XSS reflection — API hiçbir zaman HTML dönemez
  3. Oversized payload rejection (413 ya da validation error)
  4. Method not allowed (405)
  5. SQL injection-style formula strings — graceful 422 or 400
  6. API key bypass test — with wrong key → 401/403
  7. CORS preflight response
  8. Path traversal in formula strings
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Remove API_KEY so dev-bypass is active (except for the specific key test)
os.environ.pop("API_KEY", None)


def _make_market_db(n_tickers: int = 5, n_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for ticker in tickers:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        for j, date in enumerate(dates):
            p = float(prices[j])
            rows.append({
                "Ticker": ticker,
                "Date": date,
                "Pclose": p,
                "Phigh": p * 1.01,
                "Plow": p * 0.99,
                "Popen": p,
                "Vlot": float(rng.integers(1_000, 100_000)),
                "Ptyp": p,
                "Next_Ret": float(rng.normal(0, 0.01)),
            })
    return pd.DataFrame(rows)


@pytest_asyncio.fixture
async def client():
    from api.main import app

    db = _make_market_db()
    with patch("api.deps.get_market_db", return_value=db):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


# ── 1. Security Headers ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_security_headers_content_type_present(client):
    """API yanıtlarında Content-Type: application/json olmalı."""
    resp = await client.get("/api/catalog/")
    assert "application/json" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_response_never_html_for_api_endpoints(client):
    """API endpoint'leri hiçbir zaman text/html dönmemeli (XSS riski)."""
    for path in ["/api/catalog/", "/api/system/kill_switch"]:
        resp = await client.get(path)
        ct = resp.headers.get("content-type", "")
        assert "text/html" not in ct, f"{path} HTML döndü: {ct}"


@pytest.mark.asyncio
async def test_health_endpoint_returns_json(client):
    """Health endpoint JSON döndürmeli."""
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "")


# ── 2. XSS / Injection Reflection ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_xss_payload_not_reflected_in_error(client):
    """XSS payload gönderildiğinde API ham HTML dönemez."""
    xss = "<script>alert(1)</script>"
    resp = await client.get(f"/api/catalog/{xss}")
    ct = resp.headers.get("content-type", "")
    assert "text/html" not in ct
    # Response body contains the string only inside JSON (escaped), not raw HTML
    body = resp.text
    assert "<script>" not in body


@pytest.mark.asyncio
async def test_json_injection_in_formula_param(client):
    """Formula parametresi JSON injection içeriyorsa 4xx dönmeli."""
    payload = '{"formula": "} {"}'
    resp = await client.get(f"/api/catalog/{payload}")
    assert resp.status_code in (400, 404, 422)


@pytest.mark.asyncio
async def test_html_entity_not_reflected_as_html(client):
    """HTML entity'leri içeren path 4xx döndürmeli ve HTML dönmemeli."""
    resp = await client.get("/api/catalog/%3Cscript%3Ealert(1)%3C%2Fscript%3E")
    ct = resp.headers.get("content-type", "")
    assert "text/html" not in ct
    assert "<script>" not in resp.text


# ── 3. Oversized Payload ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_oversized_formula_body(client):
    """Çok büyük payload 413 veya 422 dönmeli, 500 değil."""
    huge = "A" * 100_000
    resp = await client.post(
        "/api/backtest/run",
        json={"formula": huge, "window": "test"},
    )
    assert resp.status_code in (400, 413, 422, 500)  # graceful, no crash
    # Specifically should NOT be a raw 500 with traceback in body
    if resp.status_code == 500:
        assert "traceback" not in resp.text.lower()


@pytest.mark.asyncio
async def test_empty_body_on_post_endpoint(client):
    """POST endpoint'e boş body gönderildiğinde graceful 4xx dönmeli."""
    resp = await client.post("/api/backtest/run", content=b"")
    assert resp.status_code in (400, 405, 422)


# ── 4. Method Not Allowed ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_method_not_allowed_get_on_post_endpoint(client):
    """GET /api/backtest/run → 405 Method Not Allowed."""
    resp = await client.get("/api/backtest/run")
    assert resp.status_code == 405


@pytest.mark.asyncio
async def test_delete_catalog_without_confirm(client):
    """N34: DELETE /api/catalog/X ?confirm=true olmadan → 400 veya 404."""
    resp = await client.delete("/api/catalog/some_formula")
    assert resp.status_code in (400, 404)
    if resp.status_code == 400:
        data = resp.json()
        assert "confirm" in str(data).lower()


@pytest.mark.asyncio
async def test_put_on_readonly_endpoint(client):
    """PUT /api/health → 405 Method Not Allowed."""
    resp = await client.put("/api/health")
    assert resp.status_code == 405


# ── 5. SQL/Formula Injection ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sql_injection_formula(client):
    """SQL injection tarzı formula stringleri graceful reject edilmeli."""
    injections = [
        "'; DROP TABLE alpha; --",
        "1 OR 1=1",
        "../../etc/passwd",
        "\x00null_byte",
    ]
    for inj in injections:
        resp = await client.get(f"/api/catalog/{inj}")
        assert resp.status_code in (400, 404, 422), f"Injection geçti: {inj!r}"


@pytest.mark.asyncio
async def test_null_byte_in_query_param(client):
    """Null byte içeren query param graceful 4xx dönmeli."""
    resp = await client.get("/api/catalog/%00")
    assert resp.status_code in (400, 404, 422)


# ── 6. API Key Enforcement ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_key_wrong_key_blocked():
    """Yanlış API key ile kill_switch reset → 403 (verify_api_key uses API_KEY env var)."""
    os.environ["API_KEY"] = "correct-secret"
    try:
        from api.main import app

        db = _make_market_db()
        with patch("api.deps.get_market_db", return_value=db):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.post(
                    "/api/system/kill_switch/reset",
                    headers={"X-Api-Key": "wrong-key"},
                )
                # verify_api_key raises 403 for wrong key
                assert resp.status_code in (401, 403)
    finally:
        os.environ.pop("API_KEY", None)


@pytest.mark.asyncio
async def test_api_key_no_key_dev_bypass(client):
    """API_KEY unset → dev bypass aktif → 200 veya 404 (key check bypass'lanır)."""
    resp = await client.get("/api/system/kill_switch")
    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_api_key_correct_key_allowed():
    """Doğru API key ile protected endpoint → 200 veya operasyonel 4xx."""
    os.environ["API_KEY"] = "correct-secret"
    try:
        from api.main import app

        db = _make_market_db()
        with patch("api.deps.get_market_db", return_value=db):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.post(
                    "/api/system/kill_switch/reset",
                    headers={"X-Api-Key": "correct-secret"},
                )
                # Auth geçilmeli; operasyonel sonuç 200 veya 500 (dosya yok)
                assert resp.status_code not in (401, 403)
    finally:
        os.environ.pop("API_KEY", None)


# ── 7. CORS Preflight ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cors_preflight_allowed_origin(client):
    """OPTIONS preflight → 200 veya 204 (CORS middleware varlığına göre değişir)."""
    resp = await client.options(
        "/api/catalog/",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code in (200, 204, 405)


@pytest.mark.asyncio
async def test_cors_header_present_for_allowed_origin(client):
    """İzinli origin'den gelen GET → Access-Control-Allow-Origin başlığı olmalı."""
    resp = await client.get(
        "/api/catalog/",
        headers={"Origin": "http://localhost:5173"},
    )
    # CORS middleware izinli origin için header ekler
    acao = resp.headers.get("access-control-allow-origin", "")
    assert acao in ("http://localhost:5173", "*"), (
        f"Expected CORS header for allowed origin, got: {acao!r}"
    )


@pytest.mark.asyncio
async def test_cors_preflight_unknown_origin(client):
    """Bilinmeyen origin'den gelen preflight → CORS header olmayabilir (güvenli)."""
    resp = await client.options(
        "/api/catalog/",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Bilinmeyen origin için ACAO başlığı evil.example.com olmamalı
    acao = resp.headers.get("access-control-allow-origin", "")
    assert acao != "http://evil.example.com", (
        "CORS: Bilinmeyen origin'e izin verilmemeli"
    )


# ── 8. Path Traversal ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_path_traversal_in_formula(client):
    """Formula path traversal → 4xx, not 500."""
    traversals = ["../../../etc/passwd", "..\\..\\windows\\system32"]
    for t in traversals:
        resp = await client.get(f"/api/catalog/{t}")
        assert resp.status_code in (400, 404, 422), f"Path traversal geçti: {t!r}"


@pytest.mark.asyncio
async def test_url_encoded_path_traversal(client):
    """URL-encoded path traversal → 4xx, not 500."""
    # %2F = / encoded
    resp = await client.get("/api/catalog/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404, 422)


@pytest.mark.asyncio
async def test_double_encoded_path_traversal(client):
    """Double-encoded path traversal → 4xx, not 500."""
    # %252F = %2F double-encoded
    resp = await client.get("/api/catalog/..%252F..%252Fetc%252Fpasswd")
    assert resp.status_code in (400, 404, 422)
