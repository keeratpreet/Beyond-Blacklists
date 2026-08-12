from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from phish_reimpl.dataset import CaseSample
from phish_reimpl.url_analysis import URLAnalyzer
from phish_reimpl.inference import PhishingPipeline   # YOLOv8 + URL (Heuristic+ML)

_pipeline: "PhishingPipeline | None" = None
_pipeline_error: str | None = None           # set if pipeline failed to load
_url_only: URLAnalyzer | None = None         # only when PHISH_URL_ONLY=1 is set
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _find_workspace_root() -> Path:
    """
    Locate the repository workspace root by walking up from this file until we
    find the yolo/best.pt weights file.  Stops after 5 levels to avoid wandering
    too far up the filesystem.

    Callers can override with the PHISH_WORKSPACE environment variable.
    """
    override = os.environ.get("PHISH_WORKSPACE", "").strip()
    if override:
        p = Path(override).resolve()
        if not (p / "yolo" / "best.pt").exists():
            raise FileNotFoundError(
                f"PHISH_WORKSPACE={override!r} but yolo/best.pt not found there."
            )
        return p

    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "yolo" / "best.pt").exists():
            return candidate
        candidate = candidate.parent

    raise FileNotFoundError(
        "Cannot find yolo/best.pt in this directory or any of its parents.\n"
        "Set the PHISH_WORKSPACE environment variable to the repository root.\n"
        f"Searched from: {Path(__file__).resolve().parent}"
    )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _pipeline, _pipeline_error, _url_only
    _pipeline = None
    _pipeline_error = None
    _url_only = None

    # ── Explicit URL-only mode (opt-in only, e.g. headless / no GPU env) ─────
    if os.environ.get("PHISH_URL_ONLY", "").strip().lower() in {"1", "true", "yes"}:
        print("[startup] PHISH_URL_ONLY=1 — starting in URL-only mode (no visual branch)")
        root = _find_workspace_root() if os.environ.get("PHISH_WORKSPACE") else Path(__file__).resolve().parent
        _url_only = URLAnalyzer(
            domain_map_path=root / "assets" / "models" / "domain_map.pkl",
            model_path=root / "artifacts" / "url_rf_combined.pkl",
            ml_only_mode=False,
            standalone_ml_threshold=0.97,
            borderline_threshold=0.15,
        )
        yield
        _pipeline = None
        _url_only = None
        return

    # ── Normal startup: full YOLOv8 + URL pipeline (required) ────────────────
    try:
        workspace = _find_workspace_root()
        print(f"[startup] Workspace root: {workspace}")
        _pipeline = PhishingPipeline(workspace_root=workspace)
        print("[startup] Full pipeline loaded: YOLOv8 + Heuristic+ML (url_rf_combined 125k)")
    except Exception as e:
        _pipeline_error = f"{type(e).__name__}: {e}"
        print(f"[startup] *** PIPELINE FAILED TO LOAD ***")
        print(f"[startup] {_pipeline_error}")
        print("[startup] Server will start but /analyze will return 503 until the error is fixed.")
        print("[startup] To run URL-only mode explicitly, set PHISH_URL_ONLY=1.")

    yield
    _pipeline = None
    _url_only = None
    _pipeline_error = None


app = FastAPI(title="Phishpedia — YOLOv8 + Heuristic+ML API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    case_id: str
    url: str
    final_pred: int = Field(description="1 = phishing, 0 = benign")
    predicted_brand: str | None = Field(
        description="Brand detected by visual branch (set even for benign pages "
                    "when the domain is the legitimate brand domain)"
    )
    matched_domain: list[str] | None
    confidence: float | None
    visual_time: float
    visual_pred: int = Field(description="Raw visual branch output (0=benign, 1=phish)")
    url_pred: int   = Field(description="Raw URL branch output (0=benign, 1=phish)")
    url_risk: float
    url_time: float
    mode: str       = Field(description="'full' = YOLOv8+URL, 'url_only' = URL branch only")


def _result_to_response(row, mode: str) -> AnalyzeResponse:
    md = row.matched_domain
    if md is not None and not isinstance(md, list):
        md = list(md) if md else None
    return AnalyzeResponse(
        case_id=row.case_id,
        url=row.url,
        final_pred=row.final_pred,
        predicted_brand=row.predicted_brand,
        matched_domain=md,
        confidence=row.confidence,
        visual_time=row.visual_time,
        visual_pred=row.visual_pred,
        url_pred=row.url_pred,
        url_risk=row.url_risk,
        url_time=row.url_time,
        mode=mode,
    )


@app.get("/health")
def health() -> dict:
    if _pipeline is not None:
        mode = "full"
    elif _url_only is not None:
        mode = "url_only"
    else:
        mode = "down"

    out: dict = {
        "status": "ok" if mode != "down" else "error",
        "mode": mode,
        "model": "YOLOv8 + Heuristic+ML (url_rf_combined 125k)",
    }
    if _pipeline_error:
        out["pipeline_error"] = _pipeline_error
    return out


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    url: str = Form(..., description="Full URL of the current browser tab"),
    screenshot: UploadFile = File(..., description="PNG screenshot of the visible tab"),
) -> AnalyzeResponse:
    # Visual pipeline failed to load — surface the actual error instead of silently degrading
    if _pipeline is None and _url_only is None:
        detail = (
            f"Visual pipeline not loaded: {_pipeline_error}. "
            "Check server logs. To run URL-only mode explicitly, set PHISH_URL_ONLY=1."
        ) if _pipeline_error else "Pipeline not initialized."
        raise HTTPException(status_code=503, detail=detail)

    u = (url or "").strip()
    if not u or not (u.startswith("http://") or u.startswith("https://")):
        raise HTTPException(status_code=400, detail="url must be a non-empty http(s) URL")

    raw = await screenshot.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty screenshot")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="screenshot too large (max 25 MB)")

    case_id = f"ext-{uuid.uuid4().hex[:12]}"

    # ── Explicit URL-only mode (PHISH_URL_ONLY=1) ────────────────────────────
    if _url_only is not None:
        url_out = _url_only.analyze(u)
        # URL-only mode never produces a real ResultRow (that's built by
        # PhishingPipeline.analyze_case, which isn't invoked here) — this
        # builds a minimal stand-in object with the same attributes so
        # _result_to_response() can stay a single shared code path for both modes
        row_like = type("R", (), {
            "case_id":         case_id,
            "url":             u,
            "final_pred":      url_out.prediction,
            "predicted_brand": url_out.predicted_brand,
            "matched_domain":  url_out.matched_domain,
            "confidence":      url_out.risk_score,
            "visual_time":     0.0,
            "visual_pred":     0,
            "url_pred":        url_out.prediction,
            "url_risk":        url_out.risk_score,
            "url_time":        url_out.analysis_time,
        })()
        return _result_to_response(row_like, mode="url_only")

    # ── Full YOLOv8 + URL pipeline ───────────────────────────────────────────
    # the visual pipeline (YOLO + PIL) expects a file path, not raw bytes, so
    # the uploaded screenshot is spooled to a temp file for the duration of
    # this request and always cleaned up in the `finally` block below
    tmp_path: Path | None = None
    try:
        import tempfile as _tmp
        fd, tmp_str = _tmp.mkstemp(suffix=".png")
        os.close(fd)
        tmp_path = Path(tmp_str)
        tmp_path.write_bytes(raw)

        case = CaseSample(
            case_id=case_id,
            url=u,
            screenshot_path=tmp_path,
            html_path=None,
        )
        row = _pipeline.analyze_case(case)  # type: ignore[union-attr]
        return _result_to_response(row, mode="full")
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    import uvicorn
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8765"))
    print(f"Starting Phishpedia API on {host}:{port}")
    print("Model: YOLOv8 + Heuristic+ML (url_rf_combined 125k)")
    uvicorn.run("api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
