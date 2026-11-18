from __future__ import annotations

from pathlib import Path

from phish_reimpl.dataset import CaseSample
from phish_reimpl.results import ResultRow
from phish_reimpl.url_analysis import URLAnalyzer
from phish_reimpl.visual_yolo import YoloVisualAnalyzer   # YOLOv8 — final chosen detector


def url_model_path(workspace_root: Path) -> Path:
    """The URL model to load: retrained path-aware model if present."""
    new = workspace_root / "artifacts" / "url_rf_pathaware.pkl"
    return new if new.exists() else workspace_root / "artifacts" / "url_rf_combined.pkl"


class PhishingPipeline:
    """
    Final production pipeline: YOLOv8 visual branch + Heuristic+ML URL branch
    using the new combined 125k URL model (url_rf_combined.pkl).

    Loads heavy models once at startup; reused for every request.
    """

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root or Path(__file__).resolve().parents[1]

        # ── Visual branch: YOLOv8 detector + ResNetV2 Siamese ────────────
        self.visual = YoloVisualAnalyzer(
            yolo_weights_path=self.workspace_root / "yolo" / "best.pt",
        )

        # ── URL branch: Heuristic+ML mode ────────────────────────────────
        # Prefer the retrained path-aware model (trained with real inner-page
        # URLs, see train_url_model_v2.py); fall back to the original model,
        # which URLAnalyzer then only shows the hostname.
        self.url_branch = URLAnalyzer(
            domain_map_path=self.workspace_root / "assets" / "models" / "domain_map.pkl",
            model_path=url_model_path(self.workspace_root),
            ml_only_mode=False,           # Heuristic+ML mode
            standalone_ml_threshold=0.90,
            ml_conf_threshold=0.90,
            borderline_threshold=0.15,
        )

    def analyze_case(self, case: CaseSample) -> ResultRow:
        visual_out = self.visual.analyze(case)
        html = None
        if case.html_path is not None and Path(case.html_path).exists():
            html = Path(case.html_path).read_text(errors="ignore")
        url_out    = self.url_branch.analyze(case.url, html)
        reason = url_out.reason
        if visual_out.prediction == 1:
            reason = (f"{visual_out.predicted_brand} logo on a site that is not "
                      f"{visual_out.predicted_brand}'s")

        # Primary decision comes from visual branch.
        # URL branch overrides only when visual says benign but URL flags phishing.
        final_pred   = visual_out.prediction
        final_brand  = visual_out.predicted_brand   # keep visual brand even if benign
        final_domain = visual_out.matched_domain
        final_conf   = visual_out.confidence

        if final_pred == 0 and url_out.prediction == 1:
            final_pred   = 1
            # Only override brand if visual branch found no brand
            if not final_brand:
                final_brand  = url_out.predicted_brand
                final_domain = url_out.matched_domain
            final_conf = url_out.risk_score

        return ResultRow(
            case_id=case.case_id,
            url=case.url,
            final_pred=final_pred,
            predicted_brand=final_brand,   # always set when visual detects a brand
            matched_domain=final_domain,
            confidence=final_conf,
            visual_time=visual_out.total_time,
            visual_pred=visual_out.prediction,
            url_pred=url_out.prediction,
            url_risk=url_out.risk_score,
            url_time=url_out.analysis_time,
            reason=reason,
        )
