from __future__ import annotations

import argparse
from pathlib import Path

from phish_reimpl.dataset import load_cases
from phish_reimpl.results import ResultRow, ensure_results_header, write_result_row
from phish_reimpl.url_analysis import URLAnalyzer
from phish_reimpl.visual_yolo import YoloVisualAnalyzer


def run_folder(folder: Path, output_file: Path, yolo_weights: Path, detector_threshold: float) -> None:
    workspace_root = Path(__file__).resolve().parent
    visual = YoloVisualAnalyzer(
        yolo_weights_path=yolo_weights,
        detector_threshold=detector_threshold,
    )
    url_branch = URLAnalyzer(
        domain_map_path=workspace_root / "assets" / "models" / "domain_map.pkl",
        model_path=workspace_root / "artifacts" / "url_rf_phiusiil.pkl",
    )

    ensure_results_header(output_file)
    for case in load_cases(folder):
        visual_out = visual.analyze(case)
        url_out = url_branch.analyze(case.url)

        final_pred = visual_out.prediction
        final_brand = visual_out.predicted_brand
        final_domain = visual_out.matched_domain
        final_conf = visual_out.confidence

        if final_pred == 0 and url_out.prediction == 1:
            final_pred = 1
            final_brand = url_out.predicted_brand
            final_domain = url_out.matched_domain
            final_conf = url_out.risk_score

        row = ResultRow(
            case_id=case.case_id,
            url=case.url,
            final_pred=final_pred,
            predicted_brand=final_brand,
            matched_domain=final_domain,
            confidence=final_conf,
            visual_time=visual_out.total_time,
            visual_pred=visual_out.prediction,
            url_pred=url_out.prediction,
            url_risk=url_out.risk_score,
            url_time=url_out.analysis_time,
        )
        write_result_row(output_file, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, default=Path("yolo") / "best.pt")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    args = parser.parse_args()
    run_folder(args.folder, args.output, args.yolo_weights, args.detector_threshold)


if __name__ == "__main__":
    main()
