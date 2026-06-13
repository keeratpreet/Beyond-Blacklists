from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResultRow:
    case_id: str
    url: str
    final_pred: int
    predicted_brand: str | None
    matched_domain: list[str] | None
    confidence: float | None
    visual_time: float
    visual_pred: int
    url_pred: int
    url_risk: float
    url_time: float


HEADER = (
    "case_id\turl\tfinal_pred\tpredicted_brand\tmatched_domain\tconfidence\t"
    "visual_time\tvisual_pred\turl_pred\turl_risk\turl_time\n"
)


def ensure_results_header(output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not output_file.exists() or output_file.stat().st_size == 0:
        output_file.write_text(HEADER)


def write_result_row(output_file: Path, row: ResultRow) -> None:
    with output_file.open("a") as f:
        f.write(
            f"{row.case_id}\t{row.url}\t{row.final_pred}\t{row.predicted_brand}\t"
            f"{row.matched_domain}\t{row.confidence}\t{row.visual_time:.4f}\t"
            f"{row.visual_pred}\t{row.url_pred}\t{row.url_risk:.4f}\t{row.url_time:.4f}\n"
        )
