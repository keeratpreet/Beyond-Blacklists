from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaseSample:
    # One website sample from a dataset folder.
    case_id: str
    url: str
    screenshot_path: Path
    html_path: Path | None


def _parse_info(text: str) -> str:
    """
    Extract a URL from an info.txt file.

    Supports two formats:
      1. Plain URL: the entire file content is the URL (common case).
      2. Dict literal: info.txt contains a Python/JSON dict with a 'url' key
         (used by some dataset builders, e.g. eval_auto phish cases).
    """
    text = text.strip()
    # Fast path: looks like a URL already.
    if text.startswith(("http://", "https://", "ftp://")):
        return text
    # Try to parse as a Python dict literal.
    if text.startswith("{"):
        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict) and "url" in data:
                return str(data["url"]).strip()
        except (ValueError, SyntaxError):
            pass
        # Fallback: regex extraction of url key.
        m = re.search(r"['\"]url['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
        if m:
            return m.group(1).strip()
    return text


def load_cases(folder: Path) -> list[CaseSample]:
    cases: list[CaseSample] = []
    # Each subfolder is treated as one website case.
    for case_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        screenshot_path = case_dir / "shot.png"
        info_path = case_dir / "info.txt"
        # Skip folders that do not have the minimum files needed to run inference.
        if not screenshot_path.exists() or not info_path.exists():
            continue
        html_path = case_dir / "html.txt"
        if not html_path.exists():
            html_path = case_dir / "index.html"
        if not html_path.exists():
            html_path = None
        url = _parse_info(info_path.read_text())
        # Convert the folder on disk into a structured case object for the pipeline.
        cases.append(
            CaseSample(
                case_id=case_dir.name,
                url=url,
                screenshot_path=screenshot_path,
                html_path=html_path,
            )
        )
    return cases
