from __future__ import annotations

from pathlib import Path

import numpy as np

from .visual import VisualAnalyzer


class YoloVisualAnalyzer(VisualAnalyzer):
    """
    YOLO-based visual branch. Subclasses VisualAnalyzer and overrides only
    _detect_logo_boxes() (logo bounding-box detection) and _load_detector()
    (model loading) — every other step (Siamese embedding, cosine similarity
    matching against reference logos, domain-consistency check) is inherited
    unchanged from the base class, since only the detector backend differs.
    """

    def __init__(
        self,
        *,
        yolo_weights_path: Path | None = None,
        detector_threshold: float = 0.10,
        similarity_threshold: float = 0.87,
        topk_boxes: int = 3,
        target_classes: tuple[int, ...] | None = None,
    ) -> None:
        self.yolo_weights_path = yolo_weights_path or Path(__file__).resolve().parents[1] / "yolo" / "best.pt"
        self.target_classes = target_classes
        super().__init__(
            detector_threshold=detector_threshold,
            similarity_threshold=similarity_threshold,
            topk_boxes=topk_boxes,
        )

    def _load_detector(self):
        # lazy import: ultralytics is only needed when the YOLO branch is
        # actually instantiated, not for URL-only mode
        from ultralytics import YOLO

        return YOLO(str(self.yolo_weights_path))

    def _detect_logo_boxes(self, screenshot_path: Path) -> np.ndarray:
        # low confidence threshold (0.10) is intentional — false positives here
        # are cheap (each box just gets a Siamese similarity check afterward,
        # which is the real filter at threshold 0.87), but a missed logo box
        # can never be recovered downstream
        results = self.predictor.predict(
            source=str(screenshot_path),
            conf=self.detector_threshold,
            verbose=False,
        )
        if not results:
            return np.empty((0, 4), dtype=np.float32)

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)

        xyxy = boxes.xyxy.detach().cpu().numpy()
        # optionally restrict to specific YOLO class IDs (e.g. "logo" only,
        # excluding other trained classes) — no-op if target_classes is unset
        if self.target_classes is not None and boxes.cls is not None:
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            keep = np.array([cls in self.target_classes for cls in classes], dtype=bool)
            xyxy = xyxy[keep]

        return np.asarray(xyxy, dtype=np.float32)
