#!/usr/bin/env bash
# ============================================================
#  RUN_FINAL.sh  —  YOLOv8 + Heuristic+ML (new 125k model)
#
#  PREREQUISITES (install once):
#    pip install torch torchvision ultralytics
#    pip install scikit-learn tldextract pandas numpy Pillow opencv-python-headless
#    # detectron2 is NOT required for this configuration
#
#  USAGE:
#    # Batch evaluation on a dataset folder:
#    bash RUN_FINAL.sh --eval-dir /path/to/eval_2000
#
#    # Single image + URL:
#    bash RUN_FINAL.sh --screenshot shot.png --url "http://example.com"
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
python3 run_final.py "$@"
