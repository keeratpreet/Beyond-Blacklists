# Beyond Blacklists — YOLOv8 + Heuristic+ML Phishing Detector

A complete reimplementation of the Phishpedia phishing-detection framework, extended with a YOLOv8 visual branch and a hybrid Heuristic+ML URL analysis pipeline.  
This package contains everything needed: model weights, inference, training, evaluation, and the Chrome browser extension.

---

## Folder Structure

```
Beyond Blacklists/
├── phish_reimpl/               Core Python package (import this)
│   ├── visual.py               Base visual branch: Siamese encoder + brand matching
│   ├── visual_yolo.py          YOLOv8 detector subclass (overrides _detect_logo_boxes)
│   ├── url_analysis.py         URL heuristic + ML branch (36 features)
│   ├── inference.py            PhishingPipeline: fuses visual + URL decisions
│   ├── dataset.py              CaseSample dataclass + dataset loader
│   ├── results.py              ResultRow dataclass + TSV writer
│   ├── scoring.py              Accuracy / Precision / Recall / F1 helpers
│   └── backbone.py             ResNetV2 BiT-M Siamese encoder (feature extractor)
│
├── yolo/
│   └── best.pt                 Trained YOLOv8 logo-detector weights (Git LFS)
│
├── assets/models/
│   ├── resnetv2_rgb_new.pth.tar  Siamese embedding model (ResNetV2 BiT-M, 2048-dim) [LFS]
│   ├── domain_map.pkl          Brand → official domains lookup  {brand: [list of domains]}
│   ├── LOGO_FEATS.npy          Pre-computed reference logo embeddings (bootstrap cache) [LFS]
│   ├── LOGO_FILES.npy          Matching filenames for LOGO_FEATS [LFS]
│   └── expand_targetlist/      Reference brand logo images (270 brands, one folder each)
│
├── artifacts/
│   └── url_rf_combined.pkl     Trained Random Forest URL classifier (125k corpus) [LFS]
│
├── chrome_extension/           Browser extension (Manifest V3)
│   ├── manifest.json
│   ├── popup.html / popup.js   Extension UI + analysis trigger
│   └── background.js           Service worker
│
├── api_server.py               FastAPI server — bridges extension ↔ pipeline
├── run_final.py                CLI runner: batch eval or single screenshot/URL
├── RUN_FINAL.sh                Shell wrapper for run_final.py
│
├── train_url_model.py          Train / retrain the URL Random Forest model
├── rebuild_reference_cache.py  Rebuild Siamese reference embedding cache
├── eval_comparison.py          Full multi-config comparison runner
├── main_yolo_url.py            Batch eval: YOLOv8 + URL (saves results TSV)
└── score_results.py            Score a results TSV against ground-truth labels
```

---

## How the Pipeline Works

### End-to-end inference (what runs when you click "Analyze" in the extension)

```
Browser tab
    │  screenshot (PNG) + URL
    ▼
api_server.py  (FastAPI, port 8765)
    │
    ├─► Visual Branch  [phish_reimpl/visual_yolo.py + visual.py]
    │       │
    │       ├─ YOLOv8 (best.pt)
    │       │   └─ detects logo bounding boxes in screenshot
    │       │
    │       ├─ Siamese Encoder (resnetv2_rgb_new.pth.tar)
    │       │   └─ embeds each logo crop → 2048-dim L2-normalised vector
    │       │   └─ cosine similarity vs pre-built reference embeddings (expand_targetlist/)
    │       │      match accepted if similarity ≥ 0.87
    │       │
    │       └─ Domain consistency check (tldextract)
    │           ├─ brand matched + domain MATCHES page URL  → visual_pred = 0 (benign)
    │           └─ brand matched + domain MISMATCH page URL → visual_pred = 1 (PHISHING)
    │
    ├─► URL Branch  [phish_reimpl/url_analysis.py]
    │       │
    │       ├─ infer_brand(): token match against 270-brand domain map
    │       ├─ Heuristic scoring: free-host (+0.55), brand-in-hostname (+0.45),
    │       │   hyphen (+0.15), suspicious keywords (+0.15), subdomain depth (+0.10)…
    │       └─ Random Forest (url_rf_combined.pkl, 36 features) for borderline cases
    │           └─ url_pred = 0 or 1
    │
    └─► Fusion  [phish_reimpl/inference.py]
            │
            ├─ visual_pred == 1                      →  final = 1  (PHISHING — visual wins)
            ├─ visual_pred == 0, url_pred == 1       →  final = 1  (URL catches visual miss)
            └─ visual_pred == 0, url_pred == 0       →  final = 0  (BENIGN)
```

### Decision rule in plain English
> Visual branch has priority. If YOLOv8 detects a known brand logo and the page domain does not belong to that brand → **phishing, stop**. Only consult the URL branch when the visual branch returns benign.

---

## Quickstart

### Step 1 — Install dependencies

```bash
pip install -r requirements-api.txt
pip install torch torchvision ultralytics
pip install scikit-learn tldextract pandas numpy Pillow opencv-python-headless
```

### Step 2 — Start the API server

```bash
python3 api_server.py
# Server starts on http://127.0.0.1:8765
```

Or use the shell wrapper:

```bash
bash RUN_FINAL.sh --help
```

The server automatically locates `yolo/best.pt` by walking up the directory tree. Startup log confirms which mode loaded:

```
[startup] Workspace root: /path/to/Beyond Blacklists
[startup] Full pipeline loaded: YOLOv8 + Heuristic+ML (url_rf_combined 125k)
```

If it says `PIPELINE FAILED TO LOAD` — set `PHISH_WORKSPACE=/absolute/path/to/folder` to point directly at the repo root.

### Step 3 — Load the Chrome extension

1. Open `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked** → select the `chrome_extension/` folder
4. The extension icon appears in the toolbar

### Step 4 — Analyze a page

Navigate to any website, click the extension icon → **Analyze visible tab**. Results appear within ~1–2 seconds:

- ✅ **Safe** — no phishing indicators found  
- ✅ **Safe — Brand (domain verified)** — known brand logo detected, domain matches  
- ⚠️ **Suspicious — impersonating Brand** — logo detected but domain does not match

---

## Training & Evaluation

### Retrain the URL Random Forest model

```bash
python3 train_url_model.py \
    --dataset /path/to/url_dataset.csv \
    --out artifacts/url_rf_combined.pkl
```

Input CSV must have columns `url` and `label` (0 = benign, 1 = phishing).

### Rebuild Siamese reference embedding cache

Only needed if you add new brand logos to `assets/models/expand_targetlist/`:

```bash
python3 rebuild_reference_cache.py --cache-dir cache/
```

### Batch evaluation

```bash
# YOLOv8 + URL pipeline on a dataset folder
bash RUN_FINAL.sh --eval-dir /path/to/eval_dataset

# Full multi-config comparison (all detector × URL combos)
python3 eval_comparison.py --eval-dir /path/to/eval_dataset

# Score an existing results TSV
python3 score_results.py \
    --results results/yolo_url_results.tsv \
    --dataset /path/to/eval_dataset
```

---

## Key Files Explained

### `phish_reimpl/visual_yolo.py`
Thin subclass of `VisualAnalyzer`. Overrides `_detect_logo_boxes()` to use YOLOv8 (`conf=0.10`) instead of Faster R-CNN. All Siamese matching and domain consistency logic is inherited unchanged.

### `phish_reimpl/visual.py`
Base visual analyser. Key methods:
- `_embed_image(crop)` — pads to square, resizes to 128×128, runs through ResNetV2, returns 2048-dim L2-normalised vector.
- `_match_crop(screenshot, bbox)` — cosine similarity (`reference_matrix @ crop_embedding`). Threshold: **0.87**.
- `_is_domain_consistent(url, domains)` — tldextract-based check. Handles country-TLD variants (amazon.co.uk etc.).
- `analyze(case)` → `VisualDecision`.

### `phish_reimpl/url_analysis.py`
- `extract_url_features(url)` — computes **36 structural features** (lengths, ratios, entropy, brand tokens, TLD flags, free-host detection).
- `infer_brand(url)` — O(1) token-index lookup against domain_map.pkl.
- `analyze(url)` → `URLDecision` — heuristic score → optional RF supplement (threshold **0.90** for borderline, **0.97** standalone).

### `phish_reimpl/inference.py`
`PhishingPipeline`: loads both branches at startup, runs both in sequence per case, applies fusion rule. Used by `api_server.py`.

### `api_server.py`
FastAPI server with:
- `GET /health` — returns mode (`full` / `url_only` / `down`) and pipeline error if any.
- `POST /analyze` — accepts `url` + `screenshot` (PNG multipart), returns JSON with `final_pred`, `predicted_brand`, `visual_pred`, `url_pred`, `url_risk`, `confidence`, `mode`.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PHISH_WORKSPACE` | auto-detect | Absolute path to repo root (containing `yolo/best.pt`) |
| `PHISH_URL_ONLY` | `0` | Set to `1` for URL-only mode (no visual branch, no GPU needed) |
| `API_HOST` | `127.0.0.1` | API server bind host |
| `API_PORT` | `8765` | API server port |

---

## Results

Evaluated on 2000 balanced phishing/benign cases (50 brands × 2 cases × 20 variants):

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Faster R-CNN only (baseline) | 77.69% | 94.59% | 58.33% | 72.16% |
| Faster R-CNN + URL (midterm) | 96.69% | 95.16% | 98.33% | 96.72% |
| **YOLOv8 + Heuristic+ML (final — this repo)** | **95.10%** | **91.07%** | **100.00%** | **95.33%** |

Key improvement midterm → final: replaced Faster R-CNN with YOLOv8 (faster, no detectron2 dependency, better small-logo recall) and retrained the URL classifier on a 125k combined corpus (vs 40k previously). The final model achieves **100% recall** — zero phishing pages missed.
