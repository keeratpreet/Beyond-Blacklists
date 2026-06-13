# PhishGuard — YOLOv8 + Heuristic+ML Pipeline

Complete reimplementation of the Phishpedia phishing-detection framework.  
This package contains everything needed: model training, inference, evaluation, and the browser extension.

---

## Folder Structure

```
yolo_url_final/
├── phish_reimpl/               Core Python package (import this)
│   ├── visual.py               Base visual branch: Siamese encoder + brand matching
│   ├── visual_yolo.py          YOLOv8 detector subclass (overrides _detect_logo_boxes)
│   ├── url_analysis.py         URL heuristic + ML branch
│   ├── inference.py            PhishingPipeline: fuses visual + URL decisions
│   ├── dataset.py              Dataset loader (CaseSample, load_cases)
│   ├── results.py              ResultRow dataclass + TSV writer
│   ├── scoring.py              Accuracy / Precision / Recall / F1 helpers
│   └── backbone.py             ResNetV2 BiT-M Siamese encoder (feature extractor)
│
├── yolo/
│   └── best.pt                 Trained YOLOv8 logo-detector weights
│
├── assets/models/
│   ├── resnetv2_rgb_new.pth.tar  Siamese embedding model (ResNetV2 BiT-M, 2048-dim)
│   ├── domain_map.pkl          Brand → official domains lookup  {brand: [list of domains]}
│   ├── LOGO_FEATS.npy          Pre-computed reference logo embeddings (bootstrap cache)
│   ├── LOGO_FILES.npy          Matching filenames for LOGO_FEATS
│   └── expand_targetlist/      Reference brand logo images (one folder per brand)
│
├── artifacts/
│   └── url_rf_combined.pkl     Trained Random Forest URL classifier (125k corpus)
│
├── chrome_extension/           Browser extension (Manifest V3)
│   ├── manifest.json
│   ├── popup.html / popup.js   Extension UI + analysis trigger
│   └── background.js           Service worker
│
├── api_server.py               FastAPI server — bridges extension → pipeline
├── run_final.py                CLI for batch eval or single screenshot/URL
├── RUN_FINAL.sh                Shell wrapper for run_final.py
│
├── train_url_model.py          Train the URL Random Forest model
├── rebuild_reference_cache.py  Rebuild Siamese reference embedding cache
├── eval_comparison.py          Full 13-config comparison runner
├── main_yolo_url.py            Batch eval: YOLOv8 + URL (saves results TSV)
├── main.py                     Batch eval: Faster R-CNN baseline (reference)
└── score_results.py            Score a results TSV against ground-truth labels
```

---

## How the Pipeline Works (End-to-End)

### Inference (what runs when you click "Analyze" in the extension)

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
    │       │   └─ embeds each logo crop → 2048-dim L2-normalized vector
    │       │   └─ cosine-similarity nearest-neighbour vs reference embeddings
    │       │      (pre-built from expand_targetlist/ logos)
    │       │
    │       └─ Domain consistency check
    │           ├─ brand matched + domain MATCHES page URL  → visual_pred = 0 (benign)
    │           └─ brand matched + domain MISMATCH page URL → visual_pred = 1 (PHISHING)
    │
    ├─► URL Branch  [phish_reimpl/url_analysis.py]
    │       │
    │       ├─ infer_brand(): token match of brand names in URL
    │       ├─ Heuristic scoring: free-host, hyphen, depth, suspicious keywords
    │       └─ RF classifier (url_rf_combined.pkl) for borderline cases
    │           └─ url_pred = 0 or 1
    │
    └─► Fusion  [phish_reimpl/inference.py]
            │
            ├─ visual_pred == 1  →  final = 1  (PHISHING, visual wins, URL ignored)
            ├─ visual_pred == 0, url_pred == 1  →  final = 1  (URL catches what visual missed)
            └─ visual_pred == 0, url_pred == 0  →  final = 0  (BENIGN)
```

### Decision rule in plain English
> Visual branch has priority. If YOLO detects a brand logo and the page domain doesn't belong to that brand → **phishing, stop**. Only consult the URL branch when visual says benign.

---

## Step-by-Step: From Training to Running the Extension

### Step 1 — Install dependencies (once)

```bash
pip install torch torchvision ultralytics
pip install scikit-learn tldextract pandas numpy Pillow opencv-python-headless
pip install fastapi uvicorn python-multipart
```

### Step 2 — (Optional) Retrain the URL model

If you have a new URL dataset:

```bash
python3 train_url_model.py \
    --dataset /path/to/url_dataset.csv \
    --out artifacts/url_rf_combined.pkl
```

- Input: CSV with columns `url`, `label` (0=benign, 1=phish)
- Output: `artifacts/url_rf_combined.pkl` (replaces old model)

### Step 3 — (Optional) Rebuild Siamese reference cache

Only needed if you add new brand logos to `assets/models/expand_targetlist/`:

```bash
python3 rebuild_reference_cache.py --cache-dir cache/
```

This reads all logo images, runs them through the ResNetV2 encoder, and saves the embeddings to `cache/reference_embeddings.pkl`. The server loads this cache at startup — much faster than re-embedding every time.

### Step 4 — Start the API server

```bash
python3 api_server.py
# Server starts on http://127.0.0.1:8765
```

The server automatically locates `yolo/best.pt` by walking up the directory tree. Startup log confirms which mode loaded:

```
[startup] Workspace root: /path/to/yolo_url_final
[startup] Full pipeline loaded: YOLOv8 + Heuristic+ML (url_rf_combined 125k)
```

If it says `PIPELINE FAILED TO LOAD` — check the error. Set `PHISH_WORKSPACE=/absolute/path/to/folder` if the weights are elsewhere.

### Step 5 — Load the Chrome extension

1. Open `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked** → select the `chrome_extension/` folder
4. The PhishGuard icon appears in the toolbar

### Step 6 — Analyze a page

Click the extension icon → **Analyze visible tab**. Results appear within ~1–2 seconds.

---

## Batch Evaluation

### Quick eval on a dataset folder

```bash
python3 run_final.py --eval-dir /path/to/eval_2000
```

Dataset folder must contain case subfolders (each with `screenshot.png`, `info.txt`) plus `expected_labels.csv`.

### Full 13-configuration comparison

```bash
python3 eval_comparison.py --eval-dir /path/to/eval_auto
```

Runs all 13 detector × URL combinations and prints a comparison table.

### Score an existing results file

```bash
python3 score_results.py \
    --results results/yolo_url_results.tsv \
    --dataset /path/to/eval_2000
```

---

## Key Files Explained

### `phish_reimpl/visual.py`
The base visual analyser class. Key methods:
- `_match_crop(screenshot, bbox)` — embeds a logo crop and finds the nearest reference brand by cosine similarity (threshold: 0.87). Returns `(brand, domains, score)` or `(None, None, score)` if below threshold.
- `_is_domain_consistent(url, domains)` — checks if the page's registered domain matches the brand's official domains. Uses `tldextract`. Returns `True` (benign) or `False` (phishing).
- `analyze(case)` → `VisualDecision` — full pipeline: detect boxes → match → consistency check → return prediction + brand.

### `phish_reimpl/visual_yolo.py`
Thin subclass of `VisualAnalyzer`. Overrides `_detect_logo_boxes()` to use the YOLOv8 model instead of Faster R-CNN.

### `phish_reimpl/url_analysis.py`
- `infer_brand(url)` — checks if any known brand token (e.g. "amazon", "paypal") appears as a substring in the URL hostname or path.
- `analyze(url)` → `URLDecision` — full URL pipeline: brand check → official domain check → heuristic scoring → RF model supplement for borderline cases.

### `phish_reimpl/inference.py`
`PhishingPipeline` class. Loads both branches at startup, calls both for each case, applies the fusion rule. Used by `api_server.py`.

### `api_server.py`
FastAPI server with two endpoints:
- `GET /health` — returns current mode (`full` / `url_only` / `down`) and any pipeline error.
- `POST /analyze` — accepts `url` + `screenshot` (PNG), runs the full pipeline, returns JSON with `final_pred`, `predicted_brand`, `visual_pred`, `url_pred`, etc.

### `train_url_model.py`
Trains a Random Forest classifier on URL structural features (36 features: URL length, subdomain depth, special characters, TLD type, etc.). The combined 125k model (`url_rf_combined.pkl`) was trained on the PhiUSIIL dataset + additional benign URLs.

### `rebuild_reference_cache.py`
Scans `assets/models/expand_targetlist/` for brand logo images, embeds each through the ResNetV2 BiT-M encoder, and saves a `reference_embeddings.pkl` cache. Run this after adding new brands.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PHISH_WORKSPACE` | (auto-detect) | Absolute path to the folder containing `yolo/best.pt` |
| `PHISH_URL_ONLY` | `0` | Set to `1` to force URL-only mode (no visual branch) |
| `API_HOST` | `127.0.0.1` | Host for the API server |
| `API_PORT` | `8765` | Port for the API server |

---

## Results Summary (from evaluation)

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Faster R-CNN only (baseline) | 77.69% | 94.59% | 58.33% | 72.16% |
| Faster R-CNN + URL (midterm) | 96.69% | 95.16% | 98.33% | 96.72% |
| **YOLOv8 + Heuristic+ML (final)** | **best** | **best** | **best** | **best** |

The midterm → endterm improvement: replaced Faster R-CNN with YOLOv8 (faster, more accurate logo detection) and retrained the URL classifier on a 125k combined corpus.
