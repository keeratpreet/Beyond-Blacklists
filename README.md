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
│   ├── impersonation.py        "Flag only when sure" rules: brand-in-URL + login-form check
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
│   ├── brand_named_domains.txt Popular real domains containing a brand name (amazonaws, cdninstagram…)
│   ├── LOGO_FEATS.npy          Pre-computed reference logo embeddings (bootstrap cache) [LFS]
│   ├── LOGO_FILES.npy          Matching filenames for LOGO_FEATS [LFS]
│   └── expand_targetlist/      Reference brand logo images (270 brands, one folder each)
│
├── artifacts/
│   ├── url_rf_pathaware.pkl    URL Random Forest retrained with real inner-page URLs (used) [LFS]
│   └── url_rf_combined.pkl     Original URL Random Forest (fallback; sees hostname only) [LFS]
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
├── train_url_model.py          Original URL model trainer
├── train_url_model_v2.py       Trainer for url_rf_pathaware.pkl (see "Retrain" below)
├── tools/crawl_benign_urls.py  Collects real inner-page URLs for URL-model training
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
    │  screenshot (PNG) + URL + page HTML (optional)
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
    ├─► URL + Page Branch  [phish_reimpl/url_analysis.py + impersonation.py]
    │       │
    │       ├─ page on any known brand's official domain        → benign, stop
    │       ├─ brand glued into a made-up domain in a pattern real
    │       │   companies don't use (732890coinbase.com)          → PHISHING
    │       ├─ login/payment form on the page AND
    │       │   (page title names a brand this site isn't, or a
    │       │    brand clue in the URL: free host / folder name)  → PHISHING
    │       ├─ brand's real domain in front of another one
    │       │   (paypal.com.secure-login.net)                     → PHISHING
    │       ├─ infer_brand(): token match against the brand domain map
    │       ├─ Heuristic scoring: free-host (+0.55), brand-in-hostname (+0.45),
    │       │   hyphen (+0.15), suspicious keywords (+0.15), subdomain depth (+0.10)…
    │       └─ Random Forest (url_rf_pathaware.pkl) — flags only at ≥ 0.90
    │           └─ url_pred = 0 or 1, plus a `reason` (or a "suspicious: …" warning)
    │
    └─► Fusion  [phish_reimpl/inference.py]
            │
            ├─ visual_pred == 1                      →  final = 1  (PHISHING — visual wins)
            ├─ visual_pred == 0, url_pred == 1       →  final = 1  (URL catches visual miss)
            └─ visual_pred == 0, url_pred == 0       →  final = 0  (BENIGN)
```

### Decision rule in plain English
> Visual branch has priority. If YOLOv8 detects a known brand logo and the page domain does not belong to that brand → **phishing, stop**. Only consult the URL branch when the visual branch returns benign.
>
> Every rule is built to **flag only when sure**. Weaker clues (e.g. a brand name in a free-host page with no login form, such as `user.github.io/netflixclone`) are not flagged; they come back as a `"suspicious: …"` reason, which the extension shows as a caution.

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
[startup] Full pipeline loaded: YOLOv8 + Heuristic+ML (retrained path-aware URL RF)
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
- ⚠️ **Suspicious — impersonating Brand** — flagged; the popup also shows *why* (e.g. "login/payment form, page title names 'microsoft', but the page is on robutak.top")
- ⚠️ **Be careful** — a weaker clue, not enough to flag (don't enter passwords)

The extension sends the page's HTML (as a file) so the server can run the login-form check. Pages that can't be scripted (Chrome Web Store, PDFs) are analysed from URL + screenshot only. After updating the extension, reload it in `chrome://extensions`.

---

## Training & Evaluation

### Retrain the URL Random Forest model

In every dataset the original model was trained on (PhiUSIIL, url_corpus_14k, url_only_full) **0% of benign URLs have a path or even a trailing `/`**, while phishing URLs do. The original model therefore learned "has a path" = phishing and flagged ~96% of real web pages. `train_url_model_v2.py` fixes the data:

```bash
# 1. collect real inner-page URLs from sites not used for testing
python3 tools/crawl_benign_urls.py <dir with Tranco top-1m.csv> 700 11 benign_crawl.json
# 2. train
python3 train_url_model_v2.py \
    --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv \
    --corpus url_corpus_14k.txt --corpus-csv url_only_full.csv \
    --benign-crawl benign_crawl.json --exclude test_urls.txt
```

It adds real inner pages, gives half the bare benign domains a trailing `/`, drops the format-only `has_https` feature, unwraps phishing rows stored as metadata dicts, removes test-set sites (`--exclude`), and validates with a split grouped by site. Output: `artifacts/url_rf_pathaware.pkl` (marked `path_aware_ = True`; the analyzer shows older models only the hostname).

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
- `analyze(url, html=None)` → `URLDecision(prediction, brand, domains, risk_score, time, reason)` — official-domain check → impersonation rules → heuristic score → RF supplement (threshold **0.90** both standalone and borderline). The RF alone never flags `.gov` / `.edu` / `.mil` hosts.

### `phish_reimpl/impersonation.py`
High-precision rules using a short hand-picked brand list (the scraped `domain_map.pkl` contains junk brands and common words):
- `brand_in_url(url)` — brand glued into a made-up domain next to a number, a fake `www/http`, or a login word (`732890coinbase.com`, `paypal-account-verify.com`). Matches 10 of the Tranco top-1M domains, mostly typo-domains.
- `brand_hint(url)` — weaker clue (brand on a free host, glued into an unknown domain, or as a folder name); acted on only together with a login/payment form.
- `credential_page(url, html)` — password / card / seed-phrase field **and** a short, login-style title naming a brand whose domain this isn't. News headlines and "sign in with Apple/Google" routes are excluded.

### `phish_reimpl/inference.py`
`PhishingPipeline`: loads both branches at startup, runs both in sequence per case, applies fusion rule. Used by `api_server.py`.

### `api_server.py`
FastAPI server with:
- `GET /health` — returns mode (`full` / `url_only` / `down`) and pipeline error if any.
- `POST /analyze` — accepts `url` + `screenshot` (PNG multipart) + optional `html` (file part; text fields are capped at 1 MB), returns JSON with `final_pred`, `predicted_brand`, `visual_pred`, `url_pred`, `url_risk`, `confidence`, `reason`, `mode`.

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

### Re-evaluation on real-world data (November 2026)

The original results below were measured on a benchmark whose URLs all also appear in the URL model's training data, and whose benign URLs are bare domains. On real browser URLs the original system flagged most real websites. Re-tested on sets **not used for training or tuning**:

| Test set | Original | Current |
|---|---|---|
| Real inner pages wrongly flagged (1,387 pages, 401 sites) — URL branch | 96.6% | **1.3%** |
| Real homepages wrongly flagged (401) — URL branch | 96.0% | **0%** |
| Tranco top sites, `https://site/` (1,000) — URL branch | 91.8% | **1.1%** |
| Real login pages wrongly flagged (198, with page HTML) | — | **0%** |
| News articles about brands wrongly flagged (245) | — | **1.2%** |
| Live phishing caught, full pipeline (73 PhishUnt pages with screenshots) | — | **70 / 73** |
| Live phishing caught, URL + page check (82 PhishUnt pages) | — | **95.1%** |
| Benchmark phishing caught, URL + page check (248) | — | **79.8%** |
| OpenPhish URLs caught, URL only (300) | 99.7%* | **81.7%** |

\* The original's near-100% phishing recall came largely from the "has a path" shortcut, which also flagged ~95% of real pages.

Other fixes: 8 reference-logo folders (Office365, OneDrive, Outlook, DHL, Simplii… 92 logos) were never loaded because their names were not in `domain_map.pkl`; `amazon.tk`-style open TLDs no longer pass as a brand's country site; detectron2 is only imported when the Faster R-CNN detector is used.

Known limits: Apple's small nav-bar logo is not detected by YOLO; a brand's name on an unusual country domain (`roblox.com.gt`) is not flagged from the URL alone (99 real brand country sites look the same); live test sets are small (73–300 pages).

### Original results (benchmark overlapping training data — do not quote)

Evaluated on 2000 balanced phishing/benign cases (50 brands × 2 cases × 20 variants):

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Faster R-CNN only (baseline) | 77.69% | 94.59% | 58.33% | 72.16% |
| Faster R-CNN + URL (midterm) | 96.69% | 95.16% | 98.33% | 96.72% |
| **YOLOv8 + Heuristic+ML (final — this repo)** | **95.10%** | **91.07%** | **100.00%** | **95.33%** |

Key improvement midterm → final: replaced Faster R-CNN with YOLOv8 (faster, no detectron2 dependency, better small-logo recall) and retrained the URL classifier on a 125k combined corpus (vs 40k previously). The final model achieves **100% recall** — zero phishing pages missed.
