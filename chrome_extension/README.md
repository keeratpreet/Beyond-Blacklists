# Chrome extension (local API)

This folder is an **unpacked** Manifest V3 extension ("**PhishGuard Local**"). It sends the **active tab URL** and a **PNG of the visible tab** to the FastAPI server in the parent directory (`api_server.py`) and shows a clean **Safe / Suspicious** verdict in the popup.

## Prerequisites

Use **Python 3.10.x** (not 3.12) and a **venv** (or Conda). The dependency pins match `PROJECT_SETUP.md`; a 3.12 venv often breaks during `pip install` with `pkgutil` / `ImpImporter` errors.

```bash
cd /path/to/reimpl_phishpedia_pipeline
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-api.txt
```

To run the **full visual+URL** pipeline (instead of the URL-only fallback), you also need the ML/vision stack (e.g. `opencv-python`, `torch`, `detectron2`, etc.). If `detectron2` fails to install from PyPI on your machine, install `torch` / `torchvision` first, then follow **`../PROJECT_SETUP.md`** for a platform-specific Detectron2 build. After the stack works, verify:

```bash
python -c "import torch, detectron2, pandas, cv2, fastapi; print('ok')"
```

If `pip install` fails with **externally-managed-environment**, you ran `pip` against the **system** Python (Ubuntu/Debian block global `pip` by policy). Use a venv (above), **or** `conda activate …` and `pip install -r requirements-api.txt` there. Do not use `sudo pip` or `--break-system-packages` unless you fully understand the risk.

## 1. Start the local API

From the **project root** (the directory that contains `main.py` and `api_server.py`):

```bash
source .venv/bin/activate   # if you use a venv; skip if you use conda
export API_HOST=127.0.0.1
export API_PORT=8765
python api_server.py
```

Wait until models finish loading (first request is slow; `/health` returns immediately but models load at startup).

Optional check:

```bash
curl -s http://127.0.0.1:8765/health
```

## 2. Load the extension in Chrome

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Click **Load unpacked** and choose this `chrome_extension` folder.

## 3. Analyze a page

1. Open a normal `http://` or `https://` page (not `chrome://`, not the Web Store).
2. Click the extension icon, then **Analyze visible tab**.

The popup shows a user-friendly summary by default, with a **Show details** button to reveal technical fields and the raw JSON response (`1` = phishing, `0` = benign for `final_pred`, `visual_pred`, `url_pred`).

## Changing the API port

1. Start the server with a different port, for example:

   ```bash
   export API_PORT=9000
   python api_server.py
   ```

2. Edit `popup.js`: set `API_BASE` to `http://127.0.0.1:9000`.
3. Edit `manifest.json`: add/update `host_permissions` to include `http://127.0.0.1:9000/*`.
4. On `chrome://extensions`, click **Reload** on this extension.

## Privacy

The screenshot and URL are sent only to the host you configure (`127.0.0.1` by default). Do not expose `api_server.py` to the public internet without authentication and TLS.
