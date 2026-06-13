#!/usr/bin/env python3
"""
run_final.py  —  Inference entry point for the final configuration:
    YOLOv8 + Heuristic+ML (new combined 125k URL model)

Usage
-----
Batch evaluation (dataset folder with case subfolders + expected_labels.csv / manifest.csv):
    python3 run_final.py --eval-dir eval_2000

Single case (screenshot + URL):
    python3 run_final.py --screenshot path/to/shot.png --url "http://suspect.example.com"
"""

import sys, csv, json, argparse, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import tldextract; tldextract.extract("warmup.example.com")
from phish_reimpl.dataset import load_cases, CaseSample
from phish_reimpl.url_analysis import URLAnalyzer
from phish_reimpl.visual_yolo import YoloVisualAnalyzer

# ── Model paths ──────────────────────────────────────────────────────────────
URL_MODEL  = REPO / "artifacts" / "url_rf_combined.pkl"
DOMAIN_MAP = REPO / "assets" / "models" / "domain_map.pkl"
YOLO_PT    = REPO / "yolo" / "best.pt"


def make_url_analyzer():
    return URLAnalyzer(
        domain_map_path=DOMAIN_MAP,
        model_path=URL_MODEL,
        ml_only_mode=False,          # Heuristic+ML mode
        borderline_threshold=0.15,
        standalone_ml_threshold=0.97,
    )


def fuse(visual_pred: int, url_pred: int) -> int:
    """Final decision rule: URL branch overrides visual only when visual=benign, URL=phish."""
    return 1 if visual_pred == 0 and url_pred == 1 else visual_pred


def metrics(preds, labs):
    tp = fp = tn = fn = 0
    for p, l in zip(preds, labs):
        if   l == 1 and p == 1: tp += 1
        elif l == 0 and p == 1: fp += 1
        elif l == 0 and p == 0: tn += 1
        else:                   fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = (tp + tn) / len(labs)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                Acc=round(acc * 100, 2),
                Prec=round(prec, 4), Rec=round(rec, 4), F1=round(f1, 4))


def load_labels(eval_dir: Path) -> dict:
    for fname, label_col, is_int in [
        ("expected_labels.csv", "label", True),
        ("manifest.csv",        "label", False),
    ]:
        p = eval_dir / fname
        if p.exists():
            labels = {}
            with open(p) as f:
                for row in csv.DictReader(f):
                    val = row[label_col].strip()
                    labels[row["case_id"]] = int(val) if is_int else (1 if val.lower() == "phish" else 0)
            return labels
    raise FileNotFoundError(f"No label file found in {eval_dir}")


def run_batch(eval_dir: Path):
    print(f"Loading cases from {eval_dir} ...")
    cases  = load_cases(eval_dir)
    labels = load_labels(eval_dir)
    labs   = [labels[c.case_id] for c in cases]
    n      = len(cases)
    n_phish = sum(labs)
    print(f"  {n} cases  ({n_phish} phish, {n - n_phish} benign)")

    print("\nInitialising models ...")
    va  = YoloVisualAnalyzer(yolo_weights_path=YOLO_PT)
    ua  = make_url_analyzer()
    print("  Models loaded.")

    print("\nRunning inference ...")
    yolo_preds, url_preds, final_preds = [], [], []
    t0 = time.time()
    for i, c in enumerate(cases):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i:4d}/{n}]  {elapsed:.1f}s elapsed", flush=True)
        v = va.analyze(c).prediction
        u = ua.analyze(c.url).prediction
        yolo_preds.append(v)
        url_preds.append(u)
        final_preds.append(fuse(v, u))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/n:.3f}s/case)")

    m_yolo  = metrics(yolo_preds,  labs)
    m_url   = metrics(url_preds,   labs)
    m_final = metrics(final_preds, labs)

    print("\n" + "=" * 76)
    print(f"  RESULTS ON {eval_dir.name} ({n} cases: {n_phish} phish + {n - n_phish} benign)")
    print("=" * 76)
    hdr = f"  {'Configuration':<32} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}"
    print(hdr)
    print("  " + "-" * 74)
    for name, m in [("Visual-only [YOLOv8]", m_yolo),
                    ("URL-only [Heuristic+ML NEW]", m_url),
                    ("YOLOv8 + Heuristic+ML NEW  [FINAL]", m_final)]:
        print(f"  {name:<32} {m['Acc']:>6.2f}%  {m['Prec']:>6.4f}  {m['Rec']:>6.4f}  {m['F1']:>6.4f}"
              f"  {m['TP']:>4}  {m['FP']:>4}  {m['TN']:>4}  {m['FN']:>4}")
    print("=" * 76)

    out = {
        "dataset": eval_dir.name,
        "n_cases": n, "n_phish": n_phish,
        "elapsed_seconds": round(elapsed, 1),
        "Visual-only [YOLOv8]": m_yolo,
        "URL-only [Heuristic+ML NEW]": m_url,
        "YOLOv8 + Heuristic+ML NEW [FINAL]": m_final,
    }
    out_path = REPO / "yolo_url_final_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {out_path}")


def run_single(screenshot: Path, url: str):
    print(f"Screenshot : {screenshot}")
    print(f"URL        : {url}")
    print("\nInitialising models ...")
    va = YoloVisualAnalyzer(yolo_weights_path=YOLO_PT)
    ua = make_url_analyzer()
    print("  Models loaded.\n")

    case = CaseSample(case_id="single", url=url,
                      screenshot_path=screenshot, html_path=None)

    t0 = time.time()
    v_out = va.analyze(case)
    u_out = ua.analyze(url)
    final = fuse(v_out.prediction, u_out.prediction)
    elapsed = time.time() - t0

    verdict = "PHISHING" if final == 1 else "BENIGN"
    brand   = v_out.predicted_brand or "N/A"
    print(f"Verdict        : {verdict}")
    print(f"Visual branch  : {'phish' if v_out.prediction else 'benign'}  (brand={brand})")
    print(f"URL branch     : {'phish' if u_out.prediction else 'benign'}")
    print(f"Inference time : {elapsed:.2f}s")


def main():
    ap = argparse.ArgumentParser(
        description="Run YOLOv8 + Heuristic+ML (new 125k model) — the final selected configuration")
    ap.add_argument("--eval-dir",   type=Path, help="Batch: path to eval dataset folder")
    ap.add_argument("--screenshot", type=Path, help="Single: path to screenshot PNG")
    ap.add_argument("--url",        type=str,  help="Single: URL string")
    args = ap.parse_args()

    if args.eval_dir:
        run_batch(args.eval_dir)
    elif args.screenshot and args.url:
        run_single(args.screenshot, args.url)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
