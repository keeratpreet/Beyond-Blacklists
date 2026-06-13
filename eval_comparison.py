"""
Full 13-configuration comparison on eval_auto (500 cases) or eval_2000 (2000 cases).
Needs: torch, detectron2 (for RCNN), ultralytics (for YOLO), tldextract, scikit-learn

Run:
    python eval_comparison.py [--eval-dir eval_auto|eval_2000] [--no-rcnn] [--no-yolo]

Label files supported automatically:
    expected_labels.csv  (eval_auto)  — columns: case_id, label (0/1)
    manifest.csv         (eval_2000)  — columns: case_id, label (benign/phish), ...
"""
import sys, csv, time, json, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import tldextract; tldextract.extract("warmup.example.com")
from phish_reimpl.dataset import load_cases
from phish_reimpl.url_analysis import URLAnalyzer

NEW_MODEL  = REPO / "artifacts" / "url_rf_combined.pkl"
OLD_MODEL  = REPO / "artifacts" / "url_rf_phiusiil.pkl"
DOMAIN_MAP = REPO / "assets" / "models" / "domain_map.pkl"


def load_labels(eval_dir: Path) -> dict[str, int]:
    """
    Load ground-truth labels from the eval directory.
    Tries expected_labels.csv first (eval_auto format),
    then manifest.csv (eval_2000 format with string labels).
    """
    # Format 1: expected_labels.csv  — case_id, label (int 0/1)
    csv1 = eval_dir / "expected_labels.csv"
    if csv1.exists():
        labels = {}
        with open(csv1) as f:
            for row in csv.DictReader(f):
                labels[row["case_id"]] = int(row["label"])
        return labels

    # Format 2: manifest.csv  — case_id, label (string "benign"/"phish"), ...
    csv2 = eval_dir / "manifest.csv"
    if csv2.exists():
        labels = {}
        with open(csv2) as f:
            for row in csv.DictReader(f):
                labels[row["case_id"]] = 1 if row["label"].strip().lower() == "phish" else 0
        return labels

    raise FileNotFoundError(
        f"No label file found in {eval_dir}. "
        "Expected 'expected_labels.csv' (eval_auto) or 'manifest.csv' (eval_2000)."
    )


def metrics(preds, labs) -> dict:
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
                Acc=round(acc * 100, 2), Prec=round(prec, 4),
                Rec=round(rec, 4), F1=round(f1, 4))


def fuse(visual_preds, url_preds) -> list:
    return [1 if v == 0 and u == 1 else v
            for v, u in zip(visual_preds, url_preds)]


def run_visual_rcnn(cases) -> list[int]:
    from phish_reimpl.visual import VisualAnalyzer
    va = VisualAnalyzer()
    preds = []
    for i, c in enumerate(cases):
        if i % 25 == 0:
            print(f"    RCNN {i}/{len(cases)} ...", flush=True)
        out = va.analyze(c)
        preds.append(out.prediction)
    return preds


def run_visual_yolo(cases) -> list[int]:
    from phish_reimpl.visual_yolo import YoloVisualAnalyzer
    va = YoloVisualAnalyzer()
    preds = []
    for i, c in enumerate(cases):
        if i % 25 == 0:
            print(f"    YOLO {i}/{len(cases)} ...", flush=True)
        out = va.analyze(c)
        preds.append(out.prediction)
    return preds


def make_url(model_path, ml_only=True, thr=0.3, bl=0.15, sa=0.97):
    return URLAnalyzer(
        domain_map_path=DOMAIN_MAP, model_path=model_path,
        ml_only_mode=ml_only, ml_only_threshold=thr,
        borderline_threshold=bl, standalone_ml_threshold=sa,
    )


def print_table(results: dict, n_cases: int, n_phish: int, eval_name: str = "eval") -> None:
    print("\n" + "=" * 102)
    print(f"  RESULTS ON {eval_name} ({n_cases} cases: {n_phish} phish + {n_cases - n_phish} benign)")
    print("=" * 102)
    hdr = (f"{'Configuration':<48} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
           f"  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print(hdr)
    print("-" * 102)
    for name, m in results.items():
        print(f"{name:<48} {m['Acc']:>6.2f}%  {m['Prec']:>6.4f}  {m['Rec']:>6.4f}  {m['F1']:>6.4f}"
              f"  {m['TP']:>4}  {m['FP']:>4}  {m['TN']:>4}  {m['FN']:>4}")
    print("=" * 102)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval_auto",
                    help="Path to evaluation dataset folder")
    ap.add_argument("--no-rcnn",  action="store_true",
                    help="Skip Faster R-CNN (use if detectron2 not installed)")
    ap.add_argument("--no-yolo",  action="store_true",
                    help="Skip YOLO (use if ultralytics not installed)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    print(f"Loading cases from {eval_dir} ...")
    cases  = load_cases(eval_dir)
    labels = load_labels(eval_dir)
    labs   = [labels[c.case_id] for c in cases]
    urls   = [c.url for c in cases]
    n      = len(cases)
    n_phish = sum(labs)
    print(f"  {n} cases  ({n_phish} phish, {n - n_phish} benign)")

    results = {}

    # ── URL-only configurations (no GPU needed) ───────────────────────────
    print("\n[URL] ML-only — OLD 14k model (t=0.3) ...")
    preds = [make_url(OLD_MODEL, True, 0.3).analyze(u).prediction for u in urls]
    results["ML-only [OLD 14k, t=0.3]"] = metrics(preds, labs)

    print("[URL] ML-only — NEW combined model (t=0.3) ...")
    a = make_url(NEW_MODEL, True, 0.3)
    url_ml03 = [a.analyze(u).prediction for u in urls]
    results["ML-only [NEW combined, t=0.3]"] = metrics(url_ml03, labs)

    print("[URL] ML-only — NEW combined model (t=0.5) ...")
    a = make_url(NEW_MODEL, True, 0.5)
    url_ml05 = [a.analyze(u).prediction for u in urls]
    results["ML-only [NEW combined, t=0.5]"] = metrics(url_ml05, labs)

    print("[URL] Heuristic+ML — OLD model ...")
    preds = [make_url(OLD_MODEL, False).analyze(u).prediction for u in urls]
    results["Heuristic+ML [OLD model]"] = metrics(preds, labs)

    print("[URL] Heuristic+ML — NEW combined model ...")
    a = make_url(NEW_MODEL, False)
    url_heur = [a.analyze(u).prediction for u in urls]
    results["Heuristic+ML [NEW combined]"] = metrics(url_heur, labs)

    # ── Visual-only (needs GPU) ───────────────────────────────────────────
    rcnn_preds = yolo_preds = None

    if not args.no_rcnn:
        try:
            print("\n[RCNN] Visual-only (Faster R-CNN) ...")
            t0 = time.time()
            rcnn_preds = run_visual_rcnn(cases)
            print(f"  Done in {time.time()-t0:.1f}s")
            results["Visual-only [Faster R-CNN]"] = metrics(rcnn_preds, labs)

            print("[RCNN] FasterRCNN + ML-only NEW (t=0.3) ...")
            results["FasterRCNN + ML-only NEW (t=0.3)"] = metrics(fuse(rcnn_preds, url_ml03), labs)

            print("[RCNN] FasterRCNN + ML-only NEW (t=0.5) ...")
            results["FasterRCNN + ML-only NEW (t=0.5)"] = metrics(fuse(rcnn_preds, url_ml05), labs)

            print("[RCNN] FasterRCNN + Heuristic+ML NEW ...")
            results["FasterRCNN + Heuristic+ML NEW"] = metrics(fuse(rcnn_preds, url_heur), labs)
        except Exception as e:
            print(f"  !! RCNN failed: {e}. Use --no-rcnn to skip.")

    if not args.no_yolo:
        try:
            print("\n[YOLO] Visual-only (YOLOv8) ...")
            t0 = time.time()
            yolo_preds = run_visual_yolo(cases)
            print(f"  Done in {time.time()-t0:.1f}s")
            results["Visual-only [YOLOv8]"] = metrics(yolo_preds, labs)

            print("[YOLO] YOLOv8 + ML-only NEW (t=0.3) ...")
            results["YOLOv8 + ML-only NEW (t=0.3)"] = metrics(fuse(yolo_preds, url_ml03), labs)

            print("[YOLO] YOLOv8 + ML-only NEW (t=0.5) ...")
            results["YOLOv8 + ML-only NEW (t=0.5)"] = metrics(fuse(yolo_preds, url_ml05), labs)

            print("[YOLO] YOLOv8 + Heuristic+ML NEW ...")
            results["YOLOv8 + Heuristic+ML NEW"] = metrics(fuse(yolo_preds, url_heur), labs)
        except Exception as e:
            print(f"  !! YOLO failed: {e}. Use --no-yolo to skip.")

    print_table(results, n, n_phish, eval_name=eval_dir.name)

    out_tsv  = REPO / "eval_comparison_results.tsv"
    out_json = REPO / "eval_comparison_results.json"
    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Configuration","Accuracy","Precision","Recall","F1","TP","FP","TN","FN"])
        for name, m in results.items():
            w.writerow([name,m["Acc"],m["Prec"],m["Rec"],m["F1"],m["TP"],m["FP"],m["TN"],m["FN"]])
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_tsv}")
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
