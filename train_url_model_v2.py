"""Train the path-aware URL Random Forest (artifacts/url_rf_pathaware.pkl).

Why v2: in every dataset the original model was trained on (PhiUSIIL,
url_corpus_14k, url_only_full) 0% of benign URLs have a path or even a
trailing '/', while phishing URLs do. The original RF learned "has a path"
== phishing and flagged ~96% of real web pages. v2 fixes the data, not just
the model:

  * benign URLs include real inner pages (from tools/crawl_benign_urls.py);
  * half of bare benign domains get a trailing '/' (how browsers show them);
  * the format-only feature has_https is dropped;
  * phishing rows stored as metadata dicts ("{'sector': ..., 'url': ...}")
    are unwrapped to their real URL;
  * registered domains listed in --exclude (test sets) are removed;
  * validation is split by registered domain, so a site never sits in both.

Example:
  python train_url_model_v2.py \
      --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv \
      --corpus url_corpus_14k.txt --corpus-csv url_only_full.csv \
      --benign-crawl benign_crawl.json --exclude test_urls.txt
"""
from __future__ import annotations

import argparse, ast, csv, json, pickle, random, re
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from phish_reimpl.url_analysis import extract_url_features, registered_domain

ROOT = Path(__file__).resolve().parent


def unwrap(s: str) -> str | None:
    s = s.strip()
    if not s.startswith("{"):
        return s
    try:
        return ast.literal_eval(unquote(s).replace("%2C", ","))["url"]
    except Exception:
        m = re.search(r"'url': '([^']+)'", unquote(s))
        return m.group(1) if m else None


def load(args) -> tuple[list[str], list[str], list[str]]:
    phish, benign, crawled = [], [], []
    if args.phiusiil:   # PhiUSIIL convention: label 1 = legitimate, 0 = phishing
        df = pd.read_csv(args.phiusiil, encoding="utf-8-sig", usecols=["URL", "label"])
        phish += df[df.label == 0].URL.tolist()
        benign += df[df.label == 1].URL.tolist()
    if args.corpus:     # tab-separated: url<TAB>label (1 = phishing)
        for line in open(args.corpus):
            u, l = line.rstrip("\n").rsplit("\t", 1)
            (phish if l == "1" else benign).append(unwrap(u))
    if args.corpus_csv:  # url,label (1 = phishing)
        for r in csv.reader(open(args.corpus_csv)):
            if r and r[0] != "url":
                (phish if r[-1] == "1" else benign).append(unwrap(",".join(r[:-1])))
    if args.benign_crawl:
        for c in json.load(open(args.benign_crawl)):
            crawled += [c["home"], *c["links"]]
    ok = lambda u: bool(u) and bool(urlparse(u).hostname)
    return ([u for u in phish if ok(u)],
            [u for u in benign if ok(u) and ".example." not in u],
            [u for u in crawled if ok(u)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phiusiil", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--corpus-csv", type=Path)
    ap.add_argument("--benign-crawl", type=Path, required=True,
                    help="JSON from tools/crawl_benign_urls.py (real inner-page URLs)")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="files with one test URL per line; their sites are kept out of training")
    ap.add_argument("--model-out", type=Path, default=ROOT / "artifacts" / "url_rf_pathaware.pkl")
    ap.add_argument("--trees", type=int, default=150)
    args = ap.parse_args()

    phish, benign, crawled = load(args)
    excl = {registered_domain(urlparse(l.strip()).hostname or "")
            for f in args.exclude for l in open(f) if l.strip()}
    keep = lambda u: registered_domain(urlparse(u).hostname or "") not in excl
    phish = list(dict.fromkeys(filter(keep, phish)))
    benign = list(dict.fromkeys(filter(keep, benign)))
    crawled = list(dict.fromkeys(filter(keep, crawled)))

    rng = random.Random(0)
    benign = [u + "/" if urlparse(u).path == "" and rng.random() < 0.5 else u for u in benign]
    urls = phish + benign + crawled
    y = np.array([1] * len(phish) + [0] * (len(benign) + len(crawled)))
    groups = [registered_domain(urlparse(u).hostname or "") for u in urls]
    X = pd.DataFrame([extract_url_features(u) for u in urls]).drop(columns=["has_https"])

    # benign URLs with real paths are rarer than phishing ones: weight them up
    has_path = np.array([urlparse(u).path not in ("", "/") for u in urls])
    w = np.ones(len(y))
    bp, pp = (y == 0) & has_path, (y == 1) & has_path
    w[bp] = pp.sum() / max(1, bp.sum())
    print(f"rows {len(y)}  phish {y.sum()}  benign-with-path {bp.sum()}  phish-with-path {pp.sum()}")

    rf = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=8, max_depth=28,
                                n_jobs=-1, random_state=0, class_weight="balanced")
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, groups))
    rf.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
    p = rf.predict_proba(X.iloc[va])[:, 1]
    for t in (0.5, 0.8, 0.9):
        pr = (p >= t).astype(int)
        print(f"site-grouped validation t={t}: acc {accuracy_score(y[va], pr):.3f} "
              f"precision {precision_score(y[va], pr):.3f} recall {recall_score(y[va], pr):.3f} "
              f"benign-with-path flagged {pr[(y[va] == 0) & has_path[va]].mean():.3f}")

    rf.fit(X, y, sample_weight=w)
    rf.path_aware_ = True   # URLAnalyzer passes the full URL (not just the hostname)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as f:
        pickle.dump(rf, f)
    print("saved", args.model_out)


if __name__ == "__main__":
    main()
