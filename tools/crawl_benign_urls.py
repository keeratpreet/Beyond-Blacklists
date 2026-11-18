"""Collect real inner-page URLs from popular benign sites (Tranco list).

The original URL model only ever saw benign *bare domains*, so any URL with a
path looked like phishing. This gathers real inner-page URLs for training
(train_url_model_v2.py --benign-crawl) or for a benign test set.

usage: python tools/crawl_benign_urls.py <dir with top-1m.csv> [n_sites] [seed] [out.json]
       (top-1m.csv: https://tranco-list.eu/top-1m.csv.zip)
"""
import csv, random, re, sys, json
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
import urllib.request
import tldextract

H = sys.argv[1]
ext = tldextract.TLDExtract(suffix_list_urls=())
rank = [r[1] for r in csv.reader(open(f"{H}/top-1m.csv"))]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 700
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 11)
OUT = sys.argv[4] if len(sys.argv) > 4 else f"{H}/crawl.json"
sites = random.sample(rank[500:300000], N)

class P(HTMLParser):
    def __init__(s): super().__init__(); s.links = []
    def handle_starttag(s, tag, a):
        if tag == "a":
            for k, v in a:
                if k == "href" and v: s.links.append(v)

def reg(h): e = ext(h); return f"{e.domain}.{e.suffix}"

def crawl(d):
    try:
        req = urllib.request.Request(f"https://{d}/", headers={"User-Agent": "Mozilla/5.0 (compatible; url-dataset-builder)"})
        r = urllib.request.urlopen(req, timeout=8)
        final = r.geturl()
        if "text/html" not in r.headers.get("Content-Type", ""): return d, final, []
        p = P(); p.feed(r.read(400_000).decode("utf-8", "ignore"))
        out = []
        for l in p.links:
            u = urljoin(final, l).split("#")[0]
            pu = urlparse(u)
            if pu.scheme in ("http", "https") and pu.hostname and reg(pu.hostname) == reg(urlparse(final).hostname or d) and pu.path not in ("", "/"):
                out.append(u)
        out = list(dict.fromkeys(out))
        random.Random(d).shuffle(out)
        return d, final, out[:4]
    except Exception:
        return d, None, []

with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(crawl, sites))
ok = [(d, f, l) for d, f, l in res if f]
json.dump([dict(site=d, home=f, links=l) for d, f, l in ok], open(OUT, "w"), indent=0)
print("sites tried", len(sites), "reachable", len(ok), "inner links", sum(len(l) for _, _, l in ok))
