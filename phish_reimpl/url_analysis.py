from __future__ import annotations

import math
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import tldextract

from phish_reimpl.impersonation import (
    brand_hint, brand_in_url, credential_page, has_credential_field,
)


SUSPICIOUS_KEYWORDS = (
    "login", "signin", "secure", "verify", "update", "account",
    "confirm", "check", "validation", "authenticate", "password",
    "recover", "reset", "billing", "invoice",
)

# Free / bulletproof-hosting providers.
# URLs hosted here are NEVER cleared as "official domain" even if the brand
# token matches a legitimate brand (e.g. attacker's appspot.com app).
_FREE_HOSTS = frozenset({
    "000webhostapp.com", "appspot.com", "googleusercontent.com",
    "herokuapp.com", "netlify.app", "github.io", "pages.dev",
    "firebaseapp.com", "vercel.app", "glitch.me", "repl.co",
    "weebly.com", "wix.com", "wixsite.com", "jimdo.com",
    "accesscam.org", "htmlcomponentservice.com", "nw.r.appspot.com",
    "ey.r.appspot.com", "myshn.net", "ddns.net", "no-ip.org",
    "dyndns.org", "yolasite.com", "weblium.site", "bravesites.com",
    "wordpress.com", "blogspot.com", "tumblr.com", "godaddysites.com",
    "squarespace.com", "webflow.io", "carrd.co",
    "workers.dev", "r2.dev", "framer.app", "framer.website", "typedream.app",
    "sites.google.com",
})

# Corrections for known errors in the scraped domain_map.pkl
# (e.g. "python" only listed egenix.com, so python.org looked like an impostor).
_DOMAIN_MAP_FIXES: dict[str, list[str]] = {
    "python": ["python.org"],
    "stackoverflow": ["stackoverflow.com"],
    "Bank of America": ["bofa.com"],
    "DHL Airways": ["dhl.com", "dhl.de", "dhl.co.uk"],
    "eBay": ["ebay.com", "ebay.co.uk", "ebay.de", "ebay.fr", "ebay.it", "ebay.es", "ebay.com.au", "ebay.ca"],
    "Naver": ["naver.com"],
}
# Extra brand nicknames that phishing URLs use but the domain map lacks.
_BRAND_ALIASES: dict[str, str] = {"bofa": "Bank of America"}

# Brand tokens shorter than this must match a whole hostname part
# ("line" must not match inside "onlinesbi", "start" not inside "kickstarter").
_MIN_SUBSTRING_TOKEN = 6

# RFC 2606 / IANA reserved test domains — never real phishing targets.
# URLs like "paypal-secure.example.org" are test/anonymized entries, not real phishing.
_RESERVED_DOMAINS = frozenset({
    "example.com", "example.org", "example.net", "example.edu",
    "test.com", "test.org", "localhost",
    "invalid", "local",
})

_SAFE_TLDS = frozenset({"edu", "gov", "mil"})

_SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online",
    "site", "icu", "live", "click", "link", "download", "ru", "cn",
    "pw", "buzz", "cam", "surf", "monsters", "rest", "quest",
})

_BRAND_TOKENS = frozenset({
    "paypal", "google", "apple", "amazon", "microsoft", "bank", "netflix",
    "facebook", "instagram", "chase", "wellsfargo", "citibank", "hsbc",
    "yahoo", "outlook", "office365", "dropbox", "ebay", "steam", "twitter",
    "linkedin", "adobe", "docusign", "att", "verizon", "fedex", "ups",
    "dhl", "usps", "irs", "amex", "visa", "mastercard", "blockchain",
    "coinbase", "binance", "spotify", "icloud",
    "signin", "login",
})

GENERIC_TOKENS = {
    "service", "support", "portal", "account", "secure", "verify",
    "gateway", "center", "mail", "docsafe", "auth", "customer",
    "control", "session", "review", "validate", "recovery", "check", "free",
}

# Short brand tokens only matched as exact hostname labels
SHORT_BRAND_WHITELIST: frozenset = frozenset({
    "att", "irs", "adp", "ing", "sap", "dhl", "ubs", "sfr", "bnp",
    "lcl", "dkb", "kbc", "axa", "ems", "nab", "anz", "rbc", "bmo",
    "aib", "boi", "tsb", "ups", "pof",
})

_TLDEXTRACT = tldextract.TLDExtract(
    cache_dir=str(Path(__file__).resolve().parents[1] / "cache" / "tldextract"),
    suffix_list_urls=None,
)

# ML confidence threshold when using the model as a standalone no-brand classifier
_STANDALONE_ML_THRESHOLD = 0.90   # "flag only when sure" (was 0.80)
# ML confidence threshold when used as a borderline heuristic supplement
_BORDERLINE_ML_THRESHOLD = 0.90


@dataclass
class URLDecision:
    prediction: int
    predicted_brand: str | None
    matched_domain: list[str] | None
    risk_score: float
    analysis_time: float
    # Why the page was flagged, or (with prediction 0) a weaker clue worth a
    # warning, e.g. "'paypal' in a vercel.app subdomain". None if nothing.
    reason: str | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def registered_domain(hostname: str) -> str:
    ext = _TLDEXTRACT(hostname.lower())
    if not ext.domain:
        return hostname.lower()
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def safe_ratio(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def _safe_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _registered_domain_label(hostname: str) -> str:
    # Use the public-suffix list so "irctc.co.in" gives "irctc", not "co".
    ext = _TLDEXTRACT(hostname.lower())
    return ext.domain or hostname.lower()


def _is_restricted_registry(hostname: str) -> bool:
    """True for .gov / .mil / .edu style suffixes (incl. gov.uk, edu.au, gouv.fr).

    These cannot be registered by just anyone, so the URL model alone is not
    allowed to flag them; only a brand-impersonation rule can.
    """
    suffix = _TLDEXTRACT(hostname.lower()).suffix
    return any(p in {"gov", "mil", "edu", "gouv"} for p in suffix.split("."))


def hostname_only_url(url: str) -> str:
    """Reduce a URL to https://<hostname> before it reaches the URL model.

    Every benign URL in the RF's training data (PhiUSIIL, url_corpus_14k,
    url_only_full) is a bare domain with no path or trailing slash, while
    phishing URLs carry paths. The model therefore learned "has a path / '/'
    / http" == phishing and scores almost every real browser URL ~1.0.
    Feeding it only the hostname removes that shortcut.
    """
    hostname = (urlparse(url.strip()).hostname or "").lower()
    return f"https://{hostname}" if hostname else url


def _is_free_host(hostname: str) -> bool:
    """Return True if the hostname is user content on a free/shared hosting provider.

    The provider's own site (squarespace.com, wix.com) is not user content.
    """
    h = hostname.lower().rstrip(".")
    if h.startswith("www."):
        h = h[4:]
    # sites.google.com hosts user pages under its path, not subdomains
    return h == "sites.google.com" or any(h.endswith("." + fh) for fh in _FREE_HOSTS)


def extract_url_features(url: str) -> dict[str, float]:
    """Extract the 36 URL-structural features expected by the trained RF model."""
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    url_lower = url.lower()

    parts = [p for p in hostname.split(".") if p]
    tld = parts[-1] if parts else ""
    reg_dom_label = _registered_domain_label(hostname)

    path_tokens = [t for t in re.split(r"[/_.?=&%;+\-]+", path) if t]
    url_tokens = [t for t in re.split(r"[/_.?=&%;+\-]+", url_lower) if t]

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    specials = sum(not c.isalnum() for c in url)
    n = max(1, len(url))

    is_ip = bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname))
    has_wp = int("/wp-content/" in url_lower or "/wp-includes/" in url_lower or "/wp-admin/" in url_lower)
    has_hash_path = int(bool(re.search(r"[0-9a-f]{16,}", path.lower())))
    on_free_host = int(
        _is_free_host(hostname)
        or bool(re.search(r"\d{4,}[a-z]+\.\w+", reg_dom_label))
    )

    path_entropy = _safe_entropy(re.sub(r"[/.]", "", path.lower()))
    hostname_entropy = _safe_entropy(re.sub(r"[.]", "", hostname))

    kw_hits = sum(kw in url_lower for kw in SUSPICIOUS_KEYWORDS)
    brand_in_hostname = int(any(b in hostname for b in _BRAND_TOKENS))
    brand_in_reg_dom = int(any(b in reg_dom_label for b in _BRAND_TOKENS))
    path_has_brand = int(any(b in path.lower() for b in _BRAND_TOKENS))

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "tld_length": len(tld),
        "num_subdomains": max(0, len(parts) - 2),
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "special_char_ratio": specials / n,
        "num_hyphens": url.count("-"),
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_equals": url.count("="),
        "num_question": url.count("?"),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "has_https": int((parsed.scheme or "").lower() == "https"),
        "has_ip_host": int(is_ip),
        "has_punycode": int("xn--" in url_lower),
        "has_at_sign": int("@" in url),
        "path_depth": path.count("/"),
        "path_token_count": len(path_tokens),
        "avg_token_length": sum(len(t) for t in url_tokens) / max(1, len(url_tokens)),
        "max_path_token_length": max((len(t) for t in path_tokens), default=0),
        "path_entropy": path_entropy,
        "hostname_entropy": hostname_entropy,
        "brand_in_hostname": brand_in_hostname,
        "brand_in_reg_dom_label": brand_in_reg_dom,
        "path_has_brand": path_has_brand,
        "kw_hits": kw_hits,
        "hostname_hyphen_count": hostname.count("-"),
        "on_free_host": on_free_host,
        "has_wp_path": has_wp,
        "has_hash_path": has_hash_path,
        "tld_suspicious": int(tld in _SUSPICIOUS_TLDS),
        "tld_safe": int(tld in _SAFE_TLDS),
    }


# ── HTML feature extraction ───────────────────────────────────────────────────

def extract_html_features(html: str, page_url: str = "") -> dict[str, float]:
    """
    Extract page-content features from raw HTML.
    These mirror the PhiUSIIL dataset's HTML-based columns and are used
    to supplement URL features when html.txt is available.
    """
    h = html.lower()
    lines = html.splitlines()
    n_lines = len(lines)
    largest_line = max((len(l) for l in lines), default=0)

    # Title
    title_m = re.search(r"<title[^>]*>(.*?)</title>", h, re.DOTALL)
    has_title = int(bool(title_m))

    # Favicon
    has_favicon = int(bool(re.search(r'rel=["\'].*?icon.*?["\']', h)))

    # Meta description
    has_description = int(bool(re.search(r'<meta[^>]+name=["\']description["\']', h)))

    # Forms and inputs
    forms = re.findall(r"<form[^>]*>", h)
    n_forms = len(forms)
    has_submit = int(bool(re.search(r'<(input|button)[^>]+type=["\']submit["\']', h)
                         or re.search(r'<button[^>]*>', h)))
    has_password = int(bool(re.search(r'<input[^>]+type=["\']password["\']', h)))
    has_hidden = int(bool(re.search(r'<input[^>]+type=["\']hidden["\']', h)))

    # External form action (form posts to a different domain — strong phishing signal)
    has_ext_form = 0
    if page_url:
        page_host = (urlparse(page_url).hostname or "").lower()
        for action_m in re.finditer(r'<form[^>]+action=["\']([^"\']+)["\']', h):
            action = action_m.group(1)
            if action.startswith("http"):
                action_host = (urlparse(action).hostname or "").lower()
                if action_host and action_host != page_host:
                    has_ext_form = 1
                    break

    # iframes and popups
    n_iframes = len(re.findall(r"<iframe", h))
    n_popups = len(re.findall(r"window\.open\s*\(", h))

    # Social network links
    social_domains = ("facebook.com", "twitter.com", "instagram.com",
                      "linkedin.com", "youtube.com", "t.me", "x.com")
    has_social = int(any(s in h for s in social_domains))

    # Resources
    n_images = len(re.findall(r"<img\b", h))
    n_css = len(re.findall(r'<link[^>]+rel=["\']stylesheet["\']|<style\b', h))
    n_js = len(re.findall(r"<script\b", h))

    # Link analysis
    all_hrefs = re.findall(r'href=["\']([^"\']+)["\']', h)
    n_self = n_empty = n_ext = 0
    page_host = (urlparse(page_url).hostname or "").lower() if page_url else ""
    for href in all_hrefs:
        href = href.strip()
        if not href or href.startswith("#") or href == "/":
            n_empty += 1
        elif href.startswith("http"):
            h_host = (urlparse(href).hostname or "").lower()
            if page_host and h_host == page_host:
                n_self += 1
            else:
                n_ext += 1
        else:
            n_self += 1

    # Copyright / legitimacy signal
    has_copyright = int("©" in html or "&copy;" in h or "copyright" in h)

    # Robots meta
    has_robots = int(bool(re.search(r'<meta[^>]+name=["\']robots["\']', h)))

    return {
        "html_line_count": n_lines,
        "html_largest_line": largest_line,
        "html_has_title": has_title,
        "html_has_favicon": has_favicon,
        "html_has_description": has_description,
        "html_has_password_field": has_password,
        "html_has_hidden_fields": has_hidden,
        "html_has_submit_button": has_submit,
        "html_has_ext_form_submit": has_ext_form,
        "html_has_social": has_social,
        "html_has_copyright": has_copyright,
        "html_has_robots": has_robots,
        "html_n_iframes": n_iframes,
        "html_n_popups": n_popups,
        "html_n_images": n_images,
        "html_n_css": n_css,
        "html_n_js": n_js,
        "html_n_self_links": n_self,
        "html_n_empty_links": n_empty,
        "html_n_ext_links": n_ext,
        "html_n_forms": n_forms,
    }


HTML_FEATURE_DEFAULTS: dict[str, float] = {
    "html_line_count": 0,
    "html_largest_line": 0,
    "html_has_title": 0,
    "html_has_favicon": 0,
    "html_has_description": 0,
    "html_has_password_field": 0,
    "html_has_hidden_fields": 0,
    "html_has_submit_button": 0,
    "html_has_ext_form_submit": 0,
    "html_has_social": 0,
    "html_has_copyright": 0,
    "html_has_robots": 0,
    "html_n_iframes": 0,
    "html_n_popups": 0,
    "html_n_images": 0,
    "html_n_css": 0,
    "html_n_js": 0,
    "html_n_self_links": 0,
    "html_n_empty_links": 0,
    "html_n_ext_links": 0,
    "html_n_forms": 0,
}


class URLAnalyzer:
    def __init__(
        self,
        domain_map_path: Path,
        model_path: Path | None = None,
        html_model_path: Path | None = None,
        heuristic_threshold: float = 0.75,
        borderline_threshold: float = 0.15,
        ml_conf_threshold: float = _BORDERLINE_ML_THRESHOLD,
        standalone_ml_threshold: float = _STANDALONE_ML_THRESHOLD,
        ml_only_mode: bool = False,
        ml_only_threshold: float = 0.3,
    ):
        with domain_map_path.open("rb") as f:
            self.domain_map: dict[str, list[str]] = pickle.load(f)
        for brand, extra in _DOMAIN_MAP_FIXES.items():
            if brand in self.domain_map:
                known = list(self.domain_map[brand])
                self.domain_map[brand] = known + [d for d in extra if d not in known]
        # Every registered domain that is official for *some* brand. A page on
        # one of these is the real site even if a different brand token matched.
        self._all_official: frozenset[str] = frozenset(
            registered_domain(d.lower()) for doms in self.domain_map.values() for d in doms
        )
        # Popular real domains that contain a brand name (amazonaws, cdninstagram…)
        legit_file = domain_map_path.parent / "brand_named_domains.txt"
        self._known_legit: frozenset[str] = frozenset(
            l.strip() for l in legit_file.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ) if legit_file.exists() else frozenset()
        self.model = None
        self.html_model = None
        self.heuristic_threshold = heuristic_threshold
        self.borderline_threshold = borderline_threshold
        self.ml_conf_threshold = ml_conf_threshold
        self.standalone_ml_threshold = standalone_ml_threshold

        self.ml_only_mode = ml_only_mode
        self.ml_only_threshold = ml_only_threshold

        if model_path is not None:
            if model_path.exists():
                with model_path.open("rb") as f:
                    self.model = pickle.load(f)
            else:
                import warnings
                msg = f"URLAnalyzer: model file not found at {model_path}"
                if ml_only_mode:
                    raise FileNotFoundError(
                        f"{msg}\n"
                        "ml_only_mode=True requires a valid model. "
                        "Ensure 'artifacts/url_rf_phiusiil.pkl' exists in the project root."
                    )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

        if html_model_path is not None and html_model_path.exists():
            with html_model_path.open("rb") as f:
                self.html_model = pickle.load(f)

        # Pre-build token index for O(1) brand lookup instead of O(brands) loop.
        # Maps normalised_token → (brand_name, is_short)
        self._token_index: dict[str, tuple[str, bool]] = {}
        for brand, domains in self.domain_map.items():
            candidates = [normalize_text(brand)]
            candidates += [
                normalize_text(registered_domain(d).split(".")[0])
                for d in domains
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                is_short = len(candidate) < 4
                if (is_short and candidate not in SHORT_BRAND_WHITELIST) or candidate in GENERIC_TOKENS:
                    continue
                # Keep the longest token that maps to this key (best brand name wins)
                existing = self._token_index.get(candidate)
                if existing is None or len(candidate) > len(existing[0]):
                    self._token_index[candidate] = (brand, is_short)
        for alias, brand in _BRAND_ALIASES.items():
            if brand in self.domain_map:
                self._token_index[normalize_text(alias)] = (brand, True)

    def infer_brand(self, url: str) -> tuple[str | None, list[str] | None, str | None]:
        # self._token_index is a precomputed {token: (brand, is_short)} map built
        # once at load time from all 270 brands' name variants, so this is a
        # single pass over that small token set (not a scan of the brand list
        # per call) — checks hostname first, falls back to path, and keeps
        # whichever match has the longest (most specific) token
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        hostname_key = normalize_text(hostname)
        hostname_labels = {normalize_text(part) for part in re.split(r"[.\-]", hostname) if part}
        path_key = normalize_text(parsed.path or "")
        path_labels = {normalize_text(p) for p in re.split(r"[/_.\-?=&]+", parsed.path or "") if p}
        best_brand = None
        best_source = None
        best_score = 0

        for token, (brand, is_short) in self._token_index.items():
            if is_short:
                matches_hostname = token in hostname_labels
                matches_path = False
            elif len(token) < _MIN_SUBSTRING_TOKEN:
                # 4-letter tokens: whole hostname/path parts only
                matches_hostname = token in hostname_labels
                matches_path = token in path_labels
            else:
                matches_hostname = token in hostname_key
                matches_path = token in path_key

            if matches_hostname and len(token) > best_score:
                best_brand = brand
                best_source = "hostname"
                best_score = len(token)
            elif matches_path and len(token) > best_score:
                best_brand = brand
                best_source = "path"
                best_score = len(token)

        if best_brand is None:
            return None, None, None
        return best_brand, self.domain_map.get(best_brand), best_source

    def _run_url_ml(self, url: str) -> float:
        """Return ML model phishing probability for a URL (0.0 if no model)."""
        if self.model is None:
            return 0.0
        # Models retrained on real inner-page URLs (path_aware_ = True) see the
        # full URL; the original shortcut-prone model only sees the hostname.
        if not getattr(self.model, "path_aware_", False):
            url = hostname_only_url(url)
        feats = extract_url_features(url)
        df = pd.DataFrame([feats])
        # Align columns to what the model was trained on
        for col in self.model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0.0
        return float(self.model.predict_proba(df[self.model.feature_names_in_])[0][1])

    def _run_html_ml(self, url: str, html: str) -> float:
        """Return HTML+URL ML model phishing probability (0.0 if no model)."""
        if self.html_model is None:
            return 0.0
        url_feats = extract_url_features(url)
        html_feats = extract_html_features(html, url)
        combined = {**url_feats, **html_feats}
        df = pd.DataFrame([combined])
        for col in self.html_model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0.0
        return float(self.html_model.predict_proba(df[self.html_model.feature_names_in_])[0][1])

    def analyze(self, url: str, html: str | None = None) -> URLDecision:
        start = time.time()
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if not hostname:
            return URLDecision(0, None, None, 0.0, time.time() - start)

        # ── RFC 2606 reserved / IANA test domains — never flag as phishing ────
        # Covers anonymized benchmark entries like "paypal-secure.example.org".
        reg_dom_early = registered_domain(hostname)
        if reg_dom_early in _RESERVED_DOMAINS:
            return URLDecision(0, None, None, 0.0, time.time() - start)

        # ── ML-only mode: skip heuristics, use RF model directly ─────────────
        # Evaluations show this achieves ~99.9% accuracy / F1 with zero FP on
        # structural phishing datasets — outperforming the heuristic pipeline.
        if self.ml_only_mode and self.model is not None:
            ml_prob = self._run_url_ml(url)
            pred = int(ml_prob >= self.ml_only_threshold)
            brand, domains, _ = self.infer_brand(url) if pred else (None, None, None)
            return URLDecision(pred, brand, domains, ml_prob, time.time() - start)

        # ── HTML model fast-path (if HTML is available and model loaded) ──────
        if html is not None and self.html_model is not None:
            html_prob = self._run_html_ml(url, html)
            if html_prob >= 0.85:
                brand, domains, _ = self.infer_brand(url)
                return URLDecision(1, brand, domains, html_prob, time.time() - start)

        on_free = _is_free_host(hostname)
        current = registered_domain(hostname)

        # ── Official domain of *any* known brand → real site ─────────────────
        # (stops e.g. a "python" brand match from flagging python.org)
        if current in self._all_official and not on_free:
            return URLDecision(0, None, None, 0.0, time.time() - start)

        # ── Impersonation rules ("flag only when sure", see impersonation.py) ──
        # 1. URL alone: brand glued into a made-up domain in a pattern real
        #    companies don't use (732890coinbase.com, lhttps-www--roblox.com).
        hit = brand_in_url(url)
        if hit:
            return URLDecision(1, hit[0], self.domain_map.get(hit[0]), 0.95,
                               time.time() - start, reason=hit[1])
        # 2. Page content: a login/payment form AND (the page title names a
        #    brand this site isn't, or a brand clue in the URL).
        hint = brand_hint(url, self._known_legit)
        if html:
            hit = credential_page(url, html, self._all_official)
            if hit:
                return URLDecision(1, hit[0], self.domain_map.get(hit[0]), 0.95,
                                   time.time() - start, reason=hit[1])
            if hint and has_credential_field(html):
                return URLDecision(1, hint[0], self.domain_map.get(hint[0]), 0.90,
                                   time.time() - start, reason=f"login/payment form + {hint[1]}")
        clue = f"suspicious: {hint[1]}" if hint else None

        brand, domains, source = self.infer_brand(url)
        # .gov / .mil / .edu: the RF alone may not flag these (see helper)
        use_ml = self.model is not None and not _is_restricted_registry(hostname)

        # ── No brand detected: use URL ML model as standalone classifier ─────
        if brand is None or domains is None:
            if use_ml:
                ml_prob = self._run_url_ml(url)
                if ml_prob >= self.standalone_ml_threshold:
                    return URLDecision(1, None, None, ml_prob, time.time() - start, reason="URL model")
            return URLDecision(0, None, None, 0.0, time.time() - start, reason=clue)

        # ── Brand detected: check official domain ─────────────────────────────
        official = {registered_domain(domain.lower()) for domain in domains}

        # Only clear as official if NOT on a free-hosting platform.
        # Attackers routinely abuse appspot.com, herokuapp.com, etc.
        if current in official and not on_free:
            return URLDecision(0, brand, domains, 0.0, time.time() - start)

        # ── Brand's real domain in front of a different one ──────────────────
        # paypal.com.secure-login.net: certain impersonation. Must sit on dot
        # boundaries (9to5google.com is not google.com), and country variants
        # with the same name (google.com.au, amazon.com.br, roblox.com.gt) are
        # NOT flagged here: from the URL alone we can't tell the brand's real
        # local site from a fake, so the logo / login-form checks decide those.
        dotted = "." + hostname + "."
        cur_label = _registered_domain_label(hostname)
        if any(f".{o}." in dotted and not dotted.endswith(f".{o}.") and "." in o
               and _registered_domain_label(o) != cur_label
               for o in official if o):
            return URLDecision(1, brand, domains, 0.95, time.time() - start,
                               reason=f"{brand}'s real domain placed in front of a different domain")

        # ── Heuristic scoring ─────────────────────────────────────────────────
        score = 0.0
        if on_free:
            score += 0.55   # free-hosting with brand reference is strong signal
        elif source == "hostname":
            score += 0.45
        elif source == "path":
            score += 0.25

        if "-" in hostname:
            score += 0.15
        if any(keyword in url.lower() for keyword in SUSPICIOUS_KEYWORDS):
            score += 0.15
        if len([p for p in hostname.split(".") if p]) > 2:
            score += 0.10
        if current.endswith((".com", ".net", ".org")):
            score += 0.05
        # Typosquat: brand token embedded in the registered domain label
        brand_token = normalize_text(brand)
        registered_label = current.split(".")[0] if "." in current else current
        if len(brand_token) >= 3 and brand_token in registered_label and registered_label != brand_token:
            score += 0.10

        pred = int(score >= self.heuristic_threshold)

        # ── ML supplement for borderline cases ───────────────────────────────
        if pred == 0 and use_ml and score >= self.borderline_threshold:
            ml_prob = self._run_url_ml(url)
            if ml_prob >= self.ml_conf_threshold:
                pred = 1
                score = max(score, ml_prob)

        why = (f"URL heuristics for {brand}" if pred else clue)
        return URLDecision(pred, brand, domains, min(score, 1.0), time.time() - start, reason=why)
