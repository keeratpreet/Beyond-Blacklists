"""High-precision "flag only when sure" impersonation rules.

1. brand_in_url():   a commonly phished brand's name sits in a host that is
   not the brand's: glued into a made-up domain (732890coinbase.com) or on
   a free host (x.github.io/instagram-login, paypal-verify.vercel.app).
2. credential_page(): the page asks for a password / card / seed phrase AND
   its <title> names a brand whose official domain this is not.

Only a short hand-picked list of distinctive brand names is used. The scraped
domain_map has hundreds of junk "brands" (kinsta, instahack, ...) and common
words (apple, chase, steam, outlook, ledger) that would cause false alarms.
"""
from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlparse

import tldextract

_TLD = tldextract.TLDExtract(suffix_list_urls=None)

# token -> (brand name, registered domains the brand really owns)
TARGET_BRANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "paypal":          ("PayPal", ("paypal.com", "paypal.me", "paypalobjects.com", "paypal-community.com")),
    "coinbase":        ("Coinbase", ("coinbase.com",)),
    "instagram":       ("Instagram", ("instagram.com",)),
    "facebook":        ("Facebook", ("facebook.com", "fb.com", "facebookmail.com", "facebook.net")),
    "whatsapp":        ("WhatsApp", ("whatsapp.com", "whatsapp.net")),
    "microsoft":       ("Microsoft", ("microsoft.com", "microsoftonline.com", "live.com", "office.com", "microsoft365.com")),
    "office365":       ("Microsoft", ("office365.com", "office.com", "microsoft.com")),
    "onedrive":        ("Microsoft", ("onedrive.com", "live.com", "microsoft.com")),
    "sharepoint":      ("Microsoft", ("sharepoint.com", "microsoft.com")),
    "hotmail":         ("Microsoft", ("hotmail.com", "live.com")),
    "netflix":         ("Netflix", ("netflix.com", "netflix.net")),
    "amazon":          ("Amazon", ("amazon.com", "amazonaws.com", "amazon.jobs")),
    "icloud":          ("Apple", ("icloud.com", "apple.com")),
    "binance":         ("Binance", ("binance.com", "binance.us")),
    "metamask":        ("MetaMask", ("metamask.io",)),
    "wellsfargo":      ("Wells Fargo", ("wellsfargo.com", "wf.com")),
    "bankofamerica":   ("Bank of America", ("bankofamerica.com", "bofa.com")),
    "linkedin":        ("LinkedIn", ("linkedin.com",)),
    "dropbox":         ("Dropbox", ("dropbox.com", "dropboxusercontent.com")),
    "docusign":        ("DocuSign", ("docusign.com", "docusign.net")),
    "roblox":          ("Roblox", ("roblox.com",)),
    "steamcommunity":  ("Steam", ("steamcommunity.com",)),
    "steampowered":    ("Steam", ("steampowered.com",)),
    "spotify":         ("Spotify", ("spotify.com",)),
    "fedex":           ("FedEx", ("fedex.com",)),
    "americanexpress": ("American Express", ("americanexpress.com",)),
    "mastercard":      ("Mastercard", ("mastercard.com",)),
    "trezor":          ("Trezor", ("trezor.io",)),
    "kucoin":          ("KuCoin", ("kucoin.com",)),
    "trustwallet":     ("Trust Wallet", ("trustwallet.com",)),
    "discord":         ("Discord", ("discord.com", "discord.gg", "discordapp.com")),
    "telegram":        ("Telegram", ("telegram.org", "t.me")),
    "gmail":           ("Google", ("gmail.com", "google.com")),
    "yahoo":           ("Yahoo", ("yahoo.com",)),
    "citibank":        ("Citibank", ("citibank.com", "citi.com")),
    "santander":       ("Santander", ("santander.com",)),
    "revolut":         ("Revolut", ("revolut.com",)),
    "usps":            ("USPS", ("usps.com",)),
}

# Hosts where anyone can publish under a subdomain or path.
FREE_HOSTS = (
    "github.io", "gitlab.io", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
    "r2.dev", "web.app", "firebaseapp.com", "herokuapp.com", "glitch.me", "repl.co",
    "000webhostapp.com", "weebly.com", "wixsite.com", "webflow.io", "framer.app",
    "framer.website", "typedream.app", "blogspot.com", "godaddysites.com", "carrd.co",
    "square.site", "ipfs.io", "dweb.link", "translate.goog",
)
# On these, a user/org page named exactly after a brand is that brand's own
# (netflix.github.io is Netflix's open-source site).
_ORG_PAGE_HOSTS = ("github.io", "gitlab.io")


def _reg(host: str) -> tuple[str, str]:
    e = _TLD(host)
    return e.domain, (f"{e.domain}.{e.suffix}" if e.suffix else e.domain)


def _free_host(host: str) -> str | None:
    return next((f for f in FREE_HOSTS if host == f or host.endswith("." + f)), None)


_URL_KEYWORDS = ("login", "signin", "secure", "verify", "update", "account", "support", "wallet",
                 "auth", "recovery", "unlock", "confirm", "billing", "help")
# OAuth "sign in with X" routes legitimately carry these brand names in the path.
_SSO_PROVIDERS = {"google", "facebook", "apple", "microsoft", "twitter", "linkedin", "github", "yahoo", "amazon"}
_SSO_PATH = re.compile(r"o?auth|sso|social|callback|connect|openid|saml", re.I)


def _at_edge(label: str, tok: str) -> bool:
    """Token starts/ends the label or is next to a digit/hyphen (not inside a word)."""
    for m in re.finditer(re.escape(tok), label):
        a, z = m.start(), m.end()
        if a == 0 or not label[a - 1].isalpha() or z == len(label) or not label[z].isalpha():
            return True
    return False


def brand_in_url(url: str) -> tuple[str, str] | None:
    """URL alone is enough: a target brand glued into a made-up domain in a
    pattern real companies don't use (732890coinbase.com, lhttps-www--roblox.com,
    paypal-account-verify.com). ~10 hits in the Tranco top 1M, mostly typo-domains."""
    host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower().rstrip(".")
    if not host or _free_host(host):
        return None
    label, registered = _reg(host)
    for tok, (brand, owned) in TARGET_BRANDS.items():
        if tok not in label or label == tok or registered in owned or not _at_edge(label, tok):
            continue
        rest = label.replace(tok, " ")
        if re.search(rf"\d{{3,}}{tok}|{tok}\d{{3,}}", label):
            return brand, f"'{tok}' glued to a number in the domain {registered}"
        if re.search(r"www|https?", rest):
            return brand, f"'{tok}' with a fake 'www/http' in the domain {registered}"
        if "-" in label and any(k in rest for k in _URL_KEYWORDS):
            return brand, f"'{tok}' plus a login/verify word in the domain {registered}"
    return None


def brand_hint(url: str, known_legit: frozenset[str] = frozenset()) -> tuple[str, str] | None:
    """Weaker clue, only acted on together with a login/payment form: a brand
    name on a free host, glued into an unknown domain, or as a path folder."""
    p = urlparse(url if "://" in url else "http://" + url)
    host = (p.hostname or "").lower().rstrip(".")
    if not host:
        return None
    label, registered = _reg(host)
    fh = _free_host(host)
    # whole folder / file names only ("/BankofAmerica/login.php", "/chasedept/");
    # words inside article slugs ("/apple-maps-gets-upgrade") don't count
    segs = [re.sub(r"\.[a-z0-9]{2,5}$", "", x) for x in p.path.lower().split("/")]
    segs = [re.sub(r"[^a-z0-9]", "", x) for x in segs]
    segs = [x for x in segs if x]
    sso = bool(_SSO_PATH.search(p.path))

    for tok, (brand, owned) in TARGET_BRANDS.items():
        if registered in owned:
            return None
        if fh:
            sub = host[: -len(fh) - 1] if host != fh else ""
            if tok in re.sub(r"[^a-z0-9]", "", sub) and not (fh in _ORG_PAGE_HOSTS and sub == tok):
                return brand, f"'{tok}' in a {fh} subdomain"
        elif tok in label and label != tok and _at_edge(label, tok) and registered not in known_legit:
            return brand, f"'{tok}' inside the domain {registered}"
    for seg in segs:
        for tok, (brand, owned) in TARGET_BRANDS.items():
            if tok in seg and len(seg) <= len(tok) + 6 and not (sso and tok in _SSO_PROVIDERS):
                return brand, f"'{tok}' as a folder in the page path"
        for b, owned in TITLE_BRANDS.items():
            bk = re.sub(r"[^a-z0-9]", "", b)
            if len(bk) >= 5 and seg == bk and owned and registered not in owned \
                    and not (sso and bk in _SSO_PROVIDERS):
                return b, f"'{b}' as a folder in the page path"
    return None


# ── Page content ────────────────────────────────────────────────────────────
# Brand names as they appear in login-page titles -> registered domains the
# brand owns. Hand-picked from the Phishpedia target list plus common targets.
# A title brand only counts when it is not also part of the page's own domain.
TITLE_BRANDS: dict[str, tuple[str, ...]] = {
    "paypal": ("paypal.com",), "coinbase": ("coinbase.com",),
    "instagram": ("instagram.com",), "facebook": ("facebook.com", "fb.com"),
    "meta": ("facebook.com", "meta.com"), "whatsapp": ("whatsapp.com",),
    "microsoft": ("microsoft.com", "microsoftonline.com", "live.com", "office.com"),
    "office 365": ("office.com", "microsoft.com"), "office365": ("office.com", "microsoft.com"),
    "outlook": ("outlook.com", "live.com", "office.com", "microsoft.com"),
    "onedrive": ("live.com", "microsoft.com", "onedrive.com"), "one drive": ("live.com", "microsoft.com"),
    "sharepoint": ("sharepoint.com", "microsoft.com"), "hotmail": ("live.com", "hotmail.com"),
    "netflix": ("netflix.com",), "amazon": ("amazon.com",), "prime video": ("amazon.com", "primevideo.com"),
    "apple": ("apple.com", "icloud.com"), "apple id": ("apple.com",), "icloud": ("icloud.com", "apple.com"),
    "itunes": ("apple.com",), "google": ("google.com",), "gmail": ("google.com", "gmail.com"),
    "binance": ("binance.com",), "metamask": ("metamask.io",), "wells fargo": ("wellsfargo.com",),
    "bank of america": ("bankofamerica.com",), "chase": ("chase.com",), "citibank": ("citi.com", "citibank.com"),
    "capital one": ("capitalone.com",), "american express": ("americanexpress.com",), "amex": ("americanexpress.com",),
    "u.s. bank": ("usbank.com",), "us bank": ("usbank.com",), "usaa": ("usaa.com",),
    "navy federal": ("navyfederal.org",), "td bank": ("tdbank.com", "td.com"), "regions": ("regions.com",),
    "huntington": ("huntington.com",), "truist": ("truist.com",), "pnc": ("pnc.com",),
    "hsbc": ("hsbc.com", "hsbc.co.uk"), "barclays": ("barclays.co.uk", "barclays.com"),
    "natwest": ("natwest.com",), "lloyds": ("lloydsbank.com",), "halifax": ("halifax.co.uk",),
    "santander": ("santander.com", "santander.co.uk"), "bbva": ("bbva.com", "bbva.es"),
    "caixabank": ("caixabank.es",), "credit agricole": ("credit-agricole.fr",),
    "la banque postale": ("labanquepostale.fr",), "societe generale": ("societegenerale.fr",),
    "bnp paribas": ("bnpparibas.com", "mabanque.bnpparibas"), "ing": ("ing.com",),
    "rabobank": ("rabobank.nl",), "sparkasse": ("sparkasse.de",), "postbank": ("postbank.de",),
    "deutsche bank": ("deutsche-bank.de", "db.com"), "commonwealth bank": ("commbank.com.au",),
    "westpac": ("westpac.com.au",), "scotiabank": ("scotiabank.com",), "rbc": ("rbcroyalbank.com", "rbc.com"),
    "desjardins": ("desjardins.com",), "interac": ("interac.ca",), "bradesco": ("bradesco.com.br",),
    "itau": ("itau.com.br",), "banco do brasil": ("bb.com.br",), "caixa": ("caixa.gov.br",),
    "icici": ("icicibank.com",), "hdfc bank": ("hdfcbank.com",), "sbi": ("onlinesbi.sbi", "sbi.co.in"),
    "linkedin": ("linkedin.com",), "dropbox": ("dropbox.com",), "docusign": ("docusign.com", "docusign.net"),
    "adobe": ("adobe.com",), "wetransfer": ("wetransfer.com",), "sharefile": ("sharefile.com", "citrix.com"),
    "roblox": ("roblox.com",), "steam": ("steamcommunity.com", "steampowered.com"),
    "spotify": ("spotify.com",), "ebay": ("ebay.com",), "alibaba": ("alibaba.com",), "walmart": ("walmart.com",),
    "rakuten": ("rakuten.co.jp", "rakuten.com"), "mercari": ("mercari.com",),
    "dhl": ("dhl.com", "dhl.de"), "fedex": ("fedex.com",), "usps": ("usps.com",), "ups": ("ups.com",),
    "royal mail": ("royalmail.com",), "la poste": ("laposte.fr",), "chronopost": ("chronopost.fr",),
    "aol": ("aol.com",), "yahoo": ("yahoo.com",), "comcast": ("comcast.net", "xfinity.com"),
    "xfinity": ("xfinity.com", "comcast.net"), "at&t": ("att.com",), "verizon": ("verizon.com",),
    "spectrum": ("spectrum.net", "spectrum.com"), "cox": ("cox.com",), "vodafone": ("vodafone.com",),
    "orange": ("orange.fr", "orange.com"), "telstra": ("telstra.com.au",), "optus": ("optus.com.au",),
    "cogeco": ("cogeco.ca",), "godaddy": ("godaddy.com",), "ionos": ("ionos.com",),
    "webmail": (), "roundcube": (), "zimbra": (),
    "mastercard": ("mastercard.com",), "visa": ("visa.com",), "stripe": ("stripe.com",),
    "square": ("squareup.com",), "venmo": ("venmo.com",), "cash app": ("cash.app",),
    "kucoin": ("kucoin.com",), "kraken": ("kraken.com",), "trezor": ("trezor.io",), "ledger": ("ledger.com",),
    "blockchain.com": ("blockchain.com",), "myetherwallet": ("myetherwallet.com",),
    "trust wallet": ("trustwallet.com",), "phantom": ("phantom.app",), "telegram": ("telegram.org",),
    "discord": ("discord.com",), "twitter": ("twitter.com", "x.com"), "tiktok": ("tiktok.com",),
    "snapchat": ("snapchat.com",), "naver": ("naver.com",), "yandex": ("yandex.ru", "yandex.com"),
    "mail.ru": ("mail.ru",), "vk": ("vk.com",), "irs": ("irs.gov",), "hmrc": ("gov.uk",),
    "gov.uk": ("gov.uk",), "ameli": ("ameli.fr",), "netbank": (),
}
# Generic mail-software / hosting names: not a brand that can be impersonated
# by being on the wrong domain (any company can run Roundcube webmail).
_NOT_A_BRAND = {"webmail", "roundcube", "zimbra", "netbank"}

_CRED_INPUT = re.compile(
    r"<input[^>]+(?:type\s*=\s*[\"']?password"
    r"|(?:name|id|autocomplete)\s*=\s*[\"'][^\"']*(?:cc-?number|card.?number|cvv|cvc|seed|mnemonic|passphrase)"
    r")", re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def page_title(html: str) -> str:
    m = _TITLE.search(html or "")
    return re.sub(r"\s+", " ", unescape(m.group(1))).strip() if m else ""


def has_credential_field(html: str) -> bool:
    return bool(_CRED_INPUT.search(html or ""))


# Sign-in words in the languages phishing kits use most.
_LOGIN_WORDS = re.compile(
    r"sign.?in|log.?in|log on|logon|account|verif|password|secure|wallet|online banking|"
    r"webmail|connexion|connecter|anmelden|einloggen|iniciar|acceso|accedi|entrar|inloggen|"
    r"zaloguj|giriş|ログイン|登录|登入|로그인|вход|войти", re.I)


def _login_like(title: str) -> bool:
    """Login-page titles are short or say "sign in"; news headlines are long sentences."""
    return len(re.findall(r"\w+", title)) <= 6 or bool(_LOGIN_WORDS.search(title))


def _title_brands(title: str) -> list[str]:
    if not _login_like(title):
        return []
    t = " " + re.sub(r"[^a-z0-9.&]+", " ", title.lower()) + " "
    return [b for b in TITLE_BRANDS if b not in _NOT_A_BRAND and f" {b} " in t]


def credential_page(url: str, html: str, official_domains: frozenset[str] = frozenset()) -> tuple[str, str] | None:
    """Return (brand, reason) if a login/payment page names a brand it is not on."""
    if not html or not has_credential_field(html):
        return None
    host = (urlparse(url).hostname or "").lower()
    label, registered = _reg(host)
    if registered in official_domains and not _free_host(host):
        return None
    if _SSO_PATH.search(urlparse(url).path):
        return None     # "sign in with Apple/Google" hand-off routes
    for b in _title_brands(page_title(html)):
        owned = TITLE_BRANDS[b]
        b_key = re.sub(r"[^a-z0-9]", "", b)
        own_labels = {_reg(d)[0] for d in owned}
        if registered in owned:
            continue
        if not _free_host(host) and (b_key in re.sub(r"[^a-z0-9]", "", label) or label in own_labels):
            continue        # page's own domain carries the brand name: its own site
        return b, f"login/payment form, page title names '{b}', but the page is on {registered}"
    return None
