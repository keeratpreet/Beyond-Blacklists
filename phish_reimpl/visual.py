from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np
import torch
import tldextract
from PIL import Image, ImageOps
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torchvision import transforms

from .backbone import KNOWN_MODELS
from .dataset import CaseSample

#normalize the text
def _normalize_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())

#normalize embedding vectors before similarity comp.
def _l2_normalize(vector: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vector, p=2, dim=1)

# resize two images to comparable dimensions
# Uses math.ceil (matching original utils.py) to prevent rounding down to 0.
def _align_resolution(left: Image.Image, right: Image.Image) -> tuple[Image.Image, Image.Image]:
    left_w, left_h = left.size
    right_w, right_h = right.size
    min_w, min_h = min(left_w, right_w), min(left_h, right_h)
    if min_w == 0 or min_h == 0:
        return left, right
    if min_w < min_h:
        return (
            left.resize((int(min_w), math.ceil(left_h * (min_w / left_w)))),
            right.resize((int(min_w), math.ceil(right_h * (min_w / right_w)))),
        )
    return (
        left.resize((math.ceil(left_w * (min_h / left_h)), int(min_h))),
        right.resize((math.ceil(right_w * (min_h / right_h)), int(min_h))),
    )


@dataclass

#output of the visual branch 
class VisualDecision:
    prediction: int
    predicted_brand: str | None
    matched_domain: list[str] | None
    confidence: float | None
    total_time: float


@dataclass

# one known reference logo entry 
class ReferenceEntry:
    brand: str
    domains: list[str]
    file_path: str
    embedding: np.ndarray

# main visual analysis pipeline 
class VisualAnalyzer:
    """Independent visual branch for the workspace.

    It loads the published detector/siamese weights directly and performs the
    detector -> crop -> brand retrieval -> domain consistency flow here.
    """
    
    #setup all the paths and runtime states 
    def __init__(
        self,
        *,
        detector_cfg_path: Path | None = None,
        detector_weights_path: Path | None = None,
        siamese_weights_path: Path | None = None,
        targetlist_dir: Path | None = None,
        domain_map_path: Path | None = None,
        cache_dir: Path | None = None,
        detector_threshold: float = 0.05,
        similarity_threshold: float = 0.87,
        topk_boxes: int = 3,
    ) -> None:
        workspace_root = Path(__file__).resolve().parents[1]
        assets_root = workspace_root / "assets" / "models"

        self.detector_cfg_path = detector_cfg_path or assets_root / "faster_rcnn.yaml"
        self.detector_weights_path = detector_weights_path or assets_root / "rcnn_bet365.pth"
        self.siamese_weights_path = siamese_weights_path or assets_root / "resnetv2_rgb_new.pth.tar"
        self.targetlist_dir = targetlist_dir or assets_root / "expand_targetlist"
        self.domain_map_path = domain_map_path or assets_root / "domain_map.pkl"
        self.cache_dir = cache_dir or workspace_root / "cache"
        self.detector_threshold = detector_threshold
        self.similarity_threshold = similarity_threshold
        self.topk_boxes = topk_boxes

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tld_extractor = tldextract.TLDExtract(
            cache_dir=str(self.cache_dir / "tldextract"),
            suffix_list_urls=None,
        )
        self.predictor = self._load_detector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_siamese_model()
        self.domain_map = self._load_domain_map()
        self.reference_entries = self._load_reference_entries()
        self.reference_matrix = np.array([entry.embedding for entry in self.reference_entries])

    def _load_detector(self) -> DefaultPredictor:
        cfg = get_cfg()
        cfg.merge_from_file(str(self.detector_cfg_path))
        cfg.MODEL.WEIGHTS = str(self.detector_weights_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detector_threshold
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        return DefaultPredictor(cfg)

    def _load_siamese_model(self):
        model = KNOWN_MODELS["BiT-M-R50x1"](head_size=277, zero_head=True)
        weights = torch.load(self.siamese_weights_path, map_location="cpu")
        state_dict = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key.split("module.", 1)[1] if key.startswith("module.") else key] = value
        model.load_state_dict(cleaned)
        model.to(self.device)
        model.eval()
        return model

    #mapping of brands to their official domains 
    def _load_domain_map(self) -> dict[str, list[str]]:
        with self.domain_map_path.open("rb") as handle:
            return pickle.load(handle)

    # convert umg to RGB , resize , normalise , and return a normalised feature vector 
    def _embed_image(self, image: Image.Image) -> np.ndarray:
        # pad to square (preserving aspect ratio) before the fixed 128x128
        # resize, so a wide/tall logo crop isn't stretched out of proportion
        image = image.convert("RGB")
        pad = (
            (max(image.size) - image.size[0]) // 2,
            (max(image.size) - image.size[1]) // 2,
            (max(image.size) - image.size[0]) // 2,
            (max(image.size) - image.size[1]) // 2,
        )
        image = ImageOps.expand(image, pad, fill=(255, 255, 255))
        image = image.resize((128, 128))
        tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )(image)
        with torch.no_grad():
            feature = self.model.features(tensor.unsqueeze(0).to(self.device))
        return _l2_normalize(feature).squeeze(0).cpu().numpy()

    def _resolve_brand_name(self, folder_name: str) -> str | None:
        normalized = _normalize_key(folder_name)
        if folder_name in self.domain_map:
            return folder_name
        for brand in self.domain_map:
            if _normalize_key(brand) == normalized:
                return brand
        return None

    # find usable logo images in folder and then assosciate the brand with the domain
    def _iter_reference_images(self) -> Iterable[tuple[str, list[str], Path]]:
        for brand_dir in sorted(self.targetlist_dir.iterdir()):
            if not brand_dir.is_dir() or brand_dir.name.startswith("."):
                continue
            brand = self._resolve_brand_name(brand_dir.name)
            if brand is None:
                continue
            domains = self.domain_map.get(brand)
            if not domains:
                continue
            for image_path in sorted(brand_dir.iterdir()):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                if image_path.stem.startswith(("loginpage", "homepage")):
                    continue
                yield brand, domains, image_path

    def _load_reference_entries(self) -> list[ReferenceEntry]:
        cache_path = self.cache_dir / "reference_embeddings.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                cached_entries: list[ReferenceEntry] = pickle.load(handle)
            remapped_entries = [self._remap_reference_entry(entry) for entry in cached_entries]
            if any(left.file_path != right.file_path for left, right in zip(cached_entries, remapped_entries)):
                with cache_path.open("wb") as handle:
                    pickle.dump(remapped_entries, handle)
            return remapped_entries

        assets_root = Path(__file__).resolve().parents[1] / "assets" / "models"
        bootstrap_feats = assets_root / "LOGO_FEATS.npy"
        bootstrap_files = assets_root / "LOGO_FILES.npy"
        if bootstrap_feats.exists() and bootstrap_files.exists():
            file_paths = np.load(bootstrap_files, allow_pickle=True)
            embeddings = np.load(bootstrap_feats, allow_pickle=True)
            entries: list[ReferenceEntry] = []
            for file_path, embedding in zip(file_paths, embeddings):
                local_file = assets_root / "expand_targetlist" / Path(str(file_path)).parent.name / Path(str(file_path)).name
                brand = self._resolve_brand_name(local_file.parent.name)
                if brand is None:
                    continue
                domains = self.domain_map.get(brand)
                if not domains:
                    continue
                entries.append(
                    ReferenceEntry(
                        brand=brand,
                        domains=list(domains),
                        file_path=str(local_file),
                        embedding=np.asarray(embedding),
                    )
                )
            with cache_path.open("wb") as handle:
                pickle.dump(entries, handle)
            return entries

        entries: list[ReferenceEntry] = []
        for brand, domains, image_path in self._iter_reference_images():
            try:
                embedding = self._embed_image(Image.open(image_path))
            except OSError:
                continue
            entries.append(
                ReferenceEntry(
                    brand=brand,
                    domains=list(domains),
                    file_path=str(image_path),
                    embedding=embedding,
                )
            )

        with cache_path.open("wb") as handle:
            pickle.dump(entries, handle)
        return entries

#so old absolute file paths in the cache can be repaired when the repo moves.
    def _remap_reference_entry(self, entry: ReferenceEntry) -> ReferenceEntry:
        current_path = Path(entry.file_path)
        if current_path.exists():
            return entry

        remapped_path = self.targetlist_dir / current_path.parent.name / current_path.name
        if remapped_path.exists():
            return ReferenceEntry(
                brand=entry.brand,
                domains=entry.domains,
                file_path=str(remapped_path),
                embedding=entry.embedding,
            )
        return entry

# returns bounding boxes as np arrays 
    def _detect_logo_boxes(self, screenshot_path: Path) -> np.ndarray:
        image = cv2.imread(str(screenshot_path))
        if image is None:
            return np.empty((0, 4), dtype=np.float32)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        outputs = self.predictor(image)
        instances = outputs["instances"]
        logo_boxes = instances.pred_boxes[instances.pred_classes == 1].tensor
        return logo_boxes.detach().cpu().numpy()

    def _match_crop(self, screenshot_path: Path, bbox: np.ndarray) -> tuple[str | None, list[str] | None, float | None]:
        # crop the detected logo region and embed it, then compare against
        # every reference logo at once via a single matrix-vector product
        # (reference_matrix is pre-stacked, one row per reference embedding —
        # since embeddings are L2-normalized, the dot product IS cosine similarity)
        screenshot = Image.open(screenshot_path)
        crop = screenshot.crop(tuple(float(v) for v in bbox))
        crop_embedding = self._embed_image(crop)
        similarities = self.reference_matrix @ crop_embedding
        top_indices = np.argsort(similarities)[::-1][:3]
        if len(top_indices) == 0:
            return None, None, None

        top_entries = [self.reference_entries[idx] for idx in top_indices]
        top_scores = [float(similarities[idx]) for idx in top_indices]
        lead_brand = top_entries[0].brand

        # only consider candidates agreeing with the single best (lead) brand —
        # avoids accepting a weaker match to a different brand just because it
        # narrowly beat the threshold
        for entry, score in zip(top_entries, top_scores):
            if entry.brand != lead_brand:
                continue
            final_score = score
            # borderline case: re-embed both images at matched resolution and
            # re-score, since resizing both crop and reference to the same
            # fixed 128x128 can lose detail that a resolution-matched
            # comparison recovers
            if score < self.similarity_threshold:
                ref_logo = Image.open(entry.file_path)
                aligned_crop, aligned_logo = _align_resolution(crop, ref_logo)
                aligned_crop_embedding = self._embed_image(aligned_crop)
                aligned_logo_embedding = self._embed_image(aligned_logo)
                final_score = float(aligned_logo_embedding.dot(aligned_crop_embedding))
            if final_score >= self.similarity_threshold:
                return entry.brand, entry.domains, final_score
        # no candidate cleared the threshold — return the top score anyway so
        # the caller can report "closest guess" without claiming a match
        return None, None, top_scores[0]
    
    # domain consistency check
    # Mirrors the original check_domain_brand_inconsistency logic:
    # 1. Exact full-domain match (e.g. "amazon.com" in official list) → consistent
    # 2. 2nd-level domain matches + URL suffix is a country TLD → treat as benign country-variant
    # Both conditions replicate what the reference code calls "Benign".
    _COUNTRY_TLDS = frozenset({
        ".af",".ax",".al",".dz",".as",".ad",".ao",".ai",".aq",".ag",".ar",".am",".aw",
        ".ac",".au",".at",".az",".bs",".bh",".bd",".bb",".eus",".by",".be",".bz",".bj",
        ".bm",".bt",".bo",".bq",".an",".nl",".ba",".bw",".bv",".br",".io",".vg",".bn",
        ".bg",".bf",".mm",".bi",".kh",".cm",".ca",".cv",".cat",".ky",".cf",".td",".cl",
        ".cn",".cx",".cc",".co",".km",".cd",".cg",".ck",".cr",".ci",".hr",".cu",".cw",
        ".cy",".cz",".dk",".dj",".dm",".do",".tl",".tp",".ec",".eg",".sv",".gq",".er",
        ".ee",".et",".eu",".fk",".fo",".fm",".fj",".fi",".fr",".gf",".pf",".tf",".ga",
        ".gal",".gm",".ps",".ge",".de",".gh",".gi",".gr",".gl",".gd",".gp",".gu",".gt",
        ".gg",".gn",".gw",".gy",".ht",".hm",".hn",".hk",".hu",".is",".in",".id",".ir",
        ".iq",".ie",".im",".il",".it",".jm",".jp",".je",".jo",".kz",".ke",".ki",".kw",
        ".kg",".la",".lv",".lb",".ls",".lr",".ly",".li",".lt",".lu",".mo",".mk",".mg",
        ".mw",".my",".mv",".ml",".mt",".mh",".mq",".mr",".mu",".yt",".mx",".md",".mc",
        ".mn",".me",".ms",".ma",".mz",".na",".nr",".np",".nl",".nc",".nz",".ni",".ne",
        ".ng",".nu",".nf",".tr",".kp",".mp",".no",".om",".pk",".pw",".ps",".pa",".pg",
        ".py",".pe",".ph",".pn",".pl",".pt",".pr",".qa",".ro",".ru",".rw",".re",".bl",
        ".sh",".kn",".lc",".mf",".pm",".vc",".ws",".sm",".st",".sa",".sn",".rs",".sc",
        ".sl",".sg",".sx",".sk",".si",".sb",".so",".za",".gs",".kr",".ss",".es",".lk",
        ".sd",".sr",".sj",".sz",".se",".ch",".sy",".tw",".tj",".tz",".th",".tg",".tk",
        ".to",".tt",".tn",".tr",".tm",".tc",".tv",".ug",".ua",".ae",".uk",".us",".vi",
        ".uy",".uz",".vu",".va",".ve",".vn",".wf",".eh",".ma",".ye",".zm",".zw",
    })

    def _is_domain_consistent(self, url: str, domains: list[str]) -> bool:
        import tldextract as _tldextract
        from urllib.parse import urlparse

        hostname = (urlparse(url).hostname or "").lower()
        ext = _tldextract.extract(hostname)
        suffix_part = ("." + ext.suffix) if ext.suffix else ""
        domain_part = ext.domain  # 2nd-level label only (e.g. "amazon")
        extracted_domain = domain_part + suffix_part  # e.g. "amazon.com" or "amazon.co.uk"

        # Precompute official domain labels for country-TLD check
        official_domain_parts = []
        for official in domains:
            oext = _tldextract.extract(official.lower())
            official_domain_parts.append(oext.domain)

        for official, official_domain_label in zip(domains, official_domain_parts):
            # 1. Exact full-domain match (mirrors `if extracted_domain in matched_domain`)
            oext = _tldextract.extract(official.lower())
            o_full = oext.domain + ("." + oext.suffix if oext.suffix else "")
            if extracted_domain == o_full:
                return True

        # 2. 2nd-level domain matches AND URL suffix is a recognised country TLD
        #    (mirrors `elif domain_part in matched_domain_parts` + COUNTRY_TLDs check)
        last_suffix_part = "." + suffix_part.split(".")[-1] if suffix_part else ""
        if domain_part and domain_part in official_domain_parts:
            if last_suffix_part in self._COUNTRY_TLDS:
                return True

        return False

    def _registered_domain(self, hostname: str) -> str:
        ext = self.tld_extractor(hostname.lower())
        if not ext.domain:
            return hostname.lower()
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    def _fallback_match(self, screenshot_path: Path) -> tuple[str | None, list[str] | None, float | None]:
        """Fallback for when the detector finds zero logo boxes.

        Tries a set of fixed crops covering the regions where logos most
        commonly appear on phishing pages (top banner strip, top-left corner,
        top-centre). Uses a slightly relaxed threshold because full-region
        crops are noisier than tight detector crops.
        """
        try:
            img = Image.open(screenshot_path)
        except OSError:
            return None, None, None
        w, h = img.size
        if w == 0 or h == 0:
            return None, None, None

        # Slightly lower threshold for fallback crops (more noise expected)
        fallback_threshold = max(0.80, self.similarity_threshold - 0.04)

        # Candidate regions ordered by how likely they are to contain the logo
        regions = [
            (0, 0, w, max(1, h // 5)),                                          # top banner (full width)
            (0, 0, max(1, w // 3), max(1, h // 4)),                             # top-left corner
            (max(1, w // 3), 0, max(2, 2 * w // 3), max(1, h // 4)),            # top-centre
        ]

        best_brand: str | None = None
        best_domains: list[str] | None = None
        best_score: float | None = None

        for box in regions:
            try:
                crop = img.crop(box)
                crop_embedding = self._embed_image(crop)
            except Exception:
                continue
            similarities = self.reference_matrix @ crop_embedding
            top_idx = int(np.argmax(similarities))
            top_score = float(similarities[top_idx])
            if top_score >= fallback_threshold:
                if best_score is None or top_score > best_score:
                    entry = self.reference_entries[top_idx]
                    best_brand = entry.brand
                    best_domains = entry.domains
                    best_score = top_score

        return best_brand, best_domains, best_score

    def analyze(self, case: CaseSample) -> VisualDecision:
        start = perf_counter()
        logo_boxes = self._detect_logo_boxes(case.screenshot_path)
        predicted_brand = None
        matched_domains = None
        confidence = None
        prediction = 0

        for bbox in logo_boxes[: self.topk_boxes]:
            brand, domains, score = self._match_crop(case.screenshot_path, bbox)
            confidence = score
            if not brand or not domains:
                continue
            # Keep the brand even when benign if the page is on an official domain.
            # Only classify as phishing when the domain is inconsistent.
            predicted_brand = brand
            matched_domains = domains
            if not self._is_domain_consistent(case.url, domains):
                prediction = 1
            # We have a confident brand match; stop at the first good match
            # (phish or benign) to avoid later boxes overwriting the result.
            break

        # When the detector found no boxes at all, try strategic fixed crops
        # covering common logo positions (top banner, top-left, top-centre).
        if prediction == 0 and len(logo_boxes) == 0:
            fb_brand, fb_domains, fb_score = self._fallback_match(case.screenshot_path)
            if fb_brand and fb_domains:
                confidence = fb_score
                predicted_brand = fb_brand
                matched_domains = fb_domains
                if not self._is_domain_consistent(case.url, fb_domains):
                    prediction = 1

        return VisualDecision(
            prediction=prediction,
            predicted_brand=predicted_brand,
            matched_domain=matched_domains,
            confidence=confidence,
            total_time=perf_counter() - start,
        )
