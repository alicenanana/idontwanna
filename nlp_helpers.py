# nlp_helpers.py

"""
NLP helpers for the BA geoparser pipeline.
Spanish & Portuguese city names (toponyms) friendly; config-driven filtering; notebook-stable APIs.

Exports:
- init_nlp
- get_stopwords
- tag_named_entities
- extract_text_from_pdf
- load_config
- normalize_punctuation
- clean_light
- preprocess_text
- segment_sentences
- clean_heavy
- load_filters
- normalize_diacritics
- apply_ocr_replacements
- is_caption_line
- filter_raw_sentences
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Iterable, List, Dict, Optional, Any, Tuple, Union


try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

import yaml

try:
    import spacy
    from spacy.language import Language
except Exception as e:
    raise RuntimeError("spaCy is required for nlp_helpers.py") from e

# NLTK is optional;
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:
    nltk = None
    nltk_stopwords = None



# Config

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with io.open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def load_filters(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull sentence filtering rules from config, with robust defaults.
    Structure (YAML):
      filters:
        min_len: 20
        max_len: 600
        max_digits_ratio: 0.4
        drop_if_matches: ["^Figure\\s", "^Fig\\.\\s", "^Table\\s"]
    """
    section = (cfg or {}).get("filters", {}) or {}
    min_len = int(section.get("min_len", 20))
    max_len = int(section.get("max_len", 600))
    max_digits_ratio = float(section.get("max_digits_ratio", 0.4))
    pats = section.get("drop_if_matches", []) or []
    compiled = []
    for p in pats:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            # skip broken regex, don’t crash the pipeline
            pass
    return {
        "min_len": min_len,
        "max_len": max_len,
        "max_digits_ratio": max_digits_ratio,
        "drop_if_matches": compiled,
    }



# NLP init (ES/PT aware)


def init_nlp(
    lang: str = "es",
    prefer: Optional[List[str]] = None,
    disable: Optional[List[str]] = None,
    use_gpu: bool = False,
    with_stanza: bool = False,
) -> Tuple["spacy.language.Language", Optional[object]]:
    """
    Initialize spaCy (and optionally Stanza) in a way that's safe for spaCy v3+.

    Args:
        lang: "es" | "pt" | "en" (used for fallbacks)
        prefer: ordered spaCy model names to try first
        disable: spaCy components to disable
        use_gpu: try to put spaCy on GPU (best-effort)
        with_stanza: if True, also build a Stanza pipeline for `lang`

    Returns:
        (nlp, stanza_pipeline) where stanza_pipeline is None if with_stanza=False
    """
    disable = disable or []
    prefer = prefer or []
    stanza_pipeline = None

    # ---GPU for spaCy (best-effort) ---
    if use_gpu:
        try:
            # IMPORTANT: use the module-level import; don't re-import inside the function.
            spacy.require_gpu()
        except Exception:
            # no GPU available; continue on CPU
            pass

    # --- Load spaCy model with reasonable language-specific fallbacks ---
    by_lang_defaults = {
        "es": ["es_core_news_md", "es_core_news_sm", "xx_sent_ud_sm"],
        "pt": ["pt_core_news_md", "pt_core_news_sm", "xx_sent_ud_sm"],
        "en": ["en_core_web_md", "en_core_web_sm"],
    }
    tried = (prefer or []) + by_lang_defaults.get(lang, [])
    nlp = None
    for name in tried:
        try:
            nlp = spacy.load(name, disable=disable)
            break
        except Exception:
            continue
    if nlp is None:
        
        try:
            nlp = spacy.blank(lang if lang in {"en", "es", "pt"} else "en")
        except Exception:
            nlp = spacy.blank("en")

    # --- Ensure sentence segmentation exists (v3+ safe) ---
    if all(p not in nlp.pipe_names for p in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")

    # --- Optional Stanza pipeline (for extra tagging/UD deps) ---
    if with_stanza:
        try:
            import stanza
            try:
                stanza.download(lang, processors="tokenize,ner,pos,lemma,depparse", verbose=False)
            except Exception:
                pass
            stanza_pipeline = stanza.Pipeline(
                lang=lang,
                processors="tokenize,ner,pos,lemma,depparse",
                tokenize_pretokenized=False,
                use_gpu=use_gpu,
                verbose=False,
            )
        except Exception:
            stanza_pipeline = None  # keep going even if stanza isn't available

    return nlp, stanza_pipeline


# =========================
# Stopwords (EN/ES/PT)
# =========================

def get_stopwords(nlp: Language, extra: Optional[Iterable[str]] = None, langs: Optional[List[str]] = None) -> set:
    """
    Union of spaCy defaults + NLTK stopwords for requested languages.
    `langs` defaults to [nlp.lang] plus English for safety in mixed corpora.
    """
    stops = set()
    # spaCy defaults
    try:
        stops |= set(getattr(nlp.Defaults, "stop_words", set()))
    except Exception:
        pass

    # NLTK (best-effort)
    if nltk is not None:
        try:
            if langs is None:
                langs = [nlp.lang, "en"]
            # Map to nltk names
            lang_map = {"en": "english", "es": "spanish", "pt": "portuguese"}
            for lg in langs:
                name = lang_map.get(lg, None)
                if not name:
                    continue
                try:
                    _ = nltk_stopwords.words(name)
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                stops |= set(nltk_stopwords.words(name))
        except Exception:
            pass

    if extra:
        stops |= {s.strip().lower() for s in extra if s and isinstance(s, str)}
    return {s.lower() for s in stops}


# PDF extraction


def extract_text_from_pdf(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for extract_text_from_pdf.")
    doc = fitz.open(pdf_path)
    try:
        n_pages = doc.page_count
        sp = 1 if start_page is None else max(1, int(start_page))
        ep = n_pages if end_page is None else min(n_pages, int(end_page))
        out = []
        for i in range(sp - 1, ep):
            page = doc.load_page(i)
            out.append(page.get_text("text"))
        return "\n".join(out)
    finally:
        doc.close()



# Normalization helpers


_PUNCT_MAP = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‹": "'", "›": "'",
    "‘": "'", "’": "'", "‚": "'", "—": "-", "–": "-", "‑": "-", "‒": "-",
    "\u00A0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ", "\u202F": " ", "\u200A": " ",
    "…": "...",
}

def normalize_punctuation(text: str) -> str:
    if not text:
        return ""
    text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"-\s*\n\s*", "-\n", text)
    text = "\n".join(part.rstrip() for part in text.splitlines())
    return text


_OCR_REPLACEMENTS = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬀ": "ff", "ﬅ": "ft", "ﬆ": "st",
    "’": "'", "¨": '"', "´": "'", "`": "'", "˝": '"', "ˮ": '"',
    "．": ".", "，": ",", "：": ":", "；": ";", "（": "(", "）": ")", "［": "[", "］": "]",
    "–": "-", "—": "-",
}

def apply_ocr_replacements(text: str, replacements: Optional[Dict[str, str]] = None) -> str:
    """
    Replace common OCR ligatures and confusions prior to tokenization.
    If `replacements` dict is provided, it overrides/extends defaults.
    """
    if not text:
        return ""
    rep = dict(_OCR_REPLACEMENTS)
    if replacements:
        rep.update(replacements)
    return "".join(rep.get(ch, ch) for ch in text)



def normalize_diacritics(text: str) -> str:
    """
    Strip diacritics for gazetteer matching. **Don’t** use before NER if
    you want the Spanish/Portuguese models to leverage accents.
    """
    if not text:
        return ""
    decomp = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomp if not unicodedata.combining(c))
    return unicodedata.normalize("NFKC", stripped)



# Cleaning

def clean_light(text: str) -> str:
    if not text:
        return ""
    text = normalize_punctuation(text)
    lines = [ln.strip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if ln == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip()


def clean_heavy(sent: str, nlp: Language, stopset: Optional[set] = None, min_len: int = 2) -> str:
    """
    Heavy normalization for feature extraction:
      - lowercase
      - alphabetic tokens only
      - lemmatize if available
      - remove stopwords
      - drop short tokens
    """
    if not sent:
        return ""
    stopset = stopset or set()
    doc = nlp.make_doc(sent.lower())
    out = []
    # If no lemmatizer in pipeline, fall back to token.text
    for t in doc:
        if not t.is_alpha:
            continue
        lemma = getattr(t, "lemma_", None) or t.text
        lemma = lemma.lower()
        if len(lemma) < min_len or lemma in stopset:
            continue
        out.append(lemma)
    return " ".join(out)



# Sentences & filtering


def segment_sentences(text: str, nlp: Language) -> List[str]:
    if not text:
        return []
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text and s.text.strip()]


_CAPTION_PATTERNS = [
    re.compile(r"^\s*(figure|fig\.)\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*(table|tab\.)\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*(map|plate|chart)\s*\d+", re.IGNORECASE),
]

def is_caption_line(line: str) -> bool:
    if not line:
        return False
    ln = line.strip()
    for pat in _CAPTION_PATTERNS:
        if pat.match(ln):
            return True
    if len(ln) <= 12 and re.match(r"^(fig(?:\.|ure)?|table|map|plate|chart)\s*:?", ln, re.IGNORECASE):
        return True
    if len(ln) < 15 and ":" in ln:
        return True
    return False


def _digits_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(ch.isdigit() for ch in s) / max(1, len(s))


def filter_raw_sentences(
    sentences: Iterable[Union[str, Tuple[Any, str]]],
    rules: Optional[Dict[str, Any]] = None
) -> List[Union[str, Tuple[Any, str]]]:
    """
    Apply length/digits/pattern filters and caption heuristics.
    Accepts either plain strings or (id, text) tuples and preserves the input form.
    """
    rules = rules or {"min_len": 20, "max_len": 600, "max_digits_ratio": 0.4, "drop_if_matches": []}
    min_len = int(rules.get("min_len", 20))
    max_len = int(rules.get("max_len", 600))
    max_digits_ratio = float(rules.get("max_digits_ratio", 0.4))
    patterns: List[re.Pattern] = rules.get("drop_if_matches", []) or []

    kept: List[Union[str, Tuple[Any, str]]] = []

    def _text_of(x):
        return x[1] if isinstance(x, tuple) and len(x) >= 2 else (x or "")

    for s in sentences:
        txt = _text_of(s)
        if not txt:
            continue
        ss = txt.strip()

        # caption-like or section headers
        if is_caption_line(ss):
            continue
        if any(p.search(ss) for p in patterns):
            continue

        # length & numeric constraints
        if len(ss) < min_len or len(ss) > max_len:
            continue
        if _digits_ratio(ss) > max_digits_ratio:
            continue

        kept.append(s)
    return kept



# Preprocessing text


def preprocess_text(text: str, strip_diacritics: bool = False, apply_ocr: bool = True) -> str:
    if not text:
        return ""
    if apply_ocr:
        text = apply_ocr_replacements(text)
    text = normalize_punctuation(text)
    text = clean_light(text)
    if strip_diacritics:
        text = normalize_diacritics(text)
    return text



# NER


def tag_named_entities(text: str, nlp: Language, labels: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    if not text:
        return []
    doc = nlp(text)
    allow = {l.upper() for l in labels} if labels else None
    ents = []
    for span in doc.ents:
        if allow and span.label_.upper() not in allow:
            continue
        ents.append({
            "text": span.text,
            "label": span.label_,
            "start": span.start_char,
            "end": span.end_char,
        })
    return ents

def load_filters(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull sentence filtering rules from config, with robust defaults.
    Structure (YAML):
      filters:
        min_len: 20
        max_len: 600
        max_digits_ratio: 0.4
        drop_if_matches: ["^Figure\\s", "^Fig\\.\\s", "^Table\\s"]
        drop_sections:
          - "^\s*References\b"
          - "^\s*Bibliography\b"
        ocr_replacements:
          ﬁ: fi
          ﬂ: fl
          …: "..."
    """
    section = (cfg or {}).get("filters", {}) or {}
    min_len = int(section.get("min_len", 20))
    max_len = int(section.get("max_len", 600))
    max_digits_ratio = float(section.get("max_digits_ratio", 0.4))

    pats = (section.get("drop_if_matches", []) or []) + (section.get("drop_sections", []) or [])
    compiled = []
    for p in pats:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            pass

    # OCR replacement dict (default to our built-ins)
    ocr_repl = section.get("ocr_replacements", {}) or {}
    # coerce keys/values to 1‑char -> str
    ocr_repl = {str(k): str(v) for k, v in ocr_repl.items()}

    return {
        "min_len": min_len,
        "max_len": max_len,
        "max_digits_ratio": max_digits_ratio,
        "drop_if_matches": compiled,
        "ocr_replacements": ocr_repl,
    }

