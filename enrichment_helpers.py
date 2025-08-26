# geoparser/enrichment_helpers.py
from __future__ import annotations

import re, json, unicodedata, ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd

# ---------------------- Text utils ----------------------
def strip_diacritics(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode("utf-8")

def squash_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def en_to_es_surface(s: str) -> str:
    t = (s or "").lower()
    t = re.sub(r"^lake\s+", "lago ", t)
    t = re.sub(r"^river\s+", "río ", t)
    t = re.sub(r"^(mount|mt\.?)\s+", "cerro ", t)
    return t

def es_to_en_surface(s: str) -> str:
    t = (s or "").lower()
    t = re.sub(r"^lago\s+", "lake ", t)
    t = re.sub(r"^(río|rio)\s+", "river ", t)
    t = re.sub(r"^cerro\s+", "mount ", t)
    return t

ALIASES = {
    "cuzco":"cusco",
    "easter island":"isla de pascua",
    "easter-island":"isla de pascua",
    "rapa nui":"isla de pascua",
}

def normalize_entity_surface(s: str) -> str:
    t = squash_ws(s)
    t = en_to_es_surface(t)
    t = t.lower()
    return ALIASES.get(t, t)

# ---------------------- Gazetteer I/O (Block-7 compatible) ----------------------
def _clean_name(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", s or "")
    s = re.sub(r"[^A-Za-zÀ-ÿ\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

def load_gazetteer_compat(path: Path) -> tuple[list[dict], dict, dict, dict, set]:
    """Supports list-or-dict formats saved by Block 7."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = [{"name": k, **v} for k, v in raw.items()]
    else:
        rows = []

    by_lower, by_stripped, by_clean = {}, {}, {}
    for row in rows:
        nm = (row.get("name") or "").lower()
        meta = {k:v for k,v in row.items() if k != "name"}
        by_lower[nm] = meta
        by_stripped[strip_diacritics(nm)] = meta
        by_clean[_clean_name(nm)] = meta
    return rows, by_lower, by_stripped, by_clean, set(by_lower.keys())

def gaz_lookup_meta(entity_lower: str, by_lower: dict, by_stripped: dict, by_clean: dict) -> Optional[dict]:
    if not entity_lower:
        return None
    return by_lower.get(entity_lower) or by_stripped.get(strip_diacritics(entity_lower)) or by_clean.get(_clean_name(entity_lower))

def gaz_lookup_latlon(entity_lower: str, by_lower: dict, by_stripped: dict, by_clean: dict) -> Tuple[Optional[float], Optional[float]]:
    meta = gaz_lookup_meta(entity_lower, by_lower, by_stripped, by_clean)
    if not meta:
        return None, None
    return meta.get("lat"), meta.get("lon")

# ---------------------- Person list parsing ----------------------
def safe_person_list(x) -> List[str]:
    if isinstance(x, list): return x
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return [] if (x is None or (isinstance(x, float) and pd.isna(x))) else [str(x)]

# ---------------------- Named-object (vehicle/boat/etc.) context ----------------------
VEHICLE_TERMS = {
    "motorcycle","motorbike","bike","bicycle","moto","motocicleta",
    "boat","ship","barco","lancha","car","truck","jeep","bus","camion","train","plane","avion","raft","balsa"
}
def is_named_object_context(nlp, entity: str, sentence: str) -> bool:
    e = (entity or "").lower().strip()
    if not sentence: return False
    doc = nlp(sentence)
    veh_ix = [i for i,t in enumerate(doc) if t.lemma_.lower() in VEHICLE_TERMS or t.text.lower() in VEHICLE_TERMS]
    for i in veh_ix:
        L = max(0, i-6); R = min(len(doc), i+7)
        if any(e and e in t.text.lower() for t in doc[L:R]):
            return True
    return False

# ---------------------- Movement / symbolic cues (EN + ES) ----------------------
MOVEMENT_VERBS = {
    # EN
    "travel","go","arrive","leave","depart","walk","ride","sail","drive","cross","reach","head","return",
    # ES
    "ir","llegar","salir","partir","caminar","andar","montar","navegar","conducir","cruzar","alcanzar","volver"
}
SYMBOLIC_VERBS = {"govern","rule","dominate","represent","symbolize","embody",
                  "gobernar","dominar","representar","simbolizar","encarnar"}

def _window_tokens(tok, radius=10):
    sent = tok.doc
    L, R = max(0, tok.i - radius), min(len(sent), tok.i + radius + 1)
    return sent[L:R]

def movement_verb_present(nlp, sentence: str, entity: str, persons: List[str]) -> bool:
    if not isinstance(sentence, str) or not sentence.strip(): return False
    doc = nlp(sentence); ent_l = (entity or "").lower(); ppl = [p.lower() for p in (persons or [])]
    for tok in doc:
        if tok.lemma_.lower() in MOVEMENT_VERBS:
            win = _window_tokens(tok, radius=10)
            if any(ent_l and ent_l in t.text.lower() for t in win) or any(any(p in t.text.lower() for p in ppl) for t in win):
                return True
    return False

def symbolic_context(nlp, sentence: str, entity: str, persons: List[str]) -> bool:
    if not isinstance(sentence, str) or not sentence.strip(): return False
    doc = nlp(sentence); ent_l = (entity or "").lower(); ppl = [p.lower() for p in (persons or [])]
    for tok in doc:
        if tok.lemma_.lower() in SYMBOLIC_VERBS:
            win = _window_tokens(tok, radius=10)
            if any(ent_l and ent_l in t.text.lower() for t in win) or any(any(p in t.text.lower() for p in ppl) for t in win):
                return True
    return False

# ---------------------- Metonymy (EN + ES cue nouns, wider window) ----------------------
CUE_WORDS_METONYMY = {
    # EN
    "government","policy","military","regime","parliament","industry","media","press","power","army","forces",
    # ES
    "gobierno","política","militar","régimen","regimen","parlamento","industria","medios","prensa",
    "poder","ejército","ejercito","fuerzas","fuerzas armadas"
}

def metonymy_flag(nlp, entity_text: str, sentence: str, window: int = 10) -> bool:
    if not isinstance(sentence, str) or not sentence.strip():
        return False
    doc = nlp(sentence)
    ent_l = (entity_text or "").lower()
    ent_idxs = [i for i,t in enumerate(doc) if ent_l and ent_l in t.text.lower()]
    if not ent_idxs:
        return False
    for i,tok in enumerate(doc):
        if tok.text.lower() in CUE_WORDS_METONYMY and tok.pos_ in {"NOUN","PROPN"}:
            if any(abs(i - j) <= window for j in ent_idxs):
                return True
    return False

# ---------------------- Transport (WordNet-assisted recall + regex + context backfill) ----------------------
import re as _re
try:
    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.ensure_loaded()
    except Exception:
        try:
            nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
            wn.ensure_loaded()
        except Exception:
            pass
    _WN_OK = True if getattr(wn, "synsets", None) else False
except Exception:
    _WN_OK = False

TRANSPORT_NOUNS = {
    # EN
    "motorcycle":"motorcycle","motorbike":"motorcycle","bike":"motorcycle","moped":"motorcycle",
    "bicycle":"bicycle","cycle":"bicycle",
    "car":"car","jeep":"car","van":"car","taxi":"car","cab":"car",
    "truck":"truck","lorry":"truck","pickup":"truck",
    "bus":"bus","coach":"bus","minibus":"bus",
    "train":"train","railway":"train",
    "boat":"boat","ship":"boat","ferry":"boat","canoe":"boat","launch":"boat","steamer":"boat","vessel":"boat",
    "raft":"raft",
    "plane":"plane","airplane":"plane","aeroplane":"plane","aircraft":"plane",
    "horse":"horse","mule":"horse","donkey":"horse",
    "foot":"foot",
    # ES
    "moto":"motorcycle","motocicleta":"motorcycle",
    "bicicleta":"bicycle",
    "auto":"car","coche":"car",
    "camion":"truck","camión":"truck","camioneta":"truck",
    "bus":"bus","ómnibus":"bus","omnibus":"bus","autobús":"bus","autobus":"bus","colectivo":"bus",
    "tren":"train",
    "balsa":"raft","bote":"boat","barco":"boat","lancha":"boat","ferry":"boat",
    "avion":"plane","avión":"plane",
    "caballo":"horse","mula":"horse","burro":"horse",
    "a pie":"foot","pie":"foot"
}

def _wn_lemmas_from(synset_ids) -> set:
    out = set()
    if not _WN_OK: return out
    for sid in synset_ids:
        try:
            root = wn.synset(sid)
        except Exception:
            continue
        for ss in root.closure(lambda s: s.hyponyms()):
            for l in ss.lemmas():
                w = l.name().replace("_", "-").lower()
                if _re.fullmatch(r"[a-z-]{3,}", w):
                    out.add(w)
    return out

WN_NOUNS_BY_MODE = {}
if _WN_OK:
    WN_NOUNS_BY_MODE = {
        "motorcycle": _wn_lemmas_from(["motorcycle.n.01","motorbike.n.01","moped.n.01"]),
        "bicycle":    _wn_lemmas_from(["bicycle.n.01","cycle.n.01"]),
        "car":        _wn_lemmas_from(["car.n.01","automobile.n.01","auto.n.01","van.n.05","jeep.n.01","taxi.n.01","cab.n.03"]),
        "truck":      _wn_lemmas_from(["truck.n.01","lorry.n.01","pickup.n.02"]),
        "bus":        _wn_lemmas_from(["bus.n.01","coach.n.04","minibus.n.01"]),
        "train":      _wn_lemmas_from(["train.n.01"]),
        "boat":       _wn_lemmas_from(["boat.n.01","ship.n.01","watercraft.n.01","canoe.n.01","launch.n.06","steamer.n.01","vessel.n.02"]),
        "raft":       _wn_lemmas_from(["raft.n.01"]),
        "plane":      _wn_lemmas_from(["airplane.n.01","aircraft.n.01"]),
        "horse":      _wn_lemmas_from(["horse.n.01","mule.n.01","donkey.n.01"]),
    }
    # de-ambiguate lemmas that appear in more than one mode
    lemma2modes = {}
    for mode, lex in WN_NOUNS_BY_MODE.items():
        for w in lex:
            lemma2modes.setdefault(w, set()).add(mode)
    for w, modes in list(lemma2modes.items()):
        if len(modes) > 1:
            for m in modes:
                WN_NOUNS_BY_MODE[m].discard(w)

WN_VERB_TO_MODE = {
    "walk":"foot","hike":"foot","trek":"foot","march":"foot","stroll":"foot",
    "drive":"car","motor":"car",
    "ride":"motorcycle",  # may flip to horse if horse noun nearby
    "sail":"boat","row":"boat","paddle":"boat",
    # verbs like take/catch/board require a noun nearby
}

_TR = r"(?:the\s+)?(?:el\s+|la\s+|los\s+|las\s+)?"
PATTERNS = [
    (_re.compile(rf"\b(?:by|on|in|aboard|via)\s+{_TR}(motorcycle|motorbike|bike|bicycle|bus|train|truck|car|boat|raft|ship|ferry|plane|taxi|cab)\b", _re.I), "noun"),
    (_re.compile(rf"\b(?:en|a|al)\s+{_TR}(moto|motocicleta|bicicleta|bus|ómnibus|omnibus|autobús|autobus|tren|camioneta|camion|camión|balsa|bote|barco|lancha|ferry|avion|avión|auto|coche|caballo|mula|burro)\b", _re.I), "noun"),
    (_re.compile(rf"\b(took|caught|boarded|rode)\s+{_TR}(bus|train|boat|ship|ferry|plane|motorcycle|motorbike|bike|bicycle|car|truck|taxi|cab)\b", _re.I), "noun"),
    (_re.compile(rf"\b(tomar(?:on|on?s|on?emos|é|ía)?|cog(?:er|imos|í)|sub(?:ir|imos|ió|imos))\s+(?:en|a|al)?\s*{_TR}(bus|ómnibus|omnibus|autobús|autobus|tren|balsa|bote|barco|lancha|ferry|avion|avión|moto|motocicleta|auto|coche|camion|camión|camioneta)\b", _re.I), "noun"),
    (_re.compile(r"\b(on\s+foot|a\s+pie)\b", _re.I), "foot"),
    (_re.compile(r"\b(rode|mont(?:ar|amos|ó|é|ábamos)?)\b.*?\b(caballo|mula|burro|horse|mule|donkey|moto|motocicleta|motorcycle|motorbike)\b", _re.I), "rode"),
    (_re.compile(r"\b(walk(?:ed|ing)?|hike(?:d|s|ing)?|trekk?(?:ed|ing)?|stroll(?:ed|ing)?)\b", _re.I), "foot"),
    (_re.compile(r"\b(drove|drive|driving|conduc(?:ir|imos|ía|iste|í)|manej(?:ar|amos|ó|aba))\b", _re.I), "drive"),
    (_re.compile(r"\b(sail(?:ed|ing)?|row(?:ed|ing)?|paddl(?:ed|ing)?)\b", _re.I), "boat"),
]

def _normalize_transport_token(tok: str) -> Optional[str]:
    t = (tok or "").strip().lower()
    if t in TRANSPORT_NOUNS:
        return TRANSPORT_NOUNS[t]
    if _WN_OK:
        for mode, lex in WN_NOUNS_BY_MODE.items():
            if t in lex:
                return mode
    return None

def extract_transport_from_text(text: str, enable_wordnet: bool = True) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    spans = []
    # Regex spans
    for rx, kind in PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        if kind == "noun":
            mode = _normalize_transport_token(m.group(m.lastindex).lower())
            if mode: spans.append(mode)
        elif kind == "foot":
            spans.append("foot")
        elif kind == "rode":
            noun = m.group(m.lastindex)
            mode = _normalize_transport_token(noun)
            if mode:
                spans.append(mode)
            elif _re.search(r"\b(motorcycle|motorbike|moto|motocicleta)\b", text, flags=_re.I):
                spans.append("motorcycle")
        elif kind == "drive":
            if _re.search(r"\b(motorcycle|motorbike|moto|motocicleta)\b", text, flags=_re.I):
                spans.append("motorcycle")
            else:
                spans.append("car")
        elif kind == "boat":
            spans.append("raft" if _re.search(r"\b(raft|balsa)\b", text, flags=_re.I) else "boat")

    # WordNet sweeps (if still empty)
    if not spans and enable_wordnet and _WN_OK:
        toks = _re.findall(r"[A-Za-zÀ-ÿ-]+", text.lower())
        for w in toks:
            mode = _normalize_transport_token(w)
            if mode:
                spans.append(mode)
        for w in toks:
            mode = WN_VERB_TO_MODE.get(w)
            if mode:
                spans.append(mode)

    if spans:
        uniq = []
        for m in spans:
            if m and m not in uniq:
                uniq.append(m)
        return ", ".join(uniq)
    return None

def build_sid_to_transport(df: pd.DataFrame) -> Dict[int, Optional[str]]:
    out = {}
    for r in df.itertuples(index=False):
        try:
            sid = int(r.sentence_id)
        except Exception:
            continue
        v = getattr(r, "transport", None)
        out[sid] = v if isinstance(v, str) and v.strip() else None
    return out

def context_backfill(sid_to_transport: dict, sid: int) -> Optional[str]:
    if sid is None: return None
    if not isinstance(sid, int): return None
    if not sid_to_transport: return None
    for radius in (1, 2):
        for delta in (-radius, radius):
            s2 = sid + delta
            if s2 in sid_to_transport and sid_to_transport.get(s2):
                return sid_to_transport[s2]
    return None

def infer_transport_for_row(row: dict, sid_to_transport: dict, enable_wordnet: bool = True) -> Optional[str]:
    # keep existing
    cur = row.get("transport")
    if isinstance(cur, str) and cur.strip():
        return cur
    # regex/wordnet on sentence
    tx = extract_transport_from_text(row.get("sentence",""), enable_wordnet=enable_wordnet)
    if tx:
        return tx
    # context backfill only if motion cue present
    try:
        sid = int(row.get("sentence_id"))
    except Exception:
        sid = None
    if row.get("movement_verb_present") and sid is not None:
        ctx = context_backfill(sid_to_transport, sid)
        if ctx:
            return ctx
    return None

# ---------------------- Year extraction fallback ----------------------
YEAR_RX = re.compile(r"\b(19\d{2}|20\d{2})\b")
def extract_year_regex(sentence: str) -> Optional[str]:
    if not isinstance(sentence, str): return None
    m = YEAR_RX.search(sentence)
    return m.group(0) if m else None

# ---------------------- Final label ----------------------
def final_label_decision(row: dict) -> str:
    if row.get("named_object_flag"):
        return "NOISE"
    if row.get("symbolic_context") or (row.get("movement_verb_present") and row.get("metonymy_flagged")):
        return "SYMBOLIC"
    has_geo = bool(row.get("country_valid")) or pd.notna(row.get("lat")) or pd.notna(row.get("lon"))
    if has_geo and not row.get("metonymy_flagged"):
        return "LITERAL"
    return "NOISE"

# ---------------------- Column ordering ----------------------
def reorder_columns(df: pd.DataFrame, preferred: List[str]) -> pd.DataFrame:
    return df[[c for c in preferred if c in df.columns]]

