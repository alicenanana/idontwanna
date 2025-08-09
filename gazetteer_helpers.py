# geoparser/gazetteer_helpers.py
import time
import re
from typing import List, Dict, Set, Tuple, Iterable, Pattern
import requests
import pandas as pd

# Optional (fast path) – only used if you call the FlashText variant
try:
    from flashtext import KeywordProcessor
    _HAS_FLASHTEXT = True
except Exception:
    _HAS_FLASHTEXT = False

# -------------------------------
# GeoNames download
# -------------------------------
def build_gazetteer(username: str, countries: List[str], max_rows: int = 1000) -> Dict[str, Dict[str, float]]:
    """
    Download city names (featureClass=P) from GeoNames for the given countries.
    Returns mapping: lowercased city name -> {
        'lat': float, 'lon': float,
        'country_code': str,  # ISO2 (e.g., 'AR')
        'country': str        # English name (e.g., 'Argentina')
    }
    """
    import time
    import requests

    # Minimal ISO2 -> English country mapping for your list
    COUNTRY_NAME = {
        "AR":"Argentina","CL":"Chile","PE":"Peru","CO":"Colombia","VE":"Venezuela",
        "BO":"Bolivia","EC":"Ecuador","PA":"Panama","CR":"Costa Rica","GT":"Guatemala",
        "MX":"Mexico","CU":"Cuba","BR":"Brazil","GY":"Guyana","PY":"Paraguay",
        "SR":"Suriname","UY":"Uruguay","HN":"Honduras","SV":"El Salvador","NI":"Nicaragua"
    }

    gazetteer: Dict[str, Dict[str, float]] = {}
    print("🌍 Downloading cities from GeoNames...")

    for country_code in countries:
        loaded = 0
        try:
            for start_row in range(0, 5000, max_rows):
                url = "http://api.geonames.org/searchJSON"
                params = {
                    "featureClass": "P",
                    "country": country_code,
                    "maxRows": max_rows,
                    "startRow": start_row,
                    "orderby": "population",
                    "username": username,
                }
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                cities = data.get("geonames", []) or []
                if not cities:
                    break

                for entry in cities:
                    name = (entry.get("name") or "").strip().lower()
                    lat  = entry.get("lat")
                    lon  = entry.get("lng")
                    if not (name and lat and lon):
                        continue
                    gazetteer[name] = {
                        "lat": float(lat),
                        "lon": float(lon),
                        "country_code": country_code,
                        "country": COUNTRY_NAME.get(country_code, country_code),
                    }
                    loaded += 1

                time.sleep(1)  # respect API rate limits
            print(f"✅ {country_code}: Loaded {loaded} cities with coordinates.")
        except Exception as e:
            print(f"❌ {country_code}: {e}")

    print(f"📌 Total cities in gazetteer: {len(gazetteer)}")
    return gazetteer



def gazetteer_names(gaz: Dict[str, Dict[str, float]]) -> Set[str]:
    """Return the set of place names (already lowercased in build_gazetteer)."""
    return set(gaz.keys())

# -------------------------------
# Regex-based matcher (precompiled)
# -------------------------------
_CAPITAL_RE = re.compile(r"[A-ZÁÉÍÓÚÜÑÇÃÕÂÊÔ]")

def _gaz_slice_ok(s: str) -> bool:
    """Accept if title-case or multi-token (safer in narrative prose)."""
    t = s.strip().strip(".,;:!?()[]{}\"'“”‘’")
    if not t:
        return False
    if " " in t:
        return True
    return bool(_CAPITAL_RE.match(t[0]))

def build_gazetteer_patterns(places: Iterable[str], stop_words: Set[str]) -> List[Tuple[str, Pattern[str]]]:
    """
    Build precompiled whole-word regex patterns for places (filtered & sorted).
    Places should be lowercased already; we still lower when matching.
    """
    filtered = [p for p in places if isinstance(p, str) and len(p) > 3 and p.lower() not in stop_words]
    filtered = sorted(set(filtered), key=len, reverse=True)
    return [(p, re.compile(rf"\b{re.escape(p.lower())}\b")) for p in filtered]

def match_gazetteer_precompiled(text: str, patterns: List[Tuple[str, Pattern[str]]]) -> List[Tuple[str, str, int, int]]:
    """
    Use precompiled patterns to find place mentions.
    Returns: (match_text, 'GAZETTEER', start, end)
    """
    if not text:
        return []
    text_lower = text.lower()
    out: List[Tuple[str, str, int, int]] = []
    for place, pat in patterns:
        for m in pat.finditer(text_lower):
            start, end = m.start(), m.end()
            sl = text[start:end].strip(".,;:!?()[]{}0123456789 \"'“”‘’")
            if sl and _gaz_slice_ok(sl):
                out.append((sl, "GAZETTEER", start, end))
    return out

# -------------------------------
# FlashText alternative (faster on huge dicts)
# -------------------------------
def build_keyword_processor(places: Iterable[str], stop_words: Set[str]) -> "KeywordProcessor":
    """
    Build a FlashText KeywordProcessor. Requires flashtext.
    """
    if not _HAS_FLASHTEXT:
        raise ImportError("flashtext not installed. `pip install flashtext`")
    kp = KeywordProcessor(case_sensitive=False)
    for p in places:
        if isinstance(p, str) and len(p) > 3 and p.lower() not in stop_words:
            kp.add_keyword(p)
    return kp

def match_gazetteer_flashtext(text: str, kp: "KeywordProcessor") -> List[Tuple[str, str, int, int]]:
    if not text:
        return []
    hits = []
    for kw, start, end in kp.extract_keywords(text, span_info=True):
        sl = text[start:end].strip(".,;:!?()[]{}0123456789 \"'“”‘’")
        if sl and _gaz_slice_ok(sl):
            hits.append((sl, "GAZETTEER", start, end))
    return hits

# -------------------------------
# Label-aware overlap pruning
# -------------------------------
_LABEL_PRIORITY = {"GPE": 3, "LOC": 2, "FAC": 1, "GAZETTEER": 0}

def remove_overlapping_shorter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each sentence, keep non-overlapping spans, preferring:
    (1) longer spans, then (2) higher label priority (GPE > LOC > FAC > GAZETTEER).
    """
    if df.empty:
        return df
    df = df.sort_values(["sentence_id", "start_char", "end_char"], kind="stable")
    kept_rows = []
    for sid, grp in df.groupby("sentence_id", sort=False):
        taken: List[Dict] = []
        for _, r in grp.iterrows():
            overlaps = []
            for i, t in enumerate(taken):
                if not (r["end_char"] <= t["start_char"] or r["start_char"] >= t["end_char"]):
                    overlaps.append(i)
            if not overlaps:
                taken.append(r); continue
            candidates = [taken[i] for i in overlaps] + [r]
            def _score(x):
                return (x["end_char"] - x["start_char"], _LABEL_PRIORITY.get(x["label"], 0))
            best = max(candidates, key=_score)
            for i in sorted(overlaps, reverse=True):
                taken.pop(i)
            taken.append(best)
        kept_rows.extend(taken)
    out = pd.DataFrame(kept_rows).drop_duplicates(
        subset=["sentence_id", "start_char", "end_char", "label", "entity"], keep="first"
    )
    return out.reset_index(drop=True)
