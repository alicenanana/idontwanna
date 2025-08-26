# === geoparser/gazetteer_helpers.py ===
import re
from typing import List, Dict, Set, Tuple, Iterable, Pattern
import time
import requests
import pandas as pd


try:
    from flashtext import KeywordProcessor
    _HAS_FLASHTEXT = True
except Exception:
    _HAS_FLASHTEXT = False


from requests.exceptions import SSLError, RequestException

def build_gazetteer(
    username: str,
    countries: List[str],
    max_rows: int = 1000,
    *,
    host: str = "api.geonames.org",
    https: bool = False,
    page_size: int = 1000,
    timeout: int = 20,
    retries: int = 4,
    backoff_base: float = 1.5,
    sleep_between: float = 0.6,
) -> Dict[str, Dict]:
    """
    Download city names (featureClass=P) from GeoNames for the given countries.
    Respects YAML network knobs and falls back to HTTP if HTTPS cert fails.

    Returns mapping: name_lower -> {
        'lat', 'lon', 'country_code', 'country', 'population',
        'feature_class'='P', 'feature_code' (e.g. PPL)
    }
    Tie-break: keep the candidate with the largest 'population' per name.
    """
    COUNTRY_NAME = {
        "AR":"Argentina","CL":"Chile","PE":"Peru","CO":"Colombia","VE":"Venezuela",
        "BO":"Bolivia","EC":"Ecuador","PA":"Panama","CR":"Costa Rica","GT":"Guatemala",
        "MX":"Mexico","CU":"Cuba","BR":"Brazil","GY":"Guyana","PY":"Paraguay",
        "SR":"Suriname","UY":"Uruguay","HN":"Honduras","SV":"El Salvador","NI":"Nicaragua",
        "DO":"Dominican Republic","HT":"Haiti"
    }

    # de-dupe while preserving order
    countries = list(dict.fromkeys(countries))

    # initial scheme from YAML; may switch to http on SSL fail
    scheme = "https" if https else "http"
    base_url = f"{scheme}://{host}/searchJSON"

    gazetteer: Dict[str, Dict] = {}
    print("Downloading cities from GeoNames...")

    for cc in countries:
        loaded_cc = 0
        start_row = 0

        while True:
            if loaded_cc >= max_rows:
                break

            rows_to_fetch = min(page_size, max_rows - loaded_cc)
            params = {
                "featureClass": "P",       # populated places
                "country": cc,
                "maxRows": rows_to_fetch,
                "startRow": start_row,
                "orderby": "population",
                "username": username,
            }

            attempt = 0
            geos = None
            while attempt <= retries:
                try:
                    r = requests.get(base_url, params=params, timeout=timeout)
                    r.raise_for_status()
                    data = r.json()

                    # GeoNames sometimes returns {"status": {"message": "..."}}
                    st = data.get("status")
                    if isinstance(st, dict) and st.get("message"):
                        raise RuntimeError(f"GeoNames error: {st.get('message')}")

                    geos = data.get("geonames", []) or []
                    break  # success
                except SSLError as e:
                    # fallback once to HTTP if HTTPS was requested
                    if scheme == "https":
                        print(f"{cc}: SSL error on HTTPS; falling back to HTTP…")
                        scheme = "http"
                        base_url = f"{scheme}://{host}/searchJSON"
                        # retry immediately with HTTP (same attempt count)
                        continue
                    # already on HTTP → treat as normal failure
                    err = e
                except RequestException as e:
                    err = e
                except Exception as e:
                    err = e

                # retry with backoff or give up
                if attempt == retries:
                    print(f"{cc}: giving up page startRow={start_row}: {err}")
                    geos = []
                    break
                sleep_s = backoff_base ** attempt
                time.sleep(sleep_s)
                attempt += 1

            if not geos:
                break

            for entry in geos:
                name = (entry.get("name") or "").strip().lower()
                lat  = entry.get("lat")
                lon  = entry.get("lng")
                if not (name and lat and lon):
                    continue

                try:
                    pop = int(entry.get("population") or 0)
                except Exception:
                    pop = 0

                rec = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "country_code": cc,
                    "country": COUNTRY_NAME.get(cc, cc),
                    "population": pop,
                    "feature_class": "P",
                    "feature_code": entry.get("fcode"),
                }

                # keep the most-populous candidate for this name
                if (name not in gazetteer) or (pop > int(gazetteer[name].get("population", -1))):
                    gazetteer[name] = rec

                loaded_cc += 1

            start_row += len(geos)
            time.sleep(sleep_between)

        print(f"{cc}: Loaded {loaded_cc} cities (kept most-populous per name).")

    print(f"Total unique names in gazetteer: {len(gazetteer)}")
    return gazetteer


def build_gazetteer_from_conf(gconf: dict) -> Dict[str, Dict]:
    """
    Convenience wrapper if you want to pass the whole YAML sub-dict.
    Example: build_gazetteer_from_conf(config['gazetteer'])
    """
    return build_gazetteer(
        username=gconf["username"],
        countries=gconf["countries"],
        max_rows=gconf.get("max_rows", 1000),
        host=gconf.get("host", "api.geonames.org"),
        https=bool(gconf.get("https", False)),
        page_size=gconf.get("page_size", 1000),
        timeout=gconf.get("timeout", 20),
        retries=gconf.get("retries", 4),
        backoff_base=gconf.get("backoff_base", 1.5),
        sleep_between=gconf.get("sleep_between", 0.6),
    )
def gazetteer_names(gaz: Dict[str, Dict[str, float]]) -> Set[str]:
    """Return the set of place names (lowercased)."""
    return set(gaz.keys())


# Regex-based matcher (precompiled; whole-word, case-insensitive)

_CAPITAL_RE = re.compile(r"[A-ZÁÉÍÓÚÜÑÇÃÕÂÊÔ]")

def _gaz_slice_ok(s: str) -> bool:
    """Accept if title-case or multi-token (safer in narrative prose)."""
    t = s.strip().strip(".,;:!?()[]{}\"'“”‘’")
    if not t:
        return False
    if " " in t:
        return True
    # Single-token: keep if looks capitalized in original string
    return bool(_CAPITAL_RE.match(t[0]))

def build_gazetteer_patterns(places: Iterable[str], stop_words: Set[str]) -> List[Tuple[str, Pattern[str]]]:
    """
    Build precompiled whole-word regex patterns for places.
    Places should be lowercased already; we still lowercase the haystack for matching.
    """
    filtered = [p for p in places if isinstance(p, str) and len(p) > 3 and p.lower() not in stop_words]
    filtered = sorted(set(filtered), key=len, reverse=True)  # longest first
    return [(p, re.compile(rf"\b{re.escape(p.lower())}\b")) for p in filtered]

def match_gazetteer_precompiled(text: str, patterns: List[Tuple[str, Pattern[str]]]) -> List[Tuple[str, str, int, int]]:
    """
    Use precompiled patterns to find place mentions.
    Returns: (match_text, 'GAZETTEER', start, end) with rough title-case guard.
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


# FlashText alternative (optional)

def build_keyword_processor(places: Iterable[str], stop_words: Set[str]) -> "KeywordProcessor":
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


# Label-aware overlap pruning

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
