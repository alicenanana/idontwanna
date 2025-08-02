import requests
import time
from typing import List, Set
import pandas as pd


import requests
import time
from typing import List, Dict
import pandas as pd

import requests
import time
from typing import List, Dict
import pandas as pd

def build_gazetteer(username: str, countries: List[str], max_rows: int = 1000) -> Dict[str, Dict[str, float]]:
    """
    Downloads city names from GeoNames for a list of countries,
    including latitude and longitude for each city.

    Returns:
        Dict[str, Dict[str, float]]: Mapping city name to lat/lon
    """
    gazetteer = {}
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
                    "username": username
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                cities = data.get("geonames", [])
                if not cities:
                    break

                for entry in cities:
                    name = entry.get("name", "").lower()
                    lat = entry.get("lat")
                    lon = entry.get("lng")
                    if name and lat and lon:
                        gazetteer[name] = {
                            "lat": float(lat),
                            "lon": float(lon)
                        }
                        loaded += 1

                time.sleep(1)  # Respect GeoNames API rate limits
            print(f"✅ {country_code}: Loaded {loaded} cities with coordinates.")
        except Exception as e:
            print(f"❌ {country_code}: {e}")

    print(f"📌 Total cities in gazetteer: {len(gazetteer)}")
    return gazetteer





# === Remove overlapping (shorter) entities in same sentence ===
def remove_overlapping_shorter(df: pd.DataFrame) -> pd.DataFrame:
    clean = []
    for sid in df["sentence_id"].unique():
        sent_df = df[df["sentence_id"] == sid].sort_values("start_char")
        to_keep = []
        last_end = -1
        for _, row in sent_df.iterrows():
            if row["start_char"] >= last_end:
                to_keep.append(row)
                last_end = row["end_char"]
        clean.append(pd.DataFrame(to_keep))
    return pd.concat(clean, ignore_index=True)