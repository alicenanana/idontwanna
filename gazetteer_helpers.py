import requests
import time
from typing import List, Set


def build_gazetteer(username: str, countries: List[str], max_rows: int = 1000) -> Set[str]:
    """
    Downloads city names from GeoNames for a list of countries.

    Args:
        username (str): Your GeoNames API username.
        countries (List[str]): List of ISO country codes.
        max_rows (int): Max cities per request.

    Returns:
        Set[str]: A deduplicated set of city names in lowercase.
    """
    gazetteer = set()
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
                city_names = [entry["name"].lower() for entry in cities if "name" in entry]
                gazetteer.update(city_names)
                loaded += len(city_names)
                time.sleep(1)  # To respect API rate limits
            print(f"✅ {country_code}: Loaded {loaded} cities.")
        except Exception as e:
            print(f"❌ {country_code}: {e}")

    print(f"📌 Total unique cities gathered: {len(gazetteer)}")
    return gazetteer
