
import json
import requests
import pandas as pd
from shapely.geometry import shape

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def download_buildings(lat, lon, radius=500):
    """
    Download OSM building footprints around a coordinate.
    Radius is in metres.
    """

    query = f"""
    [out:json][timeout:25];
    (
      way["building"](around:{radius},{lat},{lon});
      relation["building"](around:{radius},{lat},{lon});
    );
    out center geom;
    """

    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=35
    )

    response.raise_for_status()
    data = response.json()

    buildings = []

    for item in data.get("elements", []):
        if item.get("type") == "way":
            coordinates = [
                (p["lon"], p["lat"])
                for p in item.get("geometry", [])
            ]

            if len(coordinates) >= 4:
                buildings.append({
                    "geometry": coordinates,
                    "lat": item["center"]["lat"],
                    "lon": item["center"]["lon"]
                })

        elif item.get("type") == "relation":
            center = item.get("center")
            if center:
                buildings.append({
                    "geometry": [],
                    "lat": center["lat"],
                    "lon": center["lon"]
                })

    return buildings


def load_geojson(uploaded_file):
    data = json.load(uploaded_file)
    features = data.get("features", [])

    results = []

    for feature in features:
        geom = shape(feature["geometry"])

        results.append({
            "geometry": geom,
            "properties": feature.get("properties", {})
        })

    return results


def load_elevation_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)

    required = {"lat", "lon", "elevation"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV requires lat, lon and elevation columns"
        )

    return df


def nearest_elevation(df, lat, lon):
    if df is None or df.empty:
        return 0.0

    distances = (
        (df["lat"] - lat) ** 2
        + (df["lon"] - lon) ** 2
    )

    index = distances.idxmin()

    return float(df.loc[index, "elevation"])
