# app.py — HuggingFace Spaces FastAPI backend for Bikolpo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openrouteservice
import numpy as np
import os

app = FastAPI()

# ORS API key (set in HuggingFace Secrets)
ORS_API_KEY = os.getenv("ORS_API_KEY")
if not ORS_API_KEY:
    raise Exception("ORS_API_KEY is missing. Add it in Space Settings → Variables.")

client = openrouteservice.Client(key=ORS_API_KEY, timeout=60)

# Dhaka locations
DHAKA_LOCATIONS = {
    'motijheel': (90.4173, 23.7334),
    'gulshan': (90.4152, 23.7806),
    'banani': (90.4037, 23.7937),
    'dhanmondi': (90.3742, 23.7461),
    'mohammadpur': (90.3563, 23.7654),
    'mirpur': (90.3687, 23.8223),
    'uttara': (90.3974, 23.8759),
    'badda': (90.4263, 23.7809),
    'rampura': (90.4254, 23.7578),
    'malibagh': (90.4084, 23.7508),
    'khilgaon': (90.4251, 23.7417),
    'jatrabari': (90.4315, 23.7099),
    'sayedabad': (90.4405, 23.7146),
    'kamrangirchar': (90.3659, 23.7167),
    'old dhaka': (90.4070, 23.7104),
    'sadarghat': (90.4084, 23.7090),
    'lalbagh': (90.3879, 23.7180),
    'chawkbazar': (90.4089, 23.7247),
}

# ------------------------------
# Helpers
# ------------------------------

def get_coordinates(place: str):
    place = place.lower().strip()
    if place in DHAKA_LOCATIONS:
        return DHAKA_LOCATIONS[place]

    for key in DHAKA_LOCATIONS:
        if place in key or key in place:
            return DHAKA_LOCATIONS[key]

    raise HTTPException(status_code=404, detail=f"Unknown location: {place}")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def flood_risk(lat, lon):
    if 23.70 <= lat <= 23.74 and 90.38 <= lon <= 90.42:
        return 0.80
    if 23.80 <= lat <= 23.82 and 90.36 <= lon <= 90.38:
        return 0.70
    if 23.75 <= lat <= 23.78 and 90.35 <= lon <= 90.40:
        return 0.55
    return 0.30 + np.random.uniform(-0.05, 0.05)


def route_risk(coords):
    risks = []
    total = length = 0

    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        lat = (a[1] + b[1]) / 2
        lon = (a[0] + b[0]) / 2
        seg_len = haversine(a[1], a[0], b[1], b[0])
        r = flood_risk(lat, lon)

        risks.append({"lat": lat, "lon": lon, "risk": r, "length_km": seg_len})
        total += r * seg_len
        length += seg_len

    avg = total / length if length else 0.4
    return avg, risks

class RouteRequest(BaseModel):
    source: str
    destination: str
    profile: str = "driving-car"


# ------------------------------
# Main API Endpoint
# ------------------------------

@app.post("/predict")
def predict(data: RouteRequest):

    src = get_coordinates(data.source)
    dst = get_coordinates(data.destination)

    profile = data.profile if data.profile in ["driving-car", "foot-walking"] else "driving-car"

    routes = client.directions(
        coordinates=[src, dst],
        profile=profile,
        format="geojson",
        alternative_routes={"target_count": 2},
        instructions=False,
    )

    features = routes["features"][:2]

    route_info = []
    for feature in features:
        summary = feature["properties"]["summary"]

        avg_risk, segments = route_risk(feature["geometry"]["coordinates"])

        route_info.append({
            "distance_km": round(summary["distance"] / 1000, 2),
            "duration_minutes": round(summary["duration"] / 60, 1),
            "risk_value": round(avg_risk, 3),
            "segments": segments
        })

    return {
        "routes": {"type": "FeatureCollection", "features": features},
        "route_info": route_info,
        "source_coords": src,
        "dest_coords": dst
    }
