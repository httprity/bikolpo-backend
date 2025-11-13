"""
FastAPI backend with flood-aware routing for Bikolpo
Production-safe version (no ML training, no heavy dependencies)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openrouteservice
import numpy as np
import config

app = FastAPI()

# ORS Client
ORS_API_KEY = config.ORS_API_KEY
client = openrouteservice.Client(key=ORS_API_KEY, timeout=60)

# ML model disabled in production (routing does not require it)
flood_model = None


# ------------------------------
# LOCATION DATABASE
# ------------------------------
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
    'tongi': (90.4032, 23.8978),
    'gazipur': (90.4125, 23.9999),
    'savar': (90.2633, 23.8583),
    'pallabi': (90.3656, 23.8213),
    'kafrul': (90.3839, 23.7817),
    'cantonment': (90.4000, 23.7781),
    'tejgaon': (90.3928, 23.7562),
    'dhaka': (90.4125, 23.8103),
    'chittagong': (91.8317, 22.3569),
    'khulna': (89.5403, 22.8456),
    'rajshahi': (88.5977, 24.3745),
    'sylhet': (91.8697, 24.8949),
}


# ------------------------------
# HELPERS
# ------------------------------
def get_coordinates(place: str):
    place = place.lower().strip()
    if place in DHAKA_LOCATIONS:
        return DHAKA_LOCATIONS[place]

    # Partial match
    for key, coords in DHAKA_LOCATIONS.items():
        if key in place or place in key:
            return coords

    raise ValueError(f"Unknown location: {place}")


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def estimate_flood_risk_for_coords(lat, lon):
    # Dhaka flood zones (simplified)
    if 23.70 <= lat <= 23.74 and 90.38 <= lon <= 90.42:
        return 0.80
    if 23.80 <= lat <= 23.82 and 90.36 <= lon <= 90.38:
        return 0.70
    if 23.88 <= lat <= 23.92 and 90.40 <= lon <= 90.43:
        return 0.75
    if 23.82 <= lat <= 23.87 and 90.38 <= lon <= 90.42:
        return 0.72
    if 23.75 <= lat <= 23.78 and 90.35 <= lon <= 90.40:
        return 0.55

    return 0.25 + np.random.uniform(-0.05, 0.05)


def calculate_route_risk(coords):
    segment_risks = []
    total_risk = total_len = 0

    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]

        mid_lat = (a[1] + b[1]) / 2
        mid_lon = (a[0] + b[0]) / 2
        length_km = haversine_distance(a[1], a[0], b[1], b[0])

        risk = estimate_flood_risk_for_coords(mid_lat, mid_lon)

        segment_risks.append({
            "lat": mid_lat,
            "lon": mid_lon,
            "risk": risk,
            "length_km": length_km
        })

        total_risk += risk * length_km
        total_len += length_km

    avg_risk = total_risk / total_len if total_len > 0 else 0.4
    return avg_risk, segment_risks


# ------------------------------
# ROUTE REQUEST MODEL
# ------------------------------
class RouteRequest(BaseModel):
    source: str
    destination: str
    profile: str = "driving-car"


# ------------------------------
# MAIN ROUTING ENDPOINT
# ------------------------------
@app.post("/predict")
def predict_route(data: RouteRequest):

    try:
        src = get_coordinates(data.source)
        dst = get_coordinates(data.destination)

        # ORS profile
        profile = data.profile if data.profile in ["driving-car", "foot-walking"] else "driving-car"

        # Attempt alternative routes
        response = client.directions(
            coordinates=[src, dst],
            profile=profile,
            format="geojson",
            instructions=False,
            alternative_routes={"target_count": 2}
        )

        features = response["features"][:2]  # take 2 routes

        route_info = []
        routes_geojson = {"type": "FeatureCollection", "features": features}

        for feature in features:
            coords = feature["geometry"]["coordinates"]
            summary = feature["properties"]["summary"]

            distance_km = round(summary["distance"] / 1000, 2)
            duration_minutes = round(summary["duration"] / 60, 1)

            avg_risk, segment_risks = calculate_route_risk(coords)

            route_info.append({
                "distance_km": distance_km,
                "duration_minutes": duration_minutes,
                "risk_value": round(avg_risk, 3),
                "segments": segment_risks
            })

        return {
            "routes": routes_geojson,
            "route_info": route_info,
            "source_coords": src,
            "dest_coords": dst
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
# LOCAL TEST ENTRYPOINT
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
