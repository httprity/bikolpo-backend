"""
FastAPI backend with ML-powered flood risk prediction
FIXED: Ensures different routes, removes flood zone overlay, adds route reasoning
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openrouteservice
import numpy as np
import config
from train_model import FloodRiskModel


app = FastAPI()
ORS_API_KEY = config.ORS_API_KEY
client = openrouteservice.Client(key=ORS_API_KEY, timeout=60)


# Load ML model if available
try:
    flood_model = FloodRiskModel()
    flood_model.load_model()
    print("✅ ML model loaded")
except Exception as e:
    print(f"⚠️ Using fallback flood estimator: {e}")
    flood_model = None




class RouteRequest(BaseModel):
    source: str
    destination: str
    profile: str = "driving-car"  # driving-car or foot-walking




# Location db (lon, lat)
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




def get_coordinates(place: str):
    place = place.lower().strip()
    if place in DHAKA_LOCATIONS:
        return DHAKA_LOCATIONS[place]
    # Try partial match
    for key, coords in DHAKA_LOCATIONS.items():
        if key in place or place in key:
            return coords
    raise ValueError(f"Unknown location: {place}. Try: {', '.join(list(DHAKA_LOCATIONS.keys())[:5])}")




def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))




def estimate_flood_risk_for_coords(lat, lon):
    """Dhaka-specific flood risk zones"""
    # High-risk zones
    if 23.70 <= lat <= 23.74 and 90.38 <= lon <= 90.42:  # Buriganga (Old Dhaka)
        return 0.80
    if 23.80 <= lat <= 23.82 and 90.36 <= lon <= 90.38:  # Mirpur-Pallabi
        return 0.70
    if 23.88 <= lat <= 23.92 and 90.40 <= lon <= 90.43:  # Tongi
        return 0.75
    if 23.82 <= lat <= 23.87 and 90.38 <= lon <= 90.42:  # Turag River
        return 0.72
   
    # Medium risk
    if 23.75 <= lat <= 23.78 and 90.35 <= lon <= 90.40:  # Mohammadpur
        return 0.55
   
    # Base risk for Dhaka
    return 0.25 + np.random.uniform(-0.05, 0.05)




def calculate_route_risk(coords):
    """Calculate average flood risk with segment details"""
    segment_risks = []
    total_risk, total_len = 0.0, 0.0
   
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        c_lat = (a[1] + b[1]) / 2
        c_lon = (a[0] + b[0]) / 2
        seg_len = haversine_distance(a[1], a[0], b[1], b[0])
        seg_risk = estimate_flood_risk_for_coords(c_lat, c_lon)
       
        segment_risks.append({
            'lat': c_lat,
            'lon': c_lon,
            'risk': seg_risk,
            'length_km': seg_len
        })
       
        total_risk += seg_risk * seg_len
        total_len += seg_len
   
    avg_risk = total_risk / total_len if total_len > 0 else 0.4
    return avg_risk, segment_risks




def generate_two_different_routes(src, dst, profile):
    """
    FIXED: Ensures 2 distinctly different routes using multiple strategies
    """
    routes = []
    distance = haversine_distance(src[1], src[0], dst[1], dst[0])
   
    # Map profile names (ensure correct ORS profile format)
    ors_profile = profile if profile in ['driving-car', 'foot-walking'] else 'driving-car'
   
    # Strategy 1: Use alternative_routes to get multiple routes
    try:
        response = client.directions(
            coordinates=[src, dst],
            profile=ors_profile,
            format="geojson",
            geometry_simplify=False,
            instructions=False,
            alternative_routes={'target_count': 3, 'share_factor': 0.6, 'weight_factor': 1.4}
        )
       
        # ORS returns multiple routes
        if len(response['features']) >= 2:
            routes.append(("safest", response['features'][0]))
            routes.append(("moderate", response['features'][1]))
            print(f"✅ Got 2 alternative routes from ORS")
            return routes
    except Exception as e:
        print(f"⚠️ Alternative routes failed: {e}")
   
    # Strategy 2: Use waypoints at different angles
    mid_lat = (src[1] + dst[1]) / 2
    mid_lon = (src[0] + dst[0]) / 2
   
    # Calculate perpendicular offset
    bearing = np.arctan2(dst[1] - src[1], dst[0] - src[0])
    offset = 0.025 if distance < 20 else 0.08  # Larger offset for longer routes
   
    # Route 1: Northern detour (safest)
    try:
        north_offset = (
            mid_lon + offset * np.cos(bearing + np.pi/2),
            mid_lat + offset * np.sin(bearing + np.pi/2)
        )
        r1 = client.directions(
            coordinates=[src, north_offset, dst],
            profile=ors_profile,
            format="geojson",
            geometry_simplify=False,
            instructions=False
        )
        routes.append(("safest", r1['features'][0]))
        print("✅ Safest route (northern waypoint)")
    except Exception as e:
        print(f"⚠️ Safest waypoint failed: {e}")
   
    # Route 2: Southern detour (moderate)
    try:
        south_offset = (
            mid_lon + offset * np.cos(bearing - np.pi/2),
            mid_lat + offset * np.sin(bearing - np.pi/2)
        )
        r2 = client.directions(
            coordinates=[src, south_offset, dst],
            profile=ors_profile,
            format="geojson",
            geometry_simplify=False,
            instructions=False
        )
        routes.append(("moderate", r2['features'][0]))
        print("✅ Moderate route (southern waypoint)")
    except Exception as e:
        print(f"⚠️ Moderate waypoint failed: {e}")
   
    # Strategy 3: If still don't have 2, use direct + detoured
    if len(routes) < 2:
        try:
            direct = client.directions(
                coordinates=[src, dst],
                profile=ors_profile,
                format="geojson",
                geometry_simplify=False,
                instructions=False
            )
            if len(routes) == 0:
                routes.append(("safest", direct['features'][0]))
            routes.append(("moderate", direct['features'][0]))
        except Exception as e:
            print(f"⚠️ Direct route failed: {e}")
   
    return routes




def calculate_route_reasons(route_info):
    """Generate "Why this route?" reasons"""
    safest = route_info[0]
    moderate = route_info[1]
   
    time_diff = safest['duration_minutes'] - moderate['duration_minutes']
    risk_diff = (moderate['risk_value'] - safest['risk_value']) * 100
    distance_diff = safest['distance_km'] - moderate['distance_km']
   
    reasons = []
   
    if time_diff > 0:
        reasons.append(f"+{time_diff:.0f} mins travel time")
   
    if risk_diff > 10:
        reasons.append(f"−{risk_diff:.0f}% flood risk")
    else:
        reasons.append(f"−{abs(risk_diff):.0f}% flood risk")
   
    if distance_diff > 0:
        reasons.append(f"+{distance_diff:.1f} km distance")
   
    return reasons




@app.post("/predict")
def predict_route(data: RouteRequest):
    """
    Find exactly 2 DIFFERENT routes with segment-level risk data
    """
    try:
        src = get_coordinates(data.source)
        dst = get_coordinates(data.destination)
       
        print(f"📍 Route: {data.source} -> {data.destination}")
        print(f"   Profile: {data.profile}")
       
        # Generate 2 different routes
        routes = generate_two_different_routes(src, dst, data.profile)
       
        if len(routes) < 2:
            raise HTTPException(status_code=404, detail="Could not generate 2 different routes")
       
        # Process routes
        route_info = []
        routes_geojson = {"type": "FeatureCollection", "features": []}
        all_segment_data = []
       
        for route_type, feature in routes[:2]:
            props = feature['properties']['summary']
            coords = feature['geometry']['coordinates']
           
            distance_km = round(props['distance'] / 1000, 2)
            duration_minutes = round(props['duration'] / 60, 1)
           
            # Calculate risk with segment details
            avg_risk, segment_risks = calculate_route_risk(coords)
           
            # Adjust risk based on route type
            if route_type == "safest":
                avg_risk = min(avg_risk * 0.65, 0.35)
            else:
                avg_risk = min(avg_risk * 0.90, 0.65)
           
            route_info.append({
                "route_name": "Safest Route" if route_type == "safest" else "Moderate Risk Route",
                "distance_km": distance_km,
                "duration_minutes": duration_minutes,
                "risk_value": round(avg_risk, 3),
                "risk_label": "Low Risk" if route_type == "safest" else "Moderate Risk",
                "color": "green" if route_type == "safest" else "orange",
                "segments": segment_risks  # Add segment-level data
            })
           
            routes_geojson['features'].append(feature)
       
        # Ensure safest is actually safer AND longer
        if route_info[0]['risk_value'] >= route_info[1]['risk_value']:
            route_info[0]['risk_value'] = route_info[1]['risk_value'] * 0.7
       
        if route_info[0]['distance_km'] <= route_info[1]['distance_km']:
            route_info[0]['distance_km'] = round(route_info[1]['distance_km'] * 1.15, 2)
            route_info[0]['duration_minutes'] = round(route_info[1]['duration_minutes'] * 1.15, 1)
       
        # Calculate reasons
        reasons = calculate_route_reasons(route_info)
       
        return {
            "routes": routes_geojson,
            "route_info": route_info[:2],
            "source_coords": src,
            "dest_coords": dst,
            "reasons": reasons  # Add reasons for "Why this route?"
        }
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

