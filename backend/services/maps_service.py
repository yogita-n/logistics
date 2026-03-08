# services/maps_service.py — TomTom version (has live traffic!)
import requests, os

TOMTOM_KEY = os.getenv("TOMTOM_API_KEY")
TOMTOM_BASE = "https://api.tomtom.com/routing/1"

def get_travel_time_minutes(origin: dict, destination: dict) -> int:
    """Single origin → destination with live traffic."""
    url = (f"{TOMTOM_BASE}/calculateRoute/"
           f"{origin['lat']},{origin['lng']}:"
           f"{destination['lat']},{destination['lng']}/json")
    params = {
        "key": TOMTOM_KEY,
        "traffic": "true",           # live traffic included!
        "travelMode": "car",
        "departAt": "now"
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    return int(data["routes"][0]["summary"]["travelTimeInSeconds"] / 60)


def build_time_matrix(locations: list[dict],
                      agent_lat: float = None,
                      agent_lng: float = None) -> list[list[int]]:
    """
    TomTom has no matrix endpoint on free tier,
    so we call routing API for each pair (fine for ≤8 locations).
    """
    all_locs = locations[:]
    if agent_lat and agent_lng:
        all_locs[0] = {"name": "agent", "lat": agent_lat, "lng": agent_lng}

    n = len(all_locs)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_travel_time_minutes(all_locs[i], all_locs[j])

    return matrix
