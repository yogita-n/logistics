# test_solver.py
from dotenv import load_dotenv
load_dotenv()
from services.maps_service import build_time_matrix
from core.solver import solve

locations = [
    {"name": "Depot",                    "lat": 12.9250, "lng": 77.5938},
    {"name": "Jayanagar Restaurant",     "lat": 12.9252, "lng": 77.5940},
    {"name": "National College",         "lat": 12.9305, "lng": 77.5850},
    {"name": "South End Restaurant",     "lat": 12.9180, "lng": 77.5820},
    {"name": "JP Nagar",                 "lat": 12.9102, "lng": 77.5780},
    {"name": "Lalbagh Restaurant",       "lat": 12.9195, "lng": 77.5850},
    {"name": "Jayanagar 3rd Block",      "lat": 12.9260, "lng": 77.5810},
]

pairs    = [(1, 2), (3, 4), (5, 6)]   # (pickup, delivery) for each order
deadlines = {2: 25, 4: 30, 6: 30}    # soft window: deliver by X mins

matrix = build_time_matrix(locations)
route  = solve(locations, pairs, matrix, deadlines)

if route:
    print("\n✅ Optimised Route:")
    for stop in route:
        print(f"  → {stop['name']} (ETA: {stop['arrival_time_mins']} mins)")
else:
    print("❌ No solution found")
