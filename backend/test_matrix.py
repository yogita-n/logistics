# test_matrix.py
from dotenv import load_dotenv
load_dotenv()
from services.maps_service import build_time_matrix

locations = [
    {"name": "Depot",               "lat": 12.9250, "lng": 77.5938},
    {"name": "Jayanagar Restaurant","lat": 12.9252, "lng": 77.5940},
    {"name": "National College",    "lat": 12.9305, "lng": 77.5850},
    {"name": "South End Restaurant","lat": 12.9180, "lng": 77.5820},
    {"name": "JP Nagar",            "lat": 12.9102, "lng": 77.5780},
]

matrix = build_time_matrix(locations)
for i, row in enumerate(matrix):
    print(f"From {locations[i]['name']}: {row}")
