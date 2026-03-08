# test_tomtom.py (run from backend/)
from dotenv import load_dotenv
load_dotenv()
from services.maps_service import get_travel_time_minutes

jayanagar_restaurant = {"lat": 12.9250, "lng": 77.5938}
national_college     = {"lat": 12.9305, "lng": 77.5850}

mins = get_travel_time_minutes(jayanagar_restaurant, national_college)
print(f"Travel time: {mins} mins")   # should print a real number like 7
