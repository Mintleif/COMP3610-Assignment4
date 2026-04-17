import requests
import json

url = "http://localhost:8000/predict"

# Base payload matching your exact data
payload = {
    "pickup_hour": 14,
    "pickup_day_of_week": 3,
    "is_weekend": False,
    "trip_duration_minutes": 15.5,
    "trip_speed_mph": 20.0,
    "log_trip_distance": 1.6,
    "fare_per_mile": 2.5,
    "fare_per_minute": 0.8,
    "PU_Borough": "Manhattan",
    "DO_Borough": "Brooklyn",
    "passenger_count": 2,
    "trip_distance": 5.0,
    "fare_amount": 15.0,
    "total_amount": 18.0
}

# Loop 3 times to satisfy the Task 3.2 Rubric requirement
for i in range(1, 4):
    print(f"--- Sending Request {i} ---")
    
    # Tweak the trip distance slightly each loop just to show variance
    payload["trip_distance"] += 1.5 
    
    response = requests.post(url, json=payload)
    
    # Print the formatted JSON response
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")