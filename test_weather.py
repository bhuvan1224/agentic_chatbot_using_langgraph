import os
import requests
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

api_key = os.getenv("WEATHERAPI_KEY")

print(f"Loaded API Key: {api_key}")

if not api_key:
    print("❌ ERROR: WEATHERAPI_KEY was not found in your .env file!")
else:
    # Test request for Anantapur
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q=Anantapur&days=1"
    response = requests.get(url)
    
    print("\nAPI Response Code:", response.status_code)
    print("API Response JSON:", response.json())