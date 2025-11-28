import requests
import json

url = "http://localhost:8000/analyze"
payload = {
    "url": "https://en.wikipedia.org/wiki/Narendra_Modi"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
