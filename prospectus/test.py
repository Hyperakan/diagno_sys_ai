import requests
import json
import sys

try:
    response = requests.post(
        "http://localhost:8000/analyze",
        json={
            "current prospectuses": ["aspirin.txt", "paracetamol.txt"],
            "new prospectus": "katarin.txt"
        }
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except requests.RequestException as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)