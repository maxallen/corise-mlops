import requests
import json

requests_file = "requests.json"

all_requests = []
with open(requests_file, "r") as f:
    for line in f:
        all_requests.append(json.loads(line))
    print(f"Collected {len(all_requests)} requests.")

for i, req in enumerate(all_requests):
    requests.post("http://0.0.0.0:8000/predict", json=req)
    if i % 20 == 19:
        print(f"Finished request #{i}")

print("All requests sent")