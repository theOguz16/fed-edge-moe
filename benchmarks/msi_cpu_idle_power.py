import urllib.request
import json
import time
import statistics

URL = "http://localhost:8085/data.json"
SENSOR_ID = "/intelcpu/0/power/0"

def find_power(node):
    if isinstance(node, dict):
        if node.get("SensorId") == SENSOR_ID:
            v = str(node.get("Value", ""))
            return float(v.replace("W", "").replace(",", ".").strip())

        for child in node.get("Children", []):
            x = find_power(child)
            if x is not None:
                return x

    return None

samples = []

for _ in range(30):
    with urllib.request.urlopen(URL, timeout=2) as r:
        data = json.load(r)

    p = find_power(data)

    if p is not None:
        samples.append(p)

    time.sleep(1)

print(f"Samples     : {len(samples)}")
print(f"IDLE MEDIAN : {statistics.median(samples):.2f} W")
print(f"IDLE MEAN   : {statistics.mean(samples):.2f} W")
