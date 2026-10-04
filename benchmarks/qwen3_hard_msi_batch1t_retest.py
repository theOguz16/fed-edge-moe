import json
import re
import statistics
import subprocess
import threading
import urllib.request

EXE = r"tools\llama-cuda\bin\llama-batched-bench.exe"
MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"

REPEATS = 5
POLL_SEC = 2.0
LHM_URL = "http://localhost:8085/data.json"
CPU_POWER_ID = "/intelcpu/0/power/0"

ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)

def find_sensor(node, sensor_id):
    if isinstance(node, dict):
        if node.get("SensorId") == sensor_id:
            return node
        for value in node.values():
            found = find_sensor(value, sensor_id)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_sensor(value, sensor_id)
            if found:
                return found
    return None

def cpu_power_w():
    with urllib.request.urlopen(LHM_URL, timeout=3) as response:
        data = json.load(response)

    sensor = find_sensor(data, CPU_POWER_ID)
    if sensor is None:
        raise RuntimeError("CPU power sensor not found")

    value = str(sensor["Value"])
    return float(value.replace(" W", "").replace(",", ".").strip())

print(f"LHM CPU Package: {cpu_power_w():.2f} W")

results = []

for repeat in range(1, REPEATS + 1):
    cmd = [
        EXE,
        "-hf", MODEL,
        "-ngl", "0",
        "-t", "1",
        "-tb", "1",
        "-npp", "128",
        "-ntg", "64",
        "-npl", "4",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured = []
    powers = []
    stop = threading.Event()
    poll_thread = None

    def poll_power():
        while not stop.is_set():
            try:
                powers.append(cpu_power_w())
            except Exception:
                pass
            stop.wait(POLL_SEC)

    for line in proc.stdout:
        captured.append(line)

        if poll_thread is None and "llama_batched_bench:" in line:
            poll_thread = threading.Thread(
                target=poll_power,
                daemon=True,
            )
            poll_thread.start()

    proc.wait()
    stop.set()

    if poll_thread:
        poll_thread.join(timeout=3)

    matches = ROW_RE.findall("".join(captured))

    if not matches:
        raise RuntimeError("Benchmark output parse failed")

    if not powers:
        raise RuntimeError("No power samples collected")

    row = matches[-1]

    s_tg = float(row[7])
    total_sec = float(row[8])
    mean_power = statistics.mean(powers)

    generated_tokens = 64 * 4
    energy_run = mean_power * total_sec
    energy_token = energy_run / generated_tokens

    results.append({
        "repeat": repeat,
        "s_tg": s_tg,
        "total": total_sec,
        "power": mean_power,
        "energy_run": energy_run,
        "energy_token": energy_token,
    })

    print(
        f"batch 1t R{repeat} | "
        f"{s_tg:7.2f} tok/s | "
        f"{mean_power:6.2f} W | "
        f"{energy_token:.4f} J/token | "
        f"{len(powers)} samples"
    )

stable = results[-3:]

print("\nLAST3 MEDIAN")
print(
    f"batch 1t | "
    f"{statistics.median(x['s_tg'] for x in stable):.2f} tok/s | "
    f"{statistics.median(x['power'] for x in stable):.2f} W | "
    f"{statistics.median(x['energy_run'] for x in stable):.2f} J/run | "
    f"{statistics.median(x['energy_token'] for x in stable):.4f} J/token"
)
