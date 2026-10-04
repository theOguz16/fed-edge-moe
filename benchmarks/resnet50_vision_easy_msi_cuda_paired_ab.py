import time
import threading
import statistics
import torch
import pynvml
from torchvision.models import resnet50, ResNet50_Weights

DURATION = 30
POLL_SEC = 0.5

TESTS = [
    ("fp32_light",  torch.float32, 160, 1),
    ("fp16_medium", torch.float16, 224, 4),
]

ORDERS = [
    ["no", "mon"],
    ["mon", "no"],
    ["no", "mon"],
]

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)

def run_once(model, x, batch, monitored):
    powers = []
    utils = []
    temps = []
    stop = False

    def monitor():
        while not stop:
            try:
                powers.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                utils.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                temps.append(
                    pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass
            time.sleep(POLL_SEC)

    with torch.inference_mode():
        for _ in range(20):
            model(x)
    torch.cuda.synchronize()

    t = None
    if monitored:
        t = threading.Thread(target=monitor)
        t.start()

    images = 0
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        while time.perf_counter() - start < DURATION:
            model(x)
            torch.cuda.synchronize()
            images += batch

    elapsed = time.perf_counter() - start

    if monitored:
        stop = True
        t.join()

    tp = images / elapsed

    return {
        "throughput": tp,
        "power": statistics.mean(powers) if powers else None,
        "util": statistics.mean(utils) if utils else None,
        "temp": statistics.mean(temps) if temps else None,
        "j_image": (
            statistics.mean(powers) * elapsed / images
            if powers else None
        ),
    }


for name, dtype, res, batch in TESTS:

    print(f"\n{'='*70}")
    print(name)
    print("="*70)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to("cuda", dtype=dtype).eval()

    x = torch.randn(
        batch, 3, res, res,
        device="cuda",
        dtype=dtype
    )

    no_vals = []
    mon_vals = []
    mon_power = []
    mon_energy = []
    mon_util = []
    mon_temp = []

    for pair, order in enumerate(ORDERS, 1):

        print(f"PAIR {pair}: {' -> '.join(order)}")

        for mode in order:
            r = run_once(
                model, x, batch,
                monitored=(mode == "mon")
            )

            print(
                f"  {mode.upper():3} | "
                f"{r['throughput']:.2f} img/s"
                + (
                    f" | {r['power']:.2f} W"
                    f" | {r['util']:.1f}%"
                    f" | {r['j_image']:.4f} J/img"
                    if mode == "mon"
                    else ""
                )
            )

            if mode == "no":
                no_vals.append(r["throughput"])
            else:
                mon_vals.append(r["throughput"])
                mon_power.append(r["power"])
                mon_energy.append(r["j_image"])
                mon_util.append(r["util"])
                mon_temp.append(r["temp"])

            time.sleep(5)

    no_med = statistics.median(no_vals)
    mon_med = statistics.median(mon_vals)
    delta = (mon_med - no_med) / no_med * 100

    print("\nSUMMARY")
    print(f"NO-MONITOR : {no_med:.2f} img/s")
    print(f"MONITORED  : {mon_med:.2f} img/s")
    print(f"DELTA      : {delta:+.2f}%")
    print(f"POWER      : {statistics.median(mon_power):.2f} W")
    print(f"UTIL       : {statistics.median(mon_util):.1f}%")
    print(f"TEMP       : {statistics.median(mon_temp):.1f} C")
    print(f"ENERGY     : {statistics.median(mon_energy):.4f} J/image")

    del model, x
    torch.cuda.empty_cache()

pynvml.nvmlShutdown()
