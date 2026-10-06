import os
import threading
import time

import psutil


MB = 1024 * 1024


def _process_tree_rss_bytes(root_process):
    """
    RSS of the current process plus all currently alive child processes.
    Useful when a benchmark spawns helper/subprocess workers.
    """
    total = 0

    processes = [root_process]

    try:
        processes.extend(root_process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    seen = set()

    for proc in processes:
        if proc.pid in seen:
            continue

        seen.add(proc.pid)

        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return total


class MemorySampler:
    """
    Lightweight process/system memory sampler.

    Intended usage:
        sampler = MemorySampler(interval_sec=0.2)
        sampler.start()

        # timed benchmark workload

        metrics = sampler.stop()

    Important:
    - process RSS includes the Python process and live child processes.
    - system RAM is OS-wide memory state.
    - accelerator memory is NOT included here as a separate additive metric.
    """

    def __init__(self, interval_sec=0.2, include_children=True):
        self.interval_sec = float(interval_sec)
        self.include_children = bool(include_children)
        self.process = psutil.Process(os.getpid())

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._samples = 0

        self.process_rss_pre = None
        self.process_rss_peak = 0

        self.system_ram_used_pre = None
        self.system_ram_used_peak = 0

        self.system_ram_available_pre = None
        self.system_ram_available_min = None

    def _read_sample(self):
        if self.include_children:
            rss = _process_tree_rss_bytes(self.process)
        else:
            rss = self.process.memory_info().rss

        vm = psutil.virtual_memory()

        # Prefer total - available rather than psutil's platform-specific
        # "used" interpretation.
        system_used = vm.total - vm.available
        system_available = vm.available

        return rss, system_used, system_available

    def _record_sample(self):
        rss, system_used, system_available = self._read_sample()

        with self._lock:
            self._samples += 1

            self.process_rss_peak = max(
                self.process_rss_peak,
                rss,
            )

            self.system_ram_used_peak = max(
                self.system_ram_used_peak,
                system_used,
            )

            if self.system_ram_available_min is None:
                self.system_ram_available_min = system_available
            else:
                self.system_ram_available_min = min(
                    self.system_ram_available_min,
                    system_available,
                )

    def _loop(self):
        while not self._stop_event.wait(self.interval_sec):
            self._record_sample()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("MemorySampler already started")

        rss, system_used, system_available = self._read_sample()

        self.process_rss_pre = rss
        self.process_rss_peak = rss

        self.system_ram_used_pre = system_used
        self.system_ram_used_peak = system_used

        self.system_ram_available_pre = system_available
        self.system_ram_available_min = system_available

        self._samples = 1

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._loop,
            name="memory-sampler",
            daemon=True,
        )

        self._thread.start()

    def stop(self):
        if self._thread is None:
            raise RuntimeError("MemorySampler was not started")

        self._stop_event.set()
        self._thread.join()

        # Capture final state as well.
        self._record_sample()

        metrics = self.metrics()

        self._thread = None

        return metrics

    def metrics(self):
        with self._lock:
            return {
                "process_rss_pre_mb":
                    self.process_rss_pre / MB,

                "process_rss_peak_mb":
                    self.process_rss_peak / MB,

                "process_rss_delta_mb":
                    (
                        self.process_rss_peak
                        - self.process_rss_pre
                    ) / MB,

                "system_ram_used_pre_mb":
                    self.system_ram_used_pre / MB,

                "system_ram_used_peak_mb":
                    self.system_ram_used_peak / MB,

                "system_ram_available_pre_mb":
                    self.system_ram_available_pre / MB,

                "system_ram_available_min_mb":
                    self.system_ram_available_min / MB,

                "memory_sample_count":
                    self._samples,

                "memory_sample_interval_sec":
                    self.interval_sec,
            }


if __name__ == "__main__":
    sampler = MemorySampler(interval_sec=0.2)

    sampler.start()

    # Small self-test allocation.
    x = bytearray(128 * MB)
    time.sleep(1.0)

    metrics = sampler.stop()

    del x

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:32}: {value:.2f}")
        else:
            print(f"{key:32}: {value}")



def memory_snapshot(include_children=False):
    """
    Instantaneous process/system memory snapshot.

    Accelerator memory is intentionally excluded because
    process RSS and accelerator allocation may overlap,
    especially on unified-memory systems such as Apple Silicon.
    """
    process = psutil.Process(os.getpid())

    if include_children:
        rss = _process_tree_rss_bytes(process)
    else:
        rss = process.memory_info().rss

    vm = psutil.virtual_memory()

    return {
        "process_rss_mb": rss / MB,
        "system_ram_used_mb": (vm.total - vm.available) / MB,
        "system_ram_available_mb": vm.available / MB,
    }
