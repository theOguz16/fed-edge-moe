#!/bin/bash
set -e

source .venv/bin/activate
mkdir -p results
sudo -v

for threads in 1 2 4 6 8 10; do
  for r in 1 2 3; do

    echo "================================"
    echo "${threads} threads repeat ${r}"
    echo "================================"

    rm -f /tmp/qwen_cpu_ready /tmp/qwen_cpu_go

    RUN="results/qwen25_medium_mac_cpu_${threads}t_r${r}.txt"
    POWER="results/qwen25_medium_mac_cpu_${threads}t_r${r}_power.txt"

    rm -f "$RUN" "$POWER"

    python benchmarks/qwen25_medium_mac_cpu_power.py \
      --threads "$threads" \
      --duration 60 \
      > "$RUN" 2>&1 &

    PY_PID=$!

    while [ ! -f /tmp/qwen_cpu_ready ]; do
      sleep 0.1
    done

    sudo powermetrics -i 100 \
      -o "$POWER" \
      >/dev/null 2>&1 &

    PM_PID=$!

    sleep 0.2
    touch /tmp/qwen_cpu_go

    wait $PY_PID

    sudo kill -INT $PM_PID 2>/dev/null || true
    wait $PM_PID 2>/dev/null || true

    echo "DONE"
  done
done
