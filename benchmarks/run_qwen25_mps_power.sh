#!/bin/bash
set -e

source .venv/bin/activate
mkdir -p results

sudo -v

for profile in light medium heavy; do
  for r in 1 2 3; do

    echo "================================"
    echo "$profile repeat $r"
    echo "================================"

    rm -f /tmp/qwen_ready /tmp/qwen_go

    python benchmarks/qwen25_medium_mps_power.py \
      --profile "$profile" \
      --duration 60 \
      > "results/qwen25_medium_mps_${profile}_r${r}.txt" 2>&1 &

    PY_PID=$!

    while [ ! -f /tmp/qwen_ready ]; do
      sleep 0.1
    done

    sudo powermetrics -i 100 \
      -o "results/qwen25_medium_mps_${profile}_r${r}_power.txt" \
      >/dev/null 2>&1 &

    PM_PID=$!

    sleep 0.2
    touch /tmp/qwen_go

    wait $PY_PID

    sudo kill -INT $PM_PID 2>/dev/null || true
    wait $PM_PID 2>/dev/null || true

    echo "DONE"
  done
done
