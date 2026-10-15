# Edge AI Benchmark Dataset v2 — Draft

## Purpose

Cross-hardware characterization of six AI models:
full CPU, full GPU, hybrid CPU/GPU partitioning,
CPU thread scaling, latency, QoS, energy, memory,
and device-transfer costs.

This is an experimental dataset, not a trained
placement optimizer.

## Experiment identity

- config_id: stable hash of the complete experimental configuration
- run_id: unique identifier of one independent execution process
- experiment_family: full_model_baseline / group_cost_profile /
  hybrid_partition / resource_qos
- machine_id: mac_m4 / msi_3050 / future hardware
- model_id and exact weight identifier / revision
- weight_source: pretrained / random / other
- model_family: vision / text
- precision: actual model and activation precision
- workload_id, input_shape, batch_size, context_tokens,
  output_tokens, decoding mode
- backend_sequence: e.g. mps,cpu,mps
- group_device_mapping: ordered device per model group
- cut_points: ordered, architecture-specific group boundaries
- cpu_threads: global intra-op thread count
- repetition index, random seed, start time (UTC)
- runtime/software versions and power mode

## Recommended output tables

### experiments.csv
One row per independent process execution.
Include config_id, run_id, machine_id, model_id,
workload_id, cut_points, placement, threads,
status, validation result, timestamps, and provenance.

### raw_latency.csv
One row per timed inference or decoding step.
Include run_id, repeat_index, sample_index,
measurement_kind, latency_ms and synchronization policy.

### segment_metrics.csv
One row per measured segment or transfer sample.
Include run_id, group_start, group_end, backend,
segment_kind, duration_ms, bytes_transferred,
transfer_direction and measurement scope.

### resource_samples.csv
Timestamped telemetry.
Include run_id, timestamp, power_w, power_boundary,
memory_mb, memory_boundary, gpu_util_pct, cpu_util_pct,
temperature_c, and sampling source.
Unavailable measurements remain NULL, never zero.

### summary.csv
One row per config_id and analysis grouping.
Include valid_run_count, sample_count,
latency_mean_ms, latency_p50_ms, latency_p95_ms,
throughput, throughput_unit, energy_per_item_j,
peak_cpu_ram_mb, peak_gpu_memory_mb,
transfer_time_ms and validation_status.

### hardware.json
CPU/GPU identifiers, RAM, operating system,
driver/runtime versions, torch/transformers versions,
power mode, and supported telemetry methods.

### failures.csv
Record unsupported, OOM, timeout, numerical mismatch,
and other failures. Never silently discard attempts.

## Scientific rules

1. Historical weights=None and pretrained measurements
   belong to different experimental conditions.
2. Prefill and autoregressive decode are separate workloads.
3. Sequential hybrid execution is not pipeline parallelism.
4. A CPU thread setting applies to the whole process unless
   explicitly changed and the switching overhead is measured.
5. Energy boundaries must be named:
   CPU package, GPU board, SoC, or entire system.
6. P95 must include sample count and the independent process
   repetitions; small sample counts are exploratory.
7. Full-model baselines and hybrid experiments must use
   comparable input, precision, model revision and timing rules.
8. Estimated values must never be labeled as measured values.
9. Missing data must never be encoded as measured zero.
10. Only supported, memory-feasible jobs are scheduled.
