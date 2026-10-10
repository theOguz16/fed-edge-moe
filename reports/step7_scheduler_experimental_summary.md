# Step 7 — Scheduler Experimental Results

**Status:** Measured characterization, conservative feasibility, energy selection, uncertainty, sensitivity and Pareto analysis.

> Scope: Reproducible results from existing CSV artifacts. No new benchmark measurements are performed.

## 1. Experimental coverage

| Measurement | Count |
|---|---:|
| All characterized configurations | 385 |
| Scheduler core configurations | 109 |
| Canonical service candidates | 57 |
| Service/profile combinations | 18 |
| Candidates with latency evidence | 57 |
| Candidates with memory evidence | 57 |

## 2. Quality applicability and feasibility

| Evidence applicability | Candidates |
|---|---:|
| SUPPORTED | 42 |
| CONDITIONAL | 9 |
| UNVERIFIED | 6 |

All 57 baseline candidates are **UNVERIFIED**, not proven FEASIBLE or INFEASIBLE, because real service QoS thresholds are not configured.

SUPPORTED quality evidence means a matching benchmark reference exists; it does not independently establish a deployment SLA.

Quality limitations include native Qwen3 Q4_K_M WikiText-2 perplexity evidence only on Apple M4 (llama.cpp, 16 context chunks); the GGUF binary revision is unverified, the result is not comparable to HF seq64, and UAV task-quality requirements remain unverified. DistilGPT2 precision remains uncontrolled, and some quality references are conditional across devices.

## 3. Pareto analysis

| Pareto classification | Candidates |
|---|---:|
| Observed nondominated | 41 |
| Observed dominated | 16 |
| Missing objective metrics | 0 |

Comparisons cover **37 groups**, including **19 multi-candidate groups**. Only **4 of these 19 groups** have complete three-repeat pairwise energy evidence.

Pareto comparisons use observed energy, latency and throughput point estimates within compatible service, workload shape, device, backend, energy boundary and measurement semantics.

Qwen3 CUDA batch-completion latency was recovered from archived llama.cpp benchmark logs for three canonical profiles. Light has one sweep observation; medium and heavy have sequential sustained observations. These measurements do not represent independent experiment repetitions or end-to-end UAV request latency.

## 4. Repeat-based energy evidence

Energy uncertainty analysis covers **9 configurations**, each with three raw repetitions (**27 total runs**), across four comparison groups.

| Workload | Lower-median alternative | Median advantage | Observed range relation |
|---|---|---:|---|
| vision_easy / heavy / Apple M4 (vs precision=fp32;threads=-) | precision=fp16;threads=- | 30.42% | OBSERVED_DISJOINT |
| vision_easy / light / Apple M4 (vs precision=fp16;threads=-) | precision=fp32;threads=- | 4.80% | OBSERVED_OVERLAP |
| vision_easy / medium / Apple M4 (vs precision=fp32;threads=-) | precision=fp16;threads=- | 22.38% | OBSERVED_DISJOINT |
| vision_medium / medium / Intel i7-11800H (vs precision=fp32;threads=1) | precision=fp32;threads=16 | 56.43% | OBSERVED_DISJOINT |
| vision_medium / medium / Intel i7-11800H (vs precision=fp32;threads=1) | precision=fp32;threads=8 | 54.49% | OBSERVED_DISJOINT |
| vision_medium / medium / Intel i7-11800H (vs precision=fp32;threads=8) | precision=fp32;threads=16 | 4.26% | OBSERVED_OVERLAP |

OBSERVED_DISJOINT means only that the recorded three-repeat ranges do not overlap. It is not a confidence interval or a statistical significance claim.

## 5. Synthetic QoS sensitivity

| Analysis | Scenarios | Candidate evaluations |
|---|---:|---:|
| Latency — ResNet50 | 3 | 6 |
| Latency — Intel CPU (partial) | 4 | 12 |
| Throughput — ResNet50 | 3 | 6 |
| Quality — ResNet50 | 4 | 8 |
| **Total** | **14** | **32** |

ResNet50 medium / Apple M4 demonstrates the expected changes in feasibility as latency and throughput thresholds are tightened. A synthetic ImageNetV2 Top-5 threshold can select FP32 as the sole eligible candidate even though FP16 has lower measured energy.

Intel CPU sensitivity remains **partial evidence only**: Windows process peak-working-set measurements are now available for all three canonical thread settings, but conditional quality evidence and unconfigured deployment QoS prevent feasibility certification.

## 6. Measurement boundaries and limitations

- Apple M4 energy: `combined_soc`; RTX 3050 energy: `gpu_board`; Intel CPU energy: `cpu_package`. These boundaries must not be treated as interchangeable.
- Memory observations include different measurement semantics (allocator, process, NVML). An observed allocated-memory PASS does not prove total device capacity.
- ImageNetV2 Top-1/Top-5 accuracy, WikiText-2 perplexity and SelfCheck evidence follow distinct evaluation protocols. Benchmark-reference outcomes are not deployment-quality guarantees.
- Three energy repetitions provide descriptive ranges, not statistical proof of superiority.
- Synthetic QoS thresholds test decision logic; they are not validated UAV operating requirements.
- No baseline candidate is certified FEASIBLE and no cross-device global energy winner is reported.

## 7. Source artifacts

- `results/scheduler_candidate_registry.csv`
- `results/service_candidate_quality_applicability.csv`
- `results/service_feasibility_baseline.csv`
- `results/scheduler_pareto_candidates.csv`
- `results/scheduler_pareto_group_summary.csv`
- `results/energy_uncertainty_summary.csv`
- `results/energy_pairwise_repeat_evidence.csv`
- `results/qwen3_hard_q4_cuda_recovered_latency.csv`
- `results/qos_latency_sensitivity_resnet50_m4_medium.csv`
- `results/qos_latency_sensitivity_convnext_cpu_medium_partial.csv`
- `results/qos_throughput_sensitivity_resnet50_m4_medium.csv`
- `results/qos_quality_sensitivity_resnet50_m4_medium.csv`
- `results/qwen3_q4_native_ppl_mac_c16.json`
- `results/qwen3_q4_native_ppl_mac_c16_raw.txt`
- `results/qwen3_q4_wikitext_corpus_manifest.json`

Regeneration entrypoint: `python benchmarks/rebuild_scheduler_artifacts.py`.
