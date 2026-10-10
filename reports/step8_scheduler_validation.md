# Step 8 — Scheduler Functional Validation and Policy Ablation

## 1. Research Objective

Step 8 evaluates whether the offline, energy-aware scheduler developed
in Step 7 implements its specified decision rules correctly.

The experiments use measured heterogeneous edge-hardware characteristics
and synthetic UAV-inspired QoS scenarios.

UAV is an application scenario only. No physical drone, onboard
execution, flight experiment, or field deployment was performed.

## 2. Proposed Scheduler

The evidence-aware scheduler evaluates four constraint gates:

- Throughput
- Latency
- Memory
- Quality

A candidate is FEASIBLE only if all four gates are PASS.

A known FAIL results in INFEASIBLE. Unresolved or unconfigured gates,
in the absence of a known FAIL, result in UNVERIFIED.

Only FEASIBLE candidates participate in energy minimization.

Energy comparisons are restricted to compatible service profiles,
devices, backends, workload semantics, and energy boundaries.

The selector abstains when no eligible candidate exists, when
required energy evidence is invalid, or when minimum energy
point estimates are tied.

## 3. Policy Ablation

Three policies were compared using previously recorded synthetic
QoS gate outcomes:

1. Energy-only: ignores all QoS gates.
2. Optimistic QoS-first: excludes FAIL but accepts UNKNOWN and
   NOT_CONFIGURED.
3. Evidence-aware QoS-first: accepts only candidates with four
   PASS gates.

The evaluation includes 14 synthetic scenarios and 32 underlying
candidate evaluations, producing 42 policy decisions.

| Policy | Selections / 14 | Selected with FAIL | Selected with unresolved evidence |
|---|---:|---:|---:|
| Energy-only | 14 | 6 | 4 |
| Optimistic QoS-first | 10 | 0 | 3 |
| Evidence-aware QoS-first | 7 | 0 | 0 |

The two risk indicators may overlap and must not be added
to infer a unique number of unsafe selections.

The results demonstrate the policies' expected behavior in
these scenarios. They do not establish general superiority
over independent external scheduling algorithms.

## 4. Energy Selector Functional Tests

The actual energy-selection program was exercised using
12 controlled subprocess tests.

Tested cases included:

- No FEASIBLE candidate
- One FEASIBLE candidate
- Energy ordering in both directions
- Exact and near-equal energy estimates
- Missing, NaN, and zero energy
- Unknown energy boundary
- Mixed workload shapes
- Contradictory FEASIBLE and QoS-gate statuses

Result: 12/12 PASS.

## 5. QoS Feasibility Functional Tests

The actual feasibility-gate functions were tested using
33 controlled synthetic fixtures.

The tests covered:

- Throughput thresholds and unit compatibility
- Latency deadlines and measurement semantics
- Memory budgets and measurement contracts
- Image quality, precision, resolution, and preprocessing
- Text perplexity and evaluation protocol
- Missing or conditional quality evidence
- PASS, FAIL, UNKNOWN, and NOT_CONFIGURED outcomes
- Aggregate FEASIBLE, INFEASIBLE, and UNVERIFIED decisions

Result: 33/33 PASS.

## 6. End-to-End Validation

The actual feasibility evaluator and energy selector were
executed consecutively under seven synthetic scenarios.

| Scenario | Observed selection |
|---|---|
| FP16 and FP32 both FEASIBLE | FP16 |
| Only FP32 passes quality | FP32 |
| Only FP16 passes latency | FP16 |
| Only FP16 passes throughput | FP16 |
| Only FP16 passes memory | FP16 |
| Quality evidence not applicable | No selection |
| Both candidates fail latency | No selection |

Result: 7/7 PASS.

Each scenario processed 57 canonical candidates and produced
37 comparison groups. The targeted assertions focused on
Apple M4 ResNet50 medium configurations.

Synthetic QoS requirements apply by service/profile, whereas
memory contracts apply by service/device/backend.

The tests accounted for these different scopes.

Original benchmark data, requirements, memory contracts,
quality references, and baseline feasibility outputs
were not modified.

Total direct functional and integration tests: 52/52 PASS.

## 7. Scientific Interpretation

The scheduler correctly prioritizes feasibility evaluation
over energy minimization in the tested cases.

When both Apple M4 ResNet50 medium configurations are FEASIBLE,
the lower-energy FP16 candidate is selected.

Under a synthetic ImageNetV2 Top-5 quality threshold lying
between the observed scores, FP32 is the sole eligible
candidate despite its higher measured energy.

This demonstrates why candidates dominated on energy,
latency, and throughput should not automatically be removed
before applying quality constraints.

In Intel CPU synthetic scenarios, measured memory may
satisfy the synthetic budget while conditional quality
evidence leaves the quality gate UNKNOWN. The evidence-aware
scheduler abstains.

## 8. Limitations

The 14 scenarios are synthetic and are not independent
random samples of UAV operations.

The ablation replays recorded QoS-gate outcomes rather
than independently implementing complete external schedulers.

The small ResNet50 Top-5 precision difference does not
establish statistical quality superiority.

Limited energy repeats do not establish statistical
superiority of the selected energy point estimates.

Apple M4 combined SoC, NVIDIA GPU board, and Intel CPU
package energy have different measurement boundaries.

Platform-specific memory observations are not equivalent
total device-memory measurements.

The results validate offline decision behavior under
controlled conditions, not physical UAV deployment
or real-time scheduling performance.

## 9. Reproducibility

Run the following programs from the repository root:

- benchmarks/step8_policy_ablation.py
- benchmarks/test_step8_energy_selector.py
- benchmarks/test_step8_feasibility_gates.py
- benchmarks/test_step8_end_to_end.py

Main result file: results/step8_policy_ablation.csv

Real deployment-level QoS thresholds remain unconfigured.
