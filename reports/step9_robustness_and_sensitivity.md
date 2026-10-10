# Step 9 — Scheduler Robustness and Sensitivity Analysis

## 1. Objective and scope
Step 9 evaluates the sensitivity of the evidence-aware, QoS-constrained energy scheduler to synthetic QoS thresholds, energy point-estimate perturbations, and missing evidence.
The experiments replay existing hardware measurements offline. The UAV application is a motivating scenario only; no physical UAV, online scheduler deployment, or new inference benchmark was performed.

## 2. Data and experimental methodology
The broader characterization contains 385 measured configurations and 57 canonical scheduling candidates.
QoS and evidence-availability analyses focus on Apple M4, MPS, ResNet50, medium request profile, with FP16 and FP32 candidates.
Their measurements remain unchanged; only synthetic requirements or evidence availability are varied.
Real QoS gate functions from `benchmarks/evaluate_service_feasibility.py` are used, followed by the minimum-energy rule among fully feasible candidates.
Energy comparisons are restricted to compatible hardware and energy-measurement boundaries.

## 3. Single-variable QoS sensitivity
Source: `benchmarks/step9_qos_threshold_sweep.py` and `results/step9_qos_threshold_sweep.csv`.
Four dimensions were varied independently: latency, throughput, memory, and quality, with seven synthetic thresholds per dimension.
The resulting 28 scenarios produced 20 FP16 selections, 4 FP32 selections, and 4 abstentions.
Relaxing a maximum latency or memory constraint did not decrease the feasible candidate count.
Increasing minimum throughput or quality did not increase the feasible candidate count.
All implemented monotonicity assertions passed.

## 4. Joint latency-quality sensitivity
Source: `benchmarks/step9_latency_quality_interaction.py` and `results/step9_latency_quality_interaction.csv`.
A 5-by-5 synthetic threshold grid generated 25 scenarios and 50 candidate evaluations.
The decisions were 8 FP16 selections, 4 FP32 selections, and 13 abstentions.
When latency admitted only FP16 but the quality requirement admitted only FP32, neither candidate satisfied all gates and the scheduler abstained.
Both latency and quality monotonicity checks passed.
The matched ImageNetV2 Top-5 reference scores differ by approximately 0.01 percentage points; the synthetic threshold effect is not proof of statistically significant quality superiority.

## 5. Energy-ranking sensitivity
Source: `benchmarks/step9_energy_ranking_robustness.py`, `results/step9_energy_ranking_robustness.csv`, and `results/energy_pairwise_repeat_evidence.csv`.
Six pairwise comparisons span four compatible groups, with three energy repeats per candidate.

| Comparison | Median advantage | Critical symmetric perturbation | Observed ranges |
| --- | ---: | ---: | --- |
| ResNet50 heavy, FP16 vs FP32 | 30.42% | 17.94% | DISJOINT |
| ResNet50 light, FP32 vs FP16 | 4.80% | 2.46% | OVERLAP |
| ResNet50 medium, FP16 vs FP32 | 22.38% | 12.60% | DISJOINT |
| Intel CPU, 16 vs 1 thread | 56.43% | 39.30% | DISJOINT |
| Intel CPU, 8 vs 1 thread | 54.49% | 37.45% | DISJOINT |
| Intel CPU, 16 vs 8 thread | 4.26% | 2.17% | OVERLAP |

For positive median energies E_low < E_high, the critical symmetric perturbation is `p = (E_high - E_low) / (E_high + E_low)`.
Under hypothetical opposing perturbations of +/-1%, all 6 rankings are retained; under +/-3% and +/-5%, 4 of 6 are retained.
These bounds are mathematical sensitivity scenarios, not confidence intervals or measured error rates.
Four pairs have disjoint observed ranges and two have overlapping ranges. Neither classification alone establishes statistical significance.
Only 4 of 19 multi-candidate comparison groups have complete three-repeat pairwise evidence.
The six pairs are not six independent physical experiments because some pairs share candidates.

## 6. Evidence-availability robustness
Source: `benchmarks/step9_evidence_availability.py` and `results/step9_evidence_availability.csv`.
Two candidates each have four evidence fields: throughput, latency, memory, and quality.
All 2^8 = 256 binary availability masks were evaluated, yielding 512 candidate evaluations and 2,048 gate evaluations.
Across the gates, 1,024 results were PASS and 1,024 were UNKNOWN.
Candidate statuses were FEASIBLE in 32 evaluations and UNVERIFIED in 480 evaluations.
The final decisions were 16 FP16 selections, 15 FP32 selections, and 225 abstentions.
All 1,024 single-field evidence-addition transitions preserved feasible-set inclusion.
The transitions included 60 ABSTAIN-to-FP16, 60 ABSTAIN-to-FP32, 4 FP32-to-FP16, and 900 unchanged decisions.
The 225 abstentions are a combinatorial property of the artificial availability-mask enumeration, not an observed operational failure rate.

## 7. Consolidated verification
28 single-variable QoS scenarios and 25 joint latency-quality scenarios give 53 synthetic QoS scenarios.
Combined with 256 evidence-availability scenarios, the total is 309 synthetic QoS/evidence scenarios.
The six energy comparisons are reported separately and are not included in the 309 scenarios.
All CSV decision counts, critical energy perturbation values, evidence masks, and output formats passed the consolidated audit.
The Step 8 baseline of 42 policy decisions was also checked without modification.

## 8. Scientific limitations
The primary sensitivity case is a single Apple M4 ResNet50 FP16/FP32 candidate pair.
The experiments test expected deterministic decision properties; they do not establish general statistical superiority over independent scheduling algorithms.
Numeric QoS thresholds are synthetic and do not represent validated field SLAs.
Energy-repeat coverage is limited, and three repeats per candidate do not justify strong inferential claims.
Apple combined-SoC energy and Intel CPU-package energy are not directly comparable.
Platform memory metrics also have different measurement semantics.
No live dispatch, runtime adaptation, physical UAV operation, or field performance was evaluated.
The scheduler currently uses previously characterized measurements instead of continuously updated runtime state.

## 9. Reproduction
Run the following scripts from the repository root in the configured Python environment:
- `python benchmarks/step9_qos_threshold_sweep.py`
- `python benchmarks/step9_latency_quality_interaction.py`
- `python benchmarks/step9_energy_ranking_robustness.py`
- `python benchmarks/step9_evidence_availability.py`
Each script performs its own assertions and writes a Step 9 CSV into `results/`.

## 10. Conclusion
Step 9 extends Step 8 functional verification by identifying QoS decision boundaries, characterizing energy-ranking sensitivity, and systematically testing behavior under missing evidence.
The results support the correctness of the tested offline decision rules within the available measurement scope, without claiming real UAV deployment or general statistical dominance.
The next phase is academic-paper synthesis and reproducibility documentation.
