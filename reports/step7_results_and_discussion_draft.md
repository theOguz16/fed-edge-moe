# Results and Discussion — Step 7 Draft

## 1. Experimental Characterization and Candidate Coverage

The experimental characterization produced 385 distinct configurations, of which 109 were included in the scheduler core. After matching execution configurations against canonical workload profiles, 57 candidates were retained across 18 service-profile combinations.

Among these 57 candidates, measured batch-latency evidence and at least one memory observation were available for all 57 candidates. The final three memory gaps were closed using Windows process peak-working-set measurements for ConvNeXt-Base on Intel i7-11800H at one, eight, and 16 threads, each with three independent process runs. Memory metrics retain platform-specific semantics.

Quality evidence applicability was classified as SUPPORTED for 42 candidates, CONDITIONAL for nine, and UNVERIFIED for six. Importantly, applicability indicates whether relevant benchmark evidence exists; it does not establish satisfaction of a deployment-level quality requirement.

All 57 candidates remained UNVERIFIED in the baseline feasibility evaluation because numerical service-level QoS requirements had not been configured. Consequently, no candidate was certified as FEASIBLE or INFEASIBLE under actual UAV operating requirements.

## 2. QoS-Constrained Scheduler Behavior

The proposed evaluation pipeline applies throughput, latency, memory, and quality gates before energy-based selection. Only candidates satisfying all four configured gates are eligible for selection.

To verify this behavior without introducing unsupported deployment requirements, 14 synthetic QoS sensitivity scenarios were evaluated, comprising 32 candidate evaluations.

For ResNet50 on Apple M4 under the medium workload profile (224-pixel input resolution, batch size four), FP16 achieved a measured throughput of approximately 180.20 images/s and a batch latency of 23.60 ms, compared with 134.81 images/s and 28.72 ms for FP32.

When the synthetic latency deadline was stricter than both observed latencies, neither precision was feasible. An intermediate deadline admitted only FP16, while a relaxed deadline admitted both configurations and allowed energy-based ranking.

Throughput sensitivity demonstrated analogous threshold-dependent behavior.

Quality sensitivity further demonstrated that energy minimization does not override an explicit quality constraint. Under a synthetic ImageNetV2 Top-5 threshold between the two observed scores, FP32 was the sole eligible candidate despite its higher measured energy.

This synthetic case also illustrates why candidates dominated on energy, latency, and throughput must not be discarded before applying the quality constraint: the dominated FP32 candidate can remain the sole quality-eligible option. However, the observed Top-1 and Top-5 differences between precisions were very small. These sensitivity experiments validate decision logic rather than statistically significant differences in model quality.

## 3. Energy Efficiency and Measurement Uncertainty

For the ResNet50 medium profile on Apple M4, median energy consumption was 0.064559 J/image for FP16 and 0.083173 J/image for FP32, corresponding to a 22.38% lower median energy for FP16.

The observed three-repeat ranges were [0.064366, 0.068278] J/image for FP16 and [0.079676, 0.103142] J/image for FP32. These ranges did not overlap in the recorded experiments.

Nevertheless, three repetitions per configuration are insufficient to establish general statistical superiority. The energy selector therefore reports point-estimate-based decisions as provisional, with repeat evidence attached separately.

Across the repeat-based uncertainty analysis, nine configurations and 27 raw runs were examined. Six pairwise comparisons were available within four compatible comparison groups.

For ResNet50 under the light profile, FP32 had a 4.80% lower median energy than FP16, but the observed repetition ranges overlapped. This illustrates why a small difference in median energy should not automatically be interpreted as a reliable advantage.

On Intel i7-11800H, the ConvNeXt-Base medium workload exhibited a trade-off between thread count and latency. Eight threads achieved approximately 403.40 ms batch latency, while 16 threads achieved approximately 428.08 ms. The 16-thread configuration had a 4.26% lower median energy than the eight-thread configuration, although their observed energy ranges overlapped.

Canonical Intel CPU process-memory observations are now available for all three thread configurations. However, quality evidence remains conditional and deployment QoS thresholds remain unconfigured. Thus, these thread-level comparisons remain descriptive sensitivity evidence, not certified scheduler selections.

## 4. Pareto Analysis

A three-objective Pareto analysis considered energy per item and batch latency as minimization objectives, and throughput as a maximization objective.

The 57 canonical candidates formed 37 comparison groups, including 19 multi-candidate groups. Using measured point estimates, 41 candidates were classified as observed nondominated and 16 as observed dominated. All 57 candidates had the objective metrics required for Pareto evaluation. Eighteen comparison groups contained only one candidate; nondominance in these singleton groups does not establish superiority over an alternative.

For the ResNet50 medium workload on Apple M4, FP16 dominated FP32 on all three objectives.

For ConvNeXt-Base on Intel CPU, the one-thread configuration was dominated, whereas the eight-thread and 16-thread alternatives remained nondominated because of their performance-energy trade-off.

Only four of the 19 multi-candidate groups had complete three-repeat pairwise energy evidence. Accordingly, the Pareto results describe observed trade-offs and must not be interpreted as statistically validated superiority or proof of QoS feasibility.

## 5. Discussion and Threats to Validity

The offline experiments support a traceable, feasibility-first scheduler evaluation pipeline that separates measured performance, constraint satisfaction, energy ranking, and uncertainty evidence. The contribution established here is conservative decision behavior under measured and synthetic conditions, rather than demonstrated energy optimality in deployed UAV operation.

Several limitations remain.

First, energy measurements use different power boundaries across platforms: combined SoC power for Apple M4, GPU board power for RTX 3050, and CPU package power for Intel i7-11800H. Absolute energy values across these boundaries are therefore not directly comparable.

Second, memory observations use different measurement semantics, including framework allocator usage, process memory, and NVML GPU memory. Passing a bound on one measurement type does not establish total device-memory sufficiency.

Third, benchmark-reference quality evidence does not guarantee deployment quality under UAV operating conditions. In particular, Qwen3 Q4_K_M has a native Apple M4 llama.cpp WikiText-2 perplexity reference (PPL 21.3852, 16 context chunks), but its protocol is not equivalent to the existing Transformers FP16/FP32 seq64 evaluation and it does not establish deployment-level task quality, DistilGPT2 precision was not consistently controlled, and some model-device combinations rely on conditional cross-device references.

Fourth, Qwen3 CUDA batch-completion latency was recovered from archived llama.cpp benchmark logs: 0.588 s for light (one observation), 1.471 s for medium (median of 42 sequential observations), and 2.968 s for heavy (median of 21 sequential observations). These are not independent repetitions or end-to-end UAV service response times.

Finally, energy and performance measurements originate from specific benchmark procedures, and observed point-estimate ordering may be sensitive to measurement variability.

Therefore, the present results support the correctness and reproducibility of the scheduler's conservative decision logic under measured hardware characteristics and synthetic UAV-inspired workload scenarios. The UAV context serves as an application scenario rather than a physical experimental platform; no drone flight, onboard execution, or field deployment is evaluated.

The next evaluation stage will compare energy-only, QoS-first, and evidence-aware QoS-first scheduling policies using the same measured candidate registry and explicitly synthetic workload constraints. The comparison will examine constraint violations, selection and abstention behavior, and energy point estimates within compatible measurement boundaries. Broader energy uncertainty and task-quality evidence remain limitations, without requiring physical UAV experimentation.
