# FedEdgeMoE Research Log — M1–M10

**Project:** FedEdgeMoE  
**Status:** M1–M10 completed  
**Current stage:** Clean held-out evaluation completed; ready for dynamic expert assignment / scheduling research.

---

## 1. Project Goal

FedEdgeMoE investigates whether a **single global sparse Mixture-of-Experts model** can be federatively fine-tuned across heterogeneous edge devices by assigning only subsets of experts to clients.

The intended system differs from architectures where every edge device runs an independent small language model. FedEdgeMoE maintains one global model composed of:

- shared backbone parameters,
- routers,
- multiple sparse experts,
- a global expert/version registry.

Edge devices receive selected expert shards from this global model, train them on private local data, and return parameter updates rather than raw data.

The core research question is:

> Can sparse expert-level federated training reduce client compute, memory requirements, and communication while retaining useful global model quality under heterogeneous and non-IID edge data?

A secondary systems question is:

> Can expert assignment eventually adapt to device capacity, network conditions, expert specialization, staleness, availability, and observed training behavior?

---

## 2. Current Architecture

The prototype global model is a small custom Transformer language model with sparse MoE layers.

Core configuration:

- Vocabulary: 64
- Transformer layers: 3
- Hidden size: 128
- Attention heads: 4
- Experts per MoE layer: 8
- Top-K routing: 2
- Expert hidden size: 256
- Total model parameters: **2,574,592**
- One expert parameters: **98,304**
- One expert = approximately **3.82%** of total model parameters

Each Transformer layer contains a learned router and 8 expert MLPs. For each token, the router activates only the Top-2 experts.

Experts use a SwiGLU-like gated MLP.

The prototype currently separates:

**Learning plane**
- local expert training,
- expert delta generation,
- federated aggregation,
- model versioning,
- validation gating.

**Inference plane**
- distributed inference across physical devices is intentionally deferred.

The learning plane was prioritized because remote token/layer-level MoE inference introduces significantly harder latency and network problems.

---

## 3. Hardware Used

### Core / Coordinator

MacBook Pro M4:

- Apple M4
- 10 CPU cores
- 16 GB unified memory
- PyTorch MPS backend

Roles:

- global model owner,
- coordinator,
- expert registry,
- aggregator,
- simulated Client A,
- validation/evaluation server.

### Physical Edge Node

MSI Pulse GL66:

- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB VRAM
- 8 GB system RAM
- Windows
- PyTorch CUDA backend

Roles:

- physical federated Client B,
- local expert training,
- delta upload.

Physical LAN communication was successfully established between the MSI and Mac.

---

# MILESTONE HISTORY

## M1 — Single-Machine Sparse MiniMoE

Implemented the initial sparse MoE architecture:

- `ExpertMLP`
- `TopKRouter`
- sparse expert dispatch
- weighted expert combination
- load-balancing auxiliary loss
- Transformer blocks
- causal self-attention
- language-model output head

The model contains 3 MoE layers × 8 experts = **24 physical expert modules**.

Only Top-2 experts are active for each token in each MoE layer.

### M1 learning result

Synthetic V1 contained four deterministic sequence domains.

Initial model:

- loss ≈ 4.21
- accuracy ≈ 0.34%

After 400 training steps:

- final loss ≈ 0.0257
- accuracy = 100%

Expert utilization stayed relatively balanced rather than collapsing onto one or two experts.

### Main finding

A trainable sparse MoE Transformer was successfully implemented from scratch and trained end-to-end using PyTorch MPS.

---

## M1 — Expert Specialization Analysis

Routing analysis showed different domains preferred different experts.

A zero-ablation experiment then tested whether those routing preferences were functionally meaningful.

Example:

Removing Layer 0 Expert 6 caused:

- D0 accuracy drop: **13.88 pp**
- D1 accuracy drop: **5.74 pp**

Other experts showed selective importance for D2 and D3.

### Main finding

Experts were not merely routed differently; disabling particular experts selectively damaged particular domains.

This provided evidence of **functional expert specialization**.

It also motivated expert-level federated learning.

---

## M2 — Expert Isolation and Versioning

Implemented model checkpoint separation into:

- shared parameters,
- router parameters,
- individual expert shards.

A global snapshot was exported and reconstructed with:

- maximum parameter difference = **0.0**

Then only Layer 0 Expert 6 was modified.

Verification showed:

- only L0-E6 parameters changed,
- restoring the original expert returned the global model exactly to the original state.

### Main finding

Individual experts can be independently stored, transported, trained, replaced, and versioned without modifying unrelated model parameters.

---

## M3 — Simulated Local Edge Learning

Two simulated clients independently trained the same global expert:

- Client A → D0
- Client B → D1
- target expert → L0-E6

Only one expert was trainable.

Trainable ratio:

**98,304 / 2,574,592 = 3.82%**

The first experiment used an already saturated global model, so accuracy remained 100%.

### Important lesson

The federated mechanics worked, but the experiment could not measure learning benefit because the global model was already solved.

This revealed a **ceiling / saturation effect** in the experimental design.

---

## M4 — First Expert-Wise Federated Aggregation

Client A and Client B independently produced deltas from the same global expert.

FedAvg combined those deltas.

The first round completed successfully.

Client upload:

- **393,472 bytes/client**
- **786,944 bytes total**

Full-model float32 update size would be approximately 10.3 MB/client.

Expert-only update payload was therefore approximately:

**96.2% smaller**

However, because the global model was saturated, performance remained effectively unchanged.

### Main lesson

Federated aggregation mechanics were correct, but a weaker initial global model was required to measure actual FL benefit.

---

## M5 — Measurable Federated Adaptation

Multiple pre-federation checkpoints were generated.

Validation-style accuracies included:

- Step 20: 23.63%
- Step 30: 48.14%
- Step 50: 94.82%
- Step 75: 99.90%

Step 30 was selected because it was learned but not saturated.

### Shared expert selection

Step-30 expert routing and ablation analysis identified L0-E6 as a strong common expert for D0 and D1.

### Local learning

Client A / D0:

- 38.34% → **79.41%**

Client B / D1:

- 37.86% → **61.37%**

Only 3.82% of model parameters were trained.

### Expert FedAvg result

Global mean accuracy:

- before: **48.14%**
- after: approximately **61.8%**

Domain changes were approximately:

- D0: +36 pp
- D1: +20 pp
- D2: small degradation
- D3: approximately unchanged

### Main finding

Local expert learning could transfer useful client knowledge back into the global model.

---

## M6 — Baseline Comparisons

Three approaches were compared.

### Global V0

Mean accuracy:

**48.14%**

### Expert FedAvg

Mean accuracy:

**61.82%**

Trainable parameters:

**3.82%**

Upload:

**393,472 bytes/client**

### Full-Model FedAvg

Mean accuracy:

**50.49%**

Upload:

**10,341,944 bytes/client**

Full-model FL strongly adapted to client domains but caused large forgetting:

- D2: 55.47% → 24.22%
- D3: 60.94% → 21.09%

This demonstrated severe cross-domain interference / catastrophic forgetting in this controlled experiment.

### Centralized Expert Training

Mean accuracy:

**63.48%**

Only the same expert was trained, but D0 and D1 data were centrally available.

Difference from single-round Expert FedAvg:

approximately **1.66 percentage points**

### Main findings

In this synthetic setting:

1. Expert-only federated learning retained far more global knowledge than full-model FedAvg.
2. Expert upload was approximately **26.3× smaller** than full-model upload.
3. Expert FedAvg approached centralized expert-training quality.

These results support, but do not prove generally, the hypothesis that sparse parameter isolation can reduce federated cross-domain interference.

---

## M7 — First Physical Federated Round

The MSI became a real physical edge node.

The worker:

1. connected to the Mac coordinator over LAN,
2. downloaded the global model,
3. trained L0-E6 using CUDA,
4. generated an expert delta,
5. uploaded the delta back to the Mac,
6. participated in FedAvg.

Physical Client B result:

- D1 before: 37.89%
- D1 after: **61.72%**
- delta size: **393,472 bytes**

Physical global FedAvg:

- initial mean: **48.14%**
- physical global mean: **61.82%**

The physical result closely matched the simulated experiment.

### Cross-device reproducibility

The Mac/MPS simulated Client B and MSI/CUDA physical Client B produced very similar learning behavior.

This provided early evidence that the local expert-learning mechanism behaved consistently across different hardware and accelerator backends.

---

## M7 Engineering Issue — CUDA Driver

The MSI initially used NVIDIA driver 528.49.

PyTorch reported:

`torch.cuda.is_available() == True`

but the first real model transfer to CUDA failed.

After upgrading the NVIDIA driver to 617.14:

- CUDA tensor allocation succeeded,
- RTX 3050 training succeeded.

### Lesson

A CUDA device being detectable does not necessarily guarantee usable runtime compatibility.

Future environment validation should include an actual GPU tensor allocation or matrix operation, not only `torch.cuda.is_available()`.

---

## M8 — Versioned Multi-Round Physical FL

A version-aware coordinator and persistent physical MSI worker were implemented.

Global progression:

- V30 → V31
- V31 → V32
- V32 → V33
- V33 → V34
- V34 → V35

Results:

| Version | D0 | D1 | D2 | D3 | Mean |
|---|---:|---:|---:|---:|---:|
| V30 | 38.28% | 37.89% | 55.47% | 60.94% | 48.14% |
| V31 | 75.00% | 58.59% | 53.12% | 60.94% | 61.91% |
| V32 | 79.30% | 61.72% | 53.12% | 61.33% | **63.87%** |
| V33 | 79.30% | 61.72% | 51.95% | 61.33% | 63.57% |
| V34 | 79.69% | 61.72% | 51.56% | 60.94% | 63.48% |
| V35 | 79.69% | 61.72% | 50.78% | 60.55% | 63.18% |

Best checkpoint:

**V32 — 63.87% mean accuracy**

### Delta cosine similarity

- Round 1: +0.018851
- Round 2: -0.034529
- Round 3: -0.068306
- Round 4: -0.086835
- Round 5: -0.106295

Client updates became increasingly conflicting over time.

### Main finding

Performance converged rapidly during the first two rounds and then slowly degraded.

This suggested that blindly continuing federated rounds can cause accumulating interference.

---

## M8 Engineering Issue — Race Condition

The first M8 implementation uploaded client metadata twice.

The coordinator closed the active round after receiving the delta, causing the second metadata upload to receive:

`409 no active round`

The coordinator then attempted to access a telemetry field that had not been included in the first metadata upload and crashed with:

`KeyError: delta_upload_seconds`

### Fix

The redundant metadata upload was removed.

Upload timing was logged locally instead.

### Lesson

Federated communication protocols need explicit state transitions and should avoid telemetry updates that race with round completion.

---

## M9 — Validation-Gated Aggregation

A candidate global model was evaluated before its new version was accepted.

Results:

- V30: 48.14%
- Candidate V31: +13.77 pp → **ACCEPT**
- Candidate V32: +1.95 pp → **ACCEPT**
- Candidate V33: -0.29 pp → **REJECT**

Final accepted version:

**V32**

Early stopping:

**True**

### Important observation

Round 2 had negative cosine similarity:

**-0.034529**

but still improved validation accuracy by:

**+1.95 pp**

Therefore:

> Negative update cosine similarity alone is not sufficient reason to reject an FL update.

Cosine similarity is useful as a conflict diagnostic, while validation performance is a stronger acceptance signal.

---

# M10 — Clean Train / Validation / Held-Out Test Protocol

M10 addressed an important methodological limitation in earlier experiments.

Previous synthetic sequences did not provide enough distinct initial conditions for a meaningful train/validation/test separation.

Synthetic V2 therefore introduced recurrence rules based on two previous token states.

Each domain now contains:

**16 × 16 = 256 initial-state pairs**

Approximate split:

- 70% train
- 15% validation
- 15% held-out test

The test set remained sealed during training, expert selection, and checkpoint selection.

---

## M10A — Clean Global Base

Training progression:

| Step | Train Acc | Validation Acc |
|---|---:|---:|
| 0 | 1.61% | 1.65% |
| 25 | 15.89% | 12.81% |
| 50 | 33.42% | 29.96% |
| 75 | 48.13% | 44.20% |
| 100 | 51.22% | 49.65% |
| 150 | 57.64% | 53.51% |
| 200 | 68.04% | 61.08% |
| 250 | 72.86% | 65.84% |
| 300 | 79.48% | 71.34% |

Step 100 was chosen as the federated starting checkpoint.

Reason:

- model had learned meaningful structure,
- validation was not saturated,
- generalization gap was still relatively small,
- sufficient adaptation headroom remained.

---

## M10B — Validation-Only Expert Selection

Only the validation split was used for expert-selection analysis.

The test split remained sealed.

Selected shared expert:

**Layer 1 Expert 4 — L1-E4**

Reasons:

- D0 routing: 39.7%
- D1 routing: 25.5%
- zero-ablation increased loss in both D0 and D1
- significant shared routing compared with some alternatives.

This replaced earlier L0-E6, demonstrating that the important shared expert depends on model state and data distribution.

---

## M10C — Train-Only Federated Learning

Clients used only their respective train splits:

- Client A → D0 train
- Client B → D1 train

Server selection used only validation data.

Target expert:

**L1-E4**

Validation progression:

- V100: 49.68%
- V101: 50.02%
- V102: 51.15%
- V103: 52.27%
- V104: 52.82%
- V105: **53.21%**

Gain:

**+3.53 percentage points**

Target-domain changes:

- D0: 15.96% → 20.00%
- D1: 13.16% → 23.25%

Non-target domains:

- D2: 82.74% → 82.74%
- D3: 86.84% → 86.84%

All five rounds were accepted by validation.

Update cosine similarity again became more negative:

- +0.022972
- -0.009964
- -0.047216
- -0.077463
- -0.105159

However validation accuracy continued improving, again showing that cosine conflict alone should not control aggregation.

---

## M10D — First Sealed Held-Out Test

The test split was evaluated only after:

- training completed,
- shared expert was selected,
- round count was completed,
- final checkpoint V105 was fixed.

### Base V100 held-out test

- D0: 12.31%
- D1: 10.88%
- D2: 84.91%
- D3: 85.13%
- Mean: **48.31%**

### Final V105 held-out test

- D0: 19.32%
- D1: 27.54%
- D2: 84.91%
- D3: 85.13%
- Mean: **54.23%**

### Held-out improvements

- D0: **+7.01 pp**
- D1: **+16.67 pp**
- D2: **+0.00 pp**
- D3: **+0.00 pp**
- Mean: **+5.92 pp**

Mean held-out loss:

- 1.968117 → 1.878740

### Main M10 finding

Federated expert updates learned only from client train data and selected only with validation data improved performance on a previously unseen held-out test split.

At the same time, the two non-target domains retained exactly the same test accuracy.

Within this controlled synthetic experiment, this is the strongest evidence so far that expert-restricted federated adaptation can:

- transfer useful local knowledge,
- generalize beyond training data,
- preserve unrelated global capabilities,
- avoid full-model catastrophic forgetting.

---

# Current Strongest Findings

1. Sparse expert-level federated learning works mechanically and statistically in the controlled prototype.
2. Training only 3.82% of model parameters can produce meaningful global improvement.
3. Expert update payload is approximately 26× smaller than full-model update payload.
4. Full-model FedAvg caused severe non-target-domain forgetting in the tested non-IID setup.
5. Expert FedAvg retained substantially more non-target knowledge.
6. Federated expert performance approached centralized expert training in the original Synthetic V1 experiment.
7. Physical Mac/MPS + MSI/CUDA federated training successfully reproduced simulated behavior.
8. Multi-round FL can begin degrading after convergence.
9. Client update cosine similarity is informative but should not be used as a standalone acceptance rule.
10. Validation gating successfully prevented a degrading candidate global version from replacing the best model.
11. Clean held-out Synthetic V2 testing produced a +5.92 pp global accuracy improvement while preserving D2/D3 accuracy.

---

# Important Limitations

These results are not yet evidence that FedEdgeMoE will work at production or LLM scale.

Current limitations include:

- synthetic datasets,
- very small model,
- only two active federated client domains,
- mostly two physical compute devices,
- single or limited random seeds,
- simple FedAvg,
- manual/shared expert selection,
- no adversarial clients,
- no packet loss/dropout experiments,
- no asynchronous aggregation,
- no energy measurements,
- no LoRA expert updates yet,
- no dynamic expert placement,
- no large-scale heterogeneous cluster,
- no real language dataset,
- no real distributed MoE inference.

All current claims should therefore be framed as **prototype / controlled experimental evidence**, not general conclusions.

---

# Engineering Observations

Important non-ML findings so far:

- CUDA compatibility must be tested with a real tensor operation.
- Model version must be attached to every client update.
- Stale updates should not be accepted blindly.
- Federated round state transitions require explicit protocol design.
- Physical worker polling and server shutdown require graceful completion signaling.
- Current snapshot download is approximately 9.6 MB per round, while expert upload is only ~393 KB.
- Upload efficiency is already good, but model distribution is still inefficient.
- Future workers should keep static code/backbone locally and receive only required changed shards.
- Mac MPS local training was faster than the RTX 3050 worker for this particular small workload.
- Device performance should therefore be measured rather than inferred from GPU branding.

---

# Important Checkpoints / Results

Original Synthetic V1:

- `checkpoints/m5_candidates/global_step_0030`
- `checkpoints/m7/global_v0031_physical`
- `checkpoints/m8/global_v0032`
- `results/m8_multiround_physical.json`
- `results/m9_validation_gated_fl.json`

Clean Synthetic V2:

- Base: `checkpoints/m10_candidates/global_step_0100`
- Final accepted FL checkpoint: `checkpoints/m10c/global_v0105`
- Candidate training report: `results/m10a_base_candidates.json`
- Clean FL report: `results/m10c_clean_validation_fl.json`
- Sealed test report: `results/m10d_sealed_test.json`

---

# Next Milestone — M11

M11 should move from manually selected experts toward **dynamic expert assignment**.

The coordinator should eventually consider signals such as:

- expert routing affinity,
- expert functional importance,
- client/domain history,
- device compute capacity,
- memory capacity,
- network throughput,
- training latency,
- expert version and staleness,
- recent update conflict,
- expert utilization,
- availability / dropout state.

The key question becomes:

> Which client should train which expert during each federated round?

This is the transition from simple federated expert learning toward the intended FedEdgeMoE scheduler.

---

## Current Project State

The project has moved through four distinct stages:

**Stage 1 — Model correctness**  
Sparse MoE, routing, gradients, specialization.

**Stage 2 — Federated algorithm correctness**  
Expert isolation, local deltas, FedAvg, baselines.

**Stage 3 — Physical distributed system correctness**  
Mac coordinator, physical MSI client, CUDA/MPS heterogeneity, versioned rounds.

**Stage 4 — Experimental methodology correctness**  
Validation gating and sealed held-out evaluation.

The next stage is:

**Stage 5 — Intelligent expert scheduling and heterogeneous assignment.**