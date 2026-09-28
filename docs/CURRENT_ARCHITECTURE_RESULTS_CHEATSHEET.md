# FedEdgeMoE — Current Architecture & Results Cheat Sheet

## Project Goal

FedEdgeMoE explores whether a **single shared sparse Mixture-of-Experts model** can be trained across heterogeneous edge devices by federatively adapting only selected experts.

Core idea:

`Shared global MoE + expert-local training + expert-wise aggregation + heterogeneous scheduling`

---

## Current Architecture

```text
                    GLOBAL SERVER
                         │
        ┌────────────────┴────────────────┐
        │                                 │
 Shared Backbone / Router          Global Expert Registry
        │                                 │
        └──────── Expert Selection ───────┘
                         │
              Scheduler / Placement
                         │
          ┌──────────────┴──────────────┐
          │                             │
       Client A                      Client B
     Mac / MPS                    MSI / CUDA
          │                             │
     Local data D0                 Local data D1
          │                             │
   Expert-0 LoRA train            Expert-0 LoRA train
          │                             │
          └──── small updates ──────────┘
                         │
              Effective-ΔW Aggregation
                         │
               Retention-Aware Gate
                         │
                  ACCEPT / REJECT
                         │
                Global Version Commit
```

Optional thin-device mode:

`Server backbone → hidden states → remote expert → expert output → server`

This is split/distributed execution and complements federated learning rather than replacing it.

---

## Federated Learning Path

Current expert update path:

`Global checkpoint → client-local LoRA → effective weight delta → server aggregation → validation → version commit`

LoRA effective update:

`ΔW = scaling × (B × A)`

FedEdgeMoE aggregates the **effective weight deltas**, not the LoRA A/B factors independently.

Reason:

`mean(B × A) ≠ mean(B) × mean(A)`

in general.

---

## Current Real-Model Integration

Model implementation:

**Hugging Face Qwen2-MoE**

Current bridge:

`Qwen2 tensor expert → standalone PyTorch expert → Generic LoRA → training → merge → Qwen2 tensor`

Verified operations:

- expert shard extraction,
- exact shard reload,
- standalone materialization,
- generic LoRA injection,
- local expert training,
- LoRA merge,
- tensor commit,
- full causal-LM forward/backward.

---

## Current Physical Testbed

### Client A

MacBook Pro M4

- backend: MPS
- local domain: D0
- physical LoRA training: working

### Client B

MSI Pulse GL66

- RTX 3050 Laptop GPU
- CUDA
- local domain: D1
- physical LoRA training: working

Both devices start from the exact same global checkpoint, verified by SHA256.

---

## Physical Qwen2-MoE Result

Target:

`Layer 0 / Expert 0`

Trainable LoRA parameters per client:

`1,152`

LoRA upload per client:

`5,072 bytes`

### Client A — Mac / D0

Validation loss:

`4.156956 → 4.039117`

Effective delta norm:

`2.527514`

Training time:

`4.476 s`

### Client B — MSI / D1

Validation loss:

`4.183437 → 4.046753`

Effective delta norm:

`2.560044`

Training time:

`1.265 s`

---

## Physical Federated Aggregation Result

After aggregating both physical client updates:

| Domain | Before | After | Change |
|---|---:|---:|---:|
| D0 | 4.156956 | 4.097745 | +0.059210 improvement |
| D1 | 4.183437 | 4.113478 | +0.069959 improvement |
| D2 | 4.213578 | 4.226084 | -0.012506 retention |
| D3 | 4.175813 | 4.187648 | -0.011836 retention |

Mean loss:

`4.182446 → 4.156239`

Mean improvement:

`+0.026207`

Unrelated expert change:

`0.0`

Router change:

`0.0`

---

## Retention-Aware Gate

Current utility:

`utility = target_gain - 2 × retention_damage`

Physical round:

- target gain: `+0.064584`
- retention damage: `0.012171`
- utility: `+0.040243`

Decision:

`ACCEPT`

Global version:

`V000 → V001`

---

## LoRA Findings

Full expert size:

`393,472 bytes`

Rank-16 MiniMoE LoRA:

`74,216 bytes`

Reduction:

`81.1%`

Important findings:

- rank 8 was too restrictive,
- rank 16 represented full-expert updates substantially better,
- LoRA required a longer optimization horizon,
- validation performance was non-monotonic,
- naive LoRA factor FedAvg degraded performance,
- effective-delta aggregation produced positive global gain.

---

## System Findings

FedEdgeMoE currently supports:

- expert selection,
- heterogeneous device profiling,
- availability filtering,
- memory feasibility,
- latency-aware placement,
- expert affinity,
- dropout handling,
- stale-update rejection,
- duplicate protection,
- quorum,
- retry,
- versioned global commits.

Split execution is available for devices unable to host the full model.

Measured experiments showed that long split training can be much more communication-heavy than local federated expert training.

---

## Strongest Current Result

The current prototype has demonstrated:

> A shared Tiny Qwen2-MoE model can be federatively adapted across heterogeneous Mac/MPS and Windows/CUDA devices by training only one routed expert with LoRA, transferring approximately 5 KB client updates over the network, aggregating effective expert deltas on the server, protecting unrelated experts and the router, applying a retention-aware acceptance gate, and committing the accepted result as a new global expert version.

---

## What Is Not Yet Proven

Current evidence does **not** yet establish:

- production-scale pretrained LLM performance,
- large Qwen/Mixtral/DeepSeek-scale deployment,
- real user/private datasets,
- many-client federation,
- multi-expert simultaneous federation,
- long multi-round stability,
- secure aggregation,
- differential privacy,
- adversarial-client robustness,
- privacy of split hidden states.

Current Qwen2-MoE experiments use a **tiny randomly initialized model and synthetic data**.

---

## Current State

```text
Latest milestone: M19 COMPLETE

Global version: V001

Physical clients:
Mac M4 / MPS
MSI RTX 3050 / CUDA

Current real-model target:
Qwen2-MoE
Layer 0 / Expert 0

Next research direction:
multi-round physical federation,
multi-expert scheduling,
scaling,
and more realistic pretrained models/data.
```