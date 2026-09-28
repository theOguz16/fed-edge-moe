# FedEdgeMoE Research Log — M12 to M19

## 1. Current Research Direction

FedEdgeMoE investigates whether a **single sparse Mixture-of-Experts model** can be adapted across heterogeneous edge devices by assigning and federatively training only selected experts.

The current architecture combines two complementary mechanisms:

**Federated Expert Learning**

- A global model is maintained by the coordinator/server.
- Edge clients train only selected experts or expert adapters using local data.
- Raw local data remains on the client.
- Clients return expert updates rather than the full model.
- Updates targeting the same expert are aggregated on the server.

**Distributed / Split MoE Execution**

- Large backbone and router components can remain on the server.
- Only selected expert computation may execute on an edge device.
- Hidden-state activations are dispatched to the remote expert and expert outputs are returned.
- This allows devices that cannot store the full model to still participate.

Therefore:

> FedEdgeMoE = federated expert learning + heterogeneous expert scheduling + optional distributed/split expert execution.

---

# M12 — Multi-Objective and System-Cost-Aware Scheduling

## Goal

Extend expert selection beyond model quality by incorporating:

- adaptation gain,
- forgetting / retention,
- device speed,
- network cost,
- availability,
- and measurement confidence.

## Main Findings

A multi-objective utility was introduced:

`utility = target_gain - λ × forgetting`

Expert selection was shown to depend on the retention penalty λ.

For the tested candidates:

- E4 provided lower adaptation gain but zero measured forgetting.
- E7 provided larger target gain with small retention loss.
- Both appeared on the Pareto frontier.
- E3 was dominated.

Retention budgets changed the selected expert:

- strict retention budget → E4,
- relaxed retention budget → E7.

## System-Cost Results

Physical timing from Mac and MSI showed that simplified compute-only cost estimates are insufficient.

Measured values included approximately:

- Mac local training: ~3.46 s
- MSI local training: ~3.46 s
- physical federated round E2E: ~7.71 s
- expert delta: ~393 KB

A simulated slower edge device demonstrated that a device with better expert affinity may outperform a faster device when routing affinity is sufficiently high.

A major methodological result was:

> Scheduling decisions should prefer measured end-to-end cost over isolated component timing.

The scheduler was extended to mark decisions based on estimated costs as **PROVISIONAL** until physical measurement is available.

## Limitation

The slow-edge competitor remained simulated, so the scheduler flip involving that device is not yet a fully physical result.

---

# M13 — Communication-Efficient Expert Delivery and Split Execution

## Goal

Avoid requiring edge devices to store the entire global model.

## Expert Shard Bundle

A bundle containing only the required model pieces was constructed for expert `L0-E7`.

Measured size:

- full checkpoint: ~10.34 MB
- expert-oriented bundle: ~1.19 MB compressed
- reduction: ~88.5%
- reduction factor: ~8.7×

Shard reconstruction was exact:

`max diff = 0`

## Local Split Execution

The server computed the normal model path while selected expert execution was isolated.

Monolithic and split execution produced identical results:

`max diff = 0`

## Physical Mac → MSI Expert Execution

A physical split-execution experiment was completed:

- Mac retained the main model.
- MSI contained only the selected expert.
- Hidden-state expert jobs were sent to MSI.
- Expert results were returned to Mac.

Measured numerical difference:

`~4.8e-7`

The MSI did **not** load the full model.

## Split Backpropagation

Split training was also tested.

Local equivalence:

- same loss,
- same expert gradient,
- same expert update,
- exact / near-exact numerical agreement.

Physical Mac ↔ MSI split backpropagation also succeeded.

Expert update difference:

`~5.5e-7`

## Communication Comparison

For a representative 120-step workload:

- split training: ~37.68 MB
- expert-only federated update: ~0.787 MB
- federated shard bundle: ~1.583 MB
- old full-snapshot FL: ~10.74 MB

Thus long split training generated approximately:

`47.9×`

more communication than expert-only federated training.

## Hybrid Policy

This motivated a hybrid execution policy:

- short remote work → split execution/training,
- long local learning → federated local expert training,
- infeasible clients → skip or reschedule.

Physical one-step split latency:

`~1.94 s`

Under the tested latency and network budgets, only approximately three split steps were considered safe before local federated training became preferable.

## Important Caveat

Hidden-state activations used in split execution should **not** be treated as inherently private.

---

# M14 — Heterogeneous Edge Scheduling

## Goal

Build a scheduler that can place expert-training tasks on heterogeneous devices.

## Device Profiles

Profiles were created for:

- Mac M4 / MPS
- MSI RTX 3050 Laptop / CUDA
- simulated slow edge

Properties included:

- compute backend,
- available memory,
- federated-training capability,
- split-execution capability,
- measured or estimated latency.

## Scheduling Pipeline

The scheduler evolved into:

`capability → availability → memory feasibility → affinity/speed ranking`

Example outcomes:

- normal 3.5 GB requirement → Mac selected
- Mac unavailable → MSI selected
- 6 GB requirement → only Mac feasible
- no feasible device → task deferred

## Key Result

FedEdgeMoE can now distinguish:

> “Which expert should be trained?”

from:

> “Which device can and should train that expert?”

This becomes essential when device resources and expert affinity differ.

---

# M15 — Multi-Client Robustness

## Goal

Make federated rounds robust to unreliable clients.

The following failure modes were implemented and tested:

- dropout,
- stale updates,
- duplicate updates,
- partial participation,
- insufficient quorum,
- retry / rescheduling.

## Partial Participation

Example round:

- Client A: valid
- Client B: valid
- Client C: dropout
- Client D: stale

Result:

- accepted: 2
- rejected: 2
- participation: 50%
- round valid because quorum was satisfied.

## Robust FedAvg

Only valid updates were included in aggregation.

Stale client contamination was successfully excluded.

## Quorum

With minimum quorum = 2:

- 3 valid → COMMIT
- 2 valid → COMMIT
- 1 valid → ABORT / RETRY
- 0 valid → ABORT

## Retry

A failed first attempt with insufficient valid clients was retried.

Second attempt reached quorum and committed successfully.

## Duplicate Guard

Updates use an idempotency concept based on:

`(round, base_version, client_id)`

Duplicate updates are rejected so retries cannot accidentally increase a client's aggregation weight.

## Unified Round Controller

A final controller combined:

- dropout handling,
- stale rejection,
- duplicate rejection,
- quorum,
- retry,
- commit,
- global-version increment.

Example:

`V105 → V106`

after a successful retry.

---

# M16 — Federated LoRA for Experts

## Goal

Reduce expert-training communication by replacing full expert updates with low-rank adapters.

The base expert contained:

`98,304 parameters`

### Rank-8 LoRA

Trainable:

`9,216 parameters`

Payload:

`37,344 bytes`

Reduction versus full expert:

`~90.5%`

Initial LoRA output was exactly equal to the original expert because LoRA B matrices started at zero.

Base expert weights remained frozen during training.

## Initial Federated LoRA Failure

Naively averaging LoRA A and B matrices caused validation degradation.

Initial two-client aggregation:

- D0: negative
- D1: negative
- global mean: `-0.13pp`

This motivated investigation rather than assuming LoRA itself had failed.

## Client-Level Diagnostic

Client A / D0:

- training loss decreased,
- validation degraded.

Client B / D1:

- validation improved.

Update cosine:

`+0.005`

The updates were approximately orthogonal rather than strongly conflicting.

## Full Expert vs LoRA

Full expert training:

- D0: `+2.81pp`
- D1: `+7.35pp`

Rank-8 LoRA:

- D0: negative
- D1: small positive improvement

This showed that the selected expert and training data were viable; the limitation was primarily the LoRA configuration.

## SVD Rank Analysis

Full-expert update energy captured:

Rank 8:

- approximately 72–82%

Rank 16:

- approximately 88–94%

Therefore rank 8 was too restrictive for some client updates.

## Oracle Low-Rank Test

Approximating the trained full-expert delta using SVD produced:

D0:

- full: `+2.81pp`
- rank-8 oracle: `-0.70pp`
- rank-16 oracle: `+1.93pp`

D1:

- full: `+7.35pp`
- rank-8 oracle: `+3.25pp`
- rank-16 oracle: `+5.47pp`

This demonstrated that rank 16 possessed substantially better representational capacity.

## Training Horizon

Rank-16 LoRA required substantially more optimization steps.

D0:

- step 120: `-1.58pp`
- step 480: `+1.23pp`

D1:

- step 120: `+1.20pp`
- step 480: `+2.39pp`

Validation performance was non-monotonic.

## Effective-Delta Aggregation

Naive federated LoRA averages the factor matrices independently.

However, LoRA's effective weight update is based on:

`ΔW = B × A`

Therefore a second strategy aggregated effective weight deltas first and then factorized the resulting delta.

Results:

Naive A/B FedAvg:

`mean gain = -0.13pp`

Effective-delta aggregation:

- D0: `+0.35pp`
- D1: `+0.51pp`
- D2: `0`
- D3: `0`
- mean: `+0.22pp`

This became the preferred aggregation method.

## Validation Selection

Local validation checkpoint selection was added.

In the tested run:

- Client A best step: 480
- Client B best step: 480

Therefore checkpoint selection produced no additional gain because the final checkpoints were already the best observed checkpoints.

## Final Rank-16 Communication Cost

Rank-16 LoRA:

- parameters: `18,432`
- payload: `74,216 bytes`
- full expert: `393,472 bytes`
- payload reduction: `81.1%`
- compression factor: `5.30×`

## M16 Conclusion

Rank-16 LoRA provided positive federated adaptation with substantially lower communication, but full-expert training remained stronger.

Effective-delta aggregation was materially better than naive factor-wise FedAvg in the tested setup.

---

# M17 — Real MoE Integration Bridge

## Goal

Remove direct dependence on the custom MiniMoE implementation and connect FedEdgeMoE to an external MoE architecture.

## Generic Expert Interface

A generic expert adapter interface was introduced supporting:

- projection discovery,
- freeze / unfreeze,
- state extraction,
- delta calculation,
- parameter accounting.

## Generic LoRA Injection

LoRA injection was changed to operate through the expert adapter rather than directly depending on `ExpertMLP`.

## Projection Mapping

External expert implementations may use different projection names.

Example mapping:

- logical `gate_proj` → external `w1`
- logical `up_proj` → external `w3`
- logical `down_proj` → external `w2`

Mapping-aware injection allowed these experts to use the same FedEdgeMoE LoRA infrastructure.

## Hugging Face Qwen2-MoE

Transformers 5.17.0 was introduced specifically to use the real Qwen2-MoE model implementation rather than reimplementing the architecture.

A tiny Qwen2-MoE configuration was instantiated without downloading pretrained model weights.

The real Hugging Face expert representation uses tensor banks rather than independent expert modules.

Example:

`gate_up_proj[num_experts, ...]`

`down_proj[num_experts, ...]`

## Real Expert Shard Extraction

For Tiny Qwen2-MoE Layer 0 / Expert 0:

- parameters: `6,144`
- shard: `24,800 bytes`
- reload diff: `0`
- unrelated expert change: `0`

## Materialized Expert Bridge

A tensor-slice expert was converted into a standalone `nn.Module`.

Results:

- standalone parameters: `6,144`
- output difference: `0`
- commit difference: `0`
- unrelated expert change: `0`

## Qwen2-MoE + Generic LoRA

The materialized Qwen2 expert was then trained using the generic LoRA infrastructure.

Rank 4 result:

- trainable parameters: `1,152`
- frozen base change: `0`
- LoRA changed
- expert output changed
- merged-output difference: `~3.26e-9`
- commit difference: `0`
- unrelated expert change: `0`

## M17 Conclusion

The complete bridge became:

`Qwen2 expert tensor → standalone expert → Generic LoRA → training → merge → Qwen2 tensor commit`

FedEdgeMoE was no longer limited to the custom MiniMoE expert class.

---

# M18 — End-to-End Federated Tiny Qwen2-MoE

## Goal

Run federated expert adaptation inside the full Qwen2-MoE causal language model path.

## Router Probe

Expert 0 was confirmed to receive meaningful routing traffic.

Across D0 and D1 validation probes:

`1,170 routed Expert-0 selections`

Therefore Expert 0 was suitable for the experiment.

## Full-Model Gradient Connectivity

The complete gradient path was tested:

`causal-LM loss → model → router → Expert 0 → LoRA`

Results:

- trainable parameters: `1,152`
- routed E0 tokens: `17,087`
- train loss: `4.146285 → 4.113764`
- D0 validation loss: `4.156956 → 4.122956`
- original E0 base change: `0`
- other expert change: `0`
- router change: `0`

## Two-Client Federated Round

Client A trained on D0.

Client B trained on D1.

Local results:

- Client A: `4.156956 → 4.039118`
- Client B: `4.183437 → 4.046753`

Effective expert deltas were aggregated.

Global validation:

- D0 improvement: `+0.059211`
- D1 improvement: `+0.069959`
- D2 regression: `-0.012506`
- D3 regression: `-0.011836`

Mean loss:

`4.182446 → 4.156239`

Mean improvement:

`+0.026207`

Unrelated expert change:

`0`

Router change:

`0`

## Retention-Aware Gate

Using:

`utility = target_gain - 2 × retention_damage`

Measured:

- target gain: `+0.064585`
- retention damage: `0.012171`
- utility: `+0.040243`

Decision:

`ACCEPT`

The accepted Expert 0 shard contained:

- `6,144 parameters`
- `24,800 bytes`

## M18 Conclusion

This was the first complete Tiny Qwen2-MoE experiment combining:

- real Hugging Face MoE routing,
- full causal-LM forward/backward,
- expert-only LoRA,
- two federated clients,
- effective-delta aggregation,
- retention-aware validation,
- global expert checkpoint commit.

The model remained tiny and randomly initialized; this does not yet establish performance on a pretrained production LLM.

---

# M19 — Physical Qwen2-MoE Federation

## Goal

Reproduce the M18 federated Qwen2 experiment across two real heterogeneous devices.

## Physical Devices

### Server / Client A

MacBook Pro M4

- backend: MPS
- memory: 16 GB unified memory

### Client B

MSI Pulse GL66

- NVIDIA RTX 3050 Laptop GPU
- 4 GB GPU memory
- CUDA enabled
- PyTorch 2.14.0+cu126
- Transformers 5.17.0

## Global Checkpoint Synchronization

A single Tiny Qwen2-MoE `global_v000` checkpoint was created on the Mac.

The checkpoint was transferred to MSI.

SHA256 hashes matched between devices.

Therefore both physical clients started from exactly the same model weights.

## Physical Client B — MSI / CUDA / D1

Results:

- trainable parameters: `1,152`
- routed E0 tokens: `25,884`
- train loss: `4.195633 → 4.040597`
- D1 validation: `4.183437 → 4.046753`
- effective delta norm: `2.560044`
- training time: `1.265 s`
- adapter upload: `5,072 bytes`

The adapter was transferred back to the Mac over the LAN.

Returned adapter:

- 6 tensors
- 5,072 bytes
- successfully loaded using safetensors

## Physical Client A — Mac / MPS / D0

Results:

- trainable parameters: `1,152`
- routed E0 tokens: `22,615`
- train loss: `4.148428 → 4.041487`
- D0 validation: `4.156956 → 4.039117`
- effective delta norm: `2.527514`
- training time: `4.476 s`
- adapter upload-equivalent: `5,072 bytes`

## Cross-Device Reproducibility

Physical client values reproduced the corresponding simulated M18 clients almost exactly.

Client A delta norm:

`2.527514`

Client B delta norm:

`2.560044`

The physical aggregation reproduced the M18 simulated global result.

## Physical Aggregation

Global validation after aggregation:

- D0: `4.156956 → 4.097745`
- D1: `4.183437 → 4.113478`
- D2: `4.213578 → 4.226084`
- D3: `4.175813 → 4.187648`

Mean loss:

`4.182446 → 4.156239`

Mean improvement:

`+0.026207`

Other expert change:

`0`

Router change:

`0`

## Physical Retention Gate

Measured:

- target gain: `+0.064584`
- retention damage: `0.012171`
- utility: `+0.040243`

Decision:

`ACCEPT`

Global version:

`V000 → V001`

## M19 Conclusion

FedEdgeMoE successfully completed a physical heterogeneous federated round using:

- Mac / MPS
- MSI / CUDA
- one shared Tiny Qwen2-MoE global checkpoint
- client-local Expert 0 LoRA training
- LAN adapter transfer
- server-side effective-delta aggregation
- retention-aware validation
- versioned global commit

The physical result reproduced the earlier simulated result.

This provides evidence that the mechanism is not dependent on a single backend or single-machine simulation.

---

# Current Evidence

The project currently provides controlled evidence that:

1. Expert-only federated learning can isolate model updates to selected sparse-MoE experts.
2. Expert updates can be transferred between heterogeneous physical devices.
3. Expert-only communication is substantially smaller than full-model communication.
4. LoRA further reduces expert communication, though it sacrifices some adaptation capacity.
5. Naive LoRA factor averaging may be inferior to aggregating effective weight deltas.
6. Device placement decisions can depend on affinity, latency, memory, availability, and measurement confidence.
7. Split execution can allow thin devices to participate without storing the entire model.
8. Long split training can be significantly more communication-heavy than federated local expert training.
9. The FedEdgeMoE abstraction can operate on a real Hugging Face Qwen2-MoE implementation.
10. A physical Mac + NVIDIA CUDA federation can produce the same Tiny-Qwen2 federated result as controlled simulation.

---

# Important Limitations

The current results should not be overstated.

The main limitations are:

- Tiny Qwen2-MoE configurations are randomly initialized.
- No pretrained production-scale LLM has yet been federatively adapted.
- Only two real physical compute clients have been tested.
- Only one routed expert has been federatively trained in the Qwen2 experiments.
- Physical Qwen2 experiments currently cover one federated round.
- Synthetic data is still used.
- Seed count remains limited.
- The slower third edge profile remains simulated.
- LoRA federated gains are currently small relative to full expert fine-tuning.
- Split activations may leak information and should not be treated as private by default.
- Memory models are simplified and do not yet fully distinguish unified memory, VRAM, CPU RAM, and runtime allocator behavior.
- No secure aggregation, differential privacy, encryption protocol, or adversarial-client defense has yet been implemented.

---

# Research-Safe Current Claim

A defensible current claim is:

> FedEdgeMoE demonstrates that selected experts of a shared sparse-MoE model can be trained through heterogeneous federated clients while keeping the backbone, router, and unrelated experts unchanged. In controlled synthetic experiments and a Tiny Hugging Face Qwen2-MoE prototype, expert-only LoRA updates were trained on Mac/MPS and Windows/CUDA devices, transferred over a local network, aggregated using effective weight deltas, validated with a retention-aware gate, and committed as a versioned global expert update.

This is a systems and feasibility result.

It is **not yet** evidence that FedEdgeMoE improves a production-scale pretrained LLM.

---

# State at End of M19

Latest committed global expert version:

`V001`

Physical clients:

- Mac M4 / MPS / D0
- MSI RTX 3050 / CUDA / D1

Target:

`Layer 0 / Expert 0`

Physical client LoRA payload:

`5,072 bytes per client`

Physical federated global mean-loss improvement:

`+0.026207`

Retention-aware utility:

`+0.040243`

Commit decision:

`ACCEPT`

The next research phase should move from **basic feasibility** toward **multi-round stability, scaling, and realistic model/data conditions**.