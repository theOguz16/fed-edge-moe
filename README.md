# FedEdgeMoE

**Heterojen Edge Cihazlarda Sparse Mixture-of-Experts Modelleri için Federated Expert Learning ve Distributed Execution**

FedEdgeMoE, tek bir sparse Mixture-of-Experts (MoE) modelinin expert hesaplamasını ve expert-specific training sürecini heterojen edge cihazlara dağıtmayı araştıran bir araştırma prototipidir.

Proje şu bileşenleri bir araya getirir:

- federated expert learning,
- parameter-efficient LoRA adaptation,
- expert-wise aggregation,
- distributed / split expert execution,
- heterogeneous device scheduling,
- validation-based commit ve rollback,
- adaptive local-training budget.

## Araştırma Sorusu

Ortak bir sparse MoE modeli, expert alt kümeleri farklı heterojen edge cihazlara atanarak federatif biçimde fine-tune edilebilir mi?

Bunu yaparken cihazların memory, compute ve communication maliyetleri azaltılabilir mi ve global model kalitesi kabul edilebilir seviyede korunabilir mi?

## Mimari

FedEdgeMoE, global model altyapısını expert execution ve expert adaptation süreçlerinden ayırmayı hedefler.

```text
                 Central Server
        ┌────────────────────────────┐
        │ Sparse MoE Global Model    │
        │                            │
        │ Backbone / Attention       │
        │ Router                     │
        │ Global Expert Registry     │
        │ Aggregation + Versioning   │
        └─────────────┬──────────────┘
                      │
             expert assignment
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Edge A      Edge B      Edge C
       Expert      Expert      Expert
       LoRA        LoRA        LoRA
       Training    Training    Training
          │           │           │
          └──── expert updates ────┘
                      │
               server aggregation
```

Projede iki temel execution modu araştırılmaktadır.

**Federated expert learning:** Seçilen expert'ler local data üzerinde adapte edilir ve yalnızca expert / LoRA update'leri server'a geri gönderilir.

**Distributed expert execution:** Backbone ve router central server üzerinde tutulurken, router tarafından seçilen expert computation bir edge cihazda uzaktan gerçekleştirilebilir.

Bu iki yaklaşım birbirinin alternatifi değildir. FedEdgeMoE'nin nihai hedefi, gerekli durumlarda ikisini birlikte kullanabilen hibrit bir sistem oluşturmaktır.

## Mevcut Prototip

Mevcut deney altyapısında şu teknolojiler kullanılmaktadır:

- Python
- PyTorch
- Hugging Face Transformers
- Qwen2-MoE architecture
- LoRA expert adaptation
- safetensors
- Apple Silicon / MPS
- NVIDIA CUDA
- fiziksel LAN communication

Fiziksel test ortamı:

```text
MacBook Pro M4 / MPS
        +
MSI Pulse GL66 / RTX 3050 CUDA
        +
LAN üzerinden central coordination
```

Mevcut Qwen2-MoE deneylerinde küçük ve randomly initialized bir model ile synthetic dataset kullanılmaktadır.

Bu deneylerin amacı production-level model kalitesi göstermek değil; sistem mimarisini, federated training mekaniklerini ve distributed execution yaklaşımını doğrulamaktır.

## Şu Ana Kadarki Sonuçlar

Prototip üzerinde şu mekanizmalar çalıştırılmış ve test edilmiştir:

- expert-isolated federated training,
- expert-wise FedAvg-style aggregation,
- validation-gated federated rounds,
- heterogeneous device scheduling,
- fiziksel Mac ↔ Windows/CUDA federation,
- split expert execution,
- split backpropagation,
- LoRA-based expert adaptation,
- effective-delta LoRA aggregation,
- Qwen2-MoE expert extraction ve reintegration,
- versioned global expert checkpoints,
- transactional ACCEPT / REJECT updates,
- harmful update rollback,
- adaptive local-training retry.

## Physical Multi-Round Qwen2-MoE Experiment

Aynı Qwen2-MoE expert'i, Mac/MPS ve Windows/CUDA olmak üzere iki fiziksel client üzerinde birden fazla federated round boyunca eğitilmiştir.

Validation kararında kullanılan utility:

```text
utility = target_gain - 2 × retention_damage
```

Elde edilen round sonuçları:

| Round | Local Steps | Utility | Decision |
|---|---:|---:|---|
| V001 → V002 | 80 | +0.013283 | ACCEPT |
| V002 → V003 | 80 | -0.003060 | REJECT |
| V002 → V003 retry | 40 | +0.002998 | ACCEPT |
| V003 → V004 | 40 | +0.002317 | ACCEPT |
| V004 → V005 | 40 | +0.001541 | ACCEPT |
| V005 → V006 | 40 | +0.000080 | ACCEPT |

V002 → V003 geçişinde 80-step local training sonucu oluşan update retention gate tarafından reddedilmiştir.

Aynı global version üzerinden local-training budget 40 step'e düşürülerek round tekrar çalıştırılmış ve retention damage yeterince azaldığı için update kabul edilmiştir.

Bu deney şu mekanizmanın fiziksel heterojen cihazlar üzerinde çalıştığını göstermektedir:

```text
REJECT
  ↓
rollback
  ↓
local training budget azalt
  ↓
retry
  ↓
validation
  ↓
ACCEPT / REJECT
  ↓
versioned commit
```

## Araştırma İlerlemesi

Proje şu ana kadar aşağıdaki araştırma aşamalarından geçmiştir:

- MiniMoE implementation
- expert isolation ve versioning
- federated expert aggregation
- physical federated learning
- validation-gated rounds
- expert scheduling
- multi-objective scheduling
- system-cost-aware scheduling
- split expert execution
- heterogeneous device placement
- robustness ve quorum policies
- federated LoRA
- generic expert adapters
- Qwen2-MoE integration
- physical Qwen2-MoE federation
- multi-round adaptive federation

Detaylı deney günlükları [`docs/`](docs/) klasöründe bulunmaktadır.

Seçilmiş deney çıktıları ise [`results/`](results/) klasöründe tutulmaktadır.

## Sonraki Aşamalar

Bir sonraki araştırma aşamalarında şu konular hedeflenmektedir:

1. multi-expert federated training,
2. dynamic expert placement,
3. pretrained MoE modelleri,
4. daha gerçekçi dataset ve non-IID workloads,
5. daha büyük fiziksel edge testbed,
6. baseline comparison,
7. communication / latency / memory analysis,
8. privacy ve robustness evaluation,
9. daha büyük sparse MoE modellerinin multi-node edge cluster üzerinde çalıştırılması.

Uzun vadeli hedeflerden biri, central server üzerinde büyük bir sparse MoE modeli çalıştırırken modelin expert'lerini farklı edge cihazlara dinamik olarak yerleştiren ve bu expert'leri local data üzerinde federatif olarak adapte edebilen bir runtime geliştirmektir.

## Sınırlamalar

Mevcut sonuçlar bir **system-level proof of concept** olarak değerlendirilmelidir.

Proje henüz aşağıdakileri göstermemektedir:

- production-scale MoE training,
- pretrained large-model performance,
- convergence guarantee,
- formal privacy guarantee,
- secure aggregation,
- differential privacy,
- adversarial robustness,
- large-scale multi-node performance.

Mevcut Qwen2-MoE deneyleri:

- tiny randomly initialized model,
- synthetic data,
- az sayıda client,
- sınırlı sayıda expert

üzerinde gerçekleştirilmiştir.

Bu nedenle mevcut sonuçlar doğrudan production-scale modellere genellenmemelidir.

## Proje Durumu

**Active research prototype — work in progress.**

Mevcut kabul edilmiş fiziksel global expert version:

```text
V006
```

Bir sonraki milestone:

```text
M21 — Multi-Expert Federation
```

https://arxiv.org/abs/2601.00583?utm_source=chatgpt.com
https://arxiv.org/abs/2408.11304?utm_source=chatgpt.com