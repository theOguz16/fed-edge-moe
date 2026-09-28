# M21 — Multi-Expert Physical Federation

## Amaç

M21'in amacı, tek bir global Qwen2-MoE modeli içinde birden fazla routed expert'in heterojen fiziksel client'lar tarafından federatif olarak adapte edilebildiğini doğrulamaktı.

M20'de yalnızca tek bir expert üzerinde multi-round federation çalıştırılmıştı.

M21 ile şu yeni mekanizmalar test edildi:

- routing-affinity based expert selection
- generic `(layer_id, expert_id)` training
- same-expert multi-client federation
- multiple federated experts inside one global MoE
- expert-wise aggregation
- versioned expert registry
- multi-expert global-state reconstruction
- unrelated expert isolation
- router isolation

---

## Deney Ortamı

### Client A

- MacBook Pro M4
- Apple MPS
- Local domain: D0

### Client B

- MSI Pulse GL66
- NVIDIA RTX 3050 Laptop GPU
- CUDA
- Local domain: D1

### Model

- Tiny Qwen2-MoE
- randomly initialized
- 2 MoE layers
- 4 routed experts per layer
- Top-2 routing
- LoRA rank 4
- Synthetic V2 dataset

Başlangıç global state:

    V006

M20'den gelen accepted expert:

    L0-E0

---

## M21A — Multi-Expert Routing Affinity

Global V006 üzerinde D0–D3 validation verilerinin expert routing dağılımı ölçüldü.

Aggregate affinity:

| Domain | En yüksek affinity |
|---|---|
| D0 | E3 = 33.40% |
| D1 | E0 = 32.01% |
| D2 | E1 = 33.37% |
| D3 | E0 = 32.51% |

Layer-specific seçimlerde:

    D0 → L0-E3 = 32.97%
    D1 → L1-E2 = 27.45%

M21 hedef expert'leri bu routing sinyallerine göre seçildi.

Önemli not:

    expert identity = (layer_id, expert_id)

Örneğin L0-E2 ve L1-E2 farklı expert'lerdir.

---

## M21B — Multi-Expert Base Preparation

İki target expert global modelden izole edildi:

    Mac / D0 → L0-E3
    MSI / D1 → L1-E2

Her expert:

    parameters: 6,144
    shard size: 24,800 bytes
    reload max diff: 0

Sonuç:

    MULTI-EXPERT BASE READY: True

---

## M21C — Generic Multi-Expert Client Trainer

M20 client trainer'daki sabit:

    TARGET_LAYER = 0
    TARGET_EXPERT = 0

yaklaşımı kaldırılarak generic `(layer, expert)` training geliştirildi.

Her client artık şu bilgileri alabilmektedir:

    client_id
    domain
    layer
    expert
    base_version
    local_steps

Her training sırasında:

- yalnız target expert materialize edilir
- LoRA yalnız target expert'e inject edilir
- router frozen kalır
- tensor-bank base weights frozen kalır
- update yalnız LoRA adapter olarak kaydedilir

Isolation doğrulaması:

    Base expert bank diff = 0
    Router diff = 0

---

## L0-E3 Federated Training

Aynı L0-E3 global expert'i iki fiziksel client üzerinde local olarak adapte edildi.

### Mac / D0

    Validation improvement: +0.002777
    Delta norm: 0.464965
    Routed tokens: 16,290
    Trainable params: 1,152
    Adapter bytes: 5,072
    Base expert bank diff: 0
    Router diff: 0

### MSI / D1

    Validation improvement: +0.003451
    Delta norm: 0.572736
    Routed tokens: 11,150
    Trainable params: 1,152
    Adapter bytes: 5,072
    Base expert bank diff: 0
    Router diff: 0

Her iki client aynı global base version'dan başladı.

Effective-delta aggregation kullanıldı:

    ΔW = B @ A

    ΔW_global =
        (ΔW_mac + ΔW_msi) / 2

---

## M21D — L0-E3 Expert-Wise Aggregation

Aggregated L0-E3 update validation gate üzerinden değerlendirildi.

| Domain | Improvement |
|---|---:|
| D0 | +0.001012 |
| D1 | +0.001077 |
| D2 | +0.000130 |
| D3 | -0.000067 |

Sonuç:

    Target gain:       +0.001045
    Retention damage:   0.000033
    Utility:           +0.000978
    Decision: ACCEPT

Global version:

    V006 → V007

Yeni accepted expert:

    L0-E3

---

## Versioned Expert Registry

M21 sırasında önemli bir mimari gereksinim ortaya çıktı.

Global model version artık yalnız tek bir expert shard ile temsil edilemez.

Örneğin:

    V007 =
        immutable V000 full model
        + accepted L0-E0 shard
        + accepted L0-E3 shard

Bu nedenle client trainer registry-aware hale getirildi.

Global state artık version'a göre accepted expert shard seti üzerinden reconstruct edilmektedir.

---

## L1-E2 Federated Training

L1-E2 expert'i V007 global state üzerinden iki fiziksel client tarafından eğitildi.

### Mac / D0

    Validation improvement: +0.001715
    Delta norm: 0.485898
    Routed tokens: 4,390
    Trainable params: 1,152
    Adapter bytes: 5,072
    Base expert bank diff: 0
    Router diff: 0

### MSI / D1

    Validation improvement: +0.003953
    Delta norm: 0.598868
    Routed tokens: 12,590
    Trainable params: 1,152
    Adapter bytes: 5,072
    Base expert bank diff: 0
    Router diff: 0

Routing miktarı client/domain'e göre farklılık göstermiştir.

D1'in L1-E2 affinity'si D0'dan daha yüksek olduğu için MSI/D1 tarafında daha fazla routed training token gözlenmiştir.

---

## M21E — Second Expert Aggregation

L1-E2 için iki physical client update'i effective-delta ile aggregate edildi.

| Domain | Improvement |
|---|---:|
| D0 | +0.000638 |
| D1 | +0.001115 |
| D2 | -0.000296 |
| D3 | -0.000353 |

Sonuç:

    Target gain:       +0.000876
    Retention damage:   0.000324
    Utility:           +0.000228
    Decision: ACCEPT

Global version:

    V007 → V008

---

## V008 Global Expert Registry

Final accepted multi-expert state:

    V008
    ├── L0-E0
    │   └── M20 federated expert
    │
    ├── L0-E3
    │   └── M21 Mac + MSI federation
    │
    └── L1-E2
        └── M21 Mac + MSI federation

Bu yapı aynı global MoE içinde birden fazla federatively adapted expert bulunduğunu göstermektedir.

---

## M21F — Global V008 Integrity Verification

V008 global state sıfırdan reconstruct edilerek integrity kontrolü yapıldı.

Sonuç:

    L0-E0 shard diff:      0
    L0-E3 shard diff:      0
    L1-E2 shard diff:      0

    Unrelated expert diff: 0
    Router diff:           0

    GLOBAL V008 INTEGRITY: True

Final validation losses:

    D0: 4.017849
    D1: 4.028367
    D2: 4.253384
    D3: 4.221297

Bu sonuç accepted expert shard'larının doğru yerleştirildiğini ve unrelated expert/router parametrelerinin değişmeden kaldığını doğrulamaktadır.

---

## M21 Sonucu

M21 kapsamında aşağıdaki zincir fiziksel ortamda doğrulanmıştır:

    single global MoE
            ↓
    multiple routed experts
            ↓
    same expert on multiple clients
            ↓
    local LoRA adaptation
            ↓
    effective-delta aggregation
            ↓
    validation gate
            ↓
    expert-wise commit
            ↓
    versioned expert registry
            ↓
    multi-expert global state

M21'in temel sonucu:

> FedEdgeMoE, aynı Qwen2-MoE global modelinde birden fazla expert'i heterojen fiziksel client'lardan gelen update'lerle bağımsız şekilde federatif olarak adapte edebilmekte ve expert-level versioned registry üzerinden global state'i deterministik biçimde reconstruct edebilmektedir.

---

## Sınırlamalar

M21 hâlâ kontrollü bir proof-of-concept deneyidir.

Mevcut sınırlamalar:

- tiny randomly initialized Qwen2-MoE
- synthetic dataset
- iki physical client
- yalnız iki yeni federated expert
- equal-weight aggregation
- secure aggregation yok
- differential privacy yok
- dynamic runtime placement henüz yok
- pretrained model henüz yok

Bu nedenle M21 sonucu production-scale multi-expert convergence iddiası değildir.

---

## Sonraki Milestone

    M22 — Dynamic Expert Placement

M22'nin amacı expert training/execution placement kararlarını runtime sırasında device capability, affinity, memory, latency, availability ve communication cost sinyallerine göre otomatik hale getirmektir.
