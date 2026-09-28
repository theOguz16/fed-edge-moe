# FedEdgeMoE Research Roadmap

## Research Roadmap

FedEdgeMoE araştırması küçük ve kontrollü prototiplerden başlayarak daha gerçekçi multi-expert ve multi-device sistemlere doğru ilerlemektedir.

## Tamamlanan Aşamalar

| Milestone | Amaç | Durum |
|---|---|---|
| M1–M6 | MiniMoE, expert isolation, federated expert aggregation ve baseline deneyleri | ✅ |
| M7–M10 | Physical federation, multi-round training ve validation-gated updates | ✅ |
| M11–M12 | Expert selection, multi-objective scheduling ve system-cost modeling | ✅ |
| M13–M15 | Split execution, heterogeneous scheduling ve robust round control | ✅ |
| M16 | Federated LoRA ve communication-efficient expert adaptation | ✅ |
| M17 | Generic expert interface ve Qwen2-MoE integration | ✅ |
| M18–M19 | End-to-end ve physical Qwen2-MoE federation | ✅ |
| M20 | Multi-round federation, rollback ve adaptive retry | ✅ |

---

## M21 — Multi-Expert Federation

### Hedef

Tek bir global MoE modeli içerisinde birden fazla expert'in farklı client ve domain kombinasyonları tarafından federatif olarak eğitilmesi.

### Araştırma Sorusu

Expert-specific federated adaptation birden fazla expert aynı anda güncellenirken isolation ve global model quality'yi koruyabilir mi?

### Ölçülecek Metrikler

- expert başına validation improvement
- unrelated expert parameter drift
- router parameter drift
- cross-domain retention
- expert-specific communication cost
- accepted / rejected round oranı
- client participation

### Success Criteria

- birden fazla expert bağımsız olarak federatif şekilde güncellenebilmeli
- güncellenmeyen expert'lerde istenmeyen parameter change oluşmamalı
- global validation sonucu ölçülebilir şekilde raporlanabilmeli

---

## M22 — Dynamic Expert Placement

### Hedef

Expert'lerin heterogeneous edge cihazlara runtime sırasında dinamik olarak atanması.

Scheduler şu faktörleri birlikte değerlendirecektir:

- expert affinity
- available memory
- device availability
- estimated latency
- communication cost
- training capability
- update staleness

### KPI / Metrics

- scheduling latency
- expert placement success rate
- device utilization
- round completion time
- communication volume
- fallback / reschedule frequency
- estimated cost ile measured cost arasındaki fark

### Beklenen Sonuç

Static expert placement yerine system-aware scheduling'in heterogeneous cihazlarda daha uygun placement kararları üretebildiğinin gösterilmesi.

---

## M23 — Pretrained MoE + Realistic Data

### Hedef

Tiny randomly initialized Qwen2-MoE deneylerinden pretrained bir sparse MoE modele ve daha gerçekçi dataset'lere geçmek.

### Ölçülecek Metrikler

- task validation loss / accuracy
- centralized baseline karşılaştırması
- full-model / full-expert adaptation karşılaştırması
- federated expert adaptation quality
- forgetting / retention
- communication reduction
- LoRA rank sensitivity
- convergence behavior

### Temel Araştırma Sorusu

Synthetic prototipte gözlenen federated expert learning davranışı pretrained model ve realistic non-IID data üzerinde de korunuyor mu?

---

## M24 — Multi-Node Physical Testbed

### Hedef

FedEdgeMoE'yi daha büyük fiziksel bir edge cluster üzerinde değerlendirmek.

Hedef testbed:

    Central Server
          +
    up to 12 heterogeneous edge nodes

### KPI / Metrics

- number of simultaneously active clients
- round completion latency
- expert migration / reassignment cost
- total communication volume
- device failure recovery
- client dropout handling
- system throughput
- memory footprint per edge device
- expert utilization distribution

### Beklenen Sonuç

FedEdgeMoE scheduler ve federation mekanizmalarının iki client dışındaki daha geniş bir physical deployment üzerinde çalışabilirliğinin ölçülmesi.

---

## Long-Term Target — Large Sparse MoE Deployment

Uzun vadeli hedef, daha büyük bir sparse MoE modelini central server ve heterogeneous edge cluster üzerinde çalıştırmaktır.

Örnek hedef ölçek:

    30B-class Sparse MoE
            +
    Central Server
            +
    ~12 Edge Nodes

Burada amaç tüm modeli her edge cihazda tutmak değildir.

Central server:

- backbone
- attention layers
- router
- global expert registry
- aggregation
- model versioning

bileşenlerini yönetirken, belirli expert'ler edge cihazlara yerleştirilebilir.

İki execution strategy birlikte araştırılacaktır:

    Long local adaptation
            ↓
    Federated Expert Learning

    Short remote computation
            ↓
    Distributed / Split Expert Execution

Scheduler cihaz kapasitesine ve workload'a göre uygun execution strategy'yi seçebilecektir.

---

## Main Research KPIs

| Alan | Ölçümler |
|---|---|
| Model Quality | validation loss, accuracy, target gain, retention damage |
| Communication | bytes/client, bytes/round, compression ratio |
| System Performance | round latency, training time, inference latency |
| Resource Efficiency | device memory, active parameters, compute utilization |
| Robustness | dropout recovery, stale update rejection, rollback, retry success |

---

## Expected Research Outcome

Araştırmanın başarılı olması durumunda FedEdgeMoE'nin aşağıdaki hipotezi destekleyen deneysel kanıt üretmesi hedeflenmektedir:

> Büyük bir sparse MoE modelinin expert'leri heterogeneous edge cihazlara dinamik olarak dağıtılabilir ve bu expert'ler local data üzerinde federatif olarak adapte edilirken communication ve edge-device resource requirements azaltılabilir.

Araştırmanın amacı yalnızca model accuracy artırmak değildir.

Temel hedef model quality, communication cost ve resource cost arasındaki trade-off'u incelemektir.

---

## Project Completion Vision

Araştırmanın nihai demonstrasyonu aşağıdaki akışı hedeflemektedir:

    Global Sparse MoE
            ↓
    Router / Scheduler
            ↓
    Dynamic Expert Placement
            ↓
    Heterogeneous Edge Devices
            ↓
    Local Expert Adaptation
            ↓
    LoRA / Expert Updates
            ↓
    Expert-wise Aggregation
            ↓
    Validation Gate
            ↓
    ACCEPT / REJECT
            ↓
    Versioned Global Model

M24, Phase 1 research validation için ana milestone olarak değerlendirilmektedir.

M24 sonrası scaling, privacy, communication optimization, fault tolerance ve daha büyük pretrained MoE deployment çalışmaları Phase 2 kapsamında şekillendirilecektir.
