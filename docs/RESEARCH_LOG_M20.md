# M20 — Physical Multi-Round Qwen2-MoE Federation

## Amaç

M20'nin amacı, M19'da tek round olarak doğrulanan fiziksel Qwen2-MoE federated expert training mekanizmasını birden fazla global version boyunca çalıştırmak ve aşağıdaki davranışları test etmekti:

- version-aware local training
- multi-round expert aggregation
- validation-based ACCEPT / REJECT
- transactional commit
- rollback
- adaptive local-training budget
- diminishing returns / utility saturation

## Deney Ortamı

### Central / Client A

- MacBook Pro M4
- Apple MPS
- Domain: D0

### Client B

- MSI Pulse GL66
- NVIDIA RTX 3050 Laptop GPU
- CUDA
- Domain: D1

### Model

- Tiny Qwen2-MoE
- randomly initialized
- synthetic V2 dataset
- Layer 0 / Expert 0
- LoRA rank: 4
- LoRA alpha: 4
- expert-specific federated adaptation

Bu deney production-scale model performansını değil, federated MoE sistem mekaniklerini doğrulamaktadır.

## Aggregation

Her client LoRA adapter eğitti.

Naive LoRA factor averaging yerine effective weight delta kullanıldı:

    ΔW = B @ A

İki client için:

    ΔW_global = (ΔW_A + ΔW_B) / 2

Aggregated expert delta global expert shard üzerine uygulanarak yeni version adayı oluşturuldu.

## Validation Utility

D0 ve D1 target domain, D2 ve D3 retention domain olarak kullanıldı.

    utility = target_gain - 2 × retention_damage

Karar:

    utility > 0  → ACCEPT
    utility <= 0 → REJECT

REJECT edilen update global version'a commit edilmedi.

## Round Sonuçları

| Round | Local Steps | Target Gain | Retention Damage | Utility | Decision |
|---|---:|---:|---:|---:|---|
| V001 → V002 | 80 | +0.053016 | 0.019867 | +0.013283 | ACCEPT |
| V002 → V003 | 80 | +0.025257 | 0.014158 | -0.003060 | REJECT |
| V002 → V003 retry | 40 | +0.008935 | 0.002969 | +0.002998 | ACCEPT |
| V003 → V004 | 40 | +0.007288 | 0.002486 | +0.002317 | ACCEPT |
| V004 → V005 | 40 | +0.006305 | 0.002382 | +0.001541 | ACCEPT |
| V005 → V006 | 40 | +0.005037 | 0.002479 | +0.000080 | ACCEPT |

## Adaptive Retry

V002 → V003 round'u 80 local step ile REJECT edildi.

Aynı V002 base version üzerinden 40 step ile yeniden training yapıldı ve retention damage düştüğü için round ACCEPT edildi.

Akış:

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
    ACCEPT
      ↓
    versioned commit

## Global Validation Değişimi

| Domain | V001 Loss | V006 Loss | Değişim |
|---|---:|---:|---:|
| D0 | 4.097745 | 4.019499 | +0.078246 improvement |
| D1 | 4.113478 | 4.030559 | +0.082919 improvement |
| D2 | 4.226084 | 4.253219 | -0.027135 |
| D3 | 4.187648 | 4.220877 | -0.033229 |

Mean validation loss:

    V001: 4.156239
    V006: 4.131039
    Improvement: +0.025200

## Diminishing Returns

Accepted utility zaman içinde azaldı:

    V001 → V002   +0.013283
    V002 → V003   +0.002998
    V003 → V004   +0.002317
    V004 → V005   +0.001541
    V005 → V006   +0.000080

Bu sonuç utility saturation / diminishing returns eğilimine işaret etmektedir.

## M20 Sonucu

Fiziksel heterojen cihazlar üzerinde birlikte doğrulanan mekanizmalar:

- Mac/MPS local expert adaptation
- Windows/CUDA local expert adaptation
- LoRA adapter transfer
- effective-delta aggregation
- version-aware expert state
- multi-round federation
- validation gate
- ACCEPT / REJECT
- transactional global versioning
- rollback
- adaptive local-training retry

Final accepted global expert version:

    V006

## Sınırlamalar

- tiny randomly initialized Qwen2-MoE
- synthetic dataset
- yalnızca iki physical client
- tek routed expert
- küçük LoRA rank
- az sayıda federated round
- secure aggregation yok
- differential privacy yok
- production-scale network yok

Bu nedenle M20 sonucu production-scale convergence veya model quality iddiası değildir.

## Sonraki Milestone

    M21 — Multi-Expert Physical Federation
