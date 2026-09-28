# FedEdgeMoE — M11 Research Log

## Goal

M11'in amacı manuel expert seçimini kaldırıp client routing telemetry kullanarak expert'leri otomatik seçen bir scheduler geliştirmekti.

---

## M11A — Train-Routing Affinity Scheduler

Scheduler yalnız TRAIN routing telemetry kullandı.

İlk skor:

shared_affinity = min(client_A_affinity, client_B_affinity)

Synthetic V2 üzerinde scheduler otomatik olarak:

L1-E4

seçti.

Bu, M10'da validation + ablation analiziyle manuel seçilen expert ile aynıydı.

Sonuç:
TRAIN routing telemetry expert seçimi için anlamlı bir sinyal olabilir.

---

## M11B — Scheduler Benchmark

Top-3 candidate aynı federated training protokolünde karşılaştırıldı.

Sonuç:

L1-E4
- global: 53.21%
- gain: +3.53 pp
- target gain: +7.06 pp
- retention: +0.00 pp

L0-E7
- global: 54.07%
- gain: +4.40 pp
- target gain: +8.96 pp
- retention: -0.17 pp

L0-E3
- ilk round gain: +0.08 pp
- validation gate nedeniyle reddedildi

Validation-best expert:

L0-E7

İlk scheduler L1-E4 seçtiği için scheduler prediction correct = False.

Ana ders:
Routing yoğunluğu tek başına federated usefulness'ı tam olarak tahmin etmiyor.

---

## M11C — Fixed-Horizon Diagnostic

L0-E3 early-stop olmadan 5 round çalıştırıldı.

Sonuç:

- final global: 53.18%
- cumulative gain: +3.50 pp
- target gain: +7.35 pp
- retention: -0.35 pp

L0-E3 aslında zayıf değildi.

İlk round:
+0.08 pp

İkinci round:
+1.26 pp

Ana ders:
Tek zayıf round sonrası early stopping iyi expert'leri yanlışlıkla eleyebilir.

---

## M11D — Balanced Affinity

Yeni post-hoc scheduler skoru:

shared_min = min(A, B)

balance_ratio = min(A, B) / max(A, B)

balanced_score = shared_min * balance_ratio

Eski veri üzerinde:

Old scheduler:
L1-E4

Balanced scheduler:
L0-E7

Bu, daha önce gözlenen validation-best expert ile eşleşti.

Ancak bu skor sonuç görüldükten sonra tasarlandığı için bağımsız kanıt sayılmadı.

---

## M11E — Independent Replication

Yeni Synthetic V3 oluşturuldu.

Değiştirilenler:

- recurrence rules
- train / validation partition
- model initialization seed

Base checkpoint önceden Step 100 olarak sabitlendi.

Test split kullanılmadı.

Original scheduler:

L1-E5

Balanced scheduler:

L0-E0

5-round fixed-horizon sonucu:

Original:
- final global: 24.55%
- gain: +4.29 pp
- target gain: +8.67 pp
- retention: -0.09 pp

Balanced:
- final global: 24.74%
- gain: +4.47 pp
- target gain: +9.67 pp
- retention: -0.72 pp

Balanced global advantage:

+0.18 pp

Ancak balanced scheduler daha fazla non-target forgetting üretti.

---

## M11F — Stochastic Replication

Toplam 3 replication değerlendirildi.

Replication 1:
- Original: L1-E5
- Balanced: L0-E0
- Balanced higher
- global difference: +0.18 pp
- target difference: +1.00 pp
- retention difference: -0.63 pp

Replication 2:
- Original: L2-E6
- Balanced: L2-E6
- same expert selected

Replication 3:
- Original: L1-E3
- Balanced: L1-E3
- same expert selected

Summary:

- Balanced higher: 1
- Original higher: 0
- Same expert: 2
- Tie: 0

Mean balanced-original differences:

- global: +0.061 pp
- target gain: +0.333 pp
- retention: -0.211 pp

---

## M11 Main Findings

1. Expert assignment can be automated from client TRAIN routing telemetry.

2. The selected expert changes across model seeds and routing states.

3. Raw shared routing affinity is useful but incomplete.

4. Balanced routing may improve adaptation quality in some cases.

5. Balanced scheduling did not show strong enough evidence to claim general superiority.

6. Adaptation improvement and knowledge retention can conflict.

7. Early stopping with zero patience is too aggressive for some experts.

8. Scheduler quality should eventually consider multiple objectives rather than only routing frequency.

---

## Current Scheduler Interpretation

Current evidence supports:

routing affinity = useful signal

but not:

routing affinity = complete expert utility estimate

Future scheduling should consider both adaptation benefit and retention cost.

---

## M11 Status

M11 COMPLETE.

Next research stage should build on:
- automatic expert assignment
- retention-aware scheduling
- client/device heterogeneity
