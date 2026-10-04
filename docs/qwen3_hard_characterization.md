# Qwen3-1.7B Hard Text Characterization

## 1. Purpose

Bu çalışma, Hard/Text workload olarak Qwen3-1.7B modelinin Mac M4 ve MSI RTX 3050 Laptop GPU üzerindeki feasibility, throughput, sustained performance, power ve energy davranışını karakterize eder.

Hard aşamasının temel amacı resource-aware scheduler için şu soruları incelemektir:

- Native FP16 her cihazda çalışıyor mu?
- Aynı quantized model ve runtime kullanıldığında cihaz davranışları nasıl değişiyor?
- Batch size cihazlara göre farklı optimum noktalar oluşturuyor mu?
- Sustained workload kısa benchmark sonuçlarından farklı mı?
- Throughput ve energy/token arasında hangi trade-off oluşuyor?

---

## 2. Hardware

### Mac

- Apple M4
- 10 logical CPU
- Apple GPU / Metal
- Unified memory
- macOS

### MSI

- Intel Core i7-11800H
- NVIDIA RTX 3050 Laptop GPU
- 4 GB VRAM
- Windows 11
- CUDA

---

## 3. Native FP16 Feasibility

Primary model:

Qwen3-1.7B

### Mac MPS

Workload:

- Context = 128
- Output = 64
- Batch = 1
- FP16

Result:

- Latency = 2.901 s
- Throughput = 22.06 tok/s
- MPS allocated ≈ 3.22 GB
- Status = PASS

### MSI CUDA

Native FP16 model load failed because the RTX 3050 has only 4 GB VRAM.

Result:

- Status = OOM
- Stage = model load

Bu nedenle native FP16 sonuçları cihazlar arası kontrollü karşılaştırma için kullanılmadı.

---

## 4. Controlled Cross-Device Configuration

Mac ve MSI için aynı model ve quantization kullanıldı:

- Model: Qwen3-1.7B
- Format: GGUF
- Quantization: Q4_K_M
- Model file size: ≈ 1.19 GiB
- Runtime: llama.cpp
- GPU offload: `-ngl 99`

Runtime build:

- Mac: llama.cpp build 11146
- MSI: llama.cpp build 11321

Build numaraları birebir aynı değildir. Bu fark metodolojik bir limitation olarak not edilmiştir.

---

## 5. Initial Q4 Baseline

### Mac

- pp128: 1032.74 tok/s
- tg64: 79.99 tok/s

### MSI RTX CUDA

- pp128: 3077.76 tok/s
- tg64: 119.89 tok/s

Controlled Q4 baseline her iki cihazda PASS oldu.

---

## 6. Workload Matrix

Full controlled sweep:

- Context: 128, 512, 1024
- Output: 64, 128
- Batch: 1, 2, 4
- Total: 18 configurations

Generation throughput için `S_TG` kullanıldı.

---

## 7. Mac Q4 Sweep

| CTX | OUT | B1 | B2 | B4 |
|---:|---:|---:|---:|---:|
| 128 | 64 | 76.19 | 112.18 | 98.21 |
| 128 | 128 | 72.66 | 119.03 | 102.99 |
| 512 | 64 | 75.83 | 113.27 | 97.86 |
| 512 | 128 | 75.81 | 112.75 | 97.55 |
| 1024 | 64 | 72.72 | 105.34 | 91.62 |
| 1024 | 128 | 72.38 | 104.92 | 91.46 |

Units: generation tok/s.

Mac üzerinde Batch 2 bütün workload gruplarında Batch 1 ve Batch 4'ten daha yüksek generation throughput verdi.

Bu nedenle Qwen3 Q4 workload için Mac tarafında B2 belirgin bir sweet spot oluşturmaktadır.

---

## 8. MSI RTX CUDA Q4 Sweep

| CTX | OUT | B1 | B2 | B4 |
|---:|---:|---:|---:|---:|
| 128 | 64 | 117.88 | 223.40 | 300.64 |
| 128 | 128 | 119.70 | 222.87 | 299.03 |
| 512 | 64 | 114.82 | 208.36 | 274.67 |
| 512 | 128 | 113.83 | 207.74 | 273.17 |
| 1024 | 64 | 109.15 | 194.63 | 252.51 |
| 1024 | 128 | 109.80 | 194.70 | 250.55 |

Units: generation tok/s.

RTX üzerinde Batch 1 → Batch 2 → Batch 4 geçişinde throughput sürekli arttı.

RTX bu workload'da B4 parallelism'den hâlâ önemli ölçüde faydalanmaktadır.

---

## 9. Cross-Device Behavior

Representative comparison:

| Workload | Batch | Mac | RTX |
|---|---:|---:|---:|
| 128/64 | 1 | 76.19 | 117.88 |
| 128/64 | 2 | 112.18 | 223.40 |
| 128/64 | 4 | 98.21 | 300.64 |
| 512/128 | 1 | 75.81 | 113.83 |
| 512/128 | 2 | 112.75 | 207.74 |
| 512/128 | 4 | 97.55 | 273.17 |
| 1024/128 | 1 | 72.38 | 109.80 |
| 1024/128 | 2 | 104.92 | 194.70 |
| 1024/128 | 4 | 91.46 | 250.55 |

RTX 18/18 controlled workloads üzerinde daha yüksek generation throughput verdi.

En önemli davranış farkı batch scaling'dir:

Mac:

B1 → B2 artış  
B2 → B4 düşüş

RTX:

B1 → B2 → B4 sürekli artış

Bu sonuç scheduler açısından cihaz-spesifik batch seçiminin gerekli olduğunu göstermektedir.

---

## 10. Representative Power Profiles

### Light

- Context = 128
- Output = 64
- Batch = 1

### Medium

- Context = 512
- Output = 128
- Batch = 2

### Heavy

- Context = 1024
- Output = 128
- Batch = 4

---

## 11. RTX Sustained Performance

Yaklaşık 60 saniyelik no-monitor sustained workloads çalıştırıldı.

### Medium

Generation throughput yaklaşık:

209.90 → 201.53 tok/s

Yaklaşık düşüş:

4%

Median sustained throughput:

≈ 205.48 tok/s

### Heavy

Generation throughput yaklaşık:

257.40 → 247.70 tok/s

Yaklaşık düşüş:

3.8%

Median sustained throughput:

≈ 251.37 tok/s

Qwen3 Q4 + llama.cpp workload, daha önce gözlenen Medium/Qwen2.5 PyTorch workload'a kıyasla daha stabil sustained behavior göstermiştir.

---

## 12. RTX NVML Power

Measurement:

- NVML
- 100 ms sampling
- approximately 60-second workloads
- 3 repeats
- Power boundary: NVIDIA GPU board power

| Profile | Throughput | Power | GPU Util | Peak VRAM | Peak Temp | J/run | J/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Light | 112.80 | 59.27 W | 97.1% | 2461 MB | 80 C | 35.91 | 0.5612 |
| Medium | 191.80 | 59.34 W | 96.0% | 2471 MB | 85 C | 94.00 | 0.3672 |
| Heavy | 235.50 | 59.33 W | 96.6% | 2465 MB | 86 C | 188.06 | 0.3673 |

RTX yaklaşık 59 W GPU board power ve %96–97 GPU utilization seviyesinde çalışmıştır.

Medium ve Heavy arasında J/token neredeyse aynıdır.

Heavy daha yüksek throughput sağlasa da enerji/token açısından Medium'a kıyasla belirgin bir kazanç sağlamamaktadır.

---

## 13. RTX Monitoring Effect

No-monitor sustained ve NVML monitored throughput karşılaştırması:

### Medium

- No monitor: ≈ 205.48 tok/s
- NVML: 191.80 tok/s
- Difference: ≈ 6.7%

### Heavy

- No monitor: ≈ 251.37 tok/s
- NVML: 235.50 tok/s
- Difference: ≈ 6.3%

Power sonuçları monitored operating condition'a aittir.

---

## 14. Mac Accelerator Power

İlk Mac power ölçümünde `powermetrics`, benchmark process'i model yüklemeden önce
başlatıldığı için model-load enerjisinin ölçüme karışma ihtimali vardı.

Bu nedenle önceki power tablosu final sonuç olarak kullanılmamıştır.

Final ölçümde:

- runtime: llama.cpp
- model: Qwen3-1.7B Q4_K_M
- GPU offload: enabled
- `powermetrics` benchmark model load tamamlandıktan sonra başlatıldı
- sampling interval: 100 ms
- 3 repeats
- power boundary: CPU + GPU + ANE combined SoC power

| Profile | Throughput | Power | J/run | J/token |
|---|---:|---:|---:|---:|
| Light | 72.22 | 13.44 W | 14.03 | 0.2192 |
| Medium | 113.38 | 16.35 W | 52.81 | 0.2063 |
| Heavy | 91.29 | 17.19 W | 164.50 | 0.3213 |

No-monitor Q4 sweep ile karşılaştırıldığında:

- Light: 76.19 -> 72.22 tok/s, yaklaşık -5.2%
- Medium: 112.75 -> 113.38 tok/s, yaklaşık +0.6%
- Heavy: 91.46 -> 91.29 tok/s, yaklaşık -0.2%

Medium ve Heavy profillerinde monitoring etkisi ihmal edilebilir düzeydedir.

Light profil çok kısa sürdüğü için sampling overhead ve kısa-workload state etkilerine
daha hassastır. Bu nedenle Light power sonucu kullanılabilir ancak diğer iki profile
göre daha yüksek belirsizlik taşır.

Mac üzerinde Medium profil hem Light'tan daha yüksek throughput sağlamakta hem de
Light'a göre daha düşük J/token göstermektedir.

Heavy profil ise daha yüksek batch'e rağmen throughput kaybetmiş ve enerji/token
maliyeti belirgin biçimde artmıştır.

---

## 15. CPU Backfill Characterization

Hard/Text karakterizasyonunda eksik kalan CPU-only `long_single` ve `batch`
workload'ları Q4_K_M model ve llama.cpp kullanılarak tamamlandı.

GPU offload tamamen kapatıldı:

`-ngl 0`

### 15.1 Workload tanımları

#### Long single

- PP = 128
- TG = 128
- Batch = 1

#### Batch workload

- PP = 128
- TG = 64
- Parallel prompts = 4

---

### 15.2 Mac CPU Thread Scaling

#### Long single

| Threads | S_TG | Total time |
|---:|---:|---:|
| 1 | 19.90 tok/s | 10.006 s |
| 2 | 36.93 tok/s | 5.345 s |
| 4 | 54.56 tok/s | 3.555 s |
| 6 | 55.08 tok/s | 3.350 s |
| 8 | **58.56 tok/s** | **3.066 s** |
| 10 | 32.09 tok/s | 5.127 s |

8 thread en sağlam operating point'tir.

10 thread seviyesinde hem throughput düşüşü hem yüksek tekrar varyasyonu görülmüştür.

#### Batch

| Threads | S_TG | Total time |
|---:|---:|---:|
| 1 | 30.79 tok/s | 22.833 s |
| 2 | 56.14 tok/s | 11.976 s |
| 4 | 88.50 tok/s | 7.172 s |
| 6 | 89.23 tok/s | 6.825 s |
| 8 | **94.87 tok/s** | 6.282 s |
| 10 | 84.94 tok/s | **5.988 s** |

Generation throughput açısından 8 thread daha sağlam noktadır.

10 thread toplam latency açısından düşük sonuç verse de tekrarlar arasında daha yüksek
varyasyon gözlenmiştir.

---

### 15.3 Mac CPU Power

Power ölçümü model load tamamlandıktan sonra başlatılmıştır.

Power boundary:

- CPU + GPU + ANE combined SoC power

#### Long single

| Threads | Throughput | Power | J/token |
|---:|---:|---:|---:|
| 1 | 20.08 | 8.89 W | 0.6934 |
| 4 | 53.98 | 24.49 W | 0.6936 |
| 8 | **59.60** | 23.29 W | **0.5875** |

#### Batch

| Threads | Throughput | Power | J/token |
|---:|---:|---:|---:|
| 1 | 30.14 | 10.75 W | 0.9920 |
| 4 | 81.22 | 21.15 W | 0.6462 |
| 8 | **98.74** | 22.93 W | **0.5475** |

Mac'te 8 thread hem long_single hem batch workload için en iyi measured
throughput/energy operating point olmuştur.

---

### 15.4 MSI CPU Thread Scaling

#### Long single

| Threads | S_TG | Total time |
|---:|---:|---:|
| 1 | 9.13 tok/s | 15.285 s |
| 2 | 15.17 tok/s | 9.846 s |
| 4 | 18.23 tok/s | 7.994 s |
| 8 | **19.09 tok/s** | **7.617 s** |
| 16 | 17.89 tok/s | 8.006 s |

Long single için throughput optimumu 8 thread'dir.

#### Batch

| Threads | S_TG | Total time |
|---:|---:|---:|
| 1 | 6.43 tok/s | 42.479 s |
| 2 | 16.32 tok/s | 18.887 s |
| 4 | 28.83 tok/s | 11.533 s |
| 8 | 44.08 tok/s | 7.961 s |
| 16 | **48.85 tok/s** | **7.361 s** |

Batch workload 16 thread'e kadar ölçeklenmeye devam etmiştir.

---

### 15.5 MSI CPU Power

CPU Package power, LibreHardwareMonitor üzerinden ölçülmüştür.

#### Long single

| Threads | Throughput | Package Power | J/token |
|---:|---:|---:|---:|
| 1 | 9.20 | 21.85 W | 2.5489 |
| 4 | 18.17 | 33.84 W | **2.1102** |
| 8 | **19.27** | 40.20 W | 2.3761 |
| 16 | 18.03 | 40.48 W | 2.5332 |

Bu workload scheduler açısından önemli bir trade-off göstermektedir:

- throughput optimumu: 8 thread
- energy/token optimumu: 4 thread

8 thread yaklaşık %6 daha yüksek throughput sağlarken token başına enerji maliyeti
4 thread'e göre yaklaşık %12.6 daha yüksektir.

#### Batch

| Threads | Throughput | Package Power | J/token |
|---:|---:|---:|---:|
| 4 | 29.41 | 33.31 W | 1.5301 |
| 8 | 43.92 | 37.60 W | 1.1788 |
| 16 | **49.05** | 38.45 W | **1.0964** |

Batch workload için 16 thread hem throughput hem measured energy/token açısından
en iyi noktadır.

### MSI Batch/1t variability

Batch/1t workload farklı session'larda belirgin state sensitivity göstermiştir.

Generation throughput gözlemleri:

- ilk scaling median: 6.43 tok/s
- ilk power session median: 7.92 tok/s
- tekrar power session son üç medianı: 9.90 tok/s

Tekrar session'ındaki son üç run:

- 9.55 tok/s
- 9.90 tok/s
- 9.92 tok/s

kendi içinde stabil olmasına rağmen önceki session'larla aynı seviyede değildir.

Bu nedenle batch/1t power değeri canonical scheduler operating point olarak
kullanılmamıştır ve sonuç session-state sensitive olarak işaretlenmiştir.

---

## 16. Device-Specific Batch Behavior

Bu characterization aşamasının en önemli bulgularından biri batch optimumunun cihazdan bağımsız olmamasıdır.

### Mac M4

Best observed operating point:

- Batch = 2

B4 workload GPU/resource scaling açısından verimsizleşmektedir.

### RTX 3050

Best throughput operating point:

- Batch = 4

RTX parallel workload kapasitesinden daha uzun süre faydalanmaktadır.

Bu nedenle global olarak sabit bir batch size kullanmak optimal değildir.

---

## 17. Measurement Boundary Caveat

Mac ve RTX absolute energy değerleri doğrudan cihazlar arası winner belirlemek için kullanılmamalıdır.

Mac power boundary:

- CPU
- GPU
- ANE
- combined SoC power

RTX power boundary:

- NVIDIA GPU board only

Bu nedenle enerji sonuçları öncelikle aynı platform içerisindeki configuration trendlerini analiz etmek için kullanılır.

---

## 18. Scheduler Implications

Hard/Text sonuçları scheduler'ın en az şu özellikleri dikkate alması gerektiğini göstermektedir:

- device
- backend
- model precision
- quantization
- context length
- output length
- batch size
- memory capacity
- sustained throughput
- power
- energy/token
- thermal state
- utilization
- feasibility / OOM status

Önemli karar örneği:

Mac M4 için Qwen3 Q4 workload:

Batch 2 tercih edilebilir.

RTX 3050 için aynı workload:

Batch 4 throughput açısından daha uygun olabilir.

Dolayısıyla scheduler cihaz-spesifik operating point seçmelidir.

---

## 19. Native vs Quantized Execution

Native FP16 sonuçları:

- Mac: feasible
- MSI RTX 3050: OOM

Controlled Q4_K_M sonuçları:

- Mac: feasible
- MSI RTX 3050: feasible

Quantization yalnızca performance optimization değil, aynı zamanda heterogeneous-device feasibility mekanizmasıdır.

Bu sonuç daha sonraki expert-placement ve MoE scheduling aşaması için önemlidir.

---

## 20. Main Findings

1. Native FP16 Qwen3 Mac üzerinde çalışırken RTX 3050 4 GB üzerinde model-load aşamasında OOM oluştu.
2. Q4_K_M quantization iki cihaz üzerinde kontrollü cross-device karşılaştırmayı mümkün kıldı.
3. RTX controlled Q4 sweep'te 18/18 workload'da daha yüksek generation throughput verdi.
4. Mac accelerator tarafında Batch 2 belirgin sweet spot oluştururken Batch 4'te throughput geriledi.
5. RTX Batch 4'e kadar throughput scaling göstermeye devam etti.
6. RTX sustained throughput yaklaşık 60 saniyede yalnızca yaklaşık %4 azaldı.
7. RTX representative workload'larda yaklaşık 59 W ve %96-97 GPU utilization seviyesine ulaştı.
8. Senkronize Mac power ölçümünde Medium profil 113.38 tok/s ve 0.2063 J/token ile en dengeli accelerator operating point oldu.
9. Mac CPU tarafında 8 thread hem long_single hem batch workload için güçlü ortak throughput/energy noktasıdır.
10. MSI long_single workload'da throughput optimumu 8 thread iken energy/token optimumu 4 thread'dir.
11. MSI batch workload'da 16 thread hem throughput hem energy/token açısından en iyi measured noktadır.
12. MSI batch/1t workload session-state sensitivity göstermiştir ve canonical scheduler noktası olarak kullanılmamıştır.
13. Thread optimumu workload'a ve optimization objective'e bağlıdır; tek bir global CPU thread optimumu yoktur.
14. Scheduler cihaz, workload shape, batch, memory, latency, energy ve runtime state'i birlikte değerlendirmelidir.

---

## 21. Hard/Text Completion Checklist

- [x] Mac native FP16 feasibility
- [x] MSI native FP16 feasibility
- [x] Native OOM characterization
- [x] Controlled Q4_K_M runtime selected
- [x] Mac Q4 feasibility
- [x] MSI CUDA Q4 feasibility
- [x] Mac full workload matrix
- [x] MSI full workload matrix
- [x] Mac vs MSI controlled comparison
- [x] RTX sustained validation
- [x] RTX NVML power
- [x] Mac synchronized accelerator power
- [x] Mac CPU long_single thread scaling
- [x] Mac CPU batch thread scaling
- [x] Mac CPU power/energy
- [x] MSI CPU long_single thread scaling
- [x] MSI CPU batch thread scaling
- [x] MSI CPU power/energy
- [x] CPU workload-dependent thread optimum characterization
- [x] MSI batch/1t session variability characterization
- [x] Monitoring overhead characterization
- [x] Measurement boundary caveat
- [x] Scheduler implications

**Hard / Text: COMPLETE**

---

## 22. Next Step

Text characterization status:

- Easy: COMPLETE
- Medium: COMPLETE
- Hard: COMPLETE

Vision characterization status:

- Vision Easy / ResNet50: COMPLETE
- Vision Medium / ConvNeXt-Base: NEXT
- Vision Hard / ConvNeXt-Large: pending

Bir sonraki ana aşama:

**Vision Medium — ConvNeXt-Base characterization**

Vision Medium için aynı veri-first metodoloji korunacaktır:

- feasibility
- CPU thread scaling
- resolution × batch sweep
- FP32 / FP16 accelerator comparison
- memory behavior
- sustained performance
- power / energy
- Mac vs MSI crossover
- scheduler implications
