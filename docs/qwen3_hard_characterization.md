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

## 14. Mac Power

Measurement:

- powermetrics
- 100 ms sampling
- 3 repeats
- Power boundary: CPU + GPU + ANE combined SoC power

| Profile | Throughput | Power | J/run | J/token |
|---|---:|---:|---:|---:|
| Light | 78.56 | 12.76 W | 12.30 | 0.1921 |
| Medium | 106.72 | 13.47 W | 46.85 | 0.1830 |
| Heavy | 87.00 | 13.52 W | 138.03 | 0.2696 |

Mac üzerinde Medium profil hem en yüksek sustained throughput hem de en düşük energy/token değerini verdi.

Heavy profile daha yüksek batch kullanmasına rağmen throughput düştü ve J/token kötüleşti.

---

## 15. Device-Specific Batch Behavior

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

## 16. Measurement Boundary Caveat

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

## 17. Scheduler Implications

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

## 18. Native vs Quantized Execution

Native FP16 sonuçları:

- Mac: feasible
- MSI RTX 3050: OOM

Controlled Q4_K_M sonuçları:

- Mac: feasible
- MSI RTX 3050: feasible

Quantization yalnızca performance optimization değil, aynı zamanda heterogeneous-device feasibility mekanizmasıdır.

Bu sonuç daha sonraki expert-placement ve MoE scheduling aşaması için önemlidir.

---

## 19. Main Findings

1. Native FP16 Qwen3 Mac üzerinde çalışırken RTX 3050 4 GB üzerinde OOM oluştu.
2. Q4_K_M quantization her iki cihazda kontrollü karşılaştırmayı mümkün kıldı.
3. RTX controlled Q4 sweep'te 18/18 workload'da daha yüksek generation throughput verdi.
4. Mac Batch 2'de optimum noktaya ulaşıp Batch 4'te geriledi.
5. RTX Batch 4'e kadar throughput scaling göstermeye devam etti.
6. RTX sustained throughput yaklaşık 60 saniye içinde yalnızca yaklaşık %4 azaldı.
7. RTX workload sırasında yaklaşık 59 W ve %96–97 GPU utilization seviyesine ulaştı.
8. Mac Medium profile en iyi sustained throughput ve J/token değerini verdi.
9. RTX Medium ve Heavy enerji/token değerleri neredeyse aynıydı.
10. Scheduler cihaz, workload shape, memory ve batch behavior'ı birlikte değerlendirmelidir.

---

## 20. Hard/Text Completion Checklist

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
- [x] Mac powermetrics power
- [x] Monitoring overhead characterization
- [x] Measurement boundary caveat
- [x] Scheduler implications

**Hard / Text: COMPLETE**

---

## 21. Next Step

Text characterization status:

- Easy: COMPLETE
- Medium: COMPLETE
- Hard: COMPLETE

Next major stage:

**Vision characterization**

Planned progression:

- Vision Easy
- Vision Medium
- Vision Hard

The same data-first methodology will be reused before building the resource-aware scheduler.