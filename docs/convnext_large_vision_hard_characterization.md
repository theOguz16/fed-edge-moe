# ConvNeXt-Large Vision Hard Karakterizasyonu

## 1. Amaç

Bu çalışma, Vision Hard workload olarak seçilen ConvNeXt-Large modelinin Apple M4 ve NVIDIA RTX 3050 Laptop GPU üzerinde inference davranışını karakterize eder.

Amaç, ileride geliştirilecek resource-aware scheduler için aşağıdaki değişkenlerin etkisini ölçmektir:

- cihaz ve backend,
- input resolution,
- batch büyüklüğü,
- FP32 / FP16 precision,
- CPU thread sayısı,
- accelerator memory kullanımı,
- kısa ve sustained throughput,
- güç tüketimi,
- energy/image,
- monitoring overhead,
- cross-device crossover.

Bu aşamada inference local çalıştırılmıştır. Network ve communication maliyetleri distributed multi-device aşamasına bırakılmıştır.

---

## 2. Model

- Model: ConvNeXt-Large
- Parametre sayısı: yaklaşık 197.8M
- Framework: PyTorch / torchvision
- Weights: `ConvNeXt_Large_Weights.DEFAULT`
- Görev: image classification inference
- Training: yok
- Input: synthetic image tensor
- Preprocessing: ölçüm dışında

Sonuçlar end-to-end image pipeline yerine model inference davranışını temsil etmektedir.

---

## 3. Donanım

### Mac

- Apple M4
- 10 logical CPU
- MPS backend
- unified memory

CPU thread noktaları:

`1, 2, 4, 6, 8, 10`

### MSI

- Intel Core i7-11800H
- 8 physical / 16 logical CPU
- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB VRAM
- CUDA backend

CPU thread noktaları:

`1, 2, 4, 8, 16`

---

## 4. Ana Workload Grid

Accelerator sweep:

### Resolution

- 160×160
- 224×224
- 320×320

### Batch

- B1
- B4
- B16

### Precision

- FP32
- FP16

Toplam:

`3 resolution × 3 batch × 2 precision = 18 configuration`

Representative sustained/power profilleri:

- Light: 160×160 / B1
- Medium: 224×224 / B4
- Heavy: 320×320 / B16

---

# 5. Mac Feasibility

224×224 / B1:

| Backend | Precision | Throughput | Latency |
|---|---:|---:|---:|
| CPU | FP32 | 1.27 img/s | 785.36 ms |
| MPS | FP32 | 24.75 img/s | 40.41 ms |
| MPS | FP16 | 26.10 img/s | 38.31 ms |

MPS FP32, CPU FP32'den yaklaşık 19.5× daha yüksek throughput sağlamıştır.

MPS FP16, bu küçük B1 feasibility workload'unda FP32'den yaklaşık %5.5 daha hızlıdır.

MPS current allocated memory:

- FP32: 755.0 MB
- FP16: 381.5 MB

FP16 allocation yaklaşık %49.5 azalmıştır.

---

# 6. Mac MPS Full Sweep

## FP32

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 38.17 | 55.08 | 61.40 |
| 224 | 24.99 | 30.07 | 30.57 |
| 320 | 13.94 | 15.46 | 15.20 |

## FP16

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 46.02 | 63.65 | 74.64 |
| 224 | 28.01 | 35.79 | 37.66 |
| 320 | 16.04 | 18.65 | 18.66 |

FP16, dokuz workload noktasının tamamında FP32'den daha hızlıdır.

Yaklaşık FP16 throughput kazançları:

- 160/B1: +%20.6
- 160/B4: +%15.6
- 160/B16: +%21.6
- 224/B1: +%12.1
- 224/B4: +%19.0
- 224/B16: +%23.2
- 320/B1: +%15.1
- 320/B4: +%20.6
- 320/B16: +%22.8

320/B4 ve 320/B16 FP16 throughput değerlerinin neredeyse aynı olması bu workload'da batch scaling'in B4 civarında saturation'a yaklaştığını göstermektedir.

---

# 7. Mac MPS Memory

FP32 allocation yaklaşık:

`755–774 MB`

FP16 allocation yaklaşık:

`381–391 MB`

Heavy workload 320/B16:

- FP32: 773.5 MB
- FP16: 390.8 MB

FP16 yaklaşık %49.5 daha düşük allocation kullanmıştır.

Bu sonuç precision'ın yalnızca latency değil memory pressure açısından da scheduler değişkeni olması gerektiğini göstermektedir.

---

# 8. Mac CPU Batch Davranışı

224×224 / 4 thread:

| Batch | Throughput |
|---:|---:|
| B1 | 0.99 |
| B2 | 0.89 |
| B4 | 1.46 |
| B8 | **1.51** |

En yüksek kısa-sweep throughput B8'de elde edilmiştir.

B4 ve B8 birbirine oldukça yakın olduğundan batching kazancı B4 sonrasında sınırlanmaktadır.

Representative Mac CPU batch workload:

`224×224 / B8`

---

# 9. Mac CPU Thread Scaling

## High-resolution single — 320/B1

| Threads | Throughput | Latency |
|---:|---:|---:|
| 1 | 0.39 | 2545.14 ms |
| 2 | 0.59 | 1691.86 ms |
| 4 | **0.82** | **1214.63 ms** |
| 6 | 0.62 | 1617.61 ms |
| 8 | 0.64 | 1572.03 ms |
| 10 | 0.70 | 1433.10 ms |

Kısa-sweep throughput optimumu 4 thread'dir.

## Batch workload — 224/B8

| Threads | Throughput | Latency |
|---:|---:|---:|
| 1 | 0.89 | 8982.84 ms |
| 2 | 1.42 | 5634.13 ms |
| 4 | 1.58 | 5064.11 ms |
| 6 | 1.49 | 5382.11 ms |
| 8 | **2.09** | **3822.03 ms** |
| 10 | 1.98 | 4036.07 ms |

Batch workload için kısa-sweep optimumu 8 thread'dir.

Dolayısıyla ConvNeXt-Large Mac CPU üzerinde tek bir global thread optimumu göstermemektedir.

---

# 10. Mac CPU Power

## High-resolution single — 320/B1

| Threads | Throughput | Power | J/image |
|---:|---:|---:|---:|
| 1 | 0.42 | 6.36 W | **15.2796** |
| 4 | **0.76** | 13.63 W | 18.0103 |
| 8 | 0.73 | 13.36 W | 18.1826 |
| 10 | 0.62 | 12.86 W | 20.6838 |

Bu workload'da:

- throughput optimumu: 4 thread
- energy/image optimumu: 1 thread

En hızlı configuration ile en enerji-verimli configuration farklıdır.

## Batch workload — 224/B8

| Threads | Throughput | Power | J/image |
|---:|---:|---:|---:|
| 1 | 0.91 | 5.78 W | **6.3259** |
| 4 | 1.51 | 12.00 W | 7.9847 |
| 8 | **2.13** | 15.19 W | 7.1433 |
| 10 | 2.05 | 15.18 W | 7.3948 |

Batch workload'da:

- throughput optimumu: 8 thread
- energy/image optimumu: 1 thread

8 thread yaklaşık 2.34× daha yüksek throughput sağlarken image başına enerji maliyeti 1 thread'e göre yükselmektedir.

Bu, doğrudan latency-energy trade-off oluşturmaktadır.

---

# 11. Mac MPS Sustained Power Metodolojisi

Representative workload'lar için paired A/B protokolü kullanılmıştır:

- A: no-monitor sustained throughput
- B: `powermetrics` monitored sustained throughput
- yaklaşık 10 saniye workload
- 500 ms power sampling
- FP32 ve FP16
- Light / Medium / Heavy profilleri

İlk iki power session arasında özellikle FP16 energy/image değerlerinde session-state variability görülmüştür.

Bu nedenle Vision Hard Mac MPS power sonuçları üç ayrı session ile ölçülmüş ve canonical değerler session medyanı olarak hesaplanmıştır.

Bu yaklaşım tek bir operating-state ölçümünün canonical sonuç olarak kullanılmasını önlemiştir.

---

# 12. Mac MPS Canonical Sustained Sonuçları

## FP32

| Profile | No-monitor throughput | Power | J/image | Allocation |
|---|---:|---:|---:|---:|
| Light | 39.71 | 14.28 W | 0.3549 | 754.7 MB |
| Medium | 28.23 | 15.80 W | 0.5434 | 756.8 MB |
| Heavy | 14.27 | 14.83 W | 1.0258 | 773.4 MB |

## FP16

| Profile | No-monitor throughput | Power | J/image | Allocation |
|---|---:|---:|---:|---:|
| Light | 47.23 | 13.67 W | 0.2917 | 381.4 MB |
| Medium | 35.41 | 17.09 W | 0.5092 | 382.4 MB |
| Heavy | 17.73 | 17.29 W | 0.9593 | 390.7 MB |

FP16 throughput avantajı:

- Light: yaklaşık +%18.9
- Medium: yaklaşık +%25.4
- Heavy: yaklaşık +%24.2

FP16 energy/image avantajı:

- Light: yaklaşık %17.8 daha düşük
- Medium: yaklaşık %6.3 daha düşük
- Heavy: yaklaşık %6.5 daha düşük

Medium ve Heavy workload'larda FP16 instantaneous power daha yüksek olmasına rağmen daha yüksek throughput sayesinde toplam energy/image daha düşüktür.

Bu sonuç scheduler açısından power ve energy metriklerinin birbirinin yerine kullanılamayacağını göstermektedir.

---

# 13. Mac MPS Session Variability

Üç power session arasında bazı workload'larda gözle görülür operating-state farklılıkları oluşmuştur.

Örnek:

FP16 Medium J/image:

- Session 1: 0.6579
- Session 2: 0.4654
- Session 3: 0.5092

Canonical median:

`0.5092 J/image`

FP16 Light:

- Session 1: 0.2864
- Session 2: 0.3758
- Session 3: 0.2917

Canonical median:

`0.2917 J/image`

Bu nedenle sustained characterization'da session-level replication önemlidir.

---

# 14. MSI Feasibility

224×224 / B1:

| Backend | Precision | Throughput | Latency |
|---|---:|---:|---:|
| CPU | FP32 | 4.11 img/s | 243.52 ms |
| CUDA | FP32 | 32.76 img/s | 30.53 ms |
| CUDA | FP16 | 63.03 img/s | 15.87 ms |

CUDA FP32 CPU'dan yaklaşık 8× daha hızlıdır.

CUDA FP16, feasibility testinde CUDA FP32'ye göre yaklaşık 1.92× throughput sağlamıştır.

Memory:

- FP32 allocation: 764.0 MB
- FP32 peak: 802.4 MB
- FP16 allocation: 386.1 MB
- FP16 peak: 399.3 MB

---

# 15. MSI CUDA Full Sweep

## FP32

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 48.47 | 71.42 | 81.91 |
| 224 | 31.90 | 39.25 | 40.45 |
| 320 | 17.53 | 20.13 | 20.52 |

## FP16

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 98.96 | 178.75 | 233.47 |
| 224 | 74.28 | 98.71 | 125.17 |
| 320 | 44.47 | 52.37 | 63.43 |

FP16 / FP32 throughput oranları yaklaşık:

- 160/B1: 2.04×
- 160/B4: 2.50×
- 160/B16: 2.85×
- 224/B1: 2.33×
- 224/B4: 2.51×
- 224/B16: 3.09×
- 320/B1: 2.54×
- 320/B4: 2.60×
- 320/B16: 3.09×

Model ve batch büyüdükçe FP16 avantajı belirgin biçimde artmaktadır.

---

# 16. MSI CUDA Memory

Heavy workload 320/B16:

- FP32 peak: 1534.2 MB
- FP16 peak: 771.0 MB

FP16 peak memory yaklaşık %50 azalmıştır.

18 configuration'ın tamamı RTX 3050 Laptop GPU'nun 4 GB VRAM kapasitesi içerisinde başarıyla çalışmıştır.

Vision Hard ConvNeXt-Large için bu grid içerisinde OOM gözlenmemiştir.

---

# 17. MSI CPU Batch Davranışı

224×224 / 16 thread:

| Batch | Throughput |
|---:|---:|
| B1 | 4.37 |
| B2 | **5.05** |
| B4 | 4.17 |
| B8 | 4.26 |

MSI CPU'da batching yalnız B2'ye kadar fayda sağlamıştır.

Representative MSI CPU batch workload:

`224×224 / B2`

Bu sonuç Mac CPU'daki B8 optimumundan farklıdır ve batch optimumunun cihaz bağımlı olduğunu göstermektedir.

---

# 18. MSI CPU Thread Scaling

## High-resolution single — 320/B1

| Threads | Throughput | Latency |
|---:|---:|---:|
| 1 | 0.62 | 1608.33 ms |
| 2 | 1.18 | 844.54 ms |
| 4 | 1.69 | 592.96 ms |
| 8 | 1.96 | 510.61 ms |
| 16 | **2.25** | **443.57 ms** |

Short-sweep optimumu 16 thread'dir.

## Batch workload — 224/B2

| Threads | Throughput | Latency |
|---:|---:|---:|
| 1 | 1.41 | 1422.56 ms |
| 2 | 2.38 | 841.29 ms |
| 4 | 3.47 | 576.34 ms |
| 8 | **5.50** | **363.96 ms** |
| 16 | 5.22 | 383.34 ms |

Short-sweep optimumu 8 thread'dir.

---

# 19. MSI CPU Sustained Power

LibreHardwareMonitor üzerinden CPU Package power ölçülmüştür.

Representative thread noktaları:

`1, 8, 16`

## High-resolution single — 320/B1

| Threads | No-monitor | Monitored | Power | J/image |
|---:|---:|---:|---:|---:|
| 1 | 0.64 | 0.63 | 20.06 W | 30.9745 |
| 8 | 2.24 | 2.24 | 37.20 W | 16.6535 |
| 16 | **2.25** | **2.25** | 36.59 W | **16.3509** |

Sustained workload'da 16 thread hem throughput hem energy/image açısından en iyi measured noktadır.

## Batch workload — 224/B2

| Threads | No-monitor | Monitored | Power | J/image |
|---:|---:|---:|---:|---:|
| 1 | 1.03 | 1.04 | 19.33 W | 18.6328 |
| 8 | 3.72 | 3.76 | 32.33 W | 8.6134 |
| 16 | **4.55** | **4.60** | 36.73 W | **7.9867** |

Kısa testte 8 thread en hızlıyken sustained workload'da 16 thread öne geçmiştir.

Bu, optimum thread sayısının benchmark duration ve operating state'e bağlı olduğunu göstermektedir.

---

# 20. MSI CUDA Sustained Power

NVML paired A/B ölçümü:

- yaklaşık 10 saniye
- 500 ms polling
- 3 repeats
- power boundary: NVIDIA GPU board power

Idle:

- median: 13.68 W
- mean: 11.94 W

## FP32

| Profile | No-monitor | Monitored | Power | Util | J/image |
|---|---:|---:|---:|---:|---:|
| Light | 47.63 | 47.39 | 57.15 W | 95.0% | 1.1996 |
| Medium | 37.30 | 37.29 | 57.42 W | 95.2% | 1.5397 |
| Heavy | 19.66 | 19.69 | 57.89 W | 96.0% | 2.9397 |

## FP16

| Profile | No-monitor | Monitored | Power | Util | J/image |
|---|---:|---:|---:|---:|---:|
| Light | 102.79 | 97.97 | 57.07 W | 74.0% | 0.5825 |
| Medium | 94.10 | 94.17 | 57.28 W | 95.0% | 0.6081 |
| Heavy | 60.91 | 60.93 | 57.58 W | 95.2% | 0.9450 |

---

# 21. MSI CUDA Precision ve Enerji

FP16 / FP32 sustained throughput:

- Light: yaklaşık 2.16×
- Medium: yaklaşık 2.52×
- Heavy: yaklaşık 3.10×

FP16 energy/image iyileşmesi:

### Light

- FP32: 1.1996 J/image
- FP16: 0.5825 J/image
- yaklaşık %51.4 daha düşük

### Medium

- FP32: 1.5397 J/image
- FP16: 0.6081 J/image
- yaklaşık %60.5 daha düşük

### Heavy

- FP32: 2.9397 J/image
- FP16: 0.9450 J/image
- yaklaşık %67.9 daha düşük

Workload büyüdükçe FP16 hem throughput hem energy/image açısından daha avantajlı hale gelmektedir.

---

# 22. Monitoring Overhead

MSI CUDA medium ve heavy workload'larda NVML monitoring etkisi çok düşüktür:

- FP32 Medium: yaklaşık -%0.03
- FP32 Heavy: yaklaşık +%0.15
- FP16 Medium: yaklaşık +%0.09
- FP16 Heavy: yaklaşık +%0.08

FP16 Light ise daha monitoring-sensitive davranmıştır:

`median delta ≈ -%4.69`

Bir repeat'te yaklaşık -%9.69 sapma görülmüştür.

Bu nedenle küçük accelerator workload'larında monitoring overhead ayrıca kontrol edilmelidir.

---

# 23. Mac ↔ MSI Short-Sweep Crossover

Toplam 18 accelerator configuration karşılaştırılmıştır.

Sonuç:

- Mac wins: 0
- MSI wins: 18

## FP32

MSI avantajı yaklaşık:

`1.26× – 1.35×`

## FP16

MSI avantajı yaklaşık:

`2.15× – 3.40×`

En büyük kısa-sweep farkı:

`320/B16 FP16`

- Mac: 18.66 img/s
- MSI: 63.43 img/s
- MSI: yaklaşık 3.40×

---

# 24. Mac ↔ MSI Sustained Crossover

## FP32

| Profile | Mac | MSI | Winner |
|---|---:|---:|---|
| Light | 39.71 | 47.63 | MSI 1.20× |
| Medium | 28.23 | 37.30 | MSI 1.32× |
| Heavy | 14.27 | 19.66 | MSI 1.38× |

## FP16

| Profile | Mac | MSI | Winner |
|---|---:|---:|---|
| Light | 47.23 | 102.79 | MSI 2.18× |
| Medium | 35.41 | 94.10 | MSI 2.66× |
| Heavy | 17.73 | 60.91 | MSI 3.44× |

Sustained sonuç:

- Mac wins: 0
- MSI wins: 6

FP16 workload yoğunluğu arttıkça discrete GPU avantajı belirgin biçimde büyümektedir.

---

# 25. Vision Seviyeleri Arasında Crossover Eğilimi

Önceki Vision Easy ve Vision Medium karakterizasyonlarıyla birlikte bakıldığında genel eğilim şudur:

- küçük modeller ve hafif workload'larda Apple M4 bazı accelerator senaryolarında rekabetçi olabilir,
- model ve workload yoğunluğu büyüdükçe RTX avantajı artmaktadır,
- Vision Medium'da MSI tüm representative sustained noktaları kazanmıştır,
- Vision Hard'da MSI hem 18/18 short-sweep hem 6/6 sustained noktayı kazanmıştır.

Bu nedenle device selection yalnız input size veya batch'e göre değil, model computational intensity ile birlikte yapılmalıdır.

---

# 26. Scheduler Açısından Ana Bulgular

### 1. Model büyüklüğü cihaz seçimini değiştirir

ConvNeXt-Large seviyesinde RTX 3050 bütün accelerator comparison noktalarında Mac MPS'ten daha yüksek throughput sağlamıştır.

### 2. Precision birinci sınıf scheduler değişkenidir

FP16:

- memory kullanımını yaklaşık yarıya indirir,
- özellikle RTX üzerinde throughput'u 2–3× artırabilir,
- energy/image değerini ciddi biçimde düşürebilir.

### 3. Precision kazancı workload büyüklüğüne bağlıdır

Küçük workload ile heavy workload aynı FP16 kazancını göstermemektedir.

RTX üzerinde FP16 avantajı batch ve computational intensity arttıkça belirgin biçimde büyümüştür.

### 4. Batch optimumu cihaz bağımlıdır

Mac CPU:

`224/B8`

MSI CPU:

`224/B2`

aynı model için farklı batch optimumları göstermiştir.

### 5. Thread optimumu workload bağımlıdır

Mac:

- highres short: 4 thread
- batch short: 8 thread

MSI:

- highres short: 16 thread
- batch short: 8 thread
- batch sustained: 16 thread

Tek global thread optimumu yoktur.

### 6. Benchmark duration scheduler kararını etkileyebilir

MSI batch workload'da kısa benchmark 8 thread'i, sustained workload ise 16 thread'i tercih etmiştir.

Bu nedenle scheduler profili yalnız kısa microbenchmark değerlerine dayanmamalıdır.

### 7. Minimum power, minimum energy değildir

Mac FP16 Medium ve Heavy workload'larda instantaneous power daha yüksek olmasına rağmen daha hızlı execution nedeniyle J/image daha düşük çıkmıştır.

Power ve energy ayrı objective olarak ele alınmalıdır.

### 8. Session-state variability ölçülmelidir

Mac MPS power sonuçlarında ayrı session'lar arasında anlamlı enerji farkları gözlenmiştir.

Tek-session profiling yanlış configuration tercihine yol açabilir.

---

# 27. Energy Measurement Boundary

Mac MPS:

`CPU + GPU + ANE combined SoC power`

MSI CUDA:

`NVIDIA GPU board power`

MSI CPU:

`CPU Package power`

Bu boundary'ler birbirine eşit değildir.

Bu nedenle Mac ve MSI arasında absolute J/image değerlerinden doğrudan enerji winner çıkarılmamalıdır.

Enerji metriği esas olarak aynı cihaz içindeki configuration seçimleri için kullanılmalıdır.

---

# 28. Sınırlamalar

- Synthetic input kullanılmıştır.
- Image decode ve preprocessing ölçüm dışında bırakılmıştır.
- Host-device transfer ayrı metrik olarak ölçülmemiştir.
- Accuracy / quality ölçülmemiştir.
- FP16 ve FP32'nin aynı prediction quality sağladığı varsayılmamalıdır.
- Network / communication maliyeti dahil değildir.
- Mac MPS current allocation toplam unified-memory footprint'i tam temsil etmez.
- MSI NVML power GPU board boundary'sidir.
- MSI CPU power LibreHardwareMonitor CPU Package boundary'sidir.
- Thermal ve operating-state etkileri tamamen ortadan kaldırılamaz.
- Mac power sonuçlarında bu riski azaltmak için üç-session canonical median kullanılmıştır.
- Short ve sustained benchmark sonuçları aynı optimum configuration'ı vermeyebilir.

---

# 29. Vision Hard Completion Checklist

- [x] Mac feasibility
- [x] Mac MPS FP32 sweep
- [x] Mac MPS FP16 sweep
- [x] Mac MPS memory
- [x] Mac CPU batch characterization
- [x] Mac CPU thread scaling
- [x] Mac CPU power
- [x] Mac MPS sustained profiling
- [x] Mac MPS three-session validation
- [x] Mac MPS canonical power dataset
- [x] MSI feasibility
- [x] MSI CUDA FP32 sweep
- [x] MSI CUDA FP16 sweep
- [x] MSI CUDA memory
- [x] MSI CPU batch characterization
- [x] MSI CPU thread scaling
- [x] MSI CPU sustained power
- [x] MSI CUDA sustained profiling
- [x] MSI CUDA NVML power
- [x] Monitoring A/B validation
- [x] Mac ↔ MSI 18-point short crossover
- [x] Mac ↔ MSI 6-point sustained crossover
- [x] Scheduler implications
- [x] Measurement-boundary caveats

**Vision Hard: COMPLETE**

---

# 30. Vision Characterization Status

- Vision Easy — ResNet50: COMPLETE
- Vision Medium — ConvNeXt-Base: COMPLETE
- Vision Hard — ConvNeXt-Large: COMPLETE

Vision workload characterization phase is complete.

---

# 31. Next Step

Bir sonraki ana aşama:

**Unified Characterization Dataset**

Text ve Vision benchmark sonuçları ortak scheduler-oriented schema altında birleştirilecektir.

Hedef feature alanları:

- modality
- model
- model size
- device
- backend
- precision
- resolution / prompt size
- batch
- threads
- workload class
- throughput
- latency
- memory
- power
- energy/work item
- sustained / short flag
- monitoring state
- device characteristics

Bu dataset daha sonra:

1. crossover analysis,
2. resource-aware scheduling policy,
3. latency-energy-memory trade-off modeli,
4. MoE expert placement ve scheduling

aşamalarının girdisi olacaktır.