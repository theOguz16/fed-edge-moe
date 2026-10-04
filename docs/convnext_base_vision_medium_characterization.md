# ConvNeXt-Base Vision Medium Karakterizasyonu

## 1. Amaç

Bu çalışma, Vision Medium workload olarak seçilen ConvNeXt-Base modelinin Mac M4 ve MSI RTX 3050 Laptop GPU üzerindeki inference davranışını karakterize eder.

Amaç gelecekte geliştirilecek resource-aware scheduler için şu değişkenlerin etkisini ölçmektir:

- cihaz ve backend,
- input resolution,
- batch büyüklüğü,
- FP32 / FP16 precision,
- CPU thread sayısı,
- bellek kullanımı,
- sustained throughput,
- güç tüketimi,
- enerji/image,
- monitoring overhead.

Bu aşamada inference local çalıştırılmıştır. Network ve communication maliyetleri distributed / multi-device aşamasına bırakılmıştır.

---

## 2. Model

- Model: ConvNeXt-Base
- Parametre sayısı: yaklaşık 88.6M
- Framework: PyTorch / torchvision
- Weights: `ConvNeXt_Base_Weights.DEFAULT`
- Görev: image classification inference
- Training: yok
- Input: synthetic image tensor
- Preprocessing: ölçüm dışında

Dolayısıyla sonuçlar end-to-end image pipeline yerine model inference davranışını temsil etmektedir.

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

## 4. Workload Grid

Ana accelerator sweep:

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

Representative power profilleri:

- Light: 160×160 / B1
- Medium: 224×224 / B4
- Heavy: 320×320 / B16

---

# 5. Mac Feasibility

224×224 / B1:

| Backend | Precision | Throughput | Latency |
|---|---:|---:|---:|
| CPU | FP32 | 1.99 img/s | 501.41 ms |
| MPS | FP32 | 43.65 img/s | 22.91 ms |
| MPS | FP16 | 51.54 img/s | 19.40 ms |

MPS FP32, CPU FP32'den yaklaşık 21.9× daha yüksek throughput sağlamıştır.

MPS FP16 ise MPS FP32'ye göre yaklaşık %18 daha hızlıdır.

MPS current allocated memory:

- FP32: 338.6 MB
- FP16: 169.3 MB

FP16 model allocation yaklaşık %50 azalmaktadır.

---

# 6. Mac MPS Full Sweep

## FP32

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 67.77 | 107.20 | 101.93 |
| 224 | 42.63 | 57.70 | 58.94 |
| 320 | 25.91 | 30.19 | 28.89 |

## FP16

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 69.83 | 106.97 | 144.27 |
| 224 | 50.14 | 68.56 | 72.87 |
| 320 | 30.56 | 35.98 | 36.38 |

FP16 avantajı workload büyüdükçe belirginleşmiştir.

Yaklaşık FP16 throughput değişimleri:

- 160/B1: +%3
- 160/B4: yaklaşık aynı
- 160/B16: +%41
- 224/B1: +%18
- 224/B4: +%19
- 224/B16: +%24
- 320/B1: +%18
- 320/B4: +%19
- 320/B16: +%26

FP32'de bazı workload'lar B4 sonrası saturation göstermektedir.

FP16 ise bütün resolution gruplarında B16'ya kadar daha iyi scaling göstermiştir.

---

# 7. Mac MPS Memory Davranışı

FP32 allocation yaklaşık:

`338–357 MB`

FP16 allocation yaklaşık:

`169–179 MB`

seviyesindedir.

FP16, ConvNeXt-Base için yaklaşık yarı model/working allocation sağlamıştır.

Bu sonuç precision'ın yalnız throughput değil memory pressure açısından da scheduler karar değişkeni olması gerektiğini göstermektedir.

---

# 8. Mac CPU Batch Davranışı

224×224 / 4 thread:

| Batch | Throughput |
|---:|---:|
| 1 | 1.73 |
| 2 | 1.57 |
| 4 | 3.13 |
| 8 | **3.71** |

Mac CPU'da batch workload B8'e kadar throughput artışı göstermiştir.

B2 noktasının B1'den düşük olması batch scaling'in monoton olmadığını göstermektedir.

---

# 9. Mac CPU Thread Scaling

## High-resolution single — 320/B1

| Threads | Throughput |
|---:|---:|
| 1 | 0.63 |
| 2 | 0.97 |
| 4 | **1.22** |
| 6 | 1.01 |
| 8 | 1.15 |
| 10 | 1.08 |

## Batch workload — 224/B8

| Threads | Throughput |
|---:|---:|
| 1 | 1.37 |
| 2 | 2.29 |
| 4 | **3.73** |
| 6 | 3.25 |
| 8 | 3.32 |
| 10 | 3.18 |

Her iki workload için kısa sweep throughput optimumu 4 thread'dir.

4 thread sonrası ek thread'ler düzenli performans artışı sağlamamıştır.

---

# 10. Mac CPU Power

## High-resolution single — 320/B1

| Threads | Throughput | Power | J/image |
|---:|---:|---:|---:|
| 1 | 0.63 | 5.49 W | **8.6205** |
| 4 | **1.34** | 13.45 W | 10.0239 |
| 10 | 0.96 | 12.27 W | 12.7357 |

Bu workload'da:

- throughput optimumu: 4 thread
- energy/image optimumu: 1 thread

4 thread yaklaşık 2.1× daha hızlıdır ancak görüntü başına enerji maliyeti 1 thread'den daha yüksektir.

Bu scheduler için doğrudan latency-energy trade-off örneğidir.

## Batch workload — 224/B8

| Threads | Throughput | Power | J/image |
|---:|---:|---:|---:|
| 1 | 1.36 | 6.23 W | 4.5898 |
| 4 | **3.40** | 14.84 W | **4.3583** |
| 10 | 3.18 | 15.10 W | 4.7193 |

Batch workload için 4 thread hem throughput hem energy/image açısından en iyi measured noktadır.

---

# 11. Mac MPS Sustained Power

Paired A/B protokolü kullanılmıştır:

- A: no-monitor sustained throughput
- B: `powermetrics` monitored sustained throughput
- duration: yaklaşık 10 saniye
- sampling: 500 ms
- 3 repeats

Power boundary:

`CPU + GPU + ANE combined SoC power`

## FP32

| Profile | No-monitor | Monitored | Power | J/image |
|---|---:|---:|---:|---:|
| Light | 69.18 | 69.03 | 11.62 W | 0.1677 |
| Medium | 58.10 | 57.42 | 15.55 W | 0.2708 |
| Heavy | 27.79 | 27.58 | 14.36 W | 0.5208 |

## FP16

| Profile | No-monitor | Monitored | Power | J/image |
|---|---:|---:|---:|---:|
| Light | 78.29 | 77.71 | 12.13 W | 0.1560 |
| Medium | 64.56 | 63.92 | 13.78 W | 0.2156 |
| Heavy | 32.61 | 34.02 | 14.26 W | 0.4312 |

FP16 representative üç workload'ın tamamında FP32'den daha düşük J/image sağlamıştır.

Yaklaşık enerji/image iyileşmesi:

- Light: %7
- Medium: %20
- Heavy: %17

Monitoring etkisi genel olarak düşük kalmıştır.

---

# 12. MSI Feasibility

224×224 / B1:

| Backend | Precision | Throughput | Latency |
|---|---:|---:|---:|
| CPU | FP32 | 7.11 img/s | 140.61 ms |
| CUDA | FP32 | 61.17 img/s | 16.35 ms |
| CUDA | FP16 | 74.01 img/s | 13.51 ms |

CUDA FP16 feasibility testinde FP32'ye göre yaklaşık %21 throughput artışı görülmüştür.

Memory:

- FP32 allocation: 346.7 MB
- FP16 allocation: 177.6 MB

---

# 13. MSI CUDA Full Sweep

## FP32

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 92.81 | 148.75 | 170.15 |
| 224 | 62.52 | 79.12 | 87.32 |
| 320 | 36.39 | 41.81 | 41.74 |

## FP16

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 87.49 | 230.05 | 431.80 |
| 224 | 128.02 | 182.36 | 233.08 |
| 320 | 88.60 | 94.59 | 121.49 |

FP16 küçük `160/B1` workload'da FP32'den biraz daha yavaştır.

Ancak workload büyüdükçe FP16 avantajı çok hızlı artmaktadır.

Örnekler:

- 160/B16: yaklaşık 2.54×
- 224/B16: yaklaşık 2.67×
- 320/B16: yaklaşık 2.91×

---

# 14. MSI CUDA Memory

Heavy workload 320/B16:

- FP32 peak allocation: 864.9 MB
- FP16 peak allocation: 438.6 MB

FP16 yaklaşık %49 daha düşük peak allocation kullanmıştır.

Bütün 18 configuration başarıyla çalışmıştır.

ConvNeXt-Base Vision Medium workload RTX 3050'nin 4 GB VRAM kapasitesi içerisinde kalmıştır.

---

# 15. MSI CPU Batch Davranışı

224×224 / 16 thread:

| Batch | Throughput |
|---:|---:|
| 1 | **9.66** |
| 2 | 8.64 |
| 4 | 9.17 |
| 8 | 8.05 |

Mac CPU'nun aksine MSI CPU üzerinde batching throughput kazancı sağlamamıştır.

En yüksek kısa-sweep throughput B1'de elde edilmiştir.

Bu sonuç batch optimumunun cihaz bağımlı olduğunu tekrar göstermektedir.

---

# 16. MSI CPU Thread Scaling

## High-resolution single — 320/B1

| Threads | Throughput |
|---:|---:|
| 1 | 1.44 |
| 2 | 2.41 |
| 4 | 4.33 |
| 8 | 4.10 |
| 16 | **5.48** |

Kısa-sweep throughput optimumu 16 thread'dir.

## Batched load — 224/B4

| Threads | Throughput |
|---:|---:|
| 1 | 2.74 |
| 2 | 4.80 |
| 4 | 7.54 |
| 8 | **9.92** |
| 16 | 9.52 |

Kısa-sweep optimumu 8 thread'dir.

---

# 17. MSI CPU Sustained Power

Yaklaşık 12 saniyelik paired A/B workload'lar kullanılmıştır.

CPU Package power LibreHardwareMonitor üzerinden ölçülmüştür.

## High-resolution single

| Threads | No-monitor | Monitored | Power | J/image |
|---:|---:|---:|---:|---:|
| 1 | 1.46 | 1.46 | 18.66 W | 12.8462 |
| 8 | **4.77** | 4.71 | 37.87 W | **8.0373** |
| 16 | 4.40 | 4.33 | 37.13 W | 8.4552 |

Sustained workload'da 8 thread, kısa sweep'te en hızlı olan 16 thread'in önüne geçmiştir.

## Batched load

| Threads | No-monitor | Monitored | Power | J/image |
|---:|---:|---:|---:|---:|
| 1 | 2.30 | 2.21 | 19.21 W | 8.6695 |
| 8 | 8.05 | 8.05 | 32.13 W | 3.9453 |
| 16 | **9.92** | **9.91** | 37.42 W | **3.7773** |

Batched workload'da ise sustained optimum 16 thread olmuştur.

Bu nedenle CPU thread optimumu benchmark duration ve operating state'e bağlıdır.

---

# 18. MSI CUDA Sustained Power

NVML paired A/B ölçümü:

- duration: yaklaşık 10 saniye
- polling: 500 ms
- 3 repeats
- power boundary: NVIDIA GPU board power

Idle baseline:

- Median: 13.71 W
- Mean: 12.07 W

## FP32

| Profile | No-monitor | Monitored | Power | Util | J/image |
|---|---:|---:|---:|---:|---:|
| Light | 89.31 | 89.93 | 57.21 W | 91.2% | 0.6362 |
| Medium | 74.00 | 73.95 | 57.43 W | 95.0% | 0.7750 |
| Heavy | 39.57 | 39.60 | 57.85 W | 95.7% | 1.4625 |

## FP16

| Profile | No-monitor | Monitored | Power | Util | J/image |
|---|---:|---:|---:|---:|---:|
| Light | 88.89 | 85.30 | 47.70 W | 37.0% | 0.5479 |
| Medium | 171.54 | 171.85 | 57.52 W | 94.7% | 0.3347 |
| Heavy | 117.09 | 117.40 | 57.37 W | 95.2% | 0.4885 |

Medium ve Heavy workload'larda NVML monitoring etkisi yaklaşık sıfırdır.

FP16 Light yaklaşık %5 civarında monitoring sensitivity göstermiştir.

---

# 19. CUDA Precision ve Enerji

FP16 Medium:

- FP32: 0.7750 J/image
- FP16: 0.3347 J/image

Yaklaşık %57 daha düşük enerji/image.

FP16 Heavy:

- FP32: 1.4625 J/image
- FP16: 0.4885 J/image

Yaklaşık %67 daha düşük enerji/image.

Light workload'da throughput FP32 ve FP16 için birbirine yakın olsa da FP16 daha düşük güç ve daha düşük J/image sağlamıştır.

Bu nedenle workload büyüdükçe FP16 hem latency hem energy açısından daha güçlü bir seçim haline gelmektedir.

---

# 20. Mac ↔ MSI Short-Sweep Crossover

Toplam 18 workload karşılaştırılmıştır.

Sonuç:

- Mac wins: 0
- MSI wins: 18

ConvNeXt-Base kısa sweep'te tüm resolution × batch × precision noktalarında MSI CUDA tarafından daha hızlı çalıştırılmıştır.

Vision Easy ResNet50 karakterizasyonunda görülen bazı küçük-workload crossover noktaları Vision Medium'da görülmemiştir.

---

# 21. Mac ↔ MSI Sustained Crossover

## FP32

| Profile | Mac | MSI | Winner |
|---|---:|---:|---|
| Light | 69.18 | 89.31 | MSI 1.29× |
| Medium | 58.10 | 74.00 | MSI 1.27× |
| Heavy | 27.79 | 39.57 | MSI 1.42× |

## FP16

| Profile | Mac | MSI | Winner |
|---|---:|---:|---|
| Light | 78.29 | 88.89 | MSI 1.14× |
| Medium | 64.56 | 171.54 | MSI 2.66× |
| Heavy | 32.61 | 117.09 | MSI 3.59× |

Sustained sonuç:

- Mac wins: 0
- MSI wins: 6

Model ve workload yoğunluğu arttıkça discrete GPU avantajı belirgin biçimde büyümektedir.

---

# 22. Scheduler Açısından Ana Bulgular

### 1. Model büyüklüğü device crossover'ı değiştirir

Vision Easy / ResNet50'de küçük FP16 sustained workload'da Mac avantajı görülebilmişti.

Vision Medium / ConvNeXt-Base'te MSI bütün kısa ve sustained representative workload'ları kazanmıştır.

Dolayısıyla scheduler model büyüklüğünü ve computational intensity'yi dikkate almalıdır.

### 2. Precision workload'a bağlıdır

FP16 küçük workload'da her zaman daha hızlı değildir.

Ancak medium ve heavy workload'larda özellikle RTX üzerinde ciddi throughput ve enerji avantajı sağlamaktadır.

### 3. Precision batch scaling davranışını değiştirir

Mac ve MSI accelerator'larında FP16 büyük batch'lerde FP32'den daha iyi scaling göstermiştir.

### 4. Batch optimumu cihaz bağımlıdır

Mac CPU:

`224/B8` batching avantajı göstermiştir.

MSI CPU:

`224/B1` kısa sweep'te en iyi noktadır.

### 5. Thread optimumu workload ve süre bağımlıdır

Mac CPU:

- iki representative workload'da kısa sweep optimumu 4 thread

MSI CPU:

- short highres optimum: 16 thread
- sustained highres optimum: 8 thread
- short batched optimum: 8 thread
- sustained batched optimum: 16 thread

Tek bir global thread optimumu bulunmamaktadır.

### 6. Latency ve energy optimumu farklı olabilir

Mac highres workload:

- latency optimumu: 4 thread
- energy/image optimumu: 1 thread

Bu doğrudan multi-objective scheduler gereksinimi oluşturmaktadır.

---

# 23. Energy Boundary Sınırlaması

Mac power measurement:

`CPU + GPU + ANE combined SoC power`

MSI CUDA power measurement:

`GPU board power`

MSI CPU:

`CPU Package power`

Bu nedenle Mac ve MSI absolute `J/image` değerleri aynı measurement boundary'den gelmemektedir.

Cross-device mutlak enerji winner ilan edilmemelidir.

Enerji sonuçları öncelikle aynı cihaz içerisindeki configuration seçimleri için kullanılmalıdır.

---

# 24. Diğer Sınırlamalar

- Synthetic input kullanılmıştır.
- Image decode ve preprocessing hariç tutulmuştur.
- Accuracy / quality ölçülmemiştir.
- FP16 ile FP32 arasında classification accuracy eşitliği varsayılmamıştır.
- Network / communication maliyeti dahil değildir.
- Mac unified-memory total peak usage yalnız MPS allocation metriğiyle tam temsil edilmez.
- Short benchmark ile sustained workload arasında state / thermal / scheduling farkları oluşabilmektedir.
- MSI feasibility testinde bazı workload'lar cold/session state'e duyarlı davranmıştır.

---

# 25. Vision Medium Completion Checklist

- [x] Mac feasibility
- [x] Mac MPS FP32 sweep
- [x] Mac MPS FP16 sweep
- [x] Mac MPS memory
- [x] Mac CPU batch behavior
- [x] Mac CPU thread scaling
- [x] Mac CPU power
- [x] Mac MPS sustained validation
- [x] Mac MPS power
- [x] MSI feasibility
- [x] MSI CUDA FP32 sweep
- [x] MSI CUDA FP16 sweep
- [x] MSI CUDA memory
- [x] MSI CPU batch behavior
- [x] MSI CPU thread scaling
- [x] MSI CPU sustained validation
- [x] MSI CPU power
- [x] MSI CUDA sustained validation
- [x] MSI CUDA NVML power
- [x] Monitoring A/B validation
- [x] Mac ↔ MSI short crossover
- [x] Mac ↔ MSI sustained crossover
- [x] Scheduler implications
- [x] Measurement boundary caveat

**Vision Medium: COMPLETE**

---

# 26. Next Step

Vision characterization status:

- Vision Easy / ResNet50: COMPLETE
- Vision Medium / ConvNeXt-Base: COMPLETE
- Vision Hard / ConvNeXt-Large: NEXT

Bir sonraki aşama:

**Vision Hard — ConvNeXt-Large characterization**

Aynı metodoloji korunacaktır:

- feasibility,
- memory / OOM boundary,
- CPU characterization,
- FP32 / FP16 accelerator sweep,
- sustained performance,
- power / energy,
- Mac ↔ MSI crossover,
- scheduler implications.