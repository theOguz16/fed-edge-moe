# ResNet50 Vision Easy Karakterizasyonu

## 1. Amaç

Bu deneyin amacı, ResNet50 görüntü sınıflandırma çıkarımını heterojen edge donanımlarda karakterize ederek ileride geliştirilecek resource-aware scheduler için güvenilir bir ölçüm tabanı oluşturmaktır.

İncelenen temel değişkenler:

- cihaz ve backend,
- CPU thread sayısı,
- giriş çözünürlüğü,
- batch büyüklüğü,
- FP32 / FP16 precision,
- bellek kullanımı,
- sustained performans,
- güç tüketimi,
- enerji maliyeti.

Bu aşamada inference tamamen local çalıştırıldığı için network/communication maliyeti ölçülmemiştir. Ağ maliyeti multi-device ve distributed inference aşamasında eklenecektir.

---

## 2. Model

- Model: ResNet50
- Parametre sayısı: 25.56M
- Ağırlıklar: `ResNet50_Weights.DEFAULT`
- Framework: PyTorch / torchvision
- Görev: Image classification inference
- Training: Yok
- Input: Sentetik image tensor

Ölçümler yalnızca model inference süresini kapsamaktadır. Görüntü decode, resize ve preprocessing süresi dahil edilmemiştir.

---

## 3. Donanım

### Mac

- Apple M4
- 10 logical CPU
- Apple MPS backend
- Unified memory

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

## 4. Vision Workload Boyutları

Text workload'larında kullanılan context length'in Vision tarafındaki karşılığı input resolution olarak ele alınmıştır.

### Resolution

- 160×160
- 224×224
- 320×320

### Batch

Ana accelerator sweep:

- B1
- B4
- B16

Buna ek olarak CPU batch davranışını daha ayrıntılı görmek için ara batch noktaları da ölçülmüştür.

### Representative profiller

- Light: `160×160 / B1`
- Medium: `224×224 / B4`
- Heavy: `320×320 / B16`

CPU için ayrıca:

- High-resolution single: `320×320 / B1`
- Batch workload: cihazın uygun batch noktası

kullanılmıştır.

---

# 5. Mac Feasibility

224×224 / B1 başlangıç testi:

| Backend | Precision | Throughput |
|---|---:|---:|
| CPU | FP32 | 74.40 img/s |
| MPS | FP32 | 83.12 img/s |
| MPS | FP16 | 99.77 img/s |

MPS FP32, CPU FP32'den yaklaşık %11.7 daha hızlıdır.

MPS FP16 ise MPS FP32'ye göre yaklaşık %20 throughput artışı sağlamıştır.

MPS allocation:

- FP32: 98.3 MB
- FP16: 49.2 MB

FP16 allocation yaklaşık yarıya düşmektedir.

---

# 6. Mac MPS Resolution × Batch Sweep

## FP32

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 89.58 | 221.27 | 293.67 |
| 224 | 84.84 | 139.28 | 157.02 |
| 320 | 60.03 | 78.43 | 84.62 |

## FP16

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 119.87 | 273.83 | 405.90 |
| 224 | 97.91 | 169.46 | 217.54 |
| 320 | 72.18 | 110.88 | 120.16 |

Bütün 9 workload'da FP16, FP32'den daha yüksek throughput sağlamıştır.

Resolution büyüdükçe batch scaling kazancı azalmaktadır.

Örneğin FP32:

- 160: B1→B16 ≈ 3.28×
- 224: B1→B16 ≈ 1.85×
- 320: B1→B16 ≈ 1.41×

Bu sonuç yüksek resolution altında accelerator saturation davranışının daha erken başladığını göstermektedir.

---

# 7. Mac CPU Thread Scaling

## High-resolution single — 320×320 / B1

| Threads | Throughput |
|---:|---:|
| 1 | 30.01 |
| 2 | 36.16 |
| 4 | **41.12** |
| 6 | 38.70 |
| 8 | 39.79 |
| 10 | 38.88 |

En iyi nokta 4 thread'dir.

## Batch workload — 224×224 / B10

| Threads | Throughput |
|---:|---:|
| 1 | 51.12 |
| 2 | 58.04 |
| 4 | 73.54 |
| 6 | 79.83 |
| 8 | 83.96 |
| 10 | **90.43** |

Batch workload 10 thread'e kadar ölçeklenmektedir.

Bu nedenle tek bir global "en iyi thread sayısı" bulunmamaktadır.

---

# 8. Mac CPU Batch Boundary

224×224 üzerinde:

| Batch | Throughput | RSS |
|---:|---:|---:|
| 8 | 68.81 | ~696 MB |
| 10 | 72.85 | ~662 MB |
| 12 | 70.72 | ~783 MB |
| 14 | 72.46 | ~882 MB |
| 16 | **11.67** | ~1294 MB |

B14→B16 arasında belirgin bir performance cliff oluşmuştur.

Throughput yaklaşık %84 düşerken RSS önemli ölçüde artmaktadır.

Bu nokta OOM değildir; dolayısıyla sonuç scheduler açısından bir feasibility değil, performance-boundary olarak değerlendirilmiştir.

---

# 9. Mac CPU Power

### High-resolution single — 320/B1

| Threads | Throughput | Power | J/image |
|---:|---:|---:|---:|
| 1 | 29.04 | 7.14 W | 0.2556 |
| 4 | **41.10** | 10.11 W | **0.2459** |
| 10 | 37.96 | 11.39 W | 0.2999 |

4 thread hem throughput hem enerji açısından en iyi measured operating point'tir.

### Batch workload — 224/B10

| Threads | Throughput | Power | J/image |
|---:|---:|---:|---:|
| 1 | 51.87 | 8.09 W | 0.1559 |
| 4 | 69.38 | 14.79 W | 0.2132 |
| 10 | **90.62** | 13.00 W | **0.1436** |

Batch workload'da 10 thread en iyi noktadır.

Thread optimumunun workload'a bağlı olduğu tekrar doğrulanmıştır.

---

# 10. Mac MPS Power

100 ms `powermetrics` polling bazı workload'larda ciddi monitoring interference oluşturduğu için reddedilmiştir.

500 ms polling ile monitoring etkisi kabul edilebilir seviyeye düşmüştür.

## Final 500 ms sonuçları

| Precision | Profile | Throughput | Power | J/image |
|---|---|---:|---:|---:|
| FP32 | Light | 100.58 | 5.33 W | 0.0524 |
| FP32 | Medium | 134.81 | 11.24 W | 0.0832 |
| FP32 | Heavy | 85.10 | 13.77 W | 0.1615 |
| FP16 | Light | 111.82 | 6.16 W | 0.0551 |
| FP16 | Medium | 180.20 | 11.63 W | 0.0646 |
| FP16 | Heavy | 121.71 | 13.68 W | 0.1124 |

FP16 medium ve heavy workload'larda hem daha hızlı hem daha enerji verimlidir.

Light workload'da ise FP32:

`0.0524 J/image`

ile FP16'dan:

`0.0551 J/image`

bir miktar daha düşük enerji tüketmiştir.

Bu nedenle FP16 her workload için otomatik enerji optimumu değildir.

---

# 11. MSI Feasibility

224×224 / B1:

| Backend | Precision | Throughput |
|---|---:|---:|
| CPU | FP32 | 21.13 img/s |
| CUDA | FP32 | 169.32 img/s |
| CUDA | FP16 | 141.21 img/s |

Küçük batch'te CUDA FP32, FP16'dan daha hızlıdır.

VRAM allocation:

- FP32: 106.5 MB
- FP16: 57.9 MB

FP16 yine önemli bellek tasarrufu sağlamaktadır.

---

# 12. MSI CUDA Resolution × Batch Sweep

## FP32

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 132.93 | 479.37 | 551.42 |
| 224 | 166.35 | 275.89 | 299.66 |
| 320 | 120.87 | 146.46 | 161.60 |

## FP16

| Resolution | B1 | B4 | B16 |
|---|---:|---:|---:|
| 160 | 125.63 | 460.00 | 954.55 |
| 224 | 143.91 | 446.45 | 539.64 |
| 320 | 130.72 | 253.87 | 288.40 |

FP16 küçük batch'te her zaman daha hızlı değildir.

Örneğin:

224/B1:

- FP32: 166.35 img/s
- FP16: 143.91 img/s

Ancak batch büyüdüğünde FP16 önemli üstünlük sağlamaktadır.

224/B16:

- FP32: 299.66 img/s
- FP16: 539.64 img/s

Yaklaşık 1.80× hızlanma oluşmaktadır.

320/B16 peak VRAM:

- FP32: 449.7 MB
- FP16: 243.0 MB

---

# 13. MSI CPU Thread Scaling

## High-resolution single — 320/B1

| Threads | Throughput |
|---:|---:|
| 1 | 4.74 |
| 2 | 7.84 |
| 4 | 13.43 |
| 8 | 12.83 |
| 16 | **14.30** |

4 thread sonrası ölçeklenme sınırlıdır.

## Batch workload — 224/B10

| Threads | Throughput |
|---:|---:|
| 1 | 8.75 |
| 2 | 13.72 |
| 4 | 20.14 |
| 8 | 19.58 |
| 16 | **20.37** |

Burada da 4 thread sonrası performans kazancı oldukça küçüktür.

---

# 14. MSI CPU Batch Davranışı

## 224×224

| Batch | Throughput |
|---:|---:|
| 1 | 17.85 |
| 2 | 27.32 |
| 4 | **27.96** |
| 8 | 20.00 |
| 10 | 18.43 |
| 12 | 18.66 |
| 14 | 18.91 |
| 16 | 18.62 |

Sweet spot yaklaşık B4'tür.

## 320×320

| Batch | Throughput |
|---:|---:|
| 1 | **12.45** |
| 2 | 11.67 |
| 4 | 10.35 |
| 8 | 8.33 |
| 10 | 7.93 |
| 12 | 8.16 |
| 14 | 7.91 |
| 16 | 7.97 |

320×320 workload'da batch büyütmek throughput'u artırmamaktadır.

Mac'teki B14→B16 cliff burada görülmemiştir.

Bunun yerine MSI CPU'da daha erken başlayan yumuşak saturation bulunmaktadır.

---

# 15. MSI CPU Power

CPU Package idle baseline:

- Median: 14.60 W
- Mean: 14.87 W

## High-resolution single — 320/B1

| Threads | Throughput | Package Power | J/image |
|---:|---:|---:|---:|
| 1 | 4.34 | 29.16 W | 6.7143 |
| 4 | 11.76 | 35.35 W | 3.0065 |
| 16 | 12.93 | 36.83 W | **2.8496** |

16-thread ölçümü 2 saniyelik LHM polling ile tekrar doğrulanmıştır.

No-monitor:

`12.83 img/s`

Monitored:

`12.93 img/s`

Fark yaklaşık +%0.8'dir.

## Batch optimal — 224/B4

| Threads | Throughput | Package Power | J/image |
|---:|---:|---:|---:|
| 1 | 7.52 | 27.59 W | 3.6703 |
| 4 | 16.33 | 30.09 W | 1.8427 |
| 16 | **26.39** | 38.38 W | **1.4542** |

Batch workload için 16 thread hem throughput hem enerji açısından measured optimum noktadır.

---

# 16. MSI CUDA Power

RTX idle power:

- Median: 8.80 W
- Mean: 10.04 W

Final representative sonuçlar:

| Precision | Profile | Throughput | GPU Power | Util | Peak VRAM | J/image |
|---|---|---:|---:|---:|---:|---:|
| FP32 | Light | 146.13* | 53.33 W | 50.1% | ~116 MB | 0.3693 |
| FP32 | Medium | 266.07 | 59.90 W | 99.8% | 148 MB | 0.2251 |
| FP32 | Heavy | 158.04 | 59.43 W | 99.0% | 450 MB | 0.3768 |
| FP16 | Light | 113.18 | 24.36 W | 35.5% | 63 MB | 0.2122 |
| FP16 | Medium | 394.57* | 58.30 W | 91.6% | ~81 MB | 0.1461 |
| FP16 | Heavy | 281.48 | 59.85 W | 99.9% | 243 MB | 0.2124 |

`*` Paired A/B doğrulamasından alınan final monitored değer.

Medium ve heavy workload'larda GPU yaklaşık 60 W sınırına ulaşmakta ve utilization %90–100 seviyesine çıkmaktadır.

Light workload'da ise GPU tamamen doygun değildir.

---

# 17. Monitoring A/B Validation

## Mac

100 ms `powermetrics` bazı MPS workload'larında %50'nin üzerinde throughput bozulmasına neden olmuştur ve reddedilmiştir.

500 ms polling kabul edilmiştir.

## MSI CPU

LHM 1 saniye polling özellikle high-resolution / 16-thread testinde ciddi interference oluşturmuştur.

2 saniye polling ile:

- no-monitor: 12.83 img/s
- monitored: 12.93 img/s

elde edilmiş ve ölçüm kabul edilmiştir.

## MSI CUDA

NVML 500 ms polling paired A/B:

### FP32 Light

- No-monitor: 149.23 img/s
- Monitored: 146.13 img/s
- Delta: -2.07%

### FP16 Medium

- No-monitor: 396.67 img/s
- Monitored: 394.57 img/s
- Delta: -0.53%

Her iki test de kabul edilmiştir.

---

# 18. Mac ↔ MSI Short-Sweep Crossover

Toplam 18 workload karşılaştırılmıştır.

Sonuç:

- Mac MPS wins: 0
- MSI CUDA wins: 18

## FP16

| Workload | Mac | MSI | Winner |
|---|---:|---:|---|
| 160/B1 | 119.87 | 125.63 | MSI 1.05× |
| 160/B4 | 273.83 | 460.00 | MSI 1.68× |
| 160/B16 | 405.90 | 954.55 | MSI 2.35× |
| 224/B1 | 97.91 | 143.91 | MSI 1.47× |
| 224/B4 | 169.46 | 446.45 | MSI 2.63× |
| 224/B16 | 217.54 | 539.64 | MSI 2.48× |
| 320/B1 | 72.18 | 130.72 | MSI 1.81× |
| 320/B4 | 110.88 | 253.87 | MSI 2.29× |
| 320/B16 | 120.16 | 288.40 | MSI 2.40× |

## FP32

| Workload | Mac | MSI | Winner |
|---|---:|---:|---|
| 160/B1 | 89.58 | 132.93 | MSI 1.48× |
| 160/B4 | 221.27 | 479.37 | MSI 2.17× |
| 160/B16 | 293.67 | 551.42 | MSI 1.88× |
| 224/B1 | 84.84 | 166.35 | MSI 1.96× |
| 224/B4 | 139.28 | 275.89 | MSI 1.98× |
| 224/B16 | 157.02 | 299.66 | MSI 1.91× |
| 320/B1 | 60.03 | 120.87 | MSI 2.01× |
| 320/B4 | 78.43 | 146.46 | MSI 1.87× |
| 320/B16 | 84.62 | 161.60 | MSI 1.91× |

Short-sweep performance haritasında MSI CUDA bütün workload'ları kazanmıştır.

---

# 19. Sustained Crossover

Short sweep tek başına scheduler için yeterli değildir.

Örneğin FP16 Light sustained test:

- Mac MPS: 114.79 img/s
- MSI CUDA: 108.42 img/s

Bu noktada Mac yaklaşık 1.06× daha hızlıdır.

Dolayısıyla:

- short benchmark sonucu,
- sustained operating point,
- thermal/clock state

birlikte değerlendirilmelidir.

Bu deneyde short-sweep sonucu MSI lehine 18/18 iken sustained representative workload'lardan birinde Mac öne geçmiştir.

---

# 20. Scheduler Açısından Ana Bulgular

### 1. Tek bir optimum backend yoktur

RTX genel olarak throughput açısından güçlüdür ancak küçük sustained workload'larda Mac rekabetçi veya daha hızlı olabilir.

### 2. Precision seçimi workload'a bağlıdır

RTX üzerinde:

- küçük batch → FP32 bazen daha hızlı,
- orta/büyük batch → FP16 belirgin biçimde daha hızlı.

### 3. FP16 bellek kullanımını önemli ölçüde azaltmaktadır

Özellikle heavy workload'da RTX peak VRAM yaklaşık:

`450 MB → 243 MB`

seviyesine düşmüştür.

### 4. Thread optimumu workload'a bağlıdır

Mac:

- high-resolution single → 4 thread
- batch workload → 10 thread

MSI:

- high-resolution single → 4–16 thread arası küçük fark
- batch workload → 16 thread avantajlı

### 5. Batch davranışı cihaz bağımlıdır

Mac CPU'da B14→B16 arasında sert cliff oluşmuştur.

MSI CPU'da aynı cliff görülmemiş; daha erken başlayan saturation oluşmuştur.

### 6. Büyük batch RTX lehine güçlü bir sinyal üretmektedir

Özellikle FP16:

`160/B16 → RTX ≈ 2.35×`

`224/B16 → RTX ≈ 2.48×`

`320/B16 → RTX ≈ 2.40×`

avantaj sağlamıştır.

---

# 21. Enerji Karşılaştırması Hakkında Önemli Sınırlama

Mac power boundary:

`Combined CPU + GPU + ANE`

olarak ölçülmektedir.

RTX tarafında ise NVML:

`GPU board power`

ölçmektedir.

Bu nedenle Mac ve MSI arasında mutlak `J/image` değerleri doğrudan aynı enerji boundary'siymiş gibi karşılaştırılmamalıdır.

Enerji ölçümleri kendi cihazı içerisinde scheduler operating point seçimi için güvenilirdir.

Cross-device enerji karşılaştırması için ortak bir system-level power boundary gereklidir.

---

# 22. Network

Bu deney tamamen local inference olduğu için network ölçülmemiştir.

Network/communication karakterizasyonu şu aşamada yapılacaktır:

- Mac ↔ MSI distributed inference,
- expert placement,
- tensor transfer,
- multi-device MoE scheduling.

İleride ölçülecek değişkenler:

- RTT,
- effective bandwidth,
- payload-size sensitivity,
- transfer latency,
- communication energy.

---

# 23. Vision Easy Durumu

ResNet50 Vision Easy için aşağıdaki karakterizasyonlar tamamlanmıştır:

- Mac CPU ✅
- Mac MPS FP32 ✅
- Mac MPS FP16 ✅
- MSI CPU ✅
- MSI CUDA FP32 ✅
- MSI CUDA FP16 ✅
- Resolution scaling ✅
- Batch scaling ✅
- CPU thread scaling ✅
- Memory / VRAM ✅
- CPU power ✅
- Accelerator power ✅
- Energy / image ✅
- GPU utilization ✅
- Sustained validation ✅
- Monitoring A/B ✅
- Mac ↔ MSI crossover ✅
- Scheduler implications ✅

**Vision Easy: COMPLETE**

