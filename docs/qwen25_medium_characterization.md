# Qwen2.5-1.5B-Instruct Medium Text Characterization

## 1. Purpose

Bu çalışma, Medium/Text workload olarak `Qwen/Qwen2.5-1.5B-Instruct` modelinin heterojen edge cihazlarında performans, güç, enerji ve kaynak davranışını karakterize eder.

Amaç, ileride geliştirilecek resource-aware scheduler için şu sorulara veri sağlamaktır:

- CPU mu GPU mu?
- Hangi CPU thread sayısı daha uygun?
- Context ve batch büyüdükçe performans nasıl değişiyor?
- Kısa benchmark ile sustained inference arasında fark oluşuyor mu?
- Throughput ve energy/token arasında nasıl bir trade-off var?

---

## 2. Hardware

### Mac

- Apple M4
- 10 logical CPU
- Apple GPU / MPS
- Unified memory
- macOS
- PyTorch + Transformers

### MSI

- Intel Core i7-11800H
- 8 physical / 16 logical CPU
- NVIDIA RTX 3050 Laptop GPU
- 4 GB VRAM
- Windows 11
- CUDA backend

---

## 3. Model and Precision

Model:

`Qwen/Qwen2.5-1.5B-Instruct`

GPU experiments:

- Mac MPS: FP16
- MSI CUDA: FP16

CPU experiments:

- FP32

---

## 4. Feasibility

Model her iki GPU backend üzerinde başarıyla çalıştı.

Initial feasibility workload:

- Context = 128
- Output = 64
- Batch = 1

Mac MPS:

- Latency: 2.599 s
- Throughput: 24.62 tok/s
- MPS allocated: yaklaşık 2.90 GB
- Status: PASS

MSI CUDA:

- Latency: 3.161 s
- Throughput: 20.25 tok/s
- VRAM allocated: yaklaşık 2.88 GB
- VRAM peak: yaklaşık 2.90 GB
- Status: PASS

Bu ilk test yalnızca feasibility kontrolü olarak kullanıldı. Final performans karşılaştırmaları tekrarlı workload sweep sonuçlarına dayanmaktadır.

---

## 5. Workload Matrix

Full GPU sweep:

- Context: 128, 512, 1024
- Output: 64, 128
- Batch: 1, 2, 4
- Toplam: 18 configuration
- Repeat: 3
- Final değer: median throughput

---

## 6. Mac MPS Sweep

| CTX | OUT | B1 | B2 | B4 |
|---:|---:|---:|---:|---:|
| 128 | 64 | 21.92 | 46.38 | 84.30 |
| 128 | 128 | 25.00 | 48.03 | 89.63 |
| 512 | 64 | 21.64 | 37.02 | 57.27 |
| 512 | 128 | 23.19 | 41.97 | 70.65 |
| 1024 | 64 | 18.58 | 28.65 | 39.90 |
| 1024 | 128 | 20.33 | 34.04 | 52.33 |

Units: tok/s.

Context büyüdükçe throughput genel olarak azalırken batch büyütmek MPS throughput'unu önemli ölçüde artırmaktadır.

MPS memory API yaklaşık 2.97 GB civarında sabit değer göstermiştir. Apple unified-memory yapısı ve API semantiği nedeniyle bu değer gerçek peak system memory tüketimi olarak yorumlanmamalıdır.

---

## 7. MSI CUDA Sweep

| CTX | OUT | B1 | B2 | B4 |
|---:|---:|---:|---:|---:|
| 128 | 64 | 30.00 | 57.52 | 109.41 |
| 128 | 128 | 28.83 | 55.96 | 109.29 |
| 512 | 64 | 26.76 | 49.96 | 90.63 |
| 512 | 128 | 27.07 | 51.96 | 97.64 |
| 1024 | 64 | 24.81 | 43.81 | 75.12 |
| 1024 | 128 | 26.13 | 48.28 | 88.18 |

Units: tok/s.

Peak VRAM yaklaşık 2965–3326 MB arasında değişti.

18 workload'un tamamı PASS oldu ve OOM görülmedi.

---

## 8. Mac MPS vs RTX Short-Sweep Comparison

RTX CUDA:

- 18 / 18 workload'da daha yüksek throughput
- Ortalama throughput avantajı yaklaşık 36.3%
- Minimum avantaj yaklaşık 15.3%
- Maksimum avantaj yaklaşık 88.3%
- Ortalama latency azalması yaklaşık 25.4%

OUT=64, Batch=4 örneği:

| CTX | Mac MPS | RTX CUDA | RTX Advantage |
|---:|---:|---:|---:|
| 128 | 84.30 | 109.41 | +29.8% |
| 512 | 57.27 | 90.63 | +58.2% |
| 1024 | 39.90 | 75.12 | +88.3% |

Workload ağırlaştıkça discrete GPU avantajı belirgin biçimde artmaktadır.

---

## 9. Representative Workloads

Power characterization için üç profil seçildi.

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

## 10. Mac MPS Power and Energy

Measurement:

- AC power
- `powermetrics`
- 100 ms sampling
- Model warm-up sonrası synchronized start
- 3 repeats
- Median
- Power boundary: CPU + GPU + ANE combined SoC power

| Profile | Throughput | Power | J/run | J/token |
|---|---:|---:|---:|---:|
| Light | 24.40 | 6.76 W | 17.71 | 0.2768 |
| Medium | 41.81 | 8.64 W | 52.93 | 0.2068 |
| Heavy | 53.96 | 11.96 W | 113.46 | 0.2216 |

Medium profile Mac üzerinde en düşük J/token değerini verdi.

---

## 11. Mac CPU Scaling

CPU experiments FP32 olarak çalıştırıldı.

### Long Single

Workload:

- Context = 128
- Output = 128
- Batch = 1

| Threads | Latency | Throughput |
|---:|---:|---:|
| 1 | 12.19 s | 10.50 |
| 2 | 12.18 s | 10.51 |
| 4 | 12.15 s | 10.54 |
| 6 | 12.37 s | 10.35 |
| 8 | 12.39 s | 10.34 |
| 10 | 12.46 s | 10.27 |

### Batch4

Workload:

- Context = 128
- Output = 64
- Batch = 4

| Threads | Latency | Throughput |
|---:|---:|---:|
| 1 | 11.58 s | 22.11 |
| 2 | 11.47 s | 22.31 |
| 4 | 11.43 s | 22.40 |
| 6 | 11.58 s | 22.10 |
| 8 | 11.78 s | 21.74 |
| 10 | 11.71 s | 21.86 |

4 threads throughput açısından en iyi sonucu verdi.

Daha fazla CPU thread kullanmak otomatik olarak daha yüksek performans sağlamadı.

---

## 12. Mac CPU Power

Workload:

- Context = 128
- Output = 64
- Batch = 4

| Threads | Throughput | Power | J/run | J/token |
|---:|---:|---:|---:|---:|
| 1 | 22.01 | 6.56 W | 76.25 | 0.2979 |
| 2 | 22.23 | 6.67 W | 76.77 | 0.2999 |
| 4 | 22.33 | 6.74 W | 77.21 | 0.3016 |
| 6 | 21.97 | 6.73 W | 78.23 | 0.3056 |
| 8 | 21.91 | 6.75 W | 78.90 | 0.3082 |
| 10 | 21.76 | 6.73 W | 79.25 | 0.3096 |

Sonuç:

- Throughput-optimal: 4 threads
- Energy-optimal: 1 thread

Scheduler yalnızca maksimum thread sayısını seçmemelidir.

---

## 13. MSI CPU Scaling

Qwen2.5 FP32 CPU inference MSI üzerinde çok yavaş olduğu için full CPU sweep pratik değildi.

Bu nedenle reduced CPU screening kullanıldı:

- Context = 128
- Output = 16
- Threads = 1, 2, 4, 8, 16
- Single repeat
- Long-single ve Batch4

### Long Single

| Threads | Latency | Throughput |
|---:|---:|---:|
| 2 | 624.30 s | 0.03 |
| 4 | 395.20 s | 0.04 |
| 8 | 139.93 s | 0.11 |
| 16 | 114.76 s | 0.14 |

1-thread run tamamlanmış olsa da bu raporda kullanılan terminal çıktısında kesin değeri bulunmadığı için tabloya eklenmemiştir.

### Batch4

| Threads | Latency | Throughput |
|---:|---:|---:|
| 1 | 867.71 s | 0.07 |
| 2 | 609.30 s | 0.11 |
| 4 | 400.33 s | 0.16 |
| 8 | 171.12 s | 0.37 |
| 16 | 131.79 s | 0.49 |

Measured configurations arasında 16 threads en yüksek throughput'u sağladı.

MSI CPU, Medium Qwen workload için pratikte fallback backend durumundadır.

---

## 14. MSI CPU Power

Power characterization yalnızca 8t ve 16t üzerinde yapıldı.

Sebep:

- CPU scaling eğrisi diğer thread sayılarını zaten kapsıyordu.
- Daha düşük thread sayılarında workload aşırı uzun sürüyordu.
- 8t ve 16t viable configurations olarak seçildi.

LibreHardwareMonitor:

- CPU Package Power
- HTTP API
- 1 second polling
- Idle settle
- 3 repeats

Idle CPU package power:

15.26 W

Median results:

| Threads | Throughput | Power | Dynamic Power | J/run | J/token |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.085 | 33.21 W | 17.96 W | 6311.52 | 394.47 |
| 16 | 0.105 | 34.89 W | 19.64 W | 5363.64 | 335.23 |

16t hem throughput hem de energy/token açısından 8t'den daha iyi sonuç verdi.

---

## 15. MSI CPU Monitoring Validation

16t no-monitor validation:

- R1: 174.77 s / 0.092 tok/s
- R2: 150.69 s / 0.106 tok/s
- R3: 127.74 s / 0.125 tok/s

Median:

- Latency: 150.69 s
- Throughput: 0.106 tok/s

LHM 1-second monitored median:

- Latency: 153.09 s
- Throughput: 0.105 tok/s

Monitoring etkisi yaklaşık:

- +1.6% latency
- -0.9% throughput

Bu nedenle 1-second LHM polling kabul edilebilir kabul edildi.

100 ms CPU polling ise yüksek overhead nedeniyle reddedildi.

---

## 16. RTX NVML Power

Measurement:

- NVML
- 100 ms sampling
- 60-second sustained windows
- 3 repeats
- Power boundary: NVIDIA GPU board power

Idle GPU board power:

16.64 W

Median results:

| Profile | Throughput | Power | GPU Util | Peak VRAM | J/run | J/token |
|---|---:|---:|---:|---:|---:|---:|
| Light | 27.72 | 57.27 W | 60.7% | 3381 MB | 137.61 | 2.1501 |
| Medium | 38.14 | 54.70 W | 52.7% | 3493 MB | 367.14 | 1.4341 |
| Heavy | 66.76 | 56.87 W | 65.1% | 3823 MB | 436.18 | 0.8519 |

Batch ve workload büyüdükçe GPU'nun fixed power cost'u daha fazla token'a yayıldığı için J/token önemli ölçüde azaldı.

Heavy workload yaklaşık 3.82 GB VRAM kullandı ve 4 GB GPU sınırına yaklaştı fakat OOM oluşmadı.

---

## 17. RTX Sustained Performance

NVML olmadan 60-second sustained test:

| Profile | Median Throughput |
|---|---:|
| Light | 27.02 |
| Medium | 41.54 |
| Heavy | 71.98 |

Short sweep ile karşılaştırma:

| Profile | Short Sweep | Sustained |
|---|---:|---:|
| Light | 30.00 | 27.02 |
| Medium | 51.96 | 41.54 |
| Heavy | 88.18 | 71.98 |

Bu düşüş yalnızca monitoring overhead ile açıklanamamaktadır.

Uzun süreli workload sırasında thermal/power davranışı short benchmark sonuçlarından farklı bir operating point oluşturmaktadır.

Scheduler tasarımında kısa benchmark throughput değerlerinin sustained inference performansını doğrudan temsil ettiği varsayılmamalıdır.

---

## 18. NVML Monitoring Overhead

NVML monitored vs no-monitor sustained:

| Profile | No Monitor | NVML |
|---|---:|---:|
| Light | 27.02 | 27.72 |
| Medium | 41.54 | 38.14 |
| Heavy | 71.98 | 66.76 |

Light farkı normal run-to-run variability seviyesindedir.

Medium ve Heavy workload'larda NVML 100 ms sampling yaklaşık %7–8 throughput etkisi göstermiştir.

Power ve energy sonuçları monitored operating condition'a aittir.

---

## 19. Measurement Boundary Caveat

Mac ve RTX absolute energy değerleri doğrudan karşılaştırılmamalıdır.

Mac ölçümü:

- CPU + GPU + ANE combined SoC power

RTX ölçümü:

- GPU board power only

Bu nedenle cihazlar arası enerji değerleri absolute winner şeklinde yorumlanmamalıdır.

Enerji sonuçları esas olarak aynı platform içerisindeki workload ve configuration trendlerini analiz etmek için kullanılmalıdır.

---

## 20. Main Findings

1. RTX CUDA kısa sweep'te 18/18 workload'da Mac MPS'ten daha hızlıdır.
2. RTX avantajı özellikle büyük context ve batch workload'larda artmaktadır.
3. Mac MPS düşük-power operating point sunmaktadır.
4. Batch büyütmek her iki GPU backend üzerinde throughput'u artırmaktadır.
5. MSI CPU, Medium Qwen inference için pratik primary backend değildir.
6. Mac CPU'da maksimum thread sayısı optimum değildir.
7. Mac CPU throughput-optimum 4t, energy-optimum 1t'dir.
8. MSI CPU'da tested configurations arasında 16t hem throughput hem energy/token açısından 8t'den iyidir.
9. Short benchmark ve sustained GPU throughput arasında önemli fark vardır.
10. Scheduler yalnızca peak throughput değil; sustained throughput, energy, memory ve workload shape kullanmalıdır.

---

## 21. Scheduler Implications

Characterization sonuçları scheduler için şu özelliklerin önemli olduğunu göstermektedir:

- device
- backend
- context length
- output length
- batch size
- CPU thread count
- precision
- throughput
- sustained throughput
- power
- energy/run
- energy/token
- memory usage
- utilization
- measurement boundary
- monitoring mode

Potential objective:

`J = αL + βE + γC + δQ`

Burada:

- L = latency
- E = energy
- C = communication / resource cost
- Q = quality-related cost

Scheduler workload shape ve device state'e göre configuration seçmelidir.

---

## 22. Recommended Dataset Schema

device,  
power_mode,  
modality,  
model,  
model_parameters,  
difficulty,  
context_length,  
output_tokens,  
batch_size,  
backend,  
cpu_threads,  
precision,  
quantization,  
latency_sec,  
throughput_tok_sec,  
sustained_throughput_tok_sec,  
power_w,  
energy_run_j,  
energy_token_j,  
memory_mb,  
utilization_percent,  
status,  
repeat

---

## 23. Medium/Text Completion Checklist

- [x] Mac MPS feasibility
- [x] MSI CUDA feasibility
- [x] Full Mac MPS workload sweep
- [x] Full MSI CUDA workload sweep
- [x] Mac CPU scaling
- [x] MSI CPU reduced scaling
- [x] Mac CPU power
- [x] MSI CPU power
- [x] Mac MPS power
- [x] RTX NVML power
- [x] CPU monitoring validation
- [x] GPU monitoring validation
- [x] Sustained GPU validation
- [x] Mac vs MSI comparison
- [x] Measurement-boundary caveat
- [x] Scheduler implications
- [x] Dataset schema

**Medium / Text: COMPLETE**

---

## 24. Next Step

Next characterization target:

**Hard / Text**

Primary model:

`Qwen3-1.7B`

Native FP16 feasibility:

- Mac MPS: PASS
- MSI RTX 3050 4 GB: OOM during model load

Controlled cross-device Hard comparison:

- Qwen3-1.7B
- Q4_K_M
- llama.cpp / GGUF