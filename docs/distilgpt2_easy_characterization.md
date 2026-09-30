# DistilGPT2 — Easy Text Workload Karakterizasyon Raporu

## 1. Amaç

Bu çalışma, ileride geliştirilecek **resource-aware scheduler** için Easy/Text seviyesinin temel karakterizasyonunu oluşturmayı amaçlamaktadır.

Buradaki amaç yalnızca hangi cihazın daha hızlı olduğunu bulmak değildir. Asıl amaç;

- workload büyüdükçe sistem davranışının nasıl değiştiğini,
- CPU thread sayısının performans ve enerjiye etkisini,
- CPU ile accelerator arasındaki farkları,
- batch, context ve output uzunluğunun etkisini,
- latency-optimal ve energy-optimal konfigürasyonların aynı olup olmadığını,
- Mac MPS ile NVIDIA CUDA arasında workload-dependent crossover oluşup oluşmadığını

ölçmektir.

Model:

`DistilGPT2`

Bu model Easy/Text seviyesini temsil etmektedir.

---

# 2. Test Sistemleri

## 2.1 Apple Silicon Mac

| Özellik | Değer |
|---|---|
| Platform | Apple Silicon Mac |
| CPU | Apple M4 |
| Logical CPU | 10 |
| GPU backend | Apple MPS / Metal |
| PyTorch accelerator | MPS |
| Final power condition | AC / şarja bağlı |

CPU tarafında farklı thread sayıları test edilmiştir.

MPS tarafında ise workload matrix kullanılarak accelerator davranışı ölçülmüştür.

---

## 2.2 MSI Laptop

| Özellik | Değer |
|---|---|
| CPU | Intel Core i7-11800H |
| Physical cores | 8 |
| Logical processors | 16 |
| CPU architecture | Non-hybrid, Hyper-Threading |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 GB |
| GPU backend | CUDA |
| Final power condition | AC / şarja bağlı |

RTX power telemetry için final yöntemde `NVML` kullanılmıştır.

---

# 3. CPU Thread Karakterizasyonu

## 3.1 Mac CPU — Long Single Generation

Workload:

- Batch: `1`
- Output: `128 token`

| Threads | Latency | Throughput |
|---:|---:|---:|
| 1 | 0.755 s | 169.6 tok/s |
| 2 | 0.803 s | 159.5 tok/s |
| **4** | **0.727 s** | **176.0 tok/s** |
| 6 | 0.766 s | 167.2 tok/s |
| 8 | 0.805 s | 159.1 tok/s |
| 10 | 0.822 s | 155.8 tok/s |
| MPS | **0.576 s** | **222.1 tok/s** |

### Gözlem

Mac CPU için maksimum thread sayısı en iyi sonucu vermemiştir.

Bu workload için CPU tarafında yaklaşık `4 thread` optimum bölge olarak görünmektedir.

`10 thread`, `4 thread` konfigürasyonundan daha yavaştır.

---

# 4. Mac CPU — Batch Workload

Workload:

- Batch: `4`
- Output: `64 token`

| Threads | Latency | Throughput |
|---:|---:|---:|
| 1 | 0.575 s | 445.6 tok/s |
| **2** | **0.573 s** | **446.7 tok/s** |
| 4 | 0.599 s | 427.7 tok/s |
| 6 | 0.607 s | 421.8 tok/s |
| 8 | 0.617 s | 414.7 tok/s |
| 10 | 0.631 s | 405.9 tok/s |
| MPS | **0.361 s** | **709.2 tok/s** |

### Gözlem

Optimum CPU thread sayısı workload'a göre değişmektedir.

Long single generation için yaklaşık `4 thread` daha iyi iken, batch workload'da `1–2 thread` daha iyi davranmıştır.

Bu sonuç:

> Tek bir sabit CPU thread politikasının tüm workload'lar için optimum olmadığını

göstermektedir.

---

# 5. Mac CPU Power / Energy Karakterizasyonu

İlk power sweep sırasında yaklaşık olarak aşağıdaki sonuçlar elde edilmiştir:

| Threads | Avg Power | Energy / Run |
|---:|---:|---:|
| 1 | 8.19 W | 5.284 J |
| 2 | 6.19 W | 3.811 J |
| **4** | **5.66 W** | **3.279 J** |
| 6 | 5.71 W | 3.409 J |
| 8 | 5.80 W | 3.623 J |
| 10 | 5.93 W | 3.768 J |
| MPS | 6.36 W | **2.290 J** |

`1 thread` sonucu olası outlier olarak değerlendirilmiştir.

### Gözlem

Mac CPU tarafında yaklaşık `4 thread`, throughput ve energy açısından dengeli bir çalışma noktasıdır.

MPS biraz daha yüksek power çekmesine rağmen işi daha hızlı tamamladığı için `energy/run` daha düşük çıkmıştır.

---

# 6. MSI CPU Thread Karakterizasyonu

Final CPU power experiment üç tekrar ve median kullanılarak yapılmıştır.

| Threads | Throughput | CPU Package Power | Energy / Run | Dynamic Energy / Run |
|---:|---:|---:|---:|---:|
| **2** | 136.5 tok/s | **29.78 W** | **56.93 J** | 27.67 J |
| 4 | 127.9 tok/s | 32.01 W | 62.88 J | 33.72 J |
| **6** | **139.3 tok/s** | 38.81 W | 71.32 J | 43.91 J |
| 16 | 99.7 tok/s | 39.25 W | **101.03 J** | 62.43 J |

### Gözlem

Burada iki farklı optimum ortaya çıkmaktadır.

### Energy-oriented CPU configuration

`2 threads`

En düşük:

- package power,
- energy/run

değerlerini vermektedir.

### Throughput-oriented CPU configuration

`6 threads`

En yüksek CPU throughput değerini vermektedir.

### Oversubscription örneği

`16 threads`

hem daha fazla kaynak kullanmasına rağmen daha yavaş çalışmış hem de energy/run ciddi şekilde yükselmiştir.

Bu nedenle:

> Maximum hardware parallelism ≠ optimum execution configuration

sonucu açık şekilde görülmektedir.

---

# 7. MSI CPU vs CUDA

Aynı DistilGPT2 batch workload üzerinde elde edilen temsilci sonuçlar:

| Configuration | Throughput |
|---|---:|
| CPU 2 threads | ~136–142 tok/s |
| CPU 6 threads | ~139 tok/s |
| CPU 16 threads | ~100 tok/s |
| RTX 3050 CUDA | ~760–780 tok/s |

CUDA bu workload üzerinde CPU konfigürasyonlarından yaklaşık birkaç kat daha yüksek throughput sağlamaktadır.

Ancak bu sonuç:

> her workload için GPU seçilmelidir

anlamına gelmemektedir.

Daha sonraki workload matrix sonuçları özellikle hafif workload'larda farklı cihaz davranışları olduğunu göstermektedir.

---

# 8. Accelerator Workload Matrix

DistilGPT2 için aşağıdaki workload matrix uygulanmıştır.

| Parameter | Values |
|---|---|
| Context length | 32 / 128 / 512 |
| Output tokens | 32 / 64 / 128 |
| Batch size | 1 / 2 / 4 / 8 |
| Repeat | 3 |

Toplam:

`3 × 3 × 4 = 36 workload configuration`

hem MPS hem CUDA üzerinde test edilmiştir.

---

# 9. RTX — Batch Etkisi

Örnek:

`CTX=32, OUT=64`

| Batch | Throughput |
|---:|---:|
| 1 | 186.3 tok/s |
| 2 | 385.7 tok/s |
| 4 | 767.2 tok/s |
| 8 | **1495.7 tok/s** |

Batch büyüdükçe GPU utilization daha etkili hale gelmekte ve toplam throughput ciddi şekilde artmaktadır.

---

# 10. RTX — Context Etkisi

Örneğin Batch=8 ve Output=64 için:

| Context | Throughput |
|---:|---:|
| 32 | 1495.7 tok/s |
| 128 | 1386.4 tok/s |
| 512 | 717.6 tok/s |

Context uzunluğu büyüdükçe workload'un memory ve compute maliyeti artmaktadır.

Bu nedenle batch avantajı devam etse bile throughput düşmektedir.

---

# 11. RTX VRAM Davranışı

Workload büyüdükçe VRAM kullanımı da yükselmiştir.

Örneğin:

- küçük workload: yaklaşık `326–350 MB`
- büyük context/batch workload: yaklaşık `730 MB`
- final heavy NVML workload: yaklaşık `1055 MB`

DistilGPT2 küçük bir model olduğu için 4 GB RTX 3050 üzerinde OOM oluşmamıştır.

Bu durum Easy workload seviyesinden beklenen davranıştır.

Medium ve Hard modellerde VRAM pressure'ın çok daha önemli hale gelmesi beklenmektedir.

---

# 12. Temsilci Workload'lar

Power/energy ölçümleri için üç workload seçilmiştir.

| Scenario | Context | Output | Batch |
|---|---:|---:|---:|
| Light | 32 | 32 | 1 |
| Medium | 128 | 64 | 4 |
| Heavy | 512 | 128 | 8 |

Bu üç workload Easy/Text davranış yüzeyinin farklı bölgelerini temsil etmektedir.

---

# 13. Mac MPS Final Power / Energy — AC

Mac final ölçümleri cihaz şarja bağlıyken yapılmıştır.

`powermetrics` üzerinden:

`CPU + GPU + ANE Combined Power`

ölçülmüştür.

| Scenario | Throughput | Combined Power | Energy / Run | Energy / Token |
|---|---:|---:|---:|---:|
| Light | **206.2 tok/s** | 6.74 W | 1.05 J | 0.0328 J/token |
| Medium | 551.3 tok/s | 7.04 W | 3.27 J | **0.0128 J/token** |
| Heavy | 588.7 tok/s | 13.20 W | 23.05 J | 0.0225 J/token |

### Gözlem

Heavy workload daha yüksek throughput üretse de batch çok daha büyüktür.

Bu nedenle workload'lar arasında energy efficiency değerlendirilirken yalnızca `J/run` değil, `J/token` da kullanılmalıdır.

Bu üç temsilci workload içinde Medium workload Mac açısından en düşük `J/token` değerini vermiştir.

---

# 14. RTX 3050 Final NVML Power / Energy

Final RTX ölçümleri `NVML` ile `100 ms` sampling kullanılarak yapılmıştır.

| Scenario | Throughput | GPU Power | GPU Utilization | VRAM | Energy / Run | Energy / Token |
|---|---:|---:|---:|---:|---:|---:|
| Light | 160.3 tok/s | 45.50 W | 39.7% | 571 MB | 9.11 J | 0.2846 J/token |
| Medium | 633.5 tok/s | 56.61 W | 56.5% | 611 MB | 22.88 J | 0.0894 J/token |
| Heavy | **733.5 tok/s** | 59.67 W | **89.6%** | 1055 MB | 83.30 J | **0.0813 J/token** |

### Gözlem

Workload büyüdükçe RTX utilization:

`39.7% → 56.5% → 89.6%`

şeklinde yükselmektedir.

Heavy workload RTX'i gerçek anlamda yüksek utilization bölgesine taşımaktadır.

Ayrıca RTX tarafında batch büyüdükçe `J/token` düşmektedir.

Bu durum GPU'nun yüksek parallel workload altında daha verimli kullanılabildiğini göstermektedir.

---

# 15. Mac MPS vs RTX 3050

Final throughput karşılaştırması:

| Scenario | Mac MPS | RTX 3050 | Daha yüksek throughput |
|---|---:|---:|---|
| Light | **206.2 tok/s** | 160.3 tok/s | Mac MPS |
| Medium | 551.3 tok/s | **633.5 tok/s** | RTX |
| Heavy | 588.7 tok/s | **733.5 tok/s** | RTX |

Bu sonuç önemli bir **crossover** göstermektedir.

### Light workload

Mac MPS daha hızlıdır.

### Medium workload

RTX avantaj elde etmektedir.

### Heavy workload

RTX farkı daha da açmaktadır.

Dolayısıyla:

> “Discrete GPU varsa her zaman onu kullan”

gibi statik bir scheduling politikası gözlenen davranışı doğru temsil etmemektedir.

Workload intensity arttıkça optimum execution device değişebilmektedir.

Bu sonuç ileride geliştirilecek resource-aware scheduler için doğrudan motivasyon oluşturmaktadır.

---

# 16. Energy Ölçüm Sınırı

Mac ve RTX enerji değerleri doğrudan mutlak sistem enerji karşılaştırması olarak yorumlanmamalıdır.

| Platform | Ölçülen Power Boundary |
|---|---|
| Mac | CPU + GPU + ANE Combined SoC Power |
| RTX | NVIDIA GPU Board Power |

Dolayısıyla:

- throughput doğrudan karşılaştırılabilir,
- aynı platform içindeki energy trendleri karşılaştırılabilir,
- workload büyüdükçe energy davranışı incelenebilir,
- ancak mevcut ölçümlerle “Mac RTX'ten X kat daha energy-efficient” sonucu çıkarılmamalıdır.

Mutlak whole-system karşılaştırması için ortak bir wall-power measurement yöntemi gereklidir.

---

# 17. Power Source Etkisi

İlk Mac ölçümlerinin bazıları batarya üzerinde gerçekleştirilmiştir.

Daha sonra final deneyler Mac şarja bağlıyken tekrarlanmıştır.

Throughput büyük ölçüde benzer kalırken power tüketiminde fark gözlenmiştir.

Bu nedenle bundan sonraki deneylerde:

`power_mode = AC / Battery`

ayrı bir deney metadata alanı olarak tutulmalıdır.

Final cross-device karakterizasyonlarda iki cihaz da AC power durumunda tutulmalıdır.

---

# 18. NVIDIA Monitoring Overhead Problemi

İlk RTX power ölçümlerinde `nvidia-smi` kullanılmıştır.

Ancak measurement tool'un workload performansını ciddi biçimde etkilediği gözlenmiştir.

Medium workload A/B testi:

| Measurement Method | Throughput |
|---|---:|
| No monitoring | 734.4 tok/s |
| nvidia-smi 500 ms | 477.1 tok/s |

Fark:

`-35.0%`

Bu nedenle `nvidia-smi` polling final power measurement yöntemi olarak reddedilmiştir.

---

# 19. NVML Doğrulaması

Daha düşük overhead için `NVML` kullanılmıştır.

A/B sonucu:

| Method | Throughput |
|---|---:|
| No monitoring | 655.3 tok/s |
| NVML 100 ms | 647.8 tok/s |

Fark:

`-1.1%`

Bu fark kabul edilebilir measurement overhead olarak değerlendirilmiştir.

Bu nedenle final RTX power/energy ölçümleri:

`NVML @ 100 ms`

ile gerçekleştirilmiştir.

---

# 20. Ana Bulgular

DistilGPT2 Easy/Text karakterizasyonundan aşağıdaki sonuçlar çıkarılmıştır.

### 1. Maximum CPU thread count optimum değildir

Hem Mac hem MSI deneylerinde maksimum thread sayısı daha kötü sonuç verebilmiştir.

---

### 2. CPU optimumu workload-dependent'tır

Mac üzerinde long single generation ile batch workload farklı optimum thread bölgeleri göstermiştir.

---

### 3. Performance-optimal ve energy-optimal aynı olmak zorunda değildir

MSI CPU:

- `2 threads` → energy-oriented
- `6 threads` → throughput-oriented

olarak davranmıştır.

---

### 4. Batch GPU throughput için kritik bir değişkendir

Batch büyüdükçe hem MPS hem CUDA üzerinde accelerator utilization daha etkili hale gelmektedir.

---

### 5. Context büyümesi GPU maliyetini artırmaktadır

Context length arttıkça throughput düşmekte ve memory pressure yükselmektedir.

---

### 6. Device seçimi workload-dependent'tır

Light workload:

`Mac MPS > RTX`

Medium / Heavy workload:

`RTX > Mac MPS`

Bu durum açık bir workload crossover oluşturmaktadır.

---

### 7. RTX heavy workload altında çok daha yüksek utilization'a ulaşmaktadır

Final GPU utilization:

- Light: `39.7%`
- Medium: `56.5%`
- Heavy: `89.6%`

---

### 8. Energy karşılaştırmasında J/token önemlidir

Farklı batch ve output büyüklüklerinde yalnızca J/run kullanmak yanıltıcı olabilir.

Bu nedenle bundan sonraki deneylerde en az:

- `J/run`
- `J/token`

birlikte kaydedilmelidir.

---

### 9. Measurement overhead mutlaka doğrulanmalıdır

`nvidia-smi` power monitoring workload'u yaklaşık `%35` yavaşlatmıştır.

NVML overhead ise yaklaşık `%1.1` seviyesinde kalmıştır.

Measurement aracının kendisi deney sonucunu değiştirebildiği için telemetry yöntemi benchmark metodolojisinin bir parçası olarak doğrulanmalıdır.

---

# 21. Scheduler Açısından İlk Sonuç

Easy/Text verisi bile tek bir statik execution rule'un yeterli olmadığını göstermektedir.

İleride scheduler'ın karar uzayı en az şu değişkenleri içermelidir:

`configuration = (device, threads, batch, context, precision)`

Scheduler'ın gözlemleyebileceği workload özellikleri:

- model size,
- modality,
- context length,
- expected output length,
- batch size,
- available memory,
- device utilization,
- latency target,
- energy budget.

İleride temel objective function şu yapıya genişletilebilir:

\[
J(c)=\alpha L(c)+\beta E(c)+\gamma M(c)
\]

Burada:

- `L` = latency
- `E` = energy
- `M` = memory/resource cost
- `c` = execution configuration

MoE ve distributed execution aşamasında buna communication ve quality maliyetleri de eklenecektir.

---

# 22. Deney Verisi İçin Önerilen Ortak Şema

Bundan sonraki Medium, Hard ve Vision deneylerinde aşağıdaki ortak dataset formatının kullanılması önerilmektedir:

```text
device
power_mode
modality
model
model_parameters
difficulty
context_length
output_tokens
batch_size
backend
cpu_threads
precision
quantization
latency_sec
throughput_tok_sec
power_w
energy_run_j
energy_token_j
memory_mb
utilization_percent
status
repeat
```

Bu yapı ileride scheduler training / profiling dataset'inin temelini oluşturabilir.

---

# 23. Easy/Text Durumu

## DistilGPT2 / Easy / Text

- Hardware characterization ✅
- Mac CPU thread sweep ✅
- MSI CPU thread sweep ✅
- Mac CPU power characterization ✅
- MSI CPU power characterization ✅
- Mac MPS workload matrix ✅
- RTX CUDA workload matrix ✅
- Context sweep ✅
- Output sweep ✅
- Batch sweep ✅
- Mac AC power measurement ✅
- RTX NVML power measurement ✅
- J/run ✅
- J/token ✅
- GPU utilization ✅
- VRAM measurement ✅
- Monitoring overhead validation ✅
- Mac ↔ RTX crossover analysis ✅

**Durum: COMPLETE**

---

# 24. Sonraki Aşama

Bir sonraki adım:

## Medium / Text

Önerilen model:

`Qwen2.5-1.5B-Instruct`

Burada aynı temel metodoloji korunacak ancak model büyüklüğü nedeniyle:

- compute pressure,
- memory pressure,
- accelerator utilization,
- CPU/GPU farkı,
- context sensitivity,
- energy behavior

çok daha belirgin hale gelecektir.

Daha sonra:

`Hard / Text → yaklaşık 3B model`

ve ardından:

`Easy / Medium / Hard Vision`

karakterizasyonuna geçilecektir.

Scheduler algoritmasına ancak bu karakterizasyon dataset'i tamamlandıktan sonra geçilecektir.