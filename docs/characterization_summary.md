# Characterization Summary

## 1. Amaç

Bu aşamanın amacı, heterogeneous edge inference için resource-aware scheduler geliştirilmeden önce farklı model büyüklükleri, cihazlar ve execution configuration'larının davranışını karakterize etmektir.

İki modality ve üç difficulty seviyesi incelenmiştir:

| Modality | Easy | Medium | Hard |
|---|---|---|---|
| Text | DistilGPT2 | Qwen2.5-1.5B-Instruct | Qwen3-1.7B |
| Vision | ResNet50 | ConvNeXt-Base | ConvNeXt-Large |

Test edilen ana cihazlar:

- Apple M4
- Intel i7-11800H
- NVIDIA RTX 3050 Laptop GPU

Karakterizasyon; throughput, latency, batch, thread count, precision, memory, sustained performance, power ve energy/work-item değişkenlerini kapsamaktadır.

---

## 2. Tamamlanan Karakterizasyon

### Text Easy — DistilGPT2

DistilGPT2, küçük text workload davranışını temsil etmektedir.

Ana gözlem:

- hafif bazı workload'larda Mac rekabetçi veya avantajlıdır,
- workload büyüdükçe RTX avantajı ortaya çıkmaktadır,
- CPU thread sayısı ve batch büyüklüğü performansı anlamlı biçimde değiştirmektedir.

Bu seviye, küçük workload'larda accelerator seçiminin her zaman trivial olmadığını göstermiştir.

### Text Medium — Qwen2.5-1.5B-Instruct

Model büyüklüğü arttığında discrete GPU avantajı belirginleşmiştir.

Ana sonuç:

- RTX, full accelerator sweep içerisindeki 18/18 configuration'da daha yüksek throughput sağlamıştır.

Bu sonuç model büyüklüğünün device-selection kararında önemli bir feature olduğunu göstermektedir.

### Text Hard — Qwen3-1.7B

Native FP16 çalıştırmada Apple MPS uygulanabilirken RTX 3050'nin 4 GB VRAM kapasitesi native model yüklemesinde yetersiz kalmıştır.

Kontrollü karşılaştırma için Q4_K_M GGUF ve llama.cpp kullanılmıştır.

Ana sonuçlar:

- RTX, Q4 controlled sweep'te 18/18 configuration kazanmıştır,
- Mac ve MSI CPU için optimum thread sayısı workload'a göre değişmiştir,
- latency ve energy optimumları her zaman aynı configuration değildir,
- memory capacity execution strategy seçimini doğrudan değiştirebilir.

---

## 3. Vision Easy — ResNet50

ResNet50 küçük/orta ölçekli vision inference workload'u olarak karakterize edilmiştir.

Ana sonuçlar:

- short accelerator sweep'te MSI 18/18 configuration kazanmıştır,
- sustained representative workload'larda MSI 5/6 kazanmıştır,
- küçük FP16 light workload'da Mac avantajı gözlenmiştir,
- batch ve thread scaling cihazlar arasında farklı davranmıştır.

Bu seviye küçük workload'larda device crossover'ın mümkün olduğunu göstermiştir.

---

## 4. Vision Medium — ConvNeXt-Base

ConvNeXt-Base ile computational intensity artırılmıştır.

Ana sonuçlar:

- short crossover: MSI 18/18
- sustained crossover: MSI 6/6
- FP16 özellikle RTX üzerinde batch büyüdükçe ciddi throughput ve energy avantajı sağlamıştır,
- CPU optimum thread sayısı short ve sustained workload arasında değişebilmiştir,
- batch optimumu Mac ve MSI CPU için farklıdır.

Bu seviye workload büyüdükçe discrete GPU avantajının güçlendiğini göstermiştir.

---

## 5. Vision Hard — ConvNeXt-Large

ConvNeXt-Large yaklaşık 197.8M parametre ile Vision Hard workload'u temsil etmektedir.

Ana sonuçlar:

- short crossover: MSI 18/18
- sustained crossover: MSI 6/6
- FP16 RTX üzerinde sustained throughput'u Light → Heavy yönünde yaklaşık 2.16×, 2.52× ve 3.10× artırmıştır,
- FP16 peak accelerator memory kullanımını yaklaşık %50 azaltmıştır,
- RTX avantajı FP16 heavy workload'da yaklaşık 3.4× seviyesine çıkmıştır,
- Mac CPU'da speed ve energy optimumları farklı thread configuration'larında oluşmuştur,
- Mac MPS power ölçümlerinde session-state variability görülmüş ve üç-session median kullanılmıştır.

Bu seviye scheduler'ın yalnız cihaz değil precision, model büyüklüğü ve workload yoğunluğunu birlikte değerlendirmesi gerektiğini göstermektedir.

---

## 6. Ortak Bulgular

Altı workload birlikte değerlendirildiğinde aşağıdaki genel sonuçlar ortaya çıkmaktadır.

### Device seçimi model ve workload bağımlıdır

Tek bir cihaz bütün workload sınıfları için evrensel optimum değildir.

Küçük workload'larda Mac daha rekabetçi olabilirken, model ve workload büyüdükçe RTX throughput avantajı artmaktadır.

### Precision önemli bir scheduling değişkenidir

FP16 özellikle vision workload'larında:

- throughput'u artırabilmekte,
- memory kullanımını yaklaşık yarıya indirebilmekte,
- energy/work-item değerini düşürebilmektedir.

Ancak kazanç workload büyüklüğüne bağlıdır.

### Batch optimumu cihaz bağımlıdır

Aynı model üzerinde Mac ve MSI farklı batch optimumları gösterebilmektedir.

Bu nedenle global sabit batch configuration uygun değildir.

### Thread optimumu workload bağımlıdır

CPU inference için tek bir evrensel thread sayısı bulunmamıştır.

Optimum thread count:

- modele,
- input büyüklüğüne,
- batch'e,
- cihaz mimarisine,
- benchmark duration'a

bağlı olarak değişmektedir.

### Short benchmark her zaman sustained davranışı temsil etmez

Bazı MSI CPU workload'larında kısa test ile sustained test farklı optimum thread sayılarını göstermiştir.

Scheduler profiling aşamasında yalnız microbenchmark sonuçları kullanılmamalıdır.

### Minimum power ve minimum energy aynı değildir

Daha yüksek instantaneous power kullanan bir configuration daha kısa sürede tamamlanarak daha düşük energy/work-item sağlayabilir.

Bu nedenle scheduler objective'ında power ve energy ayrı değişkenler olarak ele alınmalıdır.

### Memory capacity execution strategy'yi değiştirebilir

Text Hard örneğinde native FP16 model Mac üzerinde çalışabilirken RTX 3050'nin VRAM kapasitesi nedeniyle kontrollü quantized execution gerekmiştir.

Memory capacity yalnız performans değil feasibility constraint olarak değerlendirilmelidir.

---

## 7. Scheduler İçin Çıkan Feature'lar

Karakterizasyon sonuçları scheduler input feature setinin en az aşağıdaki alanları içermesi gerektiğini göstermektedir:

- modality
- model
- model size
- workload size
- device
- backend
- precision / quantization
- batch size
- CPU threads
- memory requirement
- latency
- throughput
- power
- energy/work item
- execution duration / sustained state

Bunlara distributed aşamada:

- network RTT
- bandwidth
- payload size
- transfer latency
- communication energy

eklenecektir.

---

## 8. Measurement Boundary

Mac, MSI CPU ve RTX power ölçümleri aynı fiziksel boundary'yi temsil etmemektedir.

- Mac: combined SoC power
- MSI CPU: CPU Package power
- RTX: GPU board power

Bu nedenle absolute cross-device energy değerlerinden doğrudan enerji winner çıkarılmamaktadır.

Energy metriği öncelikle aynı cihaz içindeki configuration seçimlerini karşılaştırmak için kullanılmaktadır.

---

## 9. Mevcut Sınırlamalar

- Input'ların önemli bölümü synthetic workload'dur.
- Preprocessing ve decode maliyetleri vision inference süresine dahil değildir.
- Network / communication henüz dahil değildir.
- Accuracy / quality sistematik olarak ölçülmemiştir.
- FP16 veya quantization için aynı quality seviyesi varsayılmamalıdır.
- Yalnız iki ana fiziksel edge/consumer platform karakterize edilmiştir.

Bu sınırlamalar scheduler sonuçlarının yorumlanmasında açıkça korunacaktır.

---

## 10. Mevcut Durum

Tamamlanan karakterizasyon:

- Text Easy ✅
- Text Medium ✅
- Text Hard ✅
- Vision Easy ✅
- Vision Medium ✅
- Vision Hard ✅

Karakterizasyon aşaması tamamlanmıştır.

Bir sonraki aşama altı workload'dan elde edilen sonuçların ortak bir **Unified Characterization Dataset** içerisinde birleştirilmesi ve ardından crossover / scheduling analizinin yapılmasıdır.

Hedef akış:

**Characterization → Unified Dataset → Crossover Analysis → Resource-Aware Scheduler → MoE Expert Scheduling → Distributed Scheduling**