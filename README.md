# 🧠 DuyguTespiti — ResNet50 ile Yüz İfadesi Tanıma

Bu proje, insan yüzlerindeki duygusal ifadeleri gerçek zamanlı olarak tespit eden bir derin öğrenme uygulamasıdır. **ResNet50** mimarisine dayalı bir yapay sinir ağı, **AffectNet** veri seti üzerinde eğitilerek 8 farklı duygu sınıfını tanıyabilmektedir. Uygulama, **Streamlit** çerçevesiyle sunulan etkileşimli bir web arayüzü aracılığıyla kullanıcıya sunulmaktadır.

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Sistem Gereksinimleri](#-sistem-gereksinimleri)
- [Kurulum](#-kurulum)
- [Proje Yapısı](#-proje-yapısı)
- [Kullanım](#-kullanım)
  - [Veri Ön İşleme](#1-veri-ön-işleme)
  - [Model Eğitimi](#2-model-eğitimi)
  - [Web Uygulaması](#3-web-uygulaması)
- [Veri Seti](#-veri-seti)
- [Model Mimarisi](#-model-mimarisi)
- [Duygu Kategorileri](#-duygu-kategorileri)
- [Uygulama Akışı](#-uygulama-akışı)
- [Bilinen Kısıtlamalar](#-bilinen-kısıtlamalar)
- [Katkıda Bulunma](#-katkıda-bulunma)

---

## ✨ Özellikler

- 📁 **Dosya yükleme** desteği (JPG / PNG)
- 📸 **Anlık fotoğraf** çekimi (tarayıcı kamerası)
- 👤 OpenCV **Haar Cascade** ile otomatik yüz tespiti
- 🤖 **ResNet50** tabanlı 8 sınıflı duygu sınıflandırması
- 🖼️ Yüz üzerine **sınır kutusu** ve **duygu etiketi** ekleme
- 🕓 Son 5 tahmini gösteren **geçmiş paneli**
- ⚡ GPU/CPU otomatik seçimi (CUDA varsa GPU kullanır)
- 🌐 **Streamlit** tabanlı tarayıcı üzerinden erişilebilir web arayüzü

---

## 💻 Sistem Gereksinimleri

| Bileşen | Minimum |
|---------|---------|
| **Python** | 3.9 veya üzeri |
| **RAM** | 4 GB (eğitim için 8 GB önerilir) |
| **Disk** | ~500 MB (model + bağımlılıklar) |
| **GPU** | İsteğe bağlı — CUDA destekli GPU kullanımı hızı artırır |

---

## ⚙️ Kurulum

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/eyupsemihtemurok/DuyguTespiti.git
cd DuyguTespiti
```

### 2. Sanal Ortam Oluşturun (Önerilir)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install torch torchvision streamlit opencv-python pillow pandas scikit-learn numpy
```

> **Not:** GPU kullanmak istiyorsanız sisteminize uygun PyTorch CUDA sürümünü [pytorch.org](https://pytorch.org/get-started/locally/) adresinden yükleyin.

### 4. Model Yolunu Güncelleyin

`app.py` dosyasının 31. satırındaki model yolu **Windows'a özgü sabit bir yol** içermektedir. Bu satırı kendi sisteminize göre güncelleyin:

```python
# Mevcut (değiştirilmesi gerekiyor):
model.load_state_dict(torch.load("D:/DuyguTespiti/duygu_modeli_resnet50.pth", map_location=device))

# Önerilen (projenin kendi klasörü):
model.load_state_dict(torch.load("duygu_modeli_resnet50.pth", map_location=device))
```

---

## 📁 Proje Yapısı

```
DuyguTespiti/
│
├── app.py                        # Streamlit web uygulaması (tahmin arayüzü)
├── train.py                      # Model eğitim betiği
├── labels.py                     # Metin etiketlerini sayısal değerlere dönüştürme
│
├── duygu_modeli_resnet50.pth     # Eğitilmiş model ağırlıkları (~91 MB, PyTorch)
├── labels.csv                    # 31.002 görsel için yol + metin etiket
├── labels_numeric.csv            # 31.002 görsel için yol + sayısal etiket (0–7)
│
├── .gitattributes                # Git satır sonu yapılandırması
└── README.md                     # Bu belge
```

### Dosya Açıklamaları

| Dosya | Açıklama |
|-------|----------|
| `app.py` | Streamlit arayüzü; yüz tespiti, model yükleme ve tahmin görselleştirmesini içerir |
| `train.py` | ResNet50 modelini AffectNet veri seti üzerinde eğiten betik |
| `labels.py` | `labels.csv` dosyasını okuyarak sayısal `labels_numeric.csv` üretir |
| `duygu_modeli_resnet50.pth` | Eğitim sonucu kaydedilen PyTorch model ağırlık dosyası |
| `labels.csv` | `pth` (görüntü yolu) ve `label` (metin) sütunlarından oluşan etiket dosyası |
| `labels_numeric.csv` | Metin etiketlerin 0–7 arasındaki sayılara dönüştürülmüş hali |

---

## 🚀 Kullanım

### 1. Veri Ön İşleme

Eğitim için önce `labels.csv` içindeki metin etiketleri sayısal değerlere dönüştürülmelidir. `labels.py` dosyasındaki sabit yolları projenizin konumuyla güncelledikten sonra çalıştırın:

```bash
python labels.py
```

Bu işlem, `labels_numeric.csv` dosyasını üretir. (Bu dosya depoda halihazırda mevcuttur.)

---

### 2. Model Eğitimi

> **Ön Koşul:** AffectNet görüntü veri setini indirip yerel bir klasöre çıkarın.

**Veri seti kaynağı:** <https://huggingface.co/datasets/chitradrishti/AffectNet/tree/main>

`train.py` dosyasındaki aşağıdaki satırları kendi ortamınıza göre güncelleyin:

```python
csv_path = r"D:\\DuyguTespiti/labels_numeric.csv"   # labels_numeric.csv dosyasının yolu
root_dir = r"D:\\DuyguTespiti"                       # Görüntülerin bulunduğu kök klasör
```

Ardından eğitimi başlatın:

```bash
python train.py
```

Eğitim tamamlandığında `duygu_modeli_resnet50.pth` dosyası oluşturulur. (Bu dosya depoda zaten mevcuttur.)

**Eğitim parametreleri:**

| Parametre | Değer |
|-----------|-------|
| Batch boyutu | 32 |
| Epoch sayısı | 30 |
| Öğrenme oranı | 1e-4 |
| Weight decay | 1e-5 |
| Optimizer | Adam |
| LR Scheduler | StepLR (adım=3, gamma=0.1) |
| Kayıp fonksiyonu | CrossEntropyLoss |
| Veri bölümü | %80 eğitim / %20 doğrulama |

---

### 3. Web Uygulaması

Aşağıdaki komutu çalıştırın ve tarayıcınızda `http://localhost:8501` adresini açın:

```bash
streamlit run app.py
```

**Kullanım adımları:**

1. Sol kenar çubuğundan görsel kaynağı seçin:
   - **📁 Dosya Yükle** — Bilgisayarınızdan JPG/PNG yükleyin
   - **📸 Fotoğraf Çek** — Tarayıcı kamerasıyla anlık fotoğraf alın
2. Görsel yüklendikten sonra **▶️ Tahmin Et** butonuna tıklayın.
3. Tespit edilen yüzler mavi bir sınır kutusuyla çerçevelenir ve üzerlerine duygu etiketi yazılır.
4. Sol kenar çubuğundaki **Tahmin Geçmişi** bölümü son 5 tahmini gösterir.

---

## 📊 Veri Seti

**AffectNet** veri seti, çeşitli arka planlar ve aydınlatma koşulları altında çekilmiş yüz ifadesi görüntüleri içermektedir.

| Duygu | Görüntü Sayısı |
|-------|---------------|
| Şaşırmış (surprise) | 4.889 |
| Mutlu (happy) | 4.382 |
| Öfkeli (anger) | 4.160 |
| İğrenmiş (disgust) | 3.776 |
| Korkmuş (fear) | 3.753 |
| Küçümseyen (contempt) | 3.588 |
| Üzgün (sad) | 3.352 |
| Nötr (neutral) | 3.102 |
| **Toplam** | **31.002** |

Sınıflar arasındaki dengesizliği gidermek için eğitimde **WeightedRandomSampler** kullanılmıştır.

---

## 🏗️ Model Mimarisi

### ResNet50 + Transfer Learning

```
Giriş Görseli (224×224×3)
        ↓
ResNet50 Gövdesi (ImageNet ön eğitimli ağırlıklar)
  • 1 Konvolüsyon katmanı
  • 4 Residüel blok grubu (toplam 48 konvolüsyon)
  • Global Average Pooling
        ↓
Özellik Vektörü (2048 boyut)
        ↓
Tam Bağlantılı Katman: 2048 → 8
        ↓
Tahmin (8 duygu sınıfı)
```

**Yüz Tespiti:** OpenCV `haarcascade_frontalface_default.xml`

**Veri Artırma (eğitim):**
- Rastgele yatay çevirme
- ±10° rastgele döndürme
- Parlaklık ve kontrast titremesi (ColorJitter)

---

## 😊 Duygu Kategorileri

| ID | İngilizce | Türkçe |
|----|-----------|--------|
| 0 | anger | Öfkeli |
| 1 | contempt | Küçümseyen |
| 2 | disgust | İğrenmiş |
| 3 | fear | Korkmuş |
| 4 | happy | Mutlu |
| 5 | neutral | Nötr |
| 6 | sad | Üzgün |
| 7 | surprise | Şaşırmış |

---

## 🔄 Uygulama Akışı

### Eğitim Hattı

```
AffectNet Görselleri + labels.csv
           ↓
       labels.py
(Metin etiket → Sayısal etiket)
           ↓
    labels_numeric.csv
           ↓
        train.py
(EmotionDataset + DataLoader + ResNet50)
           ↓
  duygu_modeli_resnet50.pth
```

### Tahmin Hattı

```
Kullanıcı Girdisi (Dosya / Kamera)
           ↓
        app.py
           ↓
OpenCV Haar Cascade → Yüz Bölgesi
           ↓
Ön İşleme (224×224 RGB Tensörü)
           ↓
  ResNet50 (model.eval())
           ↓
argmax → Duygu Etiketi
           ↓
Sınır Kutusu + Etiket Görseli
```

---

## ⚠️ Bilinen Kısıtlamalar

| Kısıtlama | Açıklama |
|-----------|----------|
| **Sabit dosya yolları** | `app.py`, `train.py` ve `labels.py` dosyalarında Windows'a özgü sabit yollar bulunmaktadır; farklı sistemlerde manuel olarak güncellenmesi gerekir |
| **`requirements.txt` yok** | Bağımlılıklar otomatik olarak yönetilmemektedir; paketlerin manuel kurulumu gerekmektedir |
| **Görüntü veri seti dahil değil** | AffectNet görüntüleri depo dışında tutulmaktadır; yalnızca etiket dosyaları paylaşılmıştır |
| **Haar Cascade sınırlılıkları** | Profil yüzler, aşırı ışık veya düşük çözünürlüklü görsellerde yüz tespiti başarısız olabilir |
| **Güven skoru gösterilmiyor** | Uygulama yalnızca en yüksek olasılıklı sınıfı göstermektedir; olasılık dağılımı ekranda görüntülenmemektedir |
| **Windows font yolu** | `app.py`'deki font yolu (`C:/Windows/Fonts/arial.ttf`) Windows'a özgüdür; Linux/macOS'ta varsayılan font kullanılır |

---

## 🤝 Katkıda Bulunma

1. Bu depoyu **fork**'layın
2. Yeni bir dal oluşturun: `git checkout -b ozellik/yeni-ozellik`
3. Değişikliklerinizi kaydedin: `git commit -m "Yeni özellik: açıklama"`
4. Dalınızı gönderin: `git push origin ozellik/yeni-ozellik`
5. Bir **Pull Request** açın

---

*Veri seti kaynağı: [AffectNet — Hugging Face](https://huggingface.co/datasets/chitradrishti/AffectNet/tree/main)*
