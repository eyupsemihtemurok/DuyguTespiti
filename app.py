import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from datetime import datetime

import torch
from torchvision import models, transforms
from torch import nn

# Ortam ayarı
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Etiketler
etiketler = {
    0: 'Öfkeli',
    1: 'Küçümseyen',
    2: 'İğrenmiş',
    3: 'Korkmuş',
    4: 'Mutlu',
    5: 'Nötr',
    6: 'Üzgün',
    7: 'Şaşırmış'
}

# Model yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 8)
model.load_state_dict(torch.load("D:/DuyguTespiti/duygu_modeli_resnet50.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Sayfa ayarı
st.set_page_config(page_title="Duygu Tespiti", layout="wide")
st.title("🧠 ResNet50 ile Duygu Tespiti (PyTorch)")

if "gecmis" not in st.session_state:
    st.session_state.gecmis = []

# Sidebar: Görsel kaynağı
st.sidebar.title("📸 Görsel Kaynağı Seç")
secenek = st.sidebar.radio("Görseli nasıl yüklemek istersin?", ["📁 Dosya Yükle", "📸 Fotoğraf Çek"])

image = None
img_array = None

if secenek == "📁 Dosya Yükle":
    dosya = st.sidebar.file_uploader("📁 Görsel Seçin", type=["jpg", "jpeg", "png"])
    if dosya:
        try:
            image = Image.open(dosya).convert("RGB")
            img_array = np.array(image)
        except:
            st.error("Görsel yüklenirken hata oluştu.")

elif secenek == "📸 Fotoğraf Çek":
    kamera_gorseli = st.camera_input("📷 Kameranızı kullanarak fotoğraf çekin")
    if kamera_gorseli:
        try:
            image = Image.open(kamera_gorseli).convert("RGB")
            img_array = np.array(image)
        except:
            st.error("Kamera görüntüsü alınırken hata oluştu.")

# Görsele yazı ekle
def yaziyi_ekle(image_np, text, x, y):
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.array(image_pil)

tahminli_gorsel = None

if img_array is not None:
    st.subheader("📷 Yüklenen / Çekilen Görsel ve Tahmin Sonucu")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    if st.button("▶️ Tahmin Et"):
        with st.spinner("Model çalışıyor..."):
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) == 0:
                    st.warning("Yüz tespit edilemedi.")
                else:
                    for (x, y, w, h) in faces:
                        roi = img_array[y:y + h, x:x + w]
                        roi_pil = Image.fromarray(roi).convert("RGB")
                        input_tensor = transform(roi_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(input_tensor)
                            predicted = torch.argmax(output, dim=1).item()

                        duygu = etiketler.get(predicted, "Bilinmeyen")
                        zaman = datetime.now().strftime("%H:%M:%S")

                        # Görsel üzerine kutu ve metin ekle
                        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        img_array = yaziyi_ekle(img_array, duygu, x, y - 30)

                        st.session_state.gecmis.append((zaman, duygu))
                        tahminli_gorsel = img_array

            except Exception as e:
                st.error(f"Hata oluştu: {e}")

    if tahminli_gorsel is not None:
        with col2:
            st.image(tahminli_gorsel, caption="Tahmin Sonucu", use_column_width=True)

# Tahmin geçmişi
if st.session_state.gecmis:
    st.sidebar.markdown("## 🕓 Tahmin Geçmişi")
    for t in st.session_state.gecmis[::-1][:5]:
        st.sidebar.markdown(f"- {t[0]} → {t[1]}")