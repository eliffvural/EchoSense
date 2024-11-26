import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Ses verilerinin bulunduğu dizinin yolu
dataset_path = './dataset'

# Özellikler ve etiketler
x = []  # MFCC özelliklerini saklayacak olan liste
y = []  # Etiketleri saklayacak olan liste

# Ses dosyalarını yükleyelim ve MFCC özelliklerini çıkaralım
for label, speaker_folder in enumerate(os.listdir(dataset_path)):
    speaker_folder_path = os.path.join(dataset_path, speaker_folder)
    
    # Eğer bu bir klasörse (her klasör bir kişiyi temsil eder)
    if os.path.isdir(speaker_folder_path):
        for file in os.listdir(speaker_folder_path):
            if file.endswith('.wav'):  # Sadece .wav dosyalarını al
                file_path = os.path.join(speaker_folder_path, file)
                
                # Ses dosyasını yükle
                y_audio, sr = librosa.load(file_path, sr=None)  # sr=None, orijinal sample rate'i kullanır
                
                # MFCC özelliklerini çıkar
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)  # 13 temel MFCC özelliği çıkar
                mfcc = np.mean(mfcc, axis=1)  # Özellikleri ortalamaya al (her ses dosyası için tek bir vektör)
                
                # Özellikleri ve etiketleri listeye ekle
                x.append(mfcc)
                y.append(label)  # Etiket, klasör sırasına göre atanır

# Veriyi numpy arraylerine dönüştürelim
X = np.array(x)
y = np.array(y)

# Veriyi kontrol edelim
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Ses dosyasının yolu
audio_file = './audio/kayit1.wav'

# Ses dosyasını yükleme
y, sr = librosa.load(audio_file)

# MFCC özelliklerini çıkarıyoruz
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# MFCC özelliklerinin boyutlarını kontrol ediyoruz
print(f"MFCC Shape: {mfcc.shape}")

# Ses bilgilerini yazdırma
print(f"Örnekleme oranı (Sample Rate): {sr}")
print(f"Sesin toplam uzunluğu (in seconds): {librosa.get_duration(y=y, sr=sr)}")

# Ses verisinin histogramını oluşturma
plt.figure(figsize=(10, 6))
plt.hist(y, bins=100, color='blue', alpha=0.7)  # Ses verisini histogram olarak çizer.
plt.title("Ses Verisi Histogramı")
plt.xlabel("Amplitüd")
plt.ylabel("Frekans")
plt.grid(True)
plt.show()  # Grafiği ekranda görüntüler
