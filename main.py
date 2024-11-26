import os
import librosa
import matplotlib.pyplot as plt
import numpy as np


# Ses verilerinin bulundugu dizinin yolu
dataset_path='./dataset'

# Ozellikler ve etiketler
x=[] # MFCC ozelliklerini saklayacak olan liste
y=[] # Etiketleri saklayacak olan liste

# Ses dosyalarini yukleyelim ve MFCC ozelliklerini cikaralim
for label, speaker_folder in enumerate(os.listdir(dataset_path)):
    speaker_folder_path = os.path.join(dataset_path, speaker_folder)

# Ses dosyasının yolu
audio_file = os.path.join('audio', 'kayit1.wav')

# Ses dosyasını yükleme
audio_file = './audio/kayit1.wav'  # Düz eğik çizgi ile
y, sr = librosa.load(audio_file)

# MFCC ozelliklerini cikariyoruz
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# MFCC ozelliklerinin boyutlarini kontrol ediyoruz
print(mfcc.shape)


# Ses bilgilerini yazdırma
print(f"Örnekleme oranı (Sample Rate): {sr}")
print(f"Sesin toplam uzunluğu (in seconds): {librosa.get_duration(y=y, sr=sr)}")

# Ses verisinin histogramini olusturma
plt.figure(figsize=(10,6))
plt.hist(y, bins=100, color='blue', alpha=0.7) #ses verisini histogram olarak cizer.
plt.title("Ses Verisi Histogramı")
plt.xlabel("Amplitüd")
plt.ylabel("Frekans")
plt.grid(True)
plt.show() #grafigi ekranda goruntuler
