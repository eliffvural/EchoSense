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

# Veriyi numpy arraylerine donusturelim
X = np.array(x)
y = np.array(y)

# Veriyi kontrol edelim
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")

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
