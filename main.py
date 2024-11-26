import os
import pyaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Ses verilerinin bulunduğu dizinin yolu
dataset_path = './dataset'

# Ses kaydını almak için PyAudio kullaniyoruz, import edildi
p = pyaudio.PyAudio()

# Mikrofon ayarlarini belirliyoruz bu sekilde;
chunk = 1024  # Ses parçası boyutu
sample_rate = 16000  # Örnekleme oranı
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk)

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
                try:
                    y_audio, sr = librosa.load(file_path, sr=None)  # sr=None, orijinal sample rate'i kullanır
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                
                # MFCC özelliklerini çıkar
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)  # 13 temel MFCC özelliği çıkar
                mfcc = np.mean(mfcc, axis=1)  # Özellikleri ortalamaya al (her ses dosyası için tek bir vektör)
                
                # Özellikleri ve etiketleri listeye ekle
                x.append(mfcc)
                y.append(label)  # Etiket, klasör sırasına göre atanır

# Veriyi numpy arraylerine dönüştürelim
X = np.array(x)
y = np.array(y)

# Veriyi eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturalım ve eğitelim
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapalım
y_pred = model.predict(X_test)

# Doğruluk ve F1 Skoru hesaplayalım
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Sonuçları yazdıralım
print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")

# Veriyi kontrol edelim
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Ses dosyasının yolu
audio_file = './dataset/speaker2/kayit4.wav'

# Ses dosyasını yükleme
try:
    y_audio, sr = librosa.load(audio_file, sr=None)
except Exception as e:
    print(f"Error loading audio file {audio_file}: {e}")
    y_audio, sr = None, None

if y_audio is not None:
    # MFCC özelliklerini çıkarıyoruz
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)

    # MFCC özelliklerinin boyutlarını kontrol ediyoruz
    print(f"MFCC Shape: {mfcc.shape}")

    # Ses bilgilerini yazdırma
    print(f"Örnekleme oranı (Sample Rate): {sr}")
    print(f"Sesin toplam uzunluğu (in seconds): {librosa.get_duration(y=y_audio, sr=sr)}")

    # Ses verisinin histogramını oluşturma
    plt.figure(figsize=(10, 6))
    plt.hist(y_audio, bins=100, color='blue', alpha=0.7)  # Ses verisini histogram olarak çizer
    plt.title("Ses Verisi Histogramı")
    plt.xlabel("Amplitüd")
    plt.ylabel("Frekans")
    plt.grid(True)
    plt.show()  # Grafiği ekranda görüntüler
else:
    print("Ses dosyası yüklenemedi.")
