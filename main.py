import os
import librosa
import matplotlib.pyplot as plt


# Ses dosyasının yolu
audio_file = os.path.join('audio', 'kayit1.wav')

# Ses dosyasını yükleme
audio_file = './audio/kayit1.wav'  # Düz eğik çizgi ile
y, sr = librosa.load(audio_file)

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
