import os
import librosa

# Ses dosyasının yolu
audio_file = os.path.join('audio', 'kayit1.wav')

# Ses dosyasını yükleme
audio_file='kayit1.wav'
y, sr = librosa.load(audio_file)

# Ses bilgilerini yazdırma
print(f"Örnekleme oranı (Sample Rate): {sr}")
print(f"Sesin toplam uzunluğu (in seconds): {librosa.get_duration(y=y, sr=sr)}")
