import sounddevice as sd
from scipy.io.wavfile import write
from predict import predict_speaker

def record_audio(filename, duration=3, samplerate=44100):
    """
    Kullanıcıdan anlık ses kaydı alır ve bir dosyaya kaydeder.
    """
    print("Kayıt başlıyor...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Kayıt bitene kadar bekle
    write(filename, samplerate, audio_data)
    print(f"Kayıt tamamlandı: {filename}")

def live_predict():
    """
    Anlık ses kaydını alır ve tahmin işlemini gerçekleştirir.
    """
    print("Anlık Ses Tanımlama")
    output_file = "recorded_audio.wav"  # Kaydedilecek ses dosyasının adı
    
    # Ses kaydı al
    record_audio(output_file)

    # Tahmin fonksiyonunu çağır
    result = predict_speaker(output_file)
    print(f"Tahmin edilen kişi: {result}")

if __name__ == "__main__":
    live_predict()
