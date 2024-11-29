import sounddevice as sd
import numpy as np
import wave
import whisper
import librosa
from transformers import pipeline

# Adım 1: Ses Kaydını Yapma
def record_audio(file_name, duration=5, samplerate=44100):
    print("Ses kaydı başlıyor...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print(f"Ses kaydı tamamlandı: {file_name}")
    
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

# Adım 2: Ses Kaydını Metne Çevirme
def transcribe_audio(file_name):
    model = whisper.load_model("base")
    result = model.transcribe(file_name)
    print("Ses metne çevriliyor...")
    return result['text']

# Adım 3: Kategoriye Ait Anahtar Kelimeler
category_keywords = {
    "Spor": ["futbol", "basketbol", "maç", "spor", "takım", "gol", "şampiyon", "stadyum", "rakip", "antrenman"],
    "Sağlık": ["hastane", "randevu", "doktor", "tedavi", "sağlık", "ameliyat", "doktor", "sağlık hizmeti", "ilaç", "kansere", "kanser"],
    "Teknoloji": ["yazılım", "donanım", "telefon", "bilgisayar", "yapay zeka", "iot", "web", "robotik", "yeni telefon", "geliştirici"],
    "Doğa Olayları": ["yağmur", "fırtına", "sel", "deprem", "volkan", "dolu", "tsunami", "doğa", "orman yangını", "kar fırtınası"],
    "Bilim": ["araştırma", "buluş", "keşif", "astronomi", "biyoloji", "fizik", "kimya", "laboratuvar", "genetik", "kuantum"],
    "Sanat": ["resim", "heykel", "sergi", "müze", "sanatçı", "performans", "dans", "müzik", "film", "sinema", "galeri"],
    "Eğitim": ["okul", "öğrenci", "öğretmen", "sınav", "ders", "eğitim", "üniversite", "kampüs", "öğrenme", "kitap"],
    "Toplum": ["kültür", "toplum", "aile", "göç", "sosyal", "yardım", "topluluk", "toplumun", "savaş", "barış"],
    "Dünya": ["haber", "ekonomi", "politik", "gündem", "savaş", "kriz", "yönetim", "ülke", "dünya", "başkan"],
    "Müzik": ["şarkı", "albüm", "sanatçı", "müzik", "konser", "melodi", "enstrüman", "rock", "pop", "rap", "klip"],
    "Psikoloji": ["duygu", "beyin", "psikolojik", "terapi", "ruh hali", "tedavi", "stres", "depresyon", "zihinsel", "kaygı"],
    "Çevre": ["iklim", "küresel ısınma", "çevre", "orman", "doğa", "plastik", "geri dönüşüm", "sıfır atık", "biyoçeşitlilik", "karbon salınımı"],
    "Tarih": ["eski", "medeni", "tarih", "imparatorluk", "yazıt", "kağıt", "müze", "arkeoloji", "civilizasyon", "kültür"],
}

# Adım 4: Kategori Tahmini Yapma
def predict_category(text):
    if not text.strip():  # Eğer metin boşsa, kategori atama yapma
        return "Metin yok"
    
    text = text.lower()
    scores = {category: 0 for category in category_keywords}  # Her kategori için puan başlatıyoruz

    # Her kategoriye ait anahtar kelimeleri kontrol ediyoruz
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text:
                scores[category] += 1  # Anahtar kelimeyi bulursak, o kategoriye puan ekliyoruz

    # En yüksek puanı alan kategoriyi döndürüyoruz
    predicted_category = max(scores, key=scores.get)

    # Eğer hiçbir kategori ile eşleşme yoksa
    if scores[predicted_category] == 0:
        return "Kategori bulunamadı"

    return predicted_category

# Adım 5: Duygu Tahmini Yapma
def predict_emotion(audio_file):
    # Ses dosyasını yükle
    audio, sr = librosa.load(audio_file)

    # Normalize edin
    audio = librosa.util.normalize(audio)

    # RMS (Enerji), Tempo ve Spektral Özellikler
    rms = librosa.feature.rms(y=audio)
    energy = np.mean(rms)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    
    # Pitch (Tizlik) Analizi
    pitch = extract_pitch(audio, sr)

    # Zero-Crossing Rate (ZCR)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    # Duygu Tahmini
    if energy > 0.1 and tempo > 130 and spectral_centroid > 2700 and pitch > 230 and zcr > 0.3:
        emotion = "Öfkeli"  # Yüksek enerji, hızlı tempo, yüksek pitch, yüksek zcr - Öfke
    elif energy > 0.08 and tempo > 120 and spectral_centroid > 2500 and pitch > 220 and zcr > 0.2:
        emotion = "Mutlu"   # Yüksek enerji, hızlı tempo, yüksek pitch - Mutluluk
    elif energy < 0.08 and tempo < 90 and spectral_centroid < 2300 and pitch < 180 and zcr < 0.2:
        emotion = "Üzgün"   # Düşük enerji, düşük tempo, düşük pitch, düşük zcr - Üzüntü
    elif energy > 0.08 and 110 <= tempo <= 130 and spectral_centroid > 2700 and zcr > 0.3:
        emotion = "Şaşkın"  # Yüksek enerji, orta tempo, yüksek spektral merkez - Şaşkınlık
    elif energy > 0.09 and tempo > 100 and spectral_centroid > 2200 and zcr > 0.3:
        emotion = "Korkmuş" # Yüksek enerji, hızla değişen sesler - Korku
    else:
        emotion = "Nötr"  # Diğer durumlar için bilinmeyen duygu
    
    return emotion

def extract_pitch(y, sr):
    # Librosa'nın y-voice pitch çıkarma fonksiyonu
    pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    # Sadece sesli (voiced) seslerden alınan pitch'leri dikkate alıyoruz
    pitch_values = pitch[voiced_flag]
    
    # Pitch'in ortalamasını hesaplayıp, sadece sıfırdan büyük olanları kullanıyoruz
    if len(pitch_values) > 0:
        return np.mean(pitch_values)
    else:
        return 0  # Hiçbir pitch değeri bulunamazsa sıfır döndür

# Ana Program
if __name__ == "__main__":
    # Adım 1: Ses Kaydını Yap
    audio_path = "live_audio.wav"
    record_audio(audio_path, duration=5)  # 5 saniyelik ses kaydı

    # Adım 2: Metne Çevir
    text = transcribe_audio(audio_path)
    print("Transkript:", text)

    # Adım 3: Kategori Belirleme
    category = predict_category(text)
    print(f"Kategori: {category}")

    # Adım 4: Duygu Tahmini
    emotion = predict_emotion(audio_path)
    print(f"Duygu: {emotion}")
