import sounddevice as sd
import numpy as np
import wave
import whisper

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
        "Hijjj": ["seni", "çok", "seviyorum", "kalbim", "yazıt", "kağıt", "müze", "arkeoloji", "civilizasyon", "kültür"]

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
    print(f"Tespit edilen kategori: {category}")
