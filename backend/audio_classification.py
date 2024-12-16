import librosa
import numpy as np
import wave
import sounddevice as sd
import whisper
import joblib

# Model yükleme
model_path = "../Model/random_forest_model.joblib"
speaker_model = joblib.load(model_path)

# Özellik çıkarma fonksiyonu
def extract_audio_features(file_path):
    """
    Ses dosyasından MFCC, Chroma, RMS ve Zero-Crossing Rate özelliklerini çıkarır.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)

        # MFCC Özellikleri
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        # Chroma Özellikleri
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # RMS (Enerji)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)

        # Özellikleri birleştir
        features = np.hstack((mfcc_mean, chroma_mean, rms_mean, zcr_mean))
        return features

    except Exception as e:
        print(f"Özellik çıkarma hatası: {e}")
        return None

# Konuşmacı tahmini
def predict_speaker(file_path):
    """
    Ses dosyasını alır, özellikleri çıkarır ve konuşmacıyı tahmin eder.
    """
    features = extract_audio_features(file_path)
    if features is not None:
        features = features.reshape(1, -1)  # Model için yeniden şekillendir
        prediction = speaker_model.predict(features)[0]
        return prediction
    else:
        return "Bilinmiyor"

# Ses kaydını metne dönüştürme
def transcribe_audio(file_path):
    """
    Whisper modelini kullanarak ses dosyasını metne çevirir.
    """
    try:
        model = whisper.load_model("base")  # Whisper modeli yükle
        result = model.transcribe(file_path)  # Ses dosyasını transkript et
        return result['text']
    except Exception as e:
        print(f"Transkripsiyon hatası: {e}")
        return "Transkripsiyon yapılamadı"

# Kategori tahmini
category_keywords = {
    "Spor": ["futbol", "basketbol", "maç", "spor", "takım"],
    "Sağlık": ["hastane", "doktor", "tedavi", "ilaç"],
    "Teknoloji": ["bilgisayar", "yapay zeka", "telefon"]
}

def predict_category(text):
    """
    Metindeki anahtar kelimelere göre kategori tahmini yapar.
    """
    text = text.lower()
    scores = {category: sum(1 for word in keywords if word in text) 
              for category, keywords in category_keywords.items()}
    predicted_category = max(scores, key=scores.get)
    return predicted_category if scores[predicted_category] > 0 else "Kategori Bulunamadı"

# Test için örnek kullanım
if __name__ == "__main__":
    test_file = "test_audio.wav"
    
    # Transkripsiyon testi
    transcription = transcribe_audio(test_file)
    print("Transkript:", transcription)
    
    # Kategori testi
    category = predict_category(transcription)
    print("Kategori:", category)
    
    # Konuşmacı testi
    speaker = predict_speaker(test_file)
    print("Tahmin edilen konuşmacı:", speaker)
