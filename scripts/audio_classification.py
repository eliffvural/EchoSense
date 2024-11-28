import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# **Adım 1: Mikrofondan Ses Kaydı**
def record_audio(output_path, duration=5, sample_rate=44100):
    print("Ses kaydı başlıyor...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    sd.wait()  # Kaydı tamamlamak için bekle
    write(output_path, sample_rate, recording)
    print(f"Ses kaydı tamamlandı: {output_path}")

# **Adım 2: Sesi Metne Çevir (Whisper)**
def transcribe_audio(audio_path, language="tr"):
    print("Ses metne çevriliyor...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=language)
    return result['text']

# **Adım 3: Kategori Analizi için Model Eğitimi**
def train_classifier():
    # Eğitim veri kümesi
    data = [
        ("Galatasaray maçı çok heyecanlıydı!", "Spor"),
        ("Yeni iPhone 15 özellikleri tanıtıldı.", "Teknoloji"),
        ("Dünya genelinde ekonomik kriz devam ediyor.", "Dünya"),
        ("Yeni sergi modern sanat eserlerini sergiliyor.", "Sanat"),
        ("Beşiktaş ve Fenerbahçe derbisi nefes kesti!", "Spor"),
        ("Tesla yeni model arabasını duyurdu.", "Teknoloji"),
        ("Birleşmiş Milletler zirvesinde önemli kararlar alındı.", "Dünya"),
        ("Ressamın yeni sergisi çok ilgi gördü.", "Sanat"),
    ]

    texts, labels = zip(*data)

    # Metni vektörize etme
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # Model eğitimi
    classifier = MultinomialNB()
    classifier.fit(X, labels)

    return classifier, vectorizer

# **Adım 4: Kategori Tahmini**
def predict_category(text, classifier, vectorizer):
    X_test = vectorizer.transform([text])
    return classifier.predict(X_test)[0]

# **Ana Fonksiyon**
if __name__ == "__main__":
    # Adım 1: Ses Kaydı
    audio_path = "live_audio.wav"
    record_audio(audio_path, duration=5)  # 5 saniyelik ses kaydı

    # Adım 2: Metne Çevir
    text = transcribe_audio(audio_path)
    print("Transkript:", text)

    # Adım 3: Modeli Eğit
    classifier, vectorizer = train_classifier()

    # Adım 4: Kategori Belirleme
    category = predict_category(text, classifier, vectorizer)
    print(f"Tespit edilen kategori: {category}")
