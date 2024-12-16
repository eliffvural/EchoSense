from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder="../Frontend")

# Model ve scaler yükleme
model = joblib.load("../Model/random_forest_model.joblib")
scaler = joblib.load("../Model/scaler.joblib")

# Konuşmacı etiketleri
SPEAKER_LABELS = {0: "Elif", 1: "İrem", 2: "Nazlı"}

# Kategori anahtar kelimeleri
CATEGORY_KEYWORDS = {
    "Spor": ["futbol", "basketbol", "maç", "spor", "takım"],
    "Sağlık": ["hastane", "doktor", "tedavi", "ilaç"],
    "Teknoloji": ["bilgisayar", "yapay zeka", "telefon"]
}

def predict_speaker(text):
    try:
        # Basit özellik çıkarımı
        text_length = len(text.split())
        char_count = len(text)

        # Özellik vektörü oluştur
        features = np.array([text_length, char_count]).reshape(1, -1)
        features_normalized = scaler.transform(features)

        # Tahmin yap
        prediction = model.predict(features_normalized)
        return SPEAKER_LABELS.get(prediction[0], "Bilinmiyor")
    except Exception as e:
        print(f"Konuşmacı tahmin hatası: {e}")
        return "Bilinmiyor"

def predict_category(text):
    text = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(word in text for word in keywords):
            return category
    return "Kategori Bulunamadı"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_text", methods=["POST"])
def process_text():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"speaker": "Bilinmiyor", "category": "Bilinmiyor"})

        speaker = predict_speaker(text)
        category = predict_category(text)

        return jsonify({"speaker": speaker, "category": category})
    except Exception as e:
        print(f"Hata: {e}")
        return jsonify({"error": "Sunucu hatası"}), 500

if __name__ == "__main__":
    app.run(debug=True)
