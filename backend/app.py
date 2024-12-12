from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS modülünü içe aktarın
from audio_classification import record_audio, transcribe_audio, predict_category, predict_emotion
import os

app = Flask(__name__)
CORS(app)  # Tüm endpoint'ler için CORS'u etkinleştir
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return "EchoSense Backend API çalışıyor!"

@app.route('/record', methods=['POST'])
def record():
    duration = request.json.get('duration', 5)  # Varsayılan süre 5 saniye
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.wav")
    record_audio(file_path, duration=duration)
    return jsonify({'message': 'Audio recorded successfully', 'file_path': file_path})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_audio.wav")
    file.save(file_path)
    text = transcribe_audio(file_path)
    return jsonify({'text': text})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_audio.wav")
        file.save(file_path)

        # Analiz işlemleri
        text = transcribe_audio(file_path)
        category = predict_category(text)
        emotion = predict_emotion(file_path)

        return jsonify({'text': text, 'category': category, 'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
