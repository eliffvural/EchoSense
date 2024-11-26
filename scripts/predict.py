import joblib
import librosa
import numpy as np

def predict_speaker(audio_file):
    try:
        # Modeli yükle
        model = joblib.load("./models/speaker_recognition_model.pkl")

        # Ses dosyasını yükle
        y_audio, sr = librosa.load(audio_file, sr=None)

        # MFCC özelliklerini çıkar
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        mfcc = np.mean(mfcc, axis=1).reshape(1, -1)

        # Modelle tahmin yap
        prediction = model.predict(mfcc)
        return prediction[0]
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    audio_file = "./dataset/speaker3/kayit6.wav"
    result = predict_speaker(audio_file)
    print(f"Tahmin edilen kişi: {result}")
