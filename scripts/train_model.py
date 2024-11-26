import os
import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Ses verilerinin bulunduğu dizinin yolu
dataset_path = './dataset'

# Özellikler ve etiketler
x = []  # MFCC özelliklerini saklayacak olan liste
y = []  # Etiketleri saklayacak olan liste

# Ses dosyalarını yükleyip MFCC çıkar
for label, speaker_folder in enumerate(os.listdir(dataset_path)):
    speaker_folder_path = os.path.join(dataset_path, speaker_folder)
    if os.path.isdir(speaker_folder_path):
        for file in os.listdir(speaker_folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(speaker_folder_path, file)
                try:
                    y_audio, sr = librosa.load(file_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                    mfcc = np.mean(mfcc, axis=1)
                    x.append(mfcc)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Veriyi numpy arraylerine çevir
X = np.array(x)
y = np.array(y)

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Modeli değerlendir
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

# Modeli kaydet
joblib.dump(model, "./models/speaker_recognition_model.pkl")
print("Model kaydedildi: ./models/speaker_recognition_model.pkl")
