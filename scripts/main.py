from train_model import train_model
from predict import predict_speaker
from live_prediction import live_predict

print("1. Model Eğitimi")
print("2. Tahmin Yap")
print("3. Anlık Ses Tanımlama")

choice = input("Bir seçenek seçin (1/2/3): ")

if choice == "1":
    train_model()
elif choice == "2":
    audio_file = input("Tahmin yapılacak ses dosyasının yolunu girin: ")
    result = predict_speaker(audio_file)
    print(f"Tahmin edilen kişi: {result}")
elif choice == "3":
    live_predict()
else:
    print("Geçersiz seçim!")
