import sys
from predict import predict_speaker

def main():
    print("1. Model Eğitimi")
    print("2. Tahmin Yap")
    choice = input("Bir seçenek seçin (1/2): ")

    if choice == "1":
        import train_model
        print("Model eğitildi ve kaydedildi.")
    elif choice == "2":
        audio_file = input("Tahmin yapılacak ses dosyasının yolunu girin: ")
        result = predict_speaker(audio_file)
        print(f"Tahmin edilen kişi: {result}")
    else:
        print("Geçersiz seçenek.")

if __name__ == "__main__":
    main()
