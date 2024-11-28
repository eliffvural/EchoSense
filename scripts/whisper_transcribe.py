import os
from pydub import AudioSegment
from pydub.effects import normalize
import whisper

# Gürültü azaltma fonksiyonu
def reduce_noise(input_path, output_path):
    """
    Verilen ses dosyasındaki gürültüyü azaltarak normalize edilmiş bir dosya oluşturur.
    """
    if not os.path.exists(input_path):
        print(f"HATA: Ses dosyası bulunamadı: {input_path}")
        exit(1)

    audio = AudioSegment.from_file(input_path)
    normalized_audio = normalize(audio)  # Normalize işlemi
    normalized_audio.export(output_path, format="wav")  # Gürültü azaltılmış dosyayı kaydet

# Whisper ile ses transkripte çevirme fonksiyonu
def transcribe_audio(audio_path, language="tr"):
    """
    Whisper modelini kullanarak ses dosyasını transkripte çevirir.
    """
    model = whisper.load_model("base")  # Whisper modelini yükle
    result = model.transcribe(audio_path, language=language)  # Transkript işlemi
    return result

# Ana işlem
if __name__ == "__main__":
    # Windows için tam yollar
    input_audio_path = "C:\\Users\\elifv\\Desktop\\EchoSense-main\\dataset\\speaker1\\kayit7.wav"
    clean_audio_path = "C:\\Users\\elifv\\Desktop\\EchoSense-main\\dataset\\cleaned.wav"

    # Ses dosyasının varlığını kontrol et
    print("Ses dosyasındaki gürültü azaltılıyor...")
    reduce_noise(input_audio_path, clean_audio_path)
    print(f"Gürültü azaltılmış dosya kaydedildi: {clean_audio_path}")

    # Transkript işlemi
    print("Ses dosyası transkripte çevriliyor...")
    transcription_result = transcribe_audio(clean_audio_path, language="tr")
    print(f"Transkript: {transcription_result['text']}")
    print(f"Kelime Sayısı: {len(transcription_result['text'].split())}")
