from audio_classification import transcribe_audio, predict_category, predict_emotion

# Manuel analiz testi
file_path = "uploads/recorded_audio.wav"
text = transcribe_audio(file_path)
category = predict_category(text)
emotion = predict_emotion(file_path)

print("Transcribed Text:", text)
print("Category:", category)
print("Emotion:", emotion)
