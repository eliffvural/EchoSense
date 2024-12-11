const startButton = document.getElementById("start-recording");
const stopButton = document.getElementById("stop-recording");
const status = document.getElementById("status");
const result = document.getElementById("result");

let mediaRecorder;
let audioChunks = [];

// Mikrofon erişimi ve kayıt başlatma
startButton.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.start();
        status.innerText = "Durum: Kayıt yapılıyor...";
        startButton.disabled = true;
        stopButton.disabled = false;

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
    } catch (error) {
        console.error("Mikrofon erişimi hatası:", error);
        status.innerText = "Durum: Mikrofon erişimi reddedildi.";
    }
});

// Kaydı durdurma ve backend'e gönderme
stopButton.addEventListener("click", async () => {
    mediaRecorder.stop();
    status.innerText = "Durum: Kayıt durduruldu.";
    startButton.disabled = false;
    stopButton.disabled = true;

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        audioChunks = []; // Eski veriyi temizle

        // Backend'e gönderme
        const formData = new FormData();
        formData.append("file", audioBlob, "recorded_audio.wav");

        status.innerText = "Durum: Ses analizi yapılıyor...";

        try {
            const response = await fetch("http://127.0.0.1:5000/analyze", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            status.innerText = "Durum: Analiz tamamlandı.";
            result.innerText = `Metin: ${data.text}\nKategori: ${data.category}\nDuygu: ${data.emotion}`;
        } catch (error) {
            console.error("Analiz hatası:", error);
            status.innerText = "Durum: Analiz sırasında hata oluştu.";
        }
    };
});
