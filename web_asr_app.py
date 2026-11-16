#!/usr/bin/env python3
"""
Simple Web UI for Native Kashmiri Speech Recognition
Features:
  1. Direct Microphone Recording (Web Audio API MediaRecorder)
  2. Audio File Upload
  3. Real-time Transcribe via fine-tuned XLS-R 300M Kashmiri CTC model
"""
import os
import sys
import json
import base64
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from transcribe_kashmiri import KashmiriASR

print("Initializing Kashmiri ASR Model for Web App...")
asr = KashmiriASR()

HTML_PAGE = """<!DOCTYPE html>
<html lang="ks" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kashmiri ASR — Voice Recorder & Transcriber</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, -apple-system, sans-serif;
            background: #090d16;
            color: #f1f5f9;
            direction: rtl;
            text-align: right;
            padding: 30px 16px;
        }
        .container {
            max-width: 780px;
            margin: 0 auto;
            background: #131c2e;
            padding: 32px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
            border: 1px solid #1e293b;
        }
        header {
            text-align: center;
            margin-bottom: 28px;
        }
        h1 {
            color: #38bdf8;
            font-size: 28px;
            margin-bottom: 8px;
        }
        p.subtitle {
            color: #94a3b8;
            font-size: 14px;
        }
        .cards-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }
        @media(max-width: 600px) {
            .cards-grid { grid-template-columns: 1fr; }
        }
        .card {
            background: #0b1120;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .card-title {
            font-size: 15px;
            font-weight: bold;
            color: #cbd5e1;
            margin-bottom: 12px;
        }
        .record-btn {
            background: #ef4444;
            color: white;
            border: none;
            width: 70px;
            height: 70px;
            border-radius: 50%;
            font-size: 26px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
        }
        .record-btn:hover {
            transform: scale(1.05);
            background: #dc2626;
        }
        .record-btn.recording {
            background: #f59e0b;
            animation: pulse 1.5s infinite;
            box-shadow: 0 0 25px rgba(245, 158, 11, 0.6);
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .timer {
            margin-top: 10px;
            font-size: 14px;
            color: #e2e8f0;
            font-family: monospace;
            display: none;
        }
        input[type="file"] { display: none; }
        label.upload-btn {
            background: #2563eb;
            color: white;
            padding: 12px 24px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            display: inline-block;
            transition: 0.2s;
            font-size: 14px;
        }
        label.upload-btn:hover { background: #1d4ed8; }
        .file-info {
            margin-top: 10px;
            color: #94a3b8;
            font-size: 12px;
            direction: ltr;
        }
        audio {
            width: 100%;
            margin-top: 14px;
            height: 36px;
            display: none;
        }
        .action-btn {
            width: 100%;
            background: #10b981;
            color: white;
            border: none;
            padding: 16px;
            border-radius: 12px;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
        }
        .action-btn:hover { background: #059669; }
        .action-btn:disabled {
            background: #334155;
            color: #64748b;
            cursor: not-allowed;
            box-shadow: none;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 16px;
            color: #38bdf8;
            font-size: 15px;
            font-weight: bold;
        }
        .result-section {
            margin-top: 24px;
            background: #090d16;
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 24px;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .result-title {
            color: #38bdf8;
            font-size: 15px;
            font-weight: bold;
        }
        .copy-btn {
            background: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            padding: 4px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }
        .copy-btn:hover { color: #f8fafc; background: #334155; }
        .transcript-display {
            font-size: 22px;
            line-height: 1.8;
            color: #f8fafc;
            font-family: 'Scheherazade New', 'Noto Naskh Arabic', Arial, sans-serif;
            min-height: 60px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎙️ کٲشُر وائس ریکارڈر تہٕ ٹرانسکرایِبر</h1>
            <p class="subtitle">Direct Voice Recording & Automatic Speech Recognition in Native Kashmiri</p>
        </header>

        <div class="cards-grid">
            <!-- Voice Recording Card -->
            <div class="card">
                <div class="card-title">🎙️ پَنُن آواز ریکارڑ کٔرِو (Record Voice)</div>
                <button class="record-btn" id="recordBtn" onclick="toggleRecording()">🎤</button>
                <div class="timer" id="recordTimer">00:00</div>
                <audio id="recordedAudio" controls></audio>
            </div>

            <!-- Upload File Card -->
            <div class="card">
                <div class="card-title">📁 یا آڈیو فٲیل اپلوڈ کٔرِو (Or Upload Audio)</div>
                <input type="file" id="fileUpload" accept="audio/*">
                <label for="fileUpload" class="upload-btn">فٲیل ژارِو (Select File)</label>
                <div class="file-info" id="fileInfo">No file selected</div>
                <audio id="uploadedAudio" controls></audio>
            </div>
        </div>

        <button class="action-btn" id="transcribeBtn" onclick="transcribeAudio()" disabled>
            ⚡ ٹرانسکرایِب کٔرِو (Transcribe Now)
        </button>

        <div class="loading" id="loadingIndicator">
            ⏳ پروسیسنگ جاری چھِ، مہرElement بٲنۍ صَبٕر کٔرِو (Transcribing Audio)...
        </div>

        <div class="result-section">
            <div class="result-header">
                <div class="result-title">📝 کٲشُر ٹرانسکرپٹ (Kashmiri Transcript):</div>
                <button class="copy-btn" onclick="copyTranscript()">نقل کٔرِو (Copy)</button>
            </div>
            <div class="transcript-display" id="transcriptOutput">آواز ریکارڑ کٔرِو یا اپلوڈ کٔرِتھ "ٹرانسکرایِب کٔرِو" بَٹنس دَبٲیِو...</div>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let recordedBlob = null;
        let recordInterval;
        let recordSeconds = 0;

        const recordBtn = document.getElementById('recordBtn');
        const recordTimer = document.getElementById('recordTimer');
        const recordedAudio = document.getElementById('recordedAudio');
        const fileUpload = document.getElementById('fileUpload');
        const fileInfo = document.getElementById('fileInfo');
        const uploadedAudio = document.getElementById('uploadedAudio');
        const transcribeBtn = document.getElementById('transcribeBtn');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const transcriptOutput = document.getElementById('transcriptOutput');

        // File upload handler
        fileUpload.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                fileInfo.textContent = file.name + ' (' + (file.size/1024/1024).toFixed(2) + ' MB)';
                uploadedAudio.src = URL.createObjectURL(file);
                uploadedAudio.style.display = 'block';
                recordedAudio.style.display = 'none';
                recordedBlob = null;
                transcribeBtn.disabled = false;
            }
        });

        // Microphone Recording
        async function toggleRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                clearInterval(recordInterval);
                recordBtn.classList.remove('recording');
                recordBtn.innerHTML = '🎤';
                recordTimer.style.display = 'none';
            } else {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = () => {
                        recordedBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        recordedAudio.src = URL.createObjectURL(recordedBlob);
                        recordedAudio.style.display = 'block';
                        uploadedAudio.style.display = 'none';
                        fileUpload.value = '';
                        fileInfo.textContent = 'Voice recording ready';
                        transcribeBtn.disabled = false;
                    };

                    mediaRecorder.start();
                    recordBtn.classList.add('recording');
                    recordBtn.innerHTML = '⏹️';
                    recordSeconds = 0;
                    recordTimer.textContent = '00:00';
                    recordTimer.style.display = 'block';

                    recordInterval = setInterval(() => {
                        recordSeconds++;
                        const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                        const s = String(recordSeconds % 60).padStart(2, '0');
                        recordTimer.textContent = `${m}:${s}`;
                    }, 1000);

                } catch (err) {
                    alert('مائیکروفون دستیاب چھُنہٕ یا اجازت مِلٮ۪ے نہٕ (Microphone permission denied / not available): ' + err);
                }
            }
        }

        async function transcribeAudio() {
            let blobToSend = null;
            let filename = "audio.wav";

            if (recordedBlob) {
                blobToSend = recordedBlob;
                filename = "recording.wav";
            } else if (fileUpload.files.length > 0) {
                blobToSend = fileUpload.files[0];
                filename = fileUpload.files[0].name;
            }

            if (!blobToSend) return;

            loadingIndicator.style.display = 'block';
            transcribeBtn.disabled = true;
            transcriptOutput.textContent = 'پروسیسنگ جاری چھِ...';

            const reader = new FileReader();
            reader.onload = async function() {
                const base64Audio = reader.result.split(',')[1];
                try {
                    const res = await fetch('/transcribe', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            filename: filename,
                            audio_base64: base64Audio
                        })
                    });
                    const data = await res.json();
                    if (data.transcript) {
                        transcriptOutput.textContent = data.transcript;
                    } else {
                        transcriptOutput.textContent = 'Error: ' + (data.error || 'Failed to transcribe');
                    }
                } catch(e) {
                    transcriptOutput.textContent = 'Network error: ' + e;
                } finally {
                    loadingIndicator.style.display = 'none';
                    transcribeBtn.disabled = false;
                }
            };
            reader.readAsDataURL(blobToSend);
        }

        function copyTranscript() {
            const text = transcriptOutput.textContent;
            navigator.clipboard.writeText(text).then(() => {
                alert('ٹرانسکرپٹ گۆو کاپی (Transcript copied to clipboard)!');
            });
        }
    </script>
</body>
</html>
"""

class ASRRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/transcribe":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))
            
            audio_bytes = base64.b64decode(req["audio_base64"])
            ext = os.path.splitext(req["filename"])[1] or ".wav"
            
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                transcript = asr.transcribe(tmp_path)
                resp = {"transcript": transcript}
            except Exception as e:
                resp = {"error": str(e)}
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=7860):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ASRRequestHandler)
    print(f"\n=======================================================")
    print(f"  Kashmiri Voice Recorder & ASR App: http://localhost:{port}")
    print(f"=======================================================\n")
    httpd.serve_forever()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    run_server(port)
