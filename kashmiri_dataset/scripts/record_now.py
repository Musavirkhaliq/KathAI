import sys
import time
import subprocess
from pathlib import Path
from transcribe_audio import transcribe

BASE_DIR = Path("/home/musa/Kashmiri Project")
RECORDINGS_DIR = BASE_DIR / "kashmiri_dataset/recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

def record_audio(duration=5):
    timestamp = int(time.time())
    output_path = RECORDINGS_DIR / f"speech_{timestamp}.wav"
    
    print("\n" + "="*50)
    print(f"🎙️ PREPARING TO RECORD ({duration} SECONDS)")
    print("="*50)
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", flush=True)
        time.sleep(1)
        
    print(f"\n🔴 RECORDING NOW! Speak Kashmiri into your microphone: ({duration}s)...", flush=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "pulse",
        "-i", "default",
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("⏹️ Recording finished!\n")
    return output_path

if __name__ == "__main__":
    dur = 5
    if len(sys.argv) > 1:
        try:
            dur = int(sys.argv[1])
        except ValueError:
            pass
            
    try:
        wav_path = record_audio(dur)
        print("Transcribing with Kashmiri XLS-R 300M Model...")
        res = transcribe(wav_path)
        print("\n" + "="*55)
        print(f"Audio file: {wav_path.name} ({res['duration_sec']}s)")
        print("Kashmiri Perso-Arabic Transcription:")
        print(f"\n👉  {res['text']}\n")
        print("="*55 + "\n")
    except Exception as e:
        print(f"Error: {e}")
