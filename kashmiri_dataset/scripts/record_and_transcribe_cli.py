import sys
import time
import subprocess
from pathlib import Path
from transcribe_audio import transcribe

BASE_DIR = Path("/home/musa/Kashmiri Project")
RECORDINGS_DIR = BASE_DIR / "kashmiri_dataset/recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

def record_audio(duration=5, output_wav="my_speech.wav"):
    output_path = RECORDINGS_DIR / output_wav
    print(f"\n🎙️ Recording {duration} seconds of audio to {output_path}...")
    print("👉 Speak now (کٲشُر بولِو)...")
    
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
    print("✅ Recording finished!")
    return output_path

if __name__ == "__main__":
    dur = 5
    if len(sys.argv) > 1:
        try:
            dur = int(sys.argv[1])
        except ValueError:
            pass

    timestamp = int(time.time())
    wav_file = f"recorded_{timestamp}.wav"
    try:
        recorded_path = record_audio(duration=dur, output_wav=wav_file)
        print("\nTranscribing with Kashmiri XLS-R CTC model...")
        res = transcribe(recorded_path)
        print("\n" + "="*55)
        print(f"🎙️ Recorded File: {recorded_path}")
        print(f"⏱️ Duration: {res['duration_sec']}s")
        print(f"📝 Kashmiri Perso-Arabic Transcription:")
        print(f"   {res['text']}")
        print("="*55 + "\n")
    except Exception as e:
        print(f"\n❌ Error during recording/transcription: {e}")
        print("💡 Alternatively, you can use the interactive Web UI with browser mic support:")
        print("   streamlit run kashmiri_dataset/scripts/asr_web_app.py --server.port 8501")
