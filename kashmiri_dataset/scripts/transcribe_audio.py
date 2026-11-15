import sys
import torch
import unicodedata
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

BASE_DIR = Path("/home/musa/Kashmiri Project")
B02A_BEST = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02A/checkpoints/best_model"
B01_BEST = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model"
MODEL_DIR = B02A_BEST if B02A_BEST.exists() else B01_BEST

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Kashmiri ASR model from {MODEL_DIR} onto {device}...")
processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

def transcribe(audio_path):
    p = Path(audio_path)
    if not p.exists():
        print(f"Error: Audio file {audio_path} does not exist.")
        return None
        
    wav, sr = sf.read(str(p))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
        
    # Resample if needed
    if sr != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, 16000)
        wav = resampler(torch.from_numpy(wav).float()).numpy()
        sr = 16000

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs).logits[0]
            
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(pred_ids)
    
    duration = len(wav) / 16000.0
    return {
        "audio_path": str(p),
        "duration_sec": round(duration, 2),
        "transcription": transcription
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_audio.py <path_to_audio_file>")
        sys.exit(1)
        
    path = sys.argv[1]
    res = transcribe(path)
    if res:
        print("\n" + "="*50)
        print(f"Audio: {res['audio_path']} ({res['duration_sec']}s)")
        print(f"Transcription: {res['transcription']}")
        print("="*50 + "\n")
