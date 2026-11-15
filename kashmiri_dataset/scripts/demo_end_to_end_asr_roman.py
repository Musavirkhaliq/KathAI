import json
import soundfile as sf
import numpy as np
import torch
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from kashmiri_transliteration_engine import KashmiriTransliterationEngine

BASE_DIR = Path("/home/musa/Kashmiri Project")
MODEL_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model"
VAL_FILE = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl"

print("=" * 80)
print("LIVE INFERENCE: Kashmiri XLS-R 300M (Best Model) -> Dual Roman Transliteration")
print("=" * 80)

# 1. Load Model & Processor
print("Loading best ASR checkpoint from:", MODEL_DIR.name)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

# 2. Load Transliteration Engines
engine_ascii = KashmiriTransliterationEngine(mode="ascii")
engine_phonetic = KashmiriTransliterationEngine(mode="phonetic")

# 3. Select 8 representative audio samples from validation set
samples = []
with open(VAL_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i in [15, 45, 90, 130, 210, 310, 450, 600]:
            samples.append(json.loads(line))
        if len(samples) >= 8:
            break

print(f"\nProcessing {len(samples)} live audio recordings...\n")

for idx, sample in enumerate(samples, 1):
    sid = sample["sample_id"]
    audio_path = BASE_DIR / sample["audio_path"]
    ref_text = sample["text"]
    
    # Read audio
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    dur = len(audio) / sr
    
    # Forward pass
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            logits = model(inputs).logits
            
    pred_ids = torch.argmax(logits, dim=-1)[0]
    pred_perso_arabic = processor.decode(pred_ids).strip()
    
    # Transliterate
    roman_ascii = engine_ascii.transliterate(pred_perso_arabic)
    roman_phonetic = engine_phonetic.transliterate(pred_perso_arabic)
    ref_roman = engine_ascii.transliterate(ref_text)
    
    print(f"[{idx}] Utterance: {sid} ({dur:.2f}s) | Source: {sample.get('source') or sample.get('dataset_source', 'Kashmiri')}")
    print(f"    Target (Ground Truth) : {ref_text}")
    print(f"    ASR Recognized Text    : {pred_perso_arabic}")
    print(f"    ► Romanized (Colloquial): {roman_ascii}")
    print(f"    ► Romanized (Phonetic)  : {roman_phonetic}")
    print(f"    Target Romanized        : {ref_roman}")
    print("-" * 80)

print("\nLive end-to-end inference completed successfully!")
