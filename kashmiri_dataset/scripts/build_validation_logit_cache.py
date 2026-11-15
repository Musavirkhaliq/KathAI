import json, time, sys, unicodedata, hashlib
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.append("kashmiri_dataset/scripts")
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

EXP_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding")
CACHE_DIR = EXP_DIR / "logit_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model")
VAL_MANIFEST = Path("kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Stage 7 Phase B: Building Reusable Acoustic Logit Cache on {device} ===")

# Compute model checksum
model_bytes = (MODEL_DIR / "model.safetensors").read_bytes()
model_sha256 = hashlib.sha256(model_bytes).hexdigest()
vocab_sha256 = hashlib.sha256((MODEL_DIR / "vocab.json").read_bytes()).hexdigest()

val_samples = []
with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        dur = item.get("duration", 0)
        if 0.5 <= dur <= 16.0:
            val_samples.append(item)

print(f"Loaded {len(val_samples)} eligible validation samples.")

processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

batch_size = 16
cache_manifest = {
    "model_dir": str(MODEL_DIR),
    "model_sha256": model_sha256,
    "vocab_sha256": vocab_sha256,
    "precision": "float16",
    "subsampling_rate_hz": 50,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "num_samples": len(val_samples),
    "samples": {}
}

t_start = time.time()
parity_mismatches = 0
total_processed = 0

with torch.no_grad():
    for b_idx in range(0, len(val_samples), batch_size):
        batch_items = val_samples[b_idx:b_idx+batch_size]
        wavs = []
        raw_lens = []
        for item in batch_items:
            try:
                audio, sr = sf.read(item["audio_path"])
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                wavs.append(audio)
                raw_lens.append(len(audio))
            except Exception:
                wavs.append(np.zeros(16000, dtype=np.float32))
                raw_lens.append(16000)
                
        inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits  # shape [B, T_padded, 161]
            log_probs = F.log_softmax(logits, dim=-1)  # log probabilities
        
        # Process and save each sample in batch strictly up to unpadded frame length
        for idx_in_b, meta in enumerate(batch_items):
            sid = meta["sample_id"]
            dur = meta["duration"]
            audio_samples_len = raw_lens[idx_in_b]
            
            # Exact conv output length:
            t_frames = model._get_feat_extract_output_lengths(torch.tensor([audio_samples_len])).item()
            
            # Slice unpadded frames
            sample_log_probs = log_probs[idx_in_b, :t_frames, :].cpu().numpy().astype(np.float16)
            
            # Save to individual compressed npz
            npz_path = CACHE_DIR / f"{sid}.npz"
            np.savez_compressed(npz_path, log_probs=sample_log_probs, sample_id=sid, duration=dur)
            
            # Parity check: greedy decode from cached log_probs vs unpadded slice of logits
            cached_argmax = np.argmax(sample_log_probs, axis=-1)
            cached_hyp_text = processor.decode(cached_argmax)
            
            slice_logits = logits[idx_in_b, :t_frames, :]
            direct_hyp = processor.decode(torch.argmax(slice_logits, dim=-1))
            
            if cached_hyp_text != direct_hyp:
                parity_mismatches += 1
                if parity_mismatches <= 5:
                    print(f"[PARITY MISMATCH] {sid}: cached='{cached_hyp_text}' vs direct='{direct_hyp}'")
            
            cache_manifest["samples"][sid] = {
                "sample_id": sid,
                "file_path": str(npz_path.relative_to(EXP_DIR)),
                "num_frames": int(t_frames),
                "duration": dur,
                "source": meta.get("source") or meta.get("dataset_source") or "unknown",
                "text": meta.get("text", "")
            }
            
            total_processed += 1
            
        if total_processed % 800 == 0 or total_processed == len(val_samples):
            print(f"[{total_processed}/{len(val_samples)}] Cached ({time.time()-t_start:.1f}s elapsed, mismatches={parity_mismatches})", flush=True)

manifest_path = EXP_DIR / "cache_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(cache_manifest, f, indent=2)

print(f"\nSuccessfully cached {total_processed} utterances to {CACHE_DIR}")
print(f"Total parity mismatches: {parity_mismatches} / {total_processed}")
print(f"Saved manifest to {manifest_path}")
