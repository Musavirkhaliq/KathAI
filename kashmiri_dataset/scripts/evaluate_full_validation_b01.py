import json, time, sys, unicodedata
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.append("kashmiri_dataset/scripts")
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

EXP_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc")
MODEL_DIR = EXP_DIR / "best_model"
VAL_MANIFEST = Path("kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

val_samples = []
with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        dur = item.get("duration", 0)
        # Apply identical length filter (0.5s <= dur <= 16.0s)
        if 0.5 <= dur <= 16.0:
            val_samples.append(item)

print(f"Loaded {len(val_samples)} total length-valid validation samples.")

print(f"Loading B-01 best model from {MODEL_DIR}...")
processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

predictions = []
total_word_dist_raw = 0
total_ref_words_raw = 0
total_word_dist_norm = 0
total_ref_words_norm = 0

total_char_dist_raw = 0
total_ref_chars_raw = 0
total_char_dist_norm = 0
total_ref_chars_norm = 0

empty_count = 0
total_audio_sec = 0
source_metrics = {}

batch_size = 16
t_start = time.time()

with torch.no_grad():
    for b_idx in range(0, len(val_samples), batch_size):
        batch_items = val_samples[b_idx:b_idx+batch_size]
        wavs = []
        valid_items = []
        for item in batch_items:
            try:
                audio, sr = sf.read(item["audio_path"])
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                wavs.append(audio)
                valid_items.append(item)
            except Exception as e:
                # Fallback if audio decode fails
                wavs.append(np.zeros(16000, dtype=np.float32))
                valid_items.append(item)
        
        t0_rec = time.time()
        inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_values)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_texts = processor.batch_decode(pred_ids)
        rec_time = time.time() - t0_rec
        
        per_sample_latency = rec_time / len(batch_items)
        
        for idx_in_b, meta in enumerate(valid_items):
            sid = meta["sample_id"]
            ref_raw = meta.get("text", "")
            hyp_raw = pred_texts[idx_in_b]
            dur = meta.get("duration", 0)
            total_audio_sec += dur
            
            ref_norm = normalize_perso_arabic_kashmiri(ref_raw)
            hyp_norm = normalize_perso_arabic_kashmiri(hyp_raw)
            
            if len(hyp_raw.strip()) == 0:
                empty_count += 1
                
            m_raw = calculate_wer_and_cer(ref_raw, hyp_raw)
            m_norm = calculate_wer_and_cer(ref_norm, hyp_norm)
            
            total_word_dist_raw += m_raw["word_dist"]
            total_ref_words_raw += m_raw["ref_words"]
            total_char_dist_raw += m_raw["char_dist"]
            total_ref_chars_raw += m_raw["ref_chars"]
            
            total_word_dist_norm += m_norm["word_dist"]
            total_ref_words_norm += m_norm["ref_words"]
            total_char_dist_norm += m_norm["char_dist"]
            total_ref_chars_norm += m_norm["ref_chars"]
            
            src = meta.get("source") or meta.get("dataset_source") or "unknown"
            if src not in source_metrics:
                source_metrics[src] = {"word_dist_norm": 0, "ref_words_norm": 0, "char_dist_norm": 0, "ref_chars_norm": 0, "count": 0}
            source_metrics[src]["word_dist_norm"] += m_norm["word_dist"]
            source_metrics[src]["ref_words_norm"] += m_norm["ref_words"]
            source_metrics[src]["char_dist_norm"] += m_norm["char_dist"]
            source_metrics[src]["ref_chars_norm"] += m_norm["ref_chars"]
            source_metrics[src]["count"] += 1
            
            predictions.append({
                "sample_id": sid,
                "source": src,
                "speaker_id": meta.get("speaker_id"),
                "duration": dur,
                "latency_sec": round(per_sample_latency, 4),
                "ref_raw": ref_raw,
                "hyp_raw": hyp_raw,
                "ref_norm": ref_norm,
                "hyp_norm": hyp_norm,
                "raw_wer": round(m_raw["wer"], 4),
                "raw_cer": round(m_raw["cer"], 4),
                "norm_wer": round(m_norm["wer"], 4),
                "norm_cer": round(m_norm["cer"], 4)
            })
            
        cur_done = len(predictions)
        if (cur_done // batch_size) % 25 == 0 or cur_done == len(val_samples):
            curr_wer = total_word_dist_norm / max(total_ref_words_norm, 1)
            curr_cer = total_char_dist_norm / max(total_ref_chars_norm, 1)
            print(f"[{cur_done}/{len(val_samples)}] Interim Norm WER: {curr_wer*100:.2f}%, Norm CER: {curr_cer*100:.2f}% (Elapsed: {time.time()-t_start:.1f}s)")

total_eval_time = time.time() - t_start

raw_wer = total_word_dist_raw / max(total_ref_words_raw, 1)
raw_cer = total_char_dist_raw / max(total_ref_chars_raw, 1)
norm_wer = total_word_dist_norm / max(total_ref_words_norm, 1)
norm_cer = total_char_dist_norm / max(total_ref_chars_norm, 1)
empty_pct = (empty_count / len(predictions)) * 100
rtf = total_eval_time / max(total_audio_sec, 0.001)

source_breakdown = {}
for src, data in source_metrics.items():
    s_wer = data["word_dist_norm"] / max(data["ref_words_norm"], 1)
    s_cer = data["char_dist_norm"] / max(data["ref_chars_norm"], 1)
    source_breakdown[src] = {
        "sample_count": data["count"],
        "normalized_wer": round(s_wer, 4),
        "normalized_cer": round(s_cer, 4)
    }

pred_file = EXP_DIR / "full_val_b01_predictions.jsonl"
with open(pred_file, "w", encoding="utf-8") as f:
    for p in predictions:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

summary = {
    "model": "B-01 (facebook/wav2vec2-xls-r-300m + Kashmiri Character CTC, Epoch 13)",
    "evaluation_set": "Complete Length-Valid Validation Split (6,728 samples)",
    "num_samples": len(predictions),
    "total_audio_hours": round(total_audio_sec / 3600, 4),
    "raw_wer": round(raw_wer, 4),
    "raw_cer": round(raw_cer, 4),
    "normalized_wer": round(norm_wer, 4),
    "normalized_cer": round(norm_cer, 4),
    "empty_hypothesis_percentage": round(empty_pct, 2),
    "total_evaluation_time_sec": round(total_eval_time, 2),
    "real_time_factor_rtf": round(rtf, 4),
    "throughput_x_realtime": round(total_audio_sec / max(total_eval_time, 0.001), 2),
    "source_breakdown": source_breakdown
}

summary_file = EXP_DIR / "full_val_b01_metrics.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== Full Validation Set B-01 (XLS-R 300M) Results ===")
print(f"Total Samples: {len(predictions):,} ({total_audio_sec/3600:.2f} audio hours)")
print(f"Normalized WER: {norm_wer*100:.2f}%")
print(f"Normalized CER: {norm_cer*100:.2f}%")
print(f"Raw WER: {raw_wer*100:.2f}% | Raw CER: {raw_cer*100:.2f}%")
print(f"Empty Hypotheses: {empty_pct:.2f}%")
print(f"Total Wall-Clock Time: {total_eval_time/60:.2f}m | RTF: {rtf:.4f} ({total_audio_sec/total_eval_time:.1f}x real-time)")
print("Source Breakdown:")
for src, s_data in source_breakdown.items():
    print(f"  {src}: WER={s_data['normalized_wer']*100:.2f}%, CER={s_data['normalized_cer']*100:.2f}% (N={s_data['sample_count']:,})")
