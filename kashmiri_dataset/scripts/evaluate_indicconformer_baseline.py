import json, time, sys
from pathlib import Path
from collections import Counter
import soundfile as sf
import numpy as np
import onnx_asr

sys.path.append("kashmiri_dataset/scripts")
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

EXP_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/A-01-indicconformer-kashmiri-finetune")
EXP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load exact B-01 validation reference records
B01_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc")
with open(B01_DIR / "validation_predictions.jsonl", "r", encoding="utf-8") as f:
    b01_records = [json.loads(line) for line in f]

# Build mapping from sample_id to metadata from val.jsonl
val_meta = {}
with open("kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        val_meta[item["sample_id"]] = item

print(f"Loaded {len(b01_records)} validation samples to evaluate on IndicConformer baseline.")

# 2. Load model
print("Loading IndicConformer ONNX model...")
model = onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-ks-onnx")

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

t_start = time.time()

for idx, rec in enumerate(b01_records):
    sid = rec["sample_id"]
    meta = val_meta.get(sid, {})
    audio_path = meta.get("audio_path") or rec.get("audio_path")
    ref_raw = rec["ref_raw"]
    ref_norm = rec["ref_norm"]
    
    # Audio duration
    dur = meta.get("duration", 0)
    total_audio_sec += dur
    
    # Recognize
    t0_rec = time.time()
    hyp_raw = model.recognize(audio_path)
    rec_time = time.time() - t0_rec
    
    # Hyp normalization using identical function
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
    
    pred_entry = {
        "sample_id": sid,
        "source": src,
        "speaker_id": meta.get("speaker_id"),
        "duration": dur,
        "latency_sec": round(rec_time, 4),
        "ref_raw": ref_raw,
        "hyp_raw": hyp_raw,
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "raw_wer": round(m_raw["wer"], 4),
        "raw_cer": round(m_raw["cer"], 4),
        "norm_wer": round(m_norm["wer"], 4),
        "norm_cer": round(m_norm["cer"], 4)
    }
    predictions.append(pred_entry)
    
    if (idx + 1) % 50 == 0:
        curr_wer = total_word_dist_norm / max(total_ref_words_norm, 1)
        curr_cer = total_char_dist_norm / max(total_ref_chars_norm, 1)
        print(f"[{idx+1}/{len(b01_records)}] Interim Norm WER: {curr_wer*100:.2f}%, Norm CER: {curr_cer*100:.2f}%")

total_eval_time = time.time() - t_start

raw_wer = total_word_dist_raw / max(total_ref_words_raw, 1)
raw_cer = total_char_dist_raw / max(total_ref_chars_raw, 1)
norm_wer = total_word_dist_norm / max(total_ref_words_norm, 1)
norm_cer = total_char_dist_norm / max(total_ref_chars_norm, 1)
empty_pct = (empty_count / len(predictions)) * 100
rtf = total_eval_time / max(total_audio_sec, 0.001)

# Source breakdown
source_breakdown = {}
for src, data in source_metrics.items():
    s_wer = data["word_dist_norm"] / max(data["ref_words_norm"], 1)
    s_cer = data["char_dist_norm"] / max(data["ref_chars_norm"], 1)
    source_breakdown[src] = {
        "sample_count": data["count"],
        "normalized_wer": round(s_wer, 4),
        "normalized_cer": round(s_cer, 4)
    }

# Save predictions
pred_file = EXP_DIR / "baseline_validation_predictions.jsonl"
with open(pred_file, "w", encoding="utf-8") as f:
    for p in predictions:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

# Summary metrics
summary = {
    "model": "ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large (Pretrained ONNX Baseline)",
    "evaluation_set": "B-01 Matched Validation Set (512 samples)",
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

summary_file = EXP_DIR / "baseline_validation_metrics.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== IndicConformer Pretrained Baseline Results ===")
print(f"Normalized WER: {norm_wer*100:.2f}%")
print(f"Normalized CER: {norm_cer*100:.2f}%")
print(f"Raw WER: {raw_wer*100:.2f}% | Raw CER: {raw_cer*100:.2f}%")
print(f"Empty Hypotheses: {empty_pct:.2f}%")
print(f"Total Wall-Clock Time: {total_eval_time:.1f}s | RTF: {rtf:.4f} ({total_audio_sec/total_eval_time:.1f}x real-time)")
print("Source-wise Normalized WER:")
for src, s_data in source_breakdown.items():
    print(f"  {src}: WER={s_data['normalized_wer']*100:.2f}%, CER={s_data['normalized_cer']*100:.2f}% (N={s_data['sample_count']})")
