import json, time, sys, unicodedata
from pathlib import Path
import soundfile as sf
import numpy as np
import onnx_asr

sys.path.append("kashmiri_dataset/scripts")
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

OUT_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/stage6_held_out_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_MANIFEST = Path("kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl")

print("=== Stage 6 Held-Out Test Evaluation: IndicConformer Baseline ===")

test_samples = []
with open(TEST_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        dur = item.get("duration", 0)
        # Apply pre-registered inclusion criteria
        if 0.5 <= dur <= 16.0:
            t_frames = int(dur * 50)
            target_len = len(unicodedata.normalize("NFKC", item.get("text", "")))
            if target_len < t_frames and target_len > 0:
                test_samples.append(item)

print(f"Loaded {len(test_samples)} eligible held-out test samples.")

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

for idx, meta in enumerate(test_samples):
    sid = meta["sample_id"]
    audio_path = meta.get("audio_path")
    ref_raw = meta.get("text", "")
    ref_norm = normalize_perso_arabic_kashmiri(ref_raw)
    
    dur = meta.get("duration", 0)
    total_audio_sec += dur
    
    t0_rec = time.time()
    try:
        hyp_raw = model.recognize(audio_path)
    except Exception as e:
        hyp_raw = ""
    rec_time = time.time() - t0_rec
    
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
        "latency_sec": round(rec_time, 4),
        "ref_raw": ref_raw,
        "hyp_raw": hyp_raw,
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "raw_wer": round(m_raw["wer"], 4),
        "raw_cer": round(m_raw["cer"], 4),
        "norm_wer": round(m_norm["wer"], 4),
        "norm_cer": round(m_norm["cer"], 4)
    })
    
    if (idx + 1) % 250 == 0 or (idx + 1) == len(test_samples):
        curr_wer = total_word_dist_norm / max(total_ref_words_norm, 1)
        curr_cer = total_char_dist_norm / max(total_ref_chars_norm, 1)
        print(f"[{idx+1}/{len(test_samples)}] Interim Norm WER: {curr_wer*100:.2f}%, Norm CER: {curr_cer*100:.2f}% (Elapsed: {time.time()-t_start:.1f}s)", flush=True)

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

pred_file = OUT_DIR / "indicconformer_test_predictions.jsonl"
with open(pred_file, "w", encoding="utf-8") as f:
    for p in predictions:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

summary = {
    "model": "ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large (Pretrained Baseline)",
    "evaluation_set": "Held-Out Test Set (5,172 eligible samples)",
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

summary_file = OUT_DIR / "indicconformer_test_metrics.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== Held-Out Test Set IndicConformer Baseline Results ===")
print(f"Total Samples: {len(predictions):,} ({total_audio_sec/3600:.2f} audio hours)")
print(f"Normalized WER: {norm_wer*100:.2f}%")
print(f"Normalized CER: {norm_cer*100:.2f}%")
print(f"Raw WER: {raw_wer*100:.2f}% | Raw CER: {raw_cer*100:.2f}%")
print(f"Empty Hypotheses: {empty_pct:.2f}%")
print(f"Total Wall-Clock Time: {total_eval_time/60:.2f}m | RTF: {rtf:.4f} ({total_audio_sec/total_eval_time:.1f}x real-time)")
