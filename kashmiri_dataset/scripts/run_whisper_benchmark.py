import json
import time
import os
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

BASE_DIR = Path("/home/musa/Kashmiri Project")
EVAL_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_baselines" / "eval_splits"
PRED_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_baselines" / "predictions"
REP_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_baselines"

print("=== Initializing Whisper-Small ASR Baseline on CUDA ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

eval_splits = [
    "eval_clean_read",
    "eval_conversational_field",
    "eval_multispeaker_bolbosh",
    "eval_unseen_speakers",
    "eval_code_switched",
    "eval_noisy_low_snr",
    "master_benchmark_eval"
]

benchmark_summary = {}

for split_name in eval_splits:
    split_file = EVAL_DIR / f"{split_name}.jsonl"
    if not split_file.exists():
        continue
        
    print(f"\n--- Running Benchmark on {split_name} ---")
    predictions = []
    total_audio_duration = 0.0
    total_inference_time = 0.0
    
    with open(split_file, "r", encoding="utf-8") as f_in:
        items = [json.loads(line) for line in f_in]
        
    for idx, item in enumerate(items):
        audio_path = item["abs_audio_path"]
        ref_text = item["text"]
        
        try:
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            dur = len(audio_data) / sr
            total_audio_duration += dur
            
            # Whisper expects 16,000 Hz input features
            t0 = time.time()
            inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                generated_ids = model.generate(inputs, max_new_tokens=128)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            t1 = time.time()
            total_inference_time += (t1 - t0)
            
            # Metrics
            raw_metrics = calculate_wer_and_cer(ref_text, pred_text)
            norm_ref = normalize_perso_arabic_kashmiri(ref_text)
            norm_pred = normalize_perso_arabic_kashmiri(pred_text)
            norm_metrics = calculate_wer_and_cer(norm_ref, norm_pred)
            
            rec = {
                "sample_id": item["sample_id"],
                "audio_path": item["audio_path"],
                "duration": dur,
                "ref_raw": ref_text,
                "hyp_raw": pred_text,
                "ref_norm": norm_ref,
                "hyp_norm": norm_pred,
                "raw_wer": float(raw_metrics["wer"]),
                "raw_cer": float(raw_metrics["cer"]),
                "norm_wer": float(norm_metrics["wer"]),
                "norm_cer": float(norm_metrics["cer"]),
                "exact_match": bool(norm_metrics["exact_match"]),
                "source": item.get("dataset_source", item.get("source", "unknown")),
                "speaker_id": item.get("speaker_id", "unknown")
            }
            predictions.append(rec)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            
    # Calculate corpus aggregate metrics
    total_ref_words_raw = sum(len(p["ref_raw"].split()) for p in predictions)
    total_word_dist_raw = sum(calculate_wer_and_cer(p["ref_raw"], p["hyp_raw"])["word_dist"] for p in predictions)
    corpus_raw_wer = round(float(total_word_dist_raw / max(total_ref_words_raw, 1)), 4)
    
    total_ref_chars_raw = sum(len(p["ref_raw"].replace(" ", "")) for p in predictions)
    total_char_dist_raw = sum(calculate_wer_and_cer(p["ref_raw"], p["hyp_raw"])["char_dist"] for p in predictions)
    corpus_raw_cer = round(float(total_char_dist_raw / max(total_ref_chars_raw, 1)), 4)
    
    total_ref_words_norm = sum(len(p["ref_norm"].split()) for p in predictions)
    total_word_dist_norm = sum(calculate_wer_and_cer(p["ref_norm"], p["hyp_norm"])["word_dist"] for p in predictions)
    corpus_norm_wer = round(float(total_word_dist_norm / max(total_ref_words_norm, 1)), 4)
    
    total_ref_chars_norm = sum(len(p["ref_norm"].replace(" ", "")) for p in predictions)
    total_char_dist_norm = sum(calculate_wer_and_cer(p["ref_norm"], p["hyp_norm"])["char_dist"] for p in predictions)
    corpus_norm_cer = round(float(total_char_dist_norm / max(total_ref_chars_norm, 1)), 4)
    
    exact_matches = sum(1 for p in predictions if p["exact_match"])
    rtf = round(total_inference_time / max(total_audio_duration, 1e-6), 4)
    
    # Save predictions
    pred_file = PRED_DIR / f"whisper_small_{split_name}_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f_out:
        for p in predictions:
            f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    benchmark_summary[split_name] = {
        "num_samples": len(predictions),
        "total_audio_duration_sec": round(total_audio_duration, 2),
        "total_inference_time_sec": round(total_inference_time, 2),
        "real_time_factor": rtf,
        "corpus_raw_wer": corpus_raw_wer,
        "corpus_raw_cer": corpus_raw_cer,
        "corpus_norm_wer": corpus_norm_wer,
        "corpus_norm_cer": corpus_norm_cer,
        "sentence_exact_match_pct": round(exact_matches / max(len(predictions), 1) * 100, 2),
        "predictions_file": str(pred_file.relative_to(BASE_DIR))
    }
    
    print(f"[{split_name}] Samples: {len(predictions)} | Norm WER: {corpus_norm_wer*100:.1f}% | Norm CER: {corpus_norm_cer*100:.1f}% | RTF: {rtf}")

# Save full results
res_file = REP_DIR / "baseline_results.json"
with open(res_file, "w", encoding="utf-8") as f:
    json.dump({
        "model": "openai/whisper-small",
        "parameters": "244M",
        "device": device,
        "results_by_split": benchmark_summary
    }, f, indent=2)

print(f"\nAll baseline benchmarks completed successfully! Results saved to {res_file}")
