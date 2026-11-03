import json
import time
import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCTC, WhisperProcessor, WhisperForConditionalGeneration
from metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

BASE_DIR = Path("/home/musa/Kashmiri Project")
PILOT_FILE = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_sota_benchmark" / "pilot_eval" / "pilot_50.jsonl"
OUT_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_sota_benchmark" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Pilot Benchmark on device: {device}")

with open(PILOT_FILE, "r", encoding="utf-8") as f:
    eval_items = [json.loads(line) for line in f]

print(f"Loaded {len(eval_items)} pilot samples.")

def evaluate_model(model_id, is_whisper=False):
    print(f"\n================ EVALUATING: {model_id} ================")
    try:
        if is_whisper:
            proc = WhisperProcessor.from_pretrained(model_id)
            model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        else:
            proc = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForCTC.from_pretrained(model_id).to(device)
        model.eval()
    except Exception as e:
        print(f"FAILED TO LOAD {model_id}: {e}")
        return {
            "status": "LOAD_FAILED",
            "error": str(e)
        }

    preds = []
    tot_dur = 0.0
    tot_time = 0.0
    
    for idx, item in enumerate(eval_items):
        audio_path = item["abs_audio_path"]
        ref_text = item["text"]
        
        try:
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            dur = len(audio_data) / sr
            tot_dur += dur
            
            t0 = time.time()
            if is_whisper:
                inputs = proc(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                with torch.no_grad():
                    gen_ids = model.generate(inputs, max_new_tokens=128)
                hyp_text = proc.batch_decode(gen_ids, skip_special_tokens=True)[0]
            else:
                inputs = proc(audio_data, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                with torch.no_grad():
                    logits = model(inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                hyp_text = proc.batch_decode(pred_ids)[0]
            t1 = time.time()
            tot_time += (t1 - t0)
            
            # Metrics
            raw_m = calculate_wer_and_cer(ref_text, hyp_text)
            norm_ref = normalize_perso_arabic_kashmiri(ref_text)
            norm_hyp = normalize_perso_arabic_kashmiri(hyp_text)
            norm_m = calculate_wer_and_cer(norm_ref, norm_hyp)
            
            preds.append({
                "sample_id": item["sample_id"],
                "ref_raw": ref_text,
                "hyp_raw": hyp_text,
                "ref_norm": norm_ref,
                "hyp_norm": norm_hyp,
                "raw_wer": float(raw_m["wer"]),
                "norm_wer": float(norm_m["wer"]),
                "norm_cer": float(norm_m["cer"])
            })
        except Exception as e:
            print(f"Error on sample {idx}: {e}")

    # Aggregates
    tot_ref_words = sum(len(p["ref_norm"].split()) for p in preds)
    tot_word_dist = sum(calculate_wer_and_cer(p["ref_norm"], p["hyp_norm"])["word_dist"] for p in preds)
    norm_wer = round(float(tot_word_dist / max(tot_ref_words, 1)), 4)
    
    tot_ref_chars = sum(len(p["ref_norm"].replace(" ", "")) for p in preds)
    tot_char_dist = sum(calculate_wer_and_cer(p["ref_norm"], p["hyp_norm"])["char_dist"] for p in preds)
    norm_cer = round(float(tot_char_dist / max(tot_ref_chars, 1)), 4)
    
    rtf = round(tot_time / max(tot_dur, 1e-6), 4)
    
    safe_name = model_id.replace("/", "_").lower()
    out_file = OUT_DIR / f"{safe_name}_pilot_predictions.jsonl"
    with open(out_file, "w", encoding="utf-8") as f_out:
        for p in preds:
            f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    print(f"--> {model_id} | Norm WER: {norm_wer*100:.2f}% | Norm CER: {norm_cer*100:.2f}% | RTF: {rtf}")
    
    return {
        "status": "SUCCESS",
        "norm_wer": norm_wer,
        "norm_cer": norm_cer,
        "rtf": rtf,
        "predictions_file": str(out_file.relative_to(BASE_DIR))
    }

candidates = [
    ("openai/whisper-small", True),
    ("Muzaffar786/wav2vec2-kashmiri-jhon-data-one", False),
    ("KhushiDS/wav2vec2-large-xlsr-53-Kashmiri", False),
    ("helpsulaiman/kashmiri-wav2vec2", False)
]

pilot_results = {}
for mid, is_w in candidates:
    res = evaluate_model(mid, is_w)
    pilot_results[mid] = res

rep_file = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_sota_benchmark" / "pilot_results.json"
with open(rep_file, "w", encoding="utf-8") as f:
    json.dump(pilot_results, f, indent=2)

print("\n=== PILOT BENCHMARK COMPLETE ===")
print(f"Summary written to {rep_file}")
