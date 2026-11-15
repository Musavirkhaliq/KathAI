#!/usr/bin/env python3
"""
Stage 22C Chunking Strategies Benchmark on Development Audio (Dev A)
Compares uninterrupted inference vs C0, C1, C2, C3 strategies across:
- WER vs Reference
- Exact token match rate vs Uninterrupted Ground Truth
- Levenshtein character distance vs Uninterrupted Ground Truth
- Latency and peak VRAM
"""

import os, sys, json, time, csv
import numpy as np
import soundfile as sf
import scipy.signal
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import jiwer

sys.path.insert(0, "/home/musa/Kashmiri Project")
from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_func

sys.path.insert(0, "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22c_inference_certification/inference")
from stage22c_exact_output_lengths import compute_exact_ctc_length

def load_audio(p):
    if not os.path.isabs(p):
        p = os.path.join("/home/musa/Kashmiri Project", p)
    wav, sr = sf.read(p)
    if sr != 16000:
        num_samples = int(len(wav) * 16000 / sr)
        wav = scipy.signal.resample(wav, num_samples)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)

# 1. Uninterrupted Full-Audio Inference (Golden Reference)
def infer_uninterrupted(model, proc, wav, device):
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    inp = proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(inp).logits
        lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    exact_t = compute_exact_ctc_length(len(wav))
    sliced = lps[:exact_t, :]
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elapsed = time.time() - t_start
    return sliced, peak_vram, elapsed

# 2. Method C0: Historical Non-overlapping Chunking (20s chunks, trailing remainder merge)
def infer_c0_non_overlapping(model, proc, wav, device, chunk_len=20*16000):
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    chunks = []
    for i in range(0, len(wav), chunk_len):
        chunk = wav[i:i+chunk_len]
        if len(chunks) > 0 and len(chunk) < 3200:
            chunks[-1] = np.concatenate([chunks[-1], chunk])
        else:
            chunks.append(chunk)
    
    all_lps = []
    with torch.no_grad():
        for c in chunks:
            inp = proc(c, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            logits = model(inp).logits
            lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            exact_t = compute_exact_ctc_length(len(c))
            all_lps.append(lps[:exact_t, :])
    sliced = np.concatenate(all_lps, axis=0)
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elapsed = time.time() - t_start
    return sliced, peak_vram, elapsed

# 3. Method C1: Overlapping Chunking with Linear Crossfade Blend
def infer_c1_overlapping(model, proc, wav, device, chunk_len=20*16000, overlap_len=2*16000):
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    hop_len = chunk_len - overlap_len
    
    # Generate overlapping windows
    starts = list(range(0, len(wav), hop_len))
    if len(starts) > 1 and (len(wav) - starts[-1]) < overlap_len + 3200:
        starts.pop()
    
    # If only 1 chunk covers everything
    if len(starts) == 1:
        return infer_uninterrupted(model, proc, wav, device)
    
    total_frames = compute_exact_ctc_length(len(wav))
    accum_probs = np.zeros((total_frames, 161), dtype=np.float32)
    weights = np.zeros((total_frames, 1), dtype=np.float32)
    
    with torch.no_grad():
        for s in starts:
            e = min(s + chunk_len, len(wav))
            chunk = wav[s:e]
            if len(chunk) < 400:
                continue
            inp = proc(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            logits = model(inp).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Map start and end samples to frame indices
            f_start = compute_exact_ctc_length(s) if s > 0 else 0
            n_frames = compute_exact_ctc_length(len(chunk))
            f_end = min(f_start + n_frames, total_frames)
            valid_len = f_end - f_start
            
            # Linear window
            w = np.ones((valid_len, 1), dtype=np.float32)
            if s > 0 and valid_len > 10:
                ramp = np.linspace(0.1, 1.0, min(10, valid_len))[:, None]
                w[:len(ramp)] *= ramp
            if e < len(wav) and valid_len > 10:
                ramp_down = np.linspace(1.0, 0.1, min(10, valid_len))[:, None]
                w[-len(ramp_down):] *= ramp_down
            
            accum_probs[f_start:f_end] += probs[:valid_len, :] * w
            weights[f_start:f_end] += w
            
    weights = np.maximum(weights, 1e-6)
    blended_probs = accum_probs / weights
    sliced_lps = np.log(np.maximum(blended_probs, 1e-12))
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elapsed = time.time() - t_start
    return sliced_lps, peak_vram, elapsed

# 4. Method C2: Context-Preserving Central Frame Extraction
def infer_c2_central_frames(model, proc, wav, device, core_len=18*16000, ctx_len=2*16000):
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    total_samples = len(wav)
    
    if total_samples <= core_len + ctx_len:
        return infer_uninterrupted(model, proc, wav, device)
    
    all_central_lps = []
    for s_core in range(0, total_samples, core_len):
        e_core = min(s_core + core_len, total_samples)
        
        # Add acoustic context before and after
        s_ctx = max(0, s_core - ctx_len)
        e_ctx = min(total_samples, e_core + ctx_len)
        
        chunk = wav[s_ctx:e_ctx]
        inp = proc(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(inp).logits
            lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        # Compute exact frames corresponding to s_core and e_core relative to s_ctx
        f_left = compute_exact_ctc_length(s_core - s_ctx) if (s_core - s_ctx) >= 400 else 0
        f_total = compute_exact_ctc_length(e_core - s_ctx)
        
        central_slice = lps[f_left:f_total, :]
        all_central_lps.append(central_slice)
        
    sliced = np.concatenate(all_central_lps, axis=0)
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elapsed = time.time() - t_start
    return sliced, peak_vram, elapsed

# 5. Method C3: Silence / Energy-Aware Natural Boundary Segmentation
def infer_c3_silence_aware(model, proc, wav, device, max_chunk=25*16000, min_chunk=15*16000):
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    total_samples = len(wav)
    
    if total_samples <= max_chunk:
        return infer_uninterrupted(model, proc, wav, device)
    
    # Compute short-term energy profile in 50ms windows
    win_size = int(0.05 * 16000)
    energy = np.array([np.sum(wav[i:i+win_size]**2) for i in range(0, total_samples - win_size, win_size)])
    
    # Find split points at energy minima between min_chunk and max_chunk
    split_points = [0]
    curr_s = 0
    while curr_s + max_chunk < total_samples:
        win_start = (curr_s + min_chunk) // win_size
        win_end = (curr_s + max_chunk) // win_size
        min_idx = win_start + np.argmin(energy[win_start:win_end])
        split_sample = min_idx * win_size
        split_points.append(split_sample)
        curr_s = split_sample
    split_points.append(total_samples)
    
    all_lps = []
    with torch.no_grad():
        for i in range(len(split_points) - 1):
            c = wav[split_points[i]:split_points[i+1]]
            inp = proc(c, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            logits = model(inp).logits
            lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            exact_t = compute_exact_ctc_length(len(c))
            all_lps.append(lps[:exact_t, :])
            
    sliced = np.concatenate(all_lps, axis=0)
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elapsed = time.time() - t_start
    return sliced, peak_vram, elapsed

def main():
    root = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22c_inference_certification"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 22C Chunking Strategies Benchmark on Dev A ({device}) ===")
    
    e22_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"
    proc = Wav2Vec2Processor.from_pretrained(e22_path)
    model = Wav2Vec2ForCTC.from_pretrained(e22_path).to(device)
    model.eval()
    
    lm_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"
    unigrams_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt"
    with open(unigrams_path, "r", encoding="utf-8") as f:
        unigrams = [line.strip() for line in f if line.strip()]
        
    sorted_vocab = sorted(proc.tokenizer.get_vocab().items(), key=lambda x: x[1])
    labels = [token if token != "|" else " " for token, idx in sorted_vocab]
    
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        unigrams=unigrams,
        alpha=0.30,
        beta=-1.00
    )
    
    dev_a_path = "/home/musa/Kashmiri Project/kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_a_legacy_val.jsonl"
    with open(dev_a_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    long_items = [x for x in items if x.get("duration", 0.0) >= 25.0]
    print(f"Loaded {len(long_items)} long utterances (>= 25s) from Dev A.")
    
    methods = [
        ("Uninterrupted_Reference", infer_uninterrupted),
        ("C0_NonOverlapping_Historical", infer_c0_non_overlapping),
        ("C1_Overlapping_Crossfade", infer_c1_overlapping),
        ("C2_Central_Frames_Context", infer_c2_central_frames),
        ("C3_Silence_Aware_VAD", infer_c3_silence_aware)
    ]
    
    records = []
    
    for item_idx, x in enumerate(long_items):
        sid = x.get("sample_id") or x.get("id")
        dur = x.get("duration", 0.0)
        ref = norm_func(x.get("normalized_text") or x.get("text") or "")
        ap = x.get("audio") or x.get("audio_path") or x.get("audio_filepath")
        wav = load_audio(ap)
        
        print(f"\n[{item_idx+1}/{len(long_items)}] Sample {sid} ({dur:.1f}s, {len(ref.split())} words):")
        
        preds_for_item = {}
        for m_name, m_func in methods:
            torch.cuda.empty_cache()
            lps, vram, el = m_func(model, proc, wav, device)
            
            # Greedy decode
            p_ids = np.argmax(lps, axis=-1)
            greedy_pred = norm_func(proc.decode(p_ids))
            
            # KenLM decode
            kenlm_pred = norm_func(decoder.decode(lps, beam_width=64))
            
            wer_g = round(jiwer.wer(ref, greedy_pred) * 100.0, 2)
            wer_k = round(jiwer.wer(ref, kenlm_pred) * 100.0, 2)
            cer_k = round(jiwer.cer(ref, kenlm_pred) * 100.0, 2)
            
            preds_for_item[m_name] = {
                "greedy": greedy_pred,
                "kenlm": kenlm_pred,
                "wer_k": wer_k,
                "cer_k": cer_k,
                "vram": round(vram, 1),
                "time": round(el, 3)
            }
            print(f"  {m_name:30s} | WER: {wer_k:5.2f}% | CER: {cer_k:5.2f}% | VRAM: {vram:6.1f}MB | Time: {el:5.2f}s")
            
        ref_pred_k = preds_for_item["Uninterrupted_Reference"]["kenlm"]
        
        for m_name, _ in methods:
            p_k = preds_for_item[m_name]["kenlm"]
            edit_dist = jiwer.process_characters(ref_pred_k, p_k).substitutions + \
                        jiwer.process_characters(ref_pred_k, p_k).deletions + \
                        jiwer.process_characters(ref_pred_k, p_k).insertions
            exact_match = (p_k == ref_pred_k)
            
            records.append({
                "sample_id": sid,
                "duration_s": round(dur, 2),
                "strategy": m_name,
                "wer_kenlm_pct": preds_for_item[m_name]["wer_k"],
                "cer_kenlm_pct": preds_for_item[m_name]["cer_k"],
                "exact_match_vs_uninterrupted": exact_match,
                "char_diff_vs_uninterrupted": edit_dist,
                "peak_vram_mb": preds_for_item[m_name]["vram"],
                "encoder_latency_s": preds_for_item[m_name]["time"]
            })
            
    # Save CSV
    out_csv = os.path.join(root, "chunking/stage22c_chunking_comparison.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved strategy comparison to {out_csv}")
    
    # Compute aggregate strategy metrics
    print("\n================== STRATEGY AGGREGATE PERFORMANCE (Dev A Long Audio) ==================")
    for m_name, _ in methods:
        sub = [r for r in records if r["strategy"] == m_name]
        mean_wer = np.mean([r["wer_kenlm_pct"] for r in sub])
        mean_cer = np.mean([r["cer_kenlm_pct"] for r in sub])
        exact_matches = sum(1 for r in sub if r["exact_match_vs_uninterrupted"])
        mean_vram = np.mean([r["peak_vram_mb"] for r in sub])
        mean_latency = np.mean([r["encoder_latency_s"] for r in sub])
        print(f"Strategy: {m_name:30s} | Mean WER: {mean_wer:5.2f}% | Mean CER: {mean_cer:5.2f}% | Exact Matches: {exact_matches:2d}/{len(sub)} | Peak VRAM: {mean_vram:6.1f}MB | Latency: {mean_latency:5.2f}s")

if __name__ == "__main__":
    main()
