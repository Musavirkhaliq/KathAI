#!/usr/bin/env python3
"""
Stage 23 Acoustic-Language Model Interaction & Diagnostic Analysis Engine
Evaluates E-03 and E22-01 on Kashmiri development partitions under strict Stage 22C invariants:
1. Exact Conv1d output length slicing (no padded frame leaks).
2. Method C2 context-preserving central frame extraction for audio > 25.0s.
3. Length-aware bucketed dynamic batching.
4. Comprehensive frame-level posterior distribution extraction:
   - Frame entropy, Top-1 probability, Top-2 margin
   - Blank-token posterior mean, std, spike density
   - Exact CTC forward score of reference and greedy hypotheses via ctc_loss
5. N-best beam search extraction (beam_width=64):
   - Score decomposition: acoustic logit score, combined score, LM score
   - Top-1, Top-4, Top-8, Top-16, Top-32, Top-64 oracle WER
   - Candidate diversity, reference inclusion, and candidate ranking margins
"""

import os, sys, json, time, argparse, csv, math, multiprocessing, concurrent.futures
import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import jiwer

sys.path.insert(0, "/home/musa/Kashmiri Project")
from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_func

sys.path.insert(0, "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22c_inference_certification/inference")
from stage22c_exact_output_lengths import compute_exact_ctc_length, slice_exact_logits

# Global worker state for multiprocessing decoder pool
worker_kenlm_decoder = None
worker_ctc_decoder = None

def init_worker_decoders(labels, lm_path, unigrams, alpha, beta):
    global worker_kenlm_decoder, worker_ctc_decoder
    worker_kenlm_decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta
    )
    worker_ctc_decoder = build_ctcdecoder(labels=labels)

def decode_worker_task(task):
    global worker_kenlm_decoder, worker_ctc_decoder
    lps, beam_width, alpha, beta, norm_ref = task
    
    # 1. Pure CTC beam search
    try:
        ctc_beam_text = worker_ctc_decoder.decode(lps, beam_width=beam_width)
    except Exception:
        ctc_beam_text = ""
        
    # 2. KenLM beam search with N-best extraction
    try:
        raw_beams = worker_kenlm_decoder.decode_beams(lps, beam_width=beam_width)
    except Exception:
        raw_beams = []
        
    parsed_beams = []
    top1_wer = 1.0
    oracle_4 = 1.0
    oracle_8 = 1.0
    oracle_16 = 1.0
    oracle_32 = 1.0
    oracle_64 = 1.0
    has_exact_ref = False
    better_candidate_available = False
    
    ref_norm = norm_func(norm_ref) if norm_ref else ""
    
    for rank, b in enumerate(raw_beams):
        cand_text = b[0]
        logit_score = float(b[3])
        combined_score = float(b[4])
        word_count = len(cand_text.split())
        # lm_score derivation: combined = logit + alpha * lm + beta * word_count
        lm_score = (combined_score - logit_score - beta * word_count) / max(alpha, 1e-6)
        
        cand_norm = norm_func(cand_text)
        cand_wer = jiwer.wer(ref_norm, cand_norm) if ref_norm else 0.0
        cand_cer = jiwer.cer(ref_norm, cand_norm) if ref_norm else 0.0
        
        if rank == 0:
            top1_wer = cand_wer
        if cand_norm == ref_norm and len(ref_norm) > 0:
            has_exact_ref = True
        if cand_wer < top1_wer:
            better_candidate_available = True
            
        parsed_beams.append({
            "rank": rank + 1,
            "text": cand_text,
            "norm_text": cand_norm,
            "logit_score": round(logit_score, 4),
            "combined_score": round(combined_score, 4),
            "lm_score": round(lm_score, 4),
            "word_count": word_count,
            "wer": round(cand_wer, 4),
            "cer": round(cand_cer, 4)
        })
        
    if parsed_beams:
        oracle_4 = min(b["wer"] for b in parsed_beams[:4])
        oracle_8 = min(b["wer"] for b in parsed_beams[:8])
        oracle_16 = min(b["wer"] for b in parsed_beams[:16])
        oracle_32 = min(b["wer"] for b in parsed_beams[:32])
        oracle_64 = min(b["wer"] for b in parsed_beams[:64])
        top_kenlm_text = parsed_beams[0]["text"]
    else:
        top_kenlm_text = ""
        
    return {
        "ctc_beam_text": ctc_beam_text,
        "kenlm_text": top_kenlm_text,
        "beams": parsed_beams[:32], # preserve top 32 for storage efficiency
        "beam_count": len(parsed_beams),
        "top1_wer": round(top1_wer, 4),
        "oracle_4_wer": round(oracle_4, 4),
        "oracle_8_wer": round(oracle_8, 4),
        "oracle_16_wer": round(oracle_16, 4),
        "oracle_32_wer": round(oracle_32, 4),
        "oracle_64_wer": round(oracle_64, 4),
        "has_exact_ref": has_exact_ref,
        "better_candidate_available": better_candidate_available
    }

def load_audio_waveform(path):
    if not os.path.isabs(path):
        path = os.path.join("/home/musa/Kashmiri Project", path)
    wav, sr = sf.read(path)
    if sr != 16000:
        num_samples = int(len(wav) * 16000 / sr)
        wav = scipy.signal.resample(wav, num_samples)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)

def extract_long_audio_c2(model, proc, wav, device, core_len=18*16000, ctx_len=2*16000, temperature=1.0):
    total_samples = len(wav)
    all_central_lps = []
    
    for s_core in range(0, total_samples, core_len):
        e_core = min(s_core + core_len, total_samples)
        s_ctx = max(0, s_core - ctx_len)
        e_ctx = min(total_samples, e_core + ctx_len)
        
        chunk = wav[s_ctx:e_ctx]
        inp = proc(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(inp).logits / temperature
            lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        f_left = compute_exact_ctc_length(s_core - s_ctx) if (s_core - s_ctx) >= 400 else 0
        f_total = compute_exact_ctc_length(e_core - s_ctx)
        all_central_lps.append(lps[f_left:f_total, :])
        
    return np.concatenate(all_central_lps, axis=0)

def compute_posterior_metrics(lps):
    """
    Computes frame-level posterior statistics on (T, C) log-probabilities:
    - Frame entropy
    - Top-1 probability
    - Top-2 margin
    - Blank posterior mean and std
    - Spike density
    """
    probs = np.exp(lps) # (T, C)
    # 1. Entropy
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1).mean()
    
    # 2. Top-1 & Top-2 margin
    sorted_probs = np.sort(probs, axis=-1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    top1_p = float(top1.mean())
    top2_margin = float((top1 - top2).mean())
    
    # 3. Blank posterior (index 0)
    blank_p = probs[:, 0]
    blank_p_mean = float(blank_p.mean())
    blank_p_std = float(blank_p.std())
    blank_p_median = float(np.median(blank_p))
    
    # 4. Spike density: fraction of frames where non-blank is argmax
    argmax = np.argmax(probs, axis=-1)
    spike_density = float((argmax != 0).mean())
    
    return {
        "frame_entropy": round(float(entropy), 4),
        "top1_prob": round(top1_p, 4),
        "top2_margin": round(top2_margin, 4),
        "blank_p_mean": round(blank_p_mean, 4),
        "blank_p_std": round(blank_p_std, 4),
        "blank_p_median": round(blank_p_median, 4),
        "spike_density": round(spike_density, 4),
        "total_frames": int(len(lps))
    }

def compute_ctc_forward_scores(lps_torch, ref_text, greedy_text, tokenizer, device):
    """
    Calculates exact acoustic CTC log likelihood of reference and greedy texts
    using torch.nn.functional.ctc_loss(reduction='none').
    """
    T = lps_torch.shape[0]
    # ctc_loss expects (T, N, C)
    log_probs = lps_torch.unsqueeze(1).to(device)
    
    ref_norm = norm_func(ref_text)
    greedy_norm = norm_func(greedy_text)
    
    ref_tokens = tokenizer(ref_norm).input_ids if ref_norm else []
    greedy_tokens = tokenizer(greedy_norm).input_ids if greedy_norm else []
    
    ref_logp = -9999.0
    greedy_logp = -9999.0
    
    # Filter tokens >= 161 if any
    ref_tokens = [t for t in ref_tokens if t < 161]
    greedy_tokens = [t for t in greedy_tokens if t < 161]
    
    if len(ref_tokens) > 0 and len(ref_tokens) <= T:
        target_tensor = torch.tensor(ref_tokens, dtype=torch.long, device=device)
        inp_len = torch.tensor([T], dtype=torch.long, device=device)
        tgt_len = torch.tensor([len(ref_tokens)], dtype=torch.long, device=device)
        try:
            loss = F.ctc_loss(log_probs, target_tensor, inp_len, tgt_len, blank=0, reduction='none')
            ref_logp = -float(loss.item())
        except Exception:
            ref_logp = -9999.0
            
    if len(greedy_tokens) > 0 and len(greedy_tokens) <= T:
        target_tensor = torch.tensor(greedy_tokens, dtype=torch.long, device=device)
        inp_len = torch.tensor([T], dtype=torch.long, device=device)
        tgt_len = torch.tensor([len(greedy_tokens)], dtype=torch.long, device=device)
        try:
            loss = F.ctc_loss(log_probs, target_tensor, inp_len, tgt_len, blank=0, reduction='none')
            greedy_logp = -float(loss.item())
        except Exception:
            greedy_logp = -9999.0
            
    margin = ref_logp - greedy_logp if (ref_logp > -9000.0 and greedy_logp > -9000.0) else None
    
    return {
        "ref_ctc_logp": round(ref_logp, 4) if ref_logp > -9000.0 else None,
        "greedy_ctc_logp": round(greedy_logp, 4) if greedy_logp > -9000.0 else None,
        "ref_vs_greedy_margin": round(margin, 4) if margin is not None else None
    }

def run_stage23_analysis(model_name, checkpoint_path, manifest_path, out_jsonl_path,
                         lm_path, unigrams_path, alpha=0.30, beta=-1.00, beam_width=64,
                         temperature=1.0, device_str="cuda", num_workers=4):
    device = torch.device(device_str)
    print(f"\n=======================================================")
    print(f"Stage 23 Analysis Engine: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"Output:     {out_jsonl_path}")
    print(f"=======================================================")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(manifest_items)} items from manifest.")
    
    existing = {}
    if os.path.exists(out_jsonl_path):
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r["sample_id"]] = r
        print(f"Found {len(existing)} existing records in {out_jsonl_path}.")
        
    if len(existing) == len(manifest_items):
        print(f"All {len(manifest_items)} items already evaluated. Returning cached records.")
        return [existing[x.get("sample_id") or x.get("id")] for x in manifest_items]
        
    proc = Wav2Vec2Processor.from_pretrained(checkpoint_path)
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path).to(device)
    model.eval()
    
    sorted_vocab = sorted(proc.tokenizer.get_vocab().items(), key=lambda x: x[1])
    labels = [token if token != "|" else " " for token, idx in sorted_vocab]
    
    with open(unigrams_path, "r", encoding="utf-8") as f:
        unigrams = [line.strip() for line in f if line.strip()]
        
    decoder_pool = multiprocessing.Pool(
        num_workers,
        initializer=init_worker_decoders,
        initargs=(labels, lm_path, unigrams, alpha, beta)
    )
    
    remaining_indices = [i for i, x in enumerate(manifest_items) if (x.get("sample_id") or x.get("id")) not in existing]
    print(f"Processing {len(remaining_indices)} remaining items on {device} with {num_workers} decoder workers...")
    
    # Sort remaining indices by duration for optimal batching
    durations = [float(manifest_items[i].get("duration") or manifest_items[i].get("duration_seconds") or 0.0) for i in remaining_indices]
    sorted_rem_idx = [x for _, x in sorted(zip(durations, remaining_indices))]
    
    batch_size = 16
    max_batch_samples = 320000
    t0 = time.time()
    
    b_start = 0
    while b_start < len(sorted_rem_idx):
        # Accumulate batch
        curr_batch_idx = []
        curr_samples = 0
        while b_start < len(sorted_rem_idx) and len(curr_batch_idx) < batch_size:
            idx = sorted_rem_idx[b_start]
            item = manifest_items[idx]
            dur = float(item.get("duration") or item.get("duration_seconds") or 0.0)
            samples = int(dur * 16000)
            if curr_batch_idx and (curr_samples + samples > max_batch_samples):
                break
            curr_batch_idx.append(idx)
            curr_samples += samples
            b_start += 1
            if dur > 25.0: # Process long audio individually
                break
                
        batch_items = [manifest_items[i] for i in curr_batch_idx]
        
        # Audio loading & inference
        waveforms = []
        for it in batch_items:
            path = it.get("audio_path") or it.get("audio_filepath") or it.get("audio")
            wav = load_audio_waveform(path)
            waveforms.append(wav)
            
        acoustic_results = []
        for wav, it in zip(waveforms, batch_items):
            dur = len(wav) / 16000.0
            if dur > 25.0:
                lps = extract_long_audio_c2(model, proc, wav, device, temperature=temperature)
            else:
                inp = proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                with torch.no_grad():
                    logits = model(inp).logits / temperature
                    valid_frames = compute_exact_ctc_length(len(wav))
                    lps = torch.log_softmax(logits[:, :valid_frames, :], dim=-1).squeeze(0).cpu().numpy()
            acoustic_results.append(lps)
            
        # Greedy decoding, posterior stats, CTC forward scoring
        cpu_tasks = []
        interm_records = []
        for lps, it in zip(acoustic_results, batch_items):
            sid = it.get("sample_id") or it.get("id")
            ref_raw = it.get("text", "")
            ref_norm = norm_func(it.get("normalized_text") or ref_raw)
            
            # Greedy
            pred_ids = np.argmax(lps, axis=-1)
            greedy_text = proc.decode(pred_ids)
            greedy_norm = norm_func(greedy_text)
            
            # Posteriors
            post_stats = compute_posterior_metrics(lps)
            
            # CTC forward scoring
            lps_torch = torch.from_numpy(lps)
            ctc_scores = compute_ctc_forward_scores(lps_torch, ref_norm, greedy_norm, proc.tokenizer, device)
            
            # Decoder task
            cpu_tasks.append((lps, beam_width, alpha, beta, ref_norm))
            interm_records.append({
                "sample_id": sid,
                "model": model_name,
                "duration": round(len(lps) * 0.02, 3),
                "source": it.get("source_dataset") or it.get("source", "unknown"),
                "ref_text": ref_raw,
                "ref_norm": ref_norm,
                "greedy_text": greedy_text,
                "greedy_norm": greedy_norm,
                "greedy_wer": round(jiwer.wer(ref_norm, greedy_norm), 4) if ref_norm else 0.0,
                "greedy_cer": round(jiwer.cer(ref_norm, greedy_norm), 4) if ref_norm else 0.0,
                "posterior_stats": post_stats,
                "ctc_forward_scores": ctc_scores
            })
            
        # Parallel LM decoding
        dec_results = decoder_pool.map(decode_worker_task, cpu_tasks)
        
        # Merge and save
        with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
            for rec, dec in zip(interm_records, dec_results):
                rec["ctc_beam_text"] = dec["ctc_beam_text"]
                rec["ctc_beam_norm"] = norm_func(dec["ctc_beam_text"])
                rec["ctc_beam_wer"] = round(jiwer.wer(rec["ref_norm"], rec["ctc_beam_norm"]), 4) if rec["ref_norm"] else 0.0
                rec["ctc_beam_cer"] = round(jiwer.cer(rec["ref_norm"], rec["ctc_beam_norm"]), 4) if rec["ref_norm"] else 0.0
                
                rec["kenlm_text"] = dec["kenlm_text"]
                rec["kenlm_norm"] = norm_func(dec["kenlm_text"])
                rec["kenlm_wer"] = round(jiwer.wer(rec["ref_norm"], rec["kenlm_norm"]), 4) if rec["ref_norm"] else 0.0
                rec["kenlm_cer"] = round(jiwer.cer(rec["ref_norm"], rec["kenlm_norm"]), 4) if rec["ref_norm"] else 0.0
                
                rec["nbest"] = dec["beams"]
                rec["oracle_4_wer"] = dec["oracle_4_wer"]
                rec["oracle_8_wer"] = dec["oracle_8_wer"]
                rec["oracle_16_wer"] = dec["oracle_16_wer"]
                rec["oracle_32_wer"] = dec["oracle_32_wer"]
                rec["oracle_64_wer"] = dec["oracle_64_wer"]
                rec["has_exact_ref"] = dec["has_exact_ref"]
                rec["better_candidate_available"] = dec["better_candidate_available"]
                
                existing[rec["sample_id"]] = rec
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                
        curr_done = len(existing)
        if curr_done % 100 < batch_size or curr_done == len(manifest_items):
            elapsed = time.time() - t0
            print(f"  [{model_name}] Completed {curr_done}/{len(manifest_items)} ({curr_done/len(manifest_items)*100:.1f}%) in {elapsed:.1f}s")
            
    decoder_pool.close()
    decoder_pool.join()
    print(f"[{model_name}] Finished all {len(manifest_items)} items in {time.time()-t0:.1f}s.")
    return [existing[x.get("sample_id") or x.get("id")] for x in manifest_items]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--lm-path", type=str, default="/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa")
    parser.add_argument("--unigrams", type=str, default="/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt")
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=-1.00)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    
    run_stage23_analysis(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        out_jsonl_path=args.output_jsonl,
        lm_path=args.lm_path,
        unigrams_path=args.unigrams,
        alpha=args.alpha,
        beta=args.beta,
        beam_width=args.beam,
        temperature=args.temperature,
        num_workers=args.num_workers
    )
