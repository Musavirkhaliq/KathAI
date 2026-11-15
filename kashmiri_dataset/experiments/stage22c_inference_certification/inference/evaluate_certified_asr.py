#!/usr/bin/env python3
"""
Stage 22C Canonical Certified ASR Evaluation Engine
Evaluates any compatible Kashmiri ASR checkpoint on any manifest under strict
scientific invariants:
1. Exact Conv1d feature-extractor output length calculation (zero padded frames).
2. Length-aware bucketed batching with hard sample bounds.
3. Deterministic long-audio policy: Uninterrupted for <= 25s, Method C2 for > 25s.
4. Frozen decoder hyperparameters (alpha=0.30, beta=-1.00, beam=64).
5. 100% crash-proof incremental JSONL checkpointing and deterministic resumption.
"""

import os, sys, json, time, argparse, csv, multiprocessing, concurrent.futures
import numpy as np
import soundfile as sf
import scipy.signal
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import jiwer

sys.path.insert(0, "/home/musa/Kashmiri Project")
from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_func

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage22c_exact_output_lengths import compute_exact_ctc_length, slice_exact_logits

# Global worker state for multiprocessing decoder pool
worker_decoder = None

def init_decoder_worker(labels, lm_path, unigrams, alpha, beta):
    global worker_decoder
    worker_decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta
    )

def decode_task(task):
    global worker_decoder
    lps, beam_width = task
    return worker_decoder.decode(lps, beam_width=beam_width)

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

def extract_long_audio_c2(model, proc, wav, device, core_len=18*16000, ctx_len=2*16000):
    """
    Method C2: Context-Preserving Central-Frame Extraction.
    Extracts core 18s regions with 2s left/right acoustic context.
    """
    total_samples = len(wav)
    all_central_lps = []
    
    for s_core in range(0, total_samples, core_len):
        e_core = min(s_core + core_len, total_samples)
        s_ctx = max(0, s_core - ctx_len)
        e_ctx = min(total_samples, e_core + ctx_len)
        
        chunk = wav[s_ctx:e_ctx]
        inp = proc(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(inp).logits
            lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        f_left = compute_exact_ctc_length(s_core - s_ctx) if (s_core - s_ctx) >= 400 else 0
        f_total = compute_exact_ctc_length(e_core - s_ctx)
        all_central_lps.append(lps[f_left:f_total, :])
        
    return np.concatenate(all_central_lps, axis=0)

def evaluate_certified_model(model_name, model_path, manifest_items, decoder_pool, device, out_jsonl_path,
                             batch_size=32, max_batch_samples=480000, beam_width=64):
    existing_preds = {}
    if os.path.exists(out_jsonl_path):
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_preds[rec["sample_id"]] = rec
        print(f"[{model_name}] Found {len(existing_preds)} existing cached predictions in {out_jsonl_path}")
        
    if len(existing_preds) == len(manifest_items):
        print(f"[{model_name}] All {len(manifest_items)} items already evaluated. Returning cached results.")
        return [existing_preds[x.get("sample_id") or x.get("id")] for x in manifest_items]
        
    remaining_indices = [i for i, x in enumerate(manifest_items) if (x.get("sample_id") or x.get("id")) not in existing_preds]
    print(f"[{model_name}] Processing {len(remaining_indices)}/{len(manifest_items)} utterances on {device}...")
    
    proc = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    model.eval()
    
    f_out = open(out_jsonl_path, "a", encoding="utf-8")
    t0 = time.time()
    
    # Process remaining in chunks bounded by batch_size and max_batch_samples
    curr_idx = 0
    b_step = 0
    while curr_idx < len(remaining_indices):
        # Accumulate batch
        batch_indices = []
        batch_samples = 0
        while curr_idx < len(remaining_indices):
            idx = remaining_indices[curr_idx]
            item = manifest_items[idx]
            est_samples = int(item.get("duration", 4.0) * 16000)
            
            # If next item exceeds 25s, isolate it immediately
            if est_samples > 25 * 16000:
                if len(batch_indices) == 0:
                    batch_indices.append(idx)
                    curr_idx += 1
                break
                
            if len(batch_indices) > 0 and (len(batch_indices) >= batch_size or (batch_samples + est_samples) > max_batch_samples):
                break
                
            batch_indices.append(idx)
            batch_samples += est_samples
            curr_idx += 1
            
        batch = [manifest_items[i] for i in batch_indices]
        
        # Load audio in parallel
        audio_paths = [x.get("audio") or x.get("audio_path") or x.get("audio_filepath") for x in batch]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            audio_list = list(ex.map(load_audio_waveform, audio_paths))
            
        lengths = [len(w) for w in audio_list]
        max_len = max(lengths)
        
        sliced_lps_list = []
        greedy_preds = []
        
        # Branch: Long audio (>25s) vs Standard audio (<=25s)
        if max_len > 25 * 16000:
            for w in audio_list:
                if len(w) > 25 * 16000:
                    slps = extract_long_audio_c2(model, proc, w, device)
                else:
                    inp = proc(w, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                    with torch.no_grad():
                        logits = model(inp).logits
                        lps = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    exact_t = compute_exact_ctc_length(len(w))
                    slps = lps[:exact_t, :]
                sliced_lps_list.append(slps)
                p_ids = np.argmax(slps, axis=-1)
                greedy_preds.append(proc.decode(p_ids))
        else:
            inputs = proc(audio_list, sampling_rate=16000, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
                log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
                
            del inputs
            del logits
            
            for b_i in range(len(batch)):
                exact_t = compute_exact_ctc_length(lengths[b_i])
                slps = log_probs[b_i, :exact_t, :]
                sliced_lps_list.append(slps)
                p_ids = np.argmax(slps, axis=-1)
                greedy_preds.append(proc.decode(p_ids))
                
            del log_probs
            
        # KenLM beam search
        kenlm_tasks = [(slps, beam_width) for slps in sliced_lps_list]
        kenlm_preds = decoder_pool.map(decode_task, kenlm_tasks)
        
        # Flush to disk immediately
        for b_i, x in enumerate(batch):
            sid = x.get("sample_id") or x.get("id")
            ref_txt = (x.get("normalized_text") or x.get("text") or "").strip()
            rec = {
                "sample_id": sid,
                "duration": x.get("duration", lengths[b_i] / 16000.0),
                "source": x.get("source"),
                "speaker_id": x.get("speaker_id", x.get("speaker", "unknown")),
                "reference": ref_txt,
                "greedy_pred": greedy_preds[b_i],
                "kenlm_pred": kenlm_preds[b_i]
            }
            existing_preds[sid] = rec
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush()
        
        b_step += 1
        if b_step % 10 == 0:
            torch.cuda.empty_cache()
            
        curr_done = len(existing_preds)
        if b_step % 15 == 0 or curr_done == len(manifest_items):
            print(f"  [{model_name}] Progress: {curr_done}/{len(manifest_items)} ({curr_done/len(manifest_items)*100:.1f}%) in {time.time()-t0:.1f}s")
            
    f_out.close()
    del model
    del proc
    torch.cuda.empty_cache()
    print(f"[{model_name}] Completed all {len(manifest_items)} utterances in {time.time()-t0:.1f}s.")
    return [existing_preds[x.get("sample_id") or x.get("id")] for x in manifest_items]

def compute_wer_cer(refs, preds):
    n_refs = [norm_func(r) for r in refs]
    n_preds = [norm_func(p) for p in preds]
    w = jiwer.process_words(n_refs, n_preds)
    c = jiwer.process_characters(n_refs, n_preds)
    return {
        "wer_pct": round(w.wer * 100.0, 2),
        "cer_pct": round(c.cer * 100.0, 2),
        "substitutions": w.substitutions,
        "deletions": w.deletions,
        "insertions": w.insertions,
        "hits": w.hits,
        "ref_words": w.hits + w.substitutions + w.deletions,
        "ref_chars": c.hits + c.substitutions + c.deletions
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Certified Kashmiri ASR System")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to acoustic model checkpoint")
    parser.add_argument("--manifest", type=str, required=True, help="Path to evaluation JSONL manifest")
    parser.add_argument("--lm-path", type=str, default="/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa")
    parser.add_argument("--unigrams", type=str, default="/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt")
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=-1.00)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--output-jsonl", type=str, required=True, help="Path to output predictions JSONL")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    
    print(f"=== Running Certified ASR Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Manifest:   {args.manifest}")
    print(f"Output:     {args.output_jsonl}")
    
    with open(args.manifest, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(items)} manifest items.")
    
    with open(args.unigrams, "r", encoding="utf-8") as f:
        unigrams = [line.strip() for line in f if line.strip()]
        
    proc = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    sorted_vocab = sorted(proc.tokenizer.get_vocab().items(), key=lambda x: x[1])
    labels = [token if token != "|" else " " for token, idx in sorted_vocab]
    
    print(f"Initializing {args.num_workers}-worker KenLM decoder pool (alpha={args.alpha}, beta={args.beta}, beam={args.beam})...")
    pool = multiprocessing.Pool(
        args.num_workers,
        initializer=init_decoder_worker,
        initargs=(labels, args.lm_path, unigrams, args.alpha, args.beta)
    )
    
    results = evaluate_certified_model(
        model_name=os.path.basename(args.checkpoint),
        model_path=args.checkpoint,
        manifest_items=items,
        decoder_pool=pool,
        device=torch.device(args.device),
        out_jsonl_path=args.output_jsonl,
        beam_width=args.beam
    )
    
    pool.close()
    pool.join()
    
    refs = [x["reference"] for x in results]
    greedy_preds = [x["greedy_pred"] for x in results]
    kenlm_preds = [x["kenlm_pred"] for x in results]
    
    m_greedy = compute_wer_cer(refs, greedy_preds)
    m_kenlm = compute_wer_cer(refs, kenlm_preds)
    
    print("\n=== Certified Results ===")
    print(f"Greedy CTC:  WER: {m_greedy['wer_pct']:5.2f}% | CER: {m_greedy['cer_pct']:5.2f}% (Subs: {m_greedy['substitutions']}, Dels: {m_greedy['deletions']}, Ins: {m_greedy['insertions']})")
    print(f"KenLM 4-gram: WER: {m_kenlm['wer_pct']:5.2f}% | CER: {m_kenlm['cer_pct']:5.2f}% (Subs: {m_kenlm['substitutions']}, Dels: {m_kenlm['deletions']}, Ins: {m_kenlm['insertions']})")

if __name__ == "__main__":
    main()
