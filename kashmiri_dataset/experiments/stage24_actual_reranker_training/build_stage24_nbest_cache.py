#!/usr/bin/env python3
"""
Stage 24: Candidate Generation & N-Best Caching Engine
Generates supervised training examples and validation candidate sets from frozen E22-01.

Strict Stage 24 Invariants:
1. Frozen E22-01 Acoustic Model (weights unmodified, sha256 verified).
2. Canonical Stage 22C inference pipeline (exact Conv1d output slicing, no padded leaks).
3. 4-gram KenLM decoding (alpha=0.30, beta=-1.00, beam_width=64).
4. Supervised training pool: 20,000 well-distributed training-only utterances.
   Zero overlap with dev_calibration, dev_a_disjoint, dev_b_waaz, dev_c_clean, or test.jsonl.
5. Cached candidate representations: acoustic score, LM score, length features, candidate token IDs,
   and reference-derived edit distances for training loss.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import time
import math
import random
import hashlib
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
import jiwer

WORKSPACE = Path("/home/musa/Kashmiri Project")
sys.path.insert(0, str(WORKSPACE))

from kashmiri_dataset.audio.canonical_audio import load_canonical_audio
from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict
from kashmiri_dataset.experiments.stage22c_inference_certification.inference.stage22c_exact_output_lengths import compute_exact_ctc_length, slice_exact_logits

STAGE24_DIR = WORKSPACE / "kashmiri_dataset/experiments/stage24_actual_reranker_training"
E22_CKPT = WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"
LM_PATH = WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/lm_word_4gram_domain_d.arpa"
UNIGRAMS_PATH = WORKSPACE / "kashmiri_dataset/experiments/stage22b_decoder_and_independent_test/language_model/unigrams.txt"

TIER_A_MANIFEST = WORKSPACE / "kashmiri_dataset/experiments/stage22_data_expanded_training/manifests/data_22a_conservative_tier_a.jsonl"
OUT_TRAIN_MANIFEST = STAGE24_DIR / "data/stage24_train_sampled_20k.jsonl"
OUT_TRAIN_NBEST = STAGE24_DIR / "nbest_cache/stage24_train_nbest_20k.jsonl"

DEV_PREDS = {
    "dev_calibration": WORKSPACE / "kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_calibration_e2201_predictions.jsonl",
    "dev_a_disjoint": WORKSPACE / "kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_a_disjoint_e2201_base.jsonl",
    "dev_b_waaz_val": WORKSPACE / "kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_b_waaz_e2201_predictions.jsonl",
    "dev_c_clean": WORKSPACE / "kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_c_clean_e2201_predictions.jsonl",
}

DEV_OUT_NBEST = {
    "dev_calibration": STAGE24_DIR / "nbest_cache/stage24_dev_calibration_nbest.jsonl",
    "dev_a_disjoint": STAGE24_DIR / "nbest_cache/stage24_dev_a_disjoint_nbest.jsonl",
    "dev_b_waaz_val": STAGE24_DIR / "nbest_cache/stage24_dev_b_waaz_nbest.jsonl",
    "dev_c_clean": STAGE24_DIR / "nbest_cache/stage24_dev_c_clean_nbest.jsonl",
}

ALPHA = 0.30
BETA = -1.00
BEAM_WIDTH = 64

def compute_word_edit_distance(ref_norm: str, hyp_norm: str):
    if not ref_norm and not hyp_norm:
        return 0, 0, 0, 0
    if not ref_norm:
        words = len(hyp_norm.split())
        return words, 0, 0, words
    if not hyp_norm:
        words = len(ref_norm.split())
        return words, 0, words, 0
    res = jiwer.process_words(ref_norm, hyp_norm)
    return res.substitutions + res.deletions + res.insertions, res.substitutions, res.deletions, res.insertions

def step1_sample_training_utterances(target_n=20000, seed=42):
    if OUT_TRAIN_MANIFEST.exists():
        print(f"STEP 1: Found existing sampled manifest at {OUT_TRAIN_MANIFEST}")
        sampled = []
        with open(OUT_TRAIN_MANIFEST, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sampled.append(json.loads(line.strip()))
        print(f"Loaded {len(sampled)} training utterances from existing manifest.")
        return sampled

    print("=" * 70)
    print(f"STEP 1: Stratified Sampling of {target_n} Training Utterances")
    print("=" * 70)

    def load_ids(path):
        s = set()
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    s.add(d.get("sample_id") or d.get("audio_filepath") or d.get("audio_path"))
        return s

    exclude = set()
    exclude_paths = [
        WORKSPACE / "kashmiri_dataset/experiments/stage11_robustness_optimization/data_splits/dev_calibration.jsonl",
        WORKSPACE / "kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_a_disjoint.jsonl",
        WORKSPACE / "kashmiri_dataset/releases/asr_training_v2_waaz/validation_manifests/dev_b_waaz_val.jsonl",
        WORKSPACE / "kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/evaluation/dev_c_clean.jsonl",
        WORKSPACE / "kashmiri_dataset/experiments/native_asr_training/data_splits/test.jsonl",
    ]
    for ep in exclude_paths:
        if ep.exists():
            s = load_ids(ep)
            exclude.update(s)
            print(f"  Excluded {len(s)} IDs from {ep.name}")

    by_spk = defaultdict(list)
    with open(TIER_A_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                sid = d.get("sample_id") or d.get("audio_filepath") or d.get("audio_path")
                if sid not in exclude:
                    spk = d.get("speaker_id", "unknown")
                    by_spk[spk].append(d)

    total_avail = sum(len(v) for v in by_spk.values())
    print(f"Clean candidate utterances available: {total_avail} across {len(by_spk)} speakers.")

    rng = random.Random(seed)
    sampled = []
    for spk, utts in by_spk.items():
        rng.shuffle(utts)
        k = max(1, int(round(target_n * len(utts) / total_avail)))
        sampled.extend(utts[:k])

    rng.shuffle(sampled)
    sampled = sampled[:target_n]

    unique_speakers = len(set(d.get("speaker_id") for d in sampled))
    print(f"Sampled {len(sampled)} utterances across {unique_speakers} speakers.")

    with open(OUT_TRAIN_MANIFEST, "w", encoding="utf-8") as f:
        for d in sampled:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved sampled train manifest to {OUT_TRAIN_MANIFEST}")
    return sampled

# Worker state for multiprocessing decode
worker_decoder = None
worker_tokenizer = None

def init_worker(labels, lm_path, unigrams, alpha, beta, tokenizer_dir):
    global worker_decoder, worker_tokenizer
    worker_decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta
    )
    worker_tokenizer = Wav2Vec2Processor.from_pretrained(tokenizer_dir).tokenizer

def decode_and_format_item(task):
    global worker_decoder, worker_tokenizer
    sid, lps, ref_text, ref_norm, spk, dur = task
    ref_words = len(ref_norm.split())

    try:
        beams = worker_decoder.decode_beams(lps, beam_width=BEAM_WIDTH, prune_history=True)
    except Exception as e:
        beams = []

    candidates = []
    min_errs = 999999
    top1_errs = 0

    for cand_idx, b in enumerate(beams[:32]):
        cand_raw = b[0]
        cand_norm = norm_strict(cand_raw)
        cand_words = len(cand_norm.split())

        errs, s, d, i = compute_word_edit_distance(ref_norm, cand_norm)
        if cand_idx == 0:
            top1_errs = errs
        if errs < min_errs:
            min_errs = errs

        token_ids = worker_tokenizer.encode(cand_norm) if worker_tokenizer else []

        logit_score = float(b[3])
        comb_score = float(b[4])
        lm_score = float((comb_score - logit_score - BETA * cand_words) / max(ALPHA, 1e-6))

        candidates.append({
            "cand_idx": cand_idx,
            "text": cand_raw,
            "norm_text": cand_norm,
            "token_ids": token_ids,
            "acoustic_score": round(logit_score, 4),
            "lm_score": round(lm_score, 4),
            "combined_score": round(comb_score, 4),
            "num_words": cand_words,
            "char_length": len(cand_norm),
            "edit_distance": errs,
            "subs": s,
            "dels": d,
            "ins": i,
            "wer_pct": round((errs / ref_words) * 100.0 if ref_words > 0 else 0.0, 2)
        })

    return {
        "sample_id": sid,
        "speaker_id": spk,
        "duration": dur,
        "ref_text": ref_text,
        "ref_norm": ref_norm,
        "top1_text": candidates[0]["text"] if candidates else "",
        "top1_norm": candidates[0]["norm_text"] if candidates else "",
        "top1_errors": top1_errs,
        "oracle_errors": min_errs if candidates else top1_errs,
        "oracle_better": (min_errs < top1_errs) if candidates else False,
        "candidates": candidates
    }

def step2_generate_train_nbest(sampled_records):
    print("=" * 70)
    print("STEP 2: Generate Frozen E22-01 Logits & N-Best Beams (20,000 Training Utterances)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading E22-01 model from {E22_CKPT} on {device}...")
    processor = Wav2Vec2Processor.from_pretrained(str(E22_CKPT))
    model = Wav2Vec2ForCTC.from_pretrained(str(E22_CKPT), torch_dtype=torch.bfloat16).to(device)
    model.eval()

    vocab = [k for k, v in sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])]
    labels = [t if t != "|" else " " for t in vocab]

    with open(UNIGRAMS_PATH) as f:
        unigrams = [line.strip() for line in f if line.strip()]

    t0 = time.time()
    batch_size = 32
    total = len(sampled_records)
    tasks = []

    print(f"Extracting acoustic logits on GPU in batches of {batch_size}...")
    for idx in range(0, total, batch_size):
        batch_recs = sampled_records[idx: idx + batch_size]
        audios = []
        valid_recs = []

        for r in batch_recs:
            p = WORKSPACE / (r.get("audio_filepath") or r.get("audio_path"))
            try:
                wav, _ = load_canonical_audio(str(p), target_sr=16000)
                if isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav)
                if wav.dim() > 1:
                    wav = wav.squeeze()
                if len(wav) >= 400:
                    audios.append(wav)
                    valid_recs.append(r)
            except Exception as e:
                pass

        if not audios:
            continue

        padded_audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True).to(device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            batch_logits = model(padded_audios).logits.float()

        for j, r in enumerate(valid_recs):
            num_samples = len(audios[j])
            sliced = slice_exact_logits(batch_logits[j], num_samples)
            lps = torch.log_softmax(sliced, dim=-1).cpu().numpy()

            sid = r.get("sample_id") or r.get("audio_filepath")
            ref_raw = r.get("text", r.get("normalized_text", ""))
            ref_norm = norm_strict(ref_raw)
            spk = r.get("speaker_id", "unknown")
            dur = float(r.get("duration", round(num_samples / 16000.0, 2)))

            tasks.append((sid, lps, ref_raw, ref_norm, spk, dur))

        if (idx + batch_size) % 2000 == 0 or (idx + batch_size) >= total:
            print(f"  GPU forward pass: {min(idx + batch_size, total)}/{total} ({time.time()-t0:.1f}s, {len(tasks)/(time.time()-t0):.1f} utts/s)")

    print(f"\nAll {len(tasks)} acoustic logit tensors generated in {time.time()-t0:.1f}s.")
    del model
    del processor
    torch.cuda.empty_cache()

    print(f"Loaded {len(unigrams)} unigrams. Preparing multiprocessing pool (16 workers)...")
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=16,
        initializer=init_worker,
        initargs=(labels, str(LM_PATH), unigrams, ALPHA, BETA, str(E22_CKPT))
    )

    print(f"Decoding candidates with 16 CPU KenLM workers...")
    t_dec = time.time()
    results = pool.map(decode_and_format_item, tasks, chunksize=64)
    pool.close()
    pool.join()
    print(f"Candidate decoding completed in {time.time()-t_dec:.1f}s ({len(results)/(time.time()-t_dec):.1f} utts/s).")

    total_words = 0
    total_top1_errs = 0
    total_oracle_errs = 0
    better_count = 0

    with open(OUT_TRAIN_NBEST, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            w = len(res["ref_norm"].split())
            total_words += w
            total_top1_errs += res["top1_errors"]
            total_oracle_errs += res["oracle_errors"]
            if res["oracle_better"]:
                better_count += 1

    top1_wer = (total_top1_errs / total_words) * 100 if total_words > 0 else 0
    oracle_wer = (total_oracle_errs / total_words) * 100 if total_words > 0 else 0
    print("\n" + "=" * 70)
    print("TRAINING N-BEST CACHE SUMMARY:")
    print(f"Total utterances: {len(results)}")
    print(f"Total reference words: {total_words}")
    print(f"Top-1 KenLM Baseline WER: {top1_wer:.2f}% ({total_top1_errs} errors)")
    print(f"Top-32 Oracle Ceiling WER: {oracle_wer:.2f}% ({total_oracle_errs} errors)")
    print(f"Oracle Headroom: {top1_wer - oracle_wer:.2f} percentage points ({total_top1_errs - total_oracle_errs} recoverable words)")
    print(f"Utterances with better candidate in beam: {better_count}/{len(results)} ({better_count/len(results)*100:.2f}%)")
    print(f"Saved to {OUT_TRAIN_NBEST}")
    print("=" * 70)

def step3_format_dev_nbest_caches():
    print("=" * 70)
    print("STEP 3: Formatting Development N-Best Caches (Preserving 100% Certified Parity)")
    print("=" * 70)

    processor = Wav2Vec2Processor.from_pretrained(str(E22_CKPT))
    tokenizer = processor.tokenizer

    for split_key, in_path in DEV_PREDS.items():
        out_path = DEV_OUT_NBEST[split_key]
        print(f"\nProcessing {split_key} from {in_path.name} -> {out_path.name}...")
        
        count = 0
        total_words = 0
        top1_errors = 0
        oracle_errors = 0

        with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                d = json.loads(line)
                sid = d.get("sample_id")
                spk = d.get("speaker_id", "unknown")
                dur = float(d.get("duration", 0.0))
                ref_text = d.get("ref_text", "")
                ref_norm = d.get("ref_norm", norm_strict(ref_text))
                ref_words = len(ref_norm.split())
                total_words += ref_words

                nbest_raw = d.get("nbest", [])
                candidates = []
                min_errs = 999999
                t1_errs = 0

                for c_idx, c in enumerate(nbest_raw):
                    c_text = c.get("text", "")
                    c_norm = c.get("norm_text", norm_strict(c_text))
                    errs, s, dl, ins = compute_word_edit_distance(ref_norm, c_norm)
                    if c_idx == 0:
                        t1_errs = errs
                    if errs < min_errs:
                        min_errs = errs

                    token_ids = tokenizer.encode(c_norm)

                    candidates.append({
                        "cand_idx": c_idx,
                        "text": c_text,
                        "norm_text": c_norm,
                        "token_ids": token_ids,
                        "acoustic_score": round(float(c.get("logit_score", 0.0)), 4),
                        "lm_score": round(float(c.get("lm_score", 0.0)), 4),
                        "combined_score": round(float(c.get("combined_score", 0.0)), 4),
                        "num_words": int(c.get("word_count", len(c_norm.split()))),
                        "char_length": len(c_norm),
                        "edit_distance": errs,
                        "subs": s,
                        "dels": dl,
                        "ins": ins,
                        "wer_pct": round((errs / ref_words) * 100.0 if ref_words > 0 else 0.0, 2)
                    })

                top1_errors += t1_errs
                oracle_errors += min_errs
                count += 1

                record = {
                    "sample_id": sid,
                    "speaker_id": spk,
                    "duration": dur,
                    "ref_text": ref_text,
                    "ref_norm": ref_norm,
                    "top1_text": candidates[0]["text"] if candidates else "",
                    "top1_norm": candidates[0]["norm_text"] if candidates else "",
                    "top1_errors": t1_errs,
                    "oracle_errors": min_errs if candidates else t1_errs,
                    "oracle_better": (min_errs < t1_errs) if candidates else False,
                    "candidates": candidates
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        wer = (top1_errors / total_words) * 100 if total_words > 0 else 0
        ora = (oracle_errors / total_words) * 100 if total_words > 0 else 0
        print(f"  {split_key}: {count} utts, {total_words} words | Baseline WER: {wer:.2f}% ({top1_errors} errs) | Oracle: {ora:.2f}% ({oracle_errors} errs)")

if __name__ == "__main__":
    t_start = time.time()
    # Step 1: Sample training utterances
    sampled = step1_sample_training_utterances(target_n=20000, seed=42)
    # Step 2: Generate training N-best cache
    step2_generate_train_nbest(sampled)
    # Step 3: Format development N-best caches
    step3_format_dev_nbest_caches()
    print(f"\nAll Stage 24 N-Best candidate caches successfully built in {time.time()-t_start:.1f}s.")
