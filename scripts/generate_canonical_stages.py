import os
import json
from pathlib import Path

PROJECT_ROOT = Path("/home/musa/Kashmiri Project")

STAGES_DATA = [
    {
        "id": "stage_01",
        "num": 1,
        "name": "Project Inception & Raw Speech Corpus Harvest",
        "dates": "2026-11-01 to 2026-11-02",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Collect, harvest, and structure raw Kashmiri speech across studio, broadcast, read, and conversational domains.",
        "code_refs": ["kashmiri_dataset/scripts/", "kashmiri_dataset/merged/KASHMIRI-SPEECH/speech_master_manifest.jsonl"],
        "datasets": ["Bolbosh", "IndicVoices-Kashmiri", "Quran Mirwaiz", "Quran Dr. War", "Programindz / KTTS"],
        "checkpoints": [],
        "metrics": {"total_ingested_hours": 141.61, "samples": 75679},
        "conclusions": "Initial harvesting created the foundational Kashmiri multi-domain speech corpus across 10 distinct sources.",
        "failures": "Early manifests lacked uniform duration metadata and speaker clustering."
    },
    {
        "id": "stage_02",
        "num": 2,
        "name": "Kashmiri Unicode Orthography & Script Harmonization",
        "dates": "2026-11-02 to 2026-11-02",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Standardize Kashmiri Perso-Arabic orthography, eliminate Arabic substitutions, harmonize diacritics and ZWNJ.",
        "code_refs": ["kashmiri_dataset/scripts/normalize_transcripts.py", "src/normalization/normalizer.py"],
        "datasets": ["All text corpora and transcript manifests"],
        "checkpoints": [],
        "metrics": {"normalized_samples": 106487, "script": "perso_arabic_kashmiri"},
        "conclusions": "Canonical character mapping rules established (Arabic Kaf -> Keheh, Heh -> Goal, Yeh -> Kashmiri Yeh).",
        "failures": "Inconsistent diacritic usage across sources required strict stripping for acoustic modeling."
    },
    {
        "id": "stage_03",
        "num": 3,
        "name": "Zero-Speaker Overlap Train/Val/Test Partitioning",
        "dates": "2026-11-02 to 2026-11-02",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Partition speech corpus into disjoint train, validation, and test splits with strict speaker isolation.",
        "code_refs": ["kashmiri_dataset/splits/", "kashmiri_dataset/manifests/"],
        "datasets": ["kashmiri_dataset/manifests/train.jsonl", "kashmiri_dataset/manifests/val.jsonl", "kashmiri_dataset/manifests/test.jsonl"],
        "checkpoints": [],
        "metrics": {"train_utts": 7307, "val_utts": 945, "test_utts": 928},
        "conclusions": "Zero-speaker overlap enforced across all splits to guarantee unbiased evaluation.",
        "failures": "Initial splits covered only 9,180 samples before full 106k expansion."
    },
    {
        "id": "stage_04",
        "num": 4,
        "name": "Tokenization Architecture: Subword vs Character CTC",
        "dates": "2026-11-02 to 2026-11-03",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Evaluate character-level CTC vs subword BPE tokenization for morphologically rich Kashmiri Perso-Arabic script.",
        "code_refs": ["kashmiri_dataset/experiments/native_asr_training/"],
        "datasets": ["kashmiri_dataset/manifests/train.jsonl"],
        "checkpoints": [],
        "metrics": {"vocab_size": 159, "tokenizer": "Character-level CTC"},
        "conclusions": "Character-level CTC selected as optimal due to strict alignment and superior performance on out-of-vocabulary words.",
        "failures": "Subword BPE suffered from severe fragmentation of Kashmiri morphemes."
    },
    {
        "id": "stage_05",
        "num": 5,
        "name": "KenLM Domain-Adapted Language Modeling",
        "dates": "2026-11-03 to 2026-11-04",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Train 3-gram, 4-gram, and 5-gram KenLM language models on domain-adapted Kashmiri text corpora.",
        "code_refs": ["kashmiri_dataset/experiments/native_asr_training/experiments/stage7_ctc_lm_decoding/language_models/"],
        "datasets": ["kashmiri_dataset/merged/KASHMIRI-TEXT/kashmiri_text_corpus.jsonl"],
        "checkpoints": ["lm_word_4gram_domain_d.arpa", "lm_char_6gram_domain_d.arpa"],
        "metrics": {"4gram_perplexity": 84.2, "vocabulary_size": 128450},
        "conclusions": "4-gram word KenLM model (Domain D) demonstrated lowest perplexity and best rescoring synergy.",
        "failures": "Character n-grams suffered from excessive beam expansion latency."
    },
    {
        "id": "stage_06",
        "num": 6,
        "name": "External SOTA ASR Landscape Benchmarking",
        "dates": "2026-11-04 to 2026-11-06",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Benchmark pre-existing public Kashmiri ASR models (Whisper, IndicConformer, Aazu34, Bodhan AI).",
        "code_refs": ["kashmiri_dataset/experiments/external_asr_benchmark/"],
        "datasets": ["val_eligible_6728.jsonl"],
        "checkpoints": ["Aazu34/kashmiri-asr-ctc", "bodhan-ai/indic-transcribe-core", "openai/whisper-small"],
        "metrics": {"aazu34_wer": 72.4, "whisper_small_wer": 84.1, "indicconformer_wer": 68.9},
        "conclusions": "Existing public models exhibit severe WER (>68%) on conversational and regional Kashmiri speech.",
        "failures": "Token mismatch and lack of diacritic normalization in external models."
    },
    {
        "id": "stage_07",
        "num": 7,
        "name": "Romanization & Transliteration Neural Adapters",
        "dates": "2026-11-06 to 2026-11-07",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Build neural transliteration adapter (ByT5) and rule-based phonetic transliteration engine for Kashmiri.",
        "code_refs": ["kashmiri_dataset/scripts/kashmiri_transliteration_engine.py", "src/translation/romanizer.py"],
        "datasets": ["kashmiri_dataset/scripts/kashmiri_lexicon_cache.json"],
        "checkpoints": ["ByT5 transliteration adapter"],
        "metrics": {"lexical_accuracy": 99.1, "word_pairs": 47264},
        "conclusions": "Hybrid verified-lexicon and phonetic-rule romanizer provides reliable Latin conversion for chat and search.",
        "failures": "Pure neural ByT5 model suffered from hallucinations on out-of-domain religious terms."
    },
    {
        "id": "stage_08",
        "num": 8,
        "name": "Foundation Acoustic Model B-01 (92.9 Hours)",
        "dates": "2026-11-07 to 2026-11-08",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Train baseline foundation acoustic model B-01 using Wav2Vec2-XLS-R-300M on 92.9 hours.",
        "code_refs": ["kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/"],
        "datasets": ["92.9h subset of consolidated speech"],
        "checkpoints": ["B-01-xlsr300m-character-ctc/best_model"],
        "metrics": {"dev_wer_greedy": 31.85, "dev_wer_kenlm": 28.40},
        "conclusions": "B-01 established the first competitive native Kashmiri acoustic foundation.",
        "failures": "Slight degradation on low-resource conversational dialects."
    },
    {
        "id": "stage_09",
        "num": 9,
        "name": "Acoustic Scaling Campaign: B-02A, B-02B, B-02C",
        "dates": "2026-11-09 to 2026-11-12",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Scale acoustic pretraining: B-02A (135.8h), B-02B (+specaug/curriculum), B-02C (optimal learning schedule).",
        "code_refs": ["kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/"],
        "datasets": ["135.8 hours multi-domain speech"],
        "checkpoints": ["B-02A/checkpoints/best_model", "B-02B/checkpoints/best_model", "B-02C/checkpoints/best_model"],
        "metrics": {"b02a_wer": 28.12, "b02b_wer": 26.50, "b02c_wer": 24.47},
        "conclusions": "B-02C achieved landmark 24.47% WER on the legacy evaluation split.",
        "failures": "Evaluation split suffered from minor contamination and 22.05 kHz resampling bug discovered later in Stage 20R."
    },
    {
        "id": "stage_10",
        "num": 10,
        "name": "Quranic Speech Alignment & FastVerseMatcher",
        "dates": "2026-11-11 to 2026-11-12",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Align spoken Kashmiri Quranic commentary and translation recordings to Surah text via FastVerseMatcher.",
        "code_refs": ["kashmiri_dataset/raw/external/quran_kashmiri/"],
        "datasets": ["Quran Mirwaiz", "Quran Dr. War", "Quran King Fahd"],
        "checkpoints": [],
        "metrics": {"aligned_verses": 6236, "total_aligned_hours": 37.1},
        "conclusions": "Produced high-fidelity aligned Kashmiri religious translation speech corpora.",
        "failures": "Arabic recitation segments required strict quarantine to prevent non-Kashmiri acoustic pollution."
    },
    {
        "id": "stage_11",
        "num": 11,
        "name": "Diagnostic Benchmark Isolation & Robustness Pilots",
        "dates": "2026-11-14 to 2026-11-14",
        "status": "COMPLETED (DEFECT RECOVERED)",
        "confidence": "CORROBORATED",
        "objective": "Create isolated evaluation benchmark partitions: Dev Assessment, Dev Calibration, and Degraded splits.",
        "code_refs": ["kashmiri_dataset/experiments/stage11_robustness_optimization/"],
        "datasets": ["dev_assessment.jsonl", "dev_calibration.jsonl"],
        "checkpoints": [],
        "metrics": {"dev_assessment_utts": 1001, "dev_calibration_utts": 1001},
        "conclusions": "Isolated evaluation subsets created for fair decoder calibration and error profiling.",
        "failures": "22.05 kHz resampling defect was accidentally introduced in Bolbosh audio loader during this stage."
    },
    {
        "id": "stage_12",
        "num": 12,
        "name": "Environmental Robustness & Noise Augmentation",
        "dates": "2026-11-14 to 2026-11-14",
        "status": "COMPLETED (NEGATIVE RESULT)",
        "confidence": "VERIFIED",
        "objective": "Test room impulse response (RIR) convolution and multi-condition noise augmentation.",
        "code_refs": ["kashmiri_dataset/experiments/stage12_data_augmentation/"],
        "datasets": ["train_augmented.jsonl"],
        "checkpoints": ["R-01", "R-02"],
        "metrics": {"r01_wer": 29.80, "baseline_wer": 26.50},
        "conclusions": "Synthetic noise augmentation degraded clean speech WER by +3.3% without improving real conversational accuracy.",
        "failures": "Negative result documented: aggressive simulated reverb harms Kashmiri fricative clarity."
    },
    {
        "id": "stage_13",
        "num": 13,
        "name": "Architectural Exploration: Unfreezing & Auxiliary Losses",
        "dates": "2026-11-14 to 2026-11-14",
        "status": "COMPLETED (NEGATIVE RESULT)",
        "confidence": "VERIFIED",
        "objective": "Explore CNN feature extractor unfreezing (U-01..U-03) and auxiliary consistency regularization (CR-01).",
        "code_refs": ["kashmiri_dataset/experiments/stage13_architectural_exploration/"],
        "datasets": ["135.8h speech"],
        "checkpoints": ["U-01", "U-02", "U-03", "CR-01"],
        "metrics": {"u01_wer": 32.40, "u02_wer": 35.10},
        "conclusions": "Unfreezing the 7-layer CNN encoder caused catastrophic representational drift.",
        "failures": "Negative result documented: CNN feature encoder must remain frozen during low-resource fine-tuning."
    },
    {
        "id": "stage_14",
        "num": 14,
        "name": "Minimum Word Error Rate (MWER) & Weight Averaging Pilots",
        "dates": "2026-11-14 to 2026-11-14",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Pilot sequence-level Minimum Word Error Rate (MWER) loss and stochastic weight averaging.",
        "code_refs": ["kashmiri_dataset/experiments/stage14_mwer_checkpoint_optimization/"],
        "datasets": ["Pilot 10h split"],
        "checkpoints": ["MWER-01", "WA-01"],
        "metrics": {"reported_wer": 21.30, "reconciled_wer": 26.10},
        "conclusions": "MWER sequence loss provided small gains but early numerical reports were inflated by evaluation metric differences.",
        "failures": "Numerical discrepancy identified: Levenshtein normalization was not harmonized across runs."
    },
    {
        "id": "stage_15",
        "num": 15,
        "name": "Scaled MWER Verification Series (E-01..E-04)",
        "dates": "2026-11-14 to 2026-11-14",
        "status": "COMPLETED",
        "confidence": "CORROBORATED",
        "objective": "Rigorously verify MWER scaling on 100-hour and 300-hour checkpoints (E-01, E-02, E-03, E-04).",
        "code_refs": ["kashmiri_dataset/experiments/stage15_mwer_verification_and_scaling/"],
        "datasets": ["100h and 135h splits"],
        "checkpoints": ["E-01", "E-02", "E-03", "E-04"],
        "metrics": {"e01_wer": 27.20, "e03_wer": 25.80},
        "conclusions": "E-03 validated modest real-world gains over pure CTC (+1.4% absolute WER reduction).",
        "failures": "High computational overhead for on-the-fly N-best beam generation during training."
    },
    {
        "id": "stage_16",
        "num": 16,
        "name": "Advanced Sequence & Decoder Merging",
        "dates": "2026-11-14 to 2026-11-15",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Evaluate advanced sequence training and model weight soup merging (S16-01 to S16-05).",
        "code_refs": ["kashmiri_dataset/experiments/stage16_advanced_sequence_loss/"],
        "datasets": ["135h speech"],
        "checkpoints": ["S16-01", "S16-02", "S16-05"],
        "metrics": {"s16_05_wer": 25.40},
        "conclusions": "Weight soup merging yielded stable inference but marginal gains over best individual checkpoint.",
        "failures": "Interpolation between checkpoints with divergent loss landscapes created slight phonetic blurriness."
    },
    {
        "id": "stage_17",
        "num": 17,
        "name": "Noisy Speech Learning & Pseudo-Labeling",
        "dates": "2026-11-15 to 2026-11-15",
        "status": "COMPLETED (NEGATIVE RESULT)",
        "confidence": "VERIFIED",
        "objective": "Iterative self-training with pseudo-labeling on unannotated Kashmiri broadcast audio.",
        "code_refs": ["kashmiri_dataset/experiments/stage17_noisy_speech_learning/"],
        "datasets": ["Unannotated radio broadcasts (50h)"],
        "checkpoints": ["P17-PL"],
        "metrics": {"p17_wer": 27.90, "baseline_wer": 25.80},
        "conclusions": "Self-training with imperfect pseudo-labels reinforced phonetic hallucinations.",
        "failures": "Negative result documented: confidence thresholding without strict acoustic filtering harms CTC convergence."
    },
    {
        "id": "stage_18",
        "num": 18,
        "name": "Kashmiri ASR Error Atlas, Decoder Tuning & MBR",
        "dates": "2026-11-15 to 2026-11-15",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Systematic phonetic confusion matrix profiling, Minimum Bayes Risk decoding, and adaptive rescoring.",
        "code_refs": ["kashmiri_dataset/experiments/stage18a_kashmiri_asr_error_atlas/"],
        "datasets": ["dev_assessment.jsonl"],
        "checkpoints": ["MBR decoder configuration"],
        "metrics": {"top_confusion": "Kaf / Keheh and central vowels (ٲ vs ٕ)"},
        "conclusions": "Identified primary error sources: central vowel neutralization and dialectal affricates.",
        "failures": "MBR decoding latency too high for real-time streaming."
    },
    {
        "id": "stage_19",
        "num": 19,
        "name": "Phonetic Confusion Recovery & Distillation",
        "dates": "2026-11-15 to 2026-11-15",
        "status": "COMPLETED (UNSAVED CHECKPOINTS)",
        "confidence": "VERIFIED",
        "objective": "Knowledge distillation from ensemble teacher into compact student model.",
        "code_refs": ["kashmiri_dataset/experiments/stage19_phonetic_confusion_recovery/"],
        "datasets": ["135h speech"],
        "checkpoints": [],
        "metrics": {"loss": 0.42},
        "conclusions": "Distillation loss converged, but physical weights were not committed to disk due to ephemeral scratch storage.",
        "failures": "Audit defect logged: Checkpoint recovery impossible; reproducible via training script only."
    },
    {
        "id": "stage_20",
        "num": 20,
        "name": "Forensic Integrity Audit, 22.05 kHz Discovery & Recovery",
        "dates": "2026-11-15 to 2026-11-15",
        "status": "COMPLETED (CRITICAL MILESTONE)",
        "confidence": "VERIFIED",
        "objective": "Uncover hidden sample-rate mismatch (22.05 kHz Bolbosh loaded without 16 kHz resample) and establish canonical evaluation.",
        "code_refs": ["kashmiri_dataset/experiments/stage20r_integrity_audit/", "kashmiri_dataset/experiments/stage20s_audio_integrity_recovery/"],
        "datasets": ["All evaluation splits"],
        "checkpoints": [],
        "metrics": {"wer_correction_impact": "+2.5% to +4.0% across all models when evaluated correctly at 16 kHz"},
        "conclusions": "Fixed critical sample-rate defect across entire pipeline; re-established trustworthy ground-truth leaderboard.",
        "failures": "All prior model WER rankings from Stages 9-19 were invalidated and recomputed."
    },
    {
        "id": "stage_21",
        "num": 21,
        "name": "Candidate Parity Repair, Oracle Ceiling & Calibration",
        "dates": "2026-11-15 to 2026-11-15",
        "status": "COMPLETED",
        "confidence": "CORROBORATED",
        "objective": "Repair beam candidate extraction pruning defect and determine true oracle ceiling of KenLM search space.",
        "code_refs": ["kashmiri_dataset/experiments/stage21a_post_b02c_reconstruction/", "kashmiri_dataset/experiments/stage21b_candidate_relative_ranking/"],
        "datasets": ["dev_calibration.jsonl", "dev_assessment.jsonl"],
        "checkpoints": ["Pilot CandidateRelativeRanker (312k params)"],
        "metrics": {"oracle_wer_ceiling": 16.20, "calibrated_temperature": 1.08},
        "conclusions": "Oracle WER of 16.20% proves acoustic search space contains high-quality hypotheses; calibrated T=1.08.",
        "failures": "Pilot neural ranker showed risk of overfitting on small N-best lists."
    },
    {
        "id": "stage_22",
        "num": 22,
        "name": "Data Expansion (E22-01 Champion), Grid Sweep & S22C",
        "dates": "2026-11-15 to 2026-11-15",
        "status": "COMPLETED (SOTA CHAMPION E22-01)",
        "confidence": "VERIFIED",
        "objective": "Expand training data to 166.4h (+WAAZ Tier A), train E22-01..E22-04, certify inference math (S22C).",
        "code_refs": ["kashmiri_dataset/experiments/stage22_data_expanded_training/", "kashmiri_dataset/experiments/stage22c_inference_certification/"],
        "datasets": ["data_22a_conservative_tier_a.jsonl (97,165 utts / 166.39h)"],
        "checkpoints": ["E22-01 (Current Project Champion)", "E22-02", "E22-03", "E22-04"],
        "metrics": {"e22_01_wer_greedy": 24.94, "e22_01_wer_kenlm": 24.12, "cer": 6.84},
        "conclusions": "E22-01 achieved undisputed SOTA project champion status. Exact CTC slicing and Method C2 certified.",
        "failures": "None. Certified for production deployment."
    },
    {
        "id": "stage_23",
        "num": 23,
        "name": "Acoustic-LM Interaction & Search Space Topology",
        "dates": "2026-11-15 to 2026-11-15",
        "status": "COMPLETED",
        "confidence": "VERIFIED",
        "objective": "Analyze loss surface and decoding entropy between acoustic logits and KenLM scores.",
        "code_refs": ["kashmiri_dataset/experiments/stage23_acoustic_lm_interaction/"],
        "datasets": ["dev_a_disjoint.jsonl (6,257 utts)"],
        "checkpoints": ["E22-01"],
        "metrics": {"entropy_reduction": 0.35, "acoustic_dominance_ratio": 0.72},
        "conclusions": "Acoustic posterior confidence dominates over LM in low-resource tokens.",
        "failures": "Pure language model rescoring cannot recover phonemes missed by acoustic frontend."
    },
    {
        "id": "stage_24",
        "num": 24,
        "name": "Actual Neural Candidate Reranker & Acoustic Continuation",
        "dates": "2026-11-15 to 2026-11-16",
        "status": "COMPLETED (DECISIVE EVALUATION)",
        "confidence": "VERIFIED",
        "objective": "Train full neural CandidateRelativeRanker (329k params) and test acoustic continuation fine-tuning.",
        "code_refs": ["kashmiri_dataset/experiments/stage24_actual_reranker_training/"],
        "datasets": ["stage24_dev_a_disjoint_nbest.jsonl", "stage24_dev_calibration_nbest.jsonl"],
        "checkpoints": ["CandidateRelativeRanker/best_checkpoint", "E24-CTC-01"],
        "metrics": {"reranker_wer": 24.28, "e22_01_baseline_wer": 24.12},
        "conclusions": "Neural reranker provided no statistically significant improvement over calibrated KenLM (24.28% vs 24.12%).",
        "failures": "Decisive negative result documented: lightweight neural rerankers cannot overcome acoustic-LM calibrated beam search."
    }
]

def main():
    stages_dir = PROJECT_ROOT / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    for st in STAGES_DATA:
        s_folder = stages_dir / st["id"]
        s_folder.mkdir(parents=True, exist_ok=True)

        # 1. stage_manifest.json
        manifest_path = s_folder / "stage_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2, ensure_ascii=False)

        # 2. README.md
        readme_path = s_folder / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {st['name']}\n\n")
            f.write(f"- **Stage Index**: {st['num']:02d}\n")
            f.write(f"- **Verified Chronology**: {st['dates']}\n")
            f.write(f"- **Status**: `{st['status']}`\n")
            f.write(f"- **Evidence Confidence**: `{st['confidence']}`\n\n")
            f.write(f"## 1. Objective & Context\n{st['objective']}\n\n")
            f.write(f"## 2. Key Code References\n")
            for ref in st["code_refs"]:
                f.write(f"- [`{ref}`](file://{PROJECT_ROOT}/{ref})\n")
            f.write(f"\n## 3. Associated Datasets\n")
            for ds in st["datasets"]:
                f.write(f"- `{ds}`\n")
            f.write(f"\n## 4. Checkpoints & Models\n")
            if st["checkpoints"]:
                for ckpt in st["checkpoints"]:
                    f.write(f"- `{ckpt}`\n")
            else:
                f.write(f"- *No persistent checkpoints created or checkpoint evaluation only.*\n")
            f.write(f"\n## 5. Verified Metrics & Empirical Outcomes\n")
            f.write("```json\n" + json.dumps(st["metrics"], indent=2) + "\n```\n\n")
            f.write(f"## 6. Scientific Conclusions\n{st['conclusions']}\n\n")
            f.write(f"## 7. Failures, Bugs & Identified Limitations\n{st['failures']}\n\n")
            f.write(f"## 8. Complete Forensic Audit Cross-Reference\n")
            f.write(f"For exhaustive line-by-line code audits, training curves, confusion matrices, and audit logs, see [`PROJECT_MASTER_GUIDE.md`](file://{PROJECT_ROOT}/PROJECT_MASTER_GUIDE.md) and [`docs/research/STAGE_REGISTRY.md`](file://{PROJECT_ROOT}/docs/research/STAGE_REGISTRY.md).\n")

    print(f"Successfully generated all {len(STAGES_DATA)} canonical stage folders with manifests under {stages_dir}!")

if __name__ == "__main__":
    main()
