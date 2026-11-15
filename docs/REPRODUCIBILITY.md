# Kashmiri ASR Reproducibility & Research Engineering Guide

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research  
**Hardware Tested**: NVIDIA DGX Spark (NVIDIA GB10 GPU, PyTorch 2.14.1+cu130, CUDA 13.0)  
**Root Working Directory**: `/home/musa/Kashmiri Project`  
**Last Verified**: November 14, 2026  

---

## 1. Quickstart: 3-Minute Smoke Test

Before executing long-running jobs, execute the verified end-to-end smoke test to ensure CUDA devices, model checkpoints, and audio decoders are operating correctly:

```bash
cd "/home/musa/Kashmiri Project"
python3 test_batch_transcribe.py
```

Expected output:
```text
Loading Kashmiri ASR model from .../B-02C/checkpoints/best_model on cuda...
Model loaded successfully!
------------------------------------------------------------
Sample: xkaab_000221
Ref: فیورِٹ بُک چھ مےٚ قُرآنِ شریٖف تِکیازِ قُرآنِ شریٖف چھُ اَسِہ حضرت محمدصلى الله عليه وسلم سٔندِس پؠٹھ نٲزِل گوٚمُت
Hyp: فیورٹ بک چھ مےٚ قرانۍ شرف تکیاز قرانہ شرف چھ اسہ حضرت محمد صالیہ لہ السلم سنس پٮ ٹھ نٲزل گوٚمت
```

---

## 2. Standardized Entry Points for Common Research Workflows

### Workflow 1: Inspect Available Datasets
Audit the master provenance registry and verify local file existence:
```bash
python3 kashmiri_dataset/scripts/audit_master_inventory.py
```
To view the dataset breakdown:
```bash
python3 -c "
import json
with open('asr_training_eligible_manifest.jsonl') as f:
    from collections import Counter
    c = Counter(json.loads(line)['source_dataset'] for line in f)
    for k, v in c.most_common():
        print(f'{k:<35}: {v:,} samples')
"
```

---

### Workflow 2: Prepare & Normalize Training Data
Run the canonical Perso-Arabic text normalizer on an arbitrary manifest:
```bash
python3 kashmiri_dataset/scripts/normalize_transcripts.py \
    --input kashmiri_dataset/manifests/raw_manifest.jsonl \
    --output kashmiri_dataset/manifests/normalized_manifest.jsonl
```

---

### Workflow 3: Validate Manifests & Quality Gates
Execute the comprehensive 5-stage pre-training validation gates:
```bash
python3 kashmiri_dataset/scripts/pretraining_validation_gates.py \
    --manifest asr_training_eligible_manifest.jsonl \
    --check_audio \
    --check_ctc_feasibility
```

---

### Workflow 4: Train an Acoustic Model (Stage 9 Architecture)
Train a controlled acoustic model (e.g., `B-02C`) using length-bucketed dynamic batching and BF16 AMP:
```bash
python3 -u kashmiri_dataset/scripts/train_stage9_full_schedule.py \
    --experiment B-02C \
    --batch_size 32 \
    --max_epochs 20 \
    --min_epochs 3 \
    --patience 4
```
*Note: Available experiments are `B-02A`, `B-02B`, `B-02C`, or `all`.*

---

### Workflow 5: Resume an Existing Experiment
If training was interrupted by system preemption or maintenance, pass the `--resume` flag to resume from the latest checkpoint without recomputing earlier steps:
```bash
python3 -u kashmiri_dataset/scripts/train_stage9_full_schedule.py \
    --experiment B-02C \
    --resume
```

---

### Workflow 6: Run Inference on Audio
Transcribe any standard audio file (WAV, MP3, FLAC, OGG) using the CLI utility:

```bash
# SOTA Model Greedy Decoding
python3 transcribe_kashmiri.py /path/to/audio.wav

# Normalized Output
python3 transcribe_kashmiri.py /path/to/audio.wav --normalize

# Explicit Checkpoint Override
python3 transcribe_kashmiri.py /path/to/audio.wav \
    --model-dir "kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02A/checkpoints/best_model"
```

To run interactive microphone transcription from the terminal:
```bash
./record_and_transcribe.sh 5   # Record for 5 seconds and transcribe
```

To launch the web GUI:
```bash
python3 web_asr_app.py
# Open your browser at http://localhost:8080
```

---

### Workflow 7: Evaluate Word Error Rate (WER) & Character Error Rate (CER)
Compute Levenshtein distance with Kashmiri Perso-Arabic normalizations:
```python
from kashmiri_dataset.scripts.metrics_evaluator import calculate_wer_and_cer, normalize_perso_arabic_kashmiri

ref = "سۄ چھےٚ کٲشُر زَبانہِ ہِنٛز کَتھ"
hyp = "سۄ چھےٚ کٲشُر زَبانہِ ہِنٛز کَتھ"

norm_ref = normalize_perso_arabic_kashmiri(ref)
norm_hyp = normalize_perso_arabic_kashmiri(hyp)

metrics = calculate_wer_and_cer(norm_ref, norm_hyp)
print(f"WER: {metrics['wer']*100:.2f}%, CER: {metrics['cer']*100:.2f}%")
```

---

### Workflow 8: Run Decoding Comparisons (Greedy, Beam Search, KenLM)
Compare all three decoding regimes across a model checkpoint:
```bash
python3 kashmiri_dataset/scripts/evaluate_stage9_decoders.py \
    --experiment B-02C \
    --split val \
    --beam_width 16 \
    --alpha 0.25 \
    --beta 0.00
```

---

### Workflow 9: Evaluate on Frozen Benchmarks
Evaluate performance against the official sealed held-out test split:
```bash
python3 kashmiri_dataset/scripts/run_b02a_followup_evaluation.py
```
Results will be output to:
- `matched_decoder_test_comparison.csv`
- `sourcewise_test_results.csv`
- `paired_test_bootstrap.json`

---

### Workflow 10: Statistical Significance Testing
Run 10,000 paired bootstrap resamples between two model prediction files:
```bash
python3 kashmiri_dataset/scripts/paired_bootstrap_test.py \
    --ref_manifest kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl \
    --hyp1 kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02A/predictions/val_greedy.jsonl \
    --hyp2 kashmiri_dataset/experiments/native_asr_training/experiments/stage9_b02_acoustic_comparison/B-02C/predictions/val_greedy.jsonl \
    --samples 10000
```

---

### Workflow 11: Romanization & Transliteration
Convert Perso-Arabic Kashmiri to standardized ASCII or Phonetic Romanization:
```python
import sys
sys.path.insert(0, "kashmiri_dataset/scripts")
from kashmiri_transliteration_engine import KashmiriTransliterationEngine

eng_ascii = KashmiriTransliterationEngine(mode="ascii")
eng_phon = KashmiriTransliterationEngine(mode="phonetic")

text = "مےٚ چُھ کٲشُر بولُن پَسَند"
print("ASCII:", eng_ascii.transliterate(text))      # me chhu kaeshur bolun pasand
print("Phonetic:", eng_phon.transliterate(text))   # me chhu kạ̄śur bōlun pasand
```

---

### Workflow 12: Creating a New Research Experiment Safely
Follow this step-by-step protocol to prevent broken paths or overwritten checkpoints:

1. **Assign a Unique Experiment ID**: E.g., `B-04-xlsr300m-augmented` or `EXP-2026-X1`.
2. **Create an Isolated Experiment Directory**:
   ```bash
   EXP_DIR="kashmiri_dataset/experiments/native_asr_training/experiments/EXP-2026-X1"
   mkdir -p "$EXP_DIR"/{checkpoints,configs,logs,predictions,reports}
   ```
3. **Save Explicit Frozen Configuration**: Store exact learning rate, batch size, seed, and input manifest inside `$EXP_DIR/configs/config.json`.
4. **Point Checkpoints to `$EXP_DIR/checkpoints/`**: Ensure optimizer state, processor config, and tokenizer are saved alongside weights.
5. **Never Modify Existing Baselines**: Always evaluate on `kashmiri_dataset/experiments/native_asr_training/data_splits/val.jsonl` using `calculate_wer_and_cer` from `metrics_evaluator.py`.
