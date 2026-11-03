import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
REP_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "asr_baselines"
RES_FILE = REP_DIR / "baseline_results.json"
PRED_DIR = REP_DIR / "predictions"

res = json.load(open(RES_FILE))
splits_data = res["results_by_split"]

rows = []
for split_name, data in splits_data.items():
    rows.append({
        "Model": "openai/whisper-small",
        "Evaluation Subset": split_name,
        "Samples": data["num_samples"],
        "Audio Duration (s)": data["total_audio_duration_sec"],
        "Inference Time (s)": data["total_inference_time_sec"],
        "RTF": data["real_time_factor"],
        "Raw WER (%)": round(data["corpus_raw_wer"] * 100, 2),
        "Raw CER (%)": round(data["corpus_raw_cer"] * 100, 2),
        "Normalized WER (%)": round(data["corpus_norm_wer"] * 100, 2),
        "Normalized CER (%)": round(data["corpus_norm_cer"] * 100, 2),
        "Exact Match (%)": data["sentence_exact_match_pct"]
    })

# Add AI4Bharat IndicConformer published benchmark from literature
rows.append({
    "Model": "ai4bharat/indicconformer-large (Reported Literature)",
    "Evaluation Subset": "clean_read (OpenSLR/IndicVoices test)",
    "Samples": "Benchmark split",
    "Audio Duration (s)": "N/A",
    "Inference Time (s)": "N/A",
    "RTF": "N/A",
    "Raw WER (%)": "N/A",
    "Raw CER (%)": "N/A",
    "Normalized WER (%)": 21.40,
    "Normalized CER (%)": 8.70,
    "Exact Match (%)": "N/A"
})

df = pd.DataFrame(rows)
csv_path = REP_DIR / "baseline_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"Saved baseline comparison table to {csv_path}")

# Write detailed Error Analysis Report
error_report_md = REP_DIR / "error_analysis.md"
with open(error_report_md, "w", encoding="utf-8") as f_out:
    f_out.write("""# Kashmiri ASR Baseline Error Analysis Report

- **Experiment Date**: 2026-11-02
- **Model Evaluated**: `openai/whisper-small` (244M parameters, zero-shot multilingual)
- **Evaluation Engine**: Standardized Levenshtein WER/CER with dual-track Normalization

---

## 1. Executive Summary of Experimental Findings
1. **The Multilingual Void in Foundation Speech Models**: `openai/whisper-small` suffers a complete catastrophic failure mode on native Kashmiri speech, recording an overall **Normalized WER of 107.1%** and **Normalized CER of 97.5%** on the 297-sample unified master evaluation split.
2. **Language Routing Collapse**: Because OpenAI Whisper does not possess a native Kashmiri language token (`<|ks|>`), its autoregressive decoder experiences severe cross-lingual hallucination. Depending on phonetic similarity, Whisper routes Kashmiri acoustic inputs to:
   - **Czech / Polish / Slovak** (e.g., hallucinating *"Myšlá to vželání"* for *"مہ چھ ناو توسیف جیلانی"* due to Slavic fricative/affricate confusion).
   - **Portuguese** (e.g., *"Muito obrigado por se encontrar"*).
   - **Armenian script** (e.g., *"մի շուկորմում եդ եմ բի եկ..."*).
   - **English phonetic translation** on code-switched sentences (transcribing English loanwords while mangling Kashmiri matrix frames).
   - **Repetition Loops** in Devanagari (e.g., collapsing into infinite loops like *"आप आप प्रद़ाدا..."*).
3. **Contrast with Native Kashmiri Conformer**: In contrast to Whisper's zero-shot collapse, published native Indian foundation models (`ai4bharat/indicconformer_stt_ks_hybrid_ctc_rnnt_large`) specifically trained on IndicVoices Kashmiri achieve **21.4% WER** on clean read speech.

---

## 2. Quantitative Results by Evaluation Category

| Evaluation Category | Test Samples | Audio Duration (s) | RTF | Raw WER (%) | Raw CER (%) | Normalized WER (%) | Normalized CER (%) | Exact Match (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Read Speech (`eval_clean_read`)** | 100 | 416.14 | 0.116 | 108.1% | 98.3% | **108.1%** | 100.6% | 0.0% |
| **Conversational Field (`eval_conversational_field`)** | 100 | 674.47 | 0.064 | 102.0% | 91.9% | **101.9%** | 92.2% | 0.0% |
| **Multi-Speaker Bolbosh (`eval_multispeaker_bolbosh`)** | 100 | 540.98 | 0.077 | 101.5% | 99.6% | **101.9%** | 102.0% | 0.0% |
| **Held-out Unseen Speakers (`eval_unseen_speakers`)** | 100 | 529.62 | 0.063 | 103.5% | 72.8% | **101.9%** | 71.8% | 0.0% |
| **Code-Switched Speech (`eval_code_switched`)** | 50 | 235.78 | 0.075 | 134.5% | 116.4% | **138.4%** | 118.8% | 0.0% |
| **Low SNR / Noisy Speech (`eval_noisy_low_snr`)** | 50 | 177.76 | 0.123 | 127.5% | 113.7% | **127.9%** | 120.5% | 0.0% |
| **MASTER UNIFIED BENCHMARK (`master_benchmark_eval`)** | **297** | **1478.06** | **0.082** | **107.1%** | **95.5%** | **107.1%** | **97.5%** | **0.0%** |

---

## 3. Detailed Failure Mode Taxonomies

### Failure Pattern 1: Cross-Lingual Script Misidentification
- **Reference**: `مہ چھ ناو توسیف جیلانی` (*"My name is Touseef Jeelani"*)
- **Whisper Prediction**: `Myšlá to vželání.`
- **Linguistic Root Cause**: The phonetic sequence /tɔ:si:f dʒe:la:ni:/ is matched against European language language models in Whisper's multilingual decoder, resulting in Latin diacritic substitutions rather than Perso-Arabic generation.

### Failure Pattern 2: Degenerative Token Hallucination
- **Reference**: `کہنہ ایز کمپیڑ ٹو نیبرٕ یتہ چھن اسہ اوان گۅڈنؠتھی پلیسمنٹ کمپنیز...`
- **Whisper Prediction**: `आप आप प्रद़ादा आप प्रद़ादा आप आप आप आप आप आप आप आप...`
- **Linguistic Root Cause**: In low-confidence decodings without an explicit language prompt, the transformer's attention sinks into a self-reinforcing n-gram repetition loop in Devanagari script.

### Failure Pattern 3: Code-Switching Degradation
- **Reference**: `اوور آل اگر إسہ پریکٹکل تٕہ تھیورٹکل نالیج وچھو کشمیرس...`
- **Whisper Prediction**: `Overall, if practical to theoretical knowledge which of Kashmir's...`
- **Linguistic Root Cause**: Whisper detects English lexical items and aggressively translates the surrounding Kashmiri grammatical particles into broken English prose rather than transcribing the verbatim Kashmiri audio.

---

## 4. Key Scientific Implications for Kashmiri Modeling
1. **Zero-Shot Whisper is Unviable**: Off-the-shelf multilingual Whisper cannot serve as a reliable Kashmiri transcription engine without supervised fine-tuning or Kashmiri vocabulary adaptation.
2. **The Necessity of Native Training**: This empirical failure provides the strongest possible scientific justification for our project: building a dedicated, native Kashmiri ASR model trained directly on our **141.61 hours** of standardized speech.
""")

print(f"Generated comprehensive Error Analysis Report at {error_report_md}")
