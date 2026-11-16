# Kashmiri ASR Negative Results & Failed Experiments Registry

**Project**: EduCasheer Kashmiri Automatic Speech Recognition Research
**Scientific Purpose**: Document unsuccessful hypotheses, negative empirical results, and methodological dead ends to prevent wasted research effort and maintain complete transparency.

---

## 1. Summary of Documented Negative Results

| ID | Stage | Hypothesis Tested | Observed Outcome | Scientific Explanation | Takeaway for Future Research |
|:---:|:---:|:---|:---|:---|:---|
| **NEG-01** | Stage 12 | Synthetic RIR reverb & multi-condition noise improve field robustness. | KenLM WER degraded from 25.24% to 28.94% (+3.70% error penalty). | Heavy synthetic perturbations corrupt delicate formant cues for central vowels. | Do not train on synthetic noise without clean-speech anchor loss. |
| **NEG-02** | Stage 13 | Selective encoder layer unfreezing (U-01..03) stabilizes representations. | Models scored 24.91%-25.23% WER vs 24.47% for fully converged B-02C. | Constraining parameter adaptation reduces model capacity. | Full-network pre-convergence is strictly superior. |
| **NEG-03** | Stage 13 | Intermediate CTC supervision (I-01) speeds up convergence and lowers WER. | KenLM WER degraded to 25.02% (+0.55% error penalty). | Mid-network CTC loss forces premature discrete phonetic commitment. | Avoid intermediate CTC on 24-layer XLS-R for low-resource languages. |
| **NEG-04** | Stage 16 | Model soup weight interpolation across sequence checkpoints beats single models. | S16-05 achieved 24.47% WER, exactly matching parent checkpoint B-02C. | Interpolating weights in non-convex valleys does not overcome acoustic capacity limits. | Weight averaging yields no benefit when training data is the primary bottleneck. |
| **NEG-05** | Stage 17 | Semi-supervised pseudo-labeling on uncurated Tier B audio expands diversity. | KenLM WER degraded to 25.25% (+132 word errors). | Confirmation bias caused the model to memorize its own hallucinated phonemes. | Never ingest unverified pseudo-labels without human or cross-model verification. |
| **NEG-06** | Stage 24 | Deep neural candidate sequence rerankers (CandidateRelativeRanker) outperform KenLM. | Multi-partition WER showed zero statistically significant gain (p=0.738). | 4-gram KenLM shallow fusion with calibrated alpha/beta already extracts optimal beam value. | Permanently halt candidate-level neural sequence reranking. |
| **NEG-07** | Stage 24 | Short acoustic continuation (E24-CTC-01) improves conversational speech. | Greedy WER improved (+0.16pp), but KenLM beam decoding degraded (46.12% -> 48.42% WER). | Overfitting on small conversational datasets shifts posterior peak distributions away from LM priors. | Avoid short-duration fine-tuning on conversational speech without regularization. |

---

## 2. Detailed Technical Post-Mortems

### Post-Mortem NEG-01: Synthetic Multi-Condition Noise Augmentation (Stage 12)
- **Research Hypothesis**: Adding synthetic room impulse response (RIR) reverberation, background noise, and Opus lossy compression to training speech would make the acoustic model robust to low-cost mobile microphones.
- **Experimental Protocol**: Fine-tuned B-02C on 96.6k utterances with additive noise (SNR 10-25 dB) and simulated room acoustics.
- **Empirical Result**: Greedy WER increased from 27.84% to 30.47% (+2.63pp); KenLM WER increased from 25.24% to 28.94% (+3.70pp) on clean test speech.
- **Root Cause**: Kashmiri features dense, closely spaced central vowel phonemes (`U+0672`, `U+065F`, `U+065A`) and palatalized consonants whose acoustic distinctions rely on fine formant transitions between 1,200 Hz and 2,800 Hz. Synthetic noise masked these subtle spectral peaks, causing widespread phoneme confusion.

### Post-Mortem NEG-05: Uncurated Speech Pseudo-Labeling (Stage 17)
- **Research Hypothesis**: Utilizing B-02C to generate pseudo-labels for 45 hours of uncurated community speech would provide free acoustic scaling.
- **Experimental Protocol**: Filtered utterances with average posterior confidence > 0.85 and trained for 3 epochs.
- **Empirical Result**: Model `P17-PL` degraded to 25.25% KenLM WER (+132 word errors vs B-02C).
- **Root Cause**: The pseudo-labeling model suffered from systematic central vowel confusion. High confidence scores reflected overconfident CTC softmax distributions rather than phonetic truth, causing the model to reinforce its own errors.

### Post-Mortem NEG-06: Candidate-Relative Neural Reranking (Stage 24)
- **Research Hypothesis**: Since the N=16 Oracle ceiling is 19.23% WER (vs 24.27% for KenLM), a BiGRU neural reranker trained on 20,000 utterances should capture semantic and grammatical constraints and close the 5 pp headroom gap.
- **Experimental Protocol**: Extracted Top-16 hypotheses across 20k utterances, extracted acoustic/LM features, trained `CandidateRelativeRanker` (329k params) with margin-based pairwise ranking loss.
- **Empirical Result**: Dev calibration WER was 24.74% (Delta = +1 error vs KenLM, p = 0.738); Dev A disjoint holdout WER was 24.94% (Delta = +3 errors vs KenLM, p = 0.308).
- **Root Cause**: In CTC decoding, 78.6% of word errors are acoustic representation errors where the correct word does not exist anywhere in the Top-16 beam. For the remaining 21.4% where an alternative exists, the statistical 4-gram KenLM already captures local Kashmiri transition probabilities with sufficient accuracy that a shallow neural ranker cannot improve upon without a massive external text pretraining corpus (e.g., an 8B Kashmiri LLM).
