# Research Contributions

Our research presents five defensible scientific and empirical contributions to low-resource speech recognition and Dardic computational linguistics:

1. **Large-Scale Kashmiri Speech Data Consolidation & Graphemic Tokenization**:
   We consolidate, audit, and standardize 214.65 hours (106,487 audio recordings) of native Kashmiri speech across 12 distinct sources spanning studio, broadcast, conversational, and religious domains. We construct an audited 159-token Character CTC Tokenizer that natively encodes Kashmiri central vowel diacritics and ligatures, achieving 0.000% OOV across all splits and bypassing the subword fragmentation penalty.

2. **Systematic Empirical Scaling of Multilingual Acoustic Models**:
   We conduct controlled scaling experiments fine-tuning `facebook/wav2vec2-xls-r-300m` across four progressive data milestones (B-01 at 92.9h, B-02A at 104.9h, B-02B at 123.5h, and B-02C at 165.3h), proving that data expansion lowers validation WER from 33.86% to 27.84% (greedy) and 25.24% (KenLM) ($p < 0.0001$). We also empirically audit foundation models, showing that zero-shot Whisper fails catastrophically (>84%–107% WER) and Meta MMS lacks official standard Kashmiri adapters.

3. **Evaluation Integrity, Sample-Rate Forensics & Metric Reconciliation**:
   We identify a critical preprocessing flaw in historical low-resource ASR pipelines: ingesting 22,050 Hz audio unresampled into 16,000 Hz models inflated error rates by ~3% absolute. Mathematically exact Kaiser windowed sinc resampling eliminated 493 phantom word errors on frozen weights. Reconciling 30+ historical checkpoints on canonical 16 kHz audio proves that foundation B-02C (4,122 errors) and sequence-continuation E-03 (4,121 errors) are statistically equivalent (delta = 1 word error across 16,845 words; $p = 0.963$), disproving prior claims of massive continuation gains.

4. **Rigorous Documentation of Negative Results in Acoustic Optimization**:
   We document extensive negative results across 15+ experimental arms: synthetic multi-condition noise augmentation (R-02 regressed clean WER to 28.94%), progressive layer unfreezing (U-01..03 plateaued at 24.91%–25.19%), intermediate CTC supervision (25.02%), and noisy pseudo-labeling (P17-PL regressed to 25.25%). These controlled experiments demonstrate that short-budget architectural interventions cannot compensate for converged full-corpus acoustic representations.

5. **Search Space Characterization, Decoder Calibration & Candidate-Relative Ranking**:
   We demonstrate that beam search integrity is essential: restoring beam width 16 with history pruning eliminates an 84-error search pruning penalty. Domain calibration of KenLM parameters ($\alpha=0.20, \beta=-0.50$) and Candidate-Relative Sequence Ranking modestly lower errors to 24.27%–24.30% ($p=0.0847$). N-best Oracle analysis reveals a 19.23% WER ceiling (3,240 errors), proving that 881 word errors (21.4% of total errors) are recoverable search failures, while 78.6% of errors stem from acoustic out-of-beam deficiencies.
