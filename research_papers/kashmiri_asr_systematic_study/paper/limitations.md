# Limitations and Threats to Validity

In the interest of scientific rigor and academic transparency, we explicitly document the limitations, boundaries, and threats to validity in this work:

1. **Repeated Diagnostic Inspection of `dev_assessment.jsonl`**:
   The 1,001-sample evaluation split (`dev_assessment.jsonl`, 16,845 words) was repeatedly evaluated during exploratory diagnostic audits in Stages 11–21B. Although acoustic model weights were never trained on this set, hyperparameter tuning (KenLM grid search, beam width selection, ranker thresholds) was verified against it. Consequently, `dev_assessment.jsonl` must be interpreted as a **diagnostic development benchmark**. A definitive claim of blind generalization requires evaluating models on the untouched sealed benchmark (`benchmark_v1_1_primary.jsonl`).

2. **Marginal Statistical Significance of Decoder Rescoring**:
   While decoder calibration and Candidate-Relative Sequence Ranking reduced total word errors on `dev_assessment.jsonl` from 4,121 to 4,089–4,093 ($\Delta = -28$ to $-32$ errors; $\Delta \text{WER} = -0.17\%$), paired bootstrap resampling yielded $p = 0.0847$ with a 95% confidence interval of $[-0.402\%, +0.072\%]$, and paired t-test yielded $p = 0.317$. Because the confidence interval encompasses zero, this improvement represents a modest empirical trend rather than a statistically certified breakthrough.

3. **Sociolinguistic and Regional Dialectal Skew**:
   Over 85% of the curated audio originates from the Central (Yamraz / Srinagar) and Southern (Maraz) dialect areas. Northern dialects (Kamraz / Baramulla / Kupwara) and peripheral Dardic varieties (Kishtwari, Poguli) represent less than 15% of the data, resulting in higher acoustic error rates on non-standard dialects.

4. **Model Architecture Scope**:
   Our investigation focused on the self-supervised `facebook/wav2vec2-xls-r-300m` encoder with CTC and n-gram shallow fusion. We did not train multi-billion parameter foundation models (e.g., XLS-R 2B or Whisper-Large from scratch) due to compute constraints on a single-node DGX GB10 system.

5. **Acoustic Out-of-Beam Error Dominance**:
   Oracle analysis proves that 78.6% of remaining recognition errors (3,240 out of 4,121 errors) never enter the 16-best search beam. Consequently, language modeling and sequence rescoring cannot resolve these errors; further progress requires expanding acoustic training data volume and dialect diversity.
