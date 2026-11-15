# Related Work and Prior Art Review

This document contextualizes the EduCasheer Kashmiri ASR systematic study within the broader landscapes of low-resource speech recognition, multilingual foundation models, Indic language technologies, and acoustic evaluation reproducibility.

---

## 1. Low-Resource and Indic Speech Recognition

Building automatic speech recognition systems for low-resource languages has historically been hindered by the scarcity of paired audio-text corpora. In South Asia, while Scheduled Eighth languages such as Hindi, Tamil, and Bengali have recently accumulated hundreds or thousands of hours of speech through national initiatives like Bhashini and AI4Bharat, Dardic languages like Kashmiri (*Kōshur*) remain critically underserved.

Early Kashmiri ASR work was virtually non-existent outside isolated acoustic experiments. In 2021, Google released the OpenSLR SLR122 corpus comprising approximately 2.2 hours of clean read speech across 67 speakers, which served as the primary public benchmark for multiple years. However, baseline Kaldi models trained on SLR122 suffered from severe domain brittleness (>42% WER).

In 2024, AI4Bharat released **IndicVoices** (Javed et al., 2024) and **IndicVoices-R** (Sankar et al., 2024), providing the first large-scale open collection of spontaneous Indian speech, including ~30 hours of conversational Kashmiri. However, published benchmarks from AI4Bharat (IndicConformer) reported a critical failure: when central vowel diacritics were stripped from evaluation text, the model achieved 41.20% WER, but when diacritics were retained strictly, WER degraded catastrophically to **66.59%**. This underscored the urgent need for dedicated Kashmiri orthographic modeling.

---

## 2. Multilingual Pretrained Speech Models: XLS-R, MMS, and Whisper

The dominant paradigm in speech recognition leverages self-supervised pretraining on massive uncurated audio collections.
- **XLS-R (Babu et al., 2021)**: Trained on 436,000 hours of speech across 128 languages using the Wav2Vec2 contrastive task (Baevski et al., 2020), XLS-R provides rich cross-lingual latent representations. However, its effectiveness on a specific low-resource language depends entirely on downstream fine-tuning and the design of the character classification head.
- **Massively Multilingual Speech (MMS, Pratap et al., 2023)**: Meta's MMS project extended coverage to 1,400+ languages using adapter modules. Crucially, as documented in our empirical audit, **zero adapter weights exist in MMS-1B for standard Kashmiri (`kas`)**, restricting its out-of-the-box utility.
- **OpenAI Whisper (Radford et al., 2023)**: While Whisper excels in high-resource zero-shot tasks, our audits confirm that Whisper-Small and Whisper-Large-v3 achieve $>84\%-107\%$ WER on Kashmiri speech. Because Whisper lacks an explicit language token for Kashmiri in its 51k-token vocabulary, its autoregressive decoder falls into cross-lingual hallucination loops, outputting repetitive Slavic or Hindi text.

---

## 3. Orthographic Modeling and the "Tokenization Tax"

Low-resource Perso-Arabic scripts encounter what Petrov et al. (2023) termed the "tokenization tax". Standard subword algorithms (BPE, SentencePiece) split words containing rare Kashmiri vowel diacritics (`U+0672`, `U+065F`, `U+065A`) into multiple byte-level fragments. In CTC acoustic modeling, excessive sequence lengths break alignment feasibility ($L > T$). By designing an audited 159-token Character CTC vocabulary with 0.00% OOV, our work demonstrates that character-level modeling bypasses the subword fragmentation penalty while natively modeling contrastive central vowels.

---

## 4. Evaluation Integrity and Reproducibility in Speech AI

The speech community has increasingly recognized that evaluation pipelines can introduce massive hidden confounders. From beam search tuning to audio normalization, unstandardized evaluation can distort leaderboards. Our study provides a rigorous case study in scientific auditing: uncovering how a 22.05 kHz unresampled audio bug accounted for a 3% WER distortion, and demonstrating that rigorous canonical evaluation restores exact reproducibility across 30+ historical checkpoints.
