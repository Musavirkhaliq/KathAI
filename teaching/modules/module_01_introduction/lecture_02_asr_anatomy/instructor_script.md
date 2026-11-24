# Instructor Delivery Script: Lecture 02 (The Anatomy of Modern ASR Pipelines)

## Timetable & Pedagogical Flow (90 Minutes Total)

```
00:00 - 00:15 | Review & Motivation: Why the Classical Hybrid Pipeline Failed for Kashmiri
00:15 - 00:35 | Paradigm Face-off: CTC vs. AED (Whisper) vs. Transducer (RNN-T)
00:35 - 00:55 | Deconstructing the 6-Stage EduCasheer Production Stack
00:55 - 01:15 | Live Code Walkthrough: src/inference/pipeline.py and configs/vocab.json
01:15 - 01:25 | Decoding Mechanics: The Interplay of Alpha, Beta, and Acoustic Temperature
01:25 - 01:30 | Summary, Student Misconceptions & Lab Orientation
```

---

## Detailed Delivery Script

### 00:00 - 00:15 | Review & Motivation
- **Action**: Step to the board. Write the three components of classical ASR: `AM * Lexicon * LM`.
- **Speaking Script**:
  > "Welcome back. In Lecture 01, we saw that training an Urdu ASR model and testing it on Kashmiri resulted in an 88.4% WER disaster. Today, we look inside the engine room.
  > If you open a standard textbook from 2015, speech recognition is taught as an acoustic model, a pronunciation lexicon, and an n-gram language model composed into an HMM-WFST graph.
  > Why couldn't we just build that for Kashmiri?
  > Because of the *pronunciation lexicon*. In English, you have CMUdict—134,000 words phonetically transcribed by linguists over decades. In Kashmiri, we have no such lexicon. Kashmiri has central vowels, vowel length contrasts, and fluid orthography. If your pipeline depends on a phonetic lexicon, you are dead before you write a single line of training code.
  > Modern End-to-End ASR bypasses the lexicon entirely. Today, we map out the complete anatomical architecture of that solution."

### 00:15 - 00:35 | Paradigm Comparison
- **Board Layout**:
  - Left: `CTC (Graves 2006)` - Collapse operator $\mathcal{B}$, monotonic, frame independence assumption mitigated by SSL encoder.
  - Middle: `AED / Whisper (Vaswani 2017)` - Autoregressive decoder, hallucination risk, quadratic attention cost.
  - Right: `Transducer (Graves 2012)` - 2D lattice, forward-backward joint network, computationally heavy.
- **Key Insight**: Emphasize why CTC + Pretrained XLS-R is the golden standard for low-resource languages: it refuses to hallucinate words that have no acoustic support.

### 00:35 - 00:55 | The 6-Stage Production Stack
- Trace the lifecycle of a 16 kHz audio buffer through the pipeline.
- Detail the exact dimension transformations:
  - $\mathbf{x} \in \mathbb{R}^{B \times 160000}$ (10 seconds)
  - CNN features $\mathbf{z} \in \mathbb{R}^{B \times 499 \times 512}$ (stride 320)
  - Transformer hidden states $\mathbf{h} \in \mathbb{R}^{B \times 499 \times 1024}$
  - Projection logits $\mathbf{L} \in \mathbb{R}^{B \times 499 \times 47}$

### 00:55 - 01:15 | Live Code Walkthrough
- Open `src/inference/pipeline.py`.
- Point out line 82: `resample_to_16k(audio, orig_sr)`.
- Point out the character dictionary mapping in `configs/vocab.json`.
- Demonstrate the blank token index (0) and pad token handling.

### 01:15 - 01:30 | Wrap-up & Discussion
- Ask students: *"If we double the LM weight $\alpha$ from 0.3 to 0.6, what happens to names of small villages in Kashmir?"*
- Guide them to the realization that an aggressive language model replaces rare acoustic names with common dictionary words.
