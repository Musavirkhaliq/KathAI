# Lecture 13: Self-Supervised Speech Representations (wav2vec 2.0 and XLS-R)

---

## 1. The Pretraining Paradigm Shift in Speech

For decades, training an Automatic Speech Recognition model required thousands of hours of audio paired with word-for-word human transcriptions. For low-resource languages like Kashmiri, acquiring 1,000 hours of professional transcriptions would require millions of dollars and decades of linguistic labor.

**Self-Supervised Learning (SSL)** revolutionized speech processing by leveraging raw, unannotated speech recordings. By solving a self-supervised masked contrastive prediction task over raw audio, the model builds deep representations of acoustic physics, phonetic boundaries, and language phonotactics without requiring a single transcribed sentence.

---

## 2. The wav2vec 2.0 Objective: Masked Contrastive Loss

In **wav2vec 2.0** (Baevski et al., 2020), self-supervised pretraining consists of three interconnected components:

```
[ Raw Audio Waveform x ]
          │
          ▼
[ 7-Layer CNN Feature Extractor ] ──▶ Latent Representations z_t
          │                                  │
          ├──▶ [ Time Masking (6.5%) ]       ▼
          │         │                  [ Gumbel-Softmax Codebook ] ──▶ Quantized Targets q_t
          ▼         ▼
[ Transformer Context Network ] ──▶ Context Representations c_t
                    │
                    ▼
       [ Contrastive Loss L_m + Diversity Loss L_d ]
```

### 1. Latent Feature Extraction:
Raw audio is mapped to continuous latent feature vectors $\mathbf{z}_1, \dots, \mathbf{z}_T \in \mathbb{R}^{512}$.

### 2. Temporal Masking:
A portion of time steps ($\approx 6.5\%$ of start indices with span length 10 frames) is masked out. The masked latents $\tilde{\mathbf{z}}_t$ are replaced with a learnable mask embedding before feeding into the Transformer context network, yielding contextualized vectors $\mathbf{c}_t$.

### 3. Gumbel-Softmax Vector Quantization:
In parallel, unmasked continuous latents $\mathbf{z}_t$ are discretized into codebook entries $\mathbf{q}_t$ using Gumbel-Softmax vector quantization over 2 codebooks with 320 entries each ($320 \times 320 = 102,400$ discrete states).

### 4. Masked Contrastive Loss ($\mathcal{L}_m$):
For each masked frame $t$, the Transformer context vector $\mathbf{c}_t$ must identify the true quantized target $\mathbf{q}_t$ among a set of $K$ distractor candidates $\tilde{\mathbf{q}} \in \mathcal{Q}_t$ sampled from other masked frames within the utterance:

$$\mathcal{L}_m = -\log \frac{\exp\left( \text{sim}(\mathbf{c}_t, \mathbf{q}_t) / \tau \right)}{\sum_{\tilde{\mathbf{q}} \in \mathcal{Q}_t} \exp\left( \text{sim}(\mathbf{c}_t, \tilde{\mathbf{q}}) / \tau \right)}$$

where $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^T \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ is cosine similarity, and $\tau = 0.1$ is temperature.

---

## 3. XLS-R: Cross-Lingual Representation Learning at Planetary Scale

**XLS-R** (Babu et al., 2021) scaled the wav2vec 2.0 objective across **436,000 hours of public speech in 128 languages**, encompassing:
- VoxPopuli (parliamentary debates)
- Common Voice (crowdsourced speech)
- BABEL (low-resource conversational telephone)
- Multilingual LibriSpeech (audiobooks)
- VoxLingua107 (YouTube speech)

### Why XLS-R is Transformative for Kashmiri:
Although Kashmiri itself had minimal representation in the 436,000-hour pretraining mix, XLS-R pretrained extensively on related Indo-Aryan and Iranian languages (Urdu, Hindi, Punjabi, Sindhi, Persian, Pashto).
The shared universal acoustic manifold pre-configures the 24 Transformer layers with:
1. Vocal tract physics and formant transition dynamics.
2. Sibilant and affricate acoustic representations.
3. Cross-lingual phonemic representations that drastically reduce the supervised training data required to achieve high accuracy.
