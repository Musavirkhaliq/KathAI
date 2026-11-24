# Lecture 32: Code-Switching and Dialectal Adaptation in Kashmiri Speech

---

## 1. The Socio-Linguistic Matrix of Kashmir

In the contemporary Kashmir Valley, natural speech is rarely monolingual:
- **Intra-Sentential Code-Switching**: Speakers fluidly interweave Kashmiri with **English** (technical, administrative, academic terms) and **Urdu** (religious, formal, colloquial discourse).
- A typical sentence: `"تۄہہِ پَزِ website پؠٹھ login کَرُن"` (*You must log in on the website*).
- **Phonetic Adaptation**: English loanwords undergo phonological assimilation (e.g., English /p/ becomes aspirated /pʰ/, /t/ becomes retroflex /ʈ/).

An ASR system that assumes a strictly monolingual Kashmiri lexicon will fail whenever a speaker uses common technical English vocabulary.

---

## 2. Low-Rank Adaptation (LoRA) for Dialectal Fine-Tuning

Kashmiri possesses distinct regional dialect groups:
1. **Maraz (Southern Kashmiri)**: Characterized by distinctive vowel shifts and morphological suffixes.
2. **Kamraz (Northern Kashmiri)**: Slower cadence, differing central vowel durations.
3. **Srinagar Standard**: The administrative and media standard.

Fine-tuning all 317 million parameters of the base model for each regional dialect creates massive storage overhead and risks catastrophic forgetting of the standard language.

EduCasheer employs **Low-Rank Adaptation (LoRA)** (Hu et al., 2021) to adapt model weights with minimal parameter overhead:

$$\mathbf{W}_{\text{adapted}} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

where:
- $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ is the frozen pretrained weight matrix.
- $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ are low-rank matrices with rank $r \ll \min(d, k)$ (typically $r = 8$).
- Only $\mathbf{A}$ and $\mathbf{B}$ are trained, adding fewer than **1.2 million parameters** ($< 0.4\%$ of model size).
- Distinct LoRA adapters can be swapped dynamically at runtime depending on the speaker's detected dialect region!
