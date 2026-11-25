# Code Walkthrough: Pipeline Inspection and Execution Trace

---

## 1. Architectural Architecture & Source File Index

This walkthrough traces the execution path of the production Kashmiri ASR pipeline across four primary codebase modules:
1. [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py): Unified top-level inference engine.
2. [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py): Mathematical CTC receptive field frame slicing.
3. [`src/decoding/kenlm_decoder.py`](file:///home/musa/Kashmiri%20Project/src/decoding/kenlm_decoder.py): Calibrated KenLM beam search decoder.
4. [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri%20Project/src/normalization/normalizer.py): Certified Perso-Arabic text normalizer.

---

## 2. Audio Preprocessing and Feature Slicing Trace

In [`src/asr/transcribe.py`](file:///home/musa/Kashmiri%20Project/src/asr/transcribe.py), audio is loaded and processed through the acoustic model:

```python
# From src/asr/transcribe.py (Lines 56-82)
def transcribe(
    self,
    audio_path: Union[str, Path],
    normalize: bool = False,
    use_kenlm: bool = False,
    temperature: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None
) -> Dict[str, Any]:
    # 1. Load audio and enforce 16 kHz mono float32
    audio = load_audio(audio_path, target_sr=16000)
    audio = normalize_audio(audio)
    num_samples = len(audio)

    # 2. Check minimum sample length against receptive field (400 samples = 25 ms)
    if num_samples < 400:
        return {"text": "", "tokens": [], "confidence": 0.0, "raw_text": ""}

    # 3. Model forward pass
    inputs = self.processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False
    ).input_values.to(self.device)

    with torch.no_grad():
        logits = self.model(inputs).logits[0]  # Shape: [T_full, |V|]

    # 4. Strict receptive field slicing to eliminate padding artifacts
    logits = slice_exact_logits(logits, num_samples)  # Shape: [T_valid, 47]
```

### Key Technical Details:
- **`num_samples < 400` Guard**: If an audio sample is shorter than the 400-sample receptive field of the Conv1d feature extractor, the model cannot perform even a single convolution without error.
- **`slice_exact_logits`**: Inspect [`src/decoding/ctc_slice.py`](file:///home/musa/Kashmiri%20Project/src/decoding/ctc_slice.py):
  ```python
  def compute_exact_ctc_length(input_samples: int) -> int:
      if input_samples < 400:
          return 0
      l = input_samples
      for k, s in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
          l = (l - k) // s + 1
      return int(l)
  ```
  This iterates through the exact layer parameters, returning the precise number of valid output frames.

---

## 3. Calibrated KenLM Beam Search Decoding

Once exact logits are sliced, [`src/decoding/kenlm_decoder.py`](file:///home/musa/Kashmiri%20Project/src/decoding/kenlm_decoder.py) applies temperature scaling and shallow fusion:

```python
# From src/decoding/kenlm_decoder.py (Lines 62-84)
def decode(self, logits: Union[np.ndarray, torch.Tensor]) -> str:
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    if logits.ndim == 3:
        logits = logits[0]

    # Apply calibrated acoustic temperature scaling
    # T = 1.08 softens peaky CTC posteriors
    scaled_logits = logits / self.temperature

    # pyctcdecode executes prefix beam search with KenLM shallow fusion:
    # S(w) = log P_CTC + alpha * log P_LM + beta * |w|
    transcription = self.decoder.decode(
        scaled_logits,
        beam_width=self.beam_width,            # W = 100
        beam_prune_logp=self.beam_prune_logp,  # Prunes beams below -10.0
        token_min_logp=self.token_min_logp     # Prunes tokens below -5.0
    )
    return transcription.strip()
```

### Why PyCTCDecode Requires `logits / self.temperature`:
Notice that `scaled_logits = logits / self.temperature` occurs before passing the array to `self.decoder.decode()`. If $T = 1.0$, the argmax CTC token often carries $99.9\%$ probability. At $T = 1.08$, that probability reduces to $\sim 92\%$, granting competing phonetically close candidates enough score to survive beam pruning until the 5-gram language model can evaluate their syntactic plausibility.

---

## 4. Orthographic Normalization Execution

The final stage passes the raw decoded string through [`src/normalization/normalizer.py`](file:///home/musa/Kashmiri%20Project/src/normalization/normalizer.py):

```python
# From src/normalization/normalizer.py (Lines 54-85)
def normalize_kashmiri_text(
    text: str,
    strip_diacritics: bool = False,
    normalize_digits: bool = True,
    preserve_zwnj: bool = True
) -> str:
    if not text or not isinstance(text, str):
        return ""

    # 1. Unicode NFC canonical decomposition and recomposition
    text = unicodedata.normalize('NFC', text)

    # 2. Perso-Arabic homoglyph unification
    # Replaces Arabic Kaf (U+0643) -> Kashmiri Keheh (U+06A9)
    # Replaces Arabic Yeh (U+064A) -> Kashmiri Yeh (U+06CC)
    # Strips Tatweel/Kashida (U+0640)
    for src, tgt in CHAR_MAP.items():
        text = text.replace(src, tgt)

    # 3. Kashmiri Central Vowel Harmonization
    text = text.replace('\u0671', '\u0627')  # Alef Wasla -> Alef
    text = text.replace('\u0622', '\u0627')  # Alef Madda -> Alef (if in unvocalized context)

    # 4. Whitespace and ZWNJ cleanup
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    return text.strip()
```

This ensures that the final evaluation metrics (WER and CER) reflect authentic recognition errors rather than trivial Unicode encoding differences.
