# Practical Lab: Building a Minimal End-to-End CTC Pipeline

---

## Lab Objective
In this laboratory session, students will implement a complete, self-contained CTC inference pipeline in pure PyTorch, including:
1. Synthetic audio generation.
2. A toy 1D convolutional acoustic front-end.
3. CTC blank-collapsing decoding algorithm.
4. Unicode character mapping from `configs/vocab.json`.

---

## Step 1: Set Up the Vocabulary Mapping

```python
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal Kashmiri Perso-Arabic subset vocabulary
VOCAB = {
    "<blank>": 0,
    "<pad>": 1,
    "|": 2,      # Word separator / space
    "ا": 3, "ب": 4, "پ": 5, "ت": 6, "ٹ": 7,
    "ر": 8, "ش": 9, "ك": 10, "م": 11, "ن": 12,
    "و": 13, "ی": 14, "ٲ": 15, "ۄ": 16
}
IDX_TO_CHAR = {v: k for k, v in VOCAB.items()}
```

---

## Step 2: Implement the Deterministic CTC Greedy Collapse

```python
def ctc_greedy_decode(logits: torch.Tensor, blank_idx: int = 0) -> str:
    # [Docstring]
    Decodes frame-level logits into a text string using greedy argmax
    and the CTC collapse operator B.
    
    Args:
        logits: Tensor of shape [T, vocab_size]
        blank_idx: Index of the CTC blank token
    Returns:
        Decoded text string
    # [Docstring]
    # 1. Take argmax over vocabulary dimension at each time frame
    best_tokens = torch.argmax(logits, dim=-1).tolist()
    
    # 2. Collapse consecutive duplicates
    collapsed = []
    prev_token = None
    for token in best_tokens:
        if token != prev_token:
            collapsed.append(token)
            prev_token = token
            
    # 3. Strip blank tokens and map to characters
    emitted_chars = []
    for token in collapsed:
        if token != blank_idx and token in IDX_TO_CHAR:
            char = IDX_TO_CHAR[token]
            if char == "|":
                emitted_chars.append(" ")
            elif not char.startswith("<"):
                emitted_chars.append(char)
                
    return "".join(emitted_chars)
```

---

## Step 3: Verification on Synthetic Logits

```python
# Construct synthetic logit tensor simulating emission of 'کٲشُر'
# Target tokens: ك (10), ٲ (15), ش (9), ر (8)
T_frames = 12
vocab_size = len(VOCAB)
synthetic_logits = torch.full((T_frames, vocab_size), -10.0)

# Frame 0-1: blank
synthetic_logits[0:2, 0] = 5.0
# Frame 2-3: ك
synthetic_logits[2:4, 10] = 6.0
# Frame 4-5: ٲ
synthetic_logits[4:6, 15] = 6.0
# Frame 6-7: blank (separator)
synthetic_logits[6:8, 0] = 4.0
# Frame 8-9: ش
synthetic_logits[8:10, 9] = 6.0
# Frame 10-11: ر
synthetic_logits[10:12, 8] = 6.0

result = ctc_greedy_decode(synthetic_logits)
print(f"Decoded Output: '{result}'")
assert result == "کٲشر", f"Expected 'کٲشر', got '{result}'"
print("SUCCESS: CTC Collapse verified!")
```
