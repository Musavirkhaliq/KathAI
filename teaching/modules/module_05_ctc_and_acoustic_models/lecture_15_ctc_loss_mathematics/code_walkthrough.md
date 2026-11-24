# Code Walkthrough: CTC Forward-Backward Verification

---

## 1. PyTorch Native `nn.CTCLoss` Usage

```python
import torch
import torch.nn as nn

# Production CTC loss instantiation
ctc_loss_fn = nn.CTCLoss(
    blank=0,
    reduction='mean',
    zero_infinity=True  # Clamps infinite loss values if T < U
)

# Shapes:
# logits: [T, B, vocab_size] (PyTorch standard expects T first)
# targets: [B, U] (padded with pad_idx)
# input_lengths: [B] (actual valid frame counts)
# target_lengths: [B] (actual valid label counts)
```
