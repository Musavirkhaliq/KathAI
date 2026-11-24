# Code Walkthrough: Historical MWER Loss Implementation

---

## 1. Inspecting the Failed Stage 19 MWER Module (`stages/stage_19/mwer_loss.py`)

```python
# HISTORICAL STAGE 19 EXPERIMENT - PRESERVED FOR FORENSIC EDUCATION
import torch
import torch.nn as nn
from src.evaluation.metrics import compute_levenshtein_ops

class MinimumWordErrorRateLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, nbest_log_probs, nbest_hypotheses, reference_text):
        # nbest_log_probs: [N] tensor of sequence log-probs
        # nbest_hypotheses: list of N strings
        ref_words = reference_text.split()
        
        # 1. Compute exact word error counts
        errors = []
        for hyp in nbest_hypotheses:
            hyp_words = hyp.split()
            S, D, I, C = compute_levenshtein_ops(ref_words, hyp_words)
            errors.append(S + D + I)
            
        err_tensor = torch.tensor(errors, dtype=torch.float32, device=nbest_log_probs.device)
        
        # 2. Renormalize probabilities with temperature
        probs = torch.softmax(nbest_log_probs / self.temperature, dim=-1)
        
        # 3. Expected WER loss
        expected_loss = torch.sum(probs * err_tensor)
        return expected_loss
```
