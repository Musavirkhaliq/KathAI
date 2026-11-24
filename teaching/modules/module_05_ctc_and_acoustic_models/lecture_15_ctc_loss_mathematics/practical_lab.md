# Practical Lab: Implementing CTC Forward Recursion in NumPy

---

## Lab Objectives
1. Implement the forward variable $\alpha_t(s)$ in NumPy.
2. Verify total sequence probability $P(\mathbf{l} \mid \mathbf{X}) = \alpha_T(2U) + \alpha_T(2U+1)$.
3. Compare numerical forward-backward likelihood against PyTorch `nn.CTCLoss`.

---

## Lab Script: `lab_ctc_forward.py`

```python
import numpy as np
import torch
import torch.nn as nn

def compute_ctc_forward_prob(log_probs, target_labels, blank_idx=0):
    # log_probs: [T, vocab_size]
    # target_labels: [U]
    T, V = log_probs.shape
    U = len(target_labels)
    
    # Extended target sequence l' of length 2U + 1
    extended = [blank_idx]
    for lbl in target_labels:
        extended.extend([lbl, blank_idx])
    L_prime = len(extended)
    
    # Forward alpha table in log-space
    alpha = np.full((T, L_prime), -np.inf)
    
    # Base case t = 0
    alpha[0, 0] = log_probs[0, blank_idx]
    alpha[0, 1] = log_probs[0, extended[1]]
    
    for t in range(1, T):
        for s in range(L_prime):
            # Same state or previous state
            prev_sum = np.logaddexp(alpha[t-1, s], alpha[t-1, s-1] if s > 0 else -np.inf)
            
            # Skip blank if applicable
            if s > 1 and extended[s] != blank_idx and extended[s] != extended[s-2]:
                prev_sum = np.logaddexp(prev_sum, alpha[t-1, s-2])
                
            alpha[t, s] = prev_sum + log_probs[t, extended[s]]
            
    # Total loss
    total_log_prob = np.logaddexp(alpha[T-1, L_prime-1], alpha[T-1, L_prime-2])
    return -total_log_prob

# Verification
T = 6
V = 5
target = [1, 2] # Two character sequence
logits = torch.randn(T, 1, V)
log_probs = torch.log_softmax(logits, dim=-1)

# PyTorch Loss
ctc_loss = nn.CTCLoss(blank=0, reduction='sum')
in_len = torch.tensor([T])
tg_len = torch.tensor([len(target)])
pt_loss = ctc_loss(log_probs, torch.tensor([target]), in_len, tg_len).item()

# Custom Loss
my_loss = compute_ctc_forward_prob(log_probs.squeeze(1).numpy(), target, blank_idx=0)
print(f"PyTorch CTC Loss: {pt_loss:.5f} | Custom Forward CTC Loss: {my_loss:.5f}")
assert abs(pt_loss - my_loss) < 1e-4, "Loss mismatch!"
print("SUCCESS: CTC forward algorithm validated!")

if __name__ == "__main__":
    run_lab = None
```
