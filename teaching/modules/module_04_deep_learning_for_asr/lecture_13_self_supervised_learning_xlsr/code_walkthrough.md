# Code Walkthrough: Contrastive Loss Computation

---

## 1. Simulating InfoNCE Masked Contrastive Loss

```python
import torch
import torch.nn.functional as F

def compute_contrastive_loss(
    context_vectors: torch.Tensor,   # [batch, masked_frames, 1024]
    target_quantized: torch.Tensor,  # [batch, masked_frames, 1024]
    distractors: torch.Tensor,       # [batch, masked_frames, num_distractors, 1024]
    temperature: float = 0.1
) -> torch.Tensor:
    # 1. Normalize vectors for cosine similarity
    c = F.normalize(context_vectors, dim=-1)
    q_pos = F.normalize(target_quantized, dim=-1)
    q_neg = F.normalize(distractors, dim=-1)
    
    # 2. Positive similarity: [batch, masked_frames, 1]
    pos_sim = torch.sum(c * q_pos, dim=-1, keepdim=True) / temperature
    
    # 3. Negative similarity: [batch, masked_frames, num_distractors]
    c_expanded = c.unsqueeze(-2)
    neg_sim = torch.sum(c_expanded * q_neg, dim=-1) / temperature
    
    # 4. Concatenate: positive is at index 0
    logits = torch.cat([pos_sim, neg_sim], dim=-1) # [batch, masked_frames, 1 + K]
    
    # 5. Target labels are all zeros
    labels = torch.zeros(logits.shape[0] * logits.shape[1], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels)
    
    return loss
```
