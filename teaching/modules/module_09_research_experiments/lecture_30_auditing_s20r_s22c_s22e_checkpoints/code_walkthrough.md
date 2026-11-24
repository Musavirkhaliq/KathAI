# Code Walkthrough: Checkpoint Forensics Engine (`src/evaluation/audit_checkpoint.py`)

---

## 1. Inspecting the Checkpoint Auditor

```python
import hashlib
import torch

def audit_checkpoint_state_dict(checkpoint_path: str) -> dict:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    sha = hashlib.sha256()
    total_params = 0
    
    # Sort keys for deterministic hash
    sorted_keys = sorted(state_dict.keys())
    for k in sorted_keys:
        t = state_dict[k]
        if isinstance(t, torch.Tensor):
            total_params += t.numel()
            sha.update(k.encode('utf-8'))
            sha.update(t.contiguous().numpy().tobytes())
            
    # Inspect linear head bias
    b0 = None
    if "lm_head.bias" in state_dict:
        b0 = state_dict["lm_head.bias"][0].item()
        
    return {
        "filepath": checkpoint_path,
        "total_parameters": total_params,
        "tensor_sha256": sha.hexdigest(),
        "blank_bias": b0
    }
```
