# Practical Lab: Tensor-Level Checkpoint Auditing

---

## Lab Objectives
1. Load multiple model state dicts.
2. Compute deterministic tensor-level SHA-256 hashes.
3. Distinguish model variants based on weight hashes and linear head bias inspection.

---

## Lab Script: `lab_checkpoint_audit.py`

```python
import torch
import hashlib

def run_audit_lab():
    # Build two simulated models
    m1 = {"fc.weight": torch.ones(5, 5), "fc.bias": torch.zeros(5)}
    m2 = {"fc.weight": torch.ones(5, 5), "fc.bias": torch.zeros(5)}
    m3 = {"fc.weight": torch.ones(5, 5), "fc.bias": torch.ones(5)}
    
    def hash_sd(sd):
        sha = hashlib.sha256()
        for k in sorted(sd.keys()):
            sha.update(k.encode('utf-8'))
            sha.update(sd[k].numpy().tobytes())
        return sha.hexdigest()
        
    h1 = hash_sd(m1)
    h2 = hash_sd(m2)
    h3 = hash_sd(m3)
    
    print(f"Model 1 Tensor Hash: {h1[:16]}...")
    print(f"Model 2 Tensor Hash: {h2[:16]}...")
    print(f"Model 3 Tensor Hash: {h3[:16]}...")
    
    assert h1 == h2, "Identical models produced different hashes!"
    assert h1 != h3, "Distinct models produced identical hashes!"
    print("SUCCESS: Deterministic tensor hashing validated!")

if __name__ == "__main__":
    run_audit_lab()
```
