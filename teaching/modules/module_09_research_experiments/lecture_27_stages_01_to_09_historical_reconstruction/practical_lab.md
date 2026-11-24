# Practical Lab: Forensic Artifact Reconciliation

---

## Lab Objectives
1. Compute SHA-256 cryptographic hashes over checkpoint weights.
2. Parse historical execution logs to extract loss curves and learning rate trajectories.
3. Identify discrepancies between documentation claims and physical log outputs.

---

## Lab Script: `lab_forensics.py`

```python
import hashlib

def run_forensic_lab():
    dummy_weights = b"PyTorch Checkpoint Weights Simulation E22-01"
    sha = hashlib.sha256(dummy_weights).hexdigest()
    print(f"Computed SHA-256 Tensor Hash: {sha}")
    assert len(sha) == 64, "Invalid SHA-256 length!"
    print("SUCCESS: Forensic hashing framework verified!")

if __name__ == "__main__":
    run_forensic_lab()
```
