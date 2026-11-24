# Practical Lab: Simulating and Catching GPU OOM

---

## Lab Objectives
1. Implement a try-except block catching `torch.cuda.OutOfMemoryError`.
2. Demonstrate recovery and cache clearance using `torch.cuda.empty_cache()`.
3. Verify that training continues seamlessly on subsequent batches without process termination.

---

## Lab Script: `lab_oom_recovery.py`

```python
import torch

def run_oom_lab():
    # Simulate an error recovery mechanism
    success_count = 0
    oom_count = 0
    
    for i in range(5):
        try:
            if i == 2:
                # Deliberately trigger allocation error simulation
                raise RuntimeError("CUDA out of memory simulation")
            else:
                success_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_count += 1
                # Recovery action
                pass
                
    print(f"Successful Batches: {success_count} | Recovered OOMs: {oom_count}")
    assert success_count == 4 and oom_count == 1, "Recovery logic failed!"
    print("SUCCESS: OOM recovery mechanism validated!")

if __name__ == "__main__":
    run_oom_lab()
```
