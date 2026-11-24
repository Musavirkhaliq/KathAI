# Practical Lab: Executing the 8-Domain Benchmark Evaluation

---

## Lab Objectives
1. Load evaluation manifests across multiple simulated acoustic domains.
2. Execute automated transcription and normalization.
3. Generate the official 8-domain evaluation scorecard and composite micro-averaged WER.

---

## Lab Script: `lab_certified_eval.py`

```python
def run_benchmark_lab():
    # Simulated domain evaluation
    domain_data = {
        "Studio Clean": (120, 1000),      # E=120, N=1000 -> 12.0%
        "Broadcast News": (180, 1000),    # E=180, N=1000 -> 18.0%
        "WAAZ Sermons": (520, 1000),      # E=520, N=1000 -> 52.0%
        "Community Mobile": (280, 1000),  # E=280, N=1000 -> 28.0%
    }
    
    tot_E = sum(v[0] for v in domain_data.values())
    tot_N = sum(v[1] for v in domain_data.values())
    
    print("=== Official Benchmark Scorecard ===")
    for dom, (E, N) in domain_data.items():
        print(f"Domain: {dom:20s} | WER: {E/N*100:5.2f}%")
        
    composite_wer = tot_E / tot_N
    print("---------------------------------------------")
    print(f"Composite Multi-Domain WER: {composite_wer*100:5.2f}%")
    assert abs(composite_wer - 0.275) < 1e-4, "Composite calculation error!"
    print("SUCCESS: Benchmark protocol executed cleanly!")

if __name__ == "__main__":
    run_benchmark_lab()
```
