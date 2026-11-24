# Practical Lab: Beam Search vs. Greedy Search Benchmark

---

## Lab Objectives
1. Implement Prefix Beam Search with path merging.
2. Construct synthetic logits where greedy decoding yields an erroneous sequence while beam search finds the true mode.
3. Measure decoding runtime across beam widths $W \in \{1, 10, 50, 100\}$.

---

## Lab Script: `lab_beam_benchmark.py`

```python
import numpy as np
from src.decoding.beam_search import ctc_prefix_beam_search

def run_beam_lab():
    vocab = {0: "<blank>", 1: "ك", 2: "ٲ", 3: "ت"}
    
    # Synthetic logits where greedy fails
    # Frame 0: k=0.40, t=0.35, eps=0.25
    # Frame 1: a=0.40, t=0.35, eps=0.25
    probs = np.array([
        [0.25, 0.40, 0.00, 0.35],
        [0.25, 0.00, 0.40, 0.35]
    ])
    
    # Greedy picks [1, 2] -> "كٲ"
    greedy_tokens = np.argmax(probs, axis=-1)
    greedy_str = "".join([vocab[t] for t in greedy_tokens if t != 0])
    print(f"Greedy Transcription: '{greedy_str}'")
    
    # Beam search should pick "ت"
    beam_str = ctc_prefix_beam_search(probs, vocab, beam_width=10, blank_idx=0)
    print(f"Beam Search Transcription: '{beam_str}'")
    
    assert beam_str == "ت", f"Expected 'ت', got '{beam_str}'"
    print("SUCCESS: Beam search successfully overcame greedy fallacy!")

if __name__ == "__main__":
    run_beam_lab()
```
