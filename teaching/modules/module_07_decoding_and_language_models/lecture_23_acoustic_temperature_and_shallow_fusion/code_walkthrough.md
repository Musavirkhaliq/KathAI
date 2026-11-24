# Code Walkthrough: 2D Grid Search for Alpha and Beta

---

## 1. Grid Search Implementation (`src/decoding/tune_hyperparams.py`)

```python
import numpy as np

def run_grid_search(val_samples, decoder, alpha_range, beta_range):
    best_wer = float('inf')
    best_params = (None, None)
    
    results = {}
    for alpha in alpha_range:
        for beta in beta_range:
            decoder.set_hyperparams(alpha=alpha, beta=beta)
            total_wer = 0.0
            
            for audio, ref_text in val_samples:
                hyp = decoder.decode(audio)
                total_wer += compute_wer(ref_text, hyp)
                
            avg_wer = total_wer / len(val_samples)
            results[(alpha, beta)] = avg_wer
            
            if avg_wer < best_wer:
                best_wer = avg_wer
                best_params = (alpha, beta)
                
    return best_params, best_wer
```
