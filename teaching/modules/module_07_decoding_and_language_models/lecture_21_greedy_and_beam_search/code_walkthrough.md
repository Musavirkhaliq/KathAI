# Code Walkthrough: Prefix Beam Search (`src/decoding/beam_search.py`)

---

## 1. Pure Python Prefix Beam Search Implementation

```python
from collections import defaultdict
import numpy as np

def ctc_prefix_beam_search(probs, vocab, beam_width=64, blank_idx=0):
    # probs: [T, vocab_size]
    T, V = probs.shape
    
    # beam: map prefix -> (p_blank, p_non_blank)
    beam = {"": (1.0, 0.0)}
    
    for t in range(T):
        next_beam = defaultdict(lambda: (0.0, 0.0))
        
        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb
            
            # 1. Extend with blank
            p_b_curr, p_nb_curr = next_beam[prefix]
            next_beam[prefix] = (p_b_curr + p_total * probs[t, blank_idx], p_nb_curr)
            
            # 2. Extend with non-blank characters
            for c_idx in range(V):
                if c_idx == blank_idx:
                    continue
                c = vocab[c_idx]
                p_c = probs[t, c_idx]
                
                if len(prefix) > 0 and prefix[-1] == c:
                    # Repeating character: p_nb extends same prefix, p_b creates new prefix
                    p_b_ex, p_nb_ex = next_beam[prefix]
                    next_beam[prefix] = (p_b_ex, p_nb_ex + p_nb * p_c)
                    
                    p_b_new, p_nb_new = next_beam[prefix + c]
                    next_beam[prefix + c] = (p_b_new, p_nb_new + p_b * p_c)
                else:
                    new_prefix = prefix + c
                    p_b_new, p_nb_new = next_beam[new_prefix]
                    next_beam[new_prefix] = (p_b_new, p_nb_new + p_total * p_c)
                    
        # Prune to top W prefixes
        sorted_prefixes = sorted(
            next_beam.keys(),
            key=lambda p: next_beam[p][0] + next_beam[p][1],
            reverse=True
        )
        beam = {p: next_beam[p] for p in sorted_prefixes[:beam_width]}
        
    best_prefix = max(beam.keys(), key=lambda p: beam[p][0] + beam[p][1])
    return best_prefix
```
