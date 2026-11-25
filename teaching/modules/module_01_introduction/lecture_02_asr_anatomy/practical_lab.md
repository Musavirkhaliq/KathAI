# Practical Lab: Receptive Field Calculation, Temperature Scaling, and Prefix Decoding

## Objective
In this laboratory, students will implement and verify the core architectural components of modern CTC speech recognition:
1. Simulating the exact 7-layer Conv1d receptive field and valid frame slicing.
2. Implementing a pure Python Prefix Beam Search decoder with blank probability tracking.
3. Measuring the mathematical effect of acoustic temperature scaling on posterior distribution entropy.

---

## Exercise 1: Receptive Field & CTC Frame Slicing Verification

Run the following Python script to simulate audio lengths and verify exact CTC frame boundaries:

```python
#!/usr/bin/env python3
# Lab Exercise 1: Conv1d Feature Extractor Simulation

def compute_exact_ctc_length(input_samples: int) -> int:
    kernels = [10, 3, 3, 3, 3, 2, 2]
    strides = [5, 2, 2, 2, 2, 2, 2]
    
    if input_samples < 400:
        return 0
    
    length = input_samples
    for k, s in zip(kernels, strides):
        length = (length - k) // s + 1
    return int(length)

# Test across various audio durations at 16,000 Hz
test_durations_sec = [0.01, 0.024, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

print("Duration (s) | Num Samples | Valid CTC Frames | Effective Frame Rate (Hz)")
print("-" * 65)
for sec in test_durations_sec:
    samples = int(sec * 16000)
    frames = compute_exact_ctc_length(samples)
    rate = frames / sec if sec > 0 and frames > 0 else 0.0
    print(f"{sec:12.3f} | {samples:11d} | {frames:16d} | {rate:24.2f}")
```

### Lab Questions for Part 1:
1. Why does an audio sample of $0.024$ seconds ($384$ samples) produce exactly $0$ frames?
2. What is the minimum number of samples required to produce at least $2$ valid CTC frames?

---

## Exercise 2: Implementing a Pure Python Prefix Beam Search Decoder

The core algorithmic challenge in CTC decoding is that multiple distinct alignment paths map to the same label sequence. In prefix beam search, we track two distinct probabilities for each candidate prefix $\ell$:
- $P_b(\ell, t)$: Total probability of prefix $\ell$ at time $t$ given that the last emission was the **blank token** $\epsilon$.
- $P_{nb}(\ell, t)$: Total probability of prefix $\ell$ at time $t$ given that the last emission was a **non-blank character**.

The total probability of prefix $\ell$ is $P_{\text{tot}}(\ell, t) = P_b(\ell, t) + P_{nb}(\ell, t)$.

```python
#!/usr/bin/env python3
# Lab Exercise 2: Prefix Beam Search with Blank Tracking
import numpy as np

def prefix_beam_search(logits, vocab, beam_width=3, blank_idx=0):
    # [Docstring]
    # Pure Python CTC Prefix Beam Search Decoder
    # logits: [T, |V|] unnormalized emission scores
    # [Docstring]
    T, V = logits.shape
    # Compute frame-level probabilities via softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Initialize beams with empty prefix ""
    # beam dict: prefix -> (P_blank, P_non_blank)
    beams = {"": (1.0, 0.0)}
    
    for t in range(T):
        new_beams = {}
        p_blank = probs[t, blank_idx]
        
        for prefix, (p_b, p_nb) in beams.items():
            p_tot = p_b + p_nb
            
            # Case 1: Emit blank token ε
            # Extends prefix without changing label string
            if prefix not in new_beams:
                new_beams[prefix] = [0.0, 0.0]
            new_beams[prefix][0] += p_tot * p_blank
            
            # Case 2: Emit character c in vocab
            for c_idx in range(V):
                if c_idx == blank_idx:
                    continue
                c = vocab[c_idx]
                p_c = probs[t, c_idx]
                new_prefix = prefix + c
                
                if new_prefix not in new_beams:
                    new_beams[new_prefix] = [0.0, 0.0]
                
                # If extending identical character as previous end:
                # Must differentiate whether blank intervened!
                if len(prefix) > 0 and c == prefix[-1]:
                    # If blank intervened, can extend identical char
                    new_beams[new_prefix][1] += p_b * p_c
                    # If no blank intervened, repeating char collapses to existing prefix!
                    new_beams[prefix][1] += p_nb * p_c
                else:
                    new_beams[new_prefix][1] += p_tot * p_c
        
        # Prune to beam width W based on total probability
        sorted_prefixes = sorted(
            new_beams.keys(),
            key=lambda p: new_beams[p][0] + new_beams[p][1],
            reverse=True
        )[:beam_width]
        
        beams = {p: (new_beams[p][0], new_beams[p][1]) for p in sorted_prefixes}
        print(f"Frame {t}: Top Beams: {[(p, round(beams[p][0]+beams[p][1], 4)) for p in beams]}")
    
    best_prefix = max(beams.keys(), key=lambda p: beams[p][0] + beams[p][1])
    return best_prefix

# Test run with toy vocabulary: 0=ε, 1='ك', 2='ٲ', 3='ش'
toy_vocab = {0: "<blank>", 1: "ك", 2: "ٲ", 3: "ش"}
toy_logits = np.array([
    [2.0, 0.1, 0.1, 0.1],  # t=0: blank dominant
    [0.1, 3.0, 0.1, 0.1],  # t=1: 'ك' dominant
    [1.5, 1.5, 0.1, 0.1],  # t=2: 'ك' vs blank contest
    [0.1, 0.1, 2.5, 0.1],  # t=3: 'ٲ' dominant
])

result = prefix_beam_search(toy_logits, toy_vocab, beam_width=3)
print(f"\nFinal Decoded Transcription: {result}")
```

---

## Exercise 3: Acoustic Temperature Scaling & Distribution Entropy

Run the following script to observe how temperature $T \in \{0.5, 1.0, 1.08, 2.0\}$ alters the Shannon entropy of acoustic emissions:

```python
#!/usr/bin/env python3
# Lab Exercise 3: Temperature Scaling & Shannon Entropy
import numpy as np

# Simulate a peaked logit vector from a trained CTC model
raw_logits = np.array([8.5, 2.1, 1.4, 0.2, -1.0])

temperatures = [0.5, 1.0, 1.08, 1.5, 2.0]

print("Temp (T) | Max Prob (Top-1) | Runner-up Prob | Entropy H(Y) in bits")
print("-" * 65)
for T in temperatures:
    scaled = raw_logits / T
    exp_s = np.exp(scaled - np.max(scaled))
    p = exp_s / np.sum(exp_s)
    
    top1 = np.max(p)
    top2 = np.sort(p)[-2]
    entropy = -np.sum(p * np.log2(p + 1e-12))
    
    print(f"{T:8.2f} | {top1:16.4f} | {top2:14.4f} | {entropy:20.4f}")
```
