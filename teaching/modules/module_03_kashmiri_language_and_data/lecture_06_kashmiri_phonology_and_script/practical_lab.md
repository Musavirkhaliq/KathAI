# Practical Lab: Formant Analysis of Kashmiri Vowels

---

## Lab Objectives
1. Implement Linear Predictive Coding (LPC) to extract formant frequencies from synthetic and natural vowel frames.
2. Plot the $F_1$ vs. $F_2$ vowel quadrilateral for front, central, and back vowels.
3. Quantify the acoustic distance between /i/, /ɨ/, and /u/.

---

## Lab Script: `lab_formant_extraction.py`

```python
import torch
import numpy as np
import scipy.signal

def extract_formants_lpc(waveform: np.ndarray, sr: int = 16000, lpc_order: int = 16):
    # 1. Apply pre-emphasis filter to boost high formants
    pre_emph = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    
    # 2. Windowing
    win = np.hamming(len(pre_emph))
    windowed = pre_emph * win
    
    # 3. LPC coefficients via Levinson-Durbin
    # Autocorrelation
    r = np.correlate(windowed, windowed, mode='full')
    r = r[len(windowed)-1 : len(windowed)-1 + lpc_order + 1]
    
    # Levinson-Durbin recursion
    a = scipy.linalg.solve_toeplitz((r[:-1], r[:-1]), r[1:])
    lpc_poly = np.concatenate(([1.0], -a))
    
    # 4. Find roots of LPC polynomial
    roots = np.roots(lpc_poly)
    roots = [r for r in roots if np.imag(r) >= 0]
    
    # Calculate frequencies and bandwidths
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))
    bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots))
    
    # Filter valid formants (bandwidth < 400 Hz, frequency > 200 Hz)
    valid_formants = []
    for f, bw in zip(freqs, bandwidths):
        if 200 < f < 4000 and bw < 400:
            valid_formants.append(f)
            
    valid_formants.sort()
    return valid_formants

# Test on synthetic central vowel /ɨ/ (F1=340, F2=1450)
t = np.linspace(0, 0.05, 800, endpoint=False)
synthetic_ɨ = np.sin(2 * np.pi * 340 * t) + 0.8 * np.sin(2 * np.pi * 1450 * t)
formants = extract_formants_lpc(synthetic_ɨ)
print(f"Detected Formants: {formants[:2]}")
assert len(formants) >= 2, "Failed to extract formants"
print("SUCCESS: Central vowel formants identified!")
```
