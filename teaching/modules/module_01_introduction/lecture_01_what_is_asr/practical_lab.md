# Practical Lab: Audio Physics, Formant Synthesis, and Decision Theory

---

## Lab Overview
In this laboratory session, students will implement pure Python and PyTorch simulations to explore:
1. Physical synthesis of Kashmiri central vowels using formant resonators.
2. Analog-to-Digital Conversion (ADC) quantization noise and Signal-to-Quantization-Noise Ratio (SQNR).
3. A toy Maximum A Posteriori (MAP) vs. Minimum Bayes Risk (MBR) sequence decoder.

---

## Exercise 1: Synthesizing the Kashmiri Central Vowel /əː/ (`lab_formant_synthesis.py`)

```python
import numpy as np
import scipy.signal
import soundfile as sf

def synthesize_kashmiri_vowel(f0=130.0, f1=580.0, f2=1550.0, f3=2450.0, duration=1.0, sr=16000):
    # Time vector
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Glottal source: periodic pulse train filtered with -12 dB/octave tilt
    # Generate impulses at fundamental frequency f0
    period_samples = int(sr / f0)
    source = np.zeros_like(t)
    source[::period_samples] = 1.0
    
    # Glottal pulse shaping filter
    b_glot, a_glot = scipy.signal.butter(1, f0 * 2 / sr, btype='low')
    glottal_excitation = scipy.signal.lfilter(b_glot, a_glot, source)
    
    # 2. Resonant Formant Filters (F1, F2, F3)
    # Bandwidths for human formants
    bw1, bw2, bw3 = 80.0, 110.0, 140.0
    
    def make_resonator(freq, bw):
        r = np.exp(-np.pi * bw / sr)
        omega = 2 * np.pi * freq / sr
        a = [1.0, -2 * r * np.cos(omega), r ** 2]
        b = [1.0 - 2 * r * np.cos(omega) + r ** 2]
        return b, a
        
    b1, a1 = make_resonator(f1, bw1)
    b2, a2 = make_resonator(f2, bw2)
    b3, a3 = make_resonator(f3, bw3)
    
    # Cascade resonators
    y1 = scipy.signal.lfilter(b1, a1, glottal_excitation)
    y2 = scipy.signal.lfilter(b2, a2, y1)
    y3 = scipy.signal.lfilter(b3, a3, y2)
    
    # 3. Radiation impedance at lips (+6 dB/octave high-pass)
    b_rad, a_rad = [1.0, -0.97], [1.0]
    speech_wave = scipy.signal.lfilter(b_rad, a_rad, y3)
    
    # Peak normalize
    speech_wave = speech_wave / np.max(np.abs(speech_wave))
    return speech_wave

if __name__ == "__main__":
    vowel = synthesize_kashmiri_vowel()
    print(f"Synthesized Kashmiri central vowel /əː/: {len(vowel)} samples at 16 kHz")
    assert len(vowel) == 16000, "Vowel length mismatch!"
    print("SUCCESS: Formant synthesis verified!")
```

---

## Exercise 2: Simulating ADC Quantization and SQNR (`lab_quantization.py`)

```python
import numpy as np

def compute_sqnr(signal, bits):
    # Normalize signal to [-1.0, 1.0]
    sig_norm = signal / np.max(np.abs(signal))
    
    # Quantize to 2^bits levels
    levels = 2 ** bits
    delta = 2.0 / levels
    quantized = np.round(sig_norm / delta) * delta
    quantized = np.clip(quantized, -1.0, 1.0)
    
    # Noise = error
    noise = sig_norm - quantized
    
    p_signal = np.mean(sig_norm ** 2)
    p_noise = np.mean(noise ** 2)
    
    sqnr_db = 10 * np.log10(p_signal / max(p_noise, 1e-12))
    return sqnr_db

if __name__ == "__main__":
    # Test on pure sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = np.sin(2 * np.pi * 440.0 * t)
    
    for b in [8, 12, 16]:
        empirical_sqnr = compute_sqnr(sine, b)
        theoretical_sqnr = 6.02 * b + 1.76
        print(f"Bits: {b:2d} | Empirical: {empirical_sqnr:6.2f} dB | Theoretical: {theoretical_sqnr:6.2f} dB")
        assert abs(empirical_sqnr - theoretical_sqnr) < 1.0, "SQNR discrepancy!"
    print("SUCCESS: Quantization noise model verified!")
```

---

## Exercise 3: Toy MAP vs. MBR Decoder (`lab_map_vs_mbr.py`)

```python
from collections import defaultdict

def toy_mbr_decode():
    # 3 hypotheses with posterior probabilities
    candidates = [
        ("کٲشُر زَبان", 0.45),
        ("کٲشُر زَبانہٕ", 0.35),
        ("ہِندوستان", 0.20)
    ]
    
    # Levenshtein word distance
    def word_edit_distance(h1, h2):
        w1, w2 = h1.split(), h2.split()
        # Simple word symmetric difference
        diff = len(set(w1).symmetric_difference(set(w2)))
        return diff
        
    # 1. MAP Decode (argmax posterior)
    map_winner = max(candidates, key=lambda x: x[1])
    print(f"MAP Selected Hypothesis: '{map_winner[0]}' (Prob = {map_winner[1]:.2f})")
    
    # 2. MBR Decode (argmin expected risk)
    risks = {}
    for hyp_i, _ in candidates:
        expected_risk = 0.0
        for hyp_j, prob_j in candidates:
            dist = word_edit_distance(hyp_i, hyp_j)
            expected_risk += dist * prob_j
        risks[hyp_i] = expected_risk
        print(f"Hypothesis '{hyp_i:15s}' -> Expected Word Risk: {expected_risk:.3f}")
        
    mbr_winner = min(risks.items(), key=lambda x: x[1])
    print(f"MBR Selected Hypothesis: '{mbr_winner[0]}' (Risk = {mbr_winner[1]:.3f})")

if __name__ == "__main__":
    toy_mbr_decode()
```
