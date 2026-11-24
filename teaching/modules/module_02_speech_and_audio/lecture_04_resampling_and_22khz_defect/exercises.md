# Exercises: Lecture 04 (Resampling Algorithms and the 22.05 kHz Defect)

---

## Exercise 1: Polyphase Ratio Simplification
1. Reduce the following sampling rate conversion ratios to coprime integers $L$ and $M$:
   - $44,100 \text{ Hz} \to 16,000 \text{ Hz}$
   - $48,000 \text{ Hz} \to 16,000 \text{ Hz}$
   - $22,050 \text{ Hz} \to 16,000 \text{ Hz}$
   - $8,000 \text{ Hz} \to 16,000 \text{ Hz}$
2. Explain why converting from $48 \text{ kHz}$ to $16 \text{ kHz}$ is computationally far cheaper than converting from $22.05 \text{ kHz}$ to $16 \text{ kHz}$.

---

## Exercise 2: Kaiser Parameter Calculation
You are designing an anti-aliasing low-pass filter for an ASR pipeline.
The client demands:
- Stopband attenuation $A \ge 85 \text{ dB}$
- Normalized transition band $\Delta \omega = 0.05 \pi$ radians/sample
1. Calculate the required Kaiser parameter $\beta$.
2. Calculate the estimated FIR filter length $N$.
3. What is the filter latency in milliseconds if the sampling rate is $16 \text{ kHz}$?

---

## Exercise 3: Forensic Audio Error Analysis
In an uncorrected evaluation on the Bolbosh test set, an acoustic model achieved the following confusion matrix entries:
- Target `/t͡s/` transcribed as `/t/`: 184 occurrences
- Target `/ʃ/` transcribed as `/s/`: 142 occurrences
- Target `/m/` transcribed as `/b/`: 11 occurrences

Explain why fricative/affricate confusions were an order of magnitude higher than nasal/stop confusions under the defective resampler.
