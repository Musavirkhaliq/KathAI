# Exercises: Lecture 03 (Sound Waves and Digital Sampling)

---

## Exercise 1: Formant Extraction and Source-Filter Separation
A Kashmiri female speaker produces the central vowel /ɨ/ with fundamental frequency $F_0 = 220 \text{ Hz}$, first formant $F_1 = 340 \text{ Hz}$, and second formant $F_2 = 1450 \text{ Hz}$.
1. What is the fundamental period $T_0$ of the glottal cycle in milliseconds?
2. Which harmonics of the glottal source are closest to the first formant $F_1$?
3. If the audio is low-pass filtered at 1,000 Hz, can an ASR acoustic model still differentiate /ɨ/ from /u/ ($F_1 \approx 320 \text{ Hz}, F_2 \approx 800 \text{ Hz}$)? Justify mathematically.

---

## Exercise 2: Mathematical Aliasing Folds
A microphone records acoustic noise containing frequencies at 4.5 kHz, 8.2 kHz, 13.5 kHz, and 21.0 kHz.
The analog signal is digitized directly at $f_s = 16.0 \text{ kHz}$ without an analog anti-aliasing filter.
Calculate the exact apparent digital frequencies observed in the resulting baseband spectrum $[0, 8.0 \text{ kHz}]$ for each component.

---

## Exercise 3: Dynamic Range and Headroom Calculation
An ASR recording setup records in a classroom where the ambient background noise floor is $38 \text{ dB SPL}$, and the peak speech volume reaches $92 \text{ dB SPL}$.
1. What is the required dynamic range of the microphone pre-amplifier?
2. What is the minimum bit depth $B$ needed so that the quantization noise floor remains at least $15 \text{ dB}$ below the room noise floor?
