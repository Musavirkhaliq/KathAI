# Instructor Delivery Script: Lecture 06 (Kashmiri Phonology and Orthography)

## Timetable (90 Minutes)
```
00:00 - 00:15 | Language Context: Dardic Roots vs. Indo-Aryan Mainstream
00:15 - 00:45 | The 3D Vowel Space: The Central Vowel Matrix (/ɨ/, /ə/, /əː/)
00:45 - 01:10 | Palatalization Mechanics and Acoustic Formant Transitions
01:10 - 01:25 | Perso-Arabic Orthography and the Diacritic Crisis
01:25 - 01:30 | Synthesis & Transition to Text Normalization
```

---

## Instructor Notes & Dialogue

### 00:15 - 00:45 | The Central Vowel Matrix
- Step to the board and draw the **Acoustic Vowel Triangle**:
  - Top-left: Front /i/ ($F_2 = 2200 \text{ Hz}$)
  - Top-right: Back /u/ ($F_2 = 800 \text{ Hz}$)
  - Bottom: Low /a/ ($F_1 = 800 \text{ Hz}$)
- Now draw a vertical pillar right down the center:
  > "Every standard Indo-Aryan language leaves this center space empty. Hindi speakers don't have phonemic /ɨ/. Urdu speakers don't have /əː/.
  > Kashmiri lives in this center space.
  > If a model is pretrained on Hindi, Urdu, or English, its acoustic latent space has never partitioned this intermediate F2 zone between 1400 and 1600 Hz.
  > When a Kashmiri speaker says 'كٲشُر' (/kəːʃur/), the first vowel is /əː/.
  > If your acoustic model fails to resolve 1550 Hz F2, it will predict either 'کاشُر' (/kaːʃur/) or 'کوشُر' (/koːʃur/). Both are errors.
  > This is why cross-lingual transfer without fine-tuning fails completely."

### 00:45 - 01:10 | Palatalization Demo
- Pronounce the contrast: `/dab/` vs `/dabʲ/`.
- Show how the tongue blade touches the hard palate during the release phase.
- On the spectrogram projection, point out the sharp rising $F_2$ trajectory leading into the consonant release.
