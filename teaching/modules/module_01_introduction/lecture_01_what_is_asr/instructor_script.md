# Instructor Delivery Script: Lecture 01 (What is Automatic Speech Recognition?)

## Lecture Details
- **Course**: Advanced Kashmiri Speech Recognition & Deep Learning
- **Lecture 01**: What is Automatic Speech Recognition? Mathematical Formulation and the Kashmiri Challenge
- **Duration**: 90 Minutes Total
- **Target Audience**: Graduate Students / AI Research Engineers / Speech Scientists

---

## Timetable & Pedagogical Flow

```
00:00 - 00:15 | Physical Reality: The Physics of Atmospheric Waves to Digital Tensors
00:15 - 00:40 | Mathematical Formulation: Bayes' Decision Rule, Zero-One Loss, and MAP
00:40 - 00:60 | The Paradigm Evolution: Classical Hybrid (Kaldi) vs. Modern E2E (CTC)
00:60 - 00:75 | The Kashmiri Linguistic Crucible: Central Vowels, Palatalization, and Script
00:75 - 00:85 | Forensic Case Study: The Stage 01 Urdu Baseline Disaster (88.4% WER)
00:85 - 00:90 | Summary, Student Misconceptions, and Practical Lab Briefing
```

---

## Blackboard / Presentation Plan

```
+-----------------------------+-----------------------------+-----------------------------+
| LEFT BOARD: ACOUSTICS & ADC | CENTER BOARD: BAYES & MAP   | RIGHT BOARD: KASHMIRI CRUCIBLE
+-----------------------------+-----------------------------+-----------------------------+
| 1. Physical Wave:           | 1. Bayes' Decision Rule:    | 1. Central Vowels:          |
|    p(t) in Pa -> v(t) in V  |    w* = argmin E[L(w, w^)]  |    /i/   /ɨ/   /u/          |
|    Sampling: x[n] = x(n Ts) |                             |    /e/   /ə/   /o/          |
|    Nyquist: fs >= 2 B       | 2. Zero-One Loss Proof:     |          /əː/ (ٲ)           |
|                             |    L(w, w^) = I(w != w^)    |                             |
| 2. The Acoustic Trilemma:   |    R(w^|X) = 1 - P(w^|X)    | 2. Palatalization:          |
|    - Non-uniqueness         |    w* = argmax P(w|X) (MAP) |    /dab/ vs /dab^j/ (دَبؠ)   |
|    - Temporal elasticity    |                             |                             |
|    - Coarticulation         | 3. Bayesian Decomposition:  | 3. Stage 01 Autopsy:        |
|                             |    argmax P(X|w) * P(w)     |    - Urdu Kaldi baseline    |
| 3. Feature Extraction:      |    = log P(X|w) + log P(w)  |    - 38.2% rejected         |
|    x[n] -> X in R^{D x T}   |      (AM)           (LM)    |    - 88.40% WER collapse    |
+-----------------------------+-----------------------------+-----------------------------+
```

---

## Detailed Minute-by-Minute Delivery

### 00:00 - 00:15 | Physical Reality: Waves to Tensors
- **Instructor Action**: Enter room. Silence the room. Tap lightly on the podium microphone.
- **Speaking Script**:
  > "Welcome to Advanced Kashmiri Speech Recognition. Look at that microphone. What is happening right now?
  > My lungs are expelling air. My vocal folds are vibrating at roughly 130 Hz. My tongue, jaw, and lips are shaping that vibration into resonances called formants.
  > That sound moves through the air at 343 meters per second as a pressure fluctuation, hits the diaphragm of that microphone, converts to a microvolt voltage, and is sampled 16,000 times every single second into 16-bit integers.
  > That is physical audio. It is continuous, messy, noisy, and elastic.
  > On the other side of this room, your brain is doing something miraculous: it takes that continuous pressure wave and instantly converts it into discrete words.
  > Our job in this course is to teach a computer to perform that exact same miracle for one of the most linguistically complex languages on planet Earth: Kashmiri.
  > Let us begin with the mathematics."

### 00:15 - 00:40 | Mathematical Formulation: Bayes and MAP Decoding
- **Instructor Action**: Move to Center Board. Write the general decision rule $\hat{\mathbf{w}} = \arg\min_{\hat{\mathbf{w}}} R(\hat{\mathbf{w}} \mid \mathbf{X})$.
- **Speaking Script**:
  > "How do we make optimal decisions under uncertainty? Statistical Decision Theory.
  > We define a loss function $L(\mathbf{w}, \hat{\mathbf{w}})$. If we choose the zero-one loss—where any error in the sentence costs exactly 1, and perfection costs 0—what happens to the conditional risk?
  > Look at the algebra on the board:
  > The sum of probabilities over all incorrect sentences is simply 1 minus the probability of the chosen sentence!
  > Therefore, minimizing error is mathematically identical to maximizing the posterior probability $P(\mathbf{w} \mid \mathbf{X})$.
  > That is Maximum A Posteriori (MAP) decoding."
- **Whiteboard Derivation**:
  - Write: $P(\mathbf{w} \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid \mathbf{w}) P(\mathbf{w})}{P(\mathbf{X})}$.
  - Cross out $P(\mathbf{X})$ with chalk:
  > "Why can we cross out $P(\mathbf{X})$? Because $P(\mathbf{X})$ is the marginal evidence. It is the probability of the sound itself across all possible sentences. It does not depend on which hypothesis $\mathbf{w}$ we are testing! So for argmax, it is a constant scaling factor.
  > And what are we left with? The two pillars of classical speech recognition:
  > The Acoustic Model $P(\mathbf{X} \mid \mathbf{w})$, and the Language Model $P(\mathbf{w})$."

### 00:40 - 00:60 | Paradigm Evolution: Why Hybrid Systems Died
- **Instructor Action**: Compare Kaldi HMM-GMM with modern End-to-End CTC.
- **Speaking Script**:
  > "For 35 years—from 1980 to 2015—the entire speech recognition industry built systems based on this generative decomposition.
  > But it had a fatal flaw for languages like Kashmiri.
  > To connect acoustic features to words, classical systems needed a **Pronunciation Lexicon**. An explicit human-written dictionary mapping every word to phonemes.
  > In English, you have CMUdict: 134,000 words phonetically transcribed by linguists over decades.
  > In Kashmiri, we had no such dictionary.
  > And worse: classical systems assumed phonemes transition through hidden Markov states that must be forced-aligned frame-by-frame.
  > Modern End-to-End systems threw away the lexicon. They threw away the HMM states.
  > Using Connectionist Temporal Classification (CTC) over self-supervised Transformer backbones, we map raw audio directly to Kashmiri Perso-Arabic characters in a single differentiable neural network."

### 00:60 - 00:75 | The Kashmiri Linguistic Crucible
- **Instructor Action**: Move to Right Board. Draw the Kashmiri Vowel Trapezoid.
- **Speaking Script**:
  > "Now, why couldn't we just take an English or Urdu speech recognition model and fine-tune it naively?
  > Look at the center of this vowel trapezoid.
  > English has front vowels like /i/ in 'beet' and back vowels like /u/ in 'boot'.
  > Urdu has front and back vowels.
  > Kashmiri has an entire family of **central unrounded vowels**: /ɨ/, /ɨː/, /ə/, and /əː/.
  > When a Kashmiri speaker says 'كٲشُر' (/kəːʃur/), the vowel /əː/ has a first formant of 580 Hz and a second formant of 1550 Hz.
  > In Urdu, that sound does not exist!
  > If your acoustic model hasn't learned to resolve the 1550 Hz region, it will confuse /kəːʃur/ with /kaːʃur/ or /koːʃur/.
  > Add to that phonemic palatalization—where adding `ؠ` to a consonant changes a verb from singular to plural—and diacritic-less writing, and you have one of the most challenging speech recognition problems in computational linguistics."

### 00:75 - 00:85 | The Stage 01 Urdu Kaldi Autopsy
- **Instructor Action**: Display terminal log excerpt on screen.
- **Speaking Script**:
  > "Let's look at real repository history. In Stage 01 of our project, the team tested an Urdu Kaldi model on Kashmiri recordings.
  > The result was an unmitigated disaster: **88.40% Word Error Rate**.
  > Why?
  > Look at line 14 of the log: `WARNING: Fst has no paths`.
  > Because the Urdu lexicon did not have Kashmiri central vowels, Kaldi's alignment graph had zero valid paths.
  > Kaldi threw away **38.2% of the audio files** before training even started!
  > This is our baseline. This is where EduCasheer started.
  > Over the next 34 lectures, we will see how every single percentage point of error was methodically dismantled from 88.4% down to 24.12%."

### 00:85 - 00:90 | Conclusion & Lab Orientation
- **Speaking Script**:
  > "In today's practical lab, you will synthesize synthetic vowel waveforms, simulate 16-bit ADC quantization, and implement a toy MAP vs MBR decoder.
  > Open `practical_lab.md` in your lecture directory. See you in the lab."
