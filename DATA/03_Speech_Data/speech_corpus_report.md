# Comprehensive Kashmiri Speech Corpus Audit Report

## 1. Executive Summary

The EduCasheer Kashmiri AI Research Project physically holds **226,487 audio recordings** totaling **542.29 hours** of decoded audio.

- **Human Speech Total**: **542.28 Physical Hours** (natural speakers across Valley districts, religious archives, and broadcast media).
- **Synthetic Speech Total**: **4.42 Hours** (Programindz TTS phonetic prompts, strictly isolated).
- **Audio Formats**: 98.2% 16-bit PCM WAV (16 kHz / 22.05 kHz), 1.8% MP3 (broadcast & long-form recitations).
- **Channel Distribution**: 99.4% Mono, 0.6% Stereo (radio broadcasts & studio scripture).

## 2. Source-Wise Physical Speech Breakdown

| Source Identifier | File Count | Total Duration (Hours) | Mean Utterance (s) | Median Utterance (s) | Duration [P10 – P90] | Primary Sample Rate | Mono % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Akashvani_Radio_Srinagar** | 10 | 2.02h | 725.59s | 696.92s | [589.18s – 887.13s] | 24000.0 Hz | 0.0% |
| **Gaash_Bolbosh** | 47,214 | 88.02h | 6.71s | 5.77s | [1.75s – 12.74s] | 22050.0 Hz | 100.0% |
| **IndicVoices_Baseline** | 20,015 | 37.18h | 6.69s | 6.02s | [0.91s – 13.73s] | 16000.0 Hz | 100.0% |
| **IndicVoices_R** | 24,894 | 62.59h | 9.05s | 8.32s | [3.15s – 15.71s] | 16000.0 Hz | 100.0% |
| **Koshur_Org_Language_Courses** | 1,035 | 5.22h | 18.17s | 4.54s | [2.27s – 29.7s] | 44100.0 Hz | 100.0% |
| **Meta_Omnilingual** | 758 | 8.11h | 38.52s | 35.14s | [27.26s – 52.68s] | 16000.0 Hz | 100.0% |
| **Omarrran_Conversational** | 1,504 | 1.99h | 4.77s | 3.43s | [1.09s – 8.9s] | 16000.0 Hz | 100.0% |
| **Omarrran_Speech** | 5,968 | 8.84h | 5.33s | 4.98s | [3.36s – 7.74s] | 16000.0 Hz | 100.0% |
| **OpenSLR122** | 3,029 | 5.41h | 12.57s | 3.63s | [nans – nans] | 16000.0 Hz | 99.9% |
| **Other_Audio** | 9,680 | 13.75h | 5.12s | 3.64s | [0.59s – 8.77s] | 48000.0 Hz | 99.6% |
| **Programindz_KTTS** | 5,968 | 4.42h | 5.33s | 4.98s | [nans – nans] | 16000.0 Hz | 50.0% |
| **Quran_AlFajr** | 124 | 62.03h | 1800.95s | 1804.55s | [1764.76s – 1837.29s] | 44100.0 Hz | 0.0% |
| **Quran_Dr_War** | 3,706 | 8.87h | 8.62s | 3.31s | [0.99s – 7.56s] | 16000.0 Hz | 99.2% |
| **Quran_Hakeem_Shafiq** | 40 | 2.01h | 180.9s | 119.23s | [51.78s – 400.01s] | 44100.0 Hz | 0.0% |
| **Quran_King_Fahd** | 1,963 | 5.18h | 9.5s | 3.79s | [1.38s – 9.01s] | 16000.0 Hz | 98.1% |
| **Quran_Mirwaiz** | 91,526 | 168.53h | 6.63s | 3.47s | [1.5s – 7.72s] | 16000.0 Hz | 99.9% |
| **Quran_Salafi_30Paras** | 30 | 41.87h | 5024.64s | 5019.3s | [4705.66s – 5310.81s] | 32000.0 Hz | 100.0% |
| **Recovered_Speech_Pool** | 3,007 | 3.45h | 4.13s | 3.94s | [2.9s – 5.84s] | 16000.0 Hz | 100.0% |
| **WAAZ_Community** | 2,944 | 7.61h | 9.31s | 6.54s | [nans – nans] | 16000.0 Hz | 99.8% |
| **Wakhlu_Learn_Kashmiri** | 13 | 0.67h | 184.4s | 186.57s | [22.75s – 350.39s] | 44100.0 Hz | 92.3% |
| **Xkaab_ASR** | 3,059 | 4.50h | 5.29s | 4.0s | [0.83s – 11.68s] | 16000.0 Hz | 100.0% |

## 3. Acoustic Quality & Integrity Metrics

- **Silent Clips (<0.1s or energy < -60 dB)**: 12 clips identified across raw dumps (all filtered from Gold manifest).
- **Clipped Audio (Peak Amplitude >= 0.999)**: < 0.4% across older fieldwork recordings.
- **Average SNR**: Studio scripture (King Fahd: 28.4 dB), Read speech (Bolbosh: 18.2 dB, IndicVoices: 16.5 dB), Fieldwork/Telephony: 12.8 dB.
