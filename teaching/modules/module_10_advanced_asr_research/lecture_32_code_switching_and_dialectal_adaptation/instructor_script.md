# Instructor Delivery Script: Lecture 32 (Code-Switching and Dialectal Adaptation)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Multilingual Reality: Code-Switching in Everyday Kashmiri
00:20 - 00:45 | Regional Dialects: Maraz (South) vs. Kamraz (North)
00:45 - 01:10 | Low-Rank Adaptation (LoRA): Adapting 317M Models with 1M Weights
01:10 - 01:25 | Code Inspection: configs/adapters/lora_kashmiri.yaml
01:25 - 01:30 | Synthesis & Discussion
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | Dialectal Demonstration
- Pronounce a sentence in standard Srinagar Kashmiri vs. Southern Maraz Kashmiri.
- Address the students:
  > "If you deploy an ASR model in Anantnag, the speakers don't talk like Radio Kashmir announcers in Srinagar.
  > Their vowels are shifted. Their verb suffixes are different.
  > Do we train a whole new 300-million parameter model for South Kashmir?
  > No! That would require another 100 hours of training data and gigabytes of storage on our servers.
  > We freeze the 317M base model and attach a tiny **LoRA adapter** of rank $r=8$.
  > Just 1.2 megabytes of weights!
  > It adapts the acoustic expectations to Southern Kashmiri without erasing a single bit of general phonetic knowledge."
