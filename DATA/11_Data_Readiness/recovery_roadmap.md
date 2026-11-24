# Kashmiri Data Recovery & Pipeline Strategy Roadmap

## 1. Executive Summary

Our audit identifies **178.08 hours** of high-potential recoverable Kashmiri human speech already in our physical possession, requiring straightforward pipeline processing to convert into training-ready assets.

## 2. Immediate & Medium-Term Recovery Priorities

1. **Priority 1: WAAZ Review Queue Promotion (3.80 Hours)**
   - Action: Run validation interface on 1,559 pending contributions queued in `HUMAN_VALIDATION_QUEUE.jsonl`.
   - Expected Yield: ~3.5 hours of high-diversity spontaneous mobile audio promoted into Gold.

2. **Priority 2: Al Fajr Trust VAD Segmentation (62.03 Hours)**
   - Action: Execute Silero VAD segmentation and Surah verse alignment on the 124 continuous audio files.
   - Expected Yield: ~55 hours of verse-level scripture speech promoted into Gold/Silver.

3. **Priority 3: Molvi Salafi Para-wise Quran VAD Segmentation (41.87 Hours)**
   - Action: Segment 30 long-form Paras (~80m each) into 3–10s speech utterances for SSL pretraining.
   - Expected Yield: ~38 hours of classical Kashmiri speech representation.

4. **Priority 4: Mirwaiz Pure Commentary Transcription (23.25 Hours)**
   - Action: Pseudo-label the 17,724 pure commentary clips with our Conformer ASR model followed by human verification.
   - Expected Yield: ~23 hours of spontaneous religious speech.
