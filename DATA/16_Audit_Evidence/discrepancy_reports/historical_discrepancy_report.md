# Forensic Audit Discrepancy Report against Historical Documentation

| # | Previously Reported Statistic | Newly Verified Ground Truth | Source / Context | Root Cause & Resolution |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Omarrran TTS & Programindz reported as 8.84h of independent speech | 4.42h unique audio | Manifests `omarrran_tts` vs `programindz_tts` | Exact acoustic duplicate detected via SHA-256; resolved by retaining Programindz as canonical synthetic tier. |
| 2 | IndicVoices training split had 21 speaker overlap with test set | 0 (Zero) speaker overlap | `asr_training_eligible_manifest.jsonl` | 1,550 clips from 21 test speakers were present in earlier eligible training split; strictly filtered in `manifest_gold.jsonl`. |
| 3 | Quran King Fahd reported as 100% verified Kashmiri training speech | 2.39h total, with Arabic leaks in early raw audio | `quran_king_fahd/` | Separated Arabic Surah recitation leaks into quarantine; retained clean Kashmiri translations. |
| 4 | WAAZ Platform reported ~4 hours in training | 0.88h approved Gold, 3.80h in unreviewed queue | `datacollect.db` | Only 323 approved recordings were valid for Gold; 1,569 recordings awaiting human validation were segregated into Bronze. |
