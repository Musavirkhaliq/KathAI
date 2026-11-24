# Synthetic Speech Data Forensic Audit

- **Programindz TTS Corpus**: 2,984 clips (4.42 hours), synthesized via phonetic text-to-speech.
- **Deconfliction Findings**: In early experiments, `manifest_omarrran_tts.jsonl` and `manifest_programindz_tts.jsonl` duplicated these identical 2,984 audio files. In this audit, duplicate entries are removed, and Programindz is quarantined into `manifest_synthetic.jsonl`.
- **Scientific Isolation**: Zero synthetic hours are counted toward natural human speech totals.
