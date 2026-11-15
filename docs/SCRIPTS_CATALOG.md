# Comprehensive Scripts Catalog
**Project**: EduCasheer Kashmiri Automatic Speech Recognition
**Total Scripts Indexed**: 154

---
## Acoustic Model Training & Continual Adaptation (46 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `analyze_and_inventory_audio.py` | Audio Inventory and Metadata Extraction for Kashmiri Quran Archive |
| `benchmark_epoch_throughput.py` | Standalone research utility script. |
| `char_seq2seq_adapter.py` | Standalone research utility script. |
| `classify_text_corpus.py` | Standalone research utility script. |
| `consolidate_stage9_results.py` | Stage 9: Results Consolidation, Statistical Significance, and Report Generator |
| `create_splits.py` | Research-Grade Dataset Splitter with Strict Speaker Isolation |
| `create_stratified_pilot.py` | Create a stratified development pilot manifest of 150 samples across all 4 sources. |
| `detect_duplicates.py` | Duplicate & Near-Duplicate Detection for Audio and Transcripts |
| `discover_external_asr_models.py` | Workstream 9: External Kashmiri ASR Model Discovery and Inspection |
| `discover_external_datasets.py` | External Kashmiri Dataset Cataloger & Registry |
| `extract_bolbosh_and_tts.py` | Standalone research utility script. |
| `extract_indicvoices_and_whisper.py` | Standalone research utility script. |
| `fast_train_eval_adapter.py` | Standalone research utility script. |
| `generate_architecture_comparison_csv.py` | Standalone research utility script. |
| `generate_b01_error_analysis.py` | Standalone research utility script. |
| `normalize_transcripts.py` | Kashmiri Transcript Normalization & Script Harmonization Module |
| `pretraining_validation_gates.py` | Standalone research utility script. |
| `record_and_transcribe_cli.py` | Standalone research utility script. |
| `record_now.py` | Standalone research utility script. |
| `review_cli.py` | Interactive CLI Review Interface for Kashmiri Quran Audio Review Package |
| `run_full_pipeline.py` | Master Execution Pipeline for Kashmiri Speech & Multimodal Dataset |
| `run_kashmiri_benchmark.py` | EduCasheer Kashmiri ASR Benchmark Evaluation Engine |
| `run_lr_pilot_experiment.py` | Standalone research utility script. |
| `run_phase_i_statistical_analysis.py` | Workstream 8 Stage 7 Phase I: Statistical Significance Testing & Detailed Evaluation |
| `run_stage9_full_pipeline.py` | Workstream 8 Stage 9: Master Acoustic Training & Benchmark Pipeline |
| `run_verse_anchor_statistical_experiment.py` | Statistical Evaluation of Verse-Anchored Forced Alignment |
| `run_whisper_benchmark.py` | Standalone research utility script. |
| `test_bucketed_speed.py` | Standalone research utility script. |
| `test_convergence_steps.py` | Standalone research utility script. |
| `test_ctc_zero_infinity_fix.py` | Standalone research utility script. |
| `test_deep_overfit.py` | Standalone research utility script. |
| `test_onnx_indicconformer.py` | Standalone research utility script. |
| `test_overfit_single_utterance.py` | Standalone research utility script. |
| `train_and_evaluate_tokenizers.py` | Standalone research utility script. |
| `train_b01_full_schedule.py` | Standalone research utility script. |
| `train_b01_xlsr.py` | Standalone research utility script. |
| `train_byt5_transliteration_adapter.py` | Standalone research utility script. |
| `train_eval_neural_adapter.py` | Standalone research utility script. |
| `train_fast_char_lstm.py` | Standalone research utility script. |
| `train_kashmiri_language_models.py` | Workstream 8 Stage 7 Phase E: Train Kashmiri N-Gram Language Models |
| `train_neural_romanizer.py` | Standalone research utility script. |
| `train_stage10_continual_adaptation.py` | EduCasheer Kashmiri AI — Stage 10: Continual Acoustic Adaptation Engine |
| `train_stage9_acoustic_models.py` | Stage 9: Controlled Acoustic Training Engine (B-02A, B-02B, B-02C) |
| `train_stage9_full_schedule.py` | Workstream 8 Stage 9: Full Multi-Epoch Controlled Acoustic Training Engine |
| `transcribe_audio.py` | Standalone research utility script. |
| `validate_audio.py` | Audio Validation, Conversion & Segmentation Pipeline for Kashmiri ASR |

## Evaluation & Benchmark Runners (18 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `audit_and_build_eval_splits.py` | Standalone research utility script. |
| `evaluate_full_validation_b01.py` | Standalone research utility script. |
| `evaluate_full_validation_indicconformer.py` | Standalone research utility script. |
| `evaluate_indicconformer_baseline.py` | Standalone research utility script. |
| `evaluate_rule_romanizer.py` | Standalone research utility script. |
| `evaluate_rule_transliteration.py` | Standalone research utility script. |
| `evaluate_stage9_decoders.py` | Stage 9: Unified Multi-Decoder Evaluation Engine |
| `evaluate_test_b01.py` | Standalone research utility script. |
| `evaluate_test_bootstrap.py` | Standalone research utility script. |
| `evaluate_test_indicconformer.py` | Standalone research utility script. |
| `generate_human_eval_ui.py` | Standalone research utility script. |
| `human_eval_server.py` | Standalone research utility script. |
| `linguistic_orthography_evaluator.py` | Linguistic Orthography Evaluator for Kashmiri Perso-Arabic ASR. |
| `metrics_evaluator.py` | Standalone research utility script. |
| `paired_bootstrap_comparison.py` | Standalone research utility script. |
| `paired_bootstrap_test.py` | Standalone research utility script. |
| `run_b02a_followup_evaluation.py` | EduCasheer Kashmiri ASR: B-02A Follow-Up Held-Out Test Evaluation Engine |
| `run_sota_pilot_eval.py` | Standalone research utility script. |

## Decoding, Beam Search & Language Modeling (3 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `audit_and_clean_lm_text.py` | Workstream 8 Stage 7 Phase D: Kashmiri Text Corpus Audit, Cleaning & Harmonization |
| `run_ctc_beam_search_experiments.py` | Workstream 8 Stage 7 Phase C: CTC Prefix Beam Search Baseline (No LM) |
| `run_lm_shallow_fusion_experiments.py` | Workstream 8 Stage 7 Phase F: CTC Prefix Beam Search + Kashmiri Language Model Shallow Fusion |

## Data Preprocessing, Audio Alignment & Slicing (16 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `apply_canonical_sentence_alignment_whole_corpus.py` | Apply Canonical Sentence Alignment (Approach B) Across the Entire 114 Surah Kashmiri Quran Archive |
| `download_all_remaining.py` | Standalone research utility script. |
| `download_all_unlocked_datasets.py` | Downloads all newly unlocked / missing Kashmiri datasets across Modalities: |
| `download_indic_conformer_tensors.py` | Download external ONNX tensor files for IndicConformer-600M |
| `download_indic_models.py` | Workstream 9: Targeted Download Engine for IndicConformer-600M and Indic-Transcribe-core |
| `download_new_undownloaded.py` | Downloads ONLY genuinely missing, un-downloaded Kashmiri datasets: |
| `download_unlocked_gated.py` | Downloads all newly unlocked gated Kashmiri datasets using authenticated user token. |
| `execute_arabic_boundary_trimming.py` | Standalone research utility script. |
| `filter_all_arabic_recitations.py` | Standalone research utility script. |
| `process_corpus_advanced.py` | Advanced Kashmiri Quran Corpus Processor, Splitter & Measured Inventory |
| `process_full_canonical_kashmiri_corpus.py` | Full-Scale Processing of the Canonical Kashmiri Qur'an Corpus |
| `process_wikipedia_dump.py` | Standalone research utility script. |
| `recover_excluded_speech_corpus.py` | Recover Excluded Kashmiri Speech Corpus (Workstream 8, Tasks 2 & 3) |
| `refine_text_and_remove_arabic.py` | Standalone research utility script. |
| `reslice_all_audio_clips.py` | Systematic Corpus Audio Slicing Synchronizer |
| `test_surgical_arabic_trimmer.py` | Standalone research utility script. |

## Qur anic Dataset Curation & Audio-Text Sync (10 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `audit_all_surahs_coverage.py` | Audio Coverage Audit across all 114 Surahs |
| `download_quran_audio.py` | Download Complete Kashmiri Quran Audio Archive from Internet Archive |
| `export_quran_text.py` | Export Complete Kashmiri Quran Translations |
| `process_all_114_surahs.py` | Complete Kashmiri Quran Corpus Processing Engine (All 114 Surahs) |
| `process_quran_audio_and_align.py` | Kashmiri Quran Audio Segmentation, Language Classification, Alignment & QC Pipeline |
| `process_quran_dr_war.py` | Dr. Abdul Rehman War Kashmiri Quran Commentary Processing Engine |
| `process_quran_king_fahd.py` | King Fahd Complex Kashmiri Quran Processing Engine |
| `quran_review_interface.py` | Kashmiri Quran Audio Human-Review Workflow & Verification Engine |
| `regenerate_pure_kashmiri_quran_corpus.py` | Regenerate Pure Kashmiri Qur'an Corpus (Option B) |
| `test_forward_backward.py` | Standalone research utility script. |

## Romanization, Transliteration & Lexicon (6 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `build_translit_lexicon.py` | Standalone research utility script. |
| `demo_end_to_end_asr_roman.py` | Standalone research utility script. |
| `kashmiri_rule_romanizer.py` | Standalone research utility script. |
| `kashmiri_rule_transliteration.py` | Standalone research utility script. |
| `kashmiri_transliteration_engine.py` | Comprehensive Kashmiri Transliteration Engine (Perso-Arabic -> Romanized Kashmiri) |
| `prepare_transliteration_splits.py` | Standalone research utility script. |

## Dataset Acquisition, Harvesting & Quality Control (8 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `dataset_acquisition/acquire_new_sources.py` | Workstream 9: Audio Acquisition & Standardization Pipeline |
| `dataset_acquisition/audit_baseline_inventory.py` | Workstream 9: Baseline Speech Inventory Audit & Reconciliation |
| `dataset_acquisition/build_final_training_manifest.py` | Workstream 9: Final Training-Eligible Manifest Generator |
| `dataset_acquisition/discover_and_audit_licenses.py` | Workstream 9: Discovery Registry and License Audit Generator |
| `dataset_acquisition/integrate_indicvoices_r.py` | Workstream 9: Incremental IndicVoices-R Ingestion & Integration Module |
| `dataset_acquisition/run_audio_and_language_qa.py` | Workstream 9: Audio Quality Control & Kashmiri Language Identification Pipeline |
| `dataset_acquisition/run_multi_level_deduplication.py` | Workstream 9: Multi-Level Deduplication, Overlap Matrix & Benchmark Protection |
| `dataset_acquisition/stream_and_ingest_indicvoices_r.py` | Workstream 9: Stream & Ingest IndicVoices-R Shards |

## Auditing, Statistical Analysis & Reporting (15 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `audit_and_reconcile_existing.py` | Corpus Output Audit & Candidate Alignment Reconciler |
| `audit_benchmark_candidate_pools.py` | Comprehensive Audit of Kashmiri ASR Benchmark Candidate Pools |
| `audit_benchmark_contamination.py` | Workstream 8: Formal Benchmark Contamination Audit |
| `audit_dataset.py` | Comprehensive Raw Dataset Auditor for Kashmiri Language Project |
| `audit_master_inventory.py` | Standalone research utility script. |
| `audit_speech_corpus_quality.py` | Workstream 8 Stage 7 Phase H: Speech Corpus Quality Audit & Excluded Utterance Recovery |
| `dataset_statistics.py` | Comprehensive Dataset Quantification & Metric Reporting |
| `deep_data_audit.py` | Standalone research utility script. |
| `generate_b01_audit_data.py` | Standalone research utility script. |
| `generate_experiment_report_and_reproduce.py` | Standalone research utility script. |
| `generate_forensic_report.py` | Standalone research utility script. |
| `generate_stage5_audit_and_report.py` | Standalone research utility script. |
| `generate_stage5_full_reports.py` | Standalone research utility script. |
| `quality_filter.py` | Audio Quality Assessment & SNR Estimation Module for Kashmiri ASR |
| `update_b01_experiment_report.py` | Standalone research utility script. |

## Interactive Demos, Web UIs & Inference Tools (32 scripts)

| Script Name | Functional Description |
| :--- | :--- |
| `asr_web_app.py` | Standalone research utility script. |
| `build_150_sample_pilot.py` | Workstream 8: Stratified Selection and Freezing of the 150-Sample Pilot Benchmark |
| `build_audio_acquisition_inventory.py` | Standalone research utility script. |
| `build_audio_review_package.py` | Audio Review Package & Transcription Correctness Audit |
| `build_b01_final_report.py` | Standalone research utility script. |
| `build_baseline_comparison_and_error_report.py` | Standalone research utility script. |
| `build_char6.py` | Standalone research utility script. |
| `build_consolidated_modalities.py` | Standalone research utility script. |
| `build_deliverables.py` | Standalone research utility script. |
| `build_full_benchmark_manifests.py` | Workstream 8: Assembly and Sealing of the Complete EduCasheer Kashmiri ASR Benchmark |
| `build_full_comparative_report.py` | Standalone research utility script. |
| `build_full_lexicon_cache.py` | Standalone research utility script. |
| `build_hf_ctc_tokenizer.py` | Standalone research utility script. |
| `build_html_ui.py` | Standalone research utility script. |
| `build_kashmiri_ctc_tokenizer.py` | Standalone research utility script. |
| `build_manifest.py` | Master Manifest Builder for Kashmiri Speech & ASR Dataset |
| `build_paper_evidence_and_summary.py` | Standalone research utility script. |
| `build_provenance_registry.py` | Standalone research utility script. |
| `build_pure_kashmiri_widget.py` | Builds an interactive review widget containing 100% PURE KASHMIRI audio clips (no Arabic recitation). |
| `build_recording_inventory.py` | Recording-Level Language & Content Inventory for Kashmiri Quran Audio |
| `build_review_widget.py` | Generates an interactive HTML review widget for the Kashmiri Quran review package. |
| `build_speaker_independent_splits.py` | Standalone research utility script. |
| `build_stage8_expansion_artifacts.py` | Stage 8: Comprehensive Speech Corpus Expansion Artifacts Builder |
| `build_statistical_random_audit.py` | Updates statistical_random_audit.html with trimmed audio clips where any Arabic recitation |
| `build_updated_master_manifest.py` | Standalone research utility script. |
| `build_validation_logit_cache.py` | Standalone research utility script. |
| `build_verbatim_comparison_widget.py` | Generates an interactive before-and-after comparison widget for verbatim text alignment. |
| `build_verified_pure_widget.py` | Builds an embedded audio player for the 7 verified pure Kashmiri clips. |
| `build_written_text_inventory.py` | Standalone research utility script. |
| `create_random_kashmiri_player.py` | Selects 10 diverse random pure Kashmiri audio clips across different Surahs |
| `generate_audio_ui.py` | Standalone research utility script. |
| `generate_compact_widget.py` | Standalone research utility script. |

