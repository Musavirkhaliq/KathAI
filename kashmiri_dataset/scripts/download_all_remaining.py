import os
import sys
from huggingface_hub import snapshot_download, HfApi

TOKEN = os.environ.get("HF_TOKEN")
api = HfApi()

CANDIDATES = [
    # Maitreyajayaraj reasoning & clinical series
    "Maitreyajayaraj/kashmiri_clean_batch_4.1",
    "Maitreyajayaraj/kashmiri_clinical_reasoning_batch_3",
    "Maitreyajayaraj/kashmiri_clinical_reasoning_batch_4",
    "Maitreyajayaraj/kashmiri_medical_math_reasoning_v10",
    "Maitreyajayaraj/kashmiri_medical_math_reasoning_v5.1",
    "Maitreyajayaraj/kashmiri_medical_math_reasoning_v7",
    "Maitreyajayaraj/kashmiri_medical_math_reasoning_v8",
    "Maitreyajayaraj/kashmiri_medical_math_reasoning_v9",

    # Injilashah datasets
    "injilashah/Kashmiri-language-text-dataset",
    "injilashah/Kashmiri-terms",
    "injilashah/kashmiri_text_and_audio",

    # Omarrran accessible datasets
    "Omarrran/200_Kashmiri_food_dishes",
    "Omarrran/31K_Kashmiri_text_and_image_dataset_for_text_Recognition",
    "Omarrran/40K_kashmiri_text_and_image_dataset",
    "Omarrran/Kashmiri_News_Articles_Text_Dataset",
    "Omarrran/Kashmiri_Poet_dataset_Shruk",
    "Omarrran/Kashmiri_Text_Corpus_Cleaned_2025_HNM",
    "Omarrran/Kashmiri_Text_Recognition_Dataset",
    "Omarrran/Kashmiri__Text_Corpus_Dataset",
    "Omarrran/Kashmiri_one_word_text_reconition_dataset",
    "Omarrran/Koshur_Pixel",
    "Omarrran/The_Grierson_Kashmiri_dictionary",
    "Omarrran/kashmiri_English_parallel_corpus_49K",
    "Omarrran/kashmiri_multilingual_dictionary_dataset",
    "Omarrran/kashmiri_parallel_Diacratic_to_Non_diacratic_Text_dataset",
    "Omarrran/kashmiri_sample_text_rocognition",
    "Omarrran/kashmiri_sharegpt",
    "Omarrran/kashmiri_single_speaker_ttd_hf_dataset_2024",
    "Omarrran/kashmiri_tts_hf_dataset",
    "Omarrran/kashmiri_words_dataset_restructured",
    "Omarrran/KS-PRET-5M_5_million_kashmiri_Pretrainning_LLM_dataset_12M_tokens_2026",
    "Omarrran/3.1Million_KASHMIRI_text_Pre_training_Dataset_for_LLM_2026_by_HNM",

    # Krish / Yashwardhan Rasa / Codemixing series
    "Krish187653/rasa-kashmiri-codemixing-transliterations",
    "yashwardhan20417/indicvoices-r-kashmiri-codemixing",
    "yashwardhan20417/rasa-kashmiri-codemixing-merged",
    "yashwardhan20417/rasa-kashmiri-text-only",

    # Other text / parallel
    "Rutts07/mgap-kas_Arab",
    "SMUQamar/Kashmiri-English-Parallel-Corpus",
    "Faizaniqbal/Koshur-OCR-Synth",
    "Faizaniqbal/koshur-synthocr",
    "Aazu34/kashmiri-openslr122-segments",
    "programindz/kashmiri-tts-single-speaker"
]

base_dir = "/home/musa/Kashmiri Project/kashmiri_dataset/external"
os.makedirs(base_dir, exist_ok=True)

successes = []
gated_or_failed = []

for repo in CANDIDATES:
    safe_name = repo.replace("/", "_").lower()
    target_dir = os.path.join(base_dir, safe_name)
    print(f"\n--- Processing: {repo} ---")
    try:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=target_dir,
            token=TOKEN
        )
        print(f"--> SUCCESS: {repo} downloaded to {target_dir}")
        successes.append(repo)
    except Exception as e:
        print(f"--> FAILED/GATED: {repo} ({e})")
        gated_or_failed.append((repo, str(e)))

print("\n================ SUMMARY ================")
print(f"Successfully downloaded: {len(successes)}")
for s in successes:
    print(f"  + {s}")
print(f"Gated or Failed: {len(gated_or_failed)}")
for f, err in gated_or_failed:
    print(f"  - {f}: {err[:80]}...")
