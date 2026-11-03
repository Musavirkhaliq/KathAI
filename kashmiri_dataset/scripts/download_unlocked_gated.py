#!/usr/bin/env python3
"""
Downloads all newly unlocked gated Kashmiri datasets using authenticated user token.
"""
import os
from huggingface_hub import snapshot_download, hf_hub_download

TOKEN = os.environ.get("HF_TOKEN", "")
BASE = "/home/musa/Kashmiri Project/kashmiri_dataset/raw/external"

print("=" * 60)
print("DOWNLOADING NEWLY UNLOCKED GATED KASHMIRI DATASETS")
print("=" * 60)

# 1. SMUQamar 270K Parallel Dataset
print("\n>>> 1. Downloading SMUQamar/Kashmiri-English-Dataset-270K...")
dest = os.path.join(BASE, "smuqamar_270k")
snapshot_download(repo_id="SMUQamar/Kashmiri-English-Dataset-270K", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: SMUQamar 270K")

# 2. Omarrran KS-PRET-5M (5 Million words pretraining text)
print("\n>>> 2. Downloading Omarrran/KS-PRET-5M...")
dest = os.path.join(BASE, "ks_pret_5m")
snapshot_download(repo_id="Omarrran/KS-PRET-5M_5_million_kashmiri_Pretrainning_LLM_dataset_12M_tokens_2026", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: KS-PRET-5M")

# 3. Omarrran KS-LIT-3M (3.1 Million words literature text)
print("\n>>> 3. Downloading Omarrran/KS-LIT-3M...")
dest = os.path.join(BASE, "ks_lit_3m")
snapshot_download(repo_id="Omarrran/3.1Million_KASHMIRI_text_Pre_training_Dataset_for_LLM_2026_by_HNM", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: KS-LIT-3M")

# 4. Omarrran English-Kashmiri 1M Translations
print("\n>>> 4. Downloading Omarrran/english-kashmiri-1m-translations...")
dest = os.path.join(BASE, "omarrran_1m_translations")
snapshot_download(repo_id="Omarrran/english-kashmiri-1m-translations", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: English-Kashmiri 1M Translations")

# 5. AI4Bharat BPCC Kashmiri Files
print("\n>>> 5. Downloading AI4Bharat BPCC Kashmiri files...")
bpcc_dest = os.path.join(BASE, "ai4bharat_bpcc")
os.makedirs(bpcc_dest, exist_ok=True)
bpcc_files = [
    'bpcc-seed-latest/kas_Arab.tsv',
    'bpcc-seed-v1/kas_Arab.tsv',
    'bpcc-seed-v2/kas_Arab.tsv',
    'daily/kas_Arab.tsv',
    'nllb_filtered/kas_Arab.tsv',
    'nllb_seed/kas_Arab.tsv',
    'nllb_seed/kas_Deva.tsv'
]
for bf in bpcc_files:
    try:
        hf_hub_download(repo_id="ai4bharat/bpcc", filename=bf, repo_type="dataset", token=TOKEN, local_dir=bpcc_dest)
        print(f"  Downloaded: {bf}")
    except Exception as e:
        print(f"  Error downloading {bf}: {e}")
print("SUCCESS: AI4Bharat BPCC Kashmiri")

# 6. Omarrran 49K Parallel Corpus
print("\n>>> 6. Downloading Omarrran/kashmiri_English_parallel_corpus_49K...")
dest = os.path.join(BASE, "kashmiri_english_49k")
snapshot_download(repo_id="Omarrran/kashmiri_English_parallel_corpus_49K", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: Kashmiri-English 49K")

# 7. Omarrran Grierson Kashmiri Dictionary Structured
print("\n>>> 7. Downloading Omarrran/The_Grierson_Kashmiri_dictionary...")
dest = os.path.join(BASE, "grierson_dictionary_structured")
snapshot_download(repo_id="Omarrran/The_Grierson_Kashmiri_dictionary", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: Grierson Dictionary")

# 8. InjilaShah Kashmiri Terms
print("\n>>> 8. Downloading injilashah/Kashmiri-terms...")
dest = os.path.join(BASE, "injilashah_kashmiri_terms")
snapshot_download(repo_id="injilashah/Kashmiri-terms", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: InjilaShah Kashmiri Terms")

# 9. Omarrran Cleaned Text Corpus 2025
print("\n>>> 9. Downloading Omarrran/Kashmiri_Text_Corpus_Cleaned_2025_HNM...")
dest = os.path.join(BASE, "omarrran_cleaned_2025")
snapshot_download(repo_id="Omarrran/Kashmiri_Text_Corpus_Cleaned_2025_HNM", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: Cleaned Text Corpus 2025")

# 10. Omarrran Multilingual Dictionary
print("\n>>> 10. Downloading Omarrran/kashmiri_multilingual_dictionary_dataset...")
dest = os.path.join(BASE, "omarrran_multilingual_dict")
snapshot_download(repo_id="Omarrran/kashmiri_multilingual_dictionary_dataset", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: Multilingual Dictionary")

# 11. Omarrran 200 Food Dishes
print("\n>>> 11. Downloading Omarrran/200_Kashmiri_food_dishes...")
dest = os.path.join(BASE, "omarrran_food_dishes")
snapshot_download(repo_id="Omarrran/200_Kashmiri_food_dishes", repo_type="dataset", token=TOKEN, local_dir=dest)
print("SUCCESS: 200 Food Dishes")

# 12. Project Vaani Kashmiri Transcription
print("\n>>> 12. Downloading ARTPARK-IISc/Vaani-transcription-part (Kashmiri files)...")
vaani_dest = os.path.join(BASE, "vaani_kashmiri")
os.makedirs(vaani_dest, exist_ok=True)
try:
    snapshot_download(repo_id="ARTPARK-IISc/Vaani-transcription-part", repo_type="dataset", token=TOKEN, allow_patterns=["*Kashmiri*", "*kas*"], local_dir=vaani_dest)
    print("SUCCESS: Project Vaani Kashmiri")
except Exception as e:
    print(f"Vaani download note: {e}")

print("\n" + "=" * 60)
print("ALL NEWLY UNLOCKED GATED DATASETS DOWNLOADED SUCCESSFULLY!")
print("=" * 60)
