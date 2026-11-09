#!/usr/bin/env python3
"""
KathAI Research Papers Downloader
Downloads all 25 curated foundation papers for Kashmiri Text, Speech, and Multimodal AI.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PAPERS = {
    "text_and_tokenization": [
        {
            "filename": "01_sennrich2016_subword_bpe_nmt.pdf",
            "url": "https://arxiv.org/pdf/1508.07909.pdf",
            "title": "Neural Machine Translation of Rare Words with Subword Units",
            "authors": "Rico Sennrich, Barry Haddow, Alexandra Birch (ACL 2016)"
        },
        {
            "filename": "02_kudo2018_subword_regularization_unigram.pdf",
            "url": "https://arxiv.org/pdf/1804.10959.pdf",
            "title": "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates",
            "authors": "Taku Kudo (ACL 2018)"
        },
        {
            "filename": "03_kudo2018_sentencepiece_system.pdf",
            "url": "https://arxiv.org/pdf/1808.06226.pdf",
            "title": "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing",
            "authors": "Taku Kudo, John Richardson (EMNLP 2018)"
        },
        {
            "filename": "04_petrov2023_tokenization_tax_discrepancies.pdf",
            "url": "https://arxiv.org/pdf/2310.08754.pdf",
            "title": "Language Model Tokenizers Introduce Systematic Discrepancies Across Languages",
            "authors": "Aleksandar Petrov, Emanuele La Malfa, Philip H.S. Torr, Adel Bibi (NeurIPS 2023)"
        },
        {
            "filename": "05_ahia2023_do_all_languages_cost_the_same.pdf",
            "url": "https://arxiv.org/pdf/2305.13707.pdf",
            "title": "Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models",
            "authors": "Orevaoghene Ahia et al. (EMNLP 2023)"
        },
        {
            "filename": "06_conneau2020_xlm_roberta.pdf",
            "url": "https://arxiv.org/pdf/1911.02116.pdf",
            "title": "Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)",
            "authors": "Alexis Conneau et al. (ACL 2020)"
        },
        {
            "filename": "07_singh2024_aya_multilingual_dataset.pdf",
            "url": "https://arxiv.org/pdf/2402.06619.pdf",
            "title": "Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning",
            "authors": "Shivalika Singh et al. (Cohere For AI, 2024)"
        },
        {
            "filename": "08_gala2023_indictrans2_22_languages.pdf",
            "url": "https://arxiv.org/pdf/2305.16307.pdf",
            "title": "IndicTrans2: Towards High-Quality and Accessible Machine Translation for all 22 Scheduled Indian Languages",
            "authors": "Jay Gala et al. (AI4Bharat, 2023)"
        },
        {
            "filename": "09_xue2022_byt5_token_free_models.pdf",
            "url": "https://arxiv.org/pdf/2105.13626.pdf",
            "title": "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models",
            "authors": "Linting Xue et al. (TACL 2022)"
        },
        {
            "filename": "10_yu2023_megabyte_multiscale_transformers.pdf",
            "url": "https://arxiv.org/pdf/2305.07185.pdf",
            "title": "MEGABYTE: Predicting High-Million-Byte Sequences with Multiscale Transformers",
            "authors": "Lili Yu et al. (Meta AI, NeurIPS 2023)"
        }
    ],
    "speech_and_audio": [
        {
            "filename": "01_graves2006_connectionist_temporal_classification_ctc.pdf",
            "url": "https://www.cs.toronto.edu/~graves/icml_2006.pdf",
            "title": "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks",
            "authors": "Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber (ICML 2006)"
        },
        {
            "filename": "02_baevski2020_wav2vec2_self_supervised_speech.pdf",
            "url": "https://arxiv.org/pdf/2006.11477.pdf",
            "title": "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations",
            "authors": "Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (NeurIPS 2020)"
        },
        {
            "filename": "03_babu2021_xls_r_128_languages.pdf",
            "url": "https://arxiv.org/pdf/2111.09296.pdf",
            "title": "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale",
            "authors": "Arun Babu et al. (Meta AI, Interspeech 2022)"
        },
        {
            "filename": "04_gulati2020_conformer_speech_recognition.pdf",
            "url": "https://arxiv.org/pdf/2005.08100.pdf",
            "title": "Conformer: Convolution-augmented Transformer for Speech Recognition",
            "authors": "Anmol Gulati et al. (Google, Interspeech 2020)"
        },
        {
            "filename": "05_radford2022_whisper_robust_asr.pdf",
            "url": "https://arxiv.org/pdf/2212.04356.pdf",
            "title": "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)",
            "authors": "Alec Radford et al. (OpenAI, 2022)"
        },
        {
            "filename": "06_javed2024_indicvoices_dataset.pdf",
            "url": "https://arxiv.org/pdf/2403.01926.pdf",
            "title": "IndicVoices: Towards Building an Inclusive Multilingual Speech Dataset for Indian Languages",
            "authors": "Tahir Javed et al. (AI4Bharat, 2024)"
        },
        {
            "filename": "07_sankar2024_indicvoices_r_tts_corpus.pdf",
            "url": "https://arxiv.org/pdf/2409.05356.pdf",
            "title": "IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS",
            "authors": "Ashwin Sankar et al. (AI4Bharat, 2024)"
        },
        {
            "filename": "08_mehta2024_matcha_tts_flow_matching.pdf",
            "url": "https://arxiv.org/pdf/2309.03199.pdf",
            "title": "Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching",
            "authors": "Shivam Mehta et al. (ICASSP 2024)"
        },
        {
            "filename": "09_chen2024_f5_tts_flow_matching_voice_clone.pdf",
            "url": "https://arxiv.org/pdf/2410.06885.pdf",
            "title": "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching",
            "authors": "Yushen Chen et al. (2024)"
        }
    ],
    "speech_llms_and_multimodal": [
        {
            "filename": "01_seamless2023_seamlessm4t_multimodal.pdf",
            "url": "https://arxiv.org/pdf/2308.11596.pdf",
            "title": "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation",
            "authors": "Seamless Communication et al. (Meta AI, 2023)"
        },
        {
            "filename": "02_rubenstein2023_audiopalm_speaks_and_listens.pdf",
            "url": "https://arxiv.org/pdf/2306.12925.pdf",
            "title": "AudioPaLM: A Large Language Model That Can Speak and Listen",
            "authors": "Paul K. Rubenstein et al. (Google DeepMind, 2023)"
        },
        {
            "filename": "03_zhang2023_speechgpt_conversational_llm.pdf",
            "url": "https://arxiv.org/pdf/2305.11000.pdf",
            "title": "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities",
            "authors": "Dong Zhang et al. (EMNLP 2023)"
        },
        {
            "filename": "04_qwen2024_qwen2_audio_technical_report.pdf",
            "url": "https://arxiv.org/pdf/2407.10759.pdf",
            "title": "Qwen2-Audio Technical Report",
            "authors": "Qwen Team et al. (Alibaba, 2024)"
        }
    ],
    "kashmiri_and_low_resource": [
        {
            "filename": "01_malik2024_koshur_pixel_synthetic_ocr.pdf",
            "url": "https://arxiv.org/pdf/2606.23144.pdf",
            "title": "Koshur Pixel: A Large-Scale Synthetic OCR Dataset for Kashmiri",
            "authors": "Haq Nawaz Malik et al. (arXiv:2606.23144)"
        },
        {
            "filename": "02_malik2024_600k_ks_ocr_dataset.pdf",
            "url": "https://arxiv.org/pdf/2601.01088.pdf",
            "title": "600k-KS-OCR: A Large-Scale Synthetic Dataset for Optical Character Recognition in Kashmiri Script",
            "authors": "Haq Nawaz Malik et al. (arXiv:2601.01088)"
        }
    ]
}

def main():
    total = sum(len(items) for items in PAPERS.values())
    downloaded = 0
    failed = 0

    print(f"=== KathAI Research Papers Downloader ({total} papers) ===")
    print(f"Destination: {BASE_DIR}\n")

    for category, items in PAPERS.items():
        cat_dir = BASE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Category: {category} ({len(items)} papers) ---")
        
        for item in items:
            dest = cat_dir / item["filename"]
            if dest.exists() and dest.stat().st_size > 10000:
                print(f"  [EXISTS] {item['filename']} ({dest.stat().st_size // 1024} KB)")
                downloaded += 1
                continue
                
            print(f"  [DOWNLOADING] {item['filename']} ...")
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
            
            success = False
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                        if data[:4] == b"%PDF":
                            with open(dest, "wb") as f:
                                f.write(data)
                            print(f"    ✓ Saved {dest.name} ({len(data) // 1024} KB)")
                            downloaded += 1
                            success = True
                            break
                        else:
                            print(f"    ✗ Non-PDF header received: {data[:20]}")
                except Exception as e:
                    print(f"    ✗ Attempt {attempt+1} error: {e}")
                    time.sleep(2)
                    
            if not success:
                failed += 1
                print(f"    FAILED: {item['filename']}")
                
            time.sleep(0.5)

    print(f"\nFinished: {downloaded}/{total} papers present ({failed} failures).")

if __name__ == "__main__":
    main()
