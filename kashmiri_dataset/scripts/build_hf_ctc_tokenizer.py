import json
import os
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

BASE_DIR = Path("/home/musa/Kashmiri Project")
EXP_DIR = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "experiments" / "B-01-xlsr300m-character-ctc"
VOCAB_FILE = BASE_DIR / "kashmiri_dataset" / "experiments" / "native_asr_training" / "training_configurations" / "kashmiri_ctc_vocab.json"
TOKENIZER_DIR = EXP_DIR / "tokenizer"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

# Copy vocab file into tokenizer directory
import shutil
shutil.copyfile(VOCAB_FILE, TOKENIZER_DIR / "vocab.json")

# Instantiate and save standard Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer(
    str(TOKENIZER_DIR / "vocab.json"),
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)
tokenizer.save_pretrained(str(TOKENIZER_DIR))

# Create Feature Extractor for 16kHz audio
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)
feature_extractor.save_pretrained(str(TOKENIZER_DIR))

# Combined Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained(str(TOKENIZER_DIR))

print(f"Hugging Face CTC Processor & Tokenizer created successfully at: {TOKENIZER_DIR}")
print(f"Tokenizer Vocab Size: {len(tokenizer)}")
