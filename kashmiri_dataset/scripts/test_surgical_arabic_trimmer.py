import json
import re
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model_dir = '/home/musa/Kashmiri Project/kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model'
proc = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir).cuda().half().eval()

# Common Arabic tokens that appear at the beginning of recitations
ARABIC_PREFIX_TOKENS = {
    'الذین', 'امنوا', 'کفروا', 'قالوا', 'قال', 'علیہم', 'ان اللہ', 'سبحان', 'الحمد',
    'یا ایہا', 'کانوا', 'کنتم', 'فلا', 'ولکن', 'انما', 'یقولون', 'تبارک', 'ولا', 'وما',
    'پلنس', 'پلنُس', 'ولاکک', 'ہُدللِناس', 'یخولوٗنس', 'فاقیم', 'فاتو', 'ودعو', 'ودوٗ',
    'ویلم', 'وقال', 'افلات'
}

KASHMIRI_ANCHOR_WORDS = {
    'چھُ', 'چھِ', 'چُھ', 'چِھ', 'چھُس', 'چھِو', 'چھُکھ', 'سٕندِ', 'سٕنزِ', 'سٕند', 'ہُند', 'ہٕند',
    'سۭتۍ', 'سٍتۍ', 'یِوان', 'گوٚو', 'ٲسۍ', 'کٔرِ', 'تِمن', 'لوٗکن', 'تہٕ', 'أسۍ', 'تۅہیہِ',
    'پَنُن', 'پَنٕنۍ', 'بیٚیہِ', 'پس', 'یعنے', 'مگر', 'بلکہ', 'پٔزۍ', 'پٲٹھۍ', 'خداے', 'اللہ',
    'فرموو', 'ونان', 'بوزِو', 'توٚہۍ', 'تِمو', 'دوٚپ', 'کتھ'
}

def analyze_and_trim_clip(record, audio_root, dry_run=True):
    audio_p = Path(audio_root) / record['audio_path']
    if not audio_p.exists():
        return 'missing_audio', record
        
    audio, sr = sf.read(str(audio_p))
    if len(audio) < 1600:
        return 'too_short', record
        
    txt = record['text'].strip()
    words = txt.split()
    
    # 1. Check if first word is known Arabic prefix
    first_word = words[0] if words else ''
    first_2_words = ' '.join(words[:2]) if len(words) >= 2 else first_word
    
    has_text_arabic = any(ar in first_2_words for ar in ARABIC_PREFIX_TOKENS)
    
    # 2. Check if text has any Kashmiri anchors at all
    has_kashmiri = any(k in txt for k in KASHMIRI_ANCHOR_WORDS)
    if not has_kashmiri and has_text_arabic:
        # 100% Pure Arabic recitation
        return 'pure_arabic_purge', record
        
    # Run CTC model to find frame-level alignment
    inputs = proc(audio, sampling_rate=16000, return_tensors='pt').input_values.cuda().half()
    with torch.no_grad():
        logits = model(inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    tokens = proc.tokenizer.convert_ids_to_tokens(pred_ids.cpu().numpy())
    frame_dur = len(audio) / sr / len(pred_ids)
    
    # Decode words with timestamps
    decoded_words = []
    curr = ''
    start_f = 0
    for f_idx, t in enumerate(tokens):
        if t == '|':
            if curr:
                decoded_words.append((curr.replace('[PAD]', ''), start_f * frame_dur, f_idx * frame_dur))
                curr = ''
        elif t not in ['<pad>', '<unk>', '<s>', '</s>']:
            if not curr:
                start_f = f_idx
            curr += t
    if curr:
        decoded_words.append((curr.replace('[PAD]', ''), start_f * frame_dur, len(tokens) * frame_dur))
        
    # Check if first decoded word is Arabic
    if not decoded_words:
        return 'empty_decode', record
        
    first_dec = decoded_words[0][0]
    first_dec_ar = any(ar in first_dec for ar in ARABIC_PREFIX_TOKENS)
    
    if has_text_arabic or first_dec_ar:
        # Find where the Kashmiri speech starts
        # Look for the first Kashmiri anchor word in decoded words
        kash_onset = None
        kash_word_idx = None
        
        for w_idx, (w, s, e) in enumerate(decoded_words):
            if any(k in w for k in KASHMIRI_ANCHOR_WORDS):
                kash_onset = s
                kash_word_idx = w_idx
                break
                
        if kash_onset is not None and kash_onset > 0.8:
            # We found an Arabic prefix that can be trimmed!
            trim_time = max(0.0, kash_onset - 0.05) # 50ms padding
            new_dur = (len(audio) / sr) - trim_time
            if new_dur >= 1.0:
                # Find corresponding text start
                # Match the first kashmiri word in original text
                text_words = txt.split()
                new_text_words = []
                for idx, tw in enumerate(text_words):
                    if any(k in tw for k in KASHMIRI_ANCHOR_WORDS):
                        new_text_words = text_words[idx:]
                        break
                new_text = ' '.join(new_text_words) if new_text_words else txt
                
                return 'trimmed', {
                    'sample_id': record['sample_id'],
                    'old_dur': record['duration'],
                    'new_dur': round(new_dur, 2),
                    'trim_sec': round(trim_time, 2),
                    'old_text': txt,
                    'new_text': new_text
                }
                
    return 'clean', record

if __name__ == '__main__':
    manifest = '/home/musa/Kashmiri Project/kashmiri_dataset/manifests/source_manifests/manifest_quran_kashmiri.jsonl'
    with open(manifest, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
        
    test_ids = [
        'quran_kashmiri_s007_seg0044', 'quran_kashmiri_s002_seg1024',
        'quran_kashmiri_s005_seg0089', 'quran_kashmiri_s002_seg0004'
    ]
    test_records = [r for r in records if r['sample_id'] in test_ids]
    
    for r in test_records:
        status, res = analyze_and_trim_clip(r, '/home/musa/Kashmiri Project', dry_run=True)
        print(f"[{status}] {res['sample_id']} | trim: {res.get('trim_sec', 0)}s | {res.get('new_dur', res.get('duration'))}s")
        if status == 'trimmed':
            print(f"   OLD: {res['old_text'][:50]}...")
            print(f"   NEW: {res['new_text'][:50]}...")
