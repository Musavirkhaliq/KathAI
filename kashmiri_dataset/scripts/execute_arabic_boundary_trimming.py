import json
import os
import shutil
import subprocess
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 1. Load ASR model for re-transcribing trimmed clips
model_dir = "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model"
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 2. Mixed clips specifications with exact Kashmiri onset split times
mixed_specs = {
    "quran_kf_s001_seg0002": {
        "path": "kashmiri_dataset/processed/audio/quran_king_fahd/quran_king_fahd_s001_seg0002.wav",
        "split_s": 5.50,
        "trimmed_prefix_type": "arabic_taawwudh"
    },
    "quran_kf_s078_seg0003": {
        "path": "kashmiri_dataset/processed/audio/quran_king_fahd/quran_king_fahd_s078_seg0003.wav",
        "split_s": 4.75,
        "trimmed_prefix_type": "arabic_bismillah"
    },
    "quran_kf_s106_seg0002": {
        "path": "kashmiri_dataset/processed/audio/quran_king_fahd/quran_king_fahd_s106_seg0002.wav",
        "split_s": 5.70,
        "trimmed_prefix_type": "arabic_bismillah"
    },
    "quran_kf_s107_seg0003": {
        "path": "kashmiri_dataset/processed/audio/quran_king_fahd/quran_king_fahd_s107_seg0003.wav",
        "split_s": 5.40,
        "trimmed_prefix_type": "arabic_bismillah"
    }
}

# 3. Pure Arabic segments to quarantine
pure_arabic_ids = set([
    "quran_kf_s001_seg0004", "quran_kf_s080_seg0002", "quran_kf_s081_seg0003",
    "quran_kf_s082_seg0003", "quran_kf_s083_seg0003", "quran_kf_s087_seg0003",
    "quran_kf_s090_seg0002", "quran_kf_s113_seg0003", "quran_war_ep14_seg0003",
    "quran_war_ep16_seg0003", "quran_war_ep19_seg0041", "quran_war_ep24_seg0003",
    "quran_war_ep25_seg0003"
])

# Create backup directory
backup_dir = "kashmiri_dataset/processed/audio/quran_king_fahd/pretrim_backup"
os.makedirs(backup_dir, exist_ok=True)

trimmed_results = {}
for sid, spec in mixed_specs.items():
    orig_wav = spec["path"]
    bak_wav = os.path.join(backup_dir, os.path.basename(orig_wav))
    if not os.path.exists(bak_wav):
        shutil.copy2(orig_wav, bak_wav)
    
    # Trim losslessly using ffmpeg
    split_s = spec["split_s"]
    tmp_trimmed = orig_wav + ".trimmed.wav"
    cmd = [
        "ffmpeg", "-y", "-ss", str(split_s), "-i", orig_wav,
        "-ar", "16000", "-ac", "1", "-f", "wav", tmp_trimmed
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    shutil.move(tmp_trimmed, orig_wav)
    
    # Read trimmed audio duration
    data, sr = sf.read(orig_wav)
    new_dur = round(len(data) / sr, 3)
    
    # Transcribe pure trimmed Kashmiri audio
    inputs = processor(data, sampling_rate=16000, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        new_text = processor.decode(pred_ids)
    
    trimmed_results[sid] = {
        "new_duration": new_dur,
        "new_text": new_text,
        "trimmed_arabic_duration": split_s,
        "trimmed_prefix_type": spec["trimmed_prefix_type"]
    }
    print(f"Trimmed {sid}: New duration: {new_dur}s | Text: {new_text}")

# 4. Update QURAN_NEW_ASR_CANDIDATES.jsonl and QURAN_UNCERTAIN_SPEECH.jsonl
candidates = [json.loads(line) for line in open("QURAN_NEW_ASR_CANDIDATES.jsonl")]
uncertain = [json.loads(line) for line in open("QURAN_UNCERTAIN_SPEECH.jsonl")]

new_candidates = []
moved_to_uncertain = []

for c in candidates:
    sid = c["sample_id"]
    if sid in pure_arabic_ids:
        c["category"] = "arabic_recitation"
        c["uncertainty_reason"] = "pure_arabic_recitation_filtered"
        c["verification_status"] = "tier1_machine_detected"
        c["language"] = "ara"
        moved_to_uncertain.append(c)
    elif sid in trimmed_results:
        res = trimmed_results[sid]
        c["duration"] = res["new_duration"]
        c["start_time_s"] = round(c.get("start_time_s", 0.0) + res["trimmed_arabic_duration"], 3)
        c["text"] = res["new_text"]
        c["normalized_text"] = res["new_text"]
        c["asr_hypothesis"] = res["new_text"]
        c["flags"] = "trimmed_arabic_prefix"
        c["trimmed_arabic_duration_s"] = res["trimmed_arabic_duration"]
        c["trimmed_prefix_type"] = res["trimmed_prefix_type"]
        new_candidates.append(c)
    else:
        new_candidates.append(c)

new_uncertain = uncertain + moved_to_uncertain

with open("QURAN_NEW_ASR_CANDIDATES.jsonl", "w", encoding="utf-8") as f:
    for c in new_candidates:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

with open("QURAN_UNCERTAIN_SPEECH.jsonl", "w", encoding="utf-8") as f:
    for u in new_uncertain:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")

print(f"Updated QURAN_NEW_ASR_CANDIDATES.jsonl: {len(new_candidates)} records.")
print(f"Updated QURAN_UNCERTAIN_SPEECH.jsonl: {len(new_uncertain)} records (added {len(moved_to_uncertain)} quarantined pure Arabic segments).")

cand_dur = sum(c["duration"] for c in new_candidates)
unc_dur = sum(u["duration"] for u in new_uncertain)
print(f"New Candidates Duration: {cand_dur/3600:.4f} hrs ({cand_dur:.2f} s)")
print(f"New Uncertain Duration: {unc_dur/3600:.4f} hrs ({unc_dur:.2f} s)")
