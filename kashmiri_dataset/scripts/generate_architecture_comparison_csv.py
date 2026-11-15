import csv
from pathlib import Path

A01_DIR = Path("kashmiri_dataset/experiments/native_asr_training/experiments/A-01-indicconformer-kashmiri-finetune")

rows = [
    {
        "System": "B-01: XLS-R 300M + Character CTC",
        "Architecture": "Wav2Vec2-XLS-R (Conformer/Transformer Hybrid)",
        "Parameters": "317M",
        "Acoustic Unit": "161 Kashmiri Character Tokens",
        "Raw WER": "47.63%",
        "Norm WER": "32.13%",
        "Raw CER": "19.83%",
        "Norm CER": "15.01%",
        "Empty Hyp %": "0.00%",
        "Inference RTF": "0.0129 (77.3x RT)",
        "Training Hours": "11.87h",
        "Checkpoint Status": "Frozen Epoch 13 (Native Fine-Tuned)"
    },
    {
        "System": "A-01: IndicConformer Kashmiri Pretrained Baseline",
        "Architecture": "Emformer/Conformer Large (CTC + RNN-T)",
        "Parameters": "120M / 600M Encoder Family",
        "Acoustic Unit": "257 SentencePiece Subwords",
        "Raw WER": "48.98%",
        "Norm WER": "33.82%",
        "Raw CER": "20.13%",
        "Norm CER": "14.95%",
        "Empty Hyp %": "0.39%",
        "Inference RTF": "0.0643 (15.6x RT on CPU)",
        "Training Hours": "Pretrained (AI4Bharat / Bhashini)",
        "Checkpoint Status": "Official HuggingFace/ONNX Pretrained"
    }
]

with open(A01_DIR / "architecture_comparison.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("Saved architecture_comparison.csv successfully.")
