import json
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Set seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

BASE_DIR = Path("/home/musa/Kashmiri Project")
DATA_DIR = BASE_DIR / "kashmiri_dataset/experiments/romanization_evaluation/data"
OUTPUT_DIR = BASE_DIR / "kashmiri_dataset/experiments/romanization_evaluation/byt5_adapter"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/byt5-small"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TransliterationDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["source"], item["target_ascii"]

# Load 5,000 train samples for fast convergence (plenty for transliteration) and 500 val samples
train_dataset = TransliterationDataset(DATA_DIR / "train.jsonl", max_samples=5000)
val_dataset = TransliterationDataset(DATA_DIR / "val.jsonl", max_samples=500)

def collate_fn(batch):
    sources = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    model_inputs = tokenizer(sources, max_length=128, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=128, padding=True, truncation=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

print(f"Starting Training: {len(train_dataset)} samples, {EPOCHS} epochs, device: {device}...")
t0 = time.time()
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"--> [Epoch {epoch} Done] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time()-t0:.1f}s")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"Saving best model to {OUTPUT_DIR}...")
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"Neural Romanizer Training Completed in {(time.time()-t0)/60:.2f} minutes!")
