import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path("/home/musa/Kashmiri Project")
DATA_DIR = BASE_DIR / "kashmiri_dataset/experiments/romanization_evaluation/data"
OUTPUT_DIR = BASE_DIR / "kashmiri_dataset/experiments/romanization_evaluation/neural_seq2seq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build Char Vocabularies
src_chars = set()
tgt_chars = set()

def load_data(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for l in f:
            it = json.loads(l)
            s = it["source"].strip()
            t = it["target_ascii"].strip()
            if s and t:
                rows.append((s, t))
    return rows

train_rows = load_data(DATA_DIR / "train.jsonl")
val_rows = load_data(DATA_DIR / "val.jsonl")
test_rows = load_data(DATA_DIR / "test.jsonl")

for s, t in train_rows:
    src_chars.update(s)
    tgt_chars.update(t.lower())

src_vocab = {c: i+3 for i, c in enumerate(sorted(src_chars))}
src_vocab["<pad>"] = 0
src_vocab["<unk>"] = 1
src_vocab["<eos>"] = 2

tgt_vocab = {c: i+3 for i, c in enumerate(sorted(tgt_chars))}
tgt_vocab["<pad>"] = 0
tgt_vocab["<unk>"] = 1
tgt_vocab["<sos>"] = 2
tgt_vocab["<eos>"] = 3
rev_tgt_vocab = {i: c for c, i in tgt_vocab.items()}

print(f"Source Char Vocab Size: {len(src_vocab)}, Target Char Vocab Size: {len(tgt_vocab)}")

# Save vocabs
with open(OUTPUT_DIR / "vocab_src.json", "w", encoding="utf-8") as f:
    json.dump(src_vocab, f, ensure_ascii=False)
with open(OUTPUT_DIR / "vocab_tgt.json", "w", encoding="utf-8") as f:
    json.dump(tgt_vocab, f, ensure_ascii=False)

class CharDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s, t = self.rows[idx]
        src_ids = [src_vocab.get(c, 1) for c in s] + [2]
        tgt_ids = [2] + [tgt_vocab.get(c, 1) for c in t.lower()] + [3]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    srcs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

train_loader = DataLoader(CharDataset(train_rows), batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(CharDataset(val_rows), batch_size=32, shuffle=False, collate_fn=collate_fn)

# Compact Attention Seq2Seq Model
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        # hidden: [B, H], enc_outputs: [B, T, H]
        B, T, H = enc_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attention.unsqueeze(1), enc_outputs).squeeze(1)
        return context

class Seq2Seq(nn.Module):
    def __init__(self, src_v, tgt_v, emb_dim=128, hid_dim=256):
        super().__init__()
        self.src_emb = nn.Embedding(src_v, emb_dim, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_v, emb_dim, padding_idx=0)
        self.encoder = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(hid_dim * 2, hid_dim)
        self.attention = Attention(hid_dim)
        self.decoder = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, tgt_v)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, src_len = src.shape
        _, tgt_len = tgt.shape
        
        enc_out, enc_hidden = self.encoder(self.src_emb(src))
        # enc_hidden: [2, B, H] -> [B, H]
        dec_hidden = torch.tanh(self.enc_proj(torch.cat((enc_hidden[-2], enc_hidden[-1]), dim=1)))
        
        # enc_out: [B, T, 2H] -> project to H
        enc_out = self.enc_proj(enc_out)

        outputs = torch.zeros(B, tgt_len - 1, len(tgt_vocab)).to(device)
        dec_input = tgt[:, 0] # <sos>

        for t in range(1, tgt_len):
            emb = self.tgt_emb(dec_input).unsqueeze(1) # [B, 1, E]
            context = self.attention(dec_hidden, enc_out) # [B, H]
            dec_in_combined = torch.cat((emb, context.unsqueeze(1)), dim=2)
            dec_out, dec_hidden = self.decoder(dec_in_combined, dec_hidden.unsqueeze(0))
            dec_hidden = dec_hidden.squeeze(0)

            out = self.fc_out(torch.cat((dec_out.squeeze(1), context, emb.squeeze(1)), dim=1))
            outputs[:, t-1] = out
            
            is_teacher = torch.rand(1).item() < teacher_forcing_ratio
            top1 = out.argmax(1)
            dec_input = tgt[:, t] if is_teacher else top1

        return outputs

model = Seq2Seq(len(src_vocab), len(tgt_vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Training Neural Character Seq2Seq...")

best_loss = float("inf")
for epoch in range(1, 6):
    model.train()
    total_loss = 0
    t0 = time.time()
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        target = tgt[:, 1:]
        loss = criterion(output.reshape(-1, len(tgt_vocab)), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            target = tgt[:, 1:]
            loss = criterion(output.reshape(-1, len(tgt_vocab)), target.reshape(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch}/5 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time()-t0:.1f}s")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "best_seq2seq.pt")

print("Saved Neural Seq2Seq Model!")
