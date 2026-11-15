import os
import sys
import json
import time
import random
import unicodedata
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from metrics_evaluator import levenshtein_distance

# Determinism
torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character-level vocabulary
# Read train data to build compact char vocabulary
train_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/train.jsonl"
val_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/val.jsonl"
test_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/romanization_benchmark/test.jsonl"

src_chars = set()
tgt_chars = set()

with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        src_chars.update(it["arabic"])
        tgt_chars.update(it["roman_ascii"].lower())

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

src_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
for c in sorted(src_chars):
    if c not in src_vocab: src_vocab[c] = len(src_vocab)

tgt_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
for c in sorted(tgt_chars):
    if c not in tgt_vocab: tgt_vocab[c] = len(tgt_vocab)

inv_tgt_vocab = {idx: tok for tok, idx in tgt_vocab.items()}

print(f"Char Vocabularies -> Source (Arabic): {len(src_vocab)}, Target (Roman): {len(tgt_vocab)}")

# Dataset Definition
class TranslitDataset(Dataset):
    def __init__(self, jsonl_file, max_samples=None):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                it = json.loads(line)
                self.samples.append((it["arabic"], it["roman_ascii"].lower()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        src_ids = [src_vocab.get(c, UNK_IDX) for c in src]
        tgt_ids = [SOS_IDX] + [tgt_vocab.get(c, UNK_IDX) for c in tgt] + [EOS_IDX]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_lens = [len(s) for s in src_list]
    tgt_lens = [len(t) for t in tgt_list]
    
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    
    src_pad = torch.full((len(batch), max_src), PAD_IDX, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_tgt), PAD_IDX, dtype=torch.long)
    
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_pad[i, :len(s)] = s
        tgt_pad[i, :len(t)] = t
        
    return src_pad, tgt_pad

# Architecture: Bidirectional GRU Encoder + Bahdanau Attention Decoder (Lightweight Neural Adapter)
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim * 2, hidden_dim)
        self.U_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, mask=None):
        # query: [B, H], keys: [B, T_src, 2H]
        q_proj = self.U_a(query).unsqueeze(1) # [B, 1, H]
        k_proj = self.W_a(keys)               # [B, T_src, H]
        scores = self.v_a(torch.tanh(q_proj + k_proj)).squeeze(-1) # [B, T_src]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1) # [B, T_src]
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1) # [B, 2H]
        return context, attn_weights

class Seq2SeqAdapter(nn.Module):
    def __init__(self, src_vocab_sz, tgt_vocab_sz, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_sz, emb_dim, padding_idx=PAD_IDX)
        self.encoder = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.tgt_emb = nn.Embedding(tgt_vocab_sz, emb_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(hidden_dim)
        self.decoder_cell = nn.GRUCell(emb_dim + hidden_dim * 2, hidden_dim)
        self.out_head = nn.Linear(hidden_dim + hidden_dim * 2, tgt_vocab_sz)
        self.enc_to_dec = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, T_src = src.shape
        _, T_tgt = tgt.shape
        
        src_mask = (src != PAD_IDX)
        enc_out, _ = self.encoder(self.src_emb(src)) # [B, T_src, 2H]
        
        # Init decoder hidden
        mean_enc = enc_out.mean(dim=1)
        h = torch.tanh(self.enc_to_dec(mean_enc)) # [B, H]
        
        outputs = []
        curr_in = tgt[:, 0] # SOS
        
        for t in range(1, T_tgt):
            emb_in = self.tgt_emb(curr_in) # [B, E]
            context, _ = self.attention(h, enc_out, src_mask) # [B, 2H]
            h = self.decoder_cell(torch.cat([emb_in, context], dim=-1), h) # [B, H]
            logits = self.out_head(torch.cat([h, context], dim=-1)) # [B, V]
            outputs.append(logits.unsqueeze(1))
            
            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher:
                curr_in = tgt[:, t]
            else:
                curr_in = logits.argmax(dim=-1)
                
        return torch.cat(outputs, dim=1) # [B, T_tgt-1, V]

    def generate(self, src, max_len=120):
        self.eval()
        with torch.no_grad():
            B, T_src = src.shape
            src_mask = (src != PAD_IDX)
            enc_out, _ = self.encoder(self.src_emb(src))
            mean_enc = enc_out.mean(dim=1)
            h = torch.tanh(self.enc_to_dec(mean_enc))
            
            curr_in = torch.full((B,), SOS_IDX, dtype=torch.long, device=src.device)
            generated_ids = []
            
            for _ in range(max_len):
                emb_in = self.tgt_emb(curr_in)
                context, _ = self.attention(h, enc_out, src_mask)
                h = self.decoder_cell(torch.cat([emb_in, context], dim=-1), h)
                logits = self.out_head(torch.cat([h, context], dim=-1))
                curr_in = logits.argmax(dim=-1)
                generated_ids.append(curr_in.unsqueeze(1))
                
            return torch.cat(generated_ids, dim=1)

print("Architecture defined successfully!")
