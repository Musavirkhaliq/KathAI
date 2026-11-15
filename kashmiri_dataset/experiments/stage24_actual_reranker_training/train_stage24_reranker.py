#!/usr/bin/env python3
"""
Stage 24: Candidate-Relative Neural Sequence Ranker Trainer
Trains the CandidateRelativeRanker on supervised Kashmiri N-best hypotheses.

Strict Stage 24 Constraints:
1. Total added neural parameters <= 1,396,255 (DGX GB10 parameter budget ceiling).
2. Protected baseline fusion with zero-initialization guarantee:
   At Step 0, candidate rankings match 100% exact baseline KenLM predictions.
3. Ranking Loss: ListNet KL-divergence + Pairwise Error-Weighted Margin + Residual Regularization.
4. Model selection strictly on dev_calibration.jsonl (1,001 utterances).
5. All checkpoints saved with full metadata, hashes, optimizer states, and evaluation records.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import time
import math
import random
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import jiwer
from transformers import Wav2Vec2Processor

WORKSPACE = Path("/home/musa/Kashmiri Project")
sys.path.insert(0, str(WORKSPACE))

from kashmiri_dataset.scripts.metrics_evaluator import normalize_perso_arabic_kashmiri as norm_strict

STAGE24_DIR = WORKSPACE / "kashmiri_dataset/experiments/stage24_actual_reranker_training"
CONFIG_PATH = STAGE24_DIR / "configs/stage24_training_config.yaml"
TRAIN_NBEST = STAGE24_DIR / "nbest_cache/stage24_train_nbest_20k.jsonl"
CALIB_NBEST = STAGE24_DIR / "nbest_cache/stage24_dev_calibration_nbest.jsonl"
CHECKPOINT_DIR = STAGE24_DIR / "checkpoints"
LOG_DIR = STAGE24_DIR / "training_logs"

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# ==============================================================================
# 1. CANDIDATE RELATIVE RANKER ARCHITECTURE
# ==============================================================================

class CandidateRelativeRanker(nn.Module):
    """
    Candidate-Relative Neural Sequence Ranker (CR-Ranker).
    Evaluates N-best candidates by comparing sequence representations
    and acoustic/LM feature deltas relative to Top-1 baseline hypothesis.
    """
    def __init__(
        self,
        vocab_size: int = 161,
        embed_dim: int = 64,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_scalar_features: int = 6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Character / token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        enc_out_dim = hidden_dim * 2 # 192 dims

        # Relative difference projector: [h_diff (192), h_mult (192), scalar_features (6)] = 390 dims
        self.diff_proj = nn.Sequential(
            nn.Linear(enc_out_dim * 2 + num_scalar_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Final scalar score prediction head
        self.score_head = nn.Linear(64, 1)

        # Zero-initialize the final scoring layer to guarantee exact baseline parity at step 0
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_sequence(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of token sequences with length masking.
        Returns utterance-level pooled representations (batch_size, hidden_dim * 2).
        """
        embeds = self.embedding(input_ids)
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        mask = torch.arange(out.size(1), device=input_ids.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        sum_pooled = torch.sum(out * mask, dim=1)
        denom = lengths.unsqueeze(-1).float().clamp(min=1.0)
        pooled = sum_pooled / denom
        return pooled

    def forward(
        self,
        cand_input_ids: torch.Tensor,      # (B, N, L_cand)
        cand_lengths: torch.Tensor,        # (B, N)
        base_input_ids: torch.Tensor,      # (B, L_base)
        base_lengths: torch.Tensor,        # (B,)
        scalar_features: torch.Tensor      # (B, N, num_scalar_features)
    ) -> torch.Tensor:
        """
        Computes candidate relative score adjustments.
        Returns:
          relative_scores: (B, N) tensor of score adjustments relative to candidate 0.
        """
        B, N, L_cand = cand_input_ids.shape

        flat_cand_ids = cand_input_ids.view(B * N, L_cand)
        flat_cand_lens = cand_lengths.view(B * N)

        # Encode candidates
        h_cand = self.encode_sequence(flat_cand_ids, flat_cand_lens)
        h_cand = h_cand.view(B, N, -1)

        # Encode baseline candidate 0
        h_base = self.encode_sequence(base_input_ids, base_lengths)
        h_base_exp = h_base.unsqueeze(1).expand(-1, N, -1)

        # Interaction representation
        h_diff = h_cand - h_base_exp
        h_mult = h_cand * h_base_exp
        comp_repr = torch.cat([h_diff, h_mult, scalar_features], dim=-1)

        hidden = self.diff_proj(comp_repr)
        raw_scores = self.score_head(hidden).squeeze(-1) # (B, N)

        # Candidate 0 is baseline: strictly 0.0 relative adjustment
        scores = raw_scores - raw_scores[:, :1]
        return scores

class ProtectedBaselineFusion(nn.Module):
    """
    Applies bounded, regularized fusion between baseline decoder scores and ranker scores:
    Final_Score = Baseline_Score + lambda * tanh(Ranker_Score)
    """
    def __init__(self, lambda_init: float = 0.01, gamma_bound: float = 5.0):
        super().__init__()
        self.fusion_lambda = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.gamma_bound = gamma_bound

    def forward(self, baseline_scores: torch.Tensor, ranker_scores: torch.Tensor) -> torch.Tensor:
        bounded_ranker = torch.tanh(ranker_scores / self.gamma_bound) * self.gamma_bound
        return baseline_scores + torch.abs(self.fusion_lambda) * bounded_ranker

# ==============================================================================
# 2. DATASET LOADER & LOSS FUNCTIONS
# ==============================================================================

class NBestCandidateDataset(Dataset):
    def __init__(self, jsonl_path, max_candidates=32, max_seq_len=80):
        self.max_candidates = max_candidates
        self.max_seq_len = max_seq_len
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        cands = rec["candidates"][:self.max_candidates]
        if not cands:
            cands = [{
                "cand_idx": 0, "text": "", "norm_text": "", "token_ids": [1],
                "acoustic_score": 0.0, "lm_score": 0.0, "combined_score": 0.0,
                "num_words": 0, "char_length": 0, "edit_distance": 0
            }]

        base_cand = cands[0]
        N = len(cands)

        cand_ids = np.zeros((self.max_candidates, self.max_seq_len), dtype=np.int64)
        cand_lens = np.zeros(self.max_candidates, dtype=np.int64)
        scalar_feats = np.zeros((self.max_candidates, 6), dtype=np.float32)
        base_scores = np.zeros(self.max_candidates, dtype=np.float32)
        edit_distances = np.zeros(self.max_candidates, dtype=np.float32)

        base_tokens = base_cand.get("token_ids", [])[:self.max_seq_len]
        base_len = max(1, len(base_tokens))
        base_id_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        base_id_arr[:len(base_tokens)] = base_tokens

        b_ac = base_cand["acoustic_score"]
        b_lm = base_cand["lm_score"]
        b_wlen = base_cand["num_words"]
        b_clen = base_cand["char_length"]
        b_comb = base_cand["combined_score"]

        for i, c in enumerate(cands):
            tokens = c.get("token_ids", [])[:self.max_seq_len]
            cand_lens[i] = max(1, len(tokens))
            cand_ids[i, :len(tokens)] = tokens

            d_ac = c["acoustic_score"] - b_ac
            d_lm = c["lm_score"] - b_lm
            d_w = c["num_words"] - b_wlen
            d_c = c["char_length"] - b_clen
            disagree = abs(d_ac - d_lm)
            rel_margin = (c["combined_score"] - b_comb) / max(abs(b_comb), 1.0)

            scalar_feats[i] = [d_ac, d_lm, d_w, d_c, disagree, rel_margin]
            base_scores[i] = c["combined_score"]
            edit_distances[i] = c["edit_distance"]

        # Padding for candidate dimension if < max_candidates
        if N < self.max_candidates:
            for j in range(N, self.max_candidates):
                cand_ids[j] = cand_ids[0]
                cand_lens[j] = cand_lens[0]
                base_scores[j] = -1e9
                edit_distances[j] = 1e6

        return {
            "cand_ids": torch.tensor(cand_ids, dtype=torch.long),
            "cand_lens": torch.tensor(cand_lens, dtype=torch.long),
            "base_ids": torch.tensor(base_id_arr, dtype=torch.long),
            "base_lens": torch.tensor(base_len, dtype=torch.long),
            "scalar_feats": torch.tensor(scalar_feats, dtype=torch.float32),
            "base_scores": torch.tensor(base_scores, dtype=torch.float32),
            "edit_distances": torch.tensor(edit_distances, dtype=torch.float32)
        }

def compute_ranking_loss(
    final_scores: torch.Tensor,
    rel_scores: torch.Tensor,
    edit_distances: torch.Tensor,
    tau_true: float = 2.0,
    tau_pred: float = 1.0,
    list_weight: float = 1.0,
    pair_weight: float = 0.5,
    margin_weight: float = 0.1,
    reg_weight: float = 0.01
):
    """
    Combined differentiable ranking loss:
    1. ListNet KL-divergence loss
    2. Pairwise Margin Ranking loss
    3. Regularization on residual adjustments
    """
    # 1. ListNet
    p_true = F.softmax(-edit_distances / tau_true, dim=-1)
    log_p_pred = F.log_softmax(final_scores / tau_pred, dim=-1)
    loss_list = -(p_true * log_p_pred).sum(dim=-1).mean()

    # 2. Pairwise Margin Ranking
    # diff_err[b, i, j] = edit_distances[j] - edit_distances[i] (positive if i has fewer errors)
    diff_err = edit_distances.unsqueeze(2) - edit_distances.unsqueeze(1)
    diff_score = final_scores.unsqueeze(1) - final_scores.unsqueeze(2)
    mask_better = (diff_err > 0).float()
    margin = margin_weight * diff_err
    pair_violations = F.relu(margin - diff_score) * mask_better
    norm_denom = mask_better.sum(dim=(-1, -2)).clamp(min=1.0)
    loss_pair = (pair_violations.sum(dim=(-1, -2)) / norm_denom).mean()

    # 3. Regularization
    loss_reg = (rel_scores ** 2).mean()

    total_loss = list_weight * loss_list + pair_weight * loss_pair + reg_weight * loss_reg
    return total_loss, loss_list.item(), loss_pair.item(), loss_reg.item()

def evaluate_on_records(model, fusion, records, device="cuda"):
    model.eval()
    fusion.eval()

    total_words = 0
    total_errors = 0
    subs = 0
    dels = 0
    ins = 0
    preserved = 0
    improved = 0
    degraded = 0

    max_seq_len = 80

    with torch.no_grad():
        for rec in records:
            cands = rec["candidates"]
            if not cands:
                continue

            ref_norm = rec["ref_norm"]
            ref_words = len(ref_norm.split())
            total_words += ref_words

            N = len(cands)
            base_cand = cands[0]
            base_tokens = base_cand.get("token_ids", [])[:max_seq_len]
            base_len = max(1, len(base_tokens))
            base_id_arr = np.zeros(max_seq_len, dtype=np.int64)
            base_id_arr[:len(base_tokens)] = base_tokens

            cand_ids = np.zeros((1, N, max_seq_len), dtype=np.int64)
            cand_lens = np.zeros((1, N), dtype=np.int64)
            scalar_feats = np.zeros((1, N, 6), dtype=np.float32)
            base_scores = np.zeros((1, N), dtype=np.float32)

            b_ac = base_cand["acoustic_score"]
            b_lm = base_cand["lm_score"]
            b_wlen = base_cand["num_words"]
            b_clen = base_cand["char_length"]
            b_comb = base_cand["combined_score"]

            for i, c in enumerate(cands):
                toks = c.get("token_ids", [])[:max_seq_len]
                cand_lens[0, i] = max(1, len(toks))
                cand_ids[0, i, :len(toks)] = toks

                d_ac = c["acoustic_score"] - b_ac
                d_lm = c["lm_score"] - b_lm
                d_w = c["num_words"] - b_wlen
                d_c = c["char_length"] - b_clen
                disagree = abs(d_ac - d_lm)
                rel_margin = (c["combined_score"] - b_comb) / max(abs(b_comb), 1.0)

                scalar_feats[0, i] = [d_ac, d_lm, d_w, d_c, disagree, rel_margin]
                base_scores[0, i] = c["combined_score"]

            t_c_ids = torch.tensor(cand_ids, dtype=torch.long, device=device)
            t_c_lens = torch.tensor(cand_lens, dtype=torch.long, device=device)
            t_b_ids = torch.tensor(base_id_arr, dtype=torch.long, device=device).unsqueeze(0)
            t_b_lens = torch.tensor([base_len], dtype=torch.long, device=device)
            t_feats = torch.tensor(scalar_feats, dtype=torch.float32, device=device)
            t_base_scores = torch.tensor(base_scores, dtype=torch.float32, device=device)

            rel_scores = model(t_c_ids, t_c_lens, t_b_ids, t_b_lens, t_feats)
            fused = fusion(t_base_scores, rel_scores)

            best_idx = int(torch.argmax(fused[0]).cpu().item())
            selected_cand = cands[best_idx]

            errs = selected_cand["edit_distance"]
            s = selected_cand["subs"]
            d = selected_cand["dels"]
            i_err = selected_cand["ins"]

            total_errors += errs
            subs += s
            dels += d
            ins += i_err

            top1_errs = cands[0]["edit_distance"]
            if best_idx == 0:
                preserved += 1
            elif errs < top1_errs:
                improved += 1
            elif errs > top1_errs:
                degraded += 1
            else:
                preserved += 1

    wer = (total_errors / total_words) * 100.0 if total_words > 0 else 0.0
    return {
        "wer_pct": round(wer, 2),
        "total_errors": total_errors,
        "subs": subs,
        "dels": dels,
        "ins": ins,
        "total_words": total_words,
        "top1_preserved": preserved,
        "top1_preservation_pct": round(preserved / len(records) * 100.0, 2),
        "improved_utts": improved,
        "degraded_utts": degraded,
        "net_utt_change": improved - degraded
    }

# ==============================================================================
# 3. TRAINING ENGINE
# ==============================================================================

def train_reranker():
    print("=" * 70)
    print("STAGE 24: CANDIDATE-RELATIVE NEURAL SEQUENCE RANKER TRAINING")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize model
    model = CandidateRelativeRanker(
        vocab_size=161,
        embed_dim=64,
        hidden_dim=96,
        num_layers=2,
        dropout=0.1,
        num_scalar_features=6
    ).to(device)

    fusion = ProtectedBaselineFusion(lambda_init=0.01, gamma_bound=5.0).to(device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} (Budget: 1,396,255 params | Overhead: {param_count/1396255*100:.1f}%)")

    # Load calibration records for monitoring
    calib_records = []
    with open(CALIB_NBEST, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                calib_records.append(json.loads(line.strip()))
    print(f"Loaded {len(calib_records)} calibration utterances from {CALIB_NBEST.name}.")

    # Step 0 Baseline Parity Check
    step0_stats = evaluate_on_records(model, fusion, calib_records, device=device)
    print(f"\n[Step 0 Baseline Parity Check]")
    print(f"  Calib WER: {step0_stats['wer_pct']:.2f}% | Errors: {step0_stats['total_errors']} (S={step0_stats['subs']}, D={step0_stats['dels']}, I={step0_stats['ins']})")
    print(f"  Baseline Preservation: {step0_stats['top1_preservation_pct']}% ({step0_stats['top1_preserved']}/{len(calib_records)})")
    assert step0_stats['top1_preservation_pct'] == 100.0, "Step 0 must have 100.0% exact baseline parity!"
    print("  -> Step 0 Verification: PASSED (Exact Baseline Parity Verified)")

    # Load training dataset
    print(f"\nLoading training dataset from {TRAIN_NBEST.name}...")
    train_dataset = NBestCandidateDataset(str(TRAIN_NBEST), max_candidates=32, max_seq_len=80)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Training dataset loaded: {len(train_dataset)} utterances ({len(train_loader)} batches per epoch).")

    # Optimizer and Scheduler
    lr = 1e-4
    epochs = 10
    total_steps = len(train_loader) * epochs
    warmup_steps = 100

    optimizer = AdamW(list(model.parameters()) + list(fusion.parameters()), lr=lr, weight_decay=0.01)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training state
    training_history = []
    best_calib_errors = step0_stats["total_errors"]
    best_epoch = 0
    patience = 3
    patience_counter = 0

    t_start = time.time()
    global_step = 0

    # Save Step 0 checkpoint
    torch.save({
        "epoch": 0,
        "step": 0,
        "model_state": model.state_dict(),
        "fusion_state": fusion.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "calib_stats": step0_stats
    }, CHECKPOINT_DIR / "ranker_epoch_0.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        fusion.train()
        epoch_loss = 0.0
        epoch_list_loss = 0.0
        epoch_pair_loss = 0.0
        epoch_reg_loss = 0.0
        t_ep = time.time()

        for batch_idx, batch in enumerate(train_loader):
            c_ids = batch["cand_ids"].to(device)
            c_lens = batch["cand_lens"].to(device)
            b_ids = batch["base_ids"].to(device)
            b_lens = batch["base_lens"].to(device)
            feats = batch["scalar_feats"].to(device)
            base_scores = batch["base_scores"].to(device)
            edit_dists = batch["edit_distances"].to(device)

            optimizer.zero_grad()
            rel_scores = model(c_ids, c_lens, b_ids, b_lens, feats)
            fused_scores = fusion(base_scores, rel_scores)

            loss, l_list, l_pair, l_reg = compute_ranking_loss(
                fused_scores, rel_scores, edit_dists,
                tau_true=2.0, tau_pred=1.0,
                list_weight=1.0, pair_weight=0.5, margin_weight=0.1, reg_weight=0.01
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_list_loss += l_list
            epoch_pair_loss += l_pair
            epoch_reg_loss += l_reg

        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_list = epoch_list_loss / num_batches
        avg_pair = epoch_pair_loss / num_batches
        avg_reg = epoch_reg_loss / num_batches

        # Evaluate on calibration set
        calib_stats = evaluate_on_records(model, fusion, calib_records, device=device)
        ep_duration = time.time() - t_ep
        cur_lambda = float(fusion.fusion_lambda.item())

        print(f"Epoch {epoch:2d}/{epochs:2d} ({ep_duration:.1f}s) | Loss: {avg_loss:.4f} (List: {avg_list:.4f}, Pair: {avg_pair:.4f}, Reg: {avg_reg:.4f}) | Calib WER: {calib_stats['wer_pct']:.2f}% (Errors: {calib_stats['total_errors']}, Net Delta: {calib_stats['total_errors']-step0_stats['total_errors']:+d} words) | Lambda: {cur_lambda:.4f}")

        record_log = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(avg_loss, 4),
            "list_loss": round(avg_list, 4),
            "pair_loss": round(avg_pair, 4),
            "reg_loss": round(avg_reg, 4),
            "calib_wer_pct": calib_stats["wer_pct"],
            "calib_total_errors": calib_stats["total_errors"],
            "subs": calib_stats["subs"],
            "dels": calib_stats["dels"],
            "ins": calib_stats["ins"],
            "top1_preservation_pct": calib_stats["top1_preservation_pct"],
            "improved_utts": calib_stats["improved_utts"],
            "degraded_utts": calib_stats["degraded_utts"],
            "net_error_delta": calib_stats["total_errors"] - step0_stats["total_errors"],
            "fusion_lambda": round(cur_lambda, 4),
            "epoch_duration_sec": round(ep_duration, 1)
        }
        training_history.append(record_log)

        # Save epoch checkpoint
        ckpt_path = CHECKPOINT_DIR / f"ranker_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "step": global_step,
            "model_state": model.state_dict(),
            "fusion_state": fusion.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "calib_stats": calib_stats,
            "sha256": ""
        }, ckpt_path)

        if calib_stats["total_errors"] < best_calib_errors:
            best_calib_errors = calib_stats["total_errors"]
            best_epoch = epoch
            patience_counter = 0
            best_ckpt_path = CHECKPOINT_DIR / "best_calib_ranker.pt"
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state": model.state_dict(),
                "fusion_state": fusion.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "calib_stats": calib_stats
            }, best_ckpt_path)
            print(f"  *** NEW BEST CHECKPOINT AT EPOCH {epoch}! Total Errors: {best_calib_errors} (WER: {calib_stats['wer_pct']:.2f}%) ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nTraining completed in {time.time()-t_start:.1f}s. Best epoch: {best_epoch} with {best_calib_errors} calibration errors.")

    # Save training history CSV
    df_hist = pd.DataFrame(training_history)
    csv_path = STAGE24_DIR / "stage24_training_history.csv"
    df_hist.to_csv(csv_path, index=False)
    print(f"Saved training history to {csv_path}")

    # Build checkpoint registry
    registry = []
    for ep in range(0, epoch + 1):
        cp = CHECKPOINT_DIR / f"ranker_epoch_{ep}.pt"
        if cp.exists():
            h = get_sha256(cp)
            data = torch.load(cp, map_location="cpu", weights_only=False)
            c_stats = data.get("calib_stats", {})
            registry.append({
                "checkpoint": cp.name,
                "epoch": ep,
                "step": data.get("step", 0),
                "sha256": h,
                "calib_wer_pct": c_stats.get("wer_pct", 0.0),
                "calib_errors": c_stats.get("total_errors", 0),
                "is_best": (ep == best_epoch)
            })

    reg_path = STAGE24_DIR / "stage24_checkpoint_registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Saved checkpoint registry to {reg_path}")

if __name__ == "__main__":
    train_reranker()
