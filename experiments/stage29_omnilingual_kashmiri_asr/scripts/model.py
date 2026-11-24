#!/usr/bin/env python3
"""
Native PyTorch Implementation of Meta's Omnilingual Wav2Vec2 CTC Architecture
Supports:
- omniASR_CTC_300M_v2 (300M params, 24 layers, model_dim 1024, ffn 4096, 16 heads)
- omniASR_CTC_1B_v2   (1B params, 48 layers, model_dim 1280, ffn 5120, 16 heads)
- SentencePiece omniASR_tokenizer_written_v2 (10,288 vocab size)
- Official checkpoint loading and state_dict alignment
- Native PyTorch CTC Loss and greedy decoding
- BF16 mixed precision and gradient checkpointing
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class OmniModelConfig:
    model_dim: int = 1024
    num_encoder_layers: int = 24
    num_encoder_attn_heads: int = 16
    ffn_inner_dim: int = 4096
    feature_dim: int = 512
    vocab_size: int = 10288
    dropout_p: float = 0.0
    attn_dropout_p: float = 0.0
    ffn_inner_dropout_p: float = 0.1
    layer_drop_p: float = 0.0
    final_dropout_p: float = 0.0
    pos_conv_kernel_size: int = 128
    pos_conv_groups: int = 16

    @classmethod
    def get_300m_config(cls) -> "OmniModelConfig":
        return cls(
            model_dim=1024,
            num_encoder_layers=24,
            num_encoder_attn_heads=16,
            ffn_inner_dim=4096,
            feature_dim=512,
            vocab_size=10288,
        )

    @classmethod
    def get_1b_config(cls) -> "OmniModelConfig":
        return cls(
            model_dim=1280,
            num_encoder_layers=48,
            num_encoder_attn_heads=16,
            ffn_inner_dim=5120,
            feature_dim=512,
            vocab_size=10288,
        )


class ConvLayerBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, is_first: bool):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=False)
        self.is_first = is_first
        if is_first:
            self.layer_norm = nn.GroupNorm(out_ch, out_ch, eps=1e-5)
        else:
            self.layer_norm = nn.LayerNorm(out_ch, eps=1e-5)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.conv(x)
        if self.is_first:
            x = self.layer_norm(x)
        else:
            # LayerNorm over channels dimension
            x = x.transpose(1, 2)  # (B, T, C)
            x = self.layer_norm(x)
            x = x.transpose(1, 2)  # (B, C, T)
        x = self.activation(x)
        return x


class FeatureExtractor(nn.Module):
    """
    Standard wav2vec 2.0 7-layer 1D CNN feature extractor.
    Total subsampling factor = 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320.
    At 16kHz, frame rate is 50 Hz (20ms shift).
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        layer_specs = [
            (1, feature_dim, 10, 5),
            (feature_dim, feature_dim, 3, 2),
            (feature_dim, feature_dim, 3, 2),
            (feature_dim, feature_dim, 3, 2),
            (feature_dim, feature_dim, 3, 2),
            (feature_dim, feature_dim, 2, 2),
            (feature_dim, feature_dim, 2, 2),
        ]
        self.layers = nn.ModuleList([
            ConvLayerBlock(in_ch, out_ch, k, s, is_first=(i == 0))
            for i, (in_ch, out_ch, k, s) in enumerate(layer_specs)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T_samples) or (B, T_samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        # Returns (B, C, T_frames)
        return x


class PositionalConvEmbedding(nn.Module):
    """
    Relative convolutional positional embedding.
    Conv1d with kernel_size=128, groups=16, padding=64, followed by GELU.
    """
    def __init__(self, model_dim: int, kernel_size: int = 128, groups: int = 16):
        super().__init__()
        self.conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv)
        # Trim if odd padding
        if x_conv.size(2) > residual.size(1):
            x_conv = x_conv[:, :, :residual.size(1)]
        x_conv = self.activation(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, T, D)
        return residual + x_conv


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer with PRE-norm order (Meta fairseq2 specification).
    x = x + Dropout(SelfAttention(LayerNorm(x)))
    x = x + Dropout(FFN(LayerNorm(x)))
    """
    def __init__(self, config: OmniModelConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.num_encoder_attn_heads,
            dropout=config.attn_dropout_p,
            batch_first=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.model_dim, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout_p)

        self.ffn_inner = nn.Linear(config.model_dim, config.ffn_inner_dim)
        self.ffn_act = nn.GELU()
        self.ffn_dropout = nn.Dropout(config.ffn_inner_dropout_p)
        self.ffn_output = nn.Linear(config.ffn_inner_dim, config.model_dim)
        self.ffn_layer_norm = nn.LayerNorm(config.model_dim, eps=1e-5)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm Self-Attention
        residual = x
        x_norm = self.self_attn_layer_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False)
        x = residual + self.dropout(attn_out)

        # Pre-norm FFN
        residual = x
        x_norm = self.ffn_layer_norm(x)
        ffn_out = self.ffn_output(self.ffn_dropout(self.ffn_act(self.ffn_inner(x_norm))))
        x = residual + self.dropout(ffn_out)
        return x


class Wav2Vec2AsrModel(nn.Module):
    """
    Complete Omnilingual Wav2Vec2 ASR Architecture.
    """
    def __init__(self, config: OmniModelConfig):
        super().__init__()
        self.config = config

        # 1. Feature Extractor
        self.feature_extractor = FeatureExtractor(config.feature_dim)
        self.post_extract_layer_norm = nn.LayerNorm(config.feature_dim, eps=1e-5)

        # 2. Model dim projection
        if config.feature_dim != config.model_dim:
            self.model_dim_proj = nn.Linear(config.feature_dim, config.model_dim)
        else:
            self.model_dim_proj = nn.Identity()

        # 3. Positional encoding
        self.pos_encoder = PositionalConvEmbedding(
            config.model_dim,
            kernel_size=config.pos_conv_kernel_size,
            groups=config.pos_conv_groups,
        )
        self.dropout = nn.Dropout(config.dropout_p)

        # 4. Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])
        self.encoder_layer_norm = nn.LayerNorm(config.model_dim, eps=1e-5)

        # 5. Final Output Projection to CTC Vocab
        self.final_dropout = nn.Dropout(config.final_dropout_p) if config.final_dropout_p > 0 else nn.Identity()
        self.final_proj = nn.Linear(config.model_dim, config.vocab_size)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes output lengths after 7-layer CNN feature extractor.
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        cur_len = input_lengths
        cur_len = _conv_out_length(cur_len, 10, 5)
        cur_len = _conv_out_length(cur_len, 3, 2)
        cur_len = _conv_out_length(cur_len, 3, 2)
        cur_len = _conv_out_length(cur_len, 3, 2)
        cur_len = _conv_out_length(cur_len, 3, 2)
        cur_len = _conv_out_length(cur_len, 2, 2)
        cur_len = _conv_out_length(cur_len, 2, 2)
        return torch.clamp(cur_len, min=1)

    def forward(
        self,
        audio_seqs: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            audio_seqs: (B, T_samples) audio waveform at 16kHz
            audio_lengths: (B,) lengths in samples
            targets: (B, L) target token IDs (for CTC loss)
            target_lengths: (B,) length of target sequences
        Returns:
            If targets is None: (logits, out_lengths)
            If targets is not None: (ctc_loss, logits, out_lengths)
        """
        # Feature extraction
        feats = self.feature_extractor(audio_seqs)  # (B, C, T_frames)
        feats = feats.transpose(1, 2)  # (B, T_frames, C)
        feats = self.post_extract_layer_norm(feats)

        # Model projection & Positional Encoding
        x = self.model_dim_proj(feats)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Compute frame lengths
        if audio_lengths is not None:
            out_lens = self._get_feat_extract_output_lengths(audio_lengths)
            max_len = x.size(1)
            # Create padding mask (True for padded positions)
            mask = torch.arange(max_len, device=x.device)[None, :] >= out_lens[:, None]
        else:
            out_lens = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
            mask = None

        # Transformer encoder layers
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, key_padding_mask=mask)

        x = self.encoder_layer_norm(x)
        x = self.final_dropout(x)
        logits = self.final_proj(x)  # (B, T_frames, vocab_size)

        if targets is None:
            return logits, out_lens

        # CTC loss computation in FP32
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)  # (T, B, V)

        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=out_lens,
            target_lengths=target_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        return loss, logits, out_lens

    @torch.no_grad()
    def decode_greedy(
        self,
        logits: torch.Tensor,
        out_lens: torch.Tensor,
        tokenizer,
        blank_id: int = 0,
        pad_id: int = 1,
    ) -> List[str]:
        """
        Standard CTC greedy decoding with blank collapsing and deduplication.
        """
        pred_ids = torch.argmax(logits, dim=-1)  # (B, T)
        transcriptions = []

        for i in range(pred_ids.size(0)):
            seq = pred_ids[i, :out_lens[i]]
            # Deduplicate consecutive tokens
            mask = torch.ones(seq.size(0), dtype=torch.bool, device=seq.device)
            mask[1:] = seq[1:] != seq[:-1]
            dedup_ids = seq[mask]

            # Remove blanks and special tokens
            valid_ids = [tid.item() for tid in dedup_ids if tid.item() not in (blank_id, pad_id)]
            text = tokenizer.DecodeIds(valid_ids)
            transcriptions.append(text)

        return transcriptions


def load_omni_checkpoint(
    model: Wav2Vec2AsrModel,
    checkpoint_path: str,
    device: str = "cpu"
) -> Wav2Vec2AsrModel:
    """
    Loads Meta Omnilingual ASR checkpoint into the Wav2Vec2AsrModel,
    mapping official fairseq2 state_dict keys to our model.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        state = state["model"]

    model_state = model.state_dict()
    mapped_state = {}
    missing_keys = []

    for name, param in model_state.items():
        if name in state and param.shape == state[name].shape:
            mapped_state[name] = state[name]
        elif name.startswith("feature_extractor."):
            c_name = f"encoder_frontend.{name}"
            if c_name in state:
                mapped_state[name] = state[c_name]
        elif name.startswith("model_dim_proj."):
            c_name = f"encoder_frontend.{name}"
            if c_name in state:
                mapped_state[name] = state[c_name]
        elif name == "post_extract_layer_norm.weight":
            mapped_state[name] = state["encoder_frontend.post_extract_layer_norm.weight"]
        elif name == "post_extract_layer_norm.bias":
            mapped_state[name] = state["encoder_frontend.post_extract_layer_norm.bias"]
        elif name == "pos_encoder.conv.bias":
            mapped_state[name] = state["encoder_frontend.pos_encoder.conv.bias"]
        elif name == "pos_encoder.conv.weight":
            g = state["encoder_frontend.pos_encoder.conv.weight_g"]
            v = state["encoder_frontend.pos_encoder.conv.weight_v"]
            norm_v = torch.norm(v, dim=(1, 2), keepdim=True)
            w = v * (g / (norm_v + 1e-8))
            mapped_state[name] = w
        elif name.startswith("encoder_layer_norm."):
            suffix = name.split(".")[-1]
            c_name = f"encoder.layer_norm.{suffix}"
            if c_name in state:
                mapped_state[name] = state[c_name]
        elif name.startswith("encoder_layers."):
            parts = name.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])
            if rest == "self_attn.in_proj_weight":
                q = state[f"encoder.layers.{layer_idx}.self_attn.q_proj.weight"]
                k = state[f"encoder.layers.{layer_idx}.self_attn.k_proj.weight"]
                v = state[f"encoder.layers.{layer_idx}.self_attn.v_proj.weight"]
                mapped_state[name] = torch.cat([q, k, v], dim=0)
            elif rest == "self_attn.in_proj_bias":
                q = state[f"encoder.layers.{layer_idx}.self_attn.q_proj.bias"]
                k = state[f"encoder.layers.{layer_idx}.self_attn.k_proj.bias"]
                v = state[f"encoder.layers.{layer_idx}.self_attn.v_proj.bias"]
                mapped_state[name] = torch.cat([q, k, v], dim=0)
            elif rest.startswith("self_attn.out_proj."):
                suffix = parts[-1]
                mapped_state[name] = state[f"encoder.layers.{layer_idx}.self_attn.output_proj.{suffix}"]
            elif rest.startswith("self_attn_layer_norm."):
                suffix = parts[-1]
                mapped_state[name] = state[f"encoder.layers.{layer_idx}.self_attn_layer_norm.{suffix}"]
            elif rest.startswith("ffn_inner."):
                suffix = parts[-1]
                mapped_state[name] = state[f"encoder.layers.{layer_idx}.ffn.inner_proj.{suffix}"]
            elif rest.startswith("ffn_output."):
                suffix = parts[-1]
                mapped_state[name] = state[f"encoder.layers.{layer_idx}.ffn.output_proj.{suffix}"]
            elif rest.startswith("ffn_layer_norm."):
                suffix = parts[-1]
                mapped_state[name] = state[f"encoder.layers.{layer_idx}.ffn_layer_norm.{suffix}"]
        elif name.startswith("final_proj."):
            if name in state:
                mapped_state[name] = state[name]

        if name not in mapped_state:
            missing_keys.append(name)

    print(f"Mapped {len(mapped_state)}/{len(model_state)} parameter tensors.")
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys: {missing_keys[:5]}...")

    model.load_state_dict(mapped_state, strict=True)
    print("Checkpoint successfully loaded into model with strict=True verification.")
    return model
