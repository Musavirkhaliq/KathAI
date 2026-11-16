"""
Unified Kashmiri ASR Inference Engine.
======================================
Provides end-to-end transcription supporting:
- 16 kHz audio loading & peak normalization
- Exact CTC convolutional receptive field slicing
- Context-preserving chunking (Method C2) for long-form speech
- Greedy CTC and Calibrated KenLM beam search decoding
- Kashmiri Perso-Arabic orthographic normalization
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import numpy as np
import torch

from src.asr.model import load_asr_model, DEFAULT_MODEL_DIR
from src.data_processing.audio_processor import load_audio, normalize_audio
from src.decoding.ctc_slice import slice_exact_logits, compute_exact_ctc_length
from src.decoding.chunking import chunk_audio_c2
from src.decoding.kenlm_decoder import CalibratedKenLMDecoder
from src.normalization.normalizer import normalize_kashmiri_text
from src.shared.logging import get_logger

logger = get_logger("asr_engine")


class KashmiriASR:
    """
    Production-grade Kashmiri Speech-to-Text Engine.
    """
    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        kenlm_path: Optional[Union[str, Path]] = None,
        unigrams_path: Optional[Union[str, Path]] = None
    ):
        self.model, self.processor, self.device = load_asr_model(model_dir, device)
        self.kenlm_decoder: Optional[CalibratedKenLMDecoder] = None

        if kenlm_path and Path(kenlm_path).exists():
            try:
                vocab = self.processor.tokenizer.get_vocab()
                sorted_vocab = [k for k, v in sorted(vocab.items(), key=lambda item: item[1])]
                labels = [token if token != "|" else " " for token in sorted_vocab]
                self.kenlm_decoder = CalibratedKenLMDecoder(
                    vocab_list=labels,
                    kenlm_path=kenlm_path,
                    unigrams_path=unigrams_path
                )
                logger.info(f"Loaded KenLM decoder from {kenlm_path}")
            except Exception as e:
                logger.warning(f"Could not initialize KenLM decoder: {e}")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        normalize: bool = False,
        use_kenlm: bool = False,
        chunk_long_audio: bool = True
    ) -> str:
        """
        Transcribes audio file to Kashmiri Perso-Arabic text.
        """
        res = self.transcribe_detailed(
            audio_path=audio_path,
            normalize=normalize,
            use_kenlm=use_kenlm,
            chunk_long_audio=chunk_long_audio
        )
        return res["text"]

    def transcribe_detailed(
        self,
        audio_path: Union[str, Path],
        normalize: bool = False,
        use_kenlm: bool = False,
        chunk_long_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribes audio and returns detailed metadata (duration, text, greedy_text, kenlm_text).
        """
        wav, sr = load_audio(audio_path, target_sample_rate=16000, to_mono=True)
        wav = normalize_audio(wav)
        duration_sec = len(wav) / 16000.0

        if len(wav) < 400:
            return {
                "text": "",
                "greedy_text": "",
                "duration_seconds": duration_sec,
                "model": "KashmiriASR"
            }

        # Check if chunking is needed for long audio (> 30s)
        if chunk_long_audio and len(wav) > 30 * 16000:
            chunks = chunk_audio_c2(wav)
            all_greedy = []
            for c_wav, v_start, v_end in chunks:
                if len(c_wav) < 400:
                    continue
                inp = self.processor(c_wav, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
                with torch.no_grad():
                    logits = self.model(inp).logits
                sliced = slice_exact_logits(logits, len(c_wav))
                pred_ids = torch.argmax(sliced, dim=-1)[0].cpu().numpy()
                text_chunk = self.processor.decode(pred_ids)
                if text_chunk.strip():
                    all_greedy.append(text_chunk.strip())
            greedy_text = " ".join(all_greedy)
        else:
            inp = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(inp).logits

            sliced = slice_exact_logits(logits, len(wav))
            pred_ids = torch.argmax(sliced, dim=-1)[0].cpu().numpy()
            greedy_text = self.processor.decode(pred_ids).strip()

        final_text = greedy_text

        # KenLM decoding if requested and available
        kenlm_text = None
        if use_kenlm and self.kenlm_decoder is not None and len(wav) <= 30 * 16000:
            try:
                kenlm_text = self.kenlm_decoder.decode(sliced)
                final_text = kenlm_text
            except Exception as e:
                logger.warning(f"KenLM decoding failed, falling back to greedy: {e}")

        if normalize:
            final_text = normalize_kashmiri_text(final_text)

        return {
            "text": final_text,
            "greedy_text": greedy_text,
            "kenlm_text": kenlm_text,
            "duration_seconds": round(duration_sec, 2),
            "device": str(self.device)
        }
