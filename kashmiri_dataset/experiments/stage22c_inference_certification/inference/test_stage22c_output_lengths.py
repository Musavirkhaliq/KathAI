#!/usr/bin/env python3
"""
Unit Test Suite: Exact CTC Output Length Calculation & Padding Invariance
Validates stage22c_exact_output_lengths against actual PyTorch Wav2Vec2 forward passes.
"""

import unittest
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC
from stage22c_exact_output_lengths import (
    compute_exact_ctc_length,
    compute_exact_ctc_lengths_batch,
    slice_exact_logits
)

class TestStage22cOutputLengths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "/home/musa/Kashmiri Project/kashmiri_dataset/experiments/stage22_data_expanded_training/checkpoints/E22-01/best_model"
        cls.model = Wav2Vec2ForCTC.from_pretrained(cls.model_path)
        cls.model.eval()

    def test_receptive_field_boundary(self):
        # < 400 samples should return 0
        self.assertEqual(compute_exact_ctc_length(399), 0)
        self.assertEqual(compute_exact_ctc_length(1), 0)
        self.assertEqual(compute_exact_ctc_length(0), 0)
        
        # Exactly 400 samples produces 1 frame
        self.assertEqual(compute_exact_ctc_length(400), 1)
        x = torch.zeros(1, 400)
        with torch.no_grad():
            out = self.model(x).logits
        self.assertEqual(out.shape[1], 1)

    def test_forward_pass_parity_across_durations(self):
        test_samples = [
            400, 640, 1600, 3200, 16000, 32000, 80000,
            160000, 320000, 400000, 400001, 400256, 960000
        ]
        for n in test_samples:
            x = torch.zeros(1, n)
            with torch.no_grad():
                out = self.model(x).logits
            actual_frames = out.shape[1]
            predicted_frames = compute_exact_ctc_length(n)
            hf_model_frames = self.model._get_feat_extract_output_lengths(torch.tensor(n)).item()
            
            self.assertEqual(
                predicted_frames, actual_frames,
                f"Mismatch for sample length {n}: predicted {predicted_frames}, actual {actual_frames}"
            )
            self.assertEqual(
                predicted_frames, hf_model_frames,
                f"Mismatch vs HuggingFace internal helper for sample length {n}"
            )

    def test_padded_batch_slicing_invariance(self):
        # Create a batch with two different lengths: 16,000 (1s) and 320,000 (20s)
        len1 = 16000
        len2 = 320000
        
        wav1 = torch.randn(len1)
        wav2 = torch.randn(len2)
        
        # Padded batch
        padded_wavs = torch.zeros(2, len2)
        padded_wavs[0, :len1] = wav1
        padded_wavs[1, :] = wav2
        
        with torch.no_grad():
            padded_logits = self.model(padded_wavs).logits
            solo_logits_1 = self.model(wav1.unsqueeze(0)).logits
            solo_logits_2 = self.model(wav2.unsqueeze(0)).logits
        
        # Slice padded logits using exact calculation
        exact_t1 = compute_exact_ctc_length(len1)
        exact_t2 = compute_exact_ctc_length(len2)
        
        sliced_1 = padded_logits[0, :exact_t1, :].cpu().numpy()
        sliced_2 = padded_logits[1, :exact_t2, :].cpu().numpy()
        
        # Sliced frame counts must match solo exactly
        self.assertEqual(sliced_1.shape[0], solo_logits_1.shape[1])
        self.assertEqual(sliced_2.shape[0], solo_logits_2.shape[1])
        
        # Verify that round(N/320) would have included a padded frame
        approx_t1 = round(len1 / 320.0)
        self.assertEqual(approx_t1, exact_t1 + 1)
        print(f"\n[Test Verification] Solo frames: {exact_t1} | round(N/320): {approx_t1} (eliminated 1 padded frame)")

if __name__ == "__main__":
    unittest.main()
