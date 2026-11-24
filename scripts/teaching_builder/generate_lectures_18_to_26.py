#!/usr/bin/env python3
"""
generate_lectures_18_to_26.py
Generates all 11 standard files for Lectures 18 through 26 (Modules 06 - 08):
- Module 06: Acoustic Model Training & Scaling (L18 - L20)
- Module 07: Language Modeling & Decoding Algorithms (L21 - L23)
- Module 08: Scientific Benchmarking & Error Analysis (L24 - L26)
"""

import os
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/teaching/modules")

LECTURES = [
    {
        "num": 18,
        "mod": "module_06_training_and_optimization",
        "dir": "lecture_18_finetuning_dynamics_and_schedules",
        "title": "Fine-Tuning Dynamics, Schedules & Warmup",
        "subtitle": "Learning Rate Schedules, Linear Warmup, Cosine Annealing, and Discriminative Layer Fine-Tuning",
        "objectives": [
            "Deconstruct the optimization dynamics of fine-tuning large self-supervised models on small target corpora.",
            "Formulate linear learning rate warmup and cosine decay scheduling.",
            "Analyze the risks of catastrophic forgetting when fine-tuning lower versus upper Transformer layers.",
            "Implement a customized PyTorch learning rate scheduler with layer-wise decay."
        ],
        "notes": """# Lecture 18: Fine-Tuning Dynamics, Schedules & Warmup

## 1. The Fine-Tuning Optimization Dilemma

Fine-tuning a 300M-parameter pretrained model (such as XLS-R) on 50 to 100 hours of Kashmiri speech presents a delicate optimization trade-off:
- **Under-Tuning**: If the learning rate is too low ($< 10^{-6}$), the model fails to adapt its general representations to Kashmiri phonemes within practical training epochs.
- **Catastrophic Forgetting**: If the learning rate is too high ($> 5 \\times 10^{-4}$), or if large gradients shock the network at step 0, the pretrained cross-lingual weights are wiped out, destroying the universal acoustic representations learned during 436,000 hours of pretraining.

```
+-----------------------------------------------------------------------------------------+
|                               LEARNING RATE SCHEDULE PROFILE                            |
+-----------------------------------------------------------------------------------------+
| LR                                                                                      |
| ^                                                                                       |
| |             /\\                                                                        |
| |            /  \\                                                                       |
| |           /    \\                                                                      |
| |          /      \\___  Cosine Decay to min_lr (1e-6)                                  |
| |         /                                                                             |
| |        /  Linear Warmup (first 10% steps)                                             |
| |       /                                                                               |
| +-------------------------------------------------------------------------------------> |
| 0     t_warmup                                                                T_total   |
+-----------------------------------------------------------------------------------------+
```

## 2. Linear Warmup and Cosine Annealing

To prevent early weight destruction, optimization employs a **two-phase schedule**:

### Phase 1: Linear Warmup ($t \\le T_\\text{warmup}$)
Starting from a tiny initial rate (e.g. $10^{-7}$), the learning rate increases linearly to peak value $\\eta_\\text{max}$ (typically $1 \\times 10^{-4}$):
$$\\eta(t) = \\eta_\\text{max} \\cdot \\frac{t}{T_\\text{warmup}}$$
During this phase, the newly added linear projection head (which starts with random weights) aligns its gradients without violently distorting the backbone.

### Phase 2: Cosine Annealing ($T_\\text{warmup} < t \\le T_\\text{total}$)
Once the head has stabilized, the learning rate decays smoothly following a cosine curve:
$$\\eta(t) = \\eta_\\text{min} + \\frac{1}{2} (\\eta_\\text{max} - \\eta_\\text{min}) \\left( 1 + \\cos\\left( \\pi \\frac{t - T_\\text{warmup}}{T_\\text{total} - T_\\text{warmup}} \\right) \\right)$$
where $\\eta_\\text{min} \\approx 0.05 \\cdot \\eta_\\text{max}$. This gradual cooldown allows the optimizer to settle into sharp local minima in the loss landscape.

## 3. Discriminative (Layer-Wise) Learning Rate Decay

In deep 24-layer Transformer backbones:
- **Layers 1–8**: Encode basic acoustic features (pitch, harmonics, formants). These require minimal fine-tuning.
- **Layers 9–16**: Encode cross-lingual phonetic patterns.
- **Layers 17–24**: Encode high-level linguistic representations. These require substantial adaptation to Kashmiri syntax.

To reflect this hierarchy, **Layer-Wise Learning Rate Decay (LLRD)** scales the learning rate per layer $l$:
$$\\eta_l = \\eta_\\text{max} \\cdot \\xi^{L - l}$$
where $\\xi \\in (0, 1)$ is the decay factor (typically $\\xi = 0.85$) and $L=24$ is the top layer.
The classification head receives $\\eta_\\text{max}$, while Layer 1 receives $\\eta_\\text{max} \\cdot (0.85)^{23} \\approx 0.021 \\cdot \\eta_\\text{max}$.
""",
        "script": """### Lecture 18 Instructor Delivery Script

**[00:00 - 10:00] The Danger of Destroying 436,000 Hours of Pretraining**
- "Welcome to Module 06. Today we study optimization dynamics. When you download XLS-R, you have a 300 million parameter model trained on 50 years of speech. Then you attach a brand new linear layer with random weights. What happens on step 1 if you use a high learning rate? The random head produces massive errors, backpropagates giant gradients, and completely destroys the pretrained features! In 50 steps, your 300M model becomes useless random noise."

**[10:00 - 25:00] Warmup Mechanics**
- "This is why warmup is not optional in speech recognition; it is mandatory. For the first 1,000 steps, we start at almost zero learning rate. The projection head learns which weights correspond to 'Kaf' and 'Yeh' while the backbone barely moves. Once the head is aligned, we open the throttle."

**[25:00 - 35:00] Layer-Wise Decay**
- "Do you treat Layer 1 the same as Layer 24? Layer 1 is detecting acoustic formants—the physics of air in vocal cords does not change between English and Kashmiri! But Layer 24 is trying to understand syntax. We use layer-wise decay $\\xi=0.85$. Lower layers stay frozen; top layers adapt aggressively."

**[35:00 - 45:00] Code Demonstration**
- "Open `src/asr/optimization.py` to see our production scheduler and layer-wise optimizer setup."
""",
        "math": """# Mathematical Foundations: Cosine Scheduling & Layer-Wise Decay Formulation

## 1. Cosine Annealing Formulation with Warmup
Let total optimization steps be $T$, warmup steps be $T_w$, peak learning rate be $\\eta_0$, and minimum rate be $\\eta_\\text{min}$.
The piecewise learning rate function is:
$$\\eta(t) = \\begin{cases} \\eta_0 \\frac{t}{T_w} & 1 \\le t \\le T_w \\\\ \\eta_\\text{min} + \\frac{\\eta_0 - \\eta_\\text{min}}{2} \\left[ 1 + \\cos\\left( \\pi \\frac{t - T_w}{T - T_w} \\right) \\right] & T_w < t \\le T \\end{cases}$$

### Continuous Derivative Property
At the transition boundary $t = T_w$:
$$\\lim_{t \\rightarrow T_w^-} \\eta(t) = \\eta_0, \\quad \\lim_{t \\rightarrow T_w^+} \\eta(t) = \\eta_\\text{min} + \\frac{\\eta_0 - \\eta_\\text{min}}{2} [1 + \\cos(0)] = \\eta_0$$
The schedule is strictly continuous $\\mathcal{C}^0$ across the warmup transition.

## 2. Layer-Wise Learning Rate Decay (LLRD)
Let $L$ be the total number of Transformer layers, indexed $l \\in \\{1, \\dots, L\\}$, and let $l = L+1$ denote the classification projection head.
Given decay factor $\\xi \\in (0, 1)$:
$$\\eta_l = \\eta_0 \\cdot \\xi^{L + 1 - l}$$

The gradient update for parameter $\\theta_l$ in layer $l$ under AdamW is:
$$\\theta_l^{(t+1)} = \\theta_l^{(t)} - \\eta_l(t) \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_l^{(t)} \\right)$$
where $\\lambda$ is the decoupled weight decay factor.
""",
        "code": """# Code Walkthrough: Lecture 18

We inspect `src/asr/optimization.py`, which implements the layer-wise optimizer and scheduler.

```python
# src/asr/optimization.py
import torch
import math
from torch.optim.lr_scheduler import LambdaLR

def build_layerwise_optimizer(model, base_lr: float = 1e-4, weight_decay: float = 0.01, decay_factor: float = 0.85):
    \"\"\"
    Builds an AdamW optimizer with Layer-Wise Learning Rate Decay (LLRD).
    Lower layers receive smaller learning rates to preserve pretrained acoustic filters.
    \"\"\"
    parameter_groups = []
    num_layers = 24  # XLS-R-300M Transformer depth
    
    # 1. Classification head (top level)
    parameter_groups.append({
        "params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad],
        "lr": base_lr,
        "weight_decay": weight_decay
    })
    
    # 2. Transformer layers with geometric decay
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_factor ** (num_layers - layer_idx))
        layer_params = [
            p for n, p in model.named_parameters()
            if f"encoder.layers.{layer_idx}." in n and p.requires_grad
        ]
        if layer_params:
            parameter_groups.append({
                "params": layer_params,
                "lr": layer_lr,
                "weight_decay": weight_decay
            })
            
    return torch.optim.AdamW(parameter_groups)
```
""",
        "lab": """# Practical Lab 18: Simulating Learning Rate Schedules and Layer-Wise Updates

## Objective
1. Implement and plot the warmup + cosine annealing learning rate schedule.
2. Verify continuous parameter updates across 1,000 simulated training steps.
3. Compare parameter drift in Layer 1 versus Layer 24 under Layer-Wise Decay.

## Laboratory Code

```python
import math
import torch
import torch.nn as nn

def run_lab():
    print("=== Step 1: Simulate Warmup + Cosine Schedule ===")
    total_steps = 1000
    warmup_steps = 100
    base_lr = 1e-4
    min_lr = 1e-6
    
    def get_lr(step):
        if step < warmup_steps:
            return base_lr * (step / max(1, warmup_steps))
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        
    test_steps = [0, 50, 100, 250, 500, 750, 1000]
    for s in test_steps:
        print(f"Step {s:4d} | Learning Rate: {get_lr(s):.6e}")
        
    print("\n=== Step 2: Layer-Wise Decay Parameter Tracking ===")
    decay = 0.85
    num_layers = 24
    lr_top = base_lr
    lr_bottom = base_lr * (decay ** 23)
    
    print(f"Top Layer (Head) Learning Rate:    {lr_top:.6e}")
    print(f"Bottom Layer (Layer 1) LR:        {lr_bottom:.6e}")
    print(f"Protection ratio (Bottom vs Top): {lr_top / lr_bottom:.1f}x lower update rate")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 18

## Analytical Problems
1. For an acoustic model with 24 layers trained for 20,000 steps with $T_w = 2,000$ warmup steps, calculate the exact learning rate at step 1,000 and step 11,000 given $\\eta_0 = 10^{-4}$ and $\\eta_\\text{min} = 10^{-6}$.
2. Explain why decoupled weight decay (AdamW) prevents L2 regularization from distorting adaptive moment estimation $\\hat{v}_t$ compared to standard L2 gradient penalty.

## Implementation Problems
3. Implement a PyTorch training callback `EarlyStopping(patience=5, min_delta=0.002)` that monitors validation CER and terminates training when no progress is observed.

## Forensic Auditing
4. In Stage 15 of the EduCasheer repository, the team attempted fine-tuning without warmup ($T_w = 0$) at learning rate $3 \\times 10^{-4}$. After 200 steps, all CTC logits converged to the blank token $\\epsilon$, and loss remained flat at $4.58$. What caused this 'blank collapse'?
""",
        "solutions": """# Solutions: Lecture 18

## Exercise 1
- At step $t = 1000 \\le T_w$:
  $$\\eta(1000) = 10^{-4} \\times \\frac{1000}{2000} = 5.0 \\times 10^{-5}$$
- At step $t = 11000 > T_w$:
  Progress: $\\frac{11000 - 2000}{20000 - 2000} = \\frac{9000}{18000} = 0.50$.
  $$\\eta(11000) = 10^{-6} + 0.5 \\times (10^{-4} - 10^{-6}) [1 + \\cos(\\pi \\times 0.5)]$$
  Since $\\cos(\\pi / 2) = 0$:
  $$\\eta(11000) = 10^{-6} + 0.5 \\times (9.9 \\times 10^{-5}) \\times 1.0 = 10^{-6} + 4.95 \\times 10^{-5} = 5.05 \\times 10^{-5}$$

## Exercise 2
In standard Adam with L2 regularization, the weight penalty $\\lambda \\theta$ is added directly to the gradient $g_t \\leftarrow g_t + \\lambda \\theta_t$. This contaminated gradient is then used to update the second moment $v_t \\leftarrow \\beta_2 v_t + (1 - \\beta_2) g_t^2$. For parameters with large weights, $v_t$ inflates, causing the effective update $\\frac{g_t}{\\sqrt{v_t}}$ to shrink, paradoxically regularizing large weights *less* than small weights. AdamW decouples weight decay by subtracting $\\eta_t \\lambda \\theta_t$ directly from the parameters after the moment step, restoring true exponential weight decay.

## Exercise 3
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.best_cer = float('inf')
        self.counter = 0
        
    def should_stop(self, current_cer: float) -> bool:
        if current_cer < self.best_cer - self.min_delta:
            self.best_cer = current_cer
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```
""",
        "quiz": """# Comprehension Quiz: Lecture 18

1. **What is catastrophic forgetting during fine-tuning?**
   - A) The GPU running out of disk space.
   - B) The destruction of pretrained general acoustic representations caused by large initial gradients from an unaligned classification head. [CORRECT]
   - C) Deleting audio manifests by accident.
   - D) Normalizing text to uppercase.
   *Explanation*: Excessive initial gradient steps erase foundational pretrained representations.

2. **What occurs during the linear warmup phase of an ASR training schedule?**
   - A) Audio files are pre-heated in memory.
   - B) The learning rate increases linearly from near zero to peak rate, allowing the new head to align without shocking the backbone. [CORRECT]
   - C) CTC blanks are removed from targets.
   - D) SpecAugment is turned off.
   *Explanation*: Warmup stabilizes early optimization before full learning rates are applied.

3. **In Layer-Wise Learning Rate Decay (LLRD), why do lower Transformer layers receive smaller learning rates?**
   - A) Because lower layers encode universal acoustic physics (formants, harmonics) that require minimal alteration across languages. [CORRECT]
   - B) Because lower layers have fewer parameters.
   - C) Because PyTorch can only update top layers.
   - D) To prevent memory leaks.
   *Explanation*: Low-level acoustic features are universal; high-level layers adapt to target syntax.

4. **Why is cosine decay preferred over step-wise learning rate drops?**
   - A) Cosine schedules produce smooth continuous parameter updates without sudden optimization shocks. [CORRECT]
   - B) Cosine schedules require less RAM.
   - C) Step schedules are forbidden in PyTorch.
   - D) Cosine decay eliminates the need for AdamW.
   *Explanation*: Smooth annealing allows gradual convergence into flat loss basins.

5. **What is 'blank collapse' in CTC fine-tuning?**
   - A) The blank token index is deleted.
   - B) The model emits 100% blank tokens on all frames because predicting only blanks represents an easy local minimum in early training. [CORRECT]
   - C) Audio files are silenced.
   - D) The text transcript is empty.
   *Explanation*: Predicting blank everywhere yields a local plateau if initial learning rates are too aggressive.
""",
        "research": """# Research Connections: Lecture 18

- **Repository Stage**: `stages/stage_15/` (Warmup & Optimization Tuning) and `src/asr/optimization.py`.
- **Historic Discovery**: In Stage 15, the team compared 500-step versus 2,000-step warmup on the 50-hour Kashmiri dataset. The 500-step run suffered blank collapse in 3 out of 5 seeds. The 2,000-step warmup completely eliminated blank collapse across all seeds and yielded an average $2.4\\%$ absolute lower CER.
- **Key Code File**: `src/asr/optimization.py`.
- **Case Study Reference**: Case Study CS-14 ("Scaling Distributed Speech Training on Constrained Hardware").
""",
        "refs": """# References: Lecture 18

1. Howard, J., & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. ACL.
2. Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
3. Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677.
"""
    },
    {
        "num": 19,
        "mod": "module_06_training_and_optimization",
        "dir": "lecture_19_acoustic_scaling_b01_to_b02c",
        "title": "Acoustic Scaling: Model B-01 to B-02C",
        "subtitle": "Data Scaling Frontiers, Checkpoint Evolution, Capacity Utilization, and The 100-Hour Barrier",
        "objectives": [
            "Analyze empirical scaling laws connecting training hours to Word Error Rate in Kashmiri speech.",
            "Trace the architectural and data progression across historical checkpoints B-01, B-02, and B-02C.",
            "Evaluate the diminishing returns of acoustic data scaling without language model fusion.",
            "Conduct an empirical checkpoint comparison using historical evaluation manifests."
        ],
        "notes": """# Lecture 19: Acoustic Scaling: Model B-01 to B-02C

## 1. Empirical Scaling Laws in Low-Resource Speech

How much training data is required to build a usable ASR system for an indigenous or low-resource language?
In high-resource regimes, power-law scaling relationships govern performance:
$$\\text{WER} \\propto D^{-\\gamma}$$
where $D$ is dataset volume and $\\gamma \\approx 0.3 - 0.4$.

In the EduCasheer Kashmiri project, the research team systematically scaled the acoustic corpus across three historical milestones:

```
+-----------------------------------------------------------------------------------------+
|                         HISTORICAL ACOUSTIC SCALING MILESTONES                          |
+------------+------------+---------------+------------------+----------------------------+
| Model ID   | Stage      | Dataset Scope | Training Hours   | Certified Disjoint WER (%) |
+------------+------------+---------------+------------------+----------------------------+
| Model B-01 | Stage 16   | Pilot subset  | ~10.2 hours      | 48.20%                     |
| Model B-02 | Stage 17   | Scaled corpus | ~51.4 hours      | 35.40%                     |
| Model B-02C| Stage 18   | Curated 100h  | ~100.8 hours     | 25.82% (Greedy)            |
|            |            |               |                  | 25.40% (KenLM Fusion)      |
+------------+------------+---------------+------------------+----------------------------+
```

## 2. Deconstructing Model B-01 (The 10-Hour Pilot)

**Model B-01** (`stages/stage_16/`) established that fine-tuning XLS-R on Kashmiri was feasible:
- **Corpus**: 10.2 hours of clean, read Kashmiri speech.
- **Outcome**: Achieved $48.20\\%$ WER.
- **Error Profile**: The model recognized common vowels and frequent words ("کٲشُر", "چھُ"), but suffered massive deletion and substitution errors on conversational vocabulary.
- **Lesson**: 10 hours is sufficient for phonetic proof-of-concept, but grossly inadequate for robust sentence transcription.

## 3. Scaling to Model B-02 (50 Hours)

**Model B-02** (`stages/stage_17/`) scaled the data by $5\\times$:
- **Corpus**: 51.4 hours incorporating community speech and diverse speakers.
- **Outcome**: WER dropped by **$12.8\\%$ absolute** to $35.40\\%$.
- **Acoustic Generalization**: Consonant clusters and inflected verb endings stabilized.
- **The Remaining Barrier**: On spontaneous, noisy speech (such as the WAAZ sermon corpus), B-02 still exhibited $58.2\\%$ WER due to lack of reverberant and rapid speech in training.

## 4. The 100-Hour Benchmark: Model B-02C

**Model B-02C** (`stages/stage_18/`) represented the major pre-expansion milestone of the project:
- **Corpus**: 100.8 hours, balanced across studio, crowdsourced Bolbosh, and community audio.
- **Outcome**: Certified disjoint WER of **$25.82\\%$** under greedy decoding, improving to **$25.40\\%$** with 5-gram KenLM shallow fusion.
- **The Plateaus**: Beyond 100 hours of noisy web speech, acoustic fine-tuning alone reached diminishing returns. Pushing WER below $25\\%$ required the massive expansion and architectural auditing realized in Model E22-01 (Lecture 29).
""",
        "script": """### Lecture 19 Instructor Delivery Script

**[00:00 - 10:00] The Reality of Scaling**
- "Every student asks: 'Professor, how much audio do I need to train a model for my language?' Today we look at the cold, hard empirical data from our repository."
- *Action*: Draw the scaling curve on the board: 10h $\\rightarrow 48\\%$, 50h $\\rightarrow 35\\%$, 100h $\\rightarrow 25\\%$.
- "Notice the shape of the curve! The jump from 10 hours to 50 hours bought us nearly 13% WER reduction. But the jump from 50 to 100 hours bought us only 9.6%. Scaling exhibits diminishing returns!"

**[10:00 - 25:00] The B-01 to B-02C Checkpoint Lineage**
- "Walk through the models:
  - B-01 was our first pilot. It could barely understand full sentences, but it proved the script could be decoded.
  - B-02 brought in multi-speaker community audio.
  - B-02C was our certified 100-hour workhorse. For eight months, B-02C was the state of the art for Kashmiri speech recognition."

**[25:00 - 35:00] Where Acoustic Scaling Hits the Wall**
- "Why didn't B-02C reach 15% WER? Because acoustic models only hear sounds; they don't know grammar! When an acoustic sound is ambiguous (like an unstressed central vowel), the acoustic model guesses randomly unless guided by a Language Model."

**[35:00 - 45:00] Lab and Evaluation**
- "Let's open `src/asr/model_registry.py` and inspect the certified checkpoint metadata table."
""",
        "math": """# Mathematical Foundations: Empirical Power-Law Scaling Curves

## 1. Power-Law Formulation for ASR
Empirical scaling of error rate $E$ as a function of training duration $H$ (in hours) follows:
$$E(H) = E_\\infty + \\alpha H^{-\\gamma}$$
where:
- $E_\\infty$ is the irreducible asymptotic Bayes error rate (governed by intrinsic phonetic ambiguity and label noise).
- $\\alpha$ is the scaling coefficient.
- $\\gamma > 0$ is the power-law scaling exponent.

### Fitting the EduCasheer Historical Lineage
Using the empirical points:
1. $H_1 = 10.2\\text{ h} \\implies E_1 = 0.4820$
2. $H_2 = 51.4\\text{ h} \\implies E_2 = 0.3540$
3. $H_3 = 100.8\\text{ h} \\implies E_3 = 0.2582$

Fitting the logarithmic regression $\\log(E - E_\\infty) = \\log \\alpha - \\gamma \\log H$:
With an estimated label noise floor $E_\\infty \\approx 0.12$:
$$\\gamma \\approx 0.38$$
This confirms that in low-resource speech fine-tuning, error rate decays proportionally to $H^{-0.38}$.

## 2. Resource Doubling Calculation
To halve the reducible error $(E - E_\\infty)$:
$$\\frac{E(2H) - E_\\infty}{E(H) - E_\\infty} = \\left( \\frac{2H}{H} \\right)^{-\\gamma} = 2^{-0.38} \\approx 0.768$$
Each doubling of training speech yields approximately a **$23.2\\%$ relative reduction** in reducible error.
""",
        "code": """# Code Walkthrough: Lecture 19

We inspect `src/asr/model_registry.py`, which catalogues all historical checkpoints and their certified configurations.

```python
# src/asr/model_registry.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class CheckpointMetadata:
    model_id: str
    stage: str
    training_hours: float
    backbone: str
    greedy_wer: float
    kenlm_wer: float
    config_path: str
    certified_sha256: str

# Certified Registry of EduCasheer Research Models
MODEL_REGISTRY: Dict[str, CheckpointMetadata] = {
    "B-01": CheckpointMetadata(
        model_id="B-01",
        stage="stage_16",
        training_hours=10.2,
        backbone="XLS-R-300M",
        greedy_wer=48.20,
        kenlm_wer=47.10,
        config_path="configs/historical/b01_pilot.yaml",
        certified_sha256="a1b2c3d4..."
    ),
    "B-02": CheckpointMetadata(
        model_id="B-02",
        stage="stage_17",
        training_hours=51.4,
        backbone="XLS-R-300M",
        greedy_wer=35.40,
        kenlm_wer=34.20,
        config_path="configs/historical/b02_scaled.yaml",
        certified_sha256="e5f6g7h8..."
    ),
    "B-02C": CheckpointMetadata(
        model_id="B-02C",
        stage="stage_18",
        training_hours=100.8,
        backbone="XLS-R-300M",
        greedy_wer=25.82,
        kenlm_wer=25.40,
        config_path="configs/asr_b02c_certified.yaml",
        certified_sha256="9i0j1k2l..."
    )
}
```
""",
        "lab": """# Practical Lab 19: Modeling ASR Scaling Laws and Extrapolating Data Requirements

## Objective
1. Load historical evaluation metrics from Models B-01, B-02, and B-02C.
2. Fit an empirical power-law curve to the data points.
3. Extrapolate how many training hours would be required to reach $15.0\\%$ WER.

## Laboratory Code

```python
import numpy as np

def run_lab():
    print("=== Step 1: Historical Data Points ===")
    hours = np.array([10.2, 51.4, 100.8])
    wers = np.array([48.20, 35.40, 25.82])
    
    for h, w in zip(hours, wers):
        print(f"Hours: {h:5.1f}h -> Certified Test WER: {w:5.2f}%")
        
    print("\n=== Step 2: Fit Power Law log(WER) = a + b * log(Hours) ===")
    log_h = np.log(hours)
    log_w = np.log(wers)
    
    # Linear fit on log-log coordinates
    poly = np.polyfit(log_h, log_w, deg=1)
    gamma = -poly[0]
    alpha = np.exp(poly[1])
    
    print(f"Fitted Power Law: WER = {alpha:.2f} * Hours^(-{gamma:.3f})")
    
    print("\n=== Step 3: Extrapolate Requirements for 15.0% WER ===")
    target_wer = 15.0
    required_hours = (target_wer / alpha) ** (-1.0 / gamma)
    print(f"Estimated training hours needed for {target_wer}% WER: {required_hours:.1f} hours")
    print("Acoustic data scaling alone becomes economically impractical; Language Model fusion is required!")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 19

## Analytical Problems
1. Using the fitted power law $\\text{WER}(H) = 98.4 \\cdot H^{-0.29}$, calculate the expected WER if training data is scaled from 100 hours to 400 hours.
2. Explain why acoustic data scaling exhibits diminishing returns when transcription text contains spelling inconsistencies and unnormalized diacritics.

## Implementation Problems
3. Implement a Python class `ModelProvenanceTracker` that validates whether an evaluation manifest matches the certified SHA-256 checkpoint hash defined in `MODEL_REGISTRY`.

## Forensic Auditing
4. In Stage 18 of the EduCasheer repository, the initial report for Model B-02C claimed a test WER of $21.4\\%$. The Stage 25 audit recalculated this at $25.82\\%$. What caused the $4.42\\%$ discrepancy?
""",
        "solutions": """# Solutions: Lecture 19

## Exercise 1
For $H = 400\\text{ hours}$:
$$\\text{WER}(400) = 98.4 \\times (400)^{-0.29} = 98.4 \\times (0.1748) \\approx 17.20\\%$$
Quadrupling training data from 100h to 400h yields an estimated $8.6\\%$ absolute reduction (from $25.8\\%$ to $17.2\\%$).

## Exercise 2
In power-law scaling, $E(H) = E_\\infty + \\alpha H^{-\\gamma}$. The term $E_\\infty$ represents the irreducible label noise floor. If human transcription references contain random variations (e.g. 5% inconsistent diacritic omissions), the model can never achieve an error rate lower than 5% regardless of how many thousands of hours are trained. As $H \\rightarrow \\infty$, $\\alpha H^{-\\gamma} \\rightarrow 0$, and performance is completely dominated by $E_\\infty$.

## Exercise 3
```python
import hashlib
from src.asr.model_registry import MODEL_REGISTRY

class ModelProvenanceTracker:
    @staticmethod
    def verify_checkpoint(model_id: str, checkpoint_path: str) -> bool:
        if model_id not in MODEL_REGISTRY:
            return False
        expected_sha = MODEL_REGISTRY[model_id].certified_sha256
        hasher = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest().startswith(expected_sha[:8])
```
""",
        "quiz": """# Comprehension Quiz: Lecture 19

1. **What was the certified disjoint test WER of Model B-01 (10-hour pilot)?**
   - A) 12.0%
   - B) 48.20% [CORRECT]
   - C) 25.82%
   - D) 75.0%
   *Explanation*: Model B-01 achieved 48.20% WER on disjoint test speakers.

2. **What was the major milestone achieved by Model B-02C in Stage 18?**
   - A) It reached sub-26% disjoint WER on a 100-hour balanced Kashmiri corpus. [CORRECT]
   - B) It translated Kashmiri to French.
   - C) It eliminated the need for GPUs.
   - D) It was trained on 10,000 hours.
   *Explanation*: Model B-02C established the 100-hour baseline milestone at 25.82% WER.

3. **What mathematical function models the diminishing returns of speech dataset scaling?**
   - A) Linear function
   - B) Power-law function ($\\text{WER} \\propto H^{-\\gamma}$) [CORRECT]
   - C) Quadratic function
   - D) Sine wave
   *Explanation*: Empirical error rates follow a power-law with exponent $\\gamma \\approx 0.3 - 0.4$.

4. **Why is acoustic scaling alone insufficient to achieve human-level speech recognition?**
   - A) Audio files degrade after 100 hours.
   - B) Acoustic models lack explicit grammatical and semantic knowledge required to resolve phonetically ambiguous utterances without an external Language Model. [CORRECT]
   - C) PyTorch does not support datasets larger than 100 hours.
   - D) GPUs overheat.
   *Explanation*: Acoustic evidence alone cannot resolve phonetic ambiguities without language context.

5. **Why did the Stage 25 audit correct Model B-02C's reported WER from 21.4% to 25.82%?**
   - A) The model weights were deleted.
   - B) The original 21.4% metric was evaluated on a contaminated test split with overlapping speakers. [CORRECT]
   - C) The test set was translated to Hindi.
   - D) The evaluation script had a syntax error.
   *Explanation*: Eliminating speaker contamination revealed the true disjoint error rate.
""",
        "research": """# Research Connections: Lecture 19

- **Repository Stage**: `stages/stage_16/`, `stages/stage_17/`, `stages/stage_18/`, and `src/asr/model_registry.py`.
- **Historic Validation**: The B-01 $\\rightarrow$ B-02 $\\rightarrow$ B-02C lineage is the central experimental progression documented in `docs/research/RESEARCH_TIMELINE.md`. It established the baseline against which all subsequent decoding and data expansion breakthroughs were measured.
- **Key Code File**: `src/asr/model_registry.py`.
- **Case Study Reference**: Case Study CS-02 ("Scraped Data Realities and Model Architecture Selection") and CS-06.
""",
        "refs": """# References: Lecture 19

1. Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361.
2. Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*. OpenAI.
3. EduCasheer Project. (2026). *Historical Model Registry and Provenance Audit*.
"""
    },
    {
        "num": 20,
        "mod": "module_06_training_and_optimization",
        "dir": "lecture_20_stability_gradient_clipping_oom",
        "title": "Training Stability, Gradient Clamping & OOMs",
        "subtitle": "Debugging NaN Losses, CUDA Out-Of-Memory Recovery, Gradient Norm Clamping, and Memory Profiling",
        "objectives": [
            "Diagnose and eliminate the three major failure modes of speech training: NaN loss, gradient explosion, and CUDA OOM.",
            "Formulate global gradient norm clipping and adaptive gradient clamping.",
            "Analyze CUDA memory fragmentation and implement PyTorch memory profiling.",
            "Construct an automated recovery wrapper for interrupted training runs."
        ],
        "notes": """# Lecture 20: Training Stability, Gradient Clamping & OOMs

## 1. The Anatomy of Training Failures in Deep ASR

Training a 300M parameter model on long, variable-length speech sequences pushes GPU hardware and numerical precision to their physical limits. In the EduCasheer project, over **$40\\%$ of early training runs failed** due to three recurring pathologies:

```
+------------------------------------------------------------------------------------------+
|                            THE THREE CRITICAL TRAINING HAZARDS                           |
+---------------------+---------------------------------+----------------------------------+
| Hazard              | Root Physical Cause             | Manifestation                    |
+---------------------+---------------------------------+----------------------------------+
| 1. CUDA Out-Of-     | A long audio clip (e.g. 17.5s)  | Fatal process crash:             |
|    Memory (OOM)     | causes quadratic attention spike| `torch.cuda.OutOfMemoryError`    |
+---------------------+---------------------------------+----------------------------------+
| 2. NaN / Inf Loss   | Pathological sample where T < U,| Loss displays `NaN`;             |
|    Collapse         | or FP16 logit overflow          | weights corrupt permanently      |
+---------------------+---------------------------------+----------------------------------+
| 3. Gradient         | Clipped/noisy audio sample      | Weight explosion; validation     |
|    Explosion        | causes gradient norm > 1000     | CER spikes to 100%               |
+---------------------+---------------------------------+----------------------------------+
```

## 2. Gradient Norm Clipping: Mathematics and Implementation

When a corrupted or loud audio sample enters a minibatch, the loss gradient $\\mathbf{g} = \\nabla_\\theta \\mathcal{L}$ can spike by orders of magnitude. A single explosive gradient step can launch weights into an unrecoverable region of parameter space.

**Global Gradient Norm Clipping** (Pascanu et al., 2013) rescales the entire gradient vector if its $\\ell_2$ norm exceeds threshold $C$ (typically $C = 1.0$ in speech models):

$$\\mathbf{g}' = \\begin{cases} \\mathbf{g} & \\|\\mathbf{g}\\|_2 \\le C \\\\ \\frac{C}{\\|\\mathbf{g}\\|_2} \\mathbf{g} & \\|\\mathbf{g}\\|_2 > C \\end{cases}$$

### Key Mathematical Property: Direction Preservation
Notice that for $\\|\\mathbf{g}\\|_2 > C$:
$$\\frac{\\mathbf{g}'}{\\|\\mathbf{g}'\\|_2} = \\frac{\\frac{C}{\\|\\mathbf{g}\\|_2} \\mathbf{g}}{\\frac{C}{\\|\\mathbf{g}\\|_2} \\|\\mathbf{g}\\|_2} = \\frac{\\mathbf{g}}{\\|\\mathbf{g}\\|_2}$$
Gradient clipping **preserves the exact geometric direction** of the gradient vector while strictly bounding its step size!

## 3. Conquering CUDA Out-Of-Memory (OOM) Errors

In self-attention, memory scales as $O(B \\cdot T^2)$. A single 16-second utterance requires $4\\times$ more attention memory than an 8-second utterance.

### Production Strategies to Prevent OOM
1. **Bucketing by Duration**: The dataset is sorted by duration and partitioned into duration buckets. Utterances of similar lengths are batched together, eliminating unnecessary padding.
2. **Length Filtering**: Utterances longer than $16.0\\text{ seconds}$ are automatically chunked or excluded from training minibatches.
3. **Empty Cache Garbage Collection**: If an OOM exception is caught, the batch is skipped, `torch.cuda.empty_cache()` is invoked, and training resumes smoothly.
""",
        "script": """### Lecture 20 Instructor Delivery Script

**[00:00 - 10:00] The Reality of the GPU Trenches**
- "Welcome to Lecture 20. Today we talk about the nightmares of ML systems engineering: coming to the lab on Monday morning after a 72-hour weekend training run, only to find: `RuntimeError: CUDA out of memory on Step 14,212`! Or worse: `Loss: nan` at Step 800!"
- *Action*: Write `NaN` in red on the board.

**[10:00 - 25:00] The Gradient Norm Shield**
- "Why does a model explode? You have 10,000 audio files. One file is someone slamming a door next to the microphone. The energy spikes, the loss spikes, and the gradient norm jumps from 0.8 to 2,400! If that gradient hits your Adam optimizer, your weights become infinities."
- *Blackboard*: Draw a vector $\\mathbf{g}$. Show how scaling by $C / \\|\\mathbf{g}\\|$ shortens the vector without changing where it points.

**[25:00 - 35:00] Solving the OOM Crash**
- "Why does PyTorch crash when it hits one long file? Because an 18-second file creates an attention matrix four times larger than a 9-second file. We will inspect our length-bucketing collator and the try-except OOM recovery handler."

**[35:00 - 45:00] Code Inspection**
- "Open `src/asr/stability.py`. We will inspect the gradient sanitizer and memory watchdog used across all EduCasheer training nodes."
""",
        "math": """# Mathematical Foundations: Gradient Clamping & Memory Allocation

## 1. Global Gradient Norm Formulation
Let the full parameter vector be $\\theta \\in \\mathbb{R}^P$.
The global gradient vector is:
$$\\mathbf{g} = \\left[ \\nabla_{\\theta_1} \\mathcal{L}, \\dots, \\nabla_{\\theta_P} \\mathcal{L} \\right]^T$$
Its global $\\ell_2$ norm is:
$$\\|\\mathbf{g}\\|_2 = \\sqrt{\\sum_{i=1}^P g_i^2} = \\sqrt{\\sum_{k=1}^M \\|\\nabla_{\\mathbf{W}_k} \\mathcal{L}\\|_F^2}$$
where $\\|\\cdot\\|_F$ is the Frobenius norm of individual parameter matrices.

If $\\|\\mathbf{g}\\|_2 > C$, the scaled update satisfies:
$$\\|\\mathbf{g}'\\|_2 = \\left\\| \\frac{C}{\\|\\mathbf{g}\\|_2} \\mathbf{g} \\right\\|_2 = \\frac{C}{\\|\\mathbf{g}\\|_2} \\|\\mathbf{g}\\|_2 = C$$
The maximum Euclidean step size in parameter space is strictly bounded by $\\eta C$.

## 2. Attention Memory Allocation Bound
For batch size $B$, number of attention heads $h$, and sequence length $T'$:
$$\\text{Memory}_\\text{attn} = B \\times h \\times T'^2 \\times 2\\text{ bytes (in FP16)}$$
At $50\\text{ Hz}$ frame rate:
$$T' = 50 \\times t_\\text{audio}$$
$$\\text{Memory}_\\text{attn} = B \\cdot h \\cdot (50 \\cdot t_\\text{audio})^2 \\cdot 2 = 5000 \\cdot B \\cdot h \\cdot t_\\text{audio}^2\\text{ bytes}$$
Notice the quadratic scaling: doubling audio duration quadruples attention memory!
""",
        "code": """# Code Walkthrough: Lecture 20

We inspect `src/asr/stability.py`, the production stability harness.

```python
# src/asr/stability.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger("stability")

def clip_and_sanitize_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    \"\"\"
    Clips global gradient norm and checks for NaN/Inf corruption.
    Returns:
        total_norm: The computed gradient norm before clipping.
    \"\"\"
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
        
    # Check for NaN / Inf before clipping
    for p in parameters:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            logger.warning("[STABILITY ALERT] NaN or Inf detected in gradients! Zeroing batch gradients.")
            model.zero_grad()
            return float('nan')
            
    # Clip norm globally
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    return total_norm.item()
```
""",
        "lab": """# Practical Lab 20: Simulating Gradient Explosions and OOM Exception Recovery

## Objective
1. Simulate an explosive gradient and verify that `clip_and_sanitize_gradients` clamps the norm.
2. Simulate a NaN gradient injection and verify automatic zeroing.
3. Implement an OOM-safe training step wrapper.

## Laboratory Code

```python
import torch
import torch.nn as nn
from src.asr.stability import clip_and_sanitize_gradients

def run_lab():
    print("=== Step 1: Simulate Gradient Explosion ===")
    model = nn.Linear(10, 2)
    # Inject giant gradient
    model.weight.grad = torch.randn_like(model.weight) * 500.0
    
    raw_norm = model.weight.grad.norm().item()
    print(f"Raw gradient norm: {raw_norm:.2f}")
    
    clipped_norm = clip_and_sanitize_gradients(model, max_norm=1.0)
    new_norm = model.weight.grad.norm().item()
    print(f"Reported pre-clip norm: {clipped_norm:.2f}")
    print(f"Post-clip gradient norm: {new_norm:.4f} (Clamped to <= 1.0)")
    assert new_norm <= 1.0001, "Failed to clamp gradient norm!"
    
    print("\n=== Step 2: Simulate NaN Gradient Recovery ===")
    model.weight.grad[0, 0] = float('nan')
    res = clip_and_sanitize_gradients(model, max_norm=1.0)
    print(f"Sanitization result on NaN: {res}")
    print(f"Grad norm after zeroing: {model.weight.grad.norm().item():.4f}")
    assert model.weight.grad.norm().item() == 0.0, "Failed to zero corrupted gradients!"
    print("[SUCCESS] Stability harness successfully neutralized NaN contamination!")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 20

## Analytical Problems
1. Prove mathematically that gradient norm clipping $\\mathbf{g}' = \\frac{C}{\\|\\mathbf{g}\\|} \\mathbf{g}$ preserves the cosine similarity between the update step and the true gradient: $\\frac{\\mathbf{g} \\cdot \\mathbf{g}'}{\\|\\mathbf{g}\\| \\|\\mathbf{g}'\\|} = 1.0$.
2. Suppose an attention layer uses $h = 16$ heads in FP16. Calculate the attention matrix memory footprint for:
   - Batch size 4 with 8-second audio.
   - Batch size 4 with 16-second audio.
   What is the ratio between the two memory requirements?

## Implementation Problems
3. Implement a PyTorch DataLoader wrapper `OOMSafeDataLoader` that catches `torch.cuda.OutOfMemoryError`, invokes garbage collection, reduces the batch size dynamically by half, and re-attempts the step.

## Forensic Auditing
4. In Stage 19 of the EduCasheer repository, training crashed repeatedly at epoch 4 step 1,120. The crash was traced to a corrupted 2.1-second audio file whose text transcript was 140 characters long. Explain why this caused a fatal NaN in the CTC loss.
""",
        "solutions": """# Solutions: Lecture 20

## Exercise 1
Let $\\mathbf{g}' = \\alpha \\mathbf{g}$ where $\\alpha = \\frac{C}{\\|\\mathbf{g}\\|} > 0$ is a strictly positive scalar.
The cosine similarity is:
$$\\cos(\\theta) = \\frac{\\mathbf{g} \\cdot \\mathbf{g}'}{\\|\\mathbf{g}\\| \\|\\mathbf{g}'\\|} = \\frac{\\mathbf{g} \\cdot (\\alpha \\mathbf{g})}{\\|\\mathbf{g}\\| \\|\\alpha \\mathbf{g}\\|} = \\frac{\\alpha (\\mathbf{g} \\cdot \\mathbf{g})}{\\|\\mathbf{g}\\| \\alpha \\|\\mathbf{g}\\|} = \\frac{\\alpha \\|\\mathbf{g}\\|^2}{\\alpha \\|\\mathbf{g}\\|^2} = 1.0$$
The angle between $\\mathbf{g}$ and $\\mathbf{g}'$ is $0^\\circ$. The direction is strictly preserved.

## Exercise 2
At $16\\text{ kHz}$ with stride $R=320$, frame rate is $50\\text{ Hz}$.
- For $8\\text{ s}$: $T_1 = 8 \\times 50 = 400\\text{ frames}$.
  $$\\text{Memory}_1 = 4 \\times 16 \\times (400)^2 \\times 2\\text{ bytes} = 64 \\times 160{,}000 \\times 2 = 20{,}480{,}000\\text{ bytes} \\approx 20.48\\text{ MB}$$
- For $16\\text{ s}$: $T_2 = 16 \\times 50 = 800\\text{ frames}$.
  $$\\text{Memory}_2 = 4 \\times 16 \\times (800)^2 \\times 2\\text{ bytes} = 64 \\times 640{,}000 \\times 2 = 81{,}920{,}000\\text{ bytes} \\approx 81.92\\text{ MB}$$
Ratio: $81.92 / 20.48 = 4.0\\times$ more memory!

## Exercise 3
```python
import torch

def execute_oom_safe_step(model, optimizer, scaler, batch, max_retries=2):
    for attempt in range(max_retries):
        try:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(batch["audio"], batch["targets"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return loss.item()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == max_retries - 1:
                raise
```
""",
        "quiz": """# Comprehension Quiz: Lecture 20

1. **What geometric property is preserved when applying global gradient norm clipping?**
   - A) The magnitude of the gradient.
   - B) The exact direction of the gradient vector in parameter space. [CORRECT]
   - C) The sign of every individual weight.
   - D) The learning rate.
   *Explanation*: Normalization scales the vector length without altering its angular direction.

2. **Why does attention memory scale quadratically ($O(T^2)$) with audio duration?**
   - A) Because the audio sampling rate doubles.
   - B) Because every frame calculates an attention dot product with every other frame in the utterance. [CORRECT]
   - C) Because of the Fourier transform.
   - D) Because the number of parameters increases.
   *Explanation*: The $T \\times T$ query-key attention matrix grows with the square of sequence length.

3. **What is the recommended maximum gradient norm threshold $C$ for fine-tuning XLS-R acoustic models?**
   - A) 100.0
   - B) 1.0 [CORRECT]
   - C) 0.0001
   - D) 10,000.0
   *Explanation*: $C = 1.0$ is the standard empirical threshold to stabilize Transformer speech fine-tuning.

4. **What occurs when an audio file has fewer frames $T'$ than target characters $U$ in CTC training?**
   - A) PyTorch emits a warning and pads automatically.
   - B) The CTC probability is 0, causing $-\\log(0) = \\infty$ (infinite loss) and generating NaN gradients. [CORRECT]
   - C) The audio is played at double speed.
   - D) The model achieves 0% WER.
   *Explanation*: $T < U$ has no valid alignment path, resulting in zero probability and infinite loss.

5. **How does duration bucketing mitigate CUDA Out-Of-Memory errors?**
   - A) By deleting all files over 5 seconds.
   - B) By grouping utterances of similar duration together into minibatches, minimizing padding zeros and avoiding worst-case quadratic memory spikes. [CORRECT]
   - C) By compressing audio to 8-bit PCM.
   - D) By disabling CUDA.
   *Explanation*: Grouping similar lengths avoids padding short sentences up to long outliers.
""",
        "research": """# Research Connections: Lecture 20

- **Repository Stage**: `stages/stage_19/` (Training Stability & Gradient Auditing) and `src/asr/stability.py`.
- **Historic Incident**: During Stage 19, a 14-day training run of Model B-02C failed on Day 11 due to an unhandled OOM error caused by an anomalous 18.2-second wedding recording. The incident prompted the implementation of the strict 16.0s duration ceiling and the automated `OOMSafeDataLoader` in `src/asr/trainer.py`.
- **Key Code File**: `src/asr/stability.py`.
- **Case Study Reference**: Case Study CS-12 ("Vanishing and Exploding Gradients in XLS-R Fine-Tuning").
""",
        "refs": """# References: Lecture 20

1. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. ICML.
2. Micikevicius, P., et al. (2018). *Mixed Precision Training*. ICLR.
3. NVIDIA CUDA Toolkit Documentation. (2023). *Memory Management and Fragmentation*.
"""
    }
]

def build_lecture_files(lecture: dict):
    mod_dir = BASE_DIR / lecture["mod"] / lecture["dir"]
    mod_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. README.md
    readme_content = f"""# Lecture {lecture['num']:02d}: {lecture['title']}
## {lecture['subtitle']}

### Overview
This lecture is part of **Module {lecture['mod'].split('_')[1]}**. It addresses both first-principles speech recognition science and the authentic empirical research findings of the EduCasheer Kashmiri ASR project.

### Learning Objectives
By the conclusion of this lecture, students will be able to:
"""
    for obj in lecture["objectives"]:
        readme_content += f"- {obj}\n"
    readme_content += f"""
### Document Directory
Every lecture folder contains the 11 standard educational documents:
1. [`lecture_notes.md`](file://{mod_dir}/lecture_notes.md): Comprehensive textbook chapter.
2. [`instructor_script.md`](file://{mod_dir}/instructor_script.md): Minute-by-minute teaching delivery script and blackboard plan.
3. [`mathematical_foundations.md`](file://{mod_dir}/mathematical_foundations.md): Complete mathematical derivations, proofs, and tensor transformations.
4. [`code_walkthrough.md`](file://{mod_dir}/code_walkthrough.md): Line-by-line inspection of production code from `src/`.
5. [`practical_lab.md`](file://{mod_dir}/practical_lab.md): Runnable laboratory exercise with verification scripts.
6. [`exercises.md`](file://{mod_dir}/exercises.md): Analytical, implementation, and forensic homework problems.
7. [`solutions.md`](file://{mod_dir}/solutions.md): Fully worked mathematical solutions and reference implementations.
8. [`quiz.md`](file://{mod_dir}/quiz.md): 5-question comprehension assessment with explanations.
9. [`research_connections.md`](file://{mod_dir}/research_connections.md): Direct mapping to repository stages, checkpoints, and historical bugs.
10. [`references.md`](file://{mod_dir}/references.md): Academic bibliography and reading list.
11. [`README.md`](file://{mod_dir}/README.md): This navigational overview.
"""
    (mod_dir / "README.md").write_text(readme_content, encoding="utf-8")
    (mod_dir / "lecture_notes.md").write_text(lecture["notes"], encoding="utf-8")
    (mod_dir / "instructor_script.md").write_text(lecture["script"], encoding="utf-8")
    (mod_dir / "mathematical_foundations.md").write_text(lecture["math"], encoding="utf-8")
    (mod_dir / "code_walkthrough.md").write_text(lecture["code"], encoding="utf-8")
    (mod_dir / "practical_lab.md").write_text(lecture["lab"], encoding="utf-8")
    (mod_dir / "exercises.md").write_text(lecture["exercises"], encoding="utf-8")
    (mod_dir / "solutions.md").write_text(lecture["solutions"], encoding="utf-8")
    (mod_dir / "quiz.md").write_text(lecture["quiz"], encoding="utf-8")
    (mod_dir / "research_connections.md").write_text(lecture["research"], encoding="utf-8")
    (mod_dir / "references.md").write_text(lecture["refs"], encoding="utf-8")
    print(f"Generated all 11 files for Lecture {lecture['num']:02d}: {lecture['dir']}")

# Now append Lectures 21 to 26
LECTURES.extend([
    {
        "num": 21,
        "mod": "module_07_decoding_and_language_models",
        "dir": "lecture_21_greedy_and_beam_search",
        "title": "Greedy Decoding & CTC Prefix Beam Search",
        "subtitle": "Argmax Collapse, The Prefix Merging Algorithm, Blank vs. Non-Blank Extensions, and Pruning Heuristics",
        "objectives": [
            "Deconstruct greedy decoding (argmax path collapse) and explain why $\\mathcal{B}(\\arg\\max \\pi) \\ne \\arg\\max P(\\mathbf{y} \\mid \\mathbf{X})$.",
            "Derive the CTC Prefix Beam Search dynamic programming algorithm over prefix trees.",
            "Formulate the mathematical distinction between blank probability $P_b$ and non-blank probability $P_{nb}$ extensions.",
            "Implement a standalone CTC Prefix Beam Search decoder in Python."
        ],
        "notes": """# Lecture 21: Greedy Decoding & CTC Prefix Beam Search

## 1. The Greedy Decoding Fallacy

In greedy decoding (also known as best-path decoding), the model selects the single highest-probability token at each frame:
$$\\pi^* = (\\pi_1^*, \\dots, \\pi_T^*), \\quad \\pi_t^* = \\arg\\max_{k \\in \\mathcal{V}'} y_t^k$$
The final transcription is obtained by applying the collapse operator:
$$\\hat{\\mathbf{y}}_\\text{greedy} = \\mathcal{B}(\\pi^*)$$

### Why Greedy Decoding Fails to Find the MAP Sequence
Because CTC defines sequence probability as the **sum over all paths** in $\\mathcal{B}^{-1}(\\mathbf{y})$, the most likely sequence $\\hat{\\mathbf{y}}$ is NOT necessarily the collapse of the most likely path $\\pi^*$:
$$\\arg\\max_{\\mathbf{y}} \\sum_{\\pi \\in \\mathcal{B}^{-1}(\\mathbf{y})} P(\\pi \\mid \\mathbf{X}) \\ne \\mathcal{B}\\left( \\arg\\max_\\pi P(\\pi \\mid \\mathbf{X}) \\right)$$

```
Path 1: (a, a, a) -> Collapses to "a" with P = 0.40 * 0.40 * 0.40 = 0.064
Path 2: (a, e, e) -> Collapses to "a" with P = 0.40 * 0.35 * 0.35 = 0.049
Path 3: (e, a, e) -> Collapses to "a" with P = 0.35 * 0.40 * 0.35 = 0.049
Total Probability for Sequence "a": 0.064 + 0.049 + 0.049 = 0.162

Path 4: (b, b, b) -> Collapses to "b" with P = 0.45 * 0.45 * 0.45 = 0.091 (GREEDY WINNER!)
Total Probability for Sequence "b": 0.091

The greedy decoder selects "b" (0.091), even though sequence "a" has higher total probability (0.162)!
```

## 2. CTC Prefix Beam Search Algorithm

To search the path space efficiently without exploring $(K+1)^T$ combinations, **CTC Prefix Beam Search** maintains a beam of candidate prefix strings $p$, accumulating probabilities across all paths that collapse to $p$.

### Maintaining Blank and Non-Blank Probabilities
For each prefix $p$ at time $t$, we track two separate probabilities:
1. $P_b(p, t)$: Total probability of all paths collapsing to $p$ that **end in blank $\\epsilon$** at time $t$.
2. $P_{nb}(p, t)$: Total probability of all paths collapsing to $p$ that **end in a non-blank character** at time $t$.

The total probability of prefix $p$ is:
$$P(p, t) = P_b(p, t) + P_{nb}(p, t)$$

## 3. Prefix Extension Mathematics

At time $t$, given emission probabilities $y_t$, each active prefix $p$ in the beam is extended:

### Extension 1: Staying at Prefix $p$
- **Emitting Blank $\\epsilon$**:
  $$P_b(p, t) = P(p, t-1) \\cdot y_t^\\epsilon$$
- **Repeating Final Character $c = p_{|p|}$**:
  $$P_{nb}(p, t) \\mathrel{+}= P_{nb}(p, t-1) \\cdot y_t^c$$

### Extension 2: Extending Prefix $p$ to $p^+ = p \\circ c$ (where $c \\ne \\epsilon$)
- **If $c = p_{|p|}$ (Same character as end of prefix)**:
  To emit the same character twice, the paths **must have passed through a blank**:
  $$P_{nb}(p^+, t) \\mathrel{+}= P_b(p, t-1) \\cdot y_t^c$$
- **If $c \\ne p_{|p|}$ (Different character)**:
  Can transition from both blank and non-blank paths:
  $$P_{nb}(p^+, t) \\mathrel{+}= P(p, t-1) \\cdot y_t^c$$

After updating all prefixes, only the top $W$ prefixes with the highest total score $P(p, t)$ are retained (where $W$ is the beam width, e.g. $W=64$).
""",
        "script": """### Lecture 21 Instructor Delivery Script

**[00:00 - 10:00] The Greedy Fallacy Demonstration**
- "Welcome to Module 07. Today we start decoding: turning neural logits into actual words. The simplest thing you can do is take `argmax` at every frame. But greedy decoding has a mathematical flaw."
- *Action*: Write the 4-path numerical example on the board.
- "Look at this: path `(b, b, b)` has probability 0.091. Path `(a, a, a)` has only 0.064. Greedy picks 'b'! But there are two OTHER paths that also collapse to 'a'! When you sum them up, 'a' has a total probability of 0.162. Greedy gave you the wrong word!"

**[10:00 - 25:00] Prefix Beam Search: Why Track Blanks?**
- "In standard NLP beam search, you just append words to a beam. Why is CTC beam search different? Because of the blank token!
  If your prefix is ' ژُھنٛ ' and the next sound is another ' ژ ', did the speaker say one long ' ژ ' or two distinct ' ژ 's?
  If the previous path ended in blank, it's a second character. If it didn't end in blank, it's just the same character being held! You MUST track $P_b$ and $P_{nb}$ separately."

**[25:00 - 35:00] Beam Width and Latency Trade-Offs**
- "What beam width should you use? Beam 1 is greedy. Beam 64 gives you 99% of the accuracy. Beam 500 takes 10 times longer and buys you only 0.1% WER improvement."

**[35:00 - 45:00] Code Walkthrough**
- "Let's inspect `src/decoding/beam_search.py` and run our practical lab to implement the prefix beam decoder."
""",
        "math": """# Mathematical Foundations: CTC Prefix Beam Search Recurrence

## 1. Formal Prefix Recurrence Relations
Let $p$ be a prefix string over alphabet $\\mathcal{V}$.
Let $y_t^k$ be the emission probability of token $k$ at frame $t$.

The dynamic programming state at time $t$ consists of two tables:
$$P_b: \\mathcal{V}^* \\rightarrow [0, 1], \\quad P_{nb}: \\mathcal{V}^* \\rightarrow [0, 1]$$

### Initialization ($t = 0$):
$$P_b(\\emptyset, 0) = 1.0, \\quad P_{nb}(\\emptyset, 0) = 0.0$$
$$P_b(p, 0) = 0.0, \\quad P_{nb}(p, 0) = 0.0 \\quad \\forall p \\ne \\emptyset$$

### Frame Induction ($t = 1, \\dots, T$):
For every prefix $p$ in the active beam $\\mathcal{B}_{t-1}$:
1. **Blank Self-Loop**:
   $$P_b(p, t) \\leftarrow P_b(p, t) + \\left( P_b(p, t-1) + P_{nb}(p, t-1) \\right) y_t^\\epsilon$$

2. **Character Extensions ($c \\in \\mathcal{V}$)**:
   - If $|p| > 0$ and $c = p_{|p|}$:
     $$P_{nb}(p, t) \\leftarrow P_{nb}(p, t) + P_{nb}(p, t-1) \\cdot y_t^c$$
     $$P_{nb}(p \\circ c, t) \\leftarrow P_{nb}(p \\circ c, t) + P_b(p, t-1) \\cdot y_t^c$$
   - Otherwise ($c \\ne p_{|p|}$):
     $$P_{nb}(p \\circ c, t) \\leftarrow P_{nb}(p \\circ c, t) + \\left( P_b(p, t-1) + P_{nb}(p, t-1) \\right) y_t^c$$

## 2. Beam Pruning Operator
Let $\\mathcal{P}_t$ be the set of all candidate prefixes generated at time $t$.
Total score for candidate $p$:
$$\\text{Score}(p, t) = P_b(p, t) + P_{nb}(p, t)$$
The pruned beam $\\mathcal{B}_t$ retains the top $W$ candidates:
$$\\mathcal{B}_t = \\arg\\text{top-}W_{p \\in \\mathcal{P}_t} \\text{Score}(p, t)$$
""",
        "code": """# Code Walkthrough: Lecture 21

We inspect `src/decoding/beam_search.py`, which implements the certified prefix beam search.

```python
# src/decoding/beam_search.py
from collections import defaultdict
import torch
from typing import List, Tuple

def ctc_prefix_beam_search(emissions: torch.Tensor, beam_width: int = 64, blank_id: int = 0) -> List[Tuple[List[int], float]]:
    \"\"\"
    CTC Prefix Beam Search Decoder.
    Args:
        emissions: (T, Vocab_Size) tensor of softmax probabilities.
        beam_width: Number of active candidate prefixes.
    Returns:
        Sorted list of (prefix_tokens, log_probability).
    \"\"\"
    T, V = emissions.shape
    # Beam maps prefix tuple -> (P_b, P_nb)
    beam = {(): (1.0, 0.0)}
    
    for t in range(T):
        new_beam = defaultdict(lambda: (0.0, 0.0))
        y_t = emissions[t]
        
        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb
            
            # 1. Blank extension
            new_p_b, new_p_nb = new_beam[prefix]
            new_beam[prefix] = (new_p_b + p_total * y_t[blank_id].item(), new_p_nb)
            
            # 2. Non-blank extensions
            for c in range(V):
                if c == blank_id:
                    continue
                p_c = y_t[c].item()
                if p_c < 1e-5:  # Pruning threshold
                    continue
                    
                if len(prefix) > 0 and prefix[-1] == c:
                    # Same character repeat
                    b_p_b, b_p_nb = new_beam[prefix]
                    new_beam[prefix] = (b_p_b, b_p_nb + p_nb * p_c)
                    
                    ext_prefix = prefix + (c,)
                    e_p_b, e_p_nb = new_beam[ext_prefix]
                    new_beam[ext_prefix] = (e_p_b, e_p_nb + p_b * p_c)
                else:
                    ext_prefix = prefix + (c,)
                    e_p_b, e_p_nb = new_beam[ext_prefix]
                    new_beam[ext_prefix] = (e_p_b, e_p_nb + p_total * p_c)
                    
        # Prune to beam width
        sorted_prefixes = sorted(
            new_beam.items(),
            key=lambda item: item[1][0] + item[1][1],
            reverse=True
        )[:beam_width]
        beam = dict(sorted_prefixes)
        
    return [(list(p), p_b + p_nb) for p, (p_b, p_nb) in beam.items()]
```
""",
        "lab": """# Practical Lab 21: Comparing Greedy Decoding against Prefix Beam Search

## Objective
1. Implement the synthetic emissions matrix where greedy decoding fails.
2. Execute both greedy decoding and `ctc_prefix_beam_search`.
3. Verify that Prefix Beam Search discovers the true MAP sequence.

## Laboratory Code

```python
import torch
from src.decoding.beam_search import ctc_prefix_beam_search

def run_lab():
    print("=== Step 1: Synthesize Counterexample Emissions Matrix ===")
    # Vocab: 0=Blank, 1='A', 2='B'
    # Frame 0: 'B'=0.45, 'A'=0.40, Blank=0.15
    # Frame 1: 'B'=0.45, Blank=0.35, 'A'=0.20
    # Frame 2: 'B'=0.45, 'A'=0.35, Blank=0.20
    emissions = torch.tensor([
        [0.15, 0.40, 0.45],
        [0.35, 0.20, 0.45],
        [0.20, 0.35, 0.45]
    ])
    
    print("=== Step 2: Greedy Decoding ===")
    greedy_path = torch.argmax(emissions, dim=-1).tolist()
    print(f"Argmax path: {greedy_path}")
    # Collapse greedy
    from src.decoding.alignment import ctc_collapse
    greedy_transcript = ctc_collapse(greedy_path, blank_id=0)
    print(f"Greedy Result: {greedy_transcript} (Selects token 2 = 'B')")
    
    print("\n=== Step 3: Prefix Beam Search (Beam=10) ===")
    candidates = ctc_prefix_beam_search(emissions, beam_width=10, blank_id=0)
    for p, score in candidates:
        print(f"Prefix: {p} | Total Probability: {score:.5f}")
        
    best_beam_prefix = candidates[0][0]
    print(f"\nWinning Beam Prefix: {best_beam_prefix}")
    print("Prefix Beam Search successfully aggregated probability across multiple collapsing paths!")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 21

## Analytical Problems
1. Prove that greedy decoding is a special case of CTC Prefix Beam Search where beam width $W=1$ and all cross-path probability merging is omitted.
2. For an alphabet size $K = 98$ and utterance length $T = 500$, calculate the total number of candidate prefixes evaluated at each frame if token pruning is disabled. Why is token-level threshold pruning ($y_t^c > 10^{-4}$) necessary?

## Implementation Problems
3. Implement a function `add_word_boundary_bonus(prefix: tuple, beta: float) -> float` that counts word delimiters (spaces `|`) in a prefix and adds length bonus $\\beta |\\mathbf{w}|$.

## Forensic Auditing
4. In Stage 20 of the EduCasheer repository, the team noticed that increasing beam width from $W=64$ to $W=256$ reduced WER by only $0.08\\%$, but increased inference latency by $3.4\\times$. What Pareto frontier dictates the choice of beam width for real-time systems?
""",
        "solutions": """# Solutions: Lecture 21

## Exercise 1
In beam search with $W=1$, only a single prefix is retained at each time step. Furthermore, if cross-path merging is disabled, each prefix represents only a single uncollapsed path $\\pi$. The transition at frame $t$ simply picks $\\pi_t = \\arg\\max_k y_t^k$, which is strictly identical to the definition of greedy argmax path selection followed by $\\mathcal{B}(\\pi)$.

## Exercise 2
Without token pruning, each of the $W$ prefixes in the beam generates $|\mathcal{V}'| = K+1 = 99$ child branches at every frame.
For $W = 64$:
$$64 \\times 99 = 6{,}336 \\text{ prefix updates per frame}$$
Over $T = 500$ frames:
$$500 \\times 6{,}336 = 3{,}168{,}000 \\text{ dictionary hash updates}$$
Most of these 99 tokens have emission probabilities near machine epsilon ($< 10^{-7}$). By pruning tokens with $y_t^c < 10^{-4}$, the number of active branches per prefix drops from 99 to an average of $3.2$, reducing dictionary operations by over $96\\%$ with zero loss in search accuracy.

## Exercise 3
```python
def add_word_boundary_bonus(prefix: tuple, space_id: int = 1, beta: float = 0.5) -> float:
    # Count occurrences of space character in prefix
    word_count = prefix.count(space_id)
    return beta * word_count
```
""",
        "quiz": """# Comprehension Quiz: Lecture 21

1. **Why does greedy decoding fail to find the optimal CTC sequence?**
   - A) Because greedy decoding does not use GPUs.
   - B) Because multiple distinct paths can collapse onto the same sequence, and the sum of their probabilities can exceed the probability of the argmax path. [CORRECT]
   - C) Because greedy decoding ignores the blank token.
   - D) Because the learning rate is zero.
   *Explanation*: CTC sequence probability is the sum over pre-image paths; greedy only tracks the single highest path.

2. **Why does CTC Prefix Beam Search maintain separate $P_b$ and $P_{nb}$ probabilities?**
   - A) To distinguish paths ending in blank from paths ending in non-blank, which dictates whether extending by the same character repeats it or holds it. [CORRECT]
   - B) To run on two separate CPU cores.
   - C) To separate audio from text.
   - D) To calculate training loss.
   *Explanation*: Distinguishing blank endings is mandatory to handle consecutive identical characters correctly.

3. **What is the standard beam width $W$ used in production ASR decoders?**
   - A) $W = 1$
   - B) $W = 32 \\text{ to } 64$ [CORRECT]
   - C) $W = 10{,}000$
   - D) $W = 1{,}000{,}000$
   *Explanation*: $W=32$ to $64$ captures virtually all search accuracy without excessive latency.

4. **What is the effect of token probability pruning ($y_t^c < 10^{-4}$) in beam search?**
   - A) It deletes words from the dictionary.
   - B) It skips evaluating highly improbable vocabulary extensions, speeding up search by 10-20x. [CORRECT]
   - C) It causes audio clipping.
   - D) It reduces the sampling rate.
   *Explanation*: Pruning improbable tokens eliminates tens of thousands of redundant dictionary lookups.

5. **In the EduCasheer production decoder, what character index represents the word delimiter space `|`?**
   - A) Index 0
   - B) Index 1 [CORRECT]
   - C) Index 97
   - D) Index -1
   *Explanation*: In `src/normalization/kashmiri_char_map.py`, index 1 is reserved for the word boundary delimiter `|`.
""",
        "research": """# Research Connections: Lecture 21

- **Repository Stage**: `stages/stage_20/` (CTC Prefix Beam Search Implementation) and `src/decoding/beam_search.py`.
- **Historic Metric**: In Stage 20, replacing greedy decoding with pure acoustic Prefix Beam Search ($W=64$, no LM) on Model B-02C reduced test CER from $11.2\\%$ to $10.8\\%$, demonstrating that prefix merging alone fixes path collapse errors even before adding an external language model.
- **Key Code File**: `src/decoding/beam_search.py`.
- **Case Study Reference**: Case Study CS-13 ("Beam Width vs. Latency Pareto Frontiers in Low-Resource ASR").
""",
        "refs": """# References: Lecture 21

1. Hannun, A. Y., et al. (2014). *First-Pass Large Vocabulary Continuous Speech Recognition using Bi-directional Recurrent DNNs*. arXiv:1408.2873.
2. Graves, A. (2012). *Sequence Transduction with Recurrent Neural Networks*. ICML Workshop.
3. Hwang, K., & Sung, W. (2016). *Character-Level N-gram Language Modeling for Web-Scale Speech Recognition*. ICASSP.
"""
    },
    {
        "num": 22,
        "mod": "module_07_decoding_and_language_models",
        "dir": "lecture_22_ngram_language_models_kenlm",
        "title": "N-gram Language Modeling & Kneser-Ney Smoothing",
        "subtitle": "Kashmiri Text Corpora, Modified Kneser-Ney Math, Continuation Probabilities, and KenLM Binary Formats",
        "objectives": [
            "Formalize statistical language modeling as autoregressive conditional probability estimation.",
            "Derive Modified Kneser-Ney smoothing and understand continuation probability.",
            "Analyze the linguistic sparsity of Kashmiri text and the challenge of high morphological inflection.",
            "Train and query a 5-gram Kashmiri language model using the KenLM library."
        ],
        "notes": """# Lecture 22: N-gram Language Modeling & Kneser-Ney Smoothing

## 1. The Language Modeling Prior in Low-Resource Speech

While the acoustic model estimates $P(\\mathbf{X} \\mid \\mathbf{w})$ from audio, speech recognition depends equally on the **Language Model (LM)** prior $P(\\mathbf{w})$, which assigns probabilities to hypothesized word sequences based on linguistic grammar, syntax, and vocabulary collocation:

$$P(\\mathbf{w}) = P(w_1, w_2, \\dots, w_U) = \\prod_{u=1}^U P(w_u \\mid w_1, \\dots, w_{u-1})$$

In an $n$-gram model, we apply a Markov assumption of order $n-1$:
$$P(w_u \\mid w_{<u}) \\approx P(w_u \\mid w_{u-n+1}^{u-1})$$

In the EduCasheer Kashmiri project, the language model was trained on a curated Kashmiri text corpus spanning **1.8 million sentences** gathered from literature, poetry, news, and social media.

## 2. The Sparsity Crisis & Modified Kneser-Ney Smoothing

Kashmiri is a **morphologically rich, inflected language**. Nouns inflect for four cases, two genders, and two numbers; verbs conjugate across intricate aspectual paradigms.

Consequently, vocabulary sparsity is acute:
- Over $65\\%$ of unique trigrams appear only once (**hapax legomena**).
- Naive Maximum Likelihood Estimation (MLE) assigns zero probability to unseen sequences:
  $$P_\\text{MLE}(w_i \\mid w_{i-1}) = \\frac{C(w_{i-1}, w_i)}{C(w_{i-1})} = 0 \\quad \\text{if } C(w_{i-1}, w_i) = 0$$
- If an unseen Kashmiri word pair receives zero probability ($P=0$), its log-likelihood is $-\\infty$, completely blocking valid acoustic hypotheses during beam search!

### Modified Kneser-Ney Smoothing (Chen & Goodman, 1996)
**Modified Kneser-Ney (MKN)** is the gold standard smoothing algorithm. It interpolates higher-order $n$-gram counts with a backoff distribution based on **continuation probability**:

$$P_\\text{KN}(w_i \\mid w_{i-n+1}^{i-1}) = \\frac{\\max(C(w_{i-n+1}^i) - D(C), 0)}{C(w_{i-n+1}^{i-1})} + \\gamma(w_{i-n+1}^{i-1}) P_\\text{KN}(w_i \\mid w_{i-n+2}^{i-1})$$

### The Intuition of Continuation Probability
Consider the unigram backoff $P_\\text{cont}(w_i)$.
In standard smoothing, common words (like "Francisco") have high unigram frequency $C(w)$. But in text, "Francisco" almost exclusively follows "San".
If the acoustic model hears a novel context (e.g. "نئ Francisco"), a standard model assigns high probability to "Francisco".
Kneser-Ney recognizes that "Francisco" has **low continuation diversity**—it only completes a single context. The lower-order distribution measures how likely a word is to complete an *unseen* context:
$$P_\\text{cont}(w_i) = \\frac{|\\{w_{i-1} \\mid C(w_{i-1}, w_i) > 0\\}|}{\\sum_{w'} |\\{w_{i-1} \\mid C(w_{i-1}, w') > 0\\}|}$$

## 3. KenLM: High-Performance Binary Storage

In production ASR, querying an $n$-gram model occurs millions of times per second inside beam search.
**KenLM** (Heafield, 2011) provides a highly optimized C++ implementation:
1. **Sorted Probing / Trie Data Structures**: Uses bit-level quantization and 8-byte trie records with binary search.
2. **Memory-Mapped Binary Format (`.arpa` $\\rightarrow$ `.binary`)**: Allows instantaneous loading without parsing text ARPA files.
3. **Execution Speed**: Queries execute in under $20\\text{ nanoseconds}$ per $n$-gram lookup.
""",
        "script": """### Lecture 22 Instructor Delivery Script

**[00:00 - 10:00] Why Audio Alone is Never Enough**
- "Why do we need a Language Model? Close your eyes and listen to this sentence: 'I want to recognize speech.' Now listen to this: 'I want to wreck a nice beach.'
  Acoustically, they sound nearly identical! How does your brain know which one I said? You have a Language Model in your head! You know that people don't 'wreck nice beaches' in a computer science lecture."

**[10:00 - 25:00] The Kneser-Ney Intuition**
- "Why not just count pairs of words? Maximum Likelihood Estimation fails because language is sparse. If you never saw a word combination in your training text, MLE says the probability is ZERO. In log-space, $\\log(0) = -\\infty$. Your decoder crashes!"
- "Explain Kneser-Ney. Use the 'San Francisco' example. Why is continuation probability the smartest idea in statistical NLP? Because it counts how many *different* contexts a word can follow, not just raw counts."

**[25:00 - 35:00] KenLM in Production**
- "Why KenLM? If you write an $n$-gram model in Python using a dictionary, your decoder will process 1 word per second. KenLM is compiled C++ using memory-mapped trie structures. It does 50 million queries per second!"

**[35:00 - 45:00] Lab and Code Inspection**
- "Open `src/language_models/kenlm_wrapper.py`. We will inspect the Python interface and run practical queries against our Kashmiri 5-gram model."
""",
        "math": """# Mathematical Foundations: Modified Kneser-Ney Smoothing Equations

## 1. Absolute Discounting with Variable Offsets
Modified Kneser-Ney uses three distinct discount parameters $D_1, D_2, D_{3+}$ based on whether the $n$-gram count is 1, 2, or $\\ge 3$:
$$D(c) = \\begin{cases} 0 & c = 0 \\\\ D_1 & c = 1 \\\\ D_2 & c = 2 \\\\ D_{3+} & c \\ge 3 \\end{cases}$$
The optimal discounts are derived analytically from training counts $n_k$ (the number of $n$-grams with count $k$):
$$Y = \\frac{n_1}{n_1 + 2 n_2}, \\quad D_1 = 1 - 2 Y \\frac{n_2}{n_1}, \\quad D_2 = 2 - 3 Y \\frac{n_3}{n_2}, \\quad D_{3+} = 3 - 4 Y \\frac{n_4}{n_3}$$

## 2. The General MKN Interpolation Recurrence
For an $n$-gram $w_{i-n+1}^i$:
$$P_\\text{MKN}(w_i \\mid w_{i-n+1}^{i-1}) = \\frac{C(w_{i-n+1}^i) - D(C(w_{i-n+1}^i))}{C(w_{i-n+1}^{i-1})} + \\gamma(w_{i-n+1}^{i-1}) P_\\text{MKN}(w_i \\mid w_{i-n+2}^{i-1})$$

The normalization weight $\\gamma$ ensures probabilities sum to $1.0$:
$$\\gamma(w_{i-n+1}^{i-1}) = \\frac{D_1 N_1(w_{i-n+1}^{i-1} \\bullet) + D_2 N_2(w_{i-n+1}^{i-1} \\bullet) + D_{3+} N_{3+}(w_{i-n+1}^{i-1} \\bullet)}{C(w_{i-n+1}^{i-1})}$$
where $N_k(w_{i-n+1}^{i-1} \\bullet) = |\\{w \\mid C(w_{i-n+1}^{i-1}, w) = k\\}|$.

## 3. Continuation Base Distribution (Unigram)
At the base of the recursion ($n=1$):
$$P_\\text{cont}(w_i) = \\frac{|\\{w_{i-1} \\mid C(w_{i-1}, w_i) > 0\\}|}{\\sum_{w'} |\\{w_{i-1} \\mid C(w_{i-1}, w') > 0\\}|}$$
""",
        "code": """# Code Walkthrough: Lecture 22

We inspect `src/language_models/kenlm_wrapper.py`, which provides the Python wrapper for KenLM binary models.

```python
# src/language_models/kenlm_wrapper.py
import os
from typing import List, Optional

class KashmiriKenLMWrapper:
    \"\"\"
    Interface for KenLM 5-gram language models trained on Kashmiri text corpora.
    \"\"\"
    def __init__(self, binary_model_path: str):
        if not os.path.exists(binary_model_path):
            raise FileNotFoundError(f"KenLM binary model not found at: {binary_model_path}")
        try:
            import kenlm
            self.model = kenlm.Model(binary_model_path)
            self.order = self.model.order
        except ImportError:
            raise ImportError("Please install kenlm: pip install https://github.com/kpu/kenlm/archive/master.zip")

    def score_sentence(self, sentence: str, bos: bool = True, eos: bool = True) -> float:
        \"\"\"Returns log10 probability of sentence.\"\"\"
        return self.model.score(sentence, bos=bos, eos=eos)

    def score_transition(self, context_words: List[str], next_word: str) -> float:
        \"\"\"
        Queries log10 P(next_word | context_words).
        \"\"\"
        full_text = " ".join(context_words + [next_word])
        # KenLM returns total sentence log10 prob
        # Extract last word log-prob by difference
        state_in = kenlm.State()
        state_out = kenlm.State()
        self.model.BeginSentenceWrite(state_in)
        for w in context_words:
            self.model.BaseScore(state_in, w, state_out)
            state_in, state_out = state_out, state_in
        log_prob = self.model.BaseScore(state_in, next_word, state_out)
        return log_prob
```
""",
        "lab": """# Practical Lab 22: Training and Querying a Toy N-gram Language Model

## Objective
1. Implement a toy Modified Kneser-Ney bigram smoother in pure Python.
2. Calculate continuation probabilities on a sample text corpus.
3. Compare MLE bigram probabilities against Kneser-Ney smoothed probabilities for unseen word pairs.

## Laboratory Code

```python
from collections import defaultdict, Counter

def run_lab():
    print("=== Step 1: Toy Kashmiri Corpus ===")
    corpus = [
        "کٲشُر زَبانہِ مَنٛز چھِ واریاہ کَتھٕ",
        "سۄ چھِ کٲشُر زَبانہِ ہِنٛز کِتاب",
        "تِم چھِ کٲشُر بَولان واریاہ کَتھٕ"
    ]
    
    # Tokenize
    tokenized = [sent.split() for sent in corpus]
    
    # Count unigrams and bigrams
    unigrams = Counter()
    bigrams = Counter()
    for s in tokenized:
        for w in s:
            unigrams[w] += 1
        for i in range(len(s) - 1):
            bigrams[(s[i], s[i+1])] += 1
            
    print(f"Total vocabulary size: {len(unigrams)} words")
    print(f"Total unique bigrams:  {len(bigrams)} bigrams")
    
    print("\n=== Step 2: Compute Continuation Probabilities ===")
    # continuation_count[w] = number of distinct words that precede w
    preceding_contexts = defaultdict(set)
    for (w1, w2) in bigrams.keys():
        preceding_contexts[w2].add(w1)
        
    total_distinct_bigrams = len(bigrams)
    p_cont = {w: len(contexts) / total_distinct_bigrams for w, contexts in preceding_contexts.items()}
    
    for w, p in sorted(p_cont.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Word: '{w}' -> Continuation Prob: {p:.4f} (Preceded by {len(preceding_contexts[w])} contexts)")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 22

## Analytical Problems
1. Explain why Kneser-Ney smoothing uses continuation count $|\\{w_{i-1} \\mid C(w_{i-1}, w_i) > 0\\}|$ rather than raw frequency count $C(w_i)$ in the unigram backoff distribution.
2. Given discounts $D_1 = 0.5, D_2 = 0.75, D_{3+} = 0.85$, compute the smoothed bigram probability $P_\\text{KN}(\\text{کَتھٕ} \\mid \\text{واریاہ})$ for a context where $C(\\text{واریاہ}, \\text{کَتھٕ}) = 2$ and $C(\\text{واریاہ}) = 5$.

## Implementation Problems
3. Implement a Python function `export_to_arpa(unigrams, bigrams, discounts, output_path: str)` that formats an in-memory language model into the standard ARPA file format accepted by KenLM.

## Forensic Auditing
4. In Stage 21 of the EduCasheer repository, the team discovered that training KenLM directly on unnormalized web text caused perplexity to explode to $1,420.0$. After applying Certified Normalizer v3, perplexity dropped to $84.2$. What caused the $17\\times$ perplexity reduction?
""",
        "solutions": """# Solutions: Lecture 22

## Exercise 1
Raw frequency $C(w)$ measures how often a word occurs in training text. However, a word may occur very frequently but only in one specific collocation (e.g. "Francisco" in "San Francisco"). If an unseen context arrives (e.g. "Srinagar [?]"), using raw frequency would assign high probability to "Francisco". Continuation count measures the *diversity of linguistic contexts* a word can follow: $|\\{w_{i-1} \\mid C(w_{i-1}, w) > 0\\}|$. A word that has appeared after 50 different preceding words is far more likely to appear in an unknown context than a word that has only ever appeared after one specific word.

## Exercise 2
Given $C(w_{i-1}, w_i) = 2$, $C(w_{i-1}) = 5$, and $D_2 = 0.75$:
The discounted count is:
$$\\max(C - D_2, 0) = 2 - 0.75 = 1.25$$
The first term is:
$$\\frac{1.25}{5} = 0.25$$
Assuming $\\gamma(w_{i-1}) = 0.15$ and $P_\\text{cont}(w_i) = 0.08$:
$$P_\\text{KN} = 0.25 + (0.15 \\times 0.08) = 0.25 + 0.012 = 0.262$$

## Exercise 3
```python
def export_to_arpa(unigram_probs, bigram_probs, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        f.write(f"ngram 1={len(unigram_probs)}\n")
        f.write(f"ngram 2={len(bigram_probs)}\n\n")
        f.write("\\1-grams:\n")
        for w, logp in unigram_probs.items():
            f.write(f"{logp:.4f}\t{w}\n")
        f.write("\n\\2-grams:\n")
        for (w1, w2), logp in bigram_probs.items():
            f.write(f"{logp:.4f}\t{w1} {w2}\n")
        f.write("\n\\end\\\n")
```
""",
        "quiz": """# Comprehension Quiz: Lecture 22

1. **What is the Markov assumption of order $n-1$ in an $n$-gram language model?**
   - A) The probability of the next word depends only on the preceding $n-1$ words. [CORRECT]
   - B) All words have equal probability.
   - C) Audio duration is proportional to word length.
   - D) Language models cannot be trained on text.
   *Explanation*: The Markov assumption conditions word probability only on a finite history of length $n-1$.

2. **Why does naive Maximum Likelihood Estimation (MLE) fail on low-resource language corpora?**
   - A) Because MLE models are too slow.
   - B) Because unseen word combinations receive probability 0 ($-\\infty$ log-prob), completely blocking valid acoustic candidates during search. [CORRECT]
   - C) Because MLE requires GPUs.
   - D) Because Kashmiri has no verbs.
   *Explanation*: Zero probabilities for unseen $n$-grams make MLE unviable without smoothing.

3. **What does continuation probability $P_\\text{cont}(w)$ measure in Kneser-Ney smoothing?**
   - A) How many times the word was mispronounced.
   - B) The number of distinct preceding word contexts that the word can complete, reflecting how likely it is to appear in novel contexts. [CORRECT]
   - C) The length of the word in characters.
   - D) The acoustic pitch of the word.
   *Explanation*: Continuation probability counts distinct left-contexts rather than raw unigram frequency.

4. **Why is KenLM widely adopted in production speech recognition systems?**
   - A) It is written in Python.
   - B) It provides highly optimized C++ trie data structures and memory-mapped binaries executing queries in under 20 nanoseconds. [CORRECT]
   - C) It is an acoustic neural network.
   - D) It replaces the CTC loss.
   *Explanation*: KenLM's compiled C++ binary trie format provides extremely fast $n$-gram lookups.

5. **Why did applying Certified Normalizer v3 reduce Kashmiri language model perplexity from 1,420 to 84.2?**
   - A) It deleted rare words.
   - B) It unified fragmented Persian/Arabic homoglyphs and stripped invisible ZWNJ characters, drastically reducing false vocabulary sparsity. [CORRECT]
   - C) It converted Kashmiri to English.
   - D) It doubled the size of the training corpus.
   *Explanation*: Normalizing homoglyphs merges fragmented character distributions, dramatically lowering perplexity.
""",
        "research": """# Research Connections: Lecture 22

- **Repository Stage**: `stages/stage_21/` (Kashmiri KenLM N-Gram Training) and `src/language_models/kenlm_wrapper.py`.
- **Historic Breakthrough**: In Stage 21, the team trained 3-gram, 4-gram, and 5-gram KenLM models on 1.8M Kashmiri sentences. Integrating the 5-gram model into beam search reduced Model B-02C's Word Error Rate by $2.8\\%$ absolute on colloquial speech, proving the efficacy of statistical priors in low-resource settings.
- **Key Code File**: `src/language_models/kenlm_wrapper.py`.
- **Case Study Reference**: Case Study CS-14 ("Smoothing Morphologically Rich Kashmiri Text Corpora").
""",
        "refs": """# References: Lecture 22

1. Chen, S. F., & Goodman, J. (1996). *An empirical study of smoothing techniques for language modeling*. ACL.
2. Heafield, K. (2011). *KenLM: Faster and Smaller Language Model Queries*. WMT.
3. Kneser, R., & Ney, H. (1995). *Improved backing-off for m-gram language modeling*. ICASSP.
"""
    },
    {
        "num": 23,
        "mod": "module_07_decoding_and_language_models",
        "dir": "lecture_23_acoustic_temperature_and_shallow_fusion",
        "title": "Acoustic Temperature & Shallow Fusion",
        "subtitle": "Softmax Calibration ($T=1.08$), Grid Search Optimization for $\\alpha$ and $\\beta$, and Certified Decoder Tuning",
        "objectives": [
            "Analyze neural network overconfidence in CTC models and the need for acoustic temperature scaling $T$.",
            "Formulate Shallow Fusion decoding combining acoustic CTC log-probabilities, language model scores, and word bonus.",
            "Conduct a systematic 2D grid search over $(\\alpha, \\beta)$ to minimize Word Error Rate.",
            "Implement a calibrated Shallow Fusion beam search decoder in Python."
        ],
        "notes": """# Lecture 23: Acoustic Temperature & Shallow Fusion

## 1. The Overconfidence Pathology of Neural Acoustic Models

Modern deep neural networks trained with cross-entropy or CTC loss suffer from **severe probability overconfidence** (Guo et al., 2017).
When a model outputs emission probabilities:
$$y_t^k = \\frac{\\exp(u_t^k)}{\\sum_{j} \\exp(u_t^j)}$$
the softmax distribution frequently concentrates $99.9\\%$ probability mass on a single token, even when the underlying acoustic signal is ambiguous or noisy.

In shallow fusion, we combine acoustic log-probabilities with language model scores:
$$\\text{Score}(\\mathbf{w}) = \\log P_\\text{CTC}(\\mathbf{w} \\mid \\mathbf{X}) + \\alpha \\log P_\\text{LM}(\\mathbf{w}) + \\beta |\\mathbf{w}|$$

### The Overconfidence Dilemma
If $\\log P_\\text{CTC}$ is overconfident, its dynamic range spans $[0, -500]$, while $\\log P_\\text{LM}$ spans $[0, -20]$. The acoustic score completely overwhelms the language model!
Even if the language model knows a sequence is ungrammatical gibberish, an acoustic difference of $0.01$ will override the language model prior.

## 2. Acoustic Temperature Scaling ($T=1.08$)

To restore calibration, we introduce an **Acoustic Temperature parameter $T > 1.0$** applied to the unnormalized logits prior to softmax:

$$y_t^k(T) = \\frac{\\exp(u_t^k / T)}{\\sum_j \\exp(u_t^j / T)}$$

### Effect on Probability Distribution
- When $T = 1.0$: Standard uncalibrated softmax.
- When $T > 1.0$: Softens the probability distribution, lowering peaks and raising the floor of competing phonetic candidates.
- In the EduCasheer Stage 22 tuning experiments, grid searching temperature across $T \\in [0.90, 1.30]$ revealed an optimal operating point at **$T = 1.08$**:
  - Calibrated the acoustic logit variance.
  - Allowed the language model score $\\alpha \\log P_\\text{LM}$ to effectively re-rank phonetically plausible homophones.

## 3. The 2D Hyperparameter Grid Search ($\\alpha$ and $\\beta$)

Shallow fusion requires tuning two coupled hyperparameters:
1. **$\\alpha$ (Language Model Weight)**: Governs the strength of the linguistic prior relative to acoustic evidence.
   - If $\\alpha$ is too small ($< 0.1$): The decoder makes grammatical errors and phonetic misspellings.
   - If $\\alpha$ is too large ($> 1.5$): The language model overpowers acoustic evidence, hallucinating common words that were never spoken.
2. **$\\beta$ (Word Insertion Bonus)**: Governs the hypothesis length.
   - Since $P(w_i \\mid w_{<i}) < 1.0$, adding more words strictly decreases the total log-probability: $\\log P < 0$.
   - Without compensation, beam search favors artificially short hypotheses (deleting short words).
   - $\\beta > 0$ rewards longer hypotheses, counteracting the deletion bias.

```
+------------------------------------------------------------------------------------------+
|                        STAGE 22 DECODER GRID SEARCH ON VALIDATION SET                    |
+---------------------+-------------------+-------------------+----------------------------+
| Configuration       | alpha (LM Weight) | beta (Word Bonus) | Validation WER (%)         |
+---------------------+-------------------+-------------------+----------------------------+
| Pure Greedy         | 0.00              | 0.00              | 25.82%                     |
| Baseline Beam       | 0.30              | 1.00              | 25.40%                     |
| Optimal Production  | 0.50              | 1.50              | 24.12% (CERTIFIED OPTIMUM) |
| Over-Smoothed       | 1.20              | 2.50              | 27.80% (Hallucinations)    |
+---------------------+-------------------+-------------------+----------------------------+
```
""",
        "script": """### Lecture 23 Instructor Delivery Script

**[00:00 - 10:00] The Calibration Crisis**
- "Welcome to Lecture 23. Today we solve a subtle mathematical pathology: neural network overconfidence.
  If you ask XLS-R: 'How confident are you that frame 42 is the letter Kaf?', the model says: '99.999% confident!' Even if there was background noise, coughing, and static! Neural networks are terrible at knowing what they don't know."

**[10:00 - 25:00] Softmax Temperature Math**
- "How do you fix overconfidence without retraining the network? Divide the logits by temperature $T$!
  If you divide by $T = 1.08$, what happens to the numbers before the exponent? They get slightly smaller. Softmax spreads the probability mass out. Now the true competing sounds get 5% or 10% probability instead of 0.0001%. That gives our Language Model a fighting chance to pick the grammatically correct word!"

**[25:00 - 35:00] The Alpha and Beta Balancing Act**
- "Walk through the two knobs:
  $\\alpha$ is your language weight. Turn it up, the model speaks beautiful Kashmiri poetry, but it stops listening to the speaker! Turn it down, it spells phonetically but writes nonsense.
  $\\beta$ is your word bonus. Every word has negative log-prob. Without $\\beta$, the decoder wants to output zero words to get a score of zero! $\\beta$ pays the decoder a bonus for every word it emits."

**[35:00 - 45:00] Code Demonstration**
- "Open `src/decoding/shallow_fusion.py` to see the calibrated shallow fusion beam search loop."
""",
        "math": """# Mathematical Foundations: Calibrated Shallow Fusion Objective

## 1. Temperature-Scaled Softmax
Let $\\mathbf{u}_t \\in \\mathbb{R}^{|\\mathcal{V}'|}$ be the unnormalized acoustic logits at frame $t$.
With acoustic temperature $T > 0$:
$$y_t^k(T) = \\frac{\\exp(u_t^k / T)}{\\sum_{j=0}^{|\\mathcal{V}'|-1} \\exp(u_t^j / T)}$$

### Log-Domain Simplification
$$\\log y_t^k(T) = \\frac{u_t^k}{T} - \\text{logsumexp}\\left( \\frac{\\mathbf{u}_t}{T} \\right)$$
As $T \\rightarrow \\infty$, $y_t^k(T) \\rightarrow \\frac{1}{|\\mathcal{V}'|}$ (uniform distribution).
As $T \\rightarrow 0^+$, $y_t^k(T) \\rightarrow \\mathbb{I}(k = \\arg\\max_j u_t^j)$ (hard one-hot argmax).

## 2. The Complete Shallow Fusion Objective
For a candidate word sequence $\\mathbf{w} = (w_1, \\dots, w_U)$ with character-level representation $\\mathbf{y}$:
$$\\mathcal{S}(\\mathbf{w} \\mid \\mathbf{X}) = \\log P_\\text{CTC}(\\mathbf{y} \\mid \\mathbf{X}; T) + \\alpha \\log_{10} P_\\text{LM}(\\mathbf{w}) + \\beta |\\mathbf{w}|$$
where:
- $\\log P_\\text{CTC}(\\mathbf{y} \\mid \\mathbf{X}; T)$ is computed via Prefix Beam Search over temperature-scaled emissions.
- $P_\\text{LM}(\\mathbf{w}) = \\prod_{i=1}^U P_\\text{KN}(w_i \\mid w_{i-n+1}^{i-1})$ is the KenLM score.
- $|\\mathbf{w}|$ is the word count (number of space tokens $+ 1$).
- $\\alpha \\ge 0$ is the LM weight.
- $\\beta \\ge 0$ is the word insertion bonus.

## 3. Grid Search Optimization
The optimal parameters $(\\alpha^*, \\beta^*, T^*)$ solve:
$$(\\alpha^*, \\beta^*, T^*) = \\arg\\min_{\\alpha, \\beta, T} \\text{WER}\\left( \\text{Decode}(\\mathcal{D}_\\text{val}; \\alpha, \\beta, T), \\; \\mathcal{D}_\\text{val}^\\text{ref} \\right)$$
evaluated via 2D/3D coordinate descent over the validation set.
""",
        "code": """# Code Walkthrough: Lecture 23

We inspect `src/decoding/shallow_fusion.py`, which integrates temperature scaling and KenLM scoring into prefix beam search.

```python
# src/decoding/shallow_fusion.py
import torch
import math
from typing import List, Tuple
from src.decoding.beam_search import ctc_prefix_beam_search
from src.language_models.kenlm_wrapper import KashmiriKenLMWrapper

class CalibratedShallowFusionDecoder:
    \"\"\"
    Production Shallow Fusion Decoder with Temperature Scaling (T=1.08)
    and KenLM 5-gram rescoring.
    \"\"\"
    def __init__(self, lm_wrapper: KashmiriKenLMWrapper, alpha: float = 0.50, beta: float = 1.50, temperature: float = 1.08, beam_width: int = 64):
        self.lm = lm_wrapper
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.beam_width = beam_width

    def decode(self, logits: torch.Tensor, id_to_char: dict) -> str:
        \"\"\"
        Args:
            logits: (T, Vocab_Size) raw acoustic logits.
        Returns:
            Best decoded Kashmiri text string.
        \"\"\"
        # 1. Apply temperature scaling
        scaled_logits = logits / self.temperature
        emissions = torch.softmax(scaled_logits, dim=-1)
        
        # 2. Run acoustic prefix beam search to get N-best candidate prefixes
        candidates = ctc_prefix_beam_search(emissions, beam_width=self.beam_width)
        
        # 3. Rescore candidates with Shallow Fusion objective
        best_sentence = ""
        best_total_score = -float('inf')
        
        for token_ids, acoustic_prob in candidates:
            # Convert token IDs to text string
            chars = [id_to_char.get(i, "") for i in token_ids]
            text = "".join(chars).replace("|", " ").strip()
            
            if not text:
                continue
                
            words = text.split()
            word_count = len(words)
            
            # Query KenLM score
            lm_score = self.lm.score_sentence(text)
            acoustic_log = math.log(max(acoustic_prob, 1e-12))
            
            # Shallow fusion formula
            total_score = acoustic_log + self.alpha * lm_score + self.beta * word_count
            
            if total_score > best_total_score:
                best_total_score = total_score
                best_sentence = text
                
        return best_sentence
```
""",
        "lab": """# Practical Lab 23: Executing a 2D Grid Search for Optimal Alpha and Beta

## Objective
1. Simulate candidate ASR beam hypotheses with acoustic scores and candidate text.
2. Implement a 2D grid search over $\\alpha \\in [0.1, 0.9]$ and $\\beta \\in [0.5, 2.5]$.
3. Identify the parameter pair that minimizes Word Error Rate on a simulated validation batch.

## Laboratory Code

```python
def run_lab():
    print("=== Step 1: Simulate N-Best Hypotheses for 3 Utterances ===")
    # Each utterance has 2 candidate transcriptions with (text, acoustic_log, lm_score)
    validation_data = [
        {
            "reference": "کٲشُر زَبان",
            "candidates": [
                ("کٲشُر زَبان", -2.1, -1.8),  # Correct (high LM, moderate AM)
                ("کاشُر زَبان", -1.8, -4.2),  # Phonetic error (higher AM, terrible LM)
            ]
        },
        {
            "reference": "واریاہ کَتھٕ",
            "candidates": [
                ("واریاہ کَتھٕ", -3.2, -2.1), # Correct
                ("واریاہ",      -1.5, -3.8), # Truncated (favored if beta is too small)
            ]
        }
    ]
    
    print("=== Step 2: Run 2D Grid Search over (alpha, beta) ===")
    from src.evaluation.metrics import word_error_rate
    
    best_wer = float('inf')
    best_params = (None, None)
    
    for alpha in [0.0, 0.3, 0.5, 1.0]:
        for beta in [0.0, 1.0, 1.5, 2.5]:
            total_sub, total_del, total_ins, total_ref = 0, 0, 0, 0
            
            for item in validation_data:
                # Pick winning hypothesis under (alpha, beta)
                scored = []
                for text, am_log, lm_log in item["candidates"]:
                    words = len(text.split())
                    score = am_log + alpha * lm_log + beta * words
                    scored.append((score, text))
                scored.sort(key=lambda x: x[0], reverse=True)
                prediction = scored[0][1]
                
                # Compute simple WER
                ref_words = item["reference"].split()
                pred_words = prediction.split()
                if ref_words != pred_words:
                    total_sub += 1
                total_ref += len(ref_words)
                
            wer = (total_sub / total_ref) * 100
            if wer < best_wer:
                best_wer = wer
                best_params = (alpha, beta)
            print(f"Alpha: {alpha:4.1f} | Beta: {beta:4.1f} -> Simulated WER: {wer:5.2f}%")
            
    print(f"\n[OPTIMAL CONFIGURATION] Alpha: {best_params[0]}, Beta: {best_params[1]} with WER: {best_wer:.2f}%")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 23

## Analytical Problems
1. Prove that dividing neural network logits by temperature $T > 1.0$ strictly increases the Shannon entropy $H(Y) = -\\sum_k y_k \\log y_k$ of the resulting softmax probability distribution.
2. Explain why setting word bonus $\\beta < 0$ causes the decoder to exhibit severe word deletion errors, while setting $\\beta > 5.0$ causes insertion errors (hallucinating extra short words).

## Implementation Problems
3. Implement a function `temperature_scaled_softmax(logits: torch.Tensor, T: float = 1.08) -> torch.Tensor` using numerically stable `torch.logsumexp`.

## Forensic Auditing
4. In Stage 27 of the EduCasheer repository, the audit team investigated why Model E22-01 produced $24.74\\%$ WER under decoder config $(\\alpha=0.3, \\beta=1.0)$ but **$24.12\\%$ WER** under $(\\alpha=0.5, \\beta=1.5, T=1.08)$. Explain the provenance of this $0.62\\%$ absolute performance difference.
""",
        "solutions": """# Solutions: Lecture 23

## Exercise 1
Let $y_k(T) = \\frac{\\exp(u_k / T)}{\\sum_j \\exp(u_j / T)}$.
The derivative of Shannon entropy $H(T) = -\\sum_k y_k(T) \\ln y_k(T)$ with respect to inverse temperature $\\beta = 1/T$ is:
$$\\frac{dH}{d\\beta} = -\\beta \\cdot \\text{Var}_{y(T)}(u)$$
where $\\text{Var}_{y(T)}(u) = \\sum_k y_k (u_k - \\bar{u})^2 \\ge 0$ is the variance of the logits under distribution $y$.
Since $\\beta = 1/T$, by the chain rule:
$$\\frac{dH}{dT} = \\frac{dH}{d\\beta} \\frac{d\\beta}{dT} = \\left( -\\frac{1}{T} \\text{Var}(u) \\right) \\left( -\\frac{1}{T^2} \\right) = \\frac{1}{T^3} \\text{Var}(u)$$
Since $T > 0$ and $\\text{Var}(u) \\ge 0$ (with strict inequality whenever logits are non-identical), $\\frac{dH}{dT} > 0$.
Thus, increasing temperature $T$ strictly increases the Shannon entropy of the emission distribution.

## Exercise 2
In the decoding objective $\\log P_\\text{AM} + \\alpha \\log P_\\text{LM} + \\beta |\\mathbf{w}|$, each additional word adds a score contribution:
$$\\Delta = \\log P_\\text{LM}(w_u \\mid w_{<u}) + \\beta$$
Since $P_\\text{LM} \\in (0, 1]$, $\\log P_\\text{LM} \\le 0$.
If $\\beta < 0$, then $\\Delta < \\log P_\\text{LM} < 0$. Emitting any word heavily penalizes the total score. A shorter sequence with 2 words will score higher than a 5-word sequence, causing the decoder to delete words to avoid penalties. Conversely, if $\\beta > 5.0$, the positive bonus $\\beta$ easily outweighs the negative log-probability $\\log P_\\text{LM} \\approx -2.0$. The decoder maximizes its score by stuffing gratuitous short words into the transcript, causing massive insertion errors.

## Exercise 3
```python
import torch

def temperature_scaled_softmax(logits: torch.Tensor, T: float = 1.08) -> torch.Tensor:
    scaled = logits / T
    log_probs = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
    return torch.exp(log_probs)
```
""",
        "quiz": """# Comprehension Quiz: Lecture 23

1. **What is the primary cause of probability overconfidence in neural acoustic models?**
   - A) Low GPU memory.
   - B) Training with cross-entropy/CTC loss pushes logit margins far beyond what is necessary for classification, yielding extreme probabilities near 1.0. [CORRECT]
   - C) Audio files recorded in stereo.
   - D) Language model perplexity.
   *Explanation*: Softmax loss continues to push logits toward $\\pm \\infty$ even after correct classification, causing overconfidence.

2. **What does acoustic temperature scaling ($T=1.08$) accomplish prior to beam search?**
   - A) It heats up the GPU TensorCores.
   - B) It softens the peaked emission distribution, restoring calibration and allowing the language model prior to effectively re-rank candidates. [CORRECT]
   - C) It deletes silent audio frames.
   - D) It normalizes Unicode characters.
   *Explanation*: Temperature scaling smooths the probability distribution to allow balanced fusion with the LM.

3. **In Shallow Fusion, what happens if the language model weight $\\alpha$ is set too high ($> 2.0$)?**
   - A) The model hallucinates grammatically fluent sentences that disregard the actual spoken acoustics. [CORRECT]
   - B) The training loop crashes with an OOM.
   - C) Audio files are resampled to 8 kHz.
   - D) Word Error Rate becomes exactly 0%.
   *Explanation*: An excessively high $\\alpha$ causes the language model to dominate and ignore acoustic evidence.

4. **What is the purpose of the word insertion bonus $\\beta$ in shallow fusion decoding?**
   - A) To reward longer hypotheses and counteract the inherent penalty of multiplying probabilities $< 1.0$, preventing word deletions. [CORRECT]
   - B) To pay crowdsourced workers.
   - C) To calculate learning rate warmup.
   - D) To compress the KenLM binary file.
   *Explanation*: $\\beta$ counteracts the negative log-probability decay that penalizes longer sequences.

5. **In the certified EduCasheer production decoder, what were the optimal parameters $(\\alpha, \\beta, T)$ for Model E22-01?**
   - A) $\\alpha=0.0, \\beta=0.0, T=1.00$
   - B) $\\alpha=0.50, \\beta=1.50, T=1.08$ [CORRECT]
   - C) $\\alpha=5.0, \\beta=10.0, T=2.00$
   - D) $\\alpha=-1.0, \\beta=0.0, T=0.50$
   *Explanation*: Stage 22 grid search established $\\alpha=0.50, \\beta=1.50, T=1.08$ as the certified configuration achieving 24.12% WER.
""",
        "research": """# Research Connections: Lecture 23

- **Repository Stage**: `stages/stage_22/` (Decoder Tuning & Grid Search) and `src/decoding/shallow_fusion.py`.
- **The Metric Discrepancy Reconciliation**: In Stage 27, the reproducibility audit investigated why historical reports cited conflicting WER numbers ($24.74\\%$ vs $24.12\\%$) for the same Model E22-01 checkpoint. The audit proved that the checkpoint was identical: $24.74\\%$ was obtained under default $(\\alpha=0.3, \\beta=1.0, T=1.0)$, while **$24.12\\%$** was the true certified performance under calibrated $(\\alpha=0.5, \\beta=1.5, T=1.08)$.
- **Key Code File**: `src/decoding/shallow_fusion.py`.
- **Case Study Reference**: Case Study CS-15 ("The Sequence Optimization Illusion and Decoder Calibration").
""",
        "refs": """# References: Lecture 23

1. Guo, C., et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
2. Hannun, A., et al. (2014). *Deep Speech: Scaling up speech recognition*. arXiv:1412.5567.
3. Chorowski, J., & Jaitly, N. (2017). *Towards better decoding and language model integration in sequence to sequence models*. Interspeech.
"""
    },
    {
        "num": 24,
        "mod": "module_08_benchmarking_and_evaluation",
        "dir": "lecture_24_levenshtein_distance_wer_cer",
        "title": "Levenshtein Distance, Word & Character Error Rate",
        "subtitle": "Dynamic Programming Edit Distance, Substitution/Deletion/Insertion Accounting, and Certified Evaluation Engines",
        "objectives": [
            "Derive the Levenshtein dynamic programming algorithm for minimum edit distance.",
            "Formulate the exact definitions of Word Error Rate (WER) and Character Error Rate (CER).",
            "Analyze the pathological case where WER exceeds $100\\%$ due to insertion errors ($I > N$).",
            "Implement a high-performance Levenshtein evaluation engine in Python."
        ],
        "notes": """# Lecture 24: Levenshtein Distance, Word & Character Error Rate

## 1. Quantifying Speech Recognition Accuracy

In speech recognition, evaluation cannot be performed via simple binary accuracy because predicted sequences $\\hat{\\mathbf{w}}$ and reference sequences $\\mathbf{w}^*$ frequently differ in length.

To quantify performance, the speech community relies on **Minimum Edit Distance** (Levenshtein, 1965), which measures the minimum number of single-token editing operations required to transform the hypothesis into the reference:
1. **Substitution ($S$)**: Replacing an incorrect token with the correct token.
2. **Deletion ($D$)**: Inserting a missed token that was omitted in the hypothesis.
3. **Insertion ($I$)**: Deleting an extra token that was hallucinated in the hypothesis.

## 2. Formal Definition of WER and CER

Let $\\mathbf{w}^* = (w_1^*, \\dots, w_N^*)$ be the reference word sequence of length $N$.
Let $\\hat{\\mathbf{w}} = (\\hat{w}_1, \\dots, \\hat{w}_M)$ be the predicted word sequence of length $M$.

The **Word Error Rate (WER)** is defined as:
$$\\text{WER} = \\frac{S + D + I}{N} = \\frac{S + D + I}{S + D + C}$$
where:
- $S$ is the total count of word substitutions.
- $D$ is the total count of word deletions.
- $I$ is the total count of word insertions.
- $C$ is the count of correct words ($N = S + D + C$).

### Can WER Exceed 100%?
Yes! Unlike classification error which is bounded in $[0, 1]$, WER is **unbounded above**:
$$\\text{WER} \\in [0, \\infty)$$
If a reference has $N = 3$ words, but an unstable or hallucinating decoder emits $M = 15$ words, the number of insertions can be $I = 12$, yielding:
$$\\text{WER} = \\frac{0 + 0 + 12}{3} = \\frac{12}{3} = 400\\%$$

### Character Error Rate (CER)
In morphologically rich languages like Kashmiri, a single substituted diacritic or inflectional suffix causes an entire word to be counted as an error under WER.
**Character Error Rate (CER)** applies the exact same Levenshtein formula at the character level:
$$\\text{CER} = \\frac{S_\\text{char} + D_\\text{char} + I_\\text{char}}{N_\\text{char}}$$
CER provides a fine-grained metric of phonetic accuracy that is less sensitive to compounding morphological inflections.

## 3. Levenshtein Dynamic Programming Recurrence

Let $D(i, j)$ be the minimum edit distance between prefix $w_{1:i}^*$ and $\\hat{w}_{1:j}$.
The dynamic programming recurrence is:

$$D(i, j) = \\begin{cases} i & j = 0 \\\\ j & i = 0 \\\\ D(i-1, j-1) & w_i^* = \\hat{w}_j \\\\ 1 + \\min \\begin{cases} D(i-1, j) & (\\text{Deletion}) \\\\ D(i, j-1) & (\\text{Insertion}) \\\\ D(i-1, j-1) & (\\text{Substitution}) \\end{cases} & w_i^* \\ne \\hat{w}_j \\end{cases}$$

Time complexity: $O(N \\cdot M)$, space complexity: $O(N \\cdot M)$ (reducible to $O(\\min(N, M))$).
""",
        "script": """### Lecture 24 Instructor Delivery Script

**[00:00 - 10:00] The Standard Currency of Speech Recognition**
- "Welcome to Module 08. Today we study the single most important number in our field: Word Error Rate (WER).
  If you build an ASR system, whether for Siri, Google, or EduCasheer Kashmiri, your performance is judged by WER. But what is it really?"
- *Action*: Write the formula on the board: $\\text{WER} = (S + D + I) / N$.

**[10:00 - 25:00] The Levenshtein Dynamic Programming Grid**
- "How do you align two strings of different lengths? You can't just subtract them. You use Vladimir Levenshtein's 1965 dynamic programming algorithm.
  Draw a 2D matrix on the board: Reference down the rows, Hypothesis across the columns.
  Show how a diagonal step means Match or Substitution; vertical means Deletion; horizontal means Insertion."

**[25:00 - 35:00] The Myth of 100% Max Error**
- "Can you get 150% WER?
  Students always think 100% is the worst possible score. If the reference is 'hello world' ($N=2$) and your model hallucinates 10 words, your insertions $I=8$. Your WER is $(2 + 8)/2 = 500\\%$!
  In low-resource Kashmiri, hallucinating models easily exceed 100% WER."

**[35:00 - 45:00] Code Walkthrough**
- "Open `src/evaluation/metrics.py`. We will inspect the certified evaluation engine used to calculate our official research benchmarks."
""",
        "math": """# Mathematical Foundations: Levenshtein Distance & Metric Formulations

## 1. Dynamic Programming Formulation of Edit Distance
Let reference sequence be $\\mathbf{r} = (r_1, \\dots, r_N)$ and hypothesis be $\\mathbf{h} = (h_1, \\dots, h_M)$.
Define distance matrix $\\mathbf{D} \\in \\mathbb{Z}^{(N+1) \\times (M+1)}$.

### Boundary Conditions:
$$D(i, 0) = i \\quad \\forall i \\in \\{0, \\dots, N\\}$$
$$D(0, j) = j \\quad \\forall j \\in \\{0, \\dots, M\\}$$

### General Transition ($1 \\le i \\le N, \\; 1 \\le j \\le M$):
$$\\text{cost} = \\begin{cases} 0 & r_i = h_j \\\\ 1 & r_i \\ne h_j \\end{cases}$$
$$D(i, j) = \\min \\begin{cases} D(i-1, j) + 1 & (\\text{Deletion from reference}) \\\\ D(i, j-1) + 1 & (\\text{Insertion into hypothesis}) \\\\ D(i-1, j-1) + \\text{cost} & (\\text{Substitution / Match}) \\end{cases}$$

The total edit distance is $D(N, M)$.

## 2. Micro vs. Macro Metric Aggregation
In evaluating a test corpus of $K$ utterances $\\{(\\mathbf{r}_k, \\mathbf{h}_k)\\}_{k=1}^K$:

### Micro-Averaged WER (Scientific Standard):
$$\\text{WER}_\\text{micro} = \\frac{\\sum_{k=1}^K (S_k + D_k + I_k)}{\\sum_{k=1}^K N_k}$$
This pools all edit operations across the entire test set. Utterance weight is proportional to reference length.

### Macro-Averaged WER (Sentence-Averaged):
$$\\text{WER}_\\text{macro} = \\frac{1}{K} \\sum_{k=1}^K \\frac{S_k + D_k + I_k}{N_k}$$
Macro averaging is **scientifically invalid** in speech benchmarks because short utterances (e.g. $N_k = 1$) with 1 insertion produce a $200\\%$ error that distorts the mean. Micro-averaging is mandatory.
""",
        "code": """# Code Walkthrough: Lecture 24

We inspect `src/evaluation/metrics.py`, which implements the certified Levenshtein metric engine.

```python
# src/evaluation/metrics.py
from typing import List, Tuple, Dict
import numpy as np

def compute_levenshtein_ops(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    \"\"\"
    Computes exact counts of (Substitutions, Deletions, Insertions)
    using dynamic programming edit distance.
    \"\"\"
    N, M = len(ref), len(hyp)
    # DP matrix: (N+1, M+1)
    dp = np.zeros((N + 1, M + 1), dtype=np.int32)
    
    for i in range(N + 1):
        dp[i, 0] = i
    for j in range(M + 1):
        dp[0, j] = j
        
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(
                    dp[i - 1, j],      # Deletion
                    dp[i, j - 1],      # Insertion
                    dp[i - 1, j - 1]   # Substitution
                )
                
    # Backtrack to count S, D, I
    i, j = N, M
    s, d, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            d += 1
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break
            
    return s, d, ins

def word_error_rate(references: List[str], hypotheses: List[str]) -> float:
    \"\"\"Computes certified micro-averaged Word Error Rate (%)\"\"\"
    total_s, total_d, total_i, total_n = 0, 0, 0, 0
    for r, h in zip(references, hypotheses):
        r_words = r.strip().split()
        h_words = h.strip().split()
        s, d, ins = compute_levenshtein_ops(r_words, h_words)
        total_s += s
        total_d += d
        total_i += ins
        total_n += len(r_words)
    if total_n == 0:
        return 0.0
    return ((total_s + total_d + total_i) / total_n) * 100.0
```
""",
        "lab": """# Practical Lab 24: Implementing Levenshtein Backtracking and Alignment Visualization

## Objective
1. Implement the Levenshtein dynamic programming matrix and backtracking path.
2. Align a pair of Kashmiri sentences and visualize edit operations (S, D, I, C).
3. Compute micro-averaged WER and CER across a test set.

## Laboratory Code

```python
from src.evaluation.metrics import compute_levenshtein_ops, word_error_rate

def run_lab():
    print("=== Step 1: Align Kashmiri Word Sequences ===")
    ref_sentence = "کٲشُر زَبانہِ مَنٛز چھِ واریاہ کَتھٕ"
    hyp_sentence = "کاشُر زَبان مَنٛز چھِ واریاہ واریاہ کَتھٕ"
    
    ref_words = ref_sentence.split()
    hyp_words = hyp_sentence.split()
    
    s, d, ins = compute_levenshtein_ops(ref_words, hyp_words)
    n = len(ref_words)
    wer = ((s + d + ins) / n) * 100.0
    
    print(f"Reference:  {ref_words}")
    print(f"Hypothesis: {hyp_words}")
    print(f"Substitutions: {s} | Deletions: {d} | Insertions: {ins} | Ref Length: {n}")
    print(f"Utterance WER: {wer:.2f}%")
    
    print("\n=== Step 2: Character Error Rate (CER) ===")
    ref_chars = list(ref_sentence.replace(" ", "|"))
    hyp_chars = list(hyp_sentence.replace(" ", "|"))
    
    s_c, d_c, i_c = compute_levenshtein_ops(ref_chars, hyp_chars)
    n_c = len(ref_chars)
    cer = ((s_c + d_c + i_c) / n_c) * 100.0
    print(f"Ref Char Count: {n_c} | Sub: {s_c} | Del: {d_c} | Ins: {i_c}")
    print(f"Utterance CER:  {cer:.2f}%")
    print(f"Notice: CER ({cer:.1f}%) is significantly lower than WER ({wer:.1f}%) due to partial word credit!")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 24

## Analytical Problems
1. Given reference $\\mathbf{r} = (\\text{A, B, C, D})$ and hypothesis $\\hat{\\mathbf{h}} = (\\text{A, X, C, D, E, F})$:
   - Construct the full $5 \\times 7$ Levenshtein dynamic programming distance table.
   - Trace the optimal alignment path and count $S$, $D$, and $I$.
   - Calculate the resulting Word Error Rate.
2. Prove that micro-averaged WER is mathematically equivalent to the weighted average of individual sentence WERs, weighted by each sentence's reference word count:
$$\\text{WER}_\\text{micro} = \\sum_{k=1}^K \\left( \\frac{N_k}{\\sum_j N_j} \\right) \\text{WER}_k$$

## Implementation Problems
3. Implement a function `generate_alignment_diff(ref: list[str], hyp: list[str]) -> list[str]` that outputs a side-by-side visualization with tokens marked as `[CORRECT]`, `[SUB: a -> b]`, `[DEL: a]`, or `[INS: b]`.

## Forensic Auditing
4. In Stage 23 of the EduCasheer repository, the evaluation suite was audited after discovering that macro-averaged WER was $32.4\\%$, while micro-averaged WER was $25.82\\%$. Why was macro-averaged WER inflated, and which metric must be reported in academic publications?
""",
        "solutions": """# Solutions: Lecture 24

## Exercise 1
- Reference length $N = 4$ (`A, B, C, D`).
- Hypothesis length $M = 6$ (`A, X, C, D, E, F`).
Optimal alignment:
- `A` $\\rightarrow$ `A`: Match ($C=1$)
- `B` $\\rightarrow$ `X`: Substitution ($S=1$)
- `C` $\\rightarrow$ `C`: Match ($C=2$)
- `D` $\\rightarrow$ `D`: Match ($C=3$)
- `E`: Insertion ($I_1=1$)
- `F`: Insertion ($I_2=2$)
Total operations: $S=1, D=0, I=2$. Reference words $N=4$.
$$\\text{WER} = \\frac{1 + 0 + 2}{4} = \\frac{3}{4} = 75.0\\%$$

## Exercise 2
By definition of individual utterance WER:
$$\\text{WER}_k = \\frac{S_k + D_k + I_k}{N_k}$$
Multiplying by weight $w_k = \\frac{N_k}{\\sum_{j=1}^K N_j}$:
$$w_k \\cdot \\text{WER}_k = \\frac{N_k}{\\sum_{j=1}^K N_j} \\cdot \\frac{S_k + D_k + I_k}{N_k} = \\frac{S_k + D_k + I_k}{\\sum_{j=1}^K N_j}$$
Summing across all $K$ utterances:
$$\\sum_{k=1}^K w_k \\text{WER}_k = \\frac{\\sum_{k=1}^K (S_k + D_k + I_k)}{\\sum_{j=1}^K N_j} = \\text{WER}_\\text{micro}$$
Thus, micro-averaged WER is strictly the word-count-weighted average of individual sentence WERs.

## Exercise 3
```python
def generate_alignment_diff(ref: list[str], hyp: list[str]) -> list[str]:
    # Traceback Levenshtein alignment
    from src.evaluation.metrics import compute_levenshtein_ops
    s, d, ins = compute_levenshtein_ops(ref, hyp)
    # Simple report summary
    return [f"S={s}, D={d}, I={ins}"]
```
""",
        "quiz": """# Comprehension Quiz: Lecture 24

1. **What is the mathematical definition of Word Error Rate (WER)?**
   - A) $\\frac{S + D + I}{N}$ where $N$ is the reference word count. [CORRECT]
   - B) $\\frac{\\text{Correct}}{\\text{Total}}$.
   - C) $\\frac{S + D}{I}$.
   - D) $1 - \\text{Loss}$.
   *Explanation*: WER is total substitutions, deletions, and insertions divided by reference word length.

2. **Can Word Error Rate exceed 100%?**
   - A) No, error rates cannot mathematically exceed 100%.
   - B) Yes, when the number of insertion errors exceeds the reference word count ($I > N$). [CORRECT]
   - C) Only if the audio is clipped.
   - D) Only on GPUs.
   *Explanation*: Hallucinations and excessive insertions cause WER to scale beyond 100%.

3. **Why is Character Error Rate (CER) typically lower than WER in Kashmiri speech recognition?**
   - A) Because characters are shorter than words.
   - B) Because CER awards partial credit for words where most characters are correct, whereas WER penalizes the entire word if even a single diacritic is wrong. [CORRECT]
   - C) Because CER ignores vowels.
   - D) Because CER is computed before decoding.
   *Explanation*: A single character error flags an entire word under WER, whereas CER penalizes only that character.

4. **What is the scientific standard for aggregating WER across a multi-sentence test set?**
   - A) Macro-averaging (simple average of sentence WERs).
   - B) Micro-averaging (sum of all edit operations divided by sum of all reference words). [CORRECT]
   - C) Median WER.
   - D) Maximum WER.
   *Explanation*: Micro-averaging weights errors proportionally by utterance length, avoiding distortion from ultra-short sentences.

5. **What is the time complexity of the Levenshtein dynamic programming algorithm for reference length $N$ and hypothesis length $M$?**
   - A) $O(N + M)$
   - B) $O(N \\cdot M)$ [CORRECT]
   - C) $O(2^{N+M})$
   - D) $O(1)$
   *Explanation*: Filling the $(N+1) \\times (M+1)$ table requires $O(N \\cdot M)$ operations.
""",
        "research": """# Research Connections: Lecture 24

- **Repository Stage**: `stages/stage_23/` (Levenshtein Evaluation Suite) and `src/evaluation/metrics.py`.
- **Historic Metric Audit**: In Stage 23, the team discovered that legacy third-party evaluation scripts used macro-averaging, artificially inflating reported error rates by over $6\\%$ on short prompts. Switching to the certified micro-averaged implementation in `src/evaluation/metrics.py` resolved the discrepancy and aligned the codebase with NIST SCLITE standards.
- **Key Code File**: `src/evaluation/metrics.py`.
- **Case Study Reference**: Case Study CS-05 ("Alignment Artifacts in Morphological Errors").
""",
        "refs": """# References: Lecture 24

1. Levenshtein, V. I. (1965). *Binary codes capable of correcting deletions, insertions, and reversals*. Soviet Physics Doklady.
2. NIST. (2009). *Speech Recognition Scoring Package (SCLITE)*. National Institute of Standards and Technology.
3. Klakow, D., & Peters, J. (2002). *Testing the correlation of word error rate and perplexity*. Speech Communication.
"""
    },
    {
        "num": 25,
        "mod": "module_08_benchmarking_and_evaluation",
        "dir": "lecture_25_scientific_benchmarking_methodology",
        "title": "Scientific Benchmarking Methodology",
        "subtitle": "Certified Protocols, Multi-Domain Test Sets, Reference Normalization Invariants, and Disjoint Certification",
        "objectives": [
            "Establish a certified scientific evaluation protocol that eliminates reproducibility discrepancies.",
            "Deconstruct the EduCasheer 8-Domain Disjoint Test Suite (Studio, Bolbosh, Community, WAAZ, etc.).",
            "Enforce reference text normalization invariants before computing evaluation metrics.",
            "Inspect the Stage 27 Certified Evaluation Protocol (`docs/reproducibility/CERTIFIED_EVALUATION_PROTOCOL.md`)."
        ],
        "notes": """# Lecture 25: Scientific Benchmarking Methodology

## 1. The Reproducibility Crisis in Speech Recognition

A recurring pathology in machine learning research is the **incomparability of reported benchmarks**. Two research papers evaluating the "same" model on the "same" test set frequently report vastly different Word Error Rates because of hidden evaluation discrepancies:

```
+-----------------------------------------------------------------------------------------+
|                              SOURCES OF BENCHMARK INCONSISTENCY                         |
+---------------------+---------------------------------+---------------------------------+
| Factor              | Variant A                       | Variant B                       |
+---------------------+---------------------------------+---------------------------------+
| Text Normalization  | Punctuation stripped, numbers   | Raw punctuation kept, digits    |
|                     | converted to words              | left unnormalized (WER +8%)     |
+---------------------+---------------------------------+---------------------------------+
| Test Partition      | Contaminated speaker split      | Strict zero-speaker disjoint    |
|                     | (WER falsely low: ~18%)         | evaluation (Certified: 25.82%)  |
+---------------------+---------------------------------+---------------------------------+
| Decoding Params     | Tuned (alpha=0.5, beta=1.5, T)  | Default greedy (WER +1.7%)      |
+---------------------+---------------------------------+---------------------------------+
| Audio Preprocessing | Kaiser sinc resampler           | Naive linear resampling         |
|                     |                                 | (Bolbosh aliasing: CER +4.2%)   |
+---------------------+---------------------------------+---------------------------------+
```

To achieve scientific integrity, a benchmark must specify not just a model checkpoint, but an **immutable, end-to-end evaluation contract**.

## 2. The EduCasheer 8-Domain Multi-Test Benchmark

Evaluating a speech model on a single homogeneous test set (e.g. read studio sentences) produces dangerous over-optimism.
The EduCasheer Certified Benchmark (`Stage 27`) evaluates models across **8 distinct acoustic and linguistic domains**:

```
+------------------------------------------------------------------------------------------+
|                          THE 8-DOMAIN DISJOINT EVALUATION SUITE                          |
+----+-----------------------------+-----------------------+-------------------------------+
| #  | Domain Name                 | Acoustic Environment  | Primary Linguistic Challenge  |
+----+-----------------------------+-----------------------+-------------------------------+
| 1  | Clean Kashmiri Read         | Studio, 16 kHz WAV    | Standard literary syntax      |
| 2  | Bolbosh Crowdsourced        | Mobile MP3 (22.05kHz) | Diverse smartphone mics       |
| 3  | IndicVoices Kashmiri        | Professional studio   | Dialectal phonetic variations |
| 4  | Quranic Kashmiri Exegesis   | High SNR, reverberant | Arabic loanwords, chanting    |
| 5  | Noisy Community Speech      | Street, home, vehicle | Low SNR, background chatter   |
| 6  | WAAZ Spontaneous Sermons    | Mosque PA system, echo| Rapid tempo, oratorical shift |
| 7  | Long Audio Streams (>10s)   | Multi-sentence files  | Chunk boundary stitching      |
| 8  | Short Boundary Utterances   | Short clips (<1.5s)   | CTC frame-limit feasibility   |
+----+-----------------------------+-----------------------+-------------------------------+
```

A model is only certified for production if it demonstrates robust convergence across all 8 domains.

## 3. Reference Normalization Invariants

The fundamental rule of fair evaluation:
$$\\text{Both reference text } \\mathbf{w}^* \\text{ and predicted text } \\hat{\\mathbf{w}} \\text{ must undergo IDENTICAL normalization.}$$

Before computing Levenshtein edit distance:
1. Strip all non-printing characters (ZWNJ, ZWJ, LTR, RTL).
2. Apply canonical homoglyph substitution (Arabic Kaf $\\rightarrow$ Kashmiri Kaf).
3. Collapse multiple whitespace and strip leading/trailing spaces.
4. Remove punctuation marks (`. , ! ? - : " ' ;`).
5. Standardize numeric digits into Persian/Kashmiri digit representations.
""",
        "script": """### Lecture 25 Instructor Delivery Script

**[00:00 - 10:00] The Benchmark Chaos**
- "In 2024, if you read three different papers on Kashmiri ASR, Paper A claims 18% WER. Paper B claims 24%. Paper C claims 35%. Which model is better?
  You look under the hood: Paper A tested on their friends who were also in the training set! Paper B stripped punctuation, while Paper C left commas in the reference! It is complete apples-to-oranges chaos."

**[10:00 - 25:00] The 8-Domain Gauntlet**
- "In the EduCasheer project, we created the 8-Domain Benchmark.
  Look at the list on the board. We don't just test clean studio speech. We test Bolbosh smartphone audio. We test WAAZ religious sermons recorded on echoing mosque loudspeakers! We test short clips near CTC boundaries and long 30-second streams. A model that only works in a quiet studio is a toy, not a technology."

**[25:00 - 35:00] The Evaluation Contract**
- "Write this rule down: Normalization must be identical on both sides.
  If your decoder outputs `کٲشُر` and your reference is `کٲشُر`, but the reference has an invisible Zero-Width Non-Joiner at index 2, your evaluation script reports an error! Certified evaluation mandates identical canonicalization."

**[35:00 - 45:00] Protocol Walkthrough**
- "Open `src/evaluation/benchmark.py` and inspect `docs/reproducibility/CERTIFIED_EVALUATION_PROTOCOL.md`."
""",
        "math": """# Mathematical Foundations: Domain Decomposition & Benchmark Protocols

## 1. Domain-Decomposed Word Error Rate
Let the benchmark suite $\\mathcal{D}$ be partitioned into $K$ mutually disjoint domain subsets:
$$\\mathcal{D} = \\bigcup_{k=1}^K \\mathcal{D}_k, \\quad \\mathcal{D}_j \\cap \\mathcal{D}_m = \\emptyset \\quad \\forall j \\ne m$$

For each domain $k$, let total reference word count be $N_k$, and total edit operations be $E_k = S_k + D_k + I_k$.
The domain-specific error rate is:
$$\\text{WER}_k = \\frac{E_k}{N_k}$$

The global benchmark WER is the pooled micro-average:
$$\\text{WER}_\\text{global} = \\frac{\\sum_{k=1}^K E_k}{\\sum_{k=1}^K N_k} = \\sum_{k=1}^K \\left( \\frac{N_k}{N_\\text{total}} \\right) \\text{WER}_k$$

## 2. Normalization Projection Commutativity
Let $\\mathcal{N}$ be the certified normalization operator, and let $\\mathcal{M}$ be the metric evaluation function (Levenshtein distance).
The evaluation protocol strictly guarantees:
$$\\mathcal{M}\\left( \\mathcal{N}(\\mathbf{w}^*), \\; \\mathcal{N}(\\hat{\\mathbf{w}}) \\right) = \\mathcal{M}\\left( \\mathcal{N}(\\mathbf{w}^*), \\; \\mathcal{N}(\\mathcal{N}(\\hat{\\mathbf{w}})) \\right)$$
by the idempotence property $\\mathcal{N} \\circ \\mathcal{N} = \\mathcal{N}$.
""",
        "code": """# Code Walkthrough: Lecture 25

We inspect `src/evaluation/benchmark.py`, which executes the multi-domain benchmark protocol.

```python
# src/evaluation/benchmark.py
from typing import Dict, List
from src.normalization.normalizer import KashmiriNormalizerV3
from src.evaluation.metrics import compute_levenshtein_ops

class CertifiedBenchmarkRunner:
    \"\"\"
    Executes the Certified 8-Domain Evaluation Protocol for Kashmiri ASR.
    Enforces identical Normalizer v3 canonicalization on predictions and references.
    \"\"\"
    def __init__(self):
        self.normalizer = KashmiriNormalizerV3()

    def evaluate_domain(self, domain_name: str, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        total_s, total_d, total_i, total_n = 0, 0, 0, 0
        
        for r_raw, h_raw in zip(references, hypotheses):
            # 1. Enforce identical certified normalization
            r_norm = self.normalizer.normalize(r_raw)
            h_norm = self.normalizer.normalize(h_raw)
            
            r_words = r_norm.split()
            h_words = h_norm.split()
            
            s, d, ins = compute_levenshtein_ops(r_words, h_words)
            total_s += s
            total_d += d
            total_i += ins
            total_n += len(r_words)
            
        wer = ((total_s + total_d + total_i) / max(1, total_n)) * 100.0
        return {
            "domain": domain_name,
            "wer": wer,
            "substitutions": total_s,
            "deletions": total_d,
            "insertions": total_i,
            "word_count": total_n
        }
```
""",
        "lab": """# Practical Lab 25: Executing Multi-Domain Benchmark Evaluation and Metric Auditing

## Objective
1. Run the `CertifiedBenchmarkRunner` across 3 distinct simulated domains.
2. Observe how domain shift impacts error rates (clean vs noisy vs religious sermons).
3. Compute domain-weighted pooled WER.

## Laboratory Code

```python
from src.evaluation.benchmark import CertifiedBenchmarkRunner

def run_lab():
    print("=== Step 1: Simulate 3-Domain Test Sets ===")
    runner = CertifiedBenchmarkRunner()
    
    # Domain 1: Clean Studio
    clean_refs = ["کٲشُر زَبانہِ مَنٛز چھِ واریاہ کَتھٕ", "آفتاب نؠرَن چھُ صُبحَس"]
    clean_hyps = ["کٲشُر زَبانہِ مَنٛز چھِ واریاہ کَتھٕ", "آفتاب نؠرَن چھُ صُبحَس"]
    
    # Domain 2: Noisy Community
    noisy_refs = ["تِم چھِ گَرَس مَنٛز روزان", "بۄ چھُس سۆکول گَژھان"]
    noisy_hyps = ["تِم چھِ گَرَس مَنٛز روزان", "بۄ چھُس سۆکول گَژھ"]  # 1 deletion
    
    # Domain 3: WAAZ Mosque Sermons
    waaz_refs = ["سۄ چھِ رَحمتُ اللّٰہ عَلیہ", "دُنیاہَس مَنٛز چھِ اِنسان مُسافِر"]
    waaz_hyps = ["سۄ چھِ رَحمت اللہ", "دُنیاہَس مَنٛز اِنسان مُسافِر"] # deletions
    
    domains = [
        ("Clean Studio", clean_refs, clean_hyps),
        ("Noisy Community", noisy_refs, noisy_hyps),
        ("WAAZ Sermons", waaz_refs, waaz_hyps)
    ]
    
    total_e, total_n = 0, 0
    print("\n=== Step 2: Domain-by-Domain Evaluation ===")
    for name, refs, hyps in domains:
        res = runner.evaluate_domain(name, refs, hyps)
        print(f"Domain: {res['domain']:18s} | WER: {res['wer']:5.2f}% | Words: {res['word_count']:2d} (S={res['substitutions']}, D={res['deletions']}, I={res['insertions']})")
        total_e += res['substitutions'] + res['deletions'] + res['insertions']
        total_n += res['word_count']
        
    pooled_wer = (total_e / total_n) * 100.0
    print(f"\nPooled Multi-Domain Benchmark WER: {pooled_wer:.2f}%")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 25

## Analytical Problems
1. Suppose Model A achieves $15.0\\%$ WER on clean studio speech ($N_1 = 5{,}000$ words) and $45.0\\%$ WER on noisy speech ($N_2 = 1{,}000$ words). Model B achieves $22.0\\%$ WER on clean speech and $24.0\\%$ WER on noisy speech.
   - Calculate the pooled micro-averaged WER for both models.
   - Which model is preferred for production deployment in real-world mobile environments, and why?
2. Explain how subtle differences in punctuation stripping (e.g. treating hyphens as word separators vs deleting hyphens) can alter reference word count $N$ and distort comparative WER numbers.

## Implementation Problems
3. Write a Python function `audit_benchmark_manifest(manifest_path: str) -> dict` that inspects an evaluation manifest and asserts that no reference text contains illegal Unicode characters, unnormalized spaces, or zero durations.

## Forensic Auditing
4. In Stage 27 of the EduCasheer repository, the audit team established the certified evaluation protocol. How did the team resolve historical discrepancies between reported WER numbers ($24.74\\%, 24.40\\%, 24.12\\%$) for Model E22-01?
""",
        "solutions": """# Solutions: Lecture 25

## Exercise 1
- For Model A:
  - Clean errors: $E_1 = 0.15 \\times 5000 = 750$.
  - Noisy errors: $E_2 = 0.45 \\times 1000 = 450$.
  - Total errors: $750 + 450 = 1200$. Total words: $5000 + 1000 = 6000$.
  $$\\text{WER}_A = \\frac{1200}{6000} = 20.00\\%$$
- For Model B:
  - Clean errors: $E_1 = 0.22 \\times 5000 = 1100$.
  - Noisy errors: $E_2 = 0.24 \\times 1000 = 240$.
  - Total errors: $1100 + 240 = 1340$. Total words: $6000$.
  $$\\text{WER}_B = \\frac{1340}{6000} = 22.33\\%$$
While Model A has a lower pooled WER ($20.0\\%$ vs $22.3\\%$) because clean speech dominates $83\\%$ of the benchmark words, Model B is vastly superior for production deployment: it is robust and stable across acoustic domains ($24\\%$ on noise), whereas Model A collapses catastrophically on noisy speech ($45\\%$ WER).

## Exercise 2
Consider the compound word "well-known".
- Strategy A (treat hyphen as space): "well known" $\\rightarrow N=2$ words.
- Strategy B (delete hyphen): "wellknown" $\\rightarrow N=1$ word.
If the reference uses Strategy A ($N=2$) and the prediction outputs Strategy B ($N=1$), Strategy A counts 1 substitution and 1 deletion ($S=1, D=1 \\implies \\text{WER} = 100\\%$). Furthermore, altering $N$ changes the denominator for all error calculations across the entire corpus. Without strict normalization standards, cross-paper metric comparisons are meaningless.

## Exercise 3
```python
import json

def audit_benchmark_manifest(manifest_path: str) -> dict:
    issues = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            entry = json.loads(line)
            if entry.get("duration", 0) <= 0:
                issues.append(f"Line {idx}: zero or negative duration")
            if not entry.get("text", "").strip():
                issues.append(f"Line {idx}: empty transcript")
    return {"valid": len(issues) == 0, "issues": issues}
```
""",
        "quiz": """# Comprehension Quiz: Lecture 25

1. **What is the primary objective of a scientific benchmarking protocol?**
   - A) To make tests run faster.
   - B) To define an immutable evaluation contract (normalization, test partitions, decoding hyperparameters) that guarantees strict reproducibility. [CORRECT]
   - C) To eliminate the need for GPUs.
   - D) To translate speech to English.
   *Explanation*: A benchmark protocol enforces exact, immutable evaluation parameters.

2. **Why is evaluating speech models across multiple acoustic domains (e.g. Studio, Community, Religious Sermons) essential?**
   - A) Because single-domain benchmarks produce false optimism and fail to detect catastrophic failure on noisy, real-world acoustics. [CORRECT]
   - B) Because PyTorch requires 8 domains.
   - C) Because all languages have 8 dialects.
   - D) To reduce the number of test files.
   *Explanation*: Multi-domain benchmarks reveal domain shift degradation.

3. **What is the fundamental rule of text normalization in ASR benchmarking?**
   - A) Punctuation must only be removed from predictions.
   - B) Both reference and predicted transcripts must undergo strictly identical canonical normalization before computing Levenshtein metrics. [CORRECT]
   - C) Transcripts must be translated to Latin script.
   - D) All vowels must be deleted.
   *Explanation*: Asymmetric normalization introduces spurious edit penalties.

4. **In the EduCasheer 8-Domain Benchmark, why is the WAAZ sermon domain particularly challenging?**
   - A) It is recorded in English.
   - B) It features rapid spontaneous speech, heavy room reverberation ($RT_{60} > 1.5\\text{s}$), and loud mosque PA loudspeaker distortion. [CORRECT]
   - C) The files have no audio.
   - D) The sampling rate is 96 kHz.
   *Explanation*: High reverberation and spontaneous oratorical delivery create severe acoustic challenges.

5. **Why should speech recognition benchmarks never use random train/test splits?**
   - A) Because random numbers cannot be generated in Python.
   - B) Because random splitting causes speaker leakage, allowing models to memorize speakers' vocal tract acoustics and reporting falsely low WERs. [CORRECT]
   - C) Because random splits make the file size larger.
   - D) Because CTC loss only works on sorted data.
   *Explanation*: Disjoint speaker partitioning is required to test true acoustic generalization.
""",
        "research": """# Research Connections: Lecture 25

- **Repository Stage**: `stages/stage_27/` (Reproducibility & Parity Certification), `docs/reproducibility/CERTIFIED_EVALUATION_PROTOCOL.md`, and `src/evaluation/benchmark.py`.
- **Historic Reconciliation**: In Stage 27, the team audited the complete research history across 37 experiment directories. The audit established that the certified production performance of Model E22-01 across the 8-domain test suite is **$24.12\\%$ WER**, reconciling all previous conflicting numbers into a single immutable, evidence-backed certification.
- **Key Code File**: `src/evaluation/benchmark.py`.
- **Case Study Reference**: Case Study CS-15 ("Certified Mathematical Parity and The Final Benchmark").
""",
        "refs": """# References: Lecture 25

1. Pine, A., et al. (2022). *Requirements for Ethical and Reproducible Speech Recognition in Indigenous Languages*. ACL Findings.
2. Tatman, R. (2017). *Gender and Dialect Bias in YouTube's Automatic Speech Recognition*. EthNLP.
3. EduCasheer Project. (2026). *Certified Evaluation Protocol and Multi-Domain Benchmark Specification*.
"""
    },
    {
        "num": 26,
        "mod": "module_08_benchmarking_and_evaluation",
        "dir": "lecture_26_statistical_significance_bootstrap",
        "title": "Statistical Significance & Bootstrap Tests",
        "subtitle": "Non-Parametric Bootstrap Confidence Intervals, Paired Wilcoxon Signed-Rank Test, and Scientific Rigor",
        "objectives": [
            "Formalize statistical hypothesis testing for speech recognition benchmarks.",
            "Implement Non-Parametric Bootstrap resampling to compute $95\\%$ confidence intervals for WER.",
            "Formulate the Paired Wilcoxon Signed-Rank Test to determine if an acoustic improvement is statistically significant.",
            "Analyze the risks of $p$-hacking and ungrounded claims in low-resource speech research."
        ],
        "notes": """# Lecture 26: Statistical Significance & Bootstrap Tests

## 1. The Myth of the Raw Metric Comparison

In academic speech literature, researchers frequently report claims such as:
> *"Our proposed model achieves $24.8\\%$ WER, outperforming the baseline model's $25.2\\%$ WER by $0.4\\%$ absolute!"*

Is a $0.4\\%$ difference meaningful?
- If the test set contains only $1{,}000$ words, a $0.4\\%$ reduction represents **only 4 words** out of 1,000!
- A single speaker clearing their throat or a slight variation in random seed could account for 4 words.
- Without **Statistical Significance Testing**, claiming that Model A is superior to Model B is scientifically unfounded.

## 2. Non-Parametric Bootstrap Confidence Intervals

How do we compute a $95\\%$ Confidence Interval (CI) for Word Error Rate when the distribution of errors across sentences is non-normal and heavily skewed?

**Efron's Non-Parametric Bootstrap** (Efron, 1979; Bisani & Ney, 2004) estimates the sampling distribution of WER via empirical resampling with replacement:

```
Original Test Set D = {u_1, u_2, ..., u_K} (K sentences)
       |
       v (Sample K sentences WITH REPLACEMENT, B=2000 times)
Bootstrap Replicates: D*_1, D*_2, ..., D*_B
       |
       v (Compute WER on each replicate)
Bootstrap WERs: { WER(D*_1), WER(D*_2), ..., WER(D*_B) }
       |
       v (Sort distribution and extract 2.5th and 97.5th percentiles)
95% Confidence Interval: [ WER_0.025,  WER_0.975 ]
```

### Interpretation
If Model E22-01 reports $24.12\\%$ WER with a $95\\%$ bootstrap confidence interval of $[23.41\\%, 24.85\\%]$, we have $95\\%$ statistical confidence that the true population error rate lies within this boundary.

## 3. The Paired Wilcoxon Signed-Rank Test

To determine whether the improvement of Model B over Model A on a shared test set is statistically significant, we use the **Paired Wilcoxon Signed-Rank Test** (a non-parametric alternative to the paired Student's $t$-test that does not assume normal error distributions).

Let $d_i = L(A, u_i) - L(B, u_i)$ be the difference in edit operations between Model A and Model B on utterance $i$.
1. Exclude ties ($d_i = 0$).
2. Rank absolute differences $|d_i|$ from smallest to largest: $\\text{Rank}(|d_i|)$.
3. Compute the sum of positive ranks $W^+$ and negative ranks $W^-$:
   $$W^+ = \\sum_{d_i > 0} \\text{Rank}(|d_i|), \\quad W^- = \\sum_{d_i < 0} \\text{Rank}(|d_i|)$$
4. The test statistic is $W = \\min(W^+, W^-)$.

For large sample sizes ($K > 20$), the normalized test statistic $Z$ follows a standard normal distribution:
$$Z = \\frac{W - \\frac{N_r(N_r + 1)}{4}}{\\sqrt{\\frac{N_r(N_r + 1)(2N_r + 1)}{24}}} \\sim \\mathcal{N}(0, 1)$$
If $p < 0.01$, the null hypothesis (that Model A and Model B perform identically) is rejected with $99\\%$ confidence.
""",
        "script": """### Lecture 26 Instructor Delivery Script

**[00:00 - 10:00] The Danger of the Single Number**
- "Welcome to Lecture 26. Today we learn how to tell truth from noise in empirical machine learning.
  Suppose you train a model and get 24.1% WER. Your friend trains a model and gets 24.5%. Did you discover a breakthrough, or were you just lucky on 3 sentences? If you don't run significance tests, you have no idea."

**[10:00 - 25:00] Bootstrap Resampling Intuition**
- "How do you calculate error bars on speech recognition? You can't use standard deviation because sentence errors are not Gaussian!
  We use the Bootstrap. Take your test set of 1,000 sentences. Draw 1,000 sentences at random WITH REPLACEMENT. Some sentences get picked twice; some get skipped. Compute the WER. Repeat that 2,000 times!
  Plot the histogram of those 2,000 runs. Take the 2.5% tail and the 97.5% tail. That is your 95% Confidence Interval!"

**[25:00 - 35:00] The Paired Wilcoxon Test**
- "Why paired? Because Model A and Model B were tested on the EXACT SAME AUDIO!
  On utterance 12, did Model B do better than Model A? We rank the differences. If Model B consistently beats Model A across hundreds of sentences, the $p$-value drops below 0.001. That is how you prove a scientific claim."

**[35:00 - 45:00] Code Walkthrough**
- "Open `src/evaluation/statistics.py`. We will inspect our bootstrap and Wilcoxon implementations and run a live test."
""",
        "math": """# Mathematical Foundations: Bootstrap Confidence Limits & Wilcoxon Normal Approximation

## 1. Non-Parametric Bootstrap Resampling
Let test dataset be $\\mathcal{D} = \\{u_1, u_2, \\dots, u_K\\}$.
Each sample $u_k = (e_k, n_k)$ contains edit count $e_k$ and reference word length $n_k$.
The true sample statistic is:
$$\\theta = \\frac{\\sum_{k=1}^K e_k}{\\sum_{k=1}^K n_k}$$

For bootstrap iteration $b \\in \\{1, \\dots, B\\}$ (typically $B = 2{,}000$):
Draw sample $\\mathcal{D}^{*(b)} = \\{u_1^*, \\dots, u_K^*\\}$ uniformly with replacement from $\\mathcal{D}$.
$$\\theta^{*(b)} = \\frac{\\sum_{k=1}^K e_k^{*(b)}}{\\sum_{k=1}^K n_k^{*(b)}}$$

Sort bootstrap estimates: $\\theta^{*(1)} \\le \\theta^{*(2)} \\le \\dots \\le \\theta^{*(B)}$.
The $100(1 - \\alpha)\\%$ empirical percentile confidence interval is:
$$\\left[ \\theta^{*}_{\\lfloor (\\alpha/2) B \\rfloor}, \\; \\theta^{*}_{\\lceil (1 - \\alpha/2) B \\rceil} \\right]$$
For $\\alpha = 0.05$ and $B = 2{,}000$: $[\\theta^{*}_{(50)}, \\; \\theta^{*}_{(1950)}]$.

## 2. Wilcoxon Signed-Rank Test Statistic Normal Approximation
Let $N_r$ be the number of non-tied pairs.
The mean and variance of the test statistic $W$ under the null hypothesis $H_0$ are:
$$\\mu_W = \\frac{N_r(N_r + 1)}{4}$$
$$\\sigma_W^2 = \\frac{N_r(N_r + 1)(2N_r + 1) - \\frac{1}{2} \\sum_{j=1}^G t_j(t_j - 1)(2t_j + 1)}{24}$$
where $t_j$ is the number of tied ranks in tie group $j$.

The asymptotic $Z$-score:
$$Z = \\frac{W - \\mu_W}{\\sigma_W}$$
Two-tailed $p$-value:
$$p = 2 \\left( 1 - \\Phi(|Z|) \\right)$$
where $\\Phi$ is the standard normal cumulative distribution function.
""",
        "code": """# Code Walkthrough: Lecture 26

We inspect `src/evaluation/statistics.py`, which implements the statistical significance testing suite.

```python
# src/evaluation/statistics.py
import numpy as np
from scipy import stats
from typing import List, Tuple

def compute_bootstrap_wer_ci(errors: List[int], word_counts: List[int], n_bootstraps: int = 2000, ci: float = 0.95) -> Tuple[float, float, float]:
    \"\"\"
    Computes Non-Parametric Bootstrap Confidence Intervals for WER.
    Returns:
        (mean_wer, lower_bound, upper_bound)
    \"\"\"
    e_arr = np.array(errors, dtype=np.float64)
    n_arr = np.array(word_counts, dtype=np.float64)
    K = len(e_arr)
    
    rng = np.random.default_rng(seed=42)
    boot_wers = np.empty(n_bootstraps, dtype=np.float64)
    
    for b in range(n_bootstraps):
        # Sample sentence indices with replacement
        indices = rng.integers(0, K, size=K)
        sum_e = np.sum(e_arr[indices])
        sum_n = np.sum(n_arr[indices])
        boot_wers[b] = (sum_e / max(1.0, sum_n)) * 100.0
        
    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(boot_wers, alpha * 100.0)
    upper = np.percentile(boot_wers, (1.0 - alpha) * 100.0)
    mean_wer = (np.sum(e_arr) / np.sum(n_arr)) * 100.0
    
    return float(mean_wer), float(lower), float(upper)

def compute_paired_wilcoxon(errors_a: List[int], errors_b: List[int]) -> Tuple[float, float]:
    \"\"\"
    Performs Paired Wilcoxon Signed-Rank Test between Model A and Model B.
    Returns:
        (statistic, p_value)
    \"\"\"
    diffs = np.array(errors_a) - np.array(errors_b)
    # Remove ties
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0, 1.0
    stat, p_val = stats.wilcoxon(diffs, alternative="two-sided")
    return float(stat), float(p_val)
```
""",
        "lab": """# Practical Lab 26: Computing Bootstrap Error Bars and Running Wilcoxon Significance Tests

## Objective
1. Generate synthetic sentence-level error distributions for Model B-02C and Model E22-01.
2. Compute $95\\%$ bootstrap confidence intervals for both models.
3. Execute the Paired Wilcoxon Signed-Rank Test to verify that Model E22-01 is statistically superior ($p < 0.001$).

## Laboratory Code

```python
import numpy as np
from src.evaluation.statistics import compute_bootstrap_wer_ci, compute_paired_wilcoxon

def run_lab():
    print("=== Step 1: Simulate Sentence-Level Errors (1,000 Utterances) ===")
    np.random.seed(42)
    n_utts = 1000
    word_counts = np.random.randint(4, 15, size=n_utts).tolist()
    
    # Model B-02C errors (~25.8% WER)
    errors_b02c = [int(np.random.binomial(n, 0.258)) for n in word_counts]
    
    # Model E22-01 errors (~24.1% WER) - systematically slightly better
    errors_e22 = [int(np.random.binomial(n, 0.241)) for n in word_counts]
    
    print("=== Step 2: Non-Parametric Bootstrap 95% Confidence Intervals ===")
    mean_b, low_b, high_b = compute_bootstrap_wer_ci(errors_b02c, word_counts)
    mean_e, low_e, high_e = compute_bootstrap_wer_ci(errors_e22, word_counts)
    
    print(f"Model B-02C WER: {mean_b:.2f}% | 95% CI: [{low_b:.2f}%, {high_b:.2f}%]")
    print(f"Model E22-01 WER: {mean_e:.2f}% | 95% CI: [{low_e:.2f}%, {high_e:.2f}%]")
    
    print("\n=== Step 3: Paired Wilcoxon Signed-Rank Test ===")
    stat, p_value = compute_paired_wilcoxon(errors_b02c, errors_e22)
    print(f"Wilcoxon Statistic: {stat:.1f}")
    print(f"p-value:            {p_value:.6e}")
    
    if p_value < 0.001:
        print("[VERDICT] Model E22-01 is statistically significantly superior to Model B-02C (p < 0.001)!")
    else:
        print("[VERDICT] Difference is not statistically significant.")

if __name__ == "__main__":
    run_lab()
```
""",
        "exercises": """# Exercises: Lecture 26

## Analytical Problems
1. Explain why Student's paired $t$-test is theoretically inappropriate for evaluating speech recognition differences across sentences, and why the non-parametric Wilcoxon signed-rank test is preferred.
2. In a benchmark evaluation of 500 sentences, Model A achieves $24.2\\%$ WER and Model B achieves $24.0\\%$ WER. If the paired Wilcoxon test yields $p = 0.18$, can you scientifically reject the null hypothesis that the models perform equally? Explain what conclusions may and may not be drawn.

## Implementation Problems
3. Implement a permutation test function `compute_permutation_test(errors_a: list[int], errors_b: list[int], word_counts: list[int], n_permutations: int = 5000) -> float` that estimates the empirical $p$-value by randomly swapping sentence error labels.

## Forensic Auditing
4. In Stage 27 of the EduCasheer repository, the audit verified that Model E22-01 ($24.12\\%$ WER) outperformed Model B-02C ($25.82\\%$ WER) with $p = 1.42 \\times 10^{-7}$. Why was this formal statistical certification required before approving E22-01 as the final production checkpoint?
""",
        "solutions": """# Solutions: Lecture 26

## Exercise 1
Student's $t$-test assumes that the difference variable $d_i = L_A(i) - L_B(i)$ is normally distributed. In speech recognition:
- Edit differences are discrete integers, not continuous variables.
- On $70 - 80\\%$ of sentences, both models make identical predictions, producing a massive spike at $d_i = 0$ (heavy zero-inflation).
- For short sentences, an insertion error produces extreme outliers ($+2$ or $+3$).
The distribution of differences violates normality, homoscedasticity, and symmetry assumptions. The Wilcoxon signed-rank test is non-parametric: it relies only on the ordinal ranking of differences $|d_i|$, making it robust to heavy tails, discrete counts, and zero-inflation.

## Exercise 2
With $p = 0.18$, the $p$-value is well above the standard significance threshold $\\alpha = 0.05$ (or $\\alpha = 0.01$).
- **Conclusion**: We **cannot** reject the null hypothesis. There is no statistically significant evidence that Model B is superior to Model A.
- **Scientific Interpretation**: The $0.2\\%$ difference is entirely consistent with random sampling noise. One must not claim in an academic paper that Model B represents a meaningful advancement over Model A based on this test.

## Exercise 3
```python
import numpy as np

def compute_permutation_test(errors_a, errors_b, word_counts, n_permutations=5000):
    ea = np.array(errors_a)
    eb = np.array(errors_b)
    nw = np.array(word_counts)
    total_words = np.sum(nw)
    
    observed_diff = abs(np.sum(ea) - np.sum(eb)) / total_words
    count_exceed = 0
    rng = np.random.default_rng(seed=42)
    
    for _ in range(n_permutations):
        # Randomly swap with probability 0.5
        swaps = rng.choice([True, False], size=len(ea))
        perm_a = np.where(swaps, eb, ea)
        perm_b = np.where(swaps, ea, eb)
        diff = abs(np.sum(perm_a) - np.sum(perm_b)) / total_words
        if diff >= observed_diff:
            count_exceed += 1
            
    return count_exceed / n_permutations
```
""",
        "quiz": """# Comprehension Quiz: Lecture 26

1. **What does a 95% bootstrap confidence interval [23.4%, 24.8%] for WER indicate?**
   - A) 95% of sentences have 24% WER.
   - B) Under empirical resampling of the test set, we have 95% statistical confidence that the true population WER falls within [23.4%, 24.8%]. [CORRECT]
   - C) The test set was corrupted.
   - D) The model achieved 95% accuracy.
   *Explanation*: The confidence interval bounds the true population parameter with 95% confidence.

2. **Why is the paired Wilcoxon signed-rank test preferred over the Student's t-test for comparing speech recognition models?**
   - A) Because the Wilcoxon test runs faster on GPU.
   - B) Because sentence-level edit differences violate normality assumptions due to zero-inflation and discrete integer distributions. [CORRECT]
   - C) Because Student's t-test is illegal in Linux.
   - D) Because the Wilcoxon test does not require reference text.
   *Explanation*: Non-parametric ranking avoids unrealistic Gaussian assumptions on discrete edit differences.

3. **In statistical hypothesis testing, what does a p-value of $p = 0.0001$ indicate?**
   - A) The model made an error on 1 file.
   - B) The probability of observing such an improvement by random chance under the null hypothesis is only 1 in 10,000 (statistically significant). [CORRECT]
   - C) The model took 0.0001 seconds to decode.
   - D) The test failed.
   *Explanation*: A tiny $p$-value indicates that the observed difference is extremely unlikely under the null hypothesis.

4. **In Non-Parametric Bootstrap resampling, how are test samples drawn?**
   - A) Randomly without replacement.
   - B) Uniformly with replacement from the empirical dataset. [CORRECT]
   - C) Sorted by audio duration.
   - D) Only from clean studio speech.
   *Explanation*: Efron's bootstrap draws samples uniformly with replacement.

5. **In the EduCasheer Stage 27 certification, what was the statistical significance of Model E22-01's superiority over Model B-02C?**
   - A) $p > 0.05$ (not significant)
   - B) $p = 1.42 \\times 10^{-7}$ (highly statistically significant, $p < 0.001$) [CORRECT]
   - C) It was mathematically undefined.
   - D) $p = 1.0$
   *Explanation*: Stage 27 verified $p < 10^{-6}$, formally certifying the statistical superiority of Model E22-01.
""",
        "research": """# Research Connections: Lecture 26

- **Repository Stage**: `stages/stage_27/` (Reproducibility & Parity Certification) and `src/evaluation/statistics.py`.
- **Historic Validation**: In Stage 27, computing bootstrap confidence intervals on Model E22-01 across all 8 evaluation domains yielded an overall $95\\%$ CI of $[23.68\\%, 24.56\\%]$. The Wilcoxon test against Model B-02C confirmed $p = 1.42 \\times 10^{-7}$, providing the definitive statistical proof required to certify E22-01 as the production baseline.
- **Key Code File**: `src/evaluation/statistics.py`.
- **Case Study Reference**: Case Study CS-15 ("Certified Mathematical Parity and The Final Benchmark").
""",
        "refs": """# References: Lecture 26

1. Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife*. The Annals of Statistics.
2. Bisani, M., & Ney, H. (2004). *Bootstrap estimates for confidence intervals in ASR evaluation*. Interspeech.
3. Wilcoxon, F. (1945). *Individual comparisons by ranking methods*. Biometrics Bulletin, 1(6), 80-83.
"""
    }
])

if __name__ == "__main__":
    print("Building Lectures 18 to 26...")
    for lec in LECTURES:
        build_lecture_files(lec)
    print("Lectures 18 to 26 completed successfully!")
