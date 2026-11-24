# Code Walkthrough: Implementing LLRD Optimizer Groups

---

## 1. Inspecting LLRD Optimizer Setup (`src/training/optimizer.py`)

```python
import torch

def build_llrd_optimizer(model, base_lr=3e-4, decay_factor=0.75, weight_decay=0.01):
    num_layers = 24
    opt_params = []
    
    # 1. Classification head (top layer: scale factor 1.0)
    opt_params.append({
        "params": [p for p in model.lm_head.parameters() if p.requires_grad],
        "lr": base_lr,
        "weight_decay": weight_decay
    })
    
    # 2. Transformer layers from top (23) to bottom (0)
    layers = model.wav2vec2.encoder.layers
    for i, layer in enumerate(reversed(layers)):
        layer_scale = decay_factor ** i
        opt_params.append({
            "params": [p for p in layer.parameters() if p.requires_grad],
            "lr": base_lr * layer_scale,
            "weight_decay": weight_decay
        })
        
    optimizer = torch.optim.AdamW(opt_params, betas=(0.9, 0.98), eps=1e-8)
    return optimizer
```
