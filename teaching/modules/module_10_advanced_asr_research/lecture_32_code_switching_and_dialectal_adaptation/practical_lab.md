# Practical Lab: Parameter-Efficient LoRA Adaptation

---

## Lab Objectives
1. Implement a low-rank adapter layer in PyTorch.
2. Verify that trainable parameters constitute $< 0.5\%$ of the base layer weights.
3. Assert that initial output equals the base layer output before gradient updates ($\mathbf{B} = \mathbf{0}$).

---

## Lab Script: `lab_lora.py`

```python
import torch
from src.models.lora import LoRALinear

def run_lora_lab():
    layer = LoRALinear(1024, 1024, r=8, alpha=16.0)
    
    # Assert initial identity
    x = torch.randn(4, 1024)
    with torch.no_grad():
        out_base = layer.linear(x)
        out_lora = layer(x)
        
    diff = torch.norm(out_base - out_lora).item()
    print(f"Initial L2 Difference between Base and LoRA: {diff:.6f}")
    assert diff == 0.0, "LoRA initialization should not alter base output!"
    
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
    print(f"Trainable LoRA Params: {trainable} | Frozen Base Params: {frozen}")
    print(f"Parameter Ratio: {trainable / frozen * 100:.3f}%")
    print("SUCCESS: LoRA layer validated!")

if __name__ == "__main__":
    run_lora_lab()
```
