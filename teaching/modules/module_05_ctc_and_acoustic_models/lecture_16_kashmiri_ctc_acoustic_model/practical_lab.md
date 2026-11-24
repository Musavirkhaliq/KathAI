# Practical Lab: Freezing Layers and Tracking Trainable Parameters

---

## Lab Objectives
1. Instantiate the Kashmiri CTC architecture.
2. Implement freezing functions for feature extractor and lower Transformer layers.
3. Verify parameter counts and assert that frozen weights receive zero gradients during backprop.

---

## Lab Script: `lab_model_freezing.py`

```python
import torch
import torch.nn as nn

class ToySpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=10, stride=5)
        self.fc = nn.Linear(64, 47)
        
    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        return self.fc(h)

model = ToySpeechModel()

# Freeze conv
for p in model.conv.parameters():
    p.requires_grad = False
    
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"Trainable Parameters: {trainable} | Frozen Parameters: {frozen}")

# Run forward-backward
x = torch.randn(2, 1, 100)
out = model(x).sum()
out.backward()

assert model.conv.weight.grad is None, "Frozen layer received gradients!"
print("SUCCESS: Freezing verified!")

if __name__ == "__main__":
    pass
```
