# Code Walkthrough: XLS-R Feature Extractor Inspection

---

## 1. Inspecting the Feature Extractor (`src/models/wav2vec2.py`)

```python
import torch
import torch.nn as nn

class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Strides: (5, 2, 2, 2, 2, 2, 2) -> Total downsampling = 320
        # Kernels: (10, 3, 3, 3, 3, 2, 2)
        conv_layers = []
        in_c = 1
        configs = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2)
        ]
        
        for out_c, k, s in configs:
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, bias=False),
                    nn.Dropout(p=0.0),
                    nn.GroupNorm(num_groups=512, num_channels=out_c, affine=True),
                    nn.GELU()
                )
            )
            in_c = out_c
            
        self.conv_layers = nn.ModuleList(conv_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 1, time_samples]
        for layer in self.conv_layers:
            x = layer(x)
        return x  # Shape: [batch, 512, time_frames]
```
