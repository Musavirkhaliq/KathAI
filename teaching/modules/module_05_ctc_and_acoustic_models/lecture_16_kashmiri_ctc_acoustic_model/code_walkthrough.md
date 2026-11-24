# Code Walkthrough: Acoustic Model Architecture (`src/models/wav2vec2.py`)

---

## 1. Inspecting the Kashmiri CTC Model Class

```python
import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model

class KashmiriWav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.init_weights()
        
        # Set positive blank bias
        with torch.no_grad():
            self.lm_head.bias[0] = 3.0
            
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
```
