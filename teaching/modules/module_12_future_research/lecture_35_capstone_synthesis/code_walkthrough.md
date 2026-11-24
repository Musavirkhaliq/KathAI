# Code Walkthrough: Full Course Master API Tour

---

## 1. Unified Access to Repository Capabilities

```python
# The complete EduCasheer pipeline in 4 lines of clean Python:
from src.inference.pipeline import KashmiriASRPipeline

# 1. Initialize certified production pipeline
pipeline = KashmiriASRPipeline.from_pretrained("checkpoints/e22_01.pt")

# 2. Transcribe long-form audio with Method C2 chunking and KenLM beam search
result = pipeline.transcribe("data/audio/sample_kashmiri.wav")

print(f"Transcribed Kashmiri Text: '{result}'")
```
