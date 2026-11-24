# Code Walkthrough: Graceful OOM Handling Loop

---

## 1. Inspecting OOM Recovery (`src/training/trainer.py`)

```python
import torch

def safe_forward_backward(model, batch, optimizer, scaler, ctc_loss_fn):
    try:
        with torch.cuda.amp.autocast():
            logits = model(batch["input_values"])
            log_probs = torch.log_softmax(logits.transpose(0, 1), dim=-1)
            loss = ctc_loss_fn(
                log_probs,
                batch["labels"],
                batch["input_lengths"],
                batch["target_lengths"]
            )
            
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf loss encountered! Skipping batch.")
            optimizer.zero_grad()
            return None
            
        scaler.scale(loss).backward()
        return loss.item()
        
    except torch.cuda.OutOfMemoryError:
        print("ALERT: CUDA OOM encountered! Emptying cache and skipping batch...")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        return None
```
