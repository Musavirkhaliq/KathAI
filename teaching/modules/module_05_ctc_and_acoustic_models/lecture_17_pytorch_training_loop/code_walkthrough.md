# Code Walkthrough: Production Training Loop (`src/training/trainer.py`)

---

## 1. Inspecting the Canonical Training Loop

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, dataloader, optimizer, scheduler, ctc_loss_fn, device, accum_steps=8):
    model.train()
    scaler = GradScaler()
    optimizer.zero_grad()
    
    total_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        in_lengths = batch["input_lengths"].to(device)
        lbl_lengths = batch["target_lengths"].to(device)
        
        # Forward pass under mixed precision
        with autocast():
            logits = model(input_values)  # [B, T, V]
            # PyTorch CTCLoss expects [T, B, V]
            log_probs = torch.log_softmax(logits.transpose(0, 1), dim=-1)
            loss = ctc_loss_fn(log_probs, labels, in_lengths, lbl_lengths)
            loss = loss / accum_steps
            
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps
        
        # Optimizer step every accum_steps
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
    return total_loss / len(dataloader)
```
