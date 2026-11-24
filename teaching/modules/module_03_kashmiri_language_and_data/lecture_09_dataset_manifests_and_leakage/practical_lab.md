# Practical Lab: Building Disjoint Speaker Partitions

---

## Lab Objectives
1. Load a simulated multi-speaker manifest.
2. Implement greedy bin-packing to construct an 80/10/10 disjoint speaker partition.
3. Execute automated leakage detection to prove $C = 0.0$.

---

## Lab Script: `lab_disjoint_splitting.py`

```python
import json

def partition_speakers(samples, train_ratio=0.8, val_ratio=0.1):
    # Group samples by speaker
    speaker_durations = {}
    speaker_samples = {}
    for s in samples:
        spk = s["speaker_id"]
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + s["duration"]
        speaker_samples.setdefault(spk, []).append(s)
        
    total_dur = sum(speaker_durations.values())
    target_train = total_dur * train_ratio
    target_val = total_dur * val_ratio
    
    # Sort speakers by duration descending (greedy bin packing)
    sorted_speakers = sorted(speaker_durations.keys(), key=lambda k: speaker_durations[k], reverse=True)
    
    train, val, test = [], [], []
    curr_train, curr_val = 0.0, 0.0
    
    for spk in sorted_speakers:
        dur = speaker_durations[spk]
        if curr_train + dur <= target_train or (curr_val >= target_val and len(test) > 0):
            train.extend(speaker_samples[spk])
            curr_train += dur
        elif curr_val + dur <= target_val:
            val.extend(speaker_samples[spk])
            curr_val += dur
        else:
            test.extend(speaker_samples[spk])
            
    return train, val, test

# Test on synthetic data
toy_data = [
    {"audio": "1.wav", "duration": 5.0, "speaker_id": "SPK_1"},
    {"audio": "2.wav", "duration": 4.0, "speaker_id": "SPK_1"},
    {"audio": "3.wav", "duration": 8.0, "speaker_id": "SPK_2"},
    {"audio": "4.wav", "duration": 3.0, "speaker_id": "SPK_3"},
]
tr, va, te = partition_speakers(toy_data)
tr_spk = {x["speaker_id"] for x in tr}
te_spk = {x["speaker_id"] for x in te}
assert len(tr_spk.intersection(te_spk)) == 0, "Speaker leakage detected!"
print("SUCCESS: Disjoint partition verified!")
```
