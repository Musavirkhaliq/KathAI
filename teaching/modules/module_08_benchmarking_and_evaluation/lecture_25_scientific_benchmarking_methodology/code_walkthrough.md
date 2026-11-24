# Code Walkthrough: Certified Benchmark Runner (`benchmarks/run_certified_eval.py`)

---

## 1. Inspecting the 8-Domain Benchmark Harness

```python
import json
from pathlib import Path
from src.evaluation.metrics import compute_corpus_wer
from src.normalization.kashmiri_normalizer import KashmiriNormalizerV3

DOMAINS = [
    "dom01_studio_clean",
    "dom02_prompted_conv",
    "dom03_broadcast_news",
    "dom04_quranic_expository",
    "dom05_waaz_reverberant",
    "dom06_community_mobile",
    "dom07_dialectal_folk",
    "dom08_educational_lecture"
]

def evaluate_certified_benchmark(pipeline, benchmark_dir="benchmarks"):
    domain_results = {}
    tot_E, tot_N = 0, 0
    
    for dom in DOMAINS:
        manifest_path = Path(benchmark_dir) / f"{dom}.jsonl"
        refs, hyps = [], []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                norm_ref = KashmiriNormalizerV3.normalize(item["text"])
                pred_hyp = pipeline.transcribe(item["audio_filepath"])
                norm_hyp = KashmiriNormalizerV3.normalize(pred_hyp)
                
                refs.append(norm_ref)
                hyps.append(norm_hyp)
                
        wer, S, D, I, N = compute_corpus_wer(refs, hyps)
        domain_results[dom] = {"wer": wer, "S": S, "D": D, "I": I, "N": N}
        tot_E += (S + D + I)
        tot_N += N
        
    composite_wer = tot_E / max(1, tot_N)
    return composite_wer, domain_results
```
