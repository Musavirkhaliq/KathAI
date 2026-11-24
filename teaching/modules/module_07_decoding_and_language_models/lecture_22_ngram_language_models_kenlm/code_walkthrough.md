# Code Walkthrough: KenLM Model Training and Querying

---

## 1. Building a 5-Gram Kashmiri KenLM Model via Shell CLI

```bash
# 1. Prepare normalized text corpus
python3 src/normalization/kashmiri_normalizer.py text_raw.txt > text_normalized.txt

# 2. Train ARPA language model using KenLM lmplz with Modified Kneser-Ney
lmplz -o 5 \
      --text text_normalized.txt \
      --arpa kashmiri_5gram.arpa \
      --discount_fallback \
      --prune 0 0 1 2 2

# 3. Compile ARPA into fast binary trie format
build_binary trie kashmiri_5gram.arpa kashmiri_5gram.binary
```

---

## 2. Python KenLM Query Interface

```python
import kenlm

class KashmiriLanguageModel:
    def __init__(self, binary_path: str):
        self.model = kenlm.Model(binary_path)
        
    def score_sentence(self, sentence: str) -> float:
        # Returns log10 probability
        return self.model.score(sentence, bos=True, eos=True)
        
    def get_word_prob(self, word: str, context: str) -> float:
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        # Advance state with context
        for w in context.split():
            next_state = kenlm.State()
            self.model.BaseScore(state, w, next_state)
            state = next_state
            
        out_state = kenlm.State()
        log10_p = self.model.BaseScore(state, word, out_state)
        return log10_p
```
