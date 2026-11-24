# Practical Lab: Building and Verifying Normalizer Invariants

---

## Lab Objectives
1. Implement test cases for Kashmiri Unicode homoglyphs and decomposed central vowels.
2. Verify the mathematical idempotence property $\mathcal{N}(\mathcal{N}(s)) = \mathcal{N}(s)$.
3. Measure the reduction in out-of-vocabulary (OOV) tokens across raw crawled Kashmiri text.

---

## Lab Script: `lab_normalizer_verification.py`

```python
from src.normalization.kashmiri_normalizer import KashmiriNormalizerV3

def test_normalizer():
    # Test Case 1: Homoglyph Unification
    raw_kaf = "\u0643\u0627\u0634\u0631" # Arabic Kaf + Alef + Shin + Reh
    norm_kaf = KashmiriNormalizerV3.normalize(raw_kaf)
    assert norm_kaf.startswith("\u06a9"), "Failed to canonicalize Kaf!"
    
    # Test Case 2: Decomposed Central Vowel
    decomposed = "\u0627\u0654\u0634\u0631" # Alef + Combining Hamza
    norm_central = KashmiriNormalizerV3.normalize(decomposed)
    assert norm_central.startswith("\u0672"), "Failed to unify precomposed central vowel!"
    
    # Test Case 3: Idempotence Verification
    sample_text = "   کٲشُر   زَبان  يَه   "
    pass1 = KashmiriNormalizerV3.normalize(sample_text)
    pass2 = KashmiriNormalizerV3.normalize(pass1)
    assert pass1 == pass2, f"Idempotence violated! Pass 1: '{pass1}', Pass 2: '{pass2}'"
    
    print("SUCCESS: All normalizer invariants passed!")

if __name__ == "__main__":
    test_normalizer()
```
