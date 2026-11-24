# Code Walkthrough: Production Normalizer Inspection

---

## 1. Inspecting `src/normalization/kashmiri_normalizer.py`

```python
import re
import unicodedata

class KashmiriNormalizerV3:
    # 1. Homoglyph Replacement Map
    HOMOGLYPH_MAP = {
        '\u0643': '\u06a9',  # Arabic Kaf -> Keheh
        '\u0643': '\u06a9',
        '\u064a': '\u06cc',  # Arabic Yeh -> Farsi Yeh
        '\u06d2': '\u06d2',  # Keep Barree Yeh
        '\u0647': '\u06c1',  # Arabic Heh -> Heh Goal
        '\u0629': '\u06c1',  # Teh Marbuta -> Heh Goal
        # Precomposed central vowels
        '\u0627\u0654': '\u0672',  # Alef + Hamza Above -> Precomposed ٲ
        '\u0627\u0655': '\u0673',  # Alef + Hamza Below -> Precomposed ٳ
    }
    
    # 2. Diacritics to strip (strictly excluding Kashmiri vowel markers)
    STRIP_DIACRITICS = re.compile(r'[\u064b-\u0652\u0656-\u065f\u0670]')
    
    # 3. Punctuation removal pattern
    PUNCTUATION = re.compile(r'[^\w\s\u0600-\u06ff]')
    
    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""
            
        # Step 1: Replace homoglyphs
        for src, target in cls.HOMOGLYPH_MAP.items():
            text = text.replace(src, target)
            
        # Step 2: Strip Arabic decorative diacritics
        text = cls.STRIP_DIACRITICS.sub('', text)
        
        # Step 3: Strip extraneous punctuation
        text = cls.PUNCTUATION.sub(' ', text)
        
        # Step 4: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
```
