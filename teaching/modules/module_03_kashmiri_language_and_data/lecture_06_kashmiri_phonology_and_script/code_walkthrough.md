# Code Walkthrough: Vocabulary Design and Character Mapping

---

## 1. Inspecting the Certified Kashmiri Vocabulary (`configs/vocab.json`)

The vocabulary file defines the exact mapping between character graphemes and CTC output indices:

```json
{
  "<blank>": 0,
  "<pad>": 1,
  "<s>": 2,
  "</s>": 3,
  "<unk>": 4,
  "|": 5,
  "ا": 6,
  "ب": 7,
  "پ": 8,
  "ت": 9,
  "ٹ": 10,
  "ث": 11,
  "ج": 12,
  "چ": 13,
  "ح": 14,
  "خ": 15,
  "د": 16,
  "ڈ": 17,
  "ذ": 18,
  "ر": 19,
  "ڑ": 20,
  "ز": 21,
  "ژ": 22,
  "س": 23,
  "ش": 24,
  "ص": 25,
  "ض": 26,
  "ط": 27,
  "ظ": 28,
  "ع": 29,
  "غ": 30,
  "ف": 31,
  "ق": 32,
  "ک": 33,
  "گ": 34,
  "ل": 35,
  "م": 36,
  "ن": 37,
  "و": 38,
  "ہ": 39,
  "ی": 40,
  "ے": 41,
  "ٲ": 42,
  "ٳ": 43,
  "ۄ": 44,
  "ۆ": 45,
  "ؠ": 46
}
```

### Key Structural Highlights:
1. **Total Cardinality**: Exactly 47 tokens (0 to 46).
2. **Kashmiri Extensions**: Tokens 42 to 46 encode the unique Kashmiri phonemes (`ٲ`, `ٳ`, `ۄ`, `ۆ`, `ؠ`).
3. **Space Representation**: Token 5 (`|`) acts as the explicit word delimiter.
