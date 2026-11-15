import json
import re
import difflib

def norm_for_align(text):
    t = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    t = t.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی").replace("ہ", "ہ")
    return t

def align_subspan(spoken_text, full_verse_text):
    if not spoken_text or not full_verse_text:
        return spoken_text or full_verse_text
        
    spoken_words = spoken_text.split()
    verse_words = full_verse_text.split()
    
    if len(verse_words) <= len(spoken_words) + 1:
        return full_verse_text # Full verse roughly matches spoken length
        
    norm_spoken = [norm_for_align(w) for w in spoken_words]
    norm_verse = [norm_for_align(w) for w in verse_words]
    
    def word_sim(w1, w2):
        if w1 == w2: return 1.0
        return difflib.SequenceMatcher(None, w1, w2).ratio()
    
    matched_indices = []
    for sw in norm_spoken:
        best_j = -1
        best_sim = 0.50
        for j, vw in enumerate(norm_verse):
            sim = word_sim(sw, vw)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j != -1:
            matched_indices.append(best_j)
            
    if not matched_indices:
        return spoken_text
        
    start_j = min(matched_indices)
    end_j = max(matched_indices) + 1
    
    # If matched span covers almost everything, return full verse
    if start_j == 0 and end_j >= len(verse_words) - 1:
        return full_verse_text
        
    return " ".join(verse_words[start_j:end_j])

# Pure Arabic markers to quarantine
arabic_quarantine_patterns = [
    "وقال لزیٖن", "وقالہ لزیٖنہ", "وقالہٕ لزینہ", "بقالہٕ لزیٖنہ", "ولر رسول ولضل قربا",
    "آمنتبِللاہی صدکلل", "سد ک رسول ہن نبی", "مِن شاہدینا وشاہ رین", "ولحمد اللہ",
    "بسم لہ رہ من وحیم", "رمربسم", "صلی اللہ علیہ وسلم", "صدق الله"
]

candidates = [json.loads(line) for line in open("QURAN_NEW_ASR_CANDIDATES.jsonl")]
uncertain = [json.loads(line) for line in open("QURAN_UNCERTAIN_SPEECH.jsonl")]

refined_candidates = []
quarantined_arabic = []
subspan_adjusted_count = 0

for c in candidates:
    txt = c.get("text", "")
    asr = c.get("asr_hypothesis", "")
    full_v = c.get("canonical_verse_text", "")
    dur = c.get("duration", 1.0)
    
    # 1. Check for Arabic formula / recitation quarantine
    is_arabic = any(pat in txt or pat in asr for pat in arabic_quarantine_patterns)
    if is_arabic:
        c["category"] = "arabic_recitation"
        c["uncertainty_reason"] = "arabic_recitation_formula_quarantined"
        c["language"] = "ara"
        quarantined_arabic.append(c)
        continue
        
    # 2. Check for disproportionately long text or verse-split
    # In King Fahd translation segments, text was set to full_v, but audio is only a part
    if c.get("category") == "kashmiri_translation" and full_v:
        char_rate = len(txt) / max(dur, 0.1)
        # If character rate is unrealistic (> 18 chars/sec) or verse has more words than spoken
        if char_rate > 18.0 or len(full_v.split()) > len(asr.split()) + 3:
            aligned = align_subspan(asr, full_v)
            if aligned != txt:
                c["original_full_verse_text"] = full_v
                c["text"] = aligned
                c["normalized_text"] = aligned
                c["flags"] = "aligned_verse_subspan"
                subspan_adjusted_count += 1
                
    refined_candidates.append(c)

new_uncertain = uncertain + quarantined_arabic

print(f"Total processed candidates: {len(candidates)}")
print(f"Quarantined Arabic segments: {len(quarantined_arabic)}")
print(f"Translation segments with subspan refined (no extra text): {subspan_adjusted_count}")
print(f"Remaining Pure Kashmiri Candidates: {len(refined_candidates)}")

# Write updated manifests
with open("QURAN_NEW_ASR_CANDIDATES.jsonl", "w", encoding="utf-8") as f:
    for c in refined_candidates:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

with open("QURAN_UNCERTAIN_SPEECH.jsonl", "w", encoding="utf-8") as f:
    for u in new_uncertain:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")

cand_dur = sum(c["duration"] for c in refined_candidates)
unc_dur = sum(u["duration"] for u in new_uncertain)
print(f"Final Candidates Duration: {cand_dur/3600:.4f} hrs ({cand_dur:.2f} s)")
print(f"Final Uncertain Duration: {unc_dur/3600:.4f} hrs ({unc_dur:.2f} s)")
