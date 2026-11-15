import json
import re

candidates = [json.loads(line) for line in open("QURAN_NEW_ASR_CANDIDATES.jsonl")]
uncertain = [json.loads(line) for line in open("QURAN_UNCERTAIN_SPEECH.jsonl")]

# Extended Kashmiri lexical markers
kas_words = set([
    "چھُ", "چھِ", "چھَس", "چھوس", "چھس", "چھُس", "بہٕ", "سُہ", "سۄ", "تہٕ", "تِہ", "منٛز", 
    "پٮ۪ٹھ", "پٮٹھ", "ناوٕ", "سۭتۍ", "سٍتۍ", "کران", "کروُن", "ناو", "أسۍ", "اَسہِ", "تَس", 
    "تِمن", "ہُنٛد", "ہٕندِ", "ہٕنٛز", "سُنٛد", "سٕندِ", "سٕنٛز", "کَرُن", "دِیُن", "گژھُن", 
    "آسُن", "یُس", "یِم", "تِم", "تِمَن", "کٮ۪نٛہہ", "نہٕ", "پٲٹھۍ", "پٲٹھ", "روٚپہٕ", "شروٗع",
    "آو", "گوٚو", "گٔیہِ", "کور", "کوٚر", "دِیُت", "اوٚر", "تَتھ", "اَتھ", "کینٛہہ", "مےٚ", "توٚہۍ",
    "کٔمۍ", "کٔرِن", "کٔرِو", "وۄنۍ", "وَنان", "وَنوُن", "وَنُن", "کروس", "کَروس", "بوزُن", "تَمیُک",
    "مُطٲبِق", "ہٮوان", "ہٮ۪کو", "أتھی", "أتی", "پَس", "اَمِچ", "اَمِکِس", "سٲری", "پرٛون", "لۄکُٹ",
    "ترجمہٕ", "تَرجمہٕ", "تَرجمہ", "ترجمہ", "کٲشِر", "کٲشُر", "لفٕظ", "لَفٕظ", "معنے", "مطلب",
    "سِپار", "سِپارہ", "سِپارِ", "سِپارٕ", "سوٗرٕ", "سوٗرَتھ", "سورے", "آیت", "آیہِ", "آیَت"
])

arabic_specific_patterns = [
    r"اسلام علیکم", r"سلام علیکم", r"ورحمت الله", r"وبرکات",
    r"فصل ناباض", r"فضَّلنا", r"فضلنا", r"کِتاب اللہ علیکم", r"کتاب الله علیکم",
    r"سمیع العلیم", r"سمی الالیم", r"وصلالل", r"صلی اللہ علیہ",
    r"وقال لزیٖن", r"وقالہٕ", r"بقالہٕ", r"آمنتبِللاہی", r"صدکلل", r"سد ک رسول",
    r"مِن شاہدینا", r"ولحمد اللہ", r"رب العالمین", r"بسم لہ", r"بسماللاہ", r"آسم اله",
    r"وحمان وحیم", r"روحمان وحی", r"روحمانرحیٖم", r"لار وحمان",
    r"سورے نِسہُک.*کِتاب اللہ"
]

clean_kashmiri = []
quarantined = []

for c in candidates:
    txt = c.get("text", "")
    asr = c.get("asr_hypothesis", "")
    full_str = txt + " " + asr
    
    # 1. Regex check for known Arabic formulas / ayahs
    is_arabic_pattern = any(re.search(pat, full_str) for pat in arabic_specific_patterns)
    
    # 2. Kashmiri lexical density check
    tokens = set(re.findall(r"[\u0600-\u06FF]+", full_str))
    kas_hits = sum(1 for t in tokens if t in kas_words)
    
    # If explicit Arabic pattern found, or segment has 4+ words with ZERO Kashmiri hits
    if is_arabic_pattern or (len(tokens) >= 4 and kas_hits == 0):
        c["category"] = "arabic_recitation"
        c["uncertainty_reason"] = "arabic_recitation_or_formula_quarantined"
        c["language"] = "ara"
        quarantined.append(c)
    else:
        clean_kashmiri.append(c)

print(f"Total input candidates: {len(candidates)}")
print(f"Newly Quarantined Arabic segments: {len(quarantined)}")
print(f"Remaining Pure Kashmiri Candidates: {len(clean_kashmiri)}")

with open("QURAN_NEW_ASR_CANDIDATES.jsonl", "w", encoding="utf-8") as f:
    for c in clean_kashmiri:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

new_uncertain = uncertain + quarantined
with open("QURAN_UNCERTAIN_SPEECH.jsonl", "w", encoding="utf-8") as f:
    for u in new_uncertain:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")

cand_dur = sum(c["duration"] for c in clean_kashmiri)
unc_dur = sum(u["duration"] for u in new_uncertain)
print(f"Updated Candidates: {len(clean_kashmiri)} clips, {cand_dur/3600:.4f} hrs")
print(f"Updated Uncertain: {len(new_uncertain)} clips, {unc_dur/3600:.4f} hrs")
