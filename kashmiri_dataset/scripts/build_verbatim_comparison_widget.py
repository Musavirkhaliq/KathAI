#!/usr/bin/env python3
"""
Generates an interactive before-and-after comparison widget for verbatim text alignment.
"""

import json
import base64
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/kashmiri_dataset")
OUT_HTML = Path("/home/musa/.gemini/antigravity/brain/1361f90f-1463-44e4-bf5a-a8707b9c3e7c/verbatim_alignment_comparison.html")

samples = [
    {
        "id": "quran_kashmiri_s002_seg0009",
        "title": "Surah 2:3 (4.52s)",
        "old_text": "تِم خدایس کھوژوٕنۍ بندٕہ گٔے تِمے یِم پژھ تہٕ یقین چھِ کران تِمن چیزن یِم نہٕ تِمو وُچھمٕتی چھِ",
        "strikethrough_old": "تِم خدایس کھوژوٕنۍ بندٕہ گٔے تِمے یِم پژھ تہٕ یقین چھِ کران تِمن چیزن <span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>یِم نہٕ تِمو وُچھمٕتی چھِ</span>",
        "spoken_audio": "تِم خۄدایَس کھوژوٕنۍ بنٛدٕ گٔیہِ تِمَے یِم پَژھ تہٕ یقیٖن چھِ کَران تِمَن چیٖزَن",
        "new_text": "تِم خدایس کھوژوٕنۍ بندٕہ گٔے تِمے یِم پژھ تہٕ یقین چھِ کران تِمن چیزن",
        "fix_type": "Pruned 6 Extra Words (belong to next clip)",
        "badge_color": "bg-amber-500/10 text-amber-400 border-amber-500/30"
    },
    {
        "id": "quran_kashmiri_s002_seg0125",
        "title": "Surah 2:15 (3.15s)",
        "old_text": "مہلت چُھکھ دِوان تِم چھِ پننہِ سرکشی اندر أنین ہٕندۍ پٲٹھۍ تھپہٕ دِوان",
        "strikethrough_old": "<span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>مہلت چُھکھ دِوان</span> تِم چھِ پننہِ سرکشی اندر أنین ہٕندۍ پٲٹھۍ تھپہٕ دِوان",
        "spoken_audio": "تِم چھِ پننہِ سرکٔشی انٛدر انٮ۪ن ہِنٛدۍ پٲٹھۍ تھَپہٕ دِوان",
        "new_text": "تِم چھِ پننہِ سرکشی اندر أنین ہٕندۍ پٲٹھۍ تھپہٕ دِوان",
        "fix_type": "Pruned 3 Extra Leading Words (belong to previous clip)",
        "badge_color": "bg-amber-500/10 text-amber-400 border-amber-500/30"
    },
    {
        "id": "quran_kashmiri_s001_seg0057",
        "title": "Surah 1:7 (3.70s)",
        "old_text": "بیٚیہِ نہٕ تِہنز وَتھ یِم گمراہ سپٕدۍ",
        "strikethrough_old": "بیٚیہِ نہٕ تِہنز وَتھ یِم گمراہ سپٕدۍ <span class='text-emerald-400 font-bold bg-emerald-950/40 px-1 rounded'>[+ وَتہِ نِش دٔلۍ]</span>",
        "spoken_audio": "بیٚیہِ نہٕ تِہنٛز وَتھ یِم گُمراہ سَپٔدۍ وَتہِ نِش دٔلۍ",
        "new_text": "بیٚیہِ نہٕ تِہنز وَتھ یِم گمراہ سپٕدۍ وَتہِ نِش دٔلۍ",
        "fix_type": "Added 2 Missing Words (Mirwaiz's spoken explanation)",
        "badge_color": "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
    },
    {
        "id": "quran_kashmiri_s002_seg0109",
        "title": "Surah 2:13 (2.88s)",
        "old_text": "پژھ کٔرِو خدا وپیغمبرِ خدایس تِتھہٕ پٲٹھۍ یِتھہٕ پٲٹھۍ ساروٕے لوٗکو ایمان اوٚن تہٕ پژھ کٔر",
        "strikethrough_old": "<span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>پژھ کٔرِو خدا وپیغمبرِ خدایس</span> تِتھہٕ پٲٹھۍ یِتھہٕ پٲٹھۍ ساروٕے لوٗکو ایمان اوٚن تہٕ پژھ کٔر",
        "spoken_audio": "تِتھہٕ پٲٹھۍ یِتھہٕ پٲٹھۍ ساروٕے لوٗکو ایمان اوٚن تہٕ پَژھ کٔر",
        "new_text": "تِتھہٕ پٲٹھۍ یِتھہٕ پٲٹھۍ ساروٕے لوٗکو ایمان اوٚن تہٕ پژھ کٔر",
        "fix_type": "Pruned 6 Extra Words (belong to previous clip)",
        "badge_color": "bg-amber-500/10 text-amber-400 border-amber-500/30"
    },
    {
        "id": "quran_kashmiri_s001_seg0036",
        "title": "Surah 1:4 (2.91s)",
        "old_text": "یُس مٲلک تہٕ اختیارٕہ وول چھُ جزاکہِ دُہُک ۔ جزاہُک دۅہ گوٚو قیامتُک دۅہ ییٚمہِ دۅہ رَٹن سٲری پننیٚن عملن ہُند بدلہٕ",
        "strikethrough_old": "یُس مٲلک تہٕ اختیارٕہ وول چھُ جزاکہِ دُہُک <span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>۔ جزاہُک دۅہ گوٚو قیامتُک دۅہ ییٚمہِ دۅہ رَٹن سٲری پننیٚن عملن ہُند بدلہٕ</span>",
        "spoken_audio": "یُس مٲلِک تہٕ اختِیار وول چھُ جذعکہِ دۄہُک",
        "new_text": "یُس مٲلک تہٕ اختیارٕہ وول چھُ جزاکہِ دُہُک",
        "fix_type": "Pruned 14 Extra Words (commentary belonging to next clips)",
        "badge_color": "bg-purple-500/10 text-purple-400 border-purple-500/30"
    }
]

for s in samples:
    p = BASE_DIR / f"processed/audio/quran_kashmiri/{s['id']}.wav"
    with open(p, "rb") as af:
        s["b64"] = base64.b64encode(af.read()).decode("utf-8")

json_str = json.dumps(samples, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Verbatim Text Alignment: Before & After</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .rtl-text {
      direction: rtl;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Naskh Arabic", "Scheherazade New", Tahoma, serif;
      line-height: 2.3;
      text-align: right;
    }
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] p-4 font-sans">
  <div class="max-w-4xl mx-auto space-y-6">
    <div class="border-b border-[var(--border)] pb-4">
      <div class="inline-block px-3 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/30 text-xs font-bold rounded-full mb-2">
        VERBATIM TEXT ALIGNMENT AUDIT
      </div>
      <h1 class="text-2xl font-bold tracking-tight text-[var(--foreground)]">Old Printed Book Text vs. New Verbatim Audio Text</h1>
      <p class="text-sm text-[var(--muted-foreground)]">Eliminating extra ghost words and missing spoken words to achieve 100% word-for-word alignment</p>
    </div>

    <div id="cardsList" class="space-y-6"></div>
  </div>

  <script>
    const samples = """ + json_str + """;
    const container = document.getElementById('cardsList');

    samples.forEach((s, idx) => {
      const card = document.createElement('div');
      card.className = "bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4";

      card.innerHTML = `
        <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[var(--border)] pb-3 gap-2">
          <div class="flex items-center gap-3">
            <span class="px-2.5 py-1 bg-emerald-600 text-white font-mono text-xs font-bold rounded-md">#${idx + 1}</span>
            <span class="font-bold text-base text-[var(--foreground)]">${s.title}</span>
            <span class="text-xs px-2.5 py-0.5 rounded-full border ${s.badge_color} font-medium">${s.fix_type}</span>
          </div>
          <span class="text-xs font-mono text-[var(--muted-foreground)]">${s.id}</span>
        </div>

        <!-- Audio Player -->
        <div class="bg-[var(--background)] p-3 rounded-lg border border-[var(--border)] flex items-center gap-4">
          <audio controls class="w-full h-10">
            <source src="data:audio/wav;base64,${s.b64}" type="audio/wav">
          </audio>
        </div>

        <!-- Before & After Comparison Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Old Text Box -->
          <div class="bg-red-950/10 border border-red-500/20 p-4 rounded-xl space-y-2">
            <div class="flex items-center justify-between text-xs font-bold text-red-400">
              <span>❌ OLD Written Text (Book Text):</span>
              <span class="text-[11px] font-normal text-red-400/80">Has Extra/Missing Words</span>
            </div>
            <div class="rtl-text text-lg text-[var(--foreground)]">
              ${s.strikethrough_old}
            </div>
          </div>

          <!-- New Text Box -->
          <div class="bg-emerald-950/10 border border-emerald-500/20 p-4 rounded-xl space-y-2">
            <div class="flex items-center justify-between text-xs font-bold text-emerald-400">
              <span>✅ NEW Verbatim Text (Aligned to Speech):</span>
              <span class="text-[11px] font-normal text-emerald-400/80">100% Word-for-Word Match</span>
            </div>
            <div class="rtl-text text-lg text-emerald-300 font-medium">
              ${s.new_text}
            </div>
          </div>
        </div>

        <!-- Acoustic Spoken Reality -->
        <div class="bg-[var(--background)] p-3 rounded-lg border border-[var(--border)] text-xs text-[var(--muted-foreground)] flex items-center justify-between">
          <span>🎙️ <b>What Mirwaiz actually utters:</b></span>
          <span class="rtl-text font-mono text-sm text-[var(--foreground)]">${s.spoken_audio}</span>
        </div>
      `;

      container.appendChild(card);
    });
  </script>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Generated verbatim alignment comparison widget!")
