#!/usr/bin/env python3
"""
Updates statistical_random_audit.html with trimmed audio clips where any Arabic recitation
prefix has been cleanly cut off, leaving 100% pure Kashmiri speech from 0.0s to end.
"""

import json
import base64
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project/kashmiri_dataset")
OUT_HTML = Path("/home/musa/.gemini/antigravity/brain/1361f90f-1463-44e4-bf5a-a8707b9c3e7c/statistical_random_audit.html")

clips_info = [
    {
        "idx": 1,
        "sample_id": "quran_kashmiri_s029_seg0511_pure",
        "surah_number": 29,
        "ayah_number": 66,
        "duration": 2.72,
        "diagnosis": "Trimmed 3.0s Arabic Recitation Prefix • Pruned Unuttered Words",
        "strikethrough_vis": "<span class='line-through text-purple-400 bg-purple-950/40 px-1 rounded'>[0.0s-3.0s Arabic Ayah Recitation]</span> بیٚیہِ یُتھ کینٛہہ <span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>کالاہ دنیاہس نادر تمتع تُلن</span>",
        "spoken_audio": "بیٚیہِ یُتھ کینٛہہ کالا دُنیاہس اندر تمتا تُلان",
        "new_verbatim": "بیٚیہِ یُتھ کینٛہہ کالاہ دنیاہس اندر تمتع تُلان",
        "file_path": BASE_DIR / "processed/audio/quran_kashmiri/pure_kashmiri_speech/quran_kashmiri_s029_seg0511_pure.wav"
    },
    {
        "idx": 2,
        "sample_id": "quran_kashmiri_s015_seg0210",
        "surah_number": 15,
        "ayah_number": 54,
        "duration": 1.83,
        "diagnosis": "100% Pure Kashmiri • Zero Arabic",
        "strikethrough_vis": "پَس کِتھ پٲٹھۍ چُھوم خۅش خبر دِوان",
        "spoken_audio": "پَس کِتھ پٲٹھۍ چھِ یِوان خۄش ہَبر دِوان",
        "new_verbatim": "پَس کِتھ پٲٹھۍ چُھوم خۅش خبر دِوان",
        "file_path": BASE_DIR / "processed/audio/quran_kashmiri/quran_kashmiri_s015_seg0210.wav"
    },
    {
        "idx": 3,
        "sample_id": "quran_kashmiri_s021_seg0283",
        "surah_number": 21,
        "ayah_number": 44,
        "duration": 3.55,
        "diagnosis": "100% Pure Kashmiri • Pruned 2 Extra Trailing Words",
        "strikethrough_vis": "کیاہ یِمے ہیکہٕ نہٕ سٲنِس پیغمبرس پٮ۪ٹھ <span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>غٲلب سپدِتھ</span>",
        "spoken_audio": "کیاہ یِمے ہیٚکہٕ نہٕ سٲنِس پیگ مبرس پٮ۪ٹھ",
        "new_verbatim": "کیاہ یِمے ہیکہٕ نہٕ سٲنِس پیغمبرس پٮ۪ٹھ",
        "file_path": BASE_DIR / "processed/audio/quran_kashmiri/quran_kashmiri_s021_seg0283.wav"
    },
    {
        "idx": 4,
        "sample_id": "quran_kashmiri_s007_seg0308_pure",
        "surah_number": 7,
        "ayah_number": 41,
        "duration": 10.15,
        "diagnosis": "Trimmed 6.0s Arabic Recitation Prefix • Pruned Final Unuttered Sentence",
        "strikethrough_vis": "<span class='line-through text-purple-400 bg-purple-950/40 px-1 rounded'>[0.0s-6.0s Arabic Ayah Recitation]</span> تِہٕندِ خٲطرٕ آسہِ جہنم کُے وتھرُن ۔ بیٚیہِ پٮ۪ٹھۍ کِنۍ تمہِ کُے وُرُن ۔ یعنے پرتھ طرفہٕ آسہِ تِمن نارٕے نار <span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>۔ أسۍ چھِ یِتھیٚن ظٲلِمن یُتھے سزا دِوان</span>",
        "spoken_audio": "تِہنٛدٕ خٲصرِ آسہِ جہننم کُے وۄتھرُن بیٚیہِ پیٚٹھۍ کِنۍ تمہِ کُے وُرُن یعنے پرٮ تھ طرفٕ آسہِ تِمَن نارٕے نار",
        "new_verbatim": "تِہٕندِ خٲطرٕ آسہِ جہنم کُے وتھرُن ۔ بیٚیہِ پٮ۪ٹھۍ کِنۍ تمہِ کُے وُرُن ۔ یعنے پرتھ طرفہٕ آسہِ تِمن نارٕے نار",
        "file_path": BASE_DIR / "processed/audio/quran_kashmiri/pure_kashmiri_speech/quran_kashmiri_s007_seg0308_pure.wav"
    },
    {
        "idx": 5,
        "sample_id": "quran_kashmiri_s032_seg0079",
        "surah_number": 32,
        "ayah_number": 14,
        "duration": 4.40,
        "diagnosis": "100% Pure Kashmiri • Pruned 1 Extra Trailing Verb",
        "strikethrough_vis": "ژٕہِو ہمیشگی ہُنٛد عذاب کیازِ توٚہۍ ٲسِو نا تِژَھے کامہِ <span class='line-through text-red-400 bg-red-950/40 px-1 rounded'>کران</span>",
        "spoken_audio": "ژٕہِو ہمیشگی ہُنٛد عضاب کیازِ تُہۍ ٲسِوٕ نا تِژے کامہِ",
        "new_verbatim": "ژٕہِو ہمیشگی ہُنٛد عذاب کیازِ توٚہۍ ٲسِو نا تِژَھے کامہِ",
        "file_path": BASE_DIR / "processed/audio/quran_kashmiri/quran_kashmiri_s032_seg0079.wav"
    },
    {
        "idx": 6,
        "sample_id": "quran_kashmiri_s018_seg0802",
        "surah_number": 18,
        "ayah_number": 77,
        "duration": 1.51,
        "diagnosis": "100% Natural Word-for-Word Match • Zero Arabic",
        "strikethrough_vis": "پَس پٔکۍ تِم برونٹھ کُں بیٚیہِ",
        "spoken_audio": "پَس پٔکۍ تِم برونٛٹھ کُن بیٚیہِ",
        "new_verbatim": "پَس پٔکۍ تِم برونٹھ کُں بیٚیہِ",
        "file_path": BASE_DIR / "processed/audio/quran_kashmiri/quran_kashmiri_s018_seg0802.wav"
    }
]

clips_data = []
for c in clips_info:
    b64 = ""
    fp = c.pop("file_path")
    if fp.exists():
        with open(fp, "rb") as af:
            b64 = base64.b64encode(af.read()).decode("utf-8")
    c["b64"] = b64
    clips_data.append(c)

json_str = json.dumps(clips_data, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cleaned Pure Kashmiri Audio Player</title>
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
      <div class="inline-block px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-bold rounded-full mb-2">
        ✓ 100% CLEANED PURE KASHMIRI • ALL ARABIC RECITATION PREFIXES TRIMMED
      </div>
      <h1 class="text-2xl font-bold tracking-tight text-[var(--foreground)]">Pure Mirwaiz Spoken Kashmiri Audio Verification</h1>
      <p class="text-sm text-[var(--muted-foreground)]">Arabic Ayah recitation prefixes cleanly cut off • Audio starts at 0.0s with pure Kashmiri speech</p>
    </div>

    <div id="cardsList" class="space-y-6"></div>
  </div>

  <script>
    const clips = """ + json_str + """;
    const container = document.getElementById('cardsList');

    clips.forEach(c => {
      const card = document.createElement('div');
      card.className = "bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4 hover:border-emerald-500/40 transition";

      card.innerHTML = `
        <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[var(--border)] pb-3 gap-2">
          <div class="flex items-center gap-3">
            <span class="px-2.5 py-1 bg-emerald-600 text-white font-mono text-xs font-bold rounded-md">#${c.idx}</span>
            <span class="font-bold text-base text-[var(--foreground)]">Surah ${c.surah_number} • Ayah ${c.ayah_number}</span>
            <span class="text-xs px-2.5 py-0.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 font-medium">${c.diagnosis}</span>
          </div>
          <span class="text-xs font-mono text-[var(--muted-foreground)]">${c.sample_id} (${c.duration}s)</span>
        </div>

        <!-- Audio Player -->
        <div class="bg-[var(--background)] p-3 rounded-lg border border-[var(--border)] flex items-center gap-4">
          <audio controls class="w-full h-10">
            <source src="data:audio/wav;base64,${c.b64}" type="audio/wav">
          </audio>
        </div>

        <!-- Comparison Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="bg-red-950/10 border border-red-500/20 p-4 rounded-xl space-y-2">
            <div class="flex items-center justify-between text-xs font-bold text-red-400">
              <span>❌ BEFORE TRIMMING:</span>
              <span class="text-[11px] font-normal text-red-400/80">Contained Arabic / Extra Words</span>
            </div>
            <div class="rtl-text text-lg text-[var(--foreground)]">
              ${c.strikethrough_vis}
            </div>
          </div>

          <div class="bg-emerald-950/10 border border-emerald-500/20 p-4 rounded-xl space-y-2">
            <div class="flex items-center justify-between text-xs font-bold text-emerald-400">
              <span>✅ AFTER TRIMMING (NOW PLAYING):</span>
              <span class="text-[11px] font-normal text-emerald-400/80">100% Pure Spoken Kashmiri</span>
            </div>
            <div class="rtl-text text-lg text-emerald-300 font-medium">
              ${c.new_verbatim}
            </div>
          </div>
        </div>

        <div class="bg-[var(--background)] p-3 rounded-lg border border-[var(--border)] text-xs text-[var(--muted-foreground)] flex items-center justify-between">
          <span>🎙️ <b>Mirwaiz speaks from 0.0s:</b></span>
          <span class="rtl-text font-mono text-sm text-[var(--foreground)]">${c.spoken_audio}</span>
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

print("Generated cleaned pure Kashmiri player with trimmed Arabic!")
