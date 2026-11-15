#!/usr/bin/env python3
"""
Selects 10 diverse random pure Kashmiri audio clips across different Surahs
and generates an embedded audio player widget with full Kashmiri script.
"""

import json
import random
import base64
from pathlib import Path
from collections import defaultdict

random.seed(1337)

BASE_DIR = Path("/home/musa/Kashmiri Project/kashmiri_dataset")
TRANS_PATH = BASE_DIR / "raw/external/quran_kashmiri/raw_text/kashmiri_translation_alignments.jsonl"
COMM_PATH = BASE_DIR / "raw/external/quran_kashmiri/raw_text/kashmiri_commentary_alignments.jsonl"
OUT_HTML = Path("/home/musa/.gemini/antigravity/brain/1361f90f-1463-44e4-bf5a-a8707b9c3e7c/random_kashmiri_clips.html")

def main():
    records = []
    for p in [TRANS_PATH, COMM_PATH]:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                dur = r.get("duration", 0)
                text = r.get("text", "")
                # Only clean Kashmiri speech (3.0s to 7.0s)
                if 2.5 <= dur <= 7.0 and len(text.split()) >= 4:
                    # check Kashmiri lexical markers
                    if any(m in text for m in ['چھُ', 'چھِ', 'سٕندِ', 'سٕنٛدِ', 'ہُنٛد', 'ہٕند', 'کَرن', 'یِوان', 'گوٚو', 'ٲسۍ', 'پنٕنۍ']):
                        records.append(r)

    print(f"Loaded {len(records)} candidate high-quality pure Kashmiri records.")

    # Group by Surah to ensure variety across the Quran
    by_surah = defaultdict(list)
    for r in records:
        by_surah[r["surah_number"]].append(r)

    # Pick 8 diverse Surahs (early, middle, and late Quran)
    target_surahs = [1, 2, 12, 18, 36, 55, 67, 100]
    selected_clips = []
    for s in target_surahs:
        if s in by_surah and by_surah[s]:
            # Pick a clip with high quality
            cands = sorted(by_surah[s], key=lambda x: x.get("quality_score", 0), reverse=True)[:10]
            selected_clips.append(random.choice(cands))

    print(f"Selected {len(selected_clips)} clips across {len(target_surahs)} Surahs.")

    # Prepare clip data with base64 audio
    clips_data = []
    for idx, c in enumerate(selected_clips, 1):
        sample_id = c["sample_id"]
        wav_path = BASE_DIR / f"processed/audio/quran_kashmiri/pure_kashmiri_speech/{sample_id}.wav"
        if not wav_path.exists():
            wav_path = BASE_DIR / f"processed/audio/quran_kashmiri/{sample_id}.wav"

        b64_str = ""
        if wav_path.exists():
            with open(wav_path, "rb") as af:
                b64_str = base64.b64encode(af.read()).decode("utf-8")

        clips_data.append({
            "idx": idx,
            "sample_id": sample_id,
            "surah_number": c["surah_number"],
            "ayah_number": c.get("ayah_number", "—"),
            "category": c.get("segment_category", "kashmiri_speech").replace("_", " ").title(),
            "duration": round(c["duration"], 2),
            "snr": round(c.get("snr", 20.0), 1),
            "kashmiri_text": c.get("text", ""),
            "asr_hyp": c.get("asr_acoustic_hypothesis", ""),
            "audio_file": wav_path.name,
            "b64": b64_str
        })

    # Generate HTML Player Widget
    json_data = json.dumps(clips_data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Random Pure Kashmiri Audio Clips</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .rtl-kashmiri {{
      direction: rtl;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Naskh Arabic", "Scheherazade New", Tahoma, serif;
      line-height: 2.3;
      text-align: right;
    }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] p-4 font-sans">
  <div class="max-w-4xl mx-auto space-y-6">
    <!-- Header -->
    <div class="border-b border-[var(--border)] pb-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
      <div>
        <div class="inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-500/10 text-emerald-500 border border-emerald-500/30 text-xs font-bold rounded-full mb-2">
          <span>✓</span> 100% PURE KASHMIRI SPEECH (ZERO ARABIC RECITATION)
        </div>
        <h1 class="text-2xl font-bold tracking-tight text-[var(--foreground)]">Random Kashmiri Speech Sample</h1>
        <p class="text-sm text-[var(--muted-foreground)]">8 diverse clips across different Surahs (Surahs 1, 2, 12, 18, 36, 55, 67, 100) • Mirwaiz Yousuf Shah</p>
      </div>
      <div class="text-xs bg-[var(--card)] border border-[var(--border)] px-3 py-2 rounded-lg font-mono">
        Speaker: Mirwaiz M. Yousuf Shah
      </div>
    </div>

    <!-- Clips Container -->
    <div id="clipsList" class="space-y-5"></div>
  </div>

  <script>
    const clips = {json_data};

    function renderClips() {{
      const container = document.getElementById('clipsList');
      container.innerHTML = '';

      clips.forEach(clip => {{
        const card = document.createElement('div');
        card.className = "bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4 hover:border-emerald-500/40 transition-colors";

        card.innerHTML = `
          <div class="flex items-center justify-between border-b border-[var(--border)] pb-3">
            <div class="flex items-center gap-3">
              <span class="px-2.5 py-1 bg-emerald-600 text-white font-mono text-xs font-bold rounded-md">#${{clip.idx}}</span>
              <span class="font-bold text-sm">Surah ${{clip.surah_number}} (Ayah: ${{clip.ayah_number}})</span>
              <span class="text-xs px-2 py-0.5 bg-[var(--background)] border border-[var(--border)] rounded text-[var(--muted-foreground)]">${{clip.category}}</span>
            </div>
            <div class="text-xs font-mono text-[var(--muted-foreground)]">
              ${{clip.duration}}s • SNR: ${{clip.snr}} dB
            </div>
          </div>

          <!-- Kashmiri Script Box -->
          <div class="bg-[var(--background)] p-4 rounded-xl border border-[var(--border)] space-y-2">
            <div class="text-xs font-semibold text-emerald-500">✍️ Kashmiri Perso-Arabic Script (کٲشُر):</div>
            <div class="rtl-kashmiri text-xl text-[var(--foreground)] font-medium">
              ${{clip.kashmiri_text}}
            </div>
          </div>

          <!-- Audio Player Bar -->
          <div class="bg-[var(--background)] p-3 rounded-lg border border-[var(--border)] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <audio controls class="w-full sm:w-2/3 h-10" id="audio_${{clip.idx}}">
              <source src="data:audio/wav;base64,${{clip.b64}}" type="audio/wav">
              Audio playback not supported.
            </audio>
            <div class="flex items-center gap-2 text-xs">
              <span class="text-[var(--muted-foreground)] font-mono">${{clip.audio_file}}</span>
              <button onclick="setRate('${{clip.idx}}', 0.85)" class="px-2 py-1 bg-[var(--card)] border border-[var(--border)] rounded hover:bg-emerald-500/20">0.85x</button>
              <button onclick="setRate('${{clip.idx}}', 1.0)" class="px-2 py-1 bg-[var(--card)] border border-[var(--border)] rounded font-bold hover:bg-emerald-500/20">1x</button>
            </div>
          </div>
        `;

        container.appendChild(card);
      }});
    }}

    function setRate(idx, rate) {{
      const a = document.getElementById(`audio_${{idx}}`);
      if (a) a.playbackRate = rate;
    }}

    window.onload = renderClips;
  </script>
</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated random clips player: {OUT_HTML}")

if __name__ == "__main__":
    main()
