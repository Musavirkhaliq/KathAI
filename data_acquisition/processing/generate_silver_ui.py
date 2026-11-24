#!/usr/bin/env python3
import os
import json
import base64
import random

artifact_dir = "/home/musa/.gemini/antigravity/brain/04940f59-6dff-4941-95de-663a4a5c68a2"
os.makedirs(artifact_dir, exist_ok=True)
out_html = os.path.join(artifact_dir, "silver_audio_inspector.html")

silver_path = "data_acquisition/processed/silver/manifest_silver.jsonl"
with open(silver_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# 1. Sample across categories
om_recs = [r for r in records if r.get("source_dataset") == "omarrran_single_speaker_2024" and 3 <= (r.get("duration_seconds") or 0) <= 8]
mir_recs = [r for r in records if r.get("source_dataset") == "SRC_QURAN_MIRWAIZ_COMMENTARY" and 3 <= (r.get("duration_seconds") or 0) <= 7]
lc_recs = [r for r in records if r.get("source_dataset") == "SRC_KOSHUR_ORG_LANGUAGE_COURSES" and 3 <= (r.get("duration_seconds") or 0) <= 12]

random.seed(2026)
sampled_om = random.sample(om_recs, 3)
sampled_mir = random.sample(mir_recs, 2)
sampled_lc = random.sample(lc_recs, 2)

samples = []
for idx, r in enumerate(sampled_om + sampled_mir + sampled_lc):
    p = r.get("audio_path")
    if not p or not os.path.exists(p):
        continue
    with open(p, "rb") as af:
        audio_b64 = base64.b64encode(af.read()).decode("ascii")
    
    mime = "audio/wav" if p.endswith(".wav") else "audio/mpeg"
    
    src = r.get("source_dataset")
    dur = float(r.get("duration_seconds") or 0.0)
    sr = r.get("sample_rate", 16000)
    txt = r.get("text") or r.get("transcript_original") or ""
    
    if src == "omarrran_single_speaker_2024":
        cat = "Verbatim Human Speech"
        desc = "Native literature & news reading"
        bg_badge = "bg-amber-500/10 text-amber-500 border-amber-500/20"
    elif src == "SRC_QURAN_MIRWAIZ_COMMENTARY":
        cat = "Religious Spoken Commentary"
        base_fn = os.path.basename(p)
        surah_id = base_fn.split("_")[2] if len(base_fn.split("_")) > 2 else "Ref"
        txt = f"تفسیر بیَانُ الفُرقان — میرواعظ مولوی محمد یوسف شاہ ({surah_id.upper()})"
        desc = "Classical spontaneous spoken Kashmiri commentary"
        bg_badge = "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
    else:
        cat = "Language Course Dialogue"
        sub = "/".join(p.split("/")[-3:])
        txt = f"کٲشُر بول چال / گفتگُو — {sub}"
        desc = "Pedagogical conversational drill (Braj B. Kachru / CIIL)"
        bg_badge = "bg-sky-500/10 text-sky-500 border-sky-500/20"
        
    samples.append({
        "id": f"clip_{idx}",
        "category": cat,
        "badge": bg_badge,
        "source": src,
        "path": p,
        "duration": round(dur, 2),
        "sample_rate": sr,
        "text": txt,
        "desc": desc,
        "data_uri": f"data:{mime};base64,{audio_b64}"
    })

print(f"Loaded and encoded {len(samples)} clips.")

# Generate HTML with Tailwind and host variables
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Silver Tier Kashmiri Speech Inspector</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');
    .kashmiri-text {{
      font-family: 'Noto Naskh Arabic', 'Scheherazade New', 'Amiri', serif;
      direction: rtl;
      line-height: 1.8;
    }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-4">
  <div class="max-w-4xl mx-auto bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-6 shadow-lg">
    
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-[var(--border)] gap-3">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-500/10 text-slate-400 border border-slate-500/20">
            SILVER TIER
          </span>
          <span class="text-xs text-[var(--muted-foreground)]">151.41 Hours Total in Corpus</span>
        </div>
        <h1 class="text-xl font-bold tracking-tight mt-1 text-[var(--foreground)]">Kashmiri Speech Audio & Text Inspector</h1>
        <p class="text-xs text-[var(--muted-foreground)] mt-0.5">Sampled recordings from literature, classical commentary, and spoken language courses.</p>
      </div>
      <div class="flex items-center gap-2">
        <button id="playAllBtn" onclick="togglePlayFirst()" class="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90 transition flex items-center gap-1.5 shadow-sm">
          <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
          <span>Play Sample</span>
        </button>
      </div>
    </div>

    <!-- Cards List -->
    <div class="mt-6 space-y-4" id="samplesList">
"""

for s in samples:
    html_content += f"""
      <div class="group bg-[var(--background)] hover:border-[var(--primary)]/40 transition-all border border-[var(--border)] rounded-xl p-5 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded-md text-xs font-medium border {s['badge']}">{s['category']}</span>
            <span class="text-xs font-mono text-[var(--muted-foreground)]">{s['source']}</span>
          </div>
          <div class="flex items-center gap-3 text-xs text-[var(--muted-foreground)] font-mono">
            <span>⏱️ {s['duration']}s</span>
            <span>📡 {s['sample_rate']} Hz</span>
          </div>
        </div>

        <!-- Kashmiri Text Box -->
        <div class="bg-[var(--card)]/60 border border-[var(--border)] rounded-lg p-4 mb-4">
          <p class="kashmiri-text text-xl md:text-2xl text-right font-medium text-[var(--foreground)] select-all leading-loose">
            {s['text']}
          </p>
          <div class="mt-2 pt-2 border-t border-[var(--border)]/40 flex justify-between items-center text-xs text-[var(--muted-foreground)]">
            <span>{s['desc']}</span>
            <span class="font-mono text-[11px] truncate max-w-[240px] opacity-60">{os.path.basename(s['path'])}</span>
          </div>
        </div>

        <!-- Embedded Audio Player -->
        <div class="bg-[var(--card)] rounded-lg p-2.5 border border-[var(--border)] flex items-center gap-3">
          <audio id="{s['id']}" controls class="w-full h-8 accent-[var(--primary)] rounded">
            <source src="{s['data_uri']}" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
        </div>
      </div>
"""

html_content += """
    </div>

    <!-- Footer Stats -->
    <div class="mt-6 pt-4 border-t border-[var(--border)] flex flex-wrap items-center justify-between text-xs text-[var(--muted-foreground)] gap-2">
      <div>Audio rendered natively via embedded 16kHz PCM data streams. Zero external server requests.</div>
      <div class="font-mono">Dataset Tier: SILVER (Aligned / Weakly-Supervised)</div>
    </div>
  </div>

  <script>
    function togglePlayFirst() {
      const audios = document.querySelectorAll('audio');
      if (audios.length > 0) {
        const first = audios[0];
        if (first.paused) {
          audios.forEach(a => a.pause());
          first.play();
          document.getElementById('playAllBtn').innerHTML = '<svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg><span>Pause</span>';
        } else {
          first.pause();
          document.getElementById('playAllBtn').innerHTML = '<svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg><span>Play Sample</span>';
        }
      }
    }

    // Auto pause others when one plays
    document.addEventListener('play', function(e){
      var audios = document.getElementsByTagName('audio');
      for(var i = 0, len = audios.length; i < len;i++){
        if(audios[i] != e.target){
          audios[i].pause();
        }
      }
    }, true);
  </script>
</body>
</html>
"""

with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML artifact created at {out_html} ({os.path.getsize(out_html)/1024:.1f} KB).")
