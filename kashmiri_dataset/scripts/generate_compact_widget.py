import json
import base64
import subprocess
import os
import random

target_widget = "/home/musa/.gemini/antigravity/brain/412df5c3-70ef-4580-9101-03634714aba4/kashmiri_audio_widget.html"
target_meta = target_widget + ".metadata.json"

with open("QURAN_NEW_ASR_CANDIDATES.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

random.seed(42)
kf = [r for r in records if r["source"] == "SRC_QURAN_KING_FAHD"]
war = [r for r in records if r["source"] == "SRC_QURAN_DR_WAR"]

sample_kf = random.sample(kf, 5)
sample_war = random.sample(war, 5)
samples = sample_kf + sample_war
random.shuffle(samples)

def to_mp3_base64(wav_path):
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-ar", "16000", "-ac", "1", "-b:a", "32k", "-f", "mp3", "-"]
    res = subprocess.run(cmd, capture_output=True, check=True)
    return base64.b64encode(res.stdout).decode("ascii")

sample_data = []
for i, s in enumerate(samples, 1):
    audio_path = s["audio_path"]
    b64 = to_mp3_base64(audio_path)
    sample_data.append({
        "index": i,
        "sample_id": s["sample_id"],
        "source": "King Fahd Complex" if s["source"] == "SRC_QURAN_KING_FAHD" else "Dr. Abdul Rehman War",
        "category": s["category"].replace("_", " ").title(),
        "is_trans": "translation" in s["category"].lower(),
        "duration": round(s["duration"], 2),
        "snr": round(s.get("snr", 15.0), 1),
        "text": s.get("text", ""),
        "speaker": s.get("speaker_name", s.get("speaker_id", "Unknown")),
        "tier": s.get("verification_status", "tier1_machine_detected").replace("_", " ").title(),
        "ref": f"Surah {s.get('surah_number')}:{s.get('ayah_number')}" if s.get("surah_number") else f"Episode {s.get('episode_number')}",
        "audio_b64": b64
    })

data_json = json.dumps(sample_data, ensure_ascii=False)

html_content = f"""<!DOCTYPE html>
<html lang="ks" dir="ltr">
<head>
  <meta charset="UTF-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');
    .kashmiri-text {{
      font-family: 'Noto Naskh Arabic', 'Scheherazade New', 'Amiri', Tahoma, sans-serif;
      line-height: 2.0;
      direction: rtl;
    }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2 sm:p-3">

  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-4 shadow-sm max-w-2xl mx-auto">
    <!-- Top Header & Navigation -->
    <div class="flex items-center justify-between pb-3 border-b border-[var(--border)] gap-2">
      <div class="flex items-center gap-2">
        <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
        <h2 class="text-sm font-semibold tracking-tight text-[var(--foreground)]">Kashmiri Speech & Text Inspector</h2>
      </div>

      <!-- Prev / Next & Counter -->
      <div class="flex items-center gap-1.5">
        <button onclick="prevSample()" class="p-1 rounded-lg border border-[var(--border)] bg-[var(--background)] hover:bg-[var(--accent)] text-[var(--foreground)] transition-colors" title="Previous Sample">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
        </button>
        <span id="counter" class="text-xs font-mono font-bold px-2 py-0.5 rounded bg-[var(--background)] border border-[var(--border)]">01 / 10</span>
        <button onclick="nextSample()" class="p-1 rounded-lg border border-[var(--border)] bg-[var(--background)] hover:bg-[var(--accent)] text-[var(--foreground)] transition-colors" title="Next Sample">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
        </button>
      </div>
    </div>

    <!-- Quick Pill Selector -->
    <div class="flex items-center gap-1 my-3 overflow-x-auto py-0.5" id="pills-container"></div>

    <!-- Active Sample Metadata Badges -->
    <div class="flex flex-wrap items-center justify-between gap-1.5 text-xs mb-2">
      <div class="flex items-center gap-1.5">
        <span id="badge-id" class="font-mono font-semibold text-[var(--foreground)]"></span>
        <span id="badge-cat" class="px-2 py-0.5 rounded-full font-medium"></span>
        <span id="badge-ref" class="px-2 py-0.5 rounded-full bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)]"></span>
      </div>
      <div class="flex items-center gap-2 text-[var(--muted-foreground)] font-mono text-[11px]">
        <span id="badge-dur"></span>
        <span>•</span>
        <span id="badge-snr" class="text-emerald-600 dark:text-emerald-400 font-semibold"></span>
        <span>•</span>
        <span id="badge-tier" class="text-amber-600 dark:text-amber-400"></span>
      </div>
    </div>

    <!-- Spoken Kashmiri Text Box -->
    <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3 sm:p-4 my-2 text-right">
      <div id="kashmiri-text" class="kashmiri-text text-lg sm:text-xl font-medium text-[var(--foreground)] leading-loose"></div>
    </div>

    <!-- Audio Player Controls -->
    <div class="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
      <div class="flex items-center gap-3 w-full sm:w-auto">
        <button onclick="toggleAudio()" id="play-btn" class="w-9 h-9 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white flex items-center justify-center shadow transition-transform active:scale-95 flex-shrink-0">
          <svg id="icon-play" class="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
          <svg id="icon-pause" class="w-4 h-4 hidden" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
        </button>

        <audio id="audio-player" controls class="w-full sm:w-72 h-8">
          <source id="audio-source" src="" type="audio/mp3">
        </audio>
      </div>

      <div class="text-[11px] text-[var(--muted-foreground)] flex items-center gap-1.5 ml-auto">
        <span id="badge-source" class="font-medium text-[var(--foreground)]"></span>
        <span id="badge-speaker"></span>
      </div>
    </div>
  </div>

  <script>
    const samples = {data_json};
    let currentIndex = 0;

    const pillsContainer = document.getElementById('pills-container');
    samples.forEach((s, idx) => {{
      const btn = document.createElement('button');
      btn.className = `px-2 py-0.5 rounded text-xs font-mono font-medium border transition-colors ${{idx === 0 ? 'bg-[var(--foreground)] text-[var(--background)] border-[var(--foreground)]' : 'bg-[var(--background)] text-[var(--muted-foreground)] border-[var(--border)] hover:text-[var(--foreground)]'}}`;
      btn.innerText = `#${{idx + 1}}`;
      btn.id = `pill-${{idx}}`;
      btn.onclick = () => loadSample(idx);
      pillsContainer.appendChild(btn);
    }});

    function loadSample(idx) {{
      currentIndex = idx;
      const s = samples[idx];

      document.querySelectorAll('#pills-container button').forEach((btn, i) => {{
        if (i === idx) {{
          btn.className = 'px-2 py-0.5 rounded text-xs font-mono font-medium border transition-colors bg-[var(--foreground)] text-[var(--background)] border-[var(--foreground)]';
        }} else {{
          btn.className = 'px-2 py-0.5 rounded text-xs font-mono font-medium border transition-colors bg-[var(--background)] text-[var(--muted-foreground)] border-[var(--border)] hover:text-[var(--foreground)]';
        }}
      }});

      document.getElementById('counter').innerText = `${{String(idx + 1).padStart(2, '0')}} / 10`;
      document.getElementById('badge-id').innerText = s.sample_id;
      
      const catBadge = document.getElementById('badge-cat');
      catBadge.innerText = s.category;
      catBadge.className = `px-2 py-0.5 rounded-full text-xs font-medium border ${{s.is_trans ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20' : 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20'}}`;

      document.getElementById('badge-ref').innerText = s.ref;
      document.getElementById('badge-dur').innerText = `${{s.duration}}s`;
      document.getElementById('badge-snr').innerText = `${{s.snr}} dB SNR`;
      document.getElementById('badge-tier').innerText = s.tier;
      document.getElementById('kashmiri-text').innerText = s.text;
      document.getElementById('badge-source').innerText = s.source;
      document.getElementById('badge-speaker').innerText = `(` + s.speaker + `)`;

      const audio = document.getElementById('audio-player');
      const source = document.getElementById('audio-source');
      source.src = 'data:audio/mp3;base64,' + s.audio_b64;
      audio.load();
      resetPlayIcon();
    }}

    function prevSample() {{
      const newIdx = (currentIndex - 1 + samples.length) % samples.length;
      loadSample(newIdx);
    }}

    function nextSample() {{
      const newIdx = (currentIndex + 1) % samples.length;
      loadSample(newIdx);
    }}

    function toggleAudio() {{
      const audio = document.getElementById('audio-player');
      if (audio.paused) {{
        audio.play();
        setPlayingIcon();
      }} else {{
        audio.pause();
        resetPlayIcon();
      }}
    }}

    function setPlayingIcon() {{
      document.getElementById('icon-play').classList.add('hidden');
      document.getElementById('icon-pause').classList.remove('hidden');
    }}

    function resetPlayIcon() {{
      document.getElementById('icon-play').classList.remove('hidden');
      document.getElementById('icon-pause').classList.add('hidden');
    }}

    const audioEl = document.getElementById('audio-player');
    audioEl.addEventListener('play', setPlayingIcon);
    audioEl.addEventListener('pause', resetPlayIcon);
    audioEl.addEventListener('ended', () => {{
      resetPlayIcon();
      if (currentIndex < samples.length - 1) {{
        nextSample();
        document.getElementById('audio-player').play();
      }}
    }});

    // Initialize first sample
    loadSample(0);
  </script>
</body>
</html>
"""

with open(target_widget, "w", encoding="utf-8") as f:
    f.write(html_content)

meta_content = {
    "UserFacing": True,
    "RequestFeedback": False,
    "Summary": "Compact inline Kashmiri speech-text interactive audio player widget with 10 random samples."
}

with open(target_meta, "w", encoding="utf-8") as f:
    json.dump(meta_content, f, indent=2)

print(f"Generated compact widget successfully at: {target_widget}")
print(f"File size: {os.path.getsize(target_widget) / 1024:.2f} KB")
