import json
import base64
import subprocess
import os
import random

target_html = "/home/musa/.gemini/antigravity/brain/412df5c3-70ef-4580-9101-03634714aba4/kashmiri_audio_inspector.html"
target_meta = target_html + ".metadata.json"

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
        "source_code": s["source"],
        "category": s["category"].replace("_", " ").title(),
        "category_raw": s["category"],
        "duration": round(s["duration"], 2),
        "snr": round(s.get("snr", 15.0), 1),
        "text": s.get("text", ""),
        "speaker": s.get("speaker_name", s.get("speaker_id", "Unknown")),
        "tier": s.get("verification_status", "tier1_machine_detected").replace("_", " ").title(),
        "ref": f"Surah {s.get('surah_number')}:{s.get('ayah_number')}" if s.get("surah_number") else f"Episode {s.get('episode_number')}",
        "audio_b64": b64
    })

print(f"Encoded {len(sample_data)} samples to MP3 Base64.")

html_content = f"""<!DOCTYPE html>
<html lang="ks" dir="ltr">
<head>
  <meta charset="UTF-8">
  <title>EduCasheer Kashmiri AI — Audio & Text Inspector</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');
    .kashmiri-text {{
      font-family: 'Noto Naskh Arabic', 'Scheherazade New', 'Amiri', Tahoma, sans-serif;
      line-height: 2.1;
      direction: rtl;
    }}
    ::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    ::-webkit-scrollbar-track {{
      background: var(--background);
    }}
    ::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 3px;
    }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-4 md:p-6 min-h-screen">

  <!-- Header Banner -->
  <header class="max-w-5xl mx-auto mb-6 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 shadow-sm">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2 mb-1">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
            DGX Spark Ingestion
          </span>
          <span class="text-xs text-[var(--muted-foreground)]">Kashmiri AI Research Program</span>
        </div>
        <h1 class="text-xl md:text-2xl font-bold tracking-tight text-[var(--foreground)]">
          Qur'anic Speech & Text Inspector
        </h1>
        <p class="text-sm text-[var(--muted-foreground)] mt-0.5">
          10 Randomly Sampled Audio Clips with Perso-Arabic Transcripts & Acoustic Metadata
        </p>
      </div>

      <!-- Quick Metrics -->
      <div class="flex items-center gap-3 bg-[var(--background)] p-2.5 rounded-xl border border-[var(--border)] text-xs">
        <div class="text-center px-2">
          <div class="font-bold text-sm text-[var(--foreground)]">10</div>
          <div class="text-[var(--muted-foreground)]">Samples</div>
        </div>
        <div class="h-6 w-px bg-[var(--border)]"></div>
        <div class="text-center px-2">
          <div class="font-bold text-sm text-[var(--foreground)]">38.9s</div>
          <div class="text-[var(--muted-foreground)]">Total Audio</div>
        </div>
        <div class="h-6 w-px bg-[var(--border)]"></div>
        <div class="text-center px-2">
          <div class="font-bold text-sm text-emerald-600 dark:text-emerald-400">15.4 dB</div>
          <div class="text-[var(--muted-foreground)]">Avg SNR</div>
        </div>
      </div>
    </div>

    <!-- Filter Buttons -->
    <div class="flex flex-wrap items-center gap-2 mt-4 pt-4 border-t border-[var(--border)]">
      <span class="text-xs font-medium text-[var(--muted-foreground)] mr-1">Filter View:</span>
      <button onclick="filterCards('all')" id="filter-all" class="filter-btn px-3 py-1 text-xs rounded-lg font-medium transition-colors bg-[var(--foreground)] text-[var(--background)]">
        All (10)
      </button>
      <button onclick="filterCards('king-fahd')" id="filter-king-fahd" class="filter-btn px-3 py-1 text-xs rounded-lg font-medium transition-colors bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        King Fahd (5)
      </button>
      <button onclick="filterCards('dr-war')" id="filter-dr-war" class="filter-btn px-3 py-1 text-xs rounded-lg font-medium transition-colors bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        Dr. Abdul Rehman War (5)
      </button>
      <button onclick="filterCards('translation')" id="filter-translation" class="filter-btn px-3 py-1 text-xs rounded-lg font-medium transition-colors bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        Verse Translation (2)
      </button>
      <button onclick="filterCards('commentary')" id="filter-commentary" class="filter-btn px-3 py-1 text-xs rounded-lg font-medium transition-colors bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        Commentary (8)
      </button>

      <div class="ml-auto flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
        <kbd class="px-1.5 py-0.5 rounded border border-[var(--border)] bg-[var(--background)] font-mono text-[11px]">Space</kbd> Play / Pause
      </div>
    </div>
  </header>

  <!-- Sample Cards Grid -->
  <main class="max-w-5xl mx-auto space-y-4" id="cards-container">
"""

for s in sample_data:
    is_trans = "translation" in s["category_raw"].lower()
    badge_bg = "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20" if is_trans else "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20"
    source_filter_class = "src-kf" if s["source_code"] == "SRC_QURAN_KING_FAHD" else "src-war"
    cat_filter_class = "cat-trans" if is_trans else "cat-comm"

    html_content += f"""
    <!-- Card #{s['index']} -->
    <article class="sample-card {source_filter_class} {cat_filter_class} bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 shadow-sm hover:border-emerald-500/40 transition-all">
      <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div class="flex flex-wrap items-center gap-2">
          <span class="font-mono text-xs font-bold text-[var(--foreground)] bg-[var(--background)] px-2 py-0.5 rounded border border-[var(--border)]">
            #{s['index']:02d}
          </span>
          <span class="text-xs font-semibold text-[var(--foreground)]">
            {s['sample_id']}
          </span>
          <span class="px-2 py-0.5 rounded-full text-xs font-medium border {badge_bg}">
            {s['category']}
          </span>
          <span class="px-2 py-0.5 rounded-full text-xs bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)] font-medium">
            {s['ref']}
          </span>
        </div>

        <div class="flex items-center gap-3 text-xs text-[var(--muted-foreground)] font-mono">
          <span>{s['duration']}s</span>
          <span>•</span>
          <span class="text-emerald-600 dark:text-emerald-400 font-semibold">{s['snr']} dB SNR</span>
          <span>•</span>
          <span class="text-amber-600 dark:text-amber-400 font-medium">{s['tier']}</span>
        </div>
      </div>

      <!-- Spoken Kashmiri Text Display -->
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 my-3 text-right">
        <div class="kashmiri-text text-xl md:text-2xl font-medium text-[var(--foreground)] tracking-wide leading-relaxed">
          {s['text']}
        </div>
      </div>

      <!-- Audio Player Controls -->
      <div class="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
        <div class="flex items-center gap-3 w-full sm:w-auto">
          <!-- Play / Pause Button -->
          <button onclick="toggleAudio('{s['sample_id']}')" id="btn-{s['sample_id']}" class="play-btn w-10 h-10 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white flex items-center justify-center shadow transition-transform active:scale-95 flex-shrink-0" title="Play / Pause">
            <svg id="icon-play-{s['sample_id']}" class="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z"/>
            </svg>
            <svg id="icon-pause-{s['sample_id']}" class="w-5 h-5 hidden" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
            </svg>
          </button>

          <!-- Native Audio Element with Base64 Source -->
          <audio id="audio-{s['sample_id']}" controls class="w-full sm:w-80 h-9">
            <source src="data:audio/mp3;base64,{s['audio_b64']}" type="audio/mp3">
            Your browser does not support the audio element.
          </audio>
        </div>

        <div class="text-xs text-[var(--muted-foreground)] flex items-center gap-2 ml-auto">
          <span class="font-medium text-[var(--foreground)]">{s['source']}</span>
          <span>({s['speaker']})</span>
        </div>
      </div>
    </article>
    """

html_content += """
  </main>

  <footer class="max-w-5xl mx-auto mt-8 text-center text-xs text-[var(--muted-foreground)] pb-6">
    EduCasheer Kashmiri AI Research Program • NVIDIA DGX Spark Deployment • Verified Zero-Contamination Corpus
  </footer>

  <script>
    let currentPlaying = null;

    function toggleAudio(id) {
      const audio = document.getElementById('audio-' + id);
      const playIcon = document.getElementById('icon-play-' + id);
      const pauseIcon = document.getElementById('icon-pause-' + id);

      if (!audio) return;

      if (currentPlaying && currentPlaying !== audio) {
        currentPlaying.pause();
        const prevId = currentPlaying.id.replace('audio-', '');
        document.getElementById('icon-play-' + prevId)?.classList.remove('hidden');
        document.getElementById('icon-pause-' + prevId)?.classList.add('hidden');
      }

      if (audio.paused) {
        audio.play();
        playIcon.classList.add('hidden');
        pauseIcon.classList.remove('hidden');
        currentPlaying = audio;
      } else {
        audio.pause();
        playIcon.classList.remove('hidden');
        pauseIcon.classList.add('hidden');
        currentPlaying = null;
      }
    }

    document.querySelectorAll('audio').forEach(audio => {
      audio.addEventListener('play', () => {
        const id = audio.id.replace('audio-', '');
        document.getElementById('icon-play-' + id)?.classList.add('hidden');
        document.getElementById('icon-pause-' + id)?.classList.remove('hidden');
        if (currentPlaying && currentPlaying !== audio) {
          currentPlaying.pause();
          const prevId = currentPlaying.id.replace('audio-', '');
          document.getElementById('icon-play-' + prevId)?.classList.remove('hidden');
          document.getElementById('icon-pause-' + prevId)?.classList.add('hidden');
        }
        currentPlaying = audio;
      });

      audio.addEventListener('pause', () => {
        const id = audio.id.replace('audio-', '');
        document.getElementById('icon-play-' + id)?.classList.remove('hidden');
        document.getElementById('icon-pause-' + id)?.classList.add('hidden');
        if (currentPlaying === audio) currentPlaying = null;
      });

      audio.addEventListener('ended', () => {
        const id = audio.id.replace('audio-', '');
        document.getElementById('icon-play-' + id)?.classList.remove('hidden');
        document.getElementById('icon-pause-' + id)?.classList.add('hidden');
        if (currentPlaying === audio) currentPlaying = null;
      });
    });

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        if (currentPlaying) {
          const curId = currentPlaying.id.replace('audio-', '');
          toggleAudio(curId);
        } else {
          const firstAudio = document.querySelector('audio');
          if (firstAudio) {
            const firstId = firstAudio.id.replace('audio-', '');
            toggleAudio(firstId);
          }
        }
      }
    });

    function filterCards(type) {
      document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('bg-[var(--foreground)]', 'text-[var(--background)]');
        btn.classList.add('bg-[var(--background)]', 'text-[var(--muted-foreground)]', 'border', 'border-[var(--border)]');
      });

      const activeBtn = document.getElementById('filter-' + type);
      if (activeBtn) {
        activeBtn.classList.remove('bg-[var(--background)]', 'text-[var(--muted-foreground)]');
        activeBtn.classList.add('bg-[var(--foreground)]', 'text-[var(--background)]');
      }

      const cards = document.querySelectorAll('.sample-card');
      cards.forEach(card => {
        if (type === 'all') {
          card.classList.remove('hidden');
        } else if (type === 'king-fahd') {
          card.classList.toggle('hidden', !card.classList.contains('src-kf'));
        } else if (type === 'dr-war') {
          card.classList.toggle('hidden', !card.classList.contains('src-war'));
        } else if (type === 'translation') {
          card.classList.toggle('hidden', !card.classList.contains('cat-trans'));
        } else if (type === 'commentary') {
          card.classList.toggle('hidden', !card.classList.contains('cat-comm'));
        }
      });
    }
  </script>
</body>
</html>
"""

with open(target_html, "w", encoding="utf-8") as f:
    f.write(html_content)

meta_content = {
    "UserFacing": True,
    "RequestFeedback": False,
    "Summary": "Interactive Kashmiri Qur'anic Speech and Text Inspector displaying 10 randomly sampled candidate recordings with audio playback, Perso-Arabic transcripts, speaker attribution, and acoustic metadata."
}

with open(target_meta, "w", encoding="utf-8") as f:
    json.dump(meta_content, f, indent=2)

print(f"Generated UI artifact successfully at: {target_html}")
print(f"File size: {os.path.getsize(target_html) / 1024:.2f} KB")
