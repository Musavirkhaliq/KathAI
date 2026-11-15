import json
import base64
import os
import sys
from pathlib import Path
from generate_human_eval_ui import sample_and_build

# Sample initial 10 clips
initial_clips = sample_and_build(10)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kashmiri Speech Corpus Human Quality Evaluator</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .rtl-text {{
      direction: rtl;
      font-family: -apple-system, BlinkMacSystemFont, "Noto Naskh Arabic", "Scheherazade New", "Amiri", "Segoe UI", Tahoma, serif;
      line-height: 2.2;
      text-align: right;
    }}
    .star-rating span {{
      cursor: pointer;
      transition: transform 0.15s ease, color 0.15s ease;
    }}
    .star-rating span:hover {{
      transform: scale(1.2);
    }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased min-h-screen p-4 md:p-8 font-sans">
  <div class="max-w-5xl mx-auto space-y-6">

    <!-- Top Sticky Header / Control Bar -->
    <header class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm sticky top-2 z-30 backdrop-blur-md bg-opacity-95">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2 mb-1">
            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Corpus: 28,960 Verified Clips (38.3 hrs)
            </span>
            <span id="backendBadge" class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/30">
              ● Server Connected (Port 8765)
            </span>
          </div>
          <h1 class="text-2xl font-bold tracking-tight text-[var(--foreground)]">
            Human Quality Judgment Dashboard
          </h1>
          <p class="text-xs md:text-sm text-[var(--muted-foreground)]">
            Randomly sample 10 clips from Mirwaiz Mohammad Yousuf Shah's Kashmiri Qur'an archive. Listen, inspect verbatim text, and record human evaluation.
          </p>
        </div>

        <!-- Controls -->
        <div class="flex flex-wrap items-center gap-2">
          <button id="btnSampleNew" onclick="sampleNewBatch()" class="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm transition active:scale-95">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
            Sample 10 New Random Clips
          </button>
          <button onclick="exportJudgmentsJSON()" class="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--accent)] hover:text-[var(--accent-foreground)] transition">
            💾 Export JSON
          </button>
          <button onclick="exportJudgmentsCSV()" class="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--accent)] hover:text-[var(--accent-foreground)] transition">
            📊 Export CSV
          </button>
        </div>
      </div>

      <!-- Live Assessment Stats Bar -->
      <div class="mt-5 pt-4 border-t border-[var(--border)] grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
        <div class="bg-[var(--background)] p-3 rounded-xl border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)] font-medium">Batch Progress</div>
          <div id="statReviewed" class="text-lg font-bold text-[var(--foreground)]">0 / 10</div>
        </div>
        <div class="bg-[var(--background)] p-3 rounded-xl border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)] font-medium">Pass Rate</div>
          <div id="statPassRate" class="text-lg font-bold text-emerald-400">-%</div>
        </div>
        <div class="bg-[var(--background)] p-3 rounded-xl border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)] font-medium">Issues Flagged</div>
          <div id="statIssues" class="text-lg font-bold text-amber-400">0</div>
        </div>
        <div class="bg-[var(--background)] p-3 rounded-xl border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)] font-medium">Average Rating</div>
          <div id="statAvgRating" class="text-lg font-bold text-blue-400">- / 5★</div>
        </div>
      </div>
    </header>

    <!-- Cards Container -->
    <div id="clipsContainer" class="space-y-6">
      <!-- Cards dynamically injected here -->
    </div>

    <!-- Bottom Action Footer -->
    <div class="p-6 bg-[var(--card)] border border-[var(--border)] rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-4">
      <div class="text-sm text-[var(--muted-foreground)]">
        All human judgments are preserved locally and saved to <code class="text-xs bg-[var(--background)] px-1.5 py-0.5 rounded border border-[var(--border)]">human_judgments.jsonl</code>.
      </div>
      <div class="flex items-center gap-3">
        <button onclick="saveJudgmentsToBackend()" class="px-5 py-2.5 rounded-xl text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition active:scale-95 shadow">
          Save All 10 Judgments
        </button>
      </div>
    </div>

  </div>

  <script>
    // Embedded Initial Data (Fallback for pure offline/static artifact viewing)
    let currentClips = {json.dumps(initial_clips, ensure_ascii=False)};
    let judgments = {{}}; // sample_id -> {{ verdict, rating, issues: [], notes: '' }}

    // Check backend health
    async function checkBackend() {{
      try {{
        const res = await fetch('http://localhost:8765/api/stats');
        if (res.ok) {{
          document.getElementById('backendBadge').className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
          document.getElementById('backendBadge').textContent = "● Live Backend Connected";
          return true;
        }}
      }} catch (e) {{
        document.getElementById('backendBadge').className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/30";
        document.getElementById('backendBadge').textContent = "● Static Mode (Offline)";
        return false;
      }}
      return false;
    }}

    // Render Cards
    function renderClips() {{
      const container = document.getElementById('clipsContainer');
      container.innerHTML = '';

      currentClips.forEach((clip, index) => {{
        const j = judgments[clip.sample_id] || {{
          verdict: null,
          rating: 0,
          issues: [],
          notes: ''
        }};

        const card = document.createElement('div');
        card.id = `clip-card-${{clip.sample_id}}`;
        card.className = "bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-5 transition duration-200";

        // Status badge
        let statusBadge = '<span class="px-2.5 py-0.5 rounded-full text-xs font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/30">⏳ Pending Review</span>';
        if (j.verdict === 'pass') {{
          statusBadge = '<span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">✓ Passed / Clean</span>';
        }} else if (j.verdict === 'minor') {{
          statusBadge = '<span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">⚠️ Minor Issue</span>';
        }} else if (j.verdict === 'fail') {{
          statusBadge = '<span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30">❌ Rejected / Flaw</span>';
        }}

        card.innerHTML = `
          <!-- Header Info -->
          <div class="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
            <div class="flex items-center gap-2">
              <span class="w-7 h-7 rounded-lg bg-[var(--background)] border border-[var(--border)] flex items-center justify-center font-bold text-xs text-[var(--foreground)]">
                #${{clip.idx || index + 1}}
              </span>
              <span class="font-semibold text-sm text-[var(--foreground)]">
                Surah ${{clip.surah_number}}: ${{clip.surah_en}} (${{clip.surah_ar}}) • Ayah ${{clip.ayah_number}}
              </span>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-xs px-2 py-0.5 rounded bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)]">
                ⏱ ${{clip.duration}}s
              </span>
              <span class="text-xs font-mono text-[var(--muted-foreground)] hidden sm:inline">
                ${{clip.sample_id}}
              </span>
              ${{statusBadge}}
            </div>
          </div>

          <!-- Audio Player & Controls -->
          <div class="bg-[var(--background)] p-4 rounded-xl border border-[var(--border)] space-y-3">
            <div class="flex flex-col sm:flex-row items-center gap-4">
              <audio id="audio-${{clip.sample_id}}" controls class="w-full h-10 rounded-lg">
                <source src="data:audio/wav;base64,${{clip.b64}}" type="audio/wav">
                Your browser does not support HTML5 audio.
              </audio>
              <div class="flex items-center gap-1.5 shrink-0">
                <button onclick="setSpeed('${{clip.sample_id}}', 0.75)" class="px-2 py-1 text-xs rounded bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--accent)] font-medium">0.75x</button>
                <button onclick="setSpeed('${{clip.sample_id}}', 1.0)" class="px-2 py-1 text-xs rounded bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--accent)] font-medium">1.0x</button>
                <button onclick="setSpeed('${{clip.sample_id}}', 1.25)" class="px-2 py-1 text-xs rounded bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--accent)] font-medium">1.25x</button>
                <button onclick="seekRel('${{clip.sample_id}}', -2)" class="px-2 py-1 text-xs rounded bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--accent)] font-medium" title="Rewind 2 seconds">⏮ -2s</button>
                <button onclick="seekRel('${{clip.sample_id}}', 2)" class="px-2 py-1 text-xs rounded bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--accent)] font-medium" title="Forward 2 seconds">⏭ +2s</button>
              </div>
            </div>
          </div>

          <!-- Kashmiri Book Text -->
          <div class="space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-xs uppercase tracking-wider font-semibold text-[var(--muted-foreground)]">Spoken Kashmiri Text (Ground Truth)</span>
              <button onclick="toggleVerseContext('${{clip.sample_id}}')" class="text-xs text-emerald-400 hover:underline">
                Toggle Full Ayah Context
              </button>
            </div>
            <div class="rtl-text text-xl md:text-2xl text-[var(--foreground)] bg-[var(--background)] p-4 rounded-xl border border-[var(--border)] select-text">
              ${{clip.text}}
            </div>
            <div id="context-${{clip.sample_id}}" class="hidden text-sm text-[var(--muted-foreground)] rtl-text bg-[var(--background)]/50 p-3 rounded-lg border border-dashed border-[var(--border)] mt-2">
              <span class="text-xs font-sans font-bold text-[var(--foreground)] block mb-1">Full Verse Context (Ayah ${{clip.ayah_number}}):</span>
              ${{clip.canonical_verse_text}}
            </div>
          </div>

          <!-- Human Assessment Panel -->
          <div class="pt-3 border-t border-[var(--border)] space-y-3">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <!-- Verdict Buttons -->
              <div class="flex items-center gap-2">
                <span class="text-xs font-semibold text-[var(--muted-foreground)] mr-1">Verdict:</span>
                <button onclick="setVerdict('${{clip.sample_id}}', 'pass')" class="px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${{j.verdict === 'pass' ? 'bg-emerald-600 text-white border-emerald-500' : 'bg-[var(--background)] border-[var(--border)] text-[var(--foreground)] hover:border-emerald-500'}}">
                  ✓ Pass / Clean
                </button>
                <button onclick="setVerdict('${{clip.sample_id}}', 'minor')" class="px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${{j.verdict === 'minor' ? 'bg-amber-600 text-white border-amber-500' : 'bg-[var(--background)] border-[var(--border)] text-[var(--foreground)] hover:border-amber-500'}}">
                  ⚠️ Minor Flaw
                </button>
                <button onclick="setVerdict('${{clip.sample_id}}', 'fail')" class="px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${{j.verdict === 'fail' ? 'bg-rose-600 text-white border-rose-500' : 'bg-[var(--background)] border-[var(--border)] text-[var(--foreground)] hover:border-rose-500'}}">
                  ❌ Reject / Defect
                </button>
              </div>

              <!-- Star Rating -->
              <div class="flex items-center gap-1.5">
                <span class="text-xs font-semibold text-[var(--muted-foreground)] mr-1">Rating:</span>
                <div class="star-rating text-lg flex gap-1">
                  ${{[1,2,3,4,5].map(star => `
                    <span onclick="setRating('${{clip.sample_id}}', ${{star}})" class="${{star <= j.rating ? 'text-amber-400' : 'text-zinc-600'}}">★</span>
                  `).join('')}}
                </div>
              </div>
            </div>

            <!-- Issue Flags (Visible if verdict is minor/fail or toggled) -->
            <div class="bg-[var(--background)] p-3 rounded-xl border border-[var(--border)] space-y-2">
              <span class="text-xs font-semibold text-[var(--muted-foreground)] block">Specific Issue Checklist:</span>
              <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                ${{[
                  ['arabic_speech', 'Arabic Recitation Detected'],
                  ['audio_cut_end', 'Audio Cut Off Early at End'],
                  ['audio_truncated_start', 'Start Clipped / Truncated'],
                  ['extra_words_text', 'Extra Words in Written Text'],
                  ['missing_words_text', 'Missing Spoken Words in Text'],
                  ['noise_distortion', 'Severe Noise / Distortion']
                ].map(([key, label]) => `
                  <label class="flex items-center gap-2 cursor-pointer select-none text-[var(--foreground)]">
                    <input type="checkbox" onchange="toggleIssue('${{clip.sample_id}}', '${{key}}')" ${{j.issues.includes(key) ? 'checked' : ''}} class="rounded border-[var(--border)] text-emerald-600 focus:ring-emerald-500">
                    <span>${{label}}</span>
                  </label>
                `).join('')}}
              </div>
            </div>

            <!-- Human Notes -->
            <div class="flex items-center gap-2">
              <input type="text" id="notes-${{clip.sample_id}}" value="${{j.notes || ''}}" onchange="setNotes('${{clip.sample_id}}', this.value)" placeholder="Add personal reviewer note or observations (optional)..." class="w-full text-xs bg-[var(--background)] border border-[var(--border)] rounded-lg px-3 py-2 text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:outline-none focus:border-emerald-500">
            </div>

          </div>
        `;
        container.appendChild(card);
      }});

      updateStats();
    }}

    // Playback Helpers
    function setSpeed(id, rate) {{
      const audio = document.getElementById(`audio-${{id}}`);
      if (audio) audio.playbackRate = rate;
    }}

    function seekRel(id, sec) {{
      const audio = document.getElementById(`audio-${{id}}`);
      if (audio) audio.currentTime = Math.max(0, audio.currentTime + sec);
    }}

    function toggleVerseContext(id) {{
      const el = document.getElementById(`context-${{id}}`);
      if (el) el.classList.toggle('hidden');
    }}

    // Rating and Verdict Logic
    function ensureJudgment(id) {{
      if (!judgments[id]) {{
        judgments[id] = {{ verdict: null, rating: 0, issues: [], notes: '' }};
      }}
      return judgments[id];
    }}

    function setVerdict(id, verdict) {{
      const j = ensureJudgment(id);
      j.verdict = verdict;
      if (verdict === 'pass' && j.rating === 0) j.rating = 5;
      if (verdict === 'fail' && (j.rating === 0 || j.rating > 2)) j.rating = 1;
      if (verdict === 'minor' && (j.rating === 0 || j.rating > 3)) j.rating = 3;
      saveLocal();
      renderClips();
    }}

    function setRating(id, rating) {{
      const j = ensureJudgment(id);
      j.rating = rating;
      if (!j.verdict) {{
        if (rating >= 4) j.verdict = 'pass';
        else if (rating === 3) j.verdict = 'minor';
        else j.verdict = 'fail';
      }}
      saveLocal();
      renderClips();
    }}

    function toggleIssue(id, issue) {{
      const j = ensureJudgment(id);
      const idx = j.issues.indexOf(issue);
      if (idx > -1) j.issues.splice(idx, 1);
      else j.issues.push(issue);

      if (j.issues.length > 0 && j.verdict === 'pass') {{
        j.verdict = 'minor';
      }}
      saveLocal();
      updateStats();
    }}

    function setNotes(id, val) {{
      const j = ensureJudgment(id);
      j.notes = val;
      saveLocal();
    }}

    // Local Storage
    function saveLocal() {{
      try {{
        localStorage.setItem('kashmiri_human_eval_judgments', JSON.stringify(judgments));
      }} catch(e) {{}}
    }}

    function loadLocal() {{
      try {{
        const saved = localStorage.getItem('kashmiri_human_eval_judgments');
        if (saved) judgments = JSON.parse(saved);
      }} catch(e) {{}}
    }}

    // Update Stats Bar
    function updateStats() {{
      const total = currentClips.length;
      let reviewed = 0;
      let passes = 0;
      let issues = 0;
      let totalRating = 0;
      let ratedCount = 0;

      currentClips.forEach(c => {{
        const j = judgments[c.sample_id];
        if (j && j.verdict) {{
          reviewed++;
          if (j.verdict === 'pass') passes++;
          if (j.verdict !== 'pass' || (j.issues && j.issues.length > 0)) issues++;
          if (j.rating > 0) {{
            totalRating += j.rating;
            ratedCount++;
          }}
        }}
      }});

      document.getElementById('statReviewed').textContent = `${{reviewed}} / ${{total}}`;
      const passRate = reviewed > 0 ? Math.round((passes / reviewed) * 100) : 0;
      document.getElementById('statPassRate').textContent = reviewed > 0 ? `${{passRate}}%` : '-%';
      document.getElementById('statIssues').textContent = `${{issues}}`;
      const avg = ratedCount > 0 ? (totalRating / ratedCount).toFixed(1) : '-';
      document.getElementById('statAvgRating').textContent = `${{avg}} / 5★`;
    }}

    // Sample New Batch
    async function sampleNewBatch() {{
      const btn = document.getElementById('btnSampleNew');
      btn.disabled = true;
      btn.innerHTML = '<span class="animate-spin inline-block mr-1">↻</span> Sampling 10 clips...';

      try {{
        const res = await fetch('http://localhost:8765/api/sample?n=10');
        if (res.ok) {{
          const data = await res.json();
          if (data.clips && data.clips.length > 0) {{
            currentClips = data.clips;
            renderClips();
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
          }}
        }} else {{
          alert('Could not fetch new clips from server. Server responded with error.');
        }}
      }} catch (e) {{
        alert('Server not reachable at http://localhost:8765. In static mode, reload page or ensure server is running.');
      }} finally {{
        btn.disabled = false;
        btn.innerHTML = `
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
          Sample 10 New Random Clips
        `;
      }}
    }}

    // Backend Save
    async function saveJudgmentsToBackend() {{
      const payload = [];
      currentClips.forEach(c => {{
        const j = judgments[c.sample_id];
        if (j) {{
          payload.push({{
            sample_id: c.sample_id,
            surah_number: c.surah_number,
            ayah_number: c.ayah_number,
            duration: c.duration,
            text: c.text,
            verdict: j.verdict,
            rating: j.rating,
            issues: j.issues,
            notes: j.notes,
            timestamp: new Date().toISOString()
          }});
        }}
      }});

      if (payload.length === 0) {{
        alert('Please review and grade at least one clip before saving.');
        return;
      }}

      try {{
        const res = await fetch('http://localhost:8765/api/save_judgments', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ judgments: payload }})
        }});
        if (res.ok) {{
          alert(`Successfully saved ${{payload.length}} human evaluation judgments to disk!`);
        }} else {{
          exportJudgmentsJSON();
        }}
      }} catch (e) {{
        alert('Backend server not connected. Exporting as JSON download instead.');
        exportJudgmentsJSON();
      }}
    }}

    // Export Helpers
    function exportJudgmentsJSON() {{
      const exportData = currentClips.map(c => ({{
        sample_id: c.sample_id,
        surah: `${{c.surah_number}} (${{c.surah_en}})`,
        ayah: c.ayah_number,
        duration: c.duration,
        text: c.text,
        evaluation: judgments[c.sample_id] || {{ verdict: 'unreviewed' }}
      }}));

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `kashmiri_human_judgments_${{Date.now()}}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }}

    function exportJudgmentsCSV() {{
      let csv = "\\uFEFFsample_id,surah_number,surah_name,ayah_number,duration,verdict,rating,issues,notes,text\\n";
      currentClips.forEach(c => {{
        const j = judgments[c.sample_id] || {{ verdict: '', rating: '', issues: [], notes: '' }};
        const escapedText = '"' + (c.text || '').replace(/"/g, '""') + '"';
        const escapedNotes = '"' + (j.notes || '').replace(/"/g, '""') + '"';
        const issuesStr = '"' + (j.issues || []).join(';') + '"';
        csv += `${{c.sample_id}},${{c.surah_number}},"${{c.surah_en}}",${{c.ayah_number}},${{c.duration}},${{j.verdict || ''}},${{j.rating || ''}},${{issuesStr}},${{escapedNotes}},${{escapedText}}\\n`;
      }});

      const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `kashmiri_human_judgments_${{Date.now()}}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }}

    // Initialize
    loadLocal();
    renderClips();
    checkBackend();
  </script>
</body>
</html>
"""

artifact_path = Path("/home/musa/.gemini/antigravity/brain/1361f90f-1463-44e4-bf5a-a8707b9c3e7c/human_judge_ui.html")
with open(artifact_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Generated standalone human eval UI artifact: {artifact_path} ({os.path.getsize(artifact_path)} bytes)")
