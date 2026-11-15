#!/usr/bin/env python3
"""
Generates an interactive HTML review widget for the Kashmiri Quran review package.
"""

import os
import json
import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "raw/external/quran_kashmiri/audio_review_package/sample_100_audit_manifest.jsonl"
OUT_HTML = Path("/home/musa/.gemini/antigravity/brain/1361f90f-1463-44e4-bf5a-a8707b9c3e7c/review_widget.html")

def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()][:5]

    clips_data = []
    for r in records:
        clip_path = BASE_DIR.parent / r["relative_clip_path"]
        if not clip_path.exists():
            clip_path = BASE_DIR / r["relative_clip_path"]
        
        b64_audio = ""
        if clip_path.exists():
            with open(clip_path, "rb") as af:
                b64_audio = base64.b64encode(af.read()).decode("utf-8")

        clips_data.append({
            "audit_id": r["audit_id"],
            "sample_id": r["sample_id"],
            "surah_number": r["surah_number"],
            "ayah_number": r["ayah_number"],
            "verse_key": r.get("verse_key", f"{r['surah_number']}:{r['ayah_number']}"),
            "duration_sec": r["duration_sec"],
            "snr_db": r.get("snr_db", 0.0),
            "published_text": r.get("published_text", ""),
            "asr_acoustic_hypothesis": r.get("asr_acoustic_hypothesis", ""),
            "written_verse_reference": r.get("written_verse_reference", ""),
            "b64_audio": b64_audio
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Kashmiri Quran Human Review Interface</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .rtl-text {{
      direction: rtl;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Naskh Arabic", "Scheherazade New", Tahoma, serif;
      line-height: 2.1;
      text-align: right;
    }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] p-4 font-sans">
  <div class="max-w-4xl mx-auto space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[var(--border)] pb-4">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-[var(--foreground)]">Kashmiri Qur'an Audio Verification Portal</h1>
        <p class="text-sm text-[var(--muted-foreground)]">Stratified Research Sample • Native Kashmiri Linguistic Audit</p>
      </div>
      <div class="bg-[var(--card)] border border-[var(--border)] px-3 py-1.5 rounded-lg text-sm font-medium">
        Reviewer: <span class="text-emerald-500 font-bold" id="reviewerName">Human Reviewer (Native Speaker)</span>
      </div>
    </div>

    <!-- Instructions -->
    <div class="bg-[var(--card)] border border-[var(--border)] p-4 rounded-xl text-sm space-y-2">
      <div class="font-semibold text-emerald-500 flex items-center gap-2">
        <span>🎧 Guidelines for Review:</span>
      </div>
      <p class="text-[var(--muted-foreground)]">1. Listen to the audio clip using the built-in player.</p>
      <p class="text-[var(--muted-foreground)]">2. Compare what is spoken against the <b>Published Written Text</b> and the <b>ASR Hypothesis</b>.</p>
      <p class="text-[var(--muted-foreground)]">3. Select your decision: <b>Approve</b>, <b>Correct Text</b>, flag <b>Arabic Boundary Leakage</b>, or <b>Reject</b>.</p>
    </div>

    <!-- Review Cards Container -->
    <div id="cardsContainer" class="space-y-6"></div>

    <!-- Export Summary -->
    <div class="bg-[var(--card)] border border-[var(--border)] p-5 rounded-xl space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="font-bold text-lg">Completed Review Payload</h3>
        <button onclick="copyPayload()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-sm rounded-lg transition-colors shadow">
          📋 Copy Annotations JSON
        </button>
      </div>
      <p class="text-xs text-[var(--muted-foreground)]">Copy this JSON and paste it into the conversation or save to reviewer_annotations.json to permanently record your human verification.</p>
      <textarea id="outputPayload" readonly class="w-full h-36 bg-[var(--background)] border border-[var(--border)] rounded-lg p-3 font-mono text-xs text-[var(--muted-foreground)]"></textarea>
    </div>
  </div>

  <script>
    const clips = {json.dumps(clips_data, ensure_ascii=False)};
    const annotations = {{}};

    function renderCards() {{
      const container = document.getElementById('cardsContainer');
      container.innerHTML = '';

      clips.forEach((clip, index) => {{
        const card = document.createElement('div');
        card.className = "bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4";
        card.id = `card_${{clip.audit_id}}`;

        card.innerHTML = `
          <div class="flex items-center justify-between border-b border-[var(--border)] pb-3">
            <div class="flex items-center gap-3">
              <span class="px-2.5 py-1 bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-xs font-bold rounded-md">${{clip.audit_id}}</span>
              <span class="font-bold text-sm">Surah ${{clip.surah_number}} • Verse ${{clip.verse_key}}</span>
              <span class="text-xs text-[var(--muted-foreground)]">(${{clip.duration_sec}}s, SNR: ${{clip.snr_db}} dB)</span>
            </div>
            <div id="status_${{clip.audit_id}}" class="text-xs font-semibold text-[var(--muted-foreground)]">
              Pending Verification
            </div>
          </div>

          <!-- Audio Player -->
          <div class="bg-[var(--background)] p-3 rounded-lg border border-[var(--border)] flex items-center gap-4">
            <audio controls class="w-full h-10" id="audio_${{clip.audit_id}}">
              <source src="data:audio/wav;base64,${{clip.b64_audio}}" type="audio/wav">
              Your browser does not support audio playback.
            </audio>
            <div class="flex items-center gap-1">
              <button onclick="setSpeed('${{clip.audit_id}}', 0.8)" class="px-2 py-1 bg-[var(--card)] hover:bg-[var(--border)] border border-[var(--border)] rounded text-xs">0.8x</button>
              <button onclick="setSpeed('${{clip.audit_id}}', 1.0)" class="px-2 py-1 bg-[var(--card)] hover:bg-[var(--border)] border border-[var(--border)] rounded text-xs font-bold">1.0x</button>
            </div>
          </div>

          <!-- Transcripts Grid -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="bg-[var(--background)] p-3.5 rounded-lg border border-[var(--border)] space-y-1.5">
              <div class="text-xs font-semibold text-blue-400">📖 Published Mirwaiz Translation (Written):</div>
              <div class="rtl-text text-base text-[var(--foreground)]">${{clip.published_text || '— (Spoken Tafsir Commentary)'}}</div>
            </div>
            <div class="bg-[var(--background)] p-3.5 rounded-lg border border-[var(--border)] space-y-1.5">
              <div class="text-xs font-semibold text-emerald-400">🎙️ ASR Acoustic Hypothesis (Spoken):</div>
              <div class="rtl-text text-base text-[var(--foreground)]">${{clip.asr_acoustic_hypothesis}}</div>
            </div>
          </div>

          <!-- Editable Correction Box -->
          <div class="space-y-1.5">
            <label class="text-xs font-semibold text-[var(--muted-foreground)]">Human Corrected Kashmiri Text (Edit if needed):</label>
            <input type="text" id="input_${{clip.audit_id}}" value="${{clip.asr_acoustic_hypothesis}}" class="w-full p-2.5 bg-[var(--background)] border border-[var(--border)] rounded-lg rtl-text text-base text-[var(--foreground)] focus:outline-none focus:ring-1 focus:ring-emerald-500">
          </div>

          <!-- Action Buttons -->
          <div class="flex flex-wrap items-center justify-between gap-2 pt-2">
            <div class="flex flex-wrap gap-2">
              <button onclick="submitDecision('${{clip.audit_id}}', 'approved', 'kashmiri_translation')" class="px-3.5 py-1.5 bg-emerald-600/20 hover:bg-emerald-600 text-emerald-400 hover:text-white border border-emerald-500/30 rounded-lg text-xs font-bold transition">
                ✓ Approve (Verse Match)
              </button>
              <button onclick="submitDecision('${{clip.audit_id}}', 'approved', 'kashmiri_commentary')" class="px-3.5 py-1.5 bg-blue-600/20 hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-500/30 rounded-lg text-xs font-bold transition">
                ✓ Approve (Commentary)
              </button>
              <button onclick="submitDecision('${{clip.audit_id}}', 'corrected', 'kashmiri_translation')" class="px-3.5 py-1.5 bg-amber-600/20 hover:bg-amber-600 text-amber-400 hover:text-white border border-amber-500/30 rounded-lg text-xs font-bold transition">
                ✎ Submit Correction
              </button>
              <button onclick="submitDecision('${{clip.audit_id}}', 'boundary_leakage', 'uncertain_mixed_speech')" class="px-3.5 py-1.5 bg-purple-600/20 hover:bg-purple-600 text-purple-400 hover:text-white border border-purple-500/30 rounded-lg text-xs font-bold transition">
                ⚡ Flag Arabic Leak
              </button>
            </div>
            <button onclick="submitDecision('${{clip.audit_id}}', 'rejected', 'rejected')" class="px-3 py-1.5 bg-red-600/20 hover:bg-red-600 text-red-400 hover:text-white border border-red-500/30 rounded-lg text-xs font-semibold transition">
              ✗ Reject
            </button>
          </div>
        `;

        container.appendChild(card);
      }});
      updateOutputPayload();
    }}

    function setSpeed(auditId, speed) {{
      const a = document.getElementById(`audio_${{auditId}}`);
      if (a) a.playbackRate = speed;
    }}

    function submitDecision(auditId, decision, category) {{
      const textVal = document.getElementById(`input_${{auditId}}`).value.trim();
      const statusBadge = document.getElementById(`status_${{auditId}}`);

      annotations[auditId] = {{
        "audit_id": auditId,
        "decision": decision,
        "reviewer_id": "human_reviewer_musa",
        "timestamp": new Date().toISOString(),
        "verified_text": textVal,
        "speech_category": category,
        "arabic_boundary_trim_sec": decision === 'boundary_leakage' ? 1.0 : 0.0,
        "notes": `Verified via GUI (${{decision}})`
      }};

      statusBadge.innerHTML = `<span class="text-emerald-400 font-bold">✓ Verified: ${{decision}}</span>`;
      const card = document.getElementById(`card_${{auditId}}`);
      card.classList.add('border-emerald-500/40');

      updateOutputPayload();
    }}

    function updateOutputPayload() {{
      const out = document.getElementById('outputPayload');
      out.value = JSON.stringify(annotations, null, 2);
    }}

    function copyPayload() {{
      const out = document.getElementById('outputPayload');
      out.select();
      document.execCommand('copy');
      alert('Annotations copied to clipboard! You can paste them into chat to apply them directly.');
    }}

    window.onload = renderCards;
  </script>
</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated {OUT_HTML} ({len(records)} clips embedded).")

if __name__ == "__main__":
    main()
