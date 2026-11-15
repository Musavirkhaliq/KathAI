import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

colors = {
    'primary': '#1976D2',
    'secondary': '#388E3C',
    'accent': '#D32F2F',
    'neutral': '#455A64',
    'light': '#ECEFF1',
    'orange': '#F57C00',
    'purple': '#7B1FA2'
}

# --- Figure 1: Dataset Composition & Sample Rate Distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sources = ['Bolbosh\n(58.4h)', 'IndicVoices-R\n(41.9h)', 'IndicVoices\n(29.8h)', 'Mirwaiz Quran\n(20.9h)', 'Other 8 Sources\n(63.7h)']
hours = [58.40, 41.87, 29.80, 20.91, 63.67]
c_palette = ['#1976D2', '#0288D1', '#0097A7', '#00796B', '#388E3C']
ax1.pie(hours, labels=sources, autopct='%1.1f%%', startangle=140, colors=c_palette, textprops=dict(color="black"))
ax1.set_title("A. Speech Corpus Composition (214.65 Hours)", pad=15)

splits = ['Tier A Train', 'Tier B Train', 'Dev Assessment', 'Dev Calibration']
sr_16k = [66.19, 48.15, 56.94, 58.30]
sr_22k = [33.81, 51.85, 43.06, 41.70]
x = np.arange(len(splits))
width = 0.35
rects1 = ax2.bar(x - width/2, sr_16k, width, label='16,000 Hz Native', color='#1976D2')
rects2 = ax2.bar(x + width/2, sr_22k, width, label='22,050 Hz (Bolbosh)', color='#F57C00')
ax2.set_ylabel('Proportion of Utterances (%)')
ax2.set_title('B. Sample Rate Distribution Across Partitions', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(splits, rotation=15)
ax2.legend()
ax2.set_ylim(0, 100)
for r in rects1 + rects2:
    h = r.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig1_corpus_and_samplerate.png"))
fig.savefig(os.path.join(fig_dir, "fig1_corpus_and_samplerate.svg"))
fig.savefig(os.path.join(fig_dir, "fig1_corpus_and_samplerate.pdf"))
plt.close(fig)

# --- Figure 2: Acoustic Scaling Trajectory ---
fig, ax = plt.subplots(figsize=(8, 5))
hours_scaling = [92.94, 104.93, 123.45, 165.32]
greedy_wer = [33.86, 29.42, 29.39, 27.84]
kenlm_wer = [31.80, 26.30, 26.41, 25.24]
labels = ['B-01 (Baseline)', 'B-02A (+Recovered)', 'B-02B (+Quranic)', 'B-02C (+IndicVoices-R)']
ax.plot(hours_scaling, greedy_wer, marker='o', linewidth=2.5, markersize=8, color=colors['primary'], label='Greedy CTC WER')
ax.plot(hours_scaling, kenlm_wer, marker='s', linewidth=2.5, markersize=8, color=colors['secondary'], linestyle='--', label='KenLM 4-gram WER')
for i, txt in enumerate(labels):
    ax.annotate(f"{txt}\n{greedy_wer[i]}% / {kenlm_wer[i]}%", (hours_scaling[i], greedy_wer[i]),
                xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#90CAF9", alpha=0.8))
ax.set_xlabel('Curated Kashmiri Speech Training Hours')
ax.set_ylabel('Validation Word Error Rate (%) [val.jsonl]')
ax.set_title('Acoustic Scaling Trajectory: Validation WER vs Training Hours', pad=15)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')
ax.set_ylim(23, 36)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig2_acoustic_scaling.png"))
fig.savefig(os.path.join(fig_dir, "fig2_acoustic_scaling.svg"))
fig.savefig(os.path.join(fig_dir, "fig2_acoustic_scaling.pdf"))
plt.close(fig)

# --- Figure 3: Resampling Impact ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
categories = ['Substitutions', 'Deletions', 'Insertions', 'Total Errors']
hist_errors = [3709, 669, 236, 4614]
corr_errors = [3365, 516, 240, 4121]
x = np.arange(len(categories))
width = 0.35
ax1.bar(x - width/2, hist_errors, width, label='Historical Protocol (Unresampled)', color=colors['accent'])
ax1.bar(x + width/2, corr_errors, width, label='Corrected Protocol (16k Resampled)', color=colors['secondary'])
ax1.set_ylabel('Word Error Count [dev_assessment.jsonl]')
ax1.set_title('A. Total Word Errors by Category (-493 Errors)', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(axis='y', linestyle=':', alpha=0.6)

cohorts = ['Native 16k Cohort\n(570 utts)', 'Bolbosh 22.05k Cohort\n(431 utts)']
hist_cohort_wer = [24.08, 30.77]
corr_cohort_wer = [24.08, 24.66]
x_c = np.arange(len(cohorts))
r1 = ax2.bar(x_c - width/2, hist_cohort_wer, width, label='Historical WER (%)', color=colors['accent'])
r2 = ax2.bar(x_c + width/2, corr_cohort_wer, width, label='Corrected WER (%)', color=colors['secondary'])
ax2.set_ylabel('Cohort Word Error Rate (%)')
ax2.set_title('B. Resampling Impact by Recording Cohort', pad=15)
ax2.set_xticks(x_c)
ax2.set_xticklabels(cohorts)
ax2.set_ylim(0, 35)
ax2.legend()
ax2.grid(axis='y', linestyle=':', alpha=0.6)
for r in r1 + r2:
    h = r.get_height()
    ax2.annotate(f'{h:.2f}%', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig3_resampling_integrity_impact.png"))
fig.savefig(os.path.join(fig_dir, "fig3_resampling_integrity_impact.svg"))
fig.savefig(os.path.join(fig_dir, "fig3_resampling_integrity_impact.pdf"))
plt.close(fig)

# --- Figure 4: Oracle Headroom ---
fig, ax = plt.subplots(figsize=(8, 5))
n_candidates = [1, 4, 8, 16, 32, 64]
oracle_wer = [24.46, 21.54, 20.26, 19.23, 18.51, 18.32]
oracle_errors = [4121, 3629, 3412, 3240, 3118, 3086]
ax.plot(n_candidates, oracle_wer, marker='D', linewidth=2.5, markersize=8, color=colors['purple'], label='Oracle WER (%)')
ax.axhline(y=24.46, color=colors['neutral'], linestyle=':', label='Top-1 Baseline (24.46%)')
ax.axhline(y=19.23, color=colors['secondary'], linestyle='--', label='16-Best Oracle Ceiling (19.23%)')
for i, n in enumerate(n_candidates):
    diff = 4121 - oracle_errors[i]
    ax.annotate(f"N={n}\n{oracle_wer[i]}%\n(-{diff} errs)", (n, oracle_wer[i]),
                xytext=(0, -22 if i%2==0 else 12), textcoords='offset points', ha='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="#F3E5F5", ec="#CE93D8", alpha=0.8))
ax.set_xscale('log', base=2)
ax.set_xticks(n_candidates)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('Beam Search Candidate Depth (N)')
ax.set_ylabel('Oracle Word Error Rate (%) [dev_assessment.jsonl]')
ax.set_title('N-Best Candidate Oracle WER Ceiling & Recoverable Headroom', pad=15)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig4_oracle_headroom.png"))
fig.savefig(os.path.join(fig_dir, "fig4_oracle_headroom.svg"))
fig.savefig(os.path.join(fig_dir, "fig4_oracle_headroom.pdf"))
plt.close(fig)

# --- Figure 5: Leaderboard ---
fig, ax = plt.subplots(figsize=(10, 5.5))
models = ['E-03 (Stg 15)', 'E-04 (Stg 15)', 'B-02C (Stg 9)', 'E-02 (Stg 15)', 'S16-05 (Stg 16)',
          'WA-02 (Stg 14)', 'E-01 (Stg 15)', 'U-02 (Stg 13)', 'B-02A (Stg 9)', 'R-02 (Stg 12)']
wers = [24.46, 24.46, 24.47, 24.47, 24.47, 24.51, 24.52, 24.91, 25.71, 28.94]
errors = [4121, 4121, 4122, 4122, 4122, 4129, 4130, 4196, 4331, 4875]
bar_colors = [colors['secondary'] if w < 24.5 else (colors['primary'] if w < 26.0 else colors['accent']) for w in wers]
y_pos = np.arange(len(models))
rects = ax.barh(y_pos, wers, align='center', color=bar_colors, edgecolor='none', height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.invert_yaxis()
ax.set_xlabel('Canonical KenLM Word Error Rate (%) [dev_assessment.jsonl]')
ax.set_title('Canonical Acoustic Checkpoint Leaderboard (Audited at 16 kHz)', pad=15)
ax.set_xlim(22, 30)
ax.grid(axis='x', linestyle=':', alpha=0.6)
for i, rect in enumerate(rects):
    w = rect.get_width()
    ax.annotate(f"{w:.2f}% ({errors[i]} errs)", xy=(w, rect.get_y() + rect.get_height()/2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig5_acoustic_leaderboard.png"))
fig.savefig(os.path.join(fig_dir, "fig5_acoustic_leaderboard.svg"))
fig.savefig(os.path.join(fig_dir, "fig5_acoustic_leaderboard.pdf"))
plt.close(fig)
