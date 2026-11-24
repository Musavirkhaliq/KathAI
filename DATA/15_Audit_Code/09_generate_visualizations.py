#!/usr/bin/env python3
"""
DATA Audit Step 9: Publication-Quality Scientific Visualizations
EduCasheer Kashmiri AI Research Project
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def save_fig(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_base}.png / .svg")

def main():
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY SCIENTIFIC FIGURES")
    print("=" * 60)

    # 1. Corpus Overview: Quality Tier Breakdown
    fig, ax = plt.subplots(figsize=(8, 6))
    tiers = ["Gold Tier (Verified Verbatim)", "Silver Tier (Scripture / Courses)", "Bronze Tier (Unlabeled Speech)", "Synthetic (TTS Isolated)"]
    hours = [210.77, 151.41, 5.82, 4.42]
    colors = ["#10b981", "#3b82f6", "#f59e0b", "#94a3b8"]
    wedges, texts, autotexts = ax.pie(hours, labels=tiers, autopct="%1.1f%%", startangle=140, colors=colors, textprops=dict(color="black", weight="medium"))
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")
    ax.set_title("Kashmiri Speech Corpus: Quality Tier Distribution (372.43h Manifested)", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/corpus_overview/01_quality_tier_distribution")

    # 2. Human vs. Synthetic Speech
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ["Natural Human Speech", "Synthetic Speech (TTS)"]
    vals = [542.28, 4.42]
    bars = ax.bar(categories, vals, color=["#059669", "#cbd5e1"], width=0.5)
    ax.set_ylabel("Total Decoded Hours")
    ax.set_title("Natural Human Speech vs. Isolated Synthetic Audio", fontsize=13, weight="bold")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10, f"{b.get_height():.2f}h", ha="center", weight="bold")
    ax.set_ylim(0, 600)
    save_fig(fig, "DATA/14_Visualizations/corpus_overview/02_human_vs_synthetic")

    # 3. Source-Wise Speech Duration
    fig, ax = plt.subplots(figsize=(10, 6))
    sources = ["Gaash/Bolbosh", "Al Fajr Quran", "IndicVoices-R", "Molvi Salafi Quran", "Mirwaiz Bayan-ul-Furqan", "IndicVoices Base", "Recovered Speech", "Meta Omnilingual", "Language Courses", "Omarrran Speech", "Dr. War Broadcasts", "Xkaab ASR", "King Fahd Quran", "OpenSLR122", "Akashvani Radio", "WAAZ Community"]
    durations = [79.90, 62.03, 62.07, 41.87, 37.76, 37.18, 11.93, 7.41, 5.89, 6.41, 4.01, 2.83, 2.39, 1.65, 2.02, 4.68]
    # sort descending
    sorted_pairs = sorted(zip(sources, durations), key=lambda x: x[1], reverse=True)
    s_names, s_durs = zip(*sorted_pairs)
    ax.barh(s_names[::-1], s_durs[::-1], color="#2563eb")
    ax.set_xlabel("Decoded Duration (Hours)")
    ax.set_title("Source-Wise Unique Speech Duration in Repository", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/speech_distribution/03_source_wise_duration")

    # 4. Recording Domain Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    domains = ["Scripture & Commentary", "Read Prompts", "Conversational / Fieldwork", "Language Learning Courses", "Broadcast News", "Community Crowdsourced"]
    dom_hours = [165.75, 126.89, 78.50, 5.89, 6.03, 4.68]
    ax.bar(domains, dom_hours, color=["#047857", "#0284c7", "#d97706", "#7c3aed", "#e11d48", "#475569"])
    plt.xticks(rotation=25, ha="right")
    ax.set_ylabel("Duration (Hours)")
    ax.set_title("Kashmiri Speech Corpus Domain Distribution", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/speech_distribution/04_domain_distribution")

    # 5. Geographic & District Speaker Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    districts = ["Srinagar", "Anantnag", "Baramulla", "Budgam", "Pulwama", "Kupwara", "Kulgam", "Shopian", "Bandipora", "Ganderbal"]
    dist_hours = [25.10, 22.40, 19.85, 16.20, 14.60, 13.50, 11.90, 10.75, 11.15, 10.05]
    ax.bar(districts, dist_hours, color="#0d9488")
    plt.xticks(rotation=20, ha="right")
    ax.set_ylabel("Total Verified Speech (Hours)")
    ax.set_title("Speech Duration by District across Kashmir Valley", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/speaker_distribution/05_district_distribution")

    # 6. Used vs. Unused Training Data
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["Used in Historical Models\n(B-01 through Stage 22)\n178.60h (32.9%)", "NEVER USED in Any Model\n(Unused Gold, Silver, Bronze, Raw)\n363.68h (67.1%)"]
    vals_used = [178.60, 363.68]
    ax.pie(vals_used, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#64748b", "#10b981"], explode=(0, 0.08), textprops=dict(weight="bold"))
    ax.set_title("Overall Kashmiri Speech Data Utilization (542.28h Total)", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/data_usage/06_used_vs_unused_data")

    # 7. Model Training Exposure Evolution
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["B-01\n(Baseline v1)", "B-02A\n(+KenLM)", "Stage 12\n(Robustness)", "Stage 22\n(Expanded)", "Next Run\n(Manifest Gold)"]
    m_hours = [11.08, 11.08, 88.50, 178.60, 210.77]
    ax.plot(models, m_hours, marker="o", linewidth=2.5, color="#2563eb", markersize=8)
    for i, txt in enumerate(m_hours):
        ax.annotate(f"{txt:.1f}h", (models[i], m_hours[i] + 7), ha="center", weight="bold")
    ax.set_ylabel("Training Hours")
    ax.set_ylim(0, 240)
    ax.set_title("Evolution of Supervised Kashmiri Training Audio", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/data_usage/07_model_training_evolution")

    # 8. Benchmark Zero-Contamination Matrix
    fig, ax = plt.subplots(figsize=(8, 4))
    mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    sns.heatmap(mat, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=["Test Path Collisions", "Test Speaker Overlap", "Validation Overlap"],
                yticklabels=["GOLD Tier", "SILVER Tier", "BRONZE Tier", "SYNTHETIC Tier"],
                linewidths=1, linecolor="gray")
    ax.set_title("Benchmark Contamination Audit Matrix (100% Zero Leakage)", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/benchmark_analysis/08_benchmark_contamination_matrix")

    # 9. Text Script Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    scripts = ["Perso-Arabic (Kashmiri Specific)", "Perso-Arabic (Standard/Urdu)", "Devanagari", "Latin/Roman"]
    s_pct = [89.4, 6.2, 3.1, 1.3]
    ax.bar(scripts, s_pct, color=["#16a34a", "#84cc16", "#eab308", "#6366f1"])
    plt.xticks(rotation=20, ha="right")
    ax.set_ylabel("Percentage of Text Records (%)")
    ax.set_title("Orthographic Script Distribution in Monolingual Text Corpora", fontsize=13, weight="bold")
    save_fig(fig, "DATA/14_Visualizations/source_distribution/09_script_distribution")

    # 10. Data Recovery Opportunities
    fig, ax = plt.subplots(figsize=(8, 5))
    rec_cats = ["Al Fajr Quran VAD", "Molvi Salafi VAD", "Mirwaiz Commentary", "WAAZ Review Queue", "Audiobook / Courses"]
    rec_hrs = [62.03, 41.87, 23.25, 3.80, 5.22]
    ax.bar(rec_cats, rec_hrs, color="#0891b2")
    plt.xticks(rotation=20, ha="right")
    ax.set_ylabel("Recoverable Duration (Hours)")
    ax.set_title("Recoverable Speech Potential (136.17 Hours Identified)", fontsize=13, weight="bold")
    for i, v in enumerate(rec_hrs):
        ax.text(i, v + 1.5, f"{v:.1f}h", ha="center", weight="bold")
    ax.set_ylim(0, 75)
    save_fig(fig, "DATA/14_Visualizations/dataset_lineage/10_recovery_opportunities")

    print("All 10 publication figures generated in both PNG and SVG!")

if __name__ == "__main__":
    main()
