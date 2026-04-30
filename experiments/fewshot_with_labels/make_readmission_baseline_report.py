#!/usr/bin/env python3
"""
Generate readmission_baseline report folder:
  1. plots/vignette_baseline_vs_fewshot.png
  2. plots/lumia_context_length_bar.png
  3. plots/lumia_context_length_line.png
  4. Copies / prints example prompts for all 6 contexts
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
EHRSHOT = Path("/home/Ayeeshi/meds-mcp/experiments/fewshot_with_labels/outputs/ehrshot")
OUT = Path("/home/Ayeeshi/meds-mcp/experiments/fewshot_with_labels/outputs/readmission_baseline")
PLOTS = OUT / "plots"
CONTEXTS = OUT / "contexts"
for d in (PLOTS, CONTEXTS):
    d.mkdir(parents=True, exist_ok=True)

# ── load summary ─────────────────────────────────────────────────────────────
with open(EHRSHOT / "analysis_summary.json") as f:
    summary = json.load(f)
pv = summary["per_variant"]

def balacc(key):
    return pv[key]["balanced_accuracy"]

def n(key):
    return pv[key]["n"]

# ── shared style ──────────────────────────────────────────────────────────────
COLORS = {
    "baseline": "#4C72B0",
    "fewshot":  "#DD8452",
    "lumia":    ["#55A868", "#C44E52", "#8172B2", "#937860"],
}
BAR_W = 0.45

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def label_bars(ax, bars, vals):
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )


# ── plot 1: vignette baseline vs fewshot ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
contexts = ["baseline_vignette", "vignette"]
labels   = ["Baseline\n(zero-shot)", "Few-shot\n(top-3)"]
colors   = [COLORS["baseline"], COLORS["fewshot"]]
vals     = [balacc(c) for c in contexts]
ns       = [n(c) for c in contexts]

bars = ax.bar(labels, vals, color=colors, width=BAR_W, edgecolor="white", linewidth=0.8)
label_bars(ax, bars, vals)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
ax.set_ylim(0.45, 0.60)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax.set_ylabel("Balanced Accuracy")
ax.set_title("Readmission: Vignette Context\nBaseline vs Few-shot (n=234)")
ax.legend(fontsize=9)

for i, (bar, nval) in enumerate(zip(bars, ns)):
    ax.text(bar.get_x() + bar.get_width() / 2, 0.452, f"n={nval}",
            ha="center", va="bottom", fontsize=9, color="white")

fig.tight_layout()
out1 = PLOTS / "vignette_baseline_vs_fewshot.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved {out1}")


# ── plot 2: lumia context lengths bar chart ───────────────────────────────────
lumia_keys = ["baseline_lumia_4096", "baseline_lumia_8192",
              "baseline_lumia_16384", "baseline_lumia_32768"]
lumia_labels = ["4 K", "8 K", "16 K", "32 K"]
lumia_vals   = [balacc(k) for k in lumia_keys]
lumia_n      = [n(k) for k in lumia_keys]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(lumia_labels, lumia_vals, color=COLORS["lumia"], width=BAR_W,
              edgecolor="white", linewidth=0.8)
label_bars(ax, bars, lumia_vals)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
ax.set_ylim(0.45, 0.56)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax.set_xlabel("Max chars per patient (LUMIA timeline)")
ax.set_ylabel("Balanced Accuracy")
ax.set_title("Readmission: LUMIA Context-Length Sweep\n(zero-shot, n=173 per condition)")
ax.legend(fontsize=9)

fig.tight_layout()
out2 = PLOTS / "lumia_context_length_bar.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved {out2}")


# ── plot 3: lumia context lengths line chart ──────────────────────────────────
x_chars = [4096, 8192, 16384, 32768]
x_labels = ["4 K", "8 K", "16 K", "32 K"]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_chars, lumia_vals, marker="o", color=COLORS["lumia"][0],
        linewidth=2, markersize=7, label="LUMIA (zero-shot)")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
ax.axhline(balacc("baseline_vignette"), color=COLORS["baseline"], linestyle=":",
           linewidth=1.5, label=f"Vignette baseline ({balacc('baseline_vignette'):.3f})")
ax.axhline(balacc("vignette"), color=COLORS["fewshot"], linestyle=":",
           linewidth=1.5, label=f"Vignette few-shot ({balacc('vignette'):.3f})")

for x, y in zip(x_chars, lumia_vals):
    ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9)

ax.set_xscale("log", base=2)
ax.set_xticks(x_chars)
ax.set_xticklabels(x_labels)
ax.set_ylim(0.45, 0.58)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax.set_xlabel("Max chars per patient (LUMIA timeline)")
ax.set_ylabel("Balanced Accuracy")
ax.set_title("Readmission: Balanced Accuracy vs LUMIA Context Length")
ax.legend(fontsize=9, loc="lower right")

fig.tight_layout()
out3 = PLOTS / "lumia_context_length_line.png"
fig.savefig(out3, dpi=150)
plt.close(fig)
print(f"Saved {out3}")


# ── copy / print context files ────────────────────────────────────────────────
prompt_files = {
    "baseline_vignette":     "example_prompt_baseline_vignette.txt",
    "vignette":              "example_prompt_vignette.txt",
    "baseline_lumia_4096":   "example_prompt_baseline_lumia_4096.txt",
    "baseline_lumia_8192":   "example_prompt_baseline_lumia_8192.txt",
    "baseline_lumia_16384":  "example_prompt_baseline_lumia_16384.txt",
    "baseline_lumia_32768":  "example_prompt_baseline_lumia_32768.txt",
}

print("\n" + "=" * 70)
print("EXAMPLE PROMPTS")
print("=" * 70)

for ctx, fname in prompt_files.items():
    src = EHRSHOT / fname
    dst = CONTEXTS / fname
    if src.exists():
        shutil.copy2(src, dst)
        content = src.read_text()
        chars = len(content)
        lines = content.count("\n")
        print(f"\n{'─'*70}")
        print(f"  Context: {ctx}  ({chars:,} chars, {lines} lines)")
        print(f"  Copied → contexts/{fname}")
        print(f"{'─'*70}")
        print(content[:3000])
        if len(content) > 3000:
            print(f"\n  ... [{chars - 3000:,} more chars] ...")
    else:
        print(f"\nWARNING: {src} not found")

# ── summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BALANCED ACCURACY SUMMARY")
print("=" * 70)
all_keys = list(prompt_files.keys())
for k in all_keys:
    print(f"  {k:<30s}  bal_acc={balacc(k):.4f}  n={n(k)}")

print(f"\nAll outputs written to: {OUT}")
