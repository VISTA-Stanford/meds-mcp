#!/usr/bin/env python3
"""
Update readmission_baseline report:
  - Replace context examples with a long-timeline patient (115970492)
  - Add lumia_timeline_lengths.png (histogram + truncation summary)
"""

import copy
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, "/home/Ayeeshi/meds-mcp/src")
sys.path.insert(0, "/home/Ayeeshi/meds-mcp")

from experiments.fewshot_with_labels.lumia_filter import filter_xml_by_date

# ── paths ─────────────────────────────────────────────────────────────────────
EHRSHOT = Path("/home/Ayeeshi/meds-mcp/experiments/fewshot_with_labels/outputs/ehrshot")
OUT     = Path("/home/Ayeeshi/meds-mcp/experiments/fewshot_with_labels/outputs/readmission_baseline")
CORPUS  = Path("/home/Ayeeshi/meds-mcp/data/ehrshot_lumia/meds_corpus")
POOL    = EHRSHOT / "pool_test_100.json"
ITEMS   = EHRSHOT / "items.jsonl"
CONTEXTS = OUT / "contexts"
PLOTS    = OUT / "plots"

SYSTEM_PROMPT = (
    "You are a clinical-prediction assistant. Answer the user's question about the query patient "
    "using only the evidence in the user message. Respond with exactly one word and nothing else: "
    "Yes or No. Do not include punctuation, reasoning, or any other text."
)
TASK_DESC = (
    'Given a current patient clinical summary, written at time of discharge, '
    'predict the risk that the patient will have a hospital readmission within 30 days of discharge.\n\n'
    'Answer "Yes" if the summary contains evidence or clinical risk factors strongly associated '
    'with a new inpatient admission within 30 days after discharge. '
    'Answer "No" if there is no evidence to suggest a risk of readmission, '
    'only outpatient follow-up or ED-only visits.\n\n'
    'Respond with exactly "Yes" or "No".'
)
CAPS = [4096, 8192, 16384, 32768]
EXAMPLE_PID = "115970492"
EXAMPLE_PID_EMBED = "2019-01-06T23:59:00"


# ── helpers ───────────────────────────────────────────────────────────────────
def get_filtered_tree(pid, embed_time_str):
    cutoff = (datetime.fromisoformat(embed_time_str) + timedelta(days=1)).strftime("%Y-%m-%d")
    return filter_xml_by_date(str(CORPUS / f"{pid}.xml"), cutoff)


def tree_bytes(tree):
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def apply_cap(base_tree, max_chars):
    tree = copy.deepcopy(base_tree)
    root = tree.getroot()
    encounters = root.findall("encounter")
    while encounters:
        if len(tree_bytes(tree)) <= max_chars:
            break
        root.remove(encounters.pop(0))
    ET.indent(tree)
    return tree_bytes(tree).decode("utf-8")


# ── 1. Generate context examples for the long-timeline patient ────────────────
print(f"Generating context examples for patient {EXAMPLE_PID}...")
base_tree = get_filtered_tree(EXAMPLE_PID, EXAMPLE_PID_EMBED)
full_xml_chars = len(tree_bytes(base_tree))
print(f"  Full filtered XML: {full_xml_chars:,} chars")

for cap in CAPS:
    xml_text = apply_cap(base_tree, cap)
    actual = len(xml_text)
    prompt = (
        f"TASK:\n{TASK_DESC}\n\n"
        f"PATIENT TIMELINE:\n{xml_text}\n"
    )
    fname = f"example_prompt_baseline_lumia_{cap}.txt"
    content = (
        f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n"
        f"=== USER PROMPT ===\n{prompt}\n"
        f"\n[example patient: {EXAMPLE_PID}, untruncated filtered: {full_xml_chars:,} chars, "
        f"cap: {cap:,}, actual: {actual:,} chars]\n"
    )
    (CONTEXTS / fname).write_text(content, encoding="utf-8")
    print(f"  cap={cap:,}: {actual:,} chars → {fname}")


# ── 2. Compute lengths for all 100-pool patients ──────────────────────────────
print("\nComputing timeline lengths for all pool patients...")
pool_ids = [str(x) for x in json.load(open(POOL))]
embed_times = {}
with open(ITEMS) as f:
    for line in f:
        it = json.loads(line)
        pid = str(it["person_id"])
        if pid in pool_ids and it["task"] == "guo_readmission" and pid not in embed_times:
            embed_times[pid] = it["embed_time"]

filtered_lens = []
for pid in pool_ids:
    xml_path = CORPUS / f"{pid}.xml"
    embed_time = embed_times.get(pid)
    if not xml_path.exists() or not embed_time:
        continue
    tree = get_filtered_tree(pid, embed_time)
    fc = len(tree_bytes(tree))
    filtered_lens.append(fc)

filtered_lens = sorted(filtered_lens)
n = len(filtered_lens)
print(f"  {n} patients | min={min(filtered_lens):,} median={filtered_lens[n//2]:,} max={max(filtered_lens):,}")


# ── 3. Plot: histogram + truncation overlay ───────────────────────────────────
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: histogram of untruncated filtered lengths (in KB)
ax = axes[0]
lens_kb = [v / 1000 for v in filtered_lens]
bins = np.logspace(np.log10(min(lens_kb)), np.log10(max(lens_kb)), 25)
ax.hist(lens_kb, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6, alpha=0.85)

cap_colors = ["#55A868", "#f0a500", "#C44E52", "#8172B2"]
for cap, color, label in zip(CAPS, cap_colors, ["4 K", "8 K", "16 K", "32 K"]):
    ax.axvline(cap / 1000, color=color, linestyle="--", linewidth=1.5, label=f"{label} cap")

ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}K" if x >= 1 else f"{x*1000:.0f}"))
ax.set_xlabel("Untruncated filtered XML length (chars)")
ax.set_ylabel("Number of patients")
ax.set_title("Distribution of LUMIA Timeline Lengths\n(after date filtering, before cap)")
ax.legend(fontsize=9, loc="upper left")

# Right: stacked bar showing kept vs truncated chars at each cap
ax2 = axes[1]
cap_labels = ["4 K", "8 K", "16 K", "32 K"]
kept_means = []
truncated_means = []
pct_truncated = []

for cap in CAPS:
    kept = [min(v, cap) for v in filtered_lens]
    trunc = [max(0, v - cap) for v in filtered_lens]
    kept_means.append(np.mean(kept))
    truncated_means.append(np.mean(trunc))
    pct_truncated.append(100 * sum(v > cap for v in filtered_lens) / n)

x = np.arange(len(CAPS))
bar_w = 0.55
b1 = ax2.bar(x, [k / 1000 for k in kept_means], bar_w,
             color="#4C72B0", label="Kept (mean)", edgecolor="white")
b2 = ax2.bar(x, [t / 1000 for t in truncated_means], bar_w,
             bottom=[k / 1000 for k in kept_means],
             color="#C44E52", alpha=0.75, label="Truncated (mean)", edgecolor="white")

ax2.set_xticks(x)
ax2.set_xticklabels(cap_labels)
ax2.set_xlabel("Context length cap (chars)")
ax2.set_ylabel("Mean chars per patient (thousands)")
ax2.set_title("Mean Kept vs Truncated per Patient\nby Context Length Cap")
ax2.legend(fontsize=9)

for i, (xi, pct) in enumerate(zip(x, pct_truncated)):
    total_h = (kept_means[i] + truncated_means[i]) / 1000
    ax2.text(xi, total_h + 8, f"{pct:.0f}% truncated",
             ha="center", va="bottom", fontsize=9, color="#C44E52")

ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}K"))

fig.suptitle(f"LUMIA Timeline Analysis — {n} readmission cohort patients", fontsize=13, y=1.01)
fig.tight_layout()
out = PLOTS / "lumia_timeline_lengths.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out}")
print(f"\nAll done. Context files updated in: {CONTEXTS}")
