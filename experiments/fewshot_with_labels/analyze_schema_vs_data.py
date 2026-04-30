#!/usr/bin/env python3
"""
Analyze the ratio of schema/template text vs. patient-specific value text
in VISTA and EHRSHOT rubricified records, and show why TF-IDF retrieval
quality differs between the two datasets.

Outputs saved to outputs/plots/schema_analysis_*.png
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VISTA_BASE   = Path.home() / "LRRL_MEDS/data/vista_rubric/rubricified"
EHRSHOT_BASE = Path.home() / "LRRL_MEDS/data/rubric/rubricified"
PLOTS_DIR    = Path(__file__).parent / "outputs/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MISSING_TOKENS = {"not documented", "not specified", "none", "no data",
                  "not staged", "unknown", "n/a", "not applicable",
                  "not mentioned", "not reported", "not available"}

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _extract_json_schema_value(obj) -> tuple[int, int, int]:
    """Return (schema_chars, value_chars, missing_value_fields) from one field object."""
    schema_keys = {"field_name", "extraction_instructions", "format", "missing_value",
                   "description", "type", "extraction_guidance",
                   "name", "rubric_name", "query"}
    value_keys  = {"value", "extracted_value"}
    s, v, m = 0, 0, 0
    for k, val in obj.items():
        k_lower = k.lower()
        val_str = str(val)
        if k_lower in value_keys:
            v += len(val_str)
            if val_str.strip().lower() in MISSING_TOKENS:
                m += 1
        elif k_lower in schema_keys:
            s += len(val_str)
    return s, v, m


def parse_rubric_text(text: str) -> dict:
    """
    Parse a rubricified_text and return:
      schema_chars, value_chars, n_fields, n_missing_fields
    Handles JSON-array, nested-JSON, fields-list, and markdown formats.
    """
    text = text.strip()
    # strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\n?|```", "", text).strip()

    # -- Try JSON --
    try:
        obj = json.loads(cleaned)

        # Format 1: flat list of field dicts
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            s = v = m = 0
            for item in obj:
                si, vi, mi = _extract_json_schema_value(item)
                s += si; v += vi; m += mi
            return dict(schema_chars=s, value_chars=v,
                        n_fields=len(obj), n_missing=m)

        # Format 2: nested object (EHRSHOT style) or fields-list
        if isinstance(obj, dict):
            # Check for {"rubric_name":..., "fields":[{"name":..,"value":..}]}
            if "fields" in obj and isinstance(obj["fields"], list):
                s = v = m = 0
                n = 0
                # top-level schema keys
                for k in ("rubric_name", "query"):
                    if k in obj:
                        s += len(str(obj[k]))
                for field in obj["fields"]:
                    si, vi, mi = _extract_json_schema_value(field)
                    s += si; v += vi; m += mi; n += 1
                return dict(schema_chars=s, value_chars=v,
                            n_fields=n, n_missing=m)

            # Check if this is a flat RUBRIC_TEMPLATE style dict
            # (EHRSHOT: {"AGE": 70, "SEX": "MALE", ...} — no "value" keys anywhere)
            def _has_value_keys(d):
                for k, v in d.items():
                    if k.lower() == "value":
                        return True
                    if isinstance(v, dict) and _has_value_keys(v):
                        return True
                return False

            if not _has_value_keys(obj):
                # Flat format: keys are field names, leaf values are patient data
                def _flat_recurse(d):
                    s = v = m = n = 0
                    for k, val in d.items():
                        s += len(str(k))
                        if isinstance(val, dict):
                            si, vi, mi, ni = _flat_recurse(val)
                            s += si; v += vi; m += mi; n += ni
                        elif isinstance(val, list):
                            for item in val:
                                if isinstance(item, dict):
                                    si, vi, mi, ni = _flat_recurse(item)
                                    s += si; v += vi; m += mi; n += ni
                                else:
                                    val_str = str(item)
                                    v += len(val_str)
                                    if val_str.strip().lower() in MISSING_TOKENS:
                                        m += 1
                                    n += 1
                        else:
                            val_str = str(val)
                            v += len(val_str)
                            if val_str.strip().lower() in MISSING_TOKENS:
                                m += 1
                            n += 1
                    return s, v, m, n
                si, vi, mi, ni = _flat_recurse(obj)
                return dict(schema_chars=si, value_chars=vi,
                            n_fields=ni, n_missing=mi)

            # Recursive nested dict with "value" keys (EHRSHOT detailed style)
            def _recurse(d, depth=0):
                s = v = m = n = 0
                for k, val in d.items():
                    k_lower = k.lower()
                    if k_lower == "value":
                        val_str = str(val)
                        v += len(val_str)
                        if val_str.strip().lower() in MISSING_TOKENS:
                            m += 1
                        n += 1
                    elif isinstance(val, dict):
                        si, vi, mi, ni = _recurse(val, depth+1)
                        s += si; v += vi; m += mi; n += ni
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                si, vi, mi, ni = _recurse(item, depth+1)
                                s += si; v += vi; m += mi; n += ni
                            else:
                                s += len(str(item))
                    else:
                        s += len(str(val))
                return s, v, m, n
            si, vi, mi, ni = _recurse(obj)
            return dict(schema_chars=si, value_chars=vi,
                        n_fields=ni, n_missing=mi)
    except (json.JSONDecodeError, TypeError):
        pass

    # -- Markdown format: handles both inline and nested bullet styles:
    #    **FIELD:** value    /    N. **FIELD**: value    /    *   **FIELD:** value --
    schema_chars = 0
    value_chars  = 0
    n_fields     = 0
    n_missing    = 0
    # Match any line that contains **text**: followed by a value
    field_pat = re.compile(r".*\*\*([^*]+?)\*\*:?\s+(.*)")
    for line in text.split("\n"):
        m = field_pat.match(line.strip())
        if m:
            field_name = m.group(1).strip().rstrip(":")
            val = m.group(2).strip()
            if val:  # only count if there's an actual value on this line
                schema_chars += len(field_name)
                value_chars += len(val)
                n_fields += 1
                if val.lower() in MISSING_TOKENS:
                    n_missing += 1
            else:
                schema_chars += len(line)  # section header, no value
        else:
            schema_chars += len(line)
    if n_fields > 0:
        return dict(schema_chars=schema_chars, value_chars=value_chars,
                    n_fields=n_fields, n_missing=n_missing)

    # Fallback: treat everything as schema
    return dict(schema_chars=len(text), value_chars=0,
                n_fields=0, n_missing=0)


# ---------------------------------------------------------------------------
# Load and analyse one dataset
# ---------------------------------------------------------------------------

def analyse_dataset(base: Path, label: str, n_sample: int = 200) -> dict:
    """Return per-task stats dict and sampled texts for TF-IDF analysis."""
    task_stats = {}
    task_texts = {}   # task -> list of rubricified_text (train only, for TF-IDF)

    for task_dir in sorted(base.iterdir()):
        task = task_dir.name
        schema_chars_all = []
        value_chars_all  = []
        n_fields_all     = []
        n_missing_all    = []
        train_texts      = []

        for split_file in sorted(task_dir.iterdir()):
            split = split_file.stem
            with open(split_file) as f:
                records = json.load(f)
            for r in records:
                parsed = parse_rubric_text(r.get("rubricified_text", ""))
                schema_chars_all.append(parsed["schema_chars"])
                value_chars_all.append(parsed["value_chars"])
                n_fields_all.append(parsed["n_fields"])
                n_missing_all.append(parsed["n_missing"])
                if split == "train":
                    train_texts.append(r.get("rubricified_text", ""))

        total = sum(s + v for s, v in zip(schema_chars_all, value_chars_all))
        task_stats[task] = {
            "mean_schema_chars": np.mean(schema_chars_all),
            "mean_value_chars":  np.mean(value_chars_all),
            "value_ratio":       np.mean([v/(s+v+1) for s, v in
                                          zip(schema_chars_all, value_chars_all)]),
            "mean_n_fields":     np.mean(n_fields_all),
            "mean_n_missing":    np.mean(n_missing_all),
            "missing_ratio":     np.mean([m/(n+1) for m, n in
                                          zip(n_missing_all, n_fields_all)]),
            "n_records":         len(schema_chars_all),
        }
        task_texts[task] = train_texts[:n_sample]

    return task_stats, task_texts


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity distribution per task
# ---------------------------------------------------------------------------

def compute_cosine_stats(task_texts: dict, n_sample: int = 150) -> dict:
    """For each task, fit TF-IDF on up to n_sample train texts and return
    a random sample of pairwise cosine similarities."""
    stats = {}
    for task, texts in task_texts.items():
        if len(texts) < 5:
            continue
        sample = texts[:n_sample]
        try:
            vec = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
            mat = vec.fit_transform(sample)
            # sample 300 random pairs
            rng = np.random.default_rng(42)
            n = mat.shape[0]
            idx_a = rng.integers(0, n, 300)
            idx_b = rng.integers(0, n, 300)
            sims = np.array([
                cosine_similarity(mat[a], mat[b])[0, 0]
                for a, b in zip(idx_a, idx_b) if a != b
            ])
            stats[task] = sims
        except Exception:
            pass
    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_value_ratio(vista_stats, ehrshot_stats):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Patient Value Text as % of Total Rubric Text\n"
                 "(schema = field names + instructions + format; value = extracted patient data)",
                 fontsize=13, fontweight="bold")

    for ax, stats, title, color in [
        (axes[0], vista_stats,   "VISTA (40 tasks)",   "#d62728"),
        (axes[1], ehrshot_stats, "EHRSHOT (15 tasks)", "#1f77b4"),
    ]:
        tasks  = list(stats.keys())
        ratios = [stats[t]["value_ratio"] * 100 for t in tasks]
        schema = [stats[t]["mean_schema_chars"] for t in tasks]
        value  = [stats[t]["mean_value_chars"]  for t in tasks]

        y = range(len(tasks))
        bars = ax.barh(y, ratios, color=color, alpha=0.8)
        ax.axvline(np.mean(ratios), color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean={np.mean(ratios):.1f}%")
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace("_", "\n") for t in tasks], fontsize=7)
        ax.set_xlabel("Value text %  (higher = more patient-specific content)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, max(ratios) * 1.15)

        for bar, r in zip(bars, ratios):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{r:.1f}%", va="center", fontsize=6.5)

    plt.tight_layout()
    path = PLOTS_DIR / "schema_fig1_value_ratio.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  fig1 → {path}")


def plot_missing_ratio(vista_stats, ehrshot_stats):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Fraction of Value Fields Containing 'Not Documented' / Missing Tokens\n"
                 "(high missing rate → TF-IDF has little discriminative value signal)",
                 fontsize=13, fontweight="bold")

    for ax, stats, title, color in [
        (axes[0], vista_stats,   "VISTA (40 tasks)",   "#d62728"),
        (axes[1], ehrshot_stats, "EHRSHOT (15 tasks)", "#1f77b4"),
    ]:
        tasks   = list(stats.keys())
        missing = [stats[t]["missing_ratio"] * 100 for t in tasks]
        y = range(len(tasks))
        bars = ax.barh(y, missing, color=color, alpha=0.8)
        ax.axvline(np.mean(missing), color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean={np.mean(missing):.1f}%")
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace("_", "\n") for t in tasks], fontsize=7)
        ax.set_xlabel("% of value fields that are missing/unknown")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

        for bar, m in zip(bars, missing):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{m:.0f}%", va="center", fontsize=6.5)

    plt.tight_layout()
    path = PLOTS_DIR / "schema_fig2_missing_ratio.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  fig2 → {path}")


def plot_schema_vs_value_absolute(vista_stats, ehrshot_stats):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Mean Characters per Record: Schema Text vs. Patient Value Text",
                 fontsize=13, fontweight="bold")

    for ax, stats, title in [
        (axes[0], vista_stats,   "VISTA"),
        (axes[1], ehrshot_stats, "EHRSHOT"),
    ]:
        tasks  = list(stats.keys())
        schema = [stats[t]["mean_schema_chars"] for t in tasks]
        value  = [stats[t]["mean_value_chars"]  for t in tasks]
        y = range(len(tasks))

        ax.barh(y, schema, color="#aec7e8", label="Schema text")
        ax.barh(y, value,  color="#d62728", label="Value text", left=schema)
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace("_", "\n") for t in tasks], fontsize=7)
        ax.set_xlabel("Mean characters per record")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "schema_fig3_absolute_chars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  fig3 → {path}")


def plot_cosine_distributions(vista_sims, ehrshot_sims):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("TF-IDF Cosine Similarity Distribution Between Train Patients (Within Task)\n"
                 "High similarity = TF-IDF can't distinguish patients → bad retrieval",
                 fontsize=13, fontweight="bold")

    for ax, sims, title, color in [
        (axes[0], vista_sims,   "VISTA",   "#d62728"),
        (axes[1], ehrshot_sims, "EHRSHOT", "#1f77b4"),
    ]:
        if not sims:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        tasks = list(sims.keys())
        data  = [sims[t] for t in tasks]
        means = [np.mean(d) for d in data]
        # sort by mean similarity
        order = np.argsort(means)[::-1]
        tasks_sorted = [tasks[i] for i in order]
        data_sorted  = [data[i]  for i in order]

        bp = ax.boxplot(data_sorted, vert=False, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        ax.set_yticks(range(1, len(tasks_sorted)+1))
        ax.set_yticklabels([t.replace("_", "\n") for t in tasks_sorted], fontsize=7)
        ax.set_xlabel("Cosine similarity (pairwise, random 300 pairs)")
        ax.axvline(0.9, color="red", linestyle="--", linewidth=1.2,
                   label="0.9 threshold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        # annotate means
        for i, m in enumerate(sorted(means, reverse=True)):
            ax.text(m + 0.01, i + 1, f"{m:.2f}", va="center", fontsize=6)

    plt.tight_layout()
    path = PLOTS_DIR / "schema_fig4_cosine_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  fig4 → {path}")


def plot_summary_comparison(vista_stats, ehrshot_stats):
    """Summary scatter: value_ratio vs missing_ratio for all tasks."""
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle("Rubric Quality Space: Value Ratio vs. Missing Rate per Task\n"
                 "Ideal = high value ratio + low missing rate (top-left)",
                 fontsize=13, fontweight="bold")

    for stats, label, color, marker in [
        (vista_stats,   "VISTA",   "#d62728", "o"),
        (ehrshot_stats, "EHRSHOT", "#1f77b4", "s"),
    ]:
        vr = [stats[t]["value_ratio"] * 100    for t in stats]
        mr = [stats[t]["missing_ratio"] * 100  for t in stats]
        ax.scatter(mr, vr, c=color, marker=marker, s=80, alpha=0.8, label=label)
        for t, x, y in zip(stats, mr, vr):
            ax.annotate(t.replace("_", " "), (x, y),
                        fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Missing/unknown value fields (%)", fontsize=11)
    ax.set_ylabel("Patient value text (%)", fontsize=11)
    ax.axvline(25, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(10, color="grey", linestyle=":", alpha=0.5)
    ax.legend(fontsize=11, markerscale=1.4)
    ax.set_title("")
    plt.tight_layout()
    path = PLOTS_DIR / "schema_fig5_quality_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  fig5 → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Analysing VISTA rubricified data …")
    vista_stats, vista_texts = analyse_dataset(VISTA_BASE, "VISTA")
    print("Analysing EHRSHOT rubricified data …")
    ehrshot_stats, ehrshot_texts = analyse_dataset(EHRSHOT_BASE, "EHRSHOT")

    print("Computing TF-IDF cosine distributions …")
    vista_sims   = compute_cosine_stats(vista_texts)
    ehrshot_sims = compute_cosine_stats(ehrshot_texts)

    print("Generating figures …")
    plot_value_ratio(vista_stats, ehrshot_stats)
    plot_missing_ratio(vista_stats, ehrshot_stats)
    plot_schema_vs_value_absolute(vista_stats, ehrshot_stats)
    plot_cosine_distributions(vista_sims, ehrshot_sims)
    plot_summary_comparison(vista_stats, ehrshot_stats)

    # Print summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, stats in [("VISTA", vista_stats), ("EHRSHOT", ehrshot_stats)]:
        vr = np.mean([s["value_ratio"] for s in stats.values()]) * 100
        mr = np.mean([s["missing_ratio"] for s in stats.values()]) * 100
        print(f"{label}:")
        print(f"  Mean value text ratio : {vr:.1f}%")
        print(f"  Mean missing rate     : {mr:.1f}%")
        print(f"  Tasks                 : {len(stats)}")
        worst_vr = min(stats, key=lambda t: stats[t]["value_ratio"])
        best_vr  = max(stats, key=lambda t: stats[t]["value_ratio"])
        print(f"  Lowest value ratio    : {worst_vr} ({stats[worst_vr]['value_ratio']*100:.1f}%)")
        print(f"  Highest value ratio   : {best_vr} ({stats[best_vr]['value_ratio']*100:.1f}%)")
    print()
    print(f"All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
