#!/usr/bin/env python3
"""
Horizontal bar chart: EHR tokens available pre-prediction time vs. context limit.
Each bar is split into 'used' (≤8192) and 'truncated' (>8192) portions.
A vertical line marks the 8192-token clipping limit.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
VISTA_SERIAL_DIR  = Path.home() / "LRRL_MEDS/data/serialized/vista/naivetext"
EHRSHOT_SERIAL_DIR = Path.home() / "LRRL_MEDS/data/serialized/naivetext"
OUT_DIR           = Path.home() / "meds-mcp/experiments/fewshot_with_labels/outputs/plots"
CLIP_LIMIT        = 8192   # MAX_SERIALIZATION_TOKENS from config
SPLITS            = ["train", "val", "test"]

USED_COLOR    = "#4C8BE0"   # blue  – tokens the model actually sees
TRUNC_COLOR   = "#E05C4C"   # red   – tokens cut off by the 8192 limit
LINE_COLOR    = "#222222"

# ---------------------------------------------------------------------------

def load_task_stats(base_dir: Path) -> dict:
    """Returns {task: {"p25": ..., "p50": ..., "p75": ..., "mean": ...,
                       "clip_pct": ..., "n": ...}}"""
    stats = {}
    for task_dir in sorted(base_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        tokens = []
        clipped = []
        for split in SPLITS:
            fpath = task_dir / f"{split}.json"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                recs = json.load(f)
            for r in recs:
                ot = r.get("original_tokens")
                wc = r.get("was_clipped")
                if ot is not None and ot > 0:
                    tokens.append(ot)
                    clipped.append(bool(wc))
        if not tokens:
            continue
        arr = np.array(tokens)
        stats[task] = {
            "p25":      float(np.percentile(arr, 25)),
            "p50":      float(np.median(arr)),
            "p75":      float(np.percentile(arr, 75)),
            "mean":     float(arr.mean()),
            "clip_pct": 100.0 * sum(clipped) / len(clipped),
            "n":        len(arr),
        }
    return stats


def make_panel(ax, stats: dict, title: str, clip_limit: int):
    """Draw one horizontal-bar panel on ax."""
    tasks = sorted(stats.keys(), key=lambda t: stats[t]["p50"])
    n     = len(tasks)
    ys    = np.arange(n)
    bar_h = 0.65

    for i, task in enumerate(tasks):
        s   = stats[task]
        p50 = s["p50"]

        used  = min(p50, clip_limit)
        extra = max(0.0, p50 - clip_limit)

        # Stacked bar: used portion
        ax.barh(i, used, height=bar_h, color=USED_COLOR, linewidth=0)
        # Truncated portion
        if extra > 0:
            ax.barh(i, extra, left=used, height=bar_h,
                    color=TRUNC_COLOR, linewidth=0)

        # IQR whisker (p25 – p75) in dark colour
        p25, p75 = s["p25"], s["p75"]
        ax.plot([p25, p75], [i, i], color="#111111", linewidth=1.2, solid_capstyle="round")
        ax.plot(p25, i, "|", color="#111111", markersize=5)
        ax.plot(p75, i, "|", color="#111111", markersize=5)

        # Clip % annotation to the right
        clip_str = f"{s['clip_pct']:.0f}% clipped"
        ax.text(
            ax.get_xlim()[1] * 0.01 if ax.get_xlim()[1] > 0 else 1000,
            i,
            f"  {clip_str}",
            va="center", ha="left", fontsize=6.5, color="#555555",
        )

    # Context-limit line
    ax.axvline(clip_limit, color=LINE_COLOR, linewidth=1.6,
               linestyle="--", label=f"Context limit ({clip_limit:,} tok)")

    ax.set_yticks(ys)
    ax.set_yticklabels(tasks, fontsize=7.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("EHR tokens (pre-prediction cutoff)  |  median + IQR whiskers", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    # Set x-limit to max p75 across tasks (plus padding) so truncated bars are visible
    max_val = max(s["p75"] for s in stats.values())
    ax.set_xlim(0, max_val * 1.18)

    # Re-annotate after xlim is set (need to nudge label position)
    # Re-draw text at fixed offset from clip line
    for i, task in enumerate(tasks):
        s    = stats[task]
        p50  = s["p50"]
        clip_str = f"{s['clip_pct']:.0f}%"
        ax.text(
            max_val * 1.14,
            i,
            clip_str,
            va="center", ha="right", fontsize=6.5, color="#555555",
        )
    # Column header
    ax.text(max_val * 1.14, n - 0.1, "clipped",
            va="bottom", ha="right", fontsize=6.5, color="#555555", style="italic")


def main():
    print("Loading VISTA serialized data…")
    vista_stats   = load_task_stats(VISTA_SERIAL_DIR)
    print(f"  {len(vista_stats)} tasks")

    print("Loading EHRSHOT serialized data…")
    ehrshot_stats = load_task_stats(EHRSHOT_SERIAL_DIR)
    print(f"  {len(ehrshot_stats)} tasks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Figure: VISTA panel
    # -----------------------------------------------------------------------
    n_vista = len(vista_stats)
    fig_h   = max(8, n_vista * 0.38 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    make_panel(ax, vista_stats, f"VISTA – EHR tokens available pre-prediction cutoff  ({n_vista} tasks)", CLIP_LIMIT)

    used_patch  = mpatches.Patch(color=USED_COLOR,  label=f"Used by model (≤{CLIP_LIMIT:,} tok)")
    trunc_patch = mpatches.Patch(color=TRUNC_COLOR, label=f"Truncated (beyond {CLIP_LIMIT:,} tok)")
    limit_line  = plt.Line2D([0], [0], color=LINE_COLOR, linewidth=1.6,
                             linestyle="--", label=f"Context limit  ({CLIP_LIMIT:,} tok)")
    ax.legend(handles=[used_patch, trunc_patch, limit_line],
              loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out = OUT_DIR / "tokens_bar_vista.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # -----------------------------------------------------------------------
    # Figure: EHRSHOT panel
    # -----------------------------------------------------------------------
    n_ehr = len(ehrshot_stats)
    fig_h = max(5, n_ehr * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    make_panel(ax, ehrshot_stats, f"EHRSHOT – EHR tokens available pre-prediction cutoff  ({n_ehr} tasks)", CLIP_LIMIT)

    ax.legend(handles=[used_patch, trunc_patch, limit_line],
              loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out = OUT_DIR / "tokens_bar_ehrshot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n=== VISTA top-10 most truncated tasks ===")
    by_clip = sorted(vista_stats.items(), key=lambda x: x[1]["clip_pct"], reverse=True)
    for t, s in by_clip[:10]:
        print(f"  {t:45s}  median={s['p50']:>8,.0f}  clip={s['clip_pct']:5.1f}%")

    print("\n=== EHRSHOT tasks ===")
    by_clip = sorted(ehrshot_stats.items(), key=lambda x: x[1]["clip_pct"], reverse=True)
    for t, s in by_clip:
        print(f"  {t:45s}  median={s['p50']:>8,.0f}  clip={s['clip_pct']:5.1f}%")


if __name__ == "__main__":
    main()
