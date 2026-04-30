#!/usr/bin/env python3
"""
Analyze context length distributions for vista_bench.

Counts:
  1. All history prior to prediction_time: distribution of events per (patient_id, prediction_time)
  2. Max k tokens: how many events fit into a token budget (e.g. 2096)
  3. Single visit (last visit): distribution of events per (patient_id, prediction_time)

Output is saved in JSON format suitable for plotting (e.g. with pandas, matplotlib, plotly).

Usage:
  uv run python scripts/analyze_context_lengths.py
  uv run python scripts/analyze_context_lengths.py --config configs/vista.yaml --token-budget 4096 --limit 100
  uv run python scripts/analyze_context_lengths.py --output results/context_analysis.json
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_timestamp(ts_str):
    if not ts_str or not str(ts_str).strip():
        return None
    from datetime import datetime

    s = str(ts_str).replace("Z", "+00:00").split(".")[0]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        except ValueError:
            return None


def estimate_tokens_per_event() -> float:
    """Rough tokens per event in minimal format: 'timestamp | code | name' (~50-80 chars → ~15-25 tokens)."""
    return 20.0  # conservative


def max_events_for_token_budget(budget: int, tokens_per_event: float = None) -> int:
    """How many events fit in budget (assuming events only; no prompt overhead)."""
    if tokens_per_event is None:
        tokens_per_event = estimate_tokens_per_event()
    return int(budget / tokens_per_event)


def get_all_history_events(pid: str, prediction_time_str: str, data_dir: str) -> list:
    """All events with timestamp <= prediction_time (all past, not single visit)."""
    from lxml import etree

    xml_path = Path(data_dir) / f"{pid}.xml"
    if not xml_path.exists():
        return []

    cutoff = parse_timestamp(prediction_time_str)
    if cutoff is None:
        return []

    root = etree.parse(str(xml_path)).getroot()
    events = []

    for encounter in root.findall("encounter"):
        events_elem = encounter.find("events")
        if events_elem is None:
            continue
        for entry in events_elem.findall("entry"):
            ts_str = entry.get("timestamp")
            ts = parse_timestamp(ts_str)
            if ts is None or ts > cutoff:
                continue
            for event in entry.findall("event"):
                events.append(
                    {"timestamp": ts_str, "code": event.get("code"), "name": event.get("name")}
                )

    return events


def get_single_visit_events(pid: str, prediction_time_str: str, data_dir: str) -> list:
    """Events from the single encounter (visit) containing prediction_time."""
    from meds_mcp.server.rag.visit_filter import get_events_for_single_visit_from_xml

    return get_events_for_single_visit_from_xml(pid, prediction_time_str, data_dir)


def get_formatted_context_tokens(pid: str, pt: str, task_name: str, events: list) -> int:
    """Full formatted context token count (delta-encoded, collapse-by-day for lab tasks)."""
    from meds_mcp.server.rag.context_formatter import format_patient_context, _count_tokens

    if not events:
        return 0
    text = format_patient_context(
        events,
        patient_id=pid,
        prediction_time=pt,
        task_name=task_name,
        max_tokens=0,
        include_event_key=False,
    )
    return _count_tokens(text) if text else 0


def collect_rows_from_task_csvs(tasks: list, limit_per_task: int = None) -> list:
    """(patient_id, prediction_time) for all rows."""
    from meds_mcp.experiments.task_config import ALL_TASKS, get_csv_path_for_task

    rows = []
    seen = set()
    for task_name in tasks or ALL_TASKS:
        csv_path = get_csv_path_for_task(task_name)
        if not csv_path.exists():
            continue
        count = 0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if limit_per_task and count >= limit_per_task:
                    break
                pid = row.get("patient_id", "").strip()
                pt = row.get("prediction_time", "").strip()
                if not pid or not pt or (pid, pt) in seen:
                    continue
                seen.add((pid, pt))
                rows.append((pid, pt, task_name))
                count += 1
    return rows


def compute_distribution(counts: list) -> dict:
    """Summary stats for a list of counts."""
    if not counts:
        return {}
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    return {
        "n": n,
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / n, 2),
        "median": sorted_counts[n // 2],
        "p25": sorted_counts[int(0.25 * n)] if n >= 4 else min(counts),
        "p75": sorted_counts[int(0.75 * n)] if n >= 4 else max(counts),
        "p90": sorted_counts[int(0.90 * n)] if n >= 10 else max(counts),
    }


def compute_histogram(counts: list, bins: list) -> dict:
    """Histogram: bin label -> count of values in that bin."""
    hist = defaultdict(int)
    for c in counts:
        for i, upper in enumerate(bins[1:], start=1):
            if c <= upper:
                label = f"({bins[i-1]}, {upper}]"
                hist[label] += 1
                break
        else:
            hist[f"> {bins[-1]}"] = hist.get(f"> {bins[-1]}", 0) + 1
    return dict(hist)


def main():
    parser = argparse.ArgumentParser(description="Analyze context length distributions")
    parser.add_argument("--config", default="configs/vista.yaml", help="Config for data_dir")
    parser.add_argument("--task", default=None, help="Single task; default all")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per task")
    parser.add_argument(
        "--token-budget",
        type=int,
        default=2096,
        help="Token budget for analysis (max events that fit)",
    )
    parser.add_argument(
        "--tokens-per-event",
        type=float,
        default=20.0,
        help="Rough tokens per event for conversion",
    )
    parser.add_argument(
        "--output",
        default="results/context_length_analysis.json",
        help="Output path for JSON (plot-ready)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config.get("data", {}).get("corpus_dir") or os.getenv(
        "DATA_DIR", "data/collections/vista_bench/thoracic_cohort_lumia"
    )
    if not Path(data_dir).exists():
        print(f"Data dir not found: {data_dir}")
        sys.exit(1)

    from meds_mcp.experiments.task_config import ALL_TASKS

    tasks = [args.task] if args.task else ALL_TASKS
    rows = collect_rows_from_task_csvs(tasks, args.limit)

    print(f"Analyzing {len(rows)} (patient_id, prediction_time) rows from data_dir={data_dir}\n")

    # Common histogram bins for events
    bins = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]

    all_history_counts = []
    single_visit_counts = []
    row_data = []  # Per-row data for plotting (patient_id, prediction_time, task, all_history, single_visit)

    for pid, pt, task_name in rows:
        evs_all = get_all_history_events(pid, pt, data_dir)
        evs_single = get_single_visit_events(pid, pt, data_dir)
        n_all = len(evs_all)
        n_single = len(evs_single)
        formatted_tokens = get_formatted_context_tokens(pid, pt, task_name, evs_single)
        all_history_counts.append(n_all)
        single_visit_counts.append(n_single)
        row_data.append(
            {
                "patient_id": pid,
                "prediction_time": pt,
                "task_name": task_name,
                "all_history_event_count": n_all,
                "single_visit_event_count": n_single,
                "formatted_context_tokens": formatted_tokens,
            }
        )

    k = args.token_budget
    tpe = args.tokens_per_event
    max_ev = max_events_for_token_budget(k, tpe)

    # Full formatted context token count per row (delta-encoded, collapse-by-day for lab tasks)
    formatted_context_tokens = [r["formatted_context_tokens"] for r in row_data]
    for r in row_data:
        r["all_history_tokens"] = int(r["all_history_event_count"] * tpe)
    n_total = len(row_data)
    n_4096 = sum(1 for t in formatted_context_tokens if t <= 4096)
    n_32k = sum(1 for t in formatted_context_tokens if t <= 32768)
    context_coverage = {
        "total_patients": n_total,
        "within_4096": n_4096,
        "within_32k": n_32k,
        "pct_within_4096": round(100.0 * n_4096 / n_total, 2) if n_total else 0,
        "pct_within_32k": round(100.0 * n_32k / n_total, 2) if n_total else 0,
    }

    # Capped distributions (what you'd actually send)
    all_capped = [min(c, max_ev) for c in all_history_counts]
    single_capped = [min(c, max_ev) for c in single_visit_counts]

    # Build output suitable for plotting
    output = {
        "metadata": {
            "data_dir": data_dir,
            "n_rows": len(rows),
            "token_budget": k,
            "tokens_per_event": tpe,
            "max_events_for_budget": max_ev,
            "tasks_included": tasks if args.task else "all",
        },
        "context_coverage": context_coverage,
        "all_history_prior_to_t": {
            "summary": compute_distribution(all_history_counts),
            "histogram": compute_histogram(all_history_counts, bins),
            "raw_counts": all_history_counts,
        },
        "all_history_capped_at_k_tokens": {
            "summary": compute_distribution(all_capped),
            "histogram": compute_histogram(all_capped, bins),
            "raw_counts": all_capped,
        },
        "single_visit": {
            "summary": compute_distribution(single_visit_counts),
            "histogram": compute_histogram(single_visit_counts, bins),
            "raw_counts": single_visit_counts,
        },
        "single_visit_capped_at_k_tokens": {
            "summary": compute_distribution(single_capped),
            "histogram": compute_histogram(single_capped, bins),
            "raw_counts": single_capped,
        },
        "per_row": row_data,
    }

    # Print summary to console
    print("1. ALL HISTORY (prior to prediction_time)")
    for kk, vv in output["all_history_prior_to_t"]["summary"].items():
        print(f"   {kk}: {vv}")

    print(f"\n2. TOKEN BUDGET k={k} (tokens_per_event≈{tpe})")
    print(f"   Max events that fit: {max_ev}")
    print("   All history capped:")
    for kk, vv in output["all_history_capped_at_k_tokens"]["summary"].items():
        print(f"     {kk}: {vv}")

    print("\n3. SINGLE VISIT")
    for kk, vv in output["single_visit"]["summary"].items():
        print(f"   {kk}: {vv}")
    print("   Single visit capped:")
    for kk, vv in output["single_visit_capped_at_k_tokens"]["summary"].items():
        print(f"     {kk}: {vv}")

    print("\n4. CONTEXT COVERAGE (formatted single-visit context, delta-encoded, collapse-by-day for lab tasks)")
    cov = output["context_coverage"]
    print(f"   Within 4096 tokens: {cov['within_4096']} / {cov['total_patients']} ({cov['pct_within_4096']}%)")
    print(f"   Within 32K tokens:  {cov['within_32k']} / {cov['total_patients']} ({cov['pct_within_32k']}%)")

    # Save (include formatted token counts for reproducibility)
    output["formatted_context_tokens"] = formatted_context_tokens
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Plot: histogram of context length distribution (x = context length, y = count of patients)
    try:
        import matplotlib.pyplot as plt
        tokens = [t for t in formatted_context_tokens if t >= 0]
        if tokens:
            fig, ax = plt.subplots(figsize=(9, 5))
            bins = min(50, max(20, len(tokens) // 10))
            ax.hist(tokens, bins=bins, color="steelblue", alpha=0.8, edgecolor="white")
            ax.set_xlabel("Full context length (tokens)")
            ax.set_ylabel("Number of patients")
            ax.set_title("Context length distribution (single-visit, delta-encoded, collapse-by-day for lab tasks)")
            plt.tight_layout()
            plot_path = out_path.parent / "context_coverage.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   Plot saved to {plot_path}")
    except ImportError:
        pass

    print(f"\nResults saved to {out_path} (plot-ready: raw_counts, histogram, per_row)")
    return output


if __name__ == "__main__":
    main()
