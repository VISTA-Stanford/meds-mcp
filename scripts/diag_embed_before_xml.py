#!/usr/bin/env python3
"""
Diagnostic: for each patient in the cohort CSV, find the earliest event
timestamp in their XML timeline and report how many patients have an
embed_time that predates their first recorded event.

A patient in that bucket has nothing to summarize under a "history up to
landmark" policy — precompute_vignettes.py would emit an empty vignette
and skip them.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree

_REPO_ROOT = Path(__file__).resolve().parents[1]

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def earliest_event_date(xml_path: Path) -> str | None:
    """Return the smallest YYYY-MM-DD string found in any <entry timestamp=...>,
    or None if the XML has no parseable entries."""
    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:
        return None
    dates: list[str] = []
    for entry in root.iter("entry"):
        ts = entry.attrib.get("timestamp", "")
        m = DATE_RE.match(ts)
        if m:
            dates.append(m.group(1))
    return min(dates) if dates else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=_REPO_ROOT
        / "data/collections/vista_bench/bikia_dev-lumia_cohort_progression_tasks-000000000000.csv",
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=_REPO_ROOT / "data/collections/vista_bench/thoracic_cohort_lumia",
    )
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional CSV output with per-patient details.")
    args = p.parse_args()

    # (pid, embed_time, split) -> only keep first row per pid (embed_time is
    # the same across a patient's rows in this cohort).
    pid_info: dict[str, tuple[str, str]] = {}  # pid -> (embed_time, split)
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = (row.get("person_id") or "").strip()
            et = (row.get("embed_time") or "").strip()
            split = (row.get("split") or "").strip()
            if pid and pid not in pid_info:
                pid_info[pid] = (et, split)

    # Categorize.
    cats: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    rows_out: list[tuple[str, str, str, str | None, str]] = []

    total = len(pid_info)
    for i, (pid, (et, split)) in enumerate(pid_info.items(), 1):
        xml_path = args.corpus_dir / f"{pid}.xml"

        cat: str
        first: str | None = None
        if not et:
            cat = "no_embed_time"
        elif not xml_path.exists():
            cat = "xml_missing"
        else:
            first = earliest_event_date(xml_path)
            if first is None:
                cat = "xml_no_parseable_dates"
            elif first > et:
                cat = "embed_before_first_event"
            elif first == et:
                cat = "embed_eq_first_event"
            else:
                cat = "embed_after_first_event"

        cats[cat] += 1
        by_split[split][cat] += 1
        rows_out.append((pid, split, et, first, cat))

        if i % 500 == 0:
            print(f"  scanned {i}/{total}...", file=sys.stderr)

    # Report.
    print(f"\nTotal unique patients in CSV: {total}\n")
    print("Category breakdown:")
    for c, n in cats.most_common():
        pct = 100 * n / total if total else 0
        print(f"  {c:30s} {n:5d}  ({pct:.1f}%)")

    print("\nBy split:")
    for split in sorted(by_split):
        sp_total = sum(by_split[split].values())
        print(f"  split={split!r:10s} (n={sp_total})")
        for c, n in by_split[split].most_common():
            pct = 100 * n / sp_total if sp_total else 0
            print(f"    {c:30s} {n:5d}  ({pct:.1f}%)")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["person_id", "split", "embed_time", "first_event_date", "category"])
            for r in rows_out:
                w.writerow(r)
        print(f"\nPer-patient details written to {args.out_csv}")


if __name__ == "__main__":
    main()
