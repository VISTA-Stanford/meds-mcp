#!/usr/bin/env python3
"""
Precompute per-patient vignettes: chop Lumia XML at each patient's embed_time,
take last N encounters, LLM-summarize to vignette. Writes vignettes.jsonl for BM25 indexing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Package lives under src/; repo root is needed for experiments.* imports.
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import (
    PatientSimilarityPipeline,
    load_patient_records_from_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


def load_completed_person_ids(jsonl_path: Path) -> set[str]:
    """person_id values already written (one JSON object per line). Skips bad lines."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    bad_lines = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = str(obj.get("person_id", "")).strip()
                if pid:
                    done.add(pid)
            except json.JSONDecodeError:
                bad_lines += 1
                logger.warning("Skipping invalid JSON line %s in %s", line_no, jsonl_path)
    if bad_lines:
        logger.warning("Ignored %d malformed line(s) in %s", bad_lines, jsonl_path)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute vignettes for progression subset corpus")
    parser.add_argument(
        "--csv",
        type=Path,
        default=_REPO_ROOT / "data/collections/vista_bench/progression_subset.csv",
        help="CSV with person_id and embed_time",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_REPO_ROOT / "data/collections/vista_bench/thoracic_cohort_lumia",
        help="Directory with {person_id}.xml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "experiments/progression_subset/outputs",
        help="Output directory for vignettes.jsonl",
    )
    parser.add_argument("--n-encounters", type=int, default=2, help="Last N encounters after chop")
    parser.add_argument("--model", type=str, default="apim:gpt-4.1-mini", help="LLM for vignette")
    parser.add_argument("--limit", type=int, default=None, help="Max patients to process (debug)")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (logging only)",
    )
    parser.add_argument(
        "--no-require-complete-rows",
        action="store_true",
        help="Disable CSV filter: include all patients present in CSV even if some rows have blank columns",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing vignettes.jsonl: skip person_ids already present, append new rows",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_patient_records_from_csv(
        args.csv,
        require_all_columns_populated=not args.no_require_complete_rows,
    )
    embed_times: dict[str, str] = {
        r.person_id: r.cutoff_date for r in records if r.cutoff_date
    }
    logger.info(
        "CSV loader: patients_kept=%d (blank embed_time excluded=%d)",
        len(embed_times),
        len(records) - len(embed_times),
    )
    pipeline = PatientSimilarityPipeline(
        xml_dir=str(args.corpus_dir),
        model=args.model,
        n_encounters=args.n_encounters,
        generation_overrides={"temperature": 0.2, "max_tokens": 1024},
    )
    summarizer = pipeline.summarizer
    base_generator = pipeline.base_generator
    assert summarizer is not None  # use_llm_vignettes defaults to True

    out_path = args.output_dir / "vignettes.jsonl"
    xml_all = sorted(args.corpus_dir.glob("*.xml"))
    candidates = [p for p in xml_all if p.stem in embed_times]
    n_skip_not_in_csv = len(xml_all) - len(candidates)

    done_before: set[str] = set()
    file_mode = "w"
    if args.resume:
        done_before = load_completed_person_ids(out_path)
        file_mode = "a" if out_path.exists() and out_path.stat().st_size > 0 else "w"
        if done_before:
            logger.info(
                "Resume: found %d person_id(s) already in %s; will append new rows",
                len(done_before),
                out_path,
            )
        elif out_path.exists():
            logger.info("Resume: existing file empty or no valid rows; opening for write")
    else:
        if out_path.exists():
            logger.info("Overwriting %s (use --resume to continue instead)", out_path)

    work_queue = [p for p in candidates if p.stem not in done_before]

    n_ok = 0
    n_skip = 0

    use_tqdm = tqdm is not None and not args.no_progress
    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=len(work_queue),
            desc="Precomputing vignettes",
            unit="patient",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    with open(out_path, file_mode, encoding="utf-8") as out_f:
        for xml_path in work_queue:
            if args.limit is not None and n_ok >= args.limit:
                break
            pid = xml_path.stem
            et = embed_times[pid]
            try:
                text = base_generator.generate(
                    patient_id=pid,
                    cutoff_date=et,
                    n_encounters=args.n_encounters,
                )
            except Exception as e:
                logger.warning("Timeline extract failed %s: %s", pid, e)
                n_skip += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
                continue
            if not text.strip():
                logger.warning("Empty timeline for %s", pid)
                n_skip += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
                continue
            try:
                vignette = summarizer.summarize(text)
            except Exception as e:
                logger.error("Vignette LLM failed %s: %s", pid, e)
                n_skip += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
                continue
            rec = {
                "person_id": pid,
                "embed_time": et,
                "n_encounters": args.n_encounters,
                "vignette": vignette,
                "last_n_encounters_chars": len(text),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={n_ok} skip={n_skip} last={pid}")
            elif not use_tqdm and n_ok > 0 and n_ok % 25 == 0:
                logger.info("Wrote %d vignettes...", n_ok)

        if pbar:
            pbar.close()

    n_total_in_file = len(done_before) + n_ok
    meta = {
        "vignettes_path": str(out_path),
        "resume": args.resume,
        "n_written_this_run": n_ok,
        "n_skipped_this_run": n_skip,
        "n_already_complete_before_run": len(done_before),
        "n_total_person_ids_in_file_after_run": n_total_in_file,
        "n_xml_not_in_csv": n_skip_not_in_csv,
        "n_candidates_eligible": len(candidates),
        "n_remaining_after_resume_filter": len(work_queue),
        "n_encounters": args.n_encounters,
        "model": args.model,
        "corpus_dir": str(args.corpus_dir),
        "require_complete_rows": not args.no_require_complete_rows,
        "n_csv_rows_loaded": len(records),
        "n_patients_with_embed_time": len(embed_times),
    }
    with open(args.output_dir / "vignettes_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Done. This run wrote {n_ok} new line(s) to {out_path} "
        f"(skipped {n_skip} this run; {n_skip_not_in_csv} corpus XML not in CSV). "
        f"Total unique person_ids in file ≈ {n_total_in_file}. "
        f"Meta: vignettes_meta.json"
    )
    if tqdm is None and not args.no_progress:
        print("Install tqdm for a progress bar: pip install tqdm")


if __name__ == "__main__":
    main()
