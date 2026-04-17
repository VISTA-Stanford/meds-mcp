#!/usr/bin/env python3
"""
Patient similarity workflow demo.

Steps:
1. Build a BM25 index over vignettes generated from a patient cohort.
2. Generate a vignette for the query patient.
3. Search the index for the top-k most similar patients (excluding the query).

Cohort and landmark-date sources:
- Without ``--records-csv``: every XML under ``--corpus-dir`` is indexed with
  no landmark filter (full timeline), regardless of ``--cutoff-mode``.
- With ``--records-csv``: the CSV's ``person_id`` column defines the cohort,
  and ``--cutoff-mode`` picks how the landmark column is applied:
    * ``per-patient`` (default): each patient is chopped at its own CSV date.
    * ``query-anchored``: every patient is chopped at the *query* patient's date.
    * ``none``: CSV dates are ignored; full timelines are used.

``--n-encounters`` is orthogonal: if set, caps every patient's vignette input
to the last N qualifying encounters (after any landmark chop).

Requires VAULT_SECRET_KEY for LLM access.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.similarity import (
    PatientBM25Index,
    PatientRecord,
    PatientSimilarityPipeline,
    load_patient_records_from_csv,
)

try:
    from securellm.exceptions import RateLimitError
except ImportError:
    class RateLimitError(Exception):
        pass


def load_corpus_dir(config_file: Optional[str]) -> str:
    """Resolve corpus directory from config YAML or environment."""
    config_path = Path(config_file) if config_file else Path("configs/medalign.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("data", {}).get("corpus_dir", "data/collections/dev-corpus")
    if config_file:
        print(f"Config file not found: {config_file}, falling back to environment")
    return os.getenv("DATA_DIR", "data/collections/dev-corpus")


def _generate_vignette_with_retry(
    pipeline: PatientSimilarityPipeline,
    patient_id: str,
    *,
    cutoff_date: Optional[str],
    n_encounters: Optional[int],
    max_retries: int = 3,
    backoff_seconds: int = 65,
) -> str:
    """Call pipeline.generate_vignette with a simple rate-limit retry."""
    attempt = 0
    while True:
        try:
            return pipeline.generate_vignette(
                patient_id,
                cutoff_date=cutoff_date,
                n_encounters=n_encounters,
            )
        except RateLimitError:
            attempt += 1
            if attempt >= max_retries:
                raise
            tqdm.write(
                f"Rate limit for {patient_id}; waiting {backoff_seconds}s "
                f"(retry {attempt}/{max_retries})"
            )
            time.sleep(backoff_seconds)


def build_global_index(
    pipeline: PatientSimilarityPipeline,
    records: List[PatientRecord],
    n_encounters_default: Optional[int],
    corpus_path: Path,
    debug_output_dir: Optional[Path] = None,
) -> tuple[PatientBM25Index, dict[str, str]]:
    """Generate a vignette for every record and build a BM25 index.

    Each ``PatientRecord``'s ``cutoff_date`` (may be None) is applied per patient.
    ``n_encounters_default`` is used when a record does not override it.
    Records whose XML is missing are skipped with a warning.
    """
    print(f"\nBuilding BM25 index over {len(records)} patient record(s)...")
    doc_records: list[dict[str, str]] = []
    all_vignettes: dict[str, str] = {}
    failed = 0

    for rec in tqdm(records, desc="Generating vignettes"):
        pid = rec.person_id
        if not (corpus_path / f"{pid}.xml").exists():
            failed += 1
            tqdm.write(f"  Missing XML for {pid}, skipping")
            continue

        n_enc = rec.n_encounters if rec.n_encounters is not None else n_encounters_default
        try:
            vignette = _generate_vignette_with_retry(
                pipeline,
                pid,
                cutoff_date=rec.cutoff_date,
                n_encounters=n_enc,
            )
        except Exception as e:
            failed += 1
            tqdm.write(f"  Failed {pid}: {e}")
            continue

        if not vignette or not vignette.strip():
            failed += 1
            continue

        all_vignettes[pid] = vignette
        doc_records.append({"person_id": pid, "vignette": vignette})

    if not doc_records:
        raise ValueError(f"No vignettes generated (failed: {failed})")

    print(f"Generated vignettes for {len(doc_records)} patients ({failed} failed)\n")

    if debug_output_dir is not None:
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        out = debug_output_dir / "all_vignettes.txt"
        with open(out, "w") as f:
            for pid, vignette in sorted(all_vignettes.items()):
                f.write(f"\n{'=' * 80}\nPatient ID: {pid}\n{'=' * 80}\n{vignette}\n")
        print(f"Saved all vignettes to {out}")

    print("Building BM25 index...")
    index = PatientBM25Index.from_vignettes(doc_records)
    print(f"BM25 index built with {index.size} patient vignettes\n")
    return index, all_vignettes


def _build_records(
    args: argparse.Namespace,
    corpus_path: Path,
) -> tuple[List[PatientRecord], Optional[str]]:
    """Build the record list and the query patient's cutoff based on CLI flags.

    Returns (records, query_cutoff). ``query_cutoff`` is ``None`` when no
    landmark should apply to the query.
    """
    if args.records_csv:
        csv_records = load_patient_records_from_csv(
            args.records_csv, cutoff_col=args.cutoff_col
        )
        ids = {r.person_id for r in csv_records}
        if args.patient_id not in ids:
            raise ValueError(
                f"--patient-id {args.patient_id} not present in {args.records_csv}"
            )
        query_rec = next(r for r in csv_records if r.person_id == args.patient_id)

        if args.cutoff_mode == "per-patient":
            return csv_records, query_rec.cutoff_date
        if args.cutoff_mode == "query-anchored":
            anchored = [
                PatientRecord(person_id=r.person_id, cutoff_date=query_rec.cutoff_date)
                for r in csv_records
            ]
            return anchored, query_rec.cutoff_date
        # cutoff_mode == "none"
        stripped = [PatientRecord(person_id=r.person_id) for r in csv_records]
        return stripped, None

    # No CSV: scan the corpus, no landmark.
    xml_files = sorted(corpus_path.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files in {corpus_path}")
    records = [PatientRecord(person_id=p.stem) for p in xml_files]
    if args.patient_id not in {r.person_id for r in records}:
        raise ValueError(
            f"--patient-id {args.patient_id} has no XML in {corpus_path}"
        )
    return records, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient similarity workflow demo: vignette index → search"
    )
    parser.add_argument("--patient-id", type=str, required=True, help="Query patient ID")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (defaults to configs/medalign.yaml)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Corpus directory (overrides config)",
    )
    parser.add_argument(
        "--records-csv",
        type=str,
        default=None,
        help="CSV with person_id and landmark-date columns. Defines the cohort "
        "and drives --cutoff-mode. If omitted, all XMLs under --corpus-dir are "
        "indexed with no landmark.",
    )
    parser.add_argument(
        "--cutoff-mode",
        choices=["per-patient", "query-anchored", "none"],
        default="per-patient",
        help="Applies only when --records-csv is provided. per-patient: each "
        "patient chopped at their own CSV date. query-anchored: every patient "
        "chopped at the query patient's CSV date. none: ignore CSV dates.",
    )
    parser.add_argument(
        "--cutoff-col",
        type=str,
        default="embed_time",
        help="Landmark-date column in --records-csv (default: embed_time).",
    )
    parser.add_argument(
        "--n-encounters",
        type=int,
        default=None,
        help="Cap vignette input to the last N encounters. Omit for the full timeline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the cohort to the first N patient records (the query patient is "
        "always kept). Useful for smoke tests.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k similar patients")
    parser.add_argument("--llm-model", type=str, default="apim:gpt-4.1-mini")
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="data/vignette_debug",
        help="Directory to save vignettes/results for debugging",
    )
    args = parser.parse_args()

    if not os.getenv("VAULT_SECRET_KEY"):
        print("VAULT_SECRET_KEY required for LLM vignette generation")
        return

    corpus_dir = args.corpus_dir or load_corpus_dir(args.config)
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_dir}")
        return

    try:
        records, query_cutoff = _build_records(args, corpus_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.limit is not None and args.limit > 0 and len(records) > args.limit:
        query_rec = next((r for r in records if r.person_id == args.patient_id), None)
        head = records[: args.limit]
        if query_rec is not None and query_rec not in head:
            head = head[: args.limit - 1] + [query_rec]
        records = head

    effective_cutoff_mode = args.cutoff_mode if args.records_csv else "none (no CSV)"
    enc_mode = (
        "full timeline" if args.n_encounters is None else f"last {args.n_encounters} encounters"
    )

    print("\n" + "=" * 80)
    print("PATIENT SIMILARITY WORKFLOW DEMO")
    print("=" * 80)
    if args.config:
        print(f"Config: {args.config}")
    print(f"Corpus: {corpus_dir}")
    print(f"Query patient: {args.patient_id}")
    print(f"Cohort size: {len(records)}")
    print(f"Cutoff mode: {effective_cutoff_mode}")
    print(f"Query cutoff date: {query_cutoff if query_cutoff else '(none)'}")
    print(f"Encounter mode: {enc_mode}")
    print(f"LLM model: {args.llm_model}")
    print(f"Top-k results: {args.top_k}")
    print(f"Debug output dir: {args.debug_dir}")
    print("=" * 80)

    debug_dir = Path(args.debug_dir)

    try:
        pipeline = PatientSimilarityPipeline(
            xml_dir=str(corpus_path),
            model=args.llm_model,
            n_encounters=args.n_encounters,
            generation_overrides={"max_tokens": 1024, "temperature": 0.2},
        )
        print(f"\nPipeline initialized with {args.llm_model}")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    try:
        index, _all_vignettes = build_global_index(
            pipeline,
            records,
            n_encounters_default=args.n_encounters,
            corpus_path=corpus_path,
            debug_output_dir=debug_dir,
        )
    except Exception as e:
        import traceback
        print(f"Failed to build global index: {e}")
        traceback.print_exc()
        return

    print(f"\n{'-' * 80}")
    print(f"STEP 1: Generate Query Vignette ({args.patient_id})")
    print(f"{'-' * 80}")

    xml_path = corpus_path / f"{args.patient_id}.xml"
    if not xml_path.exists():
        print(f"Query patient XML not found: {xml_path}")
        return

    try:
        query_vignette = _generate_vignette_with_retry(
            pipeline,
            args.patient_id,
            cutoff_date=query_cutoff,
            n_encounters=args.n_encounters,
        )
        print(f"Vignette length: {len(query_vignette)} chars")
        print("Preview (first 300 chars):")
        print(query_vignette[:300] + ("..." if len(query_vignette) > 300 else ""))

        debug_dir.mkdir(parents=True, exist_ok=True)
        q_path = debug_dir / f"query_vignette_{args.patient_id}.txt"
        with open(q_path, "w") as f:
            f.write(f"Query Patient ID: {args.patient_id}\n")
            f.write(f"Cutoff date: {query_cutoff if query_cutoff else '(none)'}\n")
            f.write(f"Encounter cap: {args.n_encounters if args.n_encounters else '(none)'}\n")
            f.write(f"{'=' * 80}\n\n{query_vignette}\n")
        print(f"Saved query vignette to {q_path}")
    except Exception as e:
        import traceback
        print(f"Error processing query patient: {e}")
        traceback.print_exc()
        return

    print(f"\n{'-' * 80}")
    print("STEP 2: Search Global Vignette Index")
    print(f"{'-' * 80}")
    print(f"Searching for {args.top_k} most similar patients (excluding {args.patient_id})...")
    results = index.search(
        query_vignette=query_vignette,
        top_k=args.top_k,
        exclude_person_id=args.patient_id,
    )

    print(f"\n{'-' * 80}")
    print(f"STEP 3: Results (Top-{args.top_k} Similar Patients)")
    print(f"{'-' * 80}\n")

    if not results:
        print("No results found")
    else:
        r_path = debug_dir / f"search_results_{args.patient_id}.txt"
        with open(r_path, "w") as f:
            f.write(f"Query Patient: {args.patient_id}\n")
            f.write(f"Cutoff mode: {effective_cutoff_mode}\n")
            f.write(f"Query cutoff: {query_cutoff if query_cutoff else '(none)'}\n")
            f.write(f"Number of Results: {len(results)}\n")
            f.write("=" * 80 + "\n\n")
            for i, r in enumerate(results, 1):
                f.write(f"{i}. Patient {r.person_id}\n")
                f.write(f"   BM25 Score: {r.score:.4f}\n")
                f.write("   Vignette:\n")
                f.write(f"   {r.vignette}\n\n{'-' * 80}\n\n")
        print(f"Saved search results to {r_path}\n")

        for i, r in enumerate(results, 1):
            print(f"{i}. Patient {r.person_id}")
            print(f"   BM25 Score: {r.score:.4f}")
            print("   Vignette:")
            print(f"   {r.vignette}")
            print()

    print("=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
