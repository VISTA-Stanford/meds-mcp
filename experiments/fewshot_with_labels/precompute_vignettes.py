#!/usr/bin/env python3
"""
Precompute per-(patient, embed_time) vignettes for the fewshot_with_labels
experiment.

Reads outputs/patients.jsonl (produced by build_cohort.py) where each row is
one PatientState keyed by ``(person_id, embed_time)``. For every state with
an empty ``vignette`` field, this script:

  1) Linearizes that patient's XML timeline up to the state's embed_time,
     keeping the last N encounters (deterministic).
  2) LLM-summarizes the linearized text into a short vignette.

Dedup is automatic: only one LLM call per distinct ``(person_id, embed_time)``
even when many task items share that embed_time.

Writes the updated patients.jsonl atomically after every success (resumable).
Use --force to regenerate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import CohortStore, PatientSimilarityPipeline
from experiments.fewshot_with_labels import _paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute vignettes for fewshot_with_labels cohort"
    )
    parser.add_argument(
        "--patients",
        type=Path,
        default=_paths.patients_jsonl(),
        help="Override via env var VISTA_OUTPUTS_DIR.",
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=_paths.items_jsonl(),
        help="Override via env var VISTA_OUTPUTS_DIR.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_paths.corpus_dir(),
        help="Override via env var VISTA_CORPUS_DIR.",
    )
    parser.add_argument(
        "--n-encounters",
        type=int,
        default=0,
        help="Keep only the last N encounters before embed_time. 0 = keep ALL encounters on/before embed_time (default).",
    )
    parser.add_argument("--model", type=str, default="apim:gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=None, help="Max patients to process this run")
    parser.add_argument("--force", action="store_true", help="Regenerate even if vignette exists")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    store = CohortStore.load(args.patients, args.items)

    pipeline = PatientSimilarityPipeline(
        xml_dir=str(args.corpus_dir),
        model=args.model,
        n_encounters=args.n_encounters,
        generation_overrides={"temperature": 0.2, "max_tokens": 1024},
    )
    summarizer = pipeline.summarizer
    base_generator = pipeline.base_generator
    assert summarizer is not None

    todo = [
        p
        for p in store.patient_states()
        if (args.force or not p.vignette.strip()) and p.embed_time
    ]

    # Only process states whose XML exists.
    missing_xml = [p for p in todo if not (args.corpus_dir / f"{p.person_id}.xml").exists()]
    todo = [p for p in todo if (args.corpus_dir / f"{p.person_id}.xml").exists()]
    if missing_xml:
        logger.warning(
            "%d (person_id, embed_time) entries skipped: XML missing under %s",
            len(missing_xml),
            args.corpus_dir,
        )

    if args.limit is not None:
        todo = todo[: args.limit]

    logger.info("Vignettes to generate this run: %d (distinct (pid, embed_time))", len(todo))

    use_tqdm = tqdm is not None and not args.no_progress
    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=len(todo),
            desc="Precomputing vignettes (fewshot_with_labels)",
            unit="patient",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    n_ok = 0
    n_skip = 0
    skip_reasons = {"timeline_fail": 0, "empty_timeline": 0, "llm_fail": 0}

    def _bump_skip(reason: str) -> None:
        nonlocal n_skip
        n_skip += 1
        skip_reasons[reason] += 1

    for p in todo:
        try:
            text = base_generator.generate(
                patient_id=p.person_id,
                cutoff_date=p.embed_time,
                n_encounters=args.n_encounters,
            )
        except Exception as e:
            logger.warning("Timeline extract failed %s@%s: %s", p.person_id, p.embed_time, e)
            _bump_skip("timeline_fail")
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
            continue

        if not text.strip():
            logger.warning("Empty timeline for %s@%s", p.person_id, p.embed_time)
            _bump_skip("empty_timeline")
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
            continue

        try:
            vignette = summarizer.summarize(text)
        except Exception as e:
            logger.error("Vignette LLM failed %s@%s: %s", p.person_id, p.embed_time, e)
            _bump_skip("llm_fail")
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
            continue

        store.update_patient(replace(p, vignette=vignette))
        n_ok += 1

        # Persist after every success to make the script resumable on crash.
        store.save(args.patients, args.items)

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(f"ok={n_ok} skip={n_skip} last={p.person_id}@{p.embed_time}")
        elif n_ok % 25 == 0:
            logger.info("Wrote %d vignettes...", n_ok)

    if pbar:
        pbar.close()

    total_with_vignette = sum(1 for p in store.patient_states() if p.vignette.strip())
    logger.info(
        "Done. ok=%d skip=%d (timeline_fail=%d, empty_timeline=%d, llm_fail=%d) "
        "total_states_with_vignette=%d/%d",
        n_ok,
        n_skip,
        skip_reasons["timeline_fail"],
        skip_reasons["empty_timeline"],
        skip_reasons["llm_fail"],
        total_with_vignette,
        len(store.patient_states()),
    )


if __name__ == "__main__":
    main()
