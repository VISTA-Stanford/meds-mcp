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
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import CohortStore, PatientSimilarityPipeline
from experiments.fewshot_with_labels import _paths
from experiments.fewshot_with_labels._tokens import (
    count_tokens,
    effective_input_budget,
    truncate_to_tokens,
)

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
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=65536,
        help=(
            "Cap the linearized timeline at this many tokens before sending it "
            "to the summarizer. Oversized timelines are head+tail truncated with "
            "an explicit [truncated] marker. 0 disables truncation. Default 65536 "
            "(covers the full trajectory for >95%% of typical oncology cohorts "
            "while leaving comfortable headroom in the model context window). "
            "Must not exceed the model's effective input budget — see "
            "--model-context-tokens / --max-output-tokens / --token-safety-margin."
        ),
    )
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        default=120000,
        help=(
            "The model's advertised context window, in tokens. Used to derive "
            "the safe effective input budget and to validate --max-input-tokens. "
            "Default 120000 (observed-safe value for apim:gpt-4.1-mini; nominal "
            "is 128K but APIM deployments commonly reserve a few K for wrapper "
            "overhead)."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help=(
            "Max output tokens the summarizer may generate. Must match "
            "generation_overrides.max_tokens in the SecureLLMSummarizer "
            "configuration (also defaults to 1024). Subtracted from the model "
            "context to compute the effective input budget."
        ),
    )
    parser.add_argument(
        "--token-safety-margin",
        type=int,
        default=2048,
        help=(
            "Extra buffer reserved when computing the effective input budget, "
            "on top of --max-output-tokens. Accounts for system-prompt + "
            "tokenizer drift (cl100k_base vs o200k_base). Default 2048."
        ),
    )
    parser.add_argument("--model", type=str, default="apim:gpt-4.1-mini")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help=(
            "Retries per patient if the LLM summarization call raises "
            "(transient APIM errors, rate limits, empty responses). "
            "Backoff is exponential (base * 2**attempt). Set to 0 to fail fast. "
            "Default 3."
        ),
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base sleep between retries; doubles each attempt. Default 2s.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max patients to process this run")
    parser.add_argument("--force", action="store_true", help="Regenerate even if vignette exists")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    # Validate the token-budget configuration before any LLM call.
    effective_budget = effective_input_budget(
        model_context_tokens=args.model_context_tokens,
        max_output_tokens=args.max_output_tokens,
        safety_margin=args.token_safety_margin,
    )
    if args.max_input_tokens > 0 and args.max_input_tokens > effective_budget:
        raise SystemExit(
            f"--max-input-tokens ({args.max_input_tokens}) exceeds the "
            f"effective input budget ({effective_budget}) derived from "
            f"--model-context-tokens={args.model_context_tokens}, "
            f"--max-output-tokens={args.max_output_tokens}, "
            f"--token-safety-margin={args.token_safety_margin}. "
            "Lower --max-input-tokens or raise --model-context-tokens."
        )
    logger.info(
        "Token budget: max_input_tokens=%d effective_input_budget=%d "
        "(model_context=%d, max_output=%d, safety_margin=%d)",
        args.max_input_tokens,
        effective_budget,
        args.model_context_tokens,
        args.max_output_tokens,
        args.token_safety_margin,
    )

    store = CohortStore.load(args.patients, args.items)

    pipeline = PatientSimilarityPipeline(
        xml_dir=str(args.corpus_dir),
        model=args.model,
        n_encounters=args.n_encounters,
        generation_overrides={"temperature": 0.2, "max_tokens": args.max_output_tokens},
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
    n_truncated = 0
    n_retries_used = 0       # total retry attempts consumed across all patients
    n_ok_after_retry = 0     # patients that succeeded only after >= 1 retry
    skip_reasons = {"timeline_fail": 0, "empty_timeline": 0, "llm_fail": 0}

    def _bump_skip(reason: str) -> None:
        nonlocal n_skip
        n_skip += 1
        skip_reasons[reason] += 1

    def _summarize_with_retry(text: str, pid: str, et: str) -> tuple[str | None, int]:
        """Return (vignette_or_None, attempts_beyond_first).

        Applies exponential backoff: after the k-th failure (0-indexed, i.e. the
        first failure is k=0), sleeps ``backoff * 2**k`` seconds before the next
        attempt. Total attempts = max_retries + 1.
        """
        last_exc: Exception | None = None
        for attempt in range(args.max_retries + 1):
            try:
                return summarizer.summarize(text), attempt
            except Exception as e:
                last_exc = e
                if attempt < args.max_retries:
                    sleep_s = args.retry_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "LLM attempt %d/%d failed for %s@%s: %s — retrying in %.1fs",
                        attempt + 1, args.max_retries + 1, pid, et, e, sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    logger.error(
                        "LLM FINAL failure for %s@%s after %d attempts: %s",
                        pid, et, args.max_retries + 1, last_exc,
                    )
        return None, args.max_retries

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

        was_trunc = False
        if args.max_input_tokens > 0:
            text, orig_tokens, was_trunc = truncate_to_tokens(text, args.max_input_tokens)
            if was_trunc:
                n_truncated += 1
                logger.info(
                    "Truncated %s@%s: %d tokens -> %d",
                    p.person_id, p.embed_time, orig_tokens, args.max_input_tokens,
                )

        vignette, attempts_used = _summarize_with_retry(text, p.person_id, p.embed_time)
        n_retries_used += attempts_used
        if vignette is None:
            _bump_skip("llm_fail")
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
            continue
        if attempts_used > 0:
            n_ok_after_retry += 1

        store.update_patient(
            replace(p, vignette=vignette, vignette_input_was_truncated=was_trunc)
        )
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
        "truncated=%d (max_input_tokens=%d) "
        "retries_used=%d (ok_after_retry=%d, max_retries=%d) "
        "total_states_with_vignette=%d/%d",
        n_ok,
        n_skip,
        skip_reasons["timeline_fail"],
        skip_reasons["empty_timeline"],
        skip_reasons["llm_fail"],
        n_truncated,
        args.max_input_tokens,
        n_retries_used,
        n_ok_after_retry,
        args.max_retries,
        total_with_vignette,
        len(store.patient_states()),
    )


if __name__ == "__main__":
    main()
