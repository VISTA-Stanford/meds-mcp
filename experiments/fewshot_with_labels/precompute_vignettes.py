#!/usr/bin/env python3
"""
Precompute per-(patient, embed_time, task) vignettes for the
fewshot_with_labels experiment.

Reads outputs/patients.jsonl (produced by build_cohort.py) and outputs/items.jsonl.
For every (state, task) pair where ``state.task_vignettes[task]`` is empty,
this script:

  1) Linearizes that patient's XML timeline up to the state's embed_time
     (deterministic; cached per state to avoid redundant XML parses).
  2) LLM-summarizes the linearized text into a short, task-aware vignette.
     The system prompt is rendered from
     ``configs/prompts/vignette_prompt.example.txt`` with ``{TASK_QUESTION}``
     filled from ``item.question`` and ``{TASK_FOCUS}`` filled from
     ``TASK_DESCRIPTIONS[task]``.

Vignettes are stored in ``PatientState.task_vignettes[task]``. Writes the
updated patients.jsonl atomically after every success (resumable). Use
``--force`` to regenerate.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import (
    CohortStore,
    PatientSimilarityPipeline,
    demographics_block,
)
from meds_mcp.experiments.task_config import TASK_DESCRIPTIONS
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
        default=None,
        help=(
            "Cap the linearized timeline at this many tokens before sending it "
            "to the summarizer. Oversized timelines are head+tail truncated with "
            "an explicit [truncated] marker. 0 disables truncation. "
            "DEFAULT: unset → use the full effective input budget "
            "(model_context - max_output - safety_margin - system_prompt ≈ "
            "116K tokens for apim:gpt-4.1-mini), i.e. as close to the model's "
            "advertised 128K context as APIM deployments practically allow. "
            "Set a smaller number only if you need faster / cheaper runs at the "
            "cost of truncating longer trajectories."
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
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Restrict generation to these task names (must appear in "
            "TASK_DESCRIPTIONS). Default: every task present in the items file."
        ),
    )
    args = parser.parse_args()

    # Derive / validate the token-budget configuration before any LLM call.
    effective_budget = effective_input_budget(
        model_context_tokens=args.model_context_tokens,
        max_output_tokens=args.max_output_tokens,
        safety_margin=args.token_safety_margin,
    )
    if args.max_input_tokens is None:
        # Unset → use the full effective budget (i.e. ~the model context
        # window). This gives the summarizer the whole trajectory for every
        # patient whose linearized timeline fits; only pathological outliers
        # get head+tail-truncated.
        args.max_input_tokens = effective_budget
        logger.info(
            "Token budget: max_input_tokens defaulted to effective_input_budget=%d "
            "(model_context=%d, max_output=%d, safety_margin=%d)",
            effective_budget,
            args.model_context_tokens,
            args.max_output_tokens,
            args.token_safety_margin,
        )
    elif args.max_input_tokens > 0 and args.max_input_tokens > effective_budget:
        raise SystemExit(
            f"--max-input-tokens ({args.max_input_tokens}) exceeds the "
            f"effective input budget ({effective_budget}) derived from "
            f"--model-context-tokens={args.model_context_tokens}, "
            f"--max-output-tokens={args.max_output_tokens}, "
            f"--token-safety-margin={args.token_safety_margin}. "
            "Lower --max-input-tokens or raise --model-context-tokens."
        )
    else:
        logger.info(
            "Token budget: max_input_tokens=%d (user-supplied) "
            "effective_input_budget=%d (model_context=%d, max_output=%d, safety_margin=%d)",
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

    # Build the (state_key -> [(task, question)]) plan up front. We share one
    # linearized timeline per state across its tasks.
    per_task_plan: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    if args.tasks:
        unknown = [t for t in args.tasks if t not in TASK_DESCRIPTIONS]
        if unknown:
            raise SystemExit(
                f"Unknown task name(s) for --tasks: {unknown}. "
                f"Known: {sorted(TASK_DESCRIPTIONS)}"
            )
        allowed_tasks = set(args.tasks)
    else:
        allowed_tasks = set(TASK_DESCRIPTIONS)

    for it in store.items():
        if it.task not in allowed_tasks:
            continue
        if not it.embed_time:
            continue
        state = store.get_or_none(it.person_id, it.embed_time)
        if state is None:
            continue
        existing = state.task_vignettes.get(it.task) or ""
        if not args.force and existing.strip():
            continue
        per_task_plan.setdefault(state.key, []).append((it.task, it.question))

    todo = [store.get(pid, et) for (pid, et) in per_task_plan.keys()]

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
    n_ok_after_shrink = 0    # patients that succeeded only after input shrinking
    n_fallback_vignettes = 0 # patients given a deterministic fallback vignette
    skip_reasons = {"timeline_fail": 0, "empty_timeline": 0, "llm_fail": 0}

    # Floor below which we don't shrink input further — tokens smaller than
    # this leave essentially nothing for the summarizer to work with.
    MIN_SHRINK_TOKENS = 512

    def _bump_skip(reason: str) -> None:
        nonlocal n_skip
        n_skip += 1
        skip_reasons[reason] += 1

    def _fallback_vignette(
        demos_text: str,
        timeline_text: str,
        pid: str,
        et: str,
    ) -> str:
        """Deterministic last-resort vignette when the LLM refuses every attempt.

        Stitches the demographics block (already formatted) with a short
        event-line excerpt so BM25 has *something* indexable downstream.
        Prefixed with a [FALLBACK:deterministic] marker so it's easy to
        filter these rows out of metrics.
        """
        excerpt, _orig, _trunc = truncate_to_tokens(timeline_text, 400)
        parts = ["[FALLBACK:deterministic — LLM unavailable for this patient]"]
        if demos_text.strip():
            parts.append(demos_text.rstrip())
        if excerpt.strip():
            parts.append(
                "Excerpt of clinical events:\n" + excerpt.strip()
            )
        parts.append(
            f"(Generated deterministically for {pid} at prediction time {et}.)"
        )
        return "\n\n".join(parts)

    def _summarize_with_retry(
        text: str, pid: str, et: str,
        *,
        task_question: str,
        task_focus: str,
    ) -> tuple[str | None, int, bool]:
        """Return (vignette_or_None, attempts_used, shrunk).

        Each retry:
          - Waits exponential backoff (``retry_backoff_seconds * 2**k``).
          - Halves the input (head+tail truncation) before the next call.
            Floors at ``MIN_SHRINK_TOKENS`` — we won't shrink below that.

        This handles both context-overflow edge cases (input was too big even
        at the proactive cap) and content-filter hits where the offending
        passage might sit in the middle of the timeline.

        ``shrunk=True`` if any attempt after the first used a smaller input
        than the original.
        """
        last_exc: Exception | None = None
        current = text
        shrunk = False
        for attempt in range(args.max_retries + 1):
            try:
                return (
                    summarizer.summarize(
                        current,
                        task_question=task_question,
                        task_focus=task_focus,
                    ),
                    attempt,
                    shrunk,
                )
            except Exception as e:
                last_exc = e
                if attempt < args.max_retries:
                    sleep_s = args.retry_backoff_seconds * (2 ** attempt)
                    # Shrink for the next attempt (head+tail truncation).
                    new_budget = max(MIN_SHRINK_TOKENS, count_tokens(current) // 2)
                    shrunken_text, _orig, did_trunc = truncate_to_tokens(
                        current, new_budget
                    )
                    if did_trunc:
                        current = shrunken_text
                        shrunk = True
                    logger.warning(
                        "LLM attempt %d/%d failed for %s@%s: %s — "
                        "retrying in %.1fs (input size now: %d tokens%s)",
                        attempt + 1, args.max_retries + 1, pid, et, e, sleep_s,
                        count_tokens(current),
                        " [shrunk]" if shrunk else "",
                    )
                    time.sleep(sleep_s)
                else:
                    logger.error(
                        "LLM FINAL failure for %s@%s after %d attempts: %s",
                        pid, et, args.max_retries + 1, last_exc,
                    )
        return None, args.max_retries, shrunk

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

        # Prepend a deterministic demographics block so the LLM can open with
        # an exact USMLE-style "A N-year-old <sex>…" instead of interpolating
        # vague terms like "middle-aged" when age isn't an explicit event.
        demos = demographics_block(
            xml_dir=str(args.corpus_dir),
            patient_id=p.person_id,
            cutoff_date=p.embed_time,
        )
        if demos:
            text = demos + "\n" + text

        # One LLM call per (state, task). Timeline + demographics are reused;
        # the system prompt's TASK / FOCUS placeholders change per call.
        tasks_for_state = per_task_plan.get(p.key, [])
        if not tasks_for_state:
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={n_ok} skip={n_skip}")
            continue

        new_task_vignettes = dict(p.task_vignettes)
        any_was_shrunk = False
        for task_name, task_question in tasks_for_state:
            task_focus = TASK_DESCRIPTIONS[task_name]
            v, attempts_used, was_shrunk = _summarize_with_retry(
                text, p.person_id, p.embed_time,
                task_question=task_question,
                task_focus=task_focus,
            )
            n_retries_used += attempts_used
            any_was_shrunk = any_was_shrunk or was_shrunk
            if v is None:
                logger.warning(
                    "Using deterministic fallback vignette for %s@%s task=%s "
                    "(LLM failed after %d attempts).",
                    p.person_id, p.embed_time, task_name, args.max_retries + 1,
                )
                v = _fallback_vignette(demos, text, p.person_id, p.embed_time)
                n_fallback_vignettes += 1
                skip_reasons["llm_fail"] += 1
            else:
                if attempts_used > 0:
                    n_ok_after_retry += 1
                if was_shrunk:
                    n_ok_after_shrink += 1
            new_task_vignettes[task_name] = v

        store.update_patient(
            replace(
                p,
                task_vignettes=new_task_vignettes,
                vignette_input_was_truncated=p.vignette_input_was_truncated
                or was_trunc
                or any_was_shrunk,
            )
        )
        n_ok += 1

        # Persist after every state to keep the script resumable.
        store.save(args.patients, args.items)

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(f"ok={n_ok} skip={n_skip} last={p.person_id}@{p.embed_time}")
        elif n_ok % 25 == 0:
            logger.info("Wrote %d vignettes...", n_ok)

    if pbar:
        pbar.close()

    # Each state can hold many task-specific vignettes; count the total
    # number of populated (state, task) entries.
    total_with_vignette = sum(
        1
        for p in store.patient_states()
        for v in p.task_vignettes.values()
        if v.strip()
    )
    logger.info(
        "Done. ok=%d (of which fallback=%d, after_retry=%d, after_shrink=%d) "
        "skip=%d (timeline_fail=%d, empty_timeline=%d); "
        "llm_fail_during=%d "
        "truncated=%d (max_input_tokens=%d) "
        "retries_used=%d (max_retries=%d) "
        "total_states_with_vignette=%d/%d",
        n_ok,
        n_fallback_vignettes,
        n_ok_after_retry,
        n_ok_after_shrink,
        n_skip,
        skip_reasons["timeline_fail"],
        skip_reasons["empty_timeline"],
        skip_reasons["llm_fail"],
        n_truncated,
        args.max_input_tokens,
        n_retries_used,
        args.max_retries,
        total_with_vignette,
        len(store.patient_states()),
    )


if __name__ == "__main__":
    main()
