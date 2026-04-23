#!/usr/bin/env python3
"""
fewshot_with_labels experiment: hold retrieval fixed, vary LLM context detail.

For each (query_pid, task) item belonging to the sampled valid-split pool,
the script:

  1) Retrieves top-k similar patients via vignette<->vignette BM25 from the
     TRAIN split, restricted to patients who have a non-(-1) label for the
     SAME task (per-task index).
  2) Builds an LLM prompt according to --context:
       baseline: query timeline only (no similars shown; similars still logged).
       vignette: query vignette + question + similar patients' vignettes +
                 each similar's ground-truth Yes/No for the same task.
       timeline: query timeline + question + similar patients' timelines
                 (chopped at each similar's own embed_time) + each similar's
                 ground-truth Yes/No.
  3) Asks the LLM for a single "Yes" or "No" answer.
  4) Records raw + parsed answer + ground truth + model/seed/temperature.

One run per --context; output file is experiment_results_<context>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.server.llm import (
    extract_response_content,
    get_default_generation_config,
    get_llm_client,
)
from meds_mcp.similarity import (
    CohortStore,
    DeterministicTimelineLinearizationGenerator,
    SimilarNeighbor,
    TaskAwareRetriever,
    demographics_block,
)
from experiments.fewshot_with_labels import _paths
from experiments.fewshot_with_labels._tokens import (
    count_tokens,
    effective_input_budget,
    summary_stats,
    truncate_to_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Silence llama-index's per-index "Building index from IDs objects" DEBUG spam
# (40 lines per run_experiment invocation at startup). TaskAwareRetriever's
# own INFO summary line ("TaskAwareRetriever built: N tasks ...") remains.
logging.getLogger("llama_index").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CONTEXT_CHOICES = (
    "baseline_vignette",
    "baseline_timeline",
    "vignette",
    "timeline",
)

# Decomposed flags driven by --context. Keeping these derived in one place so
# the rest of the script can reason by capability instead of string matching.
WITH_SIMILARS = {"vignette", "timeline"}
QUERY_AS_VIGNETTE = {"baseline_vignette", "vignette"}
QUERY_AS_TIMELINE = {"baseline_timeline", "timeline"}

SYSTEM_PROMPT = (
    "You are a clinical-prediction assistant. Answer the user's question about the query "
    "patient using only the evidence in the user message. Respond with exactly one word "
    "and nothing else: Yes or No. Do not include punctuation, reasoning, or any other text."
)


def trunc_text(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + "\n...[truncated]...\n" + s[-half:]


def parse_yes_no(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Exact single-word match first.
    first = s.split()[0].strip(".,;:!?\"'`)")
    lower = first.lower()
    if lower in ("yes", "y", "1", "true"):
        return "Yes"
    if lower in ("no", "n", "0", "false"):
        return "No"
    # Fallback regex anywhere in the text.
    m = re.search(r"\b(yes|no)\b", s, re.IGNORECASE)
    if m:
        return "Yes" if m.group(1).lower() == "yes" else "No"
    return None


def label_to_yesno(label: int) -> str:
    return "Yes" if int(label) == 1 else "No"


@dataclass(frozen=True)
class PromptRenderResult:
    """Result of rendering a user prompt with optional progressive trimming.

    ``tokens_total`` is the token count of the final ``prompt`` (user message
    only — system prompt is counted separately in the caller). ``tokens_before_trim``
    is what the prompt would have been without trimming; equal to ``tokens_total``
    if nothing was dropped or truncated.
    """

    prompt: str
    tokens_total: int
    tokens_before_trim: int
    neighbors_dropped_ids: list[str]
    query_truncated: bool


def _render_neighbor_block(
    neighbor: SimilarNeighbor,
    *,
    context: str,
    base_generator: DeterministicTimelineLinearizationGenerator,
    n_encounters: int,
    max_chars: int,
    xml_dir: Optional[str] = None,
) -> Optional[str]:
    """Render a single similar-patient block. Returns None if rendering fails
    (e.g. missing XML) — caller should skip that neighbor."""
    answer = label_to_yesno(neighbor.item.label)
    if context == "vignette":
        neighbor_text = neighbor.patient.vignette
    elif context == "timeline":
        try:
            neighbor_text = base_generator.generate(
                patient_id=neighbor.patient.person_id,
                cutoff_date=neighbor.patient.embed_time,
                n_encounters=n_encounters,
            )
        except Exception as e:
            logger.warning("Similar timeline failed %s: %s", neighbor.patient.person_id, e)
            return None
        neighbor_text = trunc_text(neighbor_text, max_chars)
        # Prepend demographics so the LLM sees age/sex for the neighbor too.
        if xml_dir:
            demos = demographics_block(
                xml_dir=xml_dir,
                patient_id=neighbor.patient.person_id,
                cutoff_date=neighbor.patient.embed_time,
            )
            if demos:
                neighbor_text = demos + "\n" + neighbor_text
    else:
        raise ValueError(f"Unknown context for neighbor rendering: {context!r}")
    return f"SIMILAR PATIENT INFORMATION:\n{neighbor_text}\nANSWER: {answer}"


def _assemble_prompt(query_block: str, question_block: str, neighbor_blocks: list[str]) -> str:
    """Concatenate the prompt pieces in the canonical order."""
    if not neighbor_blocks:
        return f"{query_block}\n{question_block}"
    similars_block = "\n\n".join(neighbor_blocks)
    # Canonical layout: query info -> question -> exemplars.
    return f"{query_block}\n{question_block}\n{similars_block}"


def build_prompt(
    *,
    context: str,
    query_pid: str,
    query_embed_time: str,
    task: str,
    question: str,
    query_vignette: str,
    query_timeline: str,
    neighbors: list[SimilarNeighbor],
    base_generator: DeterministicTimelineLinearizationGenerator,
    n_encounters: int,
    max_chars: int,
    max_prompt_tokens: Optional[int] = None,
    system_tokens: int = 0,
    xml_dir: Optional[str] = None,
) -> PromptRenderResult:
    """Render the user prompt for one (query_pid, task) row.

    When ``max_prompt_tokens`` is provided, the rendered prompt is trimmed so
    that ``system_tokens + count_tokens(prompt) <= max_prompt_tokens``. Trim
    policy when the candidate prompt exceeds the budget:

      1. Neighbor timelines are linearized ONCE (cached per render) so we
         never re-invoke ``base_generator.generate`` inside a trim loop.
      2. Drop the single largest remaining neighbor block; repeat.
      3. If no neighbors left and budget still exceeded, ``truncate_to_tokens``
         the query block to the remaining budget.

    Returns a ``PromptRenderResult`` carrying the final prompt plus diagnostics.
    """
    # 1) Query representation: vignette or chopped timeline.
    if context in QUERY_AS_VIGNETTE:
        query_text = query_vignette
    elif context in QUERY_AS_TIMELINE:
        query_text = query_timeline
    else:
        raise ValueError(f"Unknown context: {context!r}")

    query_block = f"QUERY PATIENT INFORMATION:\n{query_text}\n"
    question_block = f"QUESTION:\n{question}\n"

    # 2) Render every neighbor block ONCE up front (critical: never re-generate
    #    timelines inside a trim loop). Track which pid produced each block so
    #    we can report drops.
    neighbor_pids: list[str] = []
    neighbor_blocks: list[str] = []
    if context in WITH_SIMILARS:
        for n in neighbors:
            block = _render_neighbor_block(
                n,
                context=context,
                base_generator=base_generator,
                n_encounters=n_encounters,
                max_chars=max_chars,
                xml_dir=xml_dir,
            )
            if block is None:
                continue
            neighbor_pids.append(n.patient.person_id)
            neighbor_blocks.append(block)

    # 3) Assemble candidate prompt and measure.
    prompt = _assemble_prompt(query_block, question_block, neighbor_blocks)
    tokens_before_trim = count_tokens(prompt)
    dropped_ids: list[str] = []
    query_truncated = False

    # 4) Progressive trim if --max-prompt-tokens is active.
    if max_prompt_tokens is not None and max_prompt_tokens > 0:
        budget = max_prompt_tokens - max(0, system_tokens)
        if budget <= 0:
            raise ValueError(
                f"max_prompt_tokens ({max_prompt_tokens}) <= system_tokens "
                f"({system_tokens}); no budget left for the user prompt."
            )

        # 4a) Drop largest neighbor until fits or no neighbors left.
        while neighbor_blocks and count_tokens(prompt) > budget:
            sizes = [count_tokens(b) for b in neighbor_blocks]
            idx = max(range(len(sizes)), key=lambda i: sizes[i])
            dropped_ids.append(neighbor_pids[idx])
            del neighbor_blocks[idx]
            del neighbor_pids[idx]
            prompt = _assemble_prompt(query_block, question_block, neighbor_blocks)

        # 4b) If still over, truncate the query block itself.
        if count_tokens(prompt) > budget:
            # Compute the budget available to the query block alone:
            #   budget - question_block_tokens - similars_block_tokens - framing_tokens
            non_query = _assemble_prompt("", question_block, neighbor_blocks)
            overhead_tokens = count_tokens(non_query)
            # Leave a small tail buffer for the framing lines the joins add.
            query_budget = max(64, budget - overhead_tokens - 16)
            # truncate_to_tokens operates on the inner text, not the header line.
            new_query_text, _orig, was_trunc = truncate_to_tokens(query_text, query_budget)
            if was_trunc:
                query_truncated = True
                query_block = f"QUERY PATIENT INFORMATION:\n{new_query_text}\n"
                prompt = _assemble_prompt(query_block, question_block, neighbor_blocks)

    tokens_total = count_tokens(prompt)
    return PromptRenderResult(
        prompt=prompt,
        tokens_total=tokens_total,
        tokens_before_trim=tokens_before_trim,
        neighbors_dropped_ids=dropped_ids,
        query_truncated=query_truncated,
    )


def run_llm(
    client: Any,
    model: str,
    user_prompt: str,
    *,
    temperature: float,
    max_tokens: int = 8,
) -> str:
    gen = get_default_generation_config({"temperature": temperature, "max_tokens": max_tokens})
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        **gen,
    )
    return extract_response_content(resp)


def run_llm_with_retry(
    client: Any,
    model: str,
    user_prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    retries: int,
    delay_seconds: float = 1.0,
) -> Optional[str]:
    """One-shot (or N-shot) retry wrapper around ``run_llm``.

    No input mutation on retry — this is a thin safety net against transient
    APIM blips (timeouts, rate-limit bursts, empty bodies). Returns ``None``
    if all attempts fail; caller decides how to record that row.
    """
    last_exc: Optional[Exception] = None
    total_attempts = max(1, retries + 1)
    for attempt in range(total_attempts):
        try:
            return run_llm(
                client, model, user_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            last_exc = e
            if attempt < total_attempts - 1:
                logger.warning(
                    "LLM attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, total_attempts, e, delay_seconds,
                )
                time.sleep(delay_seconds)
            else:
                logger.error(
                    "LLM FINAL failure after %d attempts: %s",
                    total_attempts, last_exc,
                )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fewshot_with_labels: one LLM context variant per invocation"
    )
    parser.add_argument("--context", choices=CONTEXT_CHOICES, required=True)
    parser.add_argument(
        "--patients",
        type=Path,
        default=_paths.patients_jsonl(),
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=_paths.items_jsonl(),
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=_paths.outputs_dir() / "pool_valid_100.json",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_paths.corpus_dir(),
        help="Override via env var VISTA_CORPUS_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_paths.outputs_dir(),
    )
    parser.add_argument("--n", type=int, default=None, help="Process first N patients from pool (default: all)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Similar patients to retrieve. Ignored for baseline_* contexts.",
    )
    parser.add_argument(
        "--n-encounters",
        type=int,
        default=0,
        help="Keep only the last N encounters before embed_time for timelines. 0 = all (default).",
    )
    parser.add_argument("--max-chars", type=int, default=120_000, help="Per-timeline-block truncation cap")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help=(
            "Cap the rendered user prompt (plus system prompt) at this many tokens. "
            "On overshoot: drop the largest neighbor block first, repeat; finally "
            "head+tail-truncate the query block. If omitted, the default is computed "
            "from --model-context-tokens / --max-output-tokens / --token-safety-margin "
            "(same formula used by precompute_vignettes.py). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        default=120000,
        help=(
            "Advertised model context size in tokens, used to derive the default "
            "--max-prompt-tokens. Default 120000 (observed-safe for apim:gpt-4.1-mini)."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8,
        help=(
            "Max tokens the LLM may generate for the Yes/No answer. Default 8 "
            "(enough for 'Yes' or 'No' + a stray whitespace). Subtracted from "
            "the model context to derive --max-prompt-tokens."
        ),
    )
    parser.add_argument(
        "--token-safety-margin",
        type=int,
        default=2048,
        help=(
            "Extra buffer reserved when computing the default --max-prompt-tokens, "
            "on top of --max-output-tokens and the system prompt. Default 2048."
        ),
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=1,
        help=(
            "One-shot retry policy on LLM exceptions (transient APIM errors). "
            "No input mutation. Default 1 (total = 2 attempts). Set to 0 to "
            "fail fast."
        ),
    )
    parser.add_argument("--model", type=str, default="apim:gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--delay-seconds", type=float, default=0.3)
    parser.add_argument("--query-split", type=str, default="valid")
    parser.add_argument("--candidate-split", type=str, default="train")
    args = parser.parse_args()

    context: str = args.context
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --top-k is meaningful only when similar patients are shown in the prompt.
    # For baseline_* contexts, coerce to None so the value in the result rows
    # and meta file honestly reflects "no retrieval happened". If the user
    # explicitly passed --top-k on a baseline context, tell them it's ignored.
    if context not in WITH_SIMILARS:
        if "--top-k" in sys.argv:
            logger.warning(
                "--top-k is ignored for --context=%s (no similars shown). "
                "Proceeding with no retrieval.",
                context,
            )
        effective_top_k: Optional[int] = None
    else:
        effective_top_k = args.top_k

    with open(args.pool, encoding="utf-8") as f:
        pool_ids: list[str] = [str(x) for x in json.load(f)]

    store = CohortStore.load(args.patients, args.items)

    # Retriever only needed when the prompt will actually include similar patients.
    # baseline_* contexts are query-only, so we skip the per-task BM25 build entirely.
    retriever: TaskAwareRetriever | None = None
    if context in WITH_SIMILARS:
        retriever = TaskAwareRetriever(
            store,
            candidate_split=args.candidate_split,
        )
    else:
        logger.info(
            "Context=%s; skipping TaskAwareRetriever build (no similars in prompt).",
            context,
        )

    base_generator = DeterministicTimelineLinearizationGenerator(str(args.corpus_dir))
    client = get_llm_client(args.model)

    # Resolve pool -> query patients and items.
    selected_pids = pool_ids if args.n is None else pool_ids[: args.n]

    out_jsonl = args.output_dir / f"experiment_results_{context}.jsonl"
    t0 = time.perf_counter()

    # Token accounting — recorded per row and summarized in the meta file.
    # system_tokens is constant across rows; user_tokens varies with query +
    # similars. total_tokens = system + user (the prompt we actually send).
    system_tokens = count_tokens(SYSTEM_PROMPT)
    user_token_counts: list[int] = []
    total_token_counts: list[int] = []
    tokens_before_trim_counts: list[int] = []

    # Trim diagnostics aggregated into meta["prompt_trim_stats"].
    rows_with_any_trim = 0
    rows_with_neighbors_dropped = 0
    rows_with_query_truncated = 0
    total_neighbors_dropped = 0

    total_rows = 0
    total_skipped = 0

    # Resolve default for --max-prompt-tokens from the configured model context.
    # 0 is a sentinel: disables trimming entirely (matches the previous
    # behaviour). None → compute automatic budget.
    if args.max_prompt_tokens is None:
        effective_prompt_budget = effective_input_budget(
            model_context_tokens=args.model_context_tokens,
            max_output_tokens=args.max_output_tokens,
            safety_margin=args.token_safety_margin,
            system_tokens=system_tokens,
        )
        # Note: the cap we pass to build_prompt is the *total* prompt budget
        # (user + system); system_tokens is subtracted inside the trimming
        # logic. So we add it back here to match the convention.
        prompt_cap_total: Optional[int] = effective_prompt_budget + system_tokens
        logger.info(
            "Default --max-prompt-tokens computed from budget: %d "
            "(model_context=%d, max_output=%d, safety_margin=%d, system=%d)",
            prompt_cap_total,
            args.model_context_tokens,
            args.max_output_tokens,
            args.token_safety_margin,
            system_tokens,
        )
    elif args.max_prompt_tokens == 0:
        prompt_cap_total = None
        logger.info("--max-prompt-tokens=0 → progressive trim disabled.")
    else:
        prompt_cap_total = args.max_prompt_tokens
        logger.info("Using user-supplied --max-prompt-tokens=%d.", prompt_cap_total)

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for i, pid in enumerate(selected_pids):
            xml_path = args.corpus_dir / f"{pid}.xml"
            if not xml_path.exists():
                logger.warning("Missing XML for %s, skipping", pid)
                continue

            items_for_patient = [
                it for it in store.items_for_patient(pid) if it.label != -1
            ]
            if not items_for_patient:
                logger.info("No non-(-1) items for %s; skipping", pid)
                continue

            # Cache per-embed_time state and per-embed_time query timeline.
            state_by_et: dict[str, Any] = {}
            timeline_by_et: dict[str, str] = {}

            for item in items_for_patient:
                et = item.embed_time
                state = state_by_et.get(et)
                if state is None:
                    state = store.get_or_none(pid, et)
                    if state is None:
                        logger.warning(
                            "No PatientState for (%s, %s); skipping item task=%s",
                            pid,
                            et,
                            item.task,
                        )
                        continue
                    if state.split != args.query_split:
                        logger.warning(
                            "Pool pid %s state at %s has split=%s (expected %s); skipping item task=%s",
                            pid,
                            et,
                            state.split,
                            args.query_split,
                            item.task,
                        )
                        continue
                    # Vignette needed when it's shown as the query rep OR used for BM25 retrieval.
                    # baseline_timeline bypasses both, so an empty vignette is fine there.
                    needs_vignette = (
                        context in QUERY_AS_VIGNETTE or context in WITH_SIMILARS
                    )
                    if needs_vignette and not state.vignette.strip():
                        logger.warning(
                            "Empty vignette for (%s, %s); skipping item task=%s",
                            pid,
                            et,
                            item.task,
                        )
                        continue
                    state_by_et[et] = state

                if context in QUERY_AS_TIMELINE and et not in timeline_by_et:
                    try:
                        raw_timeline = trunc_text(
                            base_generator.generate(pid, cutoff_date=et),
                            args.max_chars,
                        )
                        # Prepend deterministic demographics (age at prediction
                        # time, sex, race, ethnicity) so the Yes/No model sees
                        # them even when the raw event stream omits them.
                        demos = demographics_block(
                            xml_dir=str(args.corpus_dir),
                            patient_id=pid,
                            cutoff_date=et,
                        )
                        timeline_by_et[et] = (
                            (demos + "\n" + raw_timeline) if demos else raw_timeline
                        )
                    except Exception as e:
                        logger.warning("Query timeline failed %s@%s: %s", pid, et, e)
                        timeline_by_et[et] = ""

                query_timeline = timeline_by_et.get(et, "")
                query_vignette = state.vignette

                if retriever is None:
                    neighbors = []
                else:
                    # Invariant: BM25 retrieval is ALWAYS vignette<->vignette,
                    # regardless of --context. Only the prompt rendering varies
                    # (vignette vs. chopped timeline for both query and similars).
                    # effective_top_k is set to the user flag for WITH_SIMILARS
                    # contexts; falls back to args.top_k just as a type guard.
                    neighbors = retriever.retrieve(
                        query_vignette=query_vignette,
                        task=item.task,
                        top_k=effective_top_k if effective_top_k is not None else args.top_k,
                        exclude_pid=pid,
                    )

                render = build_prompt(
                    context=context,
                    query_pid=pid,
                    query_embed_time=et,
                    task=item.task,
                    question=item.question,
                    query_vignette=query_vignette,
                    query_timeline=query_timeline,
                    neighbors=neighbors,
                    base_generator=base_generator,
                    n_encounters=args.n_encounters,
                    max_chars=args.max_chars,
                    max_prompt_tokens=prompt_cap_total,
                    system_tokens=system_tokens,
                    xml_dir=str(args.corpus_dir),
                )
                prompt = render.prompt

                # Token accounting for the prompt we're about to send.
                user_tokens = render.tokens_total
                total_prompt_tokens = system_tokens + user_tokens
                user_token_counts.append(user_tokens)
                total_token_counts.append(total_prompt_tokens)
                tokens_before_trim_counts.append(render.tokens_before_trim)

                # Aggregate trim diagnostics.
                any_trim = bool(render.neighbors_dropped_ids) or render.query_truncated
                if any_trim:
                    rows_with_any_trim += 1
                if render.neighbors_dropped_ids:
                    rows_with_neighbors_dropped += 1
                    total_neighbors_dropped += len(render.neighbors_dropped_ids)
                if render.query_truncated:
                    rows_with_query_truncated += 1

                raw = run_llm_with_retry(
                    client,
                    args.model,
                    prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                    retries=args.llm_retries,
                    delay_seconds=max(0.5, args.delay_seconds),
                )
                if raw is None:
                    total_skipped += 1
                    continue

                pred = parse_yes_no(raw)
                true_label_str = label_to_yesno(item.label)
                correct = pred is not None and pred == true_label_str

                # Keep similar_patient_ids aligned with what actually appeared in
                # the prompt (post-trim). similar_labels / similar_scores mirror
                # that order. The set of dropped ids is recorded separately.
                kept_neighbors = [
                    n for n in neighbors
                    if n.patient.person_id not in render.neighbors_dropped_ids
                ]
                rec = {
                    "context": context,
                    "person_id": pid,
                    "query_split": state.split,
                    "embed_time": et,
                    "task": item.task,
                    "task_group": item.task_group,
                    "question": item.question,
                    "label": item.label,
                    "label_description": item.label_description,
                    "true_yes_no": true_label_str,
                    "pred": pred,
                    "correct": correct,
                    "raw": raw,
                    "similar_patient_ids": [n.patient.person_id for n in kept_neighbors],
                    "similar_labels": [label_to_yesno(n.item.label) for n in kept_neighbors],
                    "similar_scores": [n.score for n in kept_neighbors],
                    "top_k": effective_top_k,
                    "model": args.model,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "n_encounters": args.n_encounters,
                    "prompt_tokens_system": system_tokens,
                    "prompt_tokens_user": user_tokens,
                    "prompt_tokens_total": total_prompt_tokens,
                    "prompt_tokens_before_trim": render.tokens_before_trim,
                    "neighbors_dropped_count": len(render.neighbors_dropped_ids),
                    "neighbors_dropped_ids": list(render.neighbors_dropped_ids),
                    "query_truncated": render.query_truncated,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                total_rows += 1

                if args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)

            logger.info(
                "Patient %d/%d %s: %d items processed (context=%s)",
                i + 1,
                len(selected_pids),
                pid,
                len(items_for_patient),
                context,
            )

    elapsed = time.perf_counter() - t0
    prompt_token_stats = {
        "tokenizer": "cl100k_base",
        "system_tokens": system_tokens,  # constant across rows
        "user_tokens": summary_stats(user_token_counts),
        "total_tokens": summary_stats(total_token_counts),
        "tokens_before_trim": summary_stats(tokens_before_trim_counts),
    }
    prompt_trim_stats = {
        "max_prompt_tokens": prompt_cap_total,
        "n_rows_with_any_trim": rows_with_any_trim,
        "n_rows_with_neighbors_dropped": rows_with_neighbors_dropped,
        "n_rows_with_query_truncated": rows_with_query_truncated,
        "total_neighbors_dropped": total_neighbors_dropped,
    }
    meta = {
        "context": context,
        "output": str(out_jsonl),
        "n_pool_patients_requested": len(selected_pids),
        "n_result_rows": total_rows,
        "n_skipped_rows": total_skipped,
        "elapsed_seconds": round(elapsed, 2),
        "model": args.model,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": effective_top_k,
        "n_encounters": args.n_encounters,
        "query_split": args.query_split,
        "candidate_split": args.candidate_split,
        "retrieval": (
            "vignette<->vignette BM25, per-task, train-only, non-(-1) labels"
            if context in WITH_SIMILARS
            else "disabled (baseline_* context shows no similars)"
        ),
        "pool_source": str(args.pool),
        "patients_source": str(args.patients),
        "items_source": str(args.items),
        "prompt_token_stats": prompt_token_stats,
        "prompt_trim_stats": prompt_trim_stats,
    }
    meta_path = args.output_dir / f"experiment_meta_{context}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Human-readable token summary alongside the main "Wrote ..." line.
    if total_token_counts:
        t = prompt_token_stats["total_tokens"]
        logger.info(
            "Prompt tokens (total=system+user): n=%d min=%d median=%d p90=%d max=%d "
            "mean=%.1f total=%d (system=%d)",
            t["n"], t["min"], t["median"], t["p90"], t["max"], t["mean"],
            t["total"], system_tokens,
        )

    print(
        f"Wrote {out_jsonl} ({total_rows} rows, context={context}) "
        f"in {elapsed:.1f}s (skipped={total_skipped})"
    )


if __name__ == "__main__":
    main()
