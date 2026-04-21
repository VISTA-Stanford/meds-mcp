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
)
from experiments.fewshot_with_labels import _paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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


@dataclass
class PromptPieces:
    header: str
    query_block: str
    similars_block: str


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
) -> str:
    header = (
        f"QUESTION:\n{question}\n\n"
        f"Query patient {query_pid} - prediction time (embed time): {query_embed_time}\n"
        f"Task: {task}\n\n"
    )

    # Query representation block — identical for paired baseline / with-similars.
    if context in QUERY_AS_VIGNETTE:
        query_block = (
            f"Query patient vignette (summary of events on or before prediction time):\n"
            f"{query_vignette}\n"
        )
    elif context in QUERY_AS_TIMELINE:
        query_block = (
            f"Query patient timeline (events on or before prediction time):\n"
            f"{query_timeline}\n"
        )
    else:
        raise ValueError(f"Unknown context: {context!r}")

    # Baselines stop here — no similars shown.
    if context not in WITH_SIMILARS:
        return header + query_block

    # With-similars variants render each neighbor in the same representation as
    # the query so the prompt shape matches exactly aside from the similars block.
    blocks: list[str] = []
    for n in neighbors:
        answer = label_to_yesno(n.item.label)
        if context == "vignette":
            blocks.append(
                f"Similar patient {n.patient.person_id} "
                f"(vignette up to {n.patient.embed_time}):\n{n.patient.vignette}\n"
                f"Ground-truth answer for this patient: {answer}"
            )
        else:  # context == "timeline"
            try:
                t = base_generator.generate(
                    patient_id=n.patient.person_id,
                    cutoff_date=n.patient.embed_time,
                    n_encounters=n_encounters,
                )
            except Exception as e:
                logger.warning("Similar timeline failed %s: %s", n.patient.person_id, e)
                continue
            t = trunc_text(t, max_chars)
            blocks.append(
                f"Similar patient {n.patient.person_id} "
                f"(timeline up to this patient's embed time {n.patient.embed_time}):\n{t}\n"
                f"Ground-truth answer for this patient: {answer}"
            )

    prompt = header + query_block
    if blocks:
        suffix = (
            "vignette similarity"
            if context == "vignette"
            else "vignette similarity; timelines truncated if long"
        )
        prompt += (
            f"\n---\nSimilar patient exemplars (same task, retrieved by {suffix}):\n"
            + "\n\n".join(blocks)
        )
    return prompt


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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--n-encounters",
        type=int,
        default=0,
        help="Keep only the last N encounters before embed_time for timelines. 0 = all (default).",
    )
    parser.add_argument("--max-chars", type=int, default=120_000, help="Per-timeline-block truncation cap")
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
    total_rows = 0
    total_skipped = 0

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
                        timeline_by_et[et] = trunc_text(
                            base_generator.generate(pid, cutoff_date=et),
                            args.max_chars,
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
                    neighbors = retriever.retrieve(
                        query_vignette=query_vignette,
                        task=item.task,
                        top_k=args.top_k,
                        exclude_pid=pid,
                    )

                prompt = build_prompt(
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
                )

                try:
                    raw = run_llm(
                        client,
                        args.model,
                        prompt,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    logger.error("LLM failed for %s/%s: %s", pid, item.task, e)
                    total_skipped += 1
                    continue

                pred = parse_yes_no(raw)
                true_label_str = label_to_yesno(item.label)
                correct = pred is not None and pred == true_label_str

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
                    "similar_patient_ids": [n.patient.person_id for n in neighbors],
                    "similar_labels": [label_to_yesno(n.item.label) for n in neighbors],
                    "similar_scores": [n.score for n in neighbors],
                    "top_k": args.top_k,
                    "model": args.model,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "n_encounters": args.n_encounters,
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
        "top_k": args.top_k,
        "n_encounters": args.n_encounters,
        "query_split": args.query_split,
        "candidate_split": args.candidate_split,
        "retrieval": "vignette<->vignette BM25, per-task, train-only, non-(-1) labels",
        "pool_source": str(args.pool),
        "patients_source": str(args.patients),
        "items_source": str(args.items),
    }
    meta_path = args.output_dir / f"experiment_meta_{context}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Wrote {out_jsonl} ({total_rows} rows, context={context}) "
        f"in {elapsed:.1f}s (skipped={total_skipped})"
    )


if __name__ == "__main__":
    main()
