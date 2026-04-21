#!/usr/bin/env python3
"""
Print the LLM prompt that would be built for a given (person_id, task) under
one or more --context modes. Does NOT call the LLM.

Useful for eyeballing what the query sees with vs without few-shot exemplars.

Examples:
  # Print all 3 variants for the first item of the first pool pid
  uv run python experiments/fewshot_with_labels/show_prompt.py

  # Specific patient + task, vignette-only
  uv run python experiments/fewshot_with_labels/show_prompt.py \\
    --person-id 135908791 --task died_any_cause_1_yr --context vignette
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import (
    CohortStore,
    DeterministicTimelineLinearizationGenerator,
    TaskAwareRetriever,
)
from experiments.fewshot_with_labels import _paths

# Reuse the prompt builder + constants from run_experiment so the preview is
# guaranteed to match what run_experiment actually sends.
from experiments.fewshot_with_labels.run_experiment import (  # noqa: E402
    CONTEXT_CHOICES,
    SYSTEM_PROMPT,
    build_prompt,
    trunc_text,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the prompt for one (pid, task)")
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
        help="Used only when --person-id is not provided (picks the first pool pid).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_paths.corpus_dir(),
    )
    parser.add_argument("--person-id", type=str, default=None)
    parser.add_argument("--task", type=str, default=None,
                        help="If omitted, the first non-(-1) task of the chosen patient is used.")
    parser.add_argument(
        "--context",
        choices=list(CONTEXT_CHOICES) + ["all"],
        default="all",
        help="Which context variant(s) to print. 'all' prints every variant in CONTEXT_CHOICES.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--n-encounters",
        type=int,
        default=0,
        help="Keep only the last N encounters before embed_time. 0 = all (default).",
    )
    parser.add_argument("--max-chars", type=int, default=120_000)
    parser.add_argument("--candidate-split", type=str, default="train")
    parser.add_argument(
        "--show-system",
        action="store_true",
        help="Also print the system prompt at the top.",
    )
    args = parser.parse_args()

    store = CohortStore.load(args.patients, args.items)

    # Resolve query pid.
    pid = args.person_id
    if pid is None:
        if not args.pool.exists():
            raise SystemExit(
                f"No --person-id given and {args.pool} not found. "
                "Run sample_pool.py first or pass --person-id."
            )
        pool = json.load(open(args.pool))
        if not pool:
            raise SystemExit(f"{args.pool} is empty; pass --person-id explicitly.")
        pid = str(pool[0])

    items = [it for it in store.items_for_patient(pid) if it.label != -1]
    if not items:
        raise SystemExit(f"No non-(-1) items for patient {pid}.")

    if args.task is None:
        item = items[0]
    else:
        matches = [it for it in items if it.task == args.task]
        if not matches:
            tasks_list = sorted({it.task for it in items})
            raise SystemExit(
                f"Patient {pid} has no item for task={args.task!r}. "
                f"Available: {tasks_list}"
            )
        item = matches[0]

    state = store.get_or_none(pid, item.embed_time)
    if state is None:
        raise SystemExit(f"No PatientState for ({pid}, {item.embed_time}).")
    if not state.vignette.strip():
        logger.warning(
            "Query state (%s, %s) has empty vignette — run precompute_vignettes.py.",
            pid, item.embed_time,
        )

    retriever = TaskAwareRetriever(store, candidate_split=args.candidate_split)
    neighbors = retriever.retrieve(
        query_vignette=state.vignette,
        task=item.task,
        top_k=args.top_k,
        exclude_pid=pid,
    )

    base_generator = DeterministicTimelineLinearizationGenerator(str(args.corpus_dir))
    query_timeline = trunc_text(
        base_generator.generate(pid, cutoff_date=item.embed_time),
        args.max_chars,
    )

    contexts = list(CONTEXT_CHOICES) if args.context == "all" else [args.context]

    print("=" * 80)
    print(f"Query patient  : {pid}")
    print(f"Split          : {state.split}")
    print(f"Embed time     : {item.embed_time}")
    print(f"Task           : {item.task}")
    print(f"Ground truth   : label={item.label} ({item.label_description})")
    print(f"Retrieved top-{args.top_k} similars (train split, same task):")
    for n in neighbors:
        print(
            f"  - {n.patient.person_id} @ {n.patient.embed_time} "
            f"label={n.item.label} ({n.item.label_description}) score={n.score:.3f}"
        )
    if not neighbors:
        print("  (none — retriever returned no candidates for this task)")
    print("=" * 80)

    if args.show_system:
        print("\n[SYSTEM]")
        print(SYSTEM_PROMPT)

    for ctx in contexts:
        prompt = build_prompt(
            context=ctx,
            query_pid=pid,
            query_embed_time=item.embed_time,
            task=item.task,
            question=item.question,
            query_vignette=state.vignette,
            query_timeline=query_timeline,
            neighbors=neighbors,
            base_generator=base_generator,
            n_encounters=args.n_encounters,
            max_chars=args.max_chars,
        )
        print("\n" + "#" * 80)
        print(f"### --context {ctx}   ({len(prompt)} chars)")
        print("#" * 80)
        print(prompt)


if __name__ == "__main__":
    main()
