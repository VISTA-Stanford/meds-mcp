#!/usr/bin/env python3
"""
Vertex batch runner for fewshot_with_labels using the cohort store.

Builds prompts using the same prompt builder as run_experiment.py, submits one
batch job, and writes experiment_results_<context>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import CohortStore, DeterministicTimelineLinearizationGenerator, TaskAwareRetriever, demographics_block
from experiments.fewshot_with_labels import _paths
from experiments.fewshot_with_labels.run_experiment import (
    CONTEXT_CHOICES,
    QUERY_AS_TIMELINE,
    WITH_SIMILARS,
    SYSTEM_PROMPT,
    build_prompt,
    label_to_yesno,
    load_reason_cache,
    parse_yes_no,
    select_balanced_neighbors,
    trunc_text,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    s = uri.replace("gs://", "", 1)
    b, _, p = s.partition("/")
    return b, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fewshot_with_labels via Vertex batch inference.")
    parser.add_argument("--context", choices=CONTEXT_CHOICES, required=True)
    parser.add_argument("--patients", type=Path, default=_paths.patients_jsonl())
    parser.add_argument("--items", type=Path, default=_paths.items_jsonl())
    parser.add_argument("--pool", type=Path, default=_paths.outputs_dir() / "pool_valid_100.json")
    parser.add_argument("--corpus-dir", type=Path, default=_paths.corpus_dir())
    parser.add_argument("--output-dir", type=Path, default=_paths.outputs_dir())
    parser.add_argument("--query-split", type=str, default="valid")
    parser.add_argument("--candidate-split", type=str, default="train")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        metavar="TASK",
        help="Only run these task names (e.g. guo_readmission). Default: all tasks.",
    )
    parser.add_argument("--n-encounters", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=120000)
    parser.add_argument("--reason-cache", type=Path, default=_paths.outputs_dir() / "reason_cache.jsonl")
    parser.add_argument("--reason-missing-policy", choices=("placeholder", "omit", "fail"), default="placeholder")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--vertex-project", type=str, default="som-nero-plevriti-deidbdf")
    parser.add_argument("--vertex-location", type=str, default="us-central1")
    parser.add_argument(
        "--vertex-input-uri",
        type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/input/experiment_batch.jsonl",
    )
    parser.add_argument(
        "--vertex-output-prefix",
        type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/output/experiment_batch",
    )
    args = parser.parse_args()

    from google.cloud import storage
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob

    context = args.context
    store = CohortStore.load(args.patients, args.items)
    reason_by_key = load_reason_cache(args.reason_cache)
    with open(args.pool, encoding="utf-8") as f:
        pool_ids = [str(x) for x in json.load(f)]
    selected = pool_ids if args.n is None else pool_ids[: args.n]

    retriever = None
    if context in WITH_SIMILARS:
        retriever = TaskAwareRetriever(store, candidate_split=args.candidate_split)
    linearizer = DeterministicTimelineLinearizationGenerator(str(args.corpus_dir))

    requests: list[dict] = []
    example_prompt_path = args.output_dir / f"example_prompt_{context}.txt"
    example_prompt_saved = False
    for pid in selected:
        tasks_filter = set(args.tasks) if args.tasks else None
        items = [
            it for it in store.items_for_patient(pid)
            if it.label != -1 and (tasks_filter is None or it.task in tasks_filter)
        ]
        for item in items:
            state = store.get_or_none(pid, item.embed_time)
            if state is None or state.split != args.query_split:
                continue
            query_timeline = ""
            if context in QUERY_AS_TIMELINE:
                try:
                    raw = trunc_text(
                        linearizer.generate(pid, cutoff_date=item.embed_time, n_encounters=args.n_encounters),
                        args.max_chars,
                    )
                    demos = demographics_block(
                        xml_dir=str(args.corpus_dir),
                        patient_id=pid,
                        cutoff_date=item.embed_time,
                    )
                    query_timeline = (demos + "\n" + raw) if demos else raw
                except Exception:
                    query_timeline = ""

            neighbors = []
            if retriever is not None:
                raw_neighbors = retriever.retrieve(
                    query_vignette=state.vignette,
                    task=item.task,
                    top_k=args.top_k * 4,
                    exclude_pid=pid,
                )
                neighbors = select_balanced_neighbors(raw_neighbors, args.top_k)

            render = build_prompt(
                context=context,
                query_pid=pid,
                query_embed_time=item.embed_time,
                task=item.task,
                question=item.question,
                query_vignette=state.vignette,
                query_timeline=query_timeline,
                neighbors=neighbors,
                base_generator=linearizer,
                n_encounters=args.n_encounters,
                max_chars=args.max_chars,
                max_prompt_tokens=None,
                system_tokens=0,
                xml_dir=str(args.corpus_dir),
                reason_by_key=reason_by_key,
                reason_missing_policy=args.reason_missing_policy,
            )
            if not example_prompt_saved:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                example_prompt_path.write_text(
                    f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n"
                    f"=== USER PROMPT ===\n{render.prompt}\n",
                    encoding="utf-8",
                )
                example_prompt_saved = True
                logger.info("Saved example prompt to %s", example_prompt_path)

            requests.append(
                {
                    "request": {
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [{"role": "user", "parts": [{"text": render.prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": 16,
                            "thinkingConfig": {"thinkingBudget": 0},
                        },
                    },
                    "_meta": json.dumps(
                        {
                            "context": context,
                            "person_id": pid,
                            "embed_time": item.embed_time,
                            "task": item.task,
                            "task_group": item.task_group,
                            "question": item.question,
                            "label": int(item.label),
                            "label_description": item.label_description,
                            "top_k": args.top_k if context in WITH_SIMILARS else None,
                            "similar_patient_ids": [n.patient.person_id for n in neighbors],
                            "similar_labels": [label_to_yesno(n.item.label) for n in neighbors],
                            "similar_scores": [n.score for n in neighbors],
                        }
                    ),
                }
            )

    in_bucket, in_path = _parse_gcs_uri(args.vertex_input_uri)
    storage_client = storage.Client()
    storage_client.bucket(in_bucket).blob(in_path).upload_from_string(
        "\n".join(json.dumps(r) for r in requests).encode("utf-8"),
        content_type="application/jsonl",
    )
    logger.info("Uploaded %d experiment requests to %s", len(requests), args.vertex_input_uri)

    vertexai.init(project=args.vertex_project, location=args.vertex_location)
    job = BatchPredictionJob.submit(
        source_model=args.model,
        input_dataset=args.vertex_input_uri,
        output_uri_prefix=args.vertex_output_prefix,
        job_display_name=f"fewshot_experiment_{context}_{int(time.time())}",
    )
    logger.info("Submitted batch job: %s", job.resource_name)
    while not job.has_ended:
        time.sleep(30)
        job.refresh()
        logger.info("Batch state: %s", job.state)
    if not job.has_succeeded:
        raise RuntimeError(f"Batch job failed: {job.error}")

    out_bucket, out_prefix = _parse_gcs_uri(job.output_location)
    blobs = [
        b for b in storage_client.bucket(out_bucket).list_blobs(prefix=out_prefix)
        if b.name.endswith(".jsonl")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"experiment_results_{context}.jsonl"
    rows = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for blob in blobs:
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("status") not in ("", None):
                    continue
                meta = obj.get("_meta", {})
                if isinstance(meta, str):
                    meta = json.loads(meta)
                try:
                    raw = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    raw = ""
                pred = parse_yes_no(raw)
                true_yes_no = label_to_yesno(int(meta.get("label", 0)))
                rec = {
                    "context": meta.get("context"),
                    "person_id": meta.get("person_id"),
                    "query_split": args.query_split,
                    "embed_time": meta.get("embed_time"),
                    "task": meta.get("task"),
                    "task_group": meta.get("task_group"),
                    "question": meta.get("question"),
                    "label": int(meta.get("label", 0)),
                    "label_description": meta.get("label_description", ""),
                    "true_yes_no": true_yes_no,
                    "pred": pred,
                    "correct": pred is not None and pred == true_yes_no,
                    "raw": raw,
                    "similar_patient_ids": meta.get("similar_patient_ids", []),
                    "similar_labels": meta.get("similar_labels", []),
                    "similar_scores": meta.get("similar_scores", []),
                    "top_k": meta.get("top_k"),
                    "model": args.model,
                    "seed": None,
                    "temperature": 0.0,
                    "n_encounters": args.n_encounters,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows += 1
    logger.info("Wrote %d rows to %s", rows, out_path)


if __name__ == "__main__":
    main()

