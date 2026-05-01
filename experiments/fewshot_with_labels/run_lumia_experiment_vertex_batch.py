#!/usr/bin/env python3
"""
Vertex batch runner for LUMIA-timeline contexts.

Two context modes:
  baseline_lumia  — zero-shot: filtered+truncated LUMIA XML as query, no examples
  lumia           — few-shot:  filtered+truncated LUMIA XML for query + examples

Filtering uses lumia_filter.filter_xml_by_date (cutoff = embed_time+1 day so
discharge-day events are included). Oldest encounters are dropped first when
the serialized XML exceeds --max-chars-per-patient.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import xml.etree.ElementTree as ET

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import CohortStore, TaskAwareRetriever
from experiments.fewshot_with_labels import _paths
from experiments.fewshot_with_labels.lumia_filter import filter_xml_by_date
from experiments.fewshot_with_labels.run_experiment import (
    SYSTEM_PROMPT,
    TASK_DESCRIPTIONS,
    label_to_yesno,
    parse_yes_no,
    select_balanced_neighbors,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONTEXT_CHOICES = ("baseline_lumia", "lumia")


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    s = uri.replace("gs://", "", 1)
    b, _, p = s.partition("/")
    return b, p


def get_filtered_xml(pid: str, embed_time_str: str, corpus_dir: Path, max_chars: int) -> str:
    """Filter LUMIA XML to before embed_time, truncate oldest encounters to fit max_chars."""
    cutoff = (datetime.fromisoformat(embed_time_str) + timedelta(days=1)).strftime("%Y-%m-%d")
    xml_path = corpus_dir / f"{pid}.xml"
    tree = filter_xml_by_date(str(xml_path), cutoff)
    root = tree.getroot()
    encounters = root.findall("encounter")
    # Drop oldest encounters until serialized size fits within max_chars
    while encounters:
        buf = io.BytesIO()
        tree.write(buf, encoding="utf-8", xml_declaration=True)
        if len(buf.getvalue()) <= max_chars:
            break
        root.remove(encounters.pop(0))
    ET.indent(tree)
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


def build_lumia_prompt(
    *,
    context: str,
    pid: str,
    embed_time: str,
    task: str,
    corpus_dir: Path,
    max_chars_per_patient: int,
    neighbors: list,
) -> str:
    task_desc = TASK_DESCRIPTIONS.get(task, f"Predict the outcome for task: {task}.")
    query_xml = get_filtered_xml(pid, embed_time, corpus_dir, max_chars_per_patient)

    if context == "baseline_lumia" or not neighbors:
        return (
            f"TASK:\n{task_desc}\n\n"
            f"PATIENT TIMELINE:\n{query_xml}\n"
        )

    # Few-shot: build example blocks with LUMIA timelines
    example_blocks = []
    for n in neighbors:
        n_xml = get_filtered_xml(
            n.patient.person_id, n.patient.embed_time, corpus_dir, max_chars_per_patient
        )
        answer = label_to_yesno(n.item.label)
        example_blocks.append(f"EXAMPLE PATIENT TIMELINE:\n{n_xml}\nAnswer: {answer}")

    examples_str = "\n\n".join(example_blocks)
    return (
        f"TASK:\n{task_desc}\n\n"
        f"EXAMPLES:\n{examples_str}\n\n"
        "---\n"
        "NEW PATIENT:\n"
        f"PATIENT TIMELINE:\n{query_xml}\n\n"
        "Answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMIA-timeline Vertex batch experiment.")
    parser.add_argument("--context", choices=CONTEXT_CHOICES, required=True)
    parser.add_argument(
        "--context-name", type=str, default=None,
        help="Override the context label used for output filenames and result rows "
             "(e.g. 'baseline_lumia_4k'). Defaults to --context.",
    )
    parser.add_argument("--patients", type=Path, default=_paths.patients_jsonl())
    parser.add_argument("--items", type=Path, default=_paths.items_jsonl())
    parser.add_argument("--pool", type=Path, default=None,
                        help="JSON file of patient IDs to evaluate. If omitted, uses all patients with matching split/task items.")
    parser.add_argument("--corpus-dir", type=Path, required=True,
                        help="Directory containing per-patient {pid}.xml LUMIA files.")
    parser.add_argument("--output-dir", type=Path, default=_paths.outputs_dir() / "ehrshot")
    parser.add_argument("--query-split", type=str, default="test")
    parser.add_argument("--candidate-split", type=str, default="test")
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-chars-per-patient", type=int, default=80_000,
                        help="Per-patient XML char cap; oldest encounters dropped first.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--vertex-project", type=str, default="som-nero-plevriti-deidbdf")
    parser.add_argument("--vertex-location", type=str, default="us-central1")
    parser.add_argument(
        "--vertex-input-uri", type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/input/ehrshot_lumia_batch.jsonl",
    )
    parser.add_argument(
        "--vertex-output-prefix", type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/output/ehrshot_lumia_batch",
    )
    args = parser.parse_args()

    from google.cloud import storage
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob

    context = args.context
    context_name = args.context_name or context
    store = CohortStore.load(args.patients, args.items)

    if args.pool is not None:
        with open(args.pool, encoding="utf-8") as f:
            pool_ids = [str(x) for x in json.load(f)]
    else:
        tasks_filter = set(args.tasks) if args.tasks else None
        pool_ids = [
            str(json.loads(l)["person_id"])
            for l in args.patients.open()
        ]
        pool_ids = [
            pid for pid in pool_ids
            if any(
                it.label != -1
                and (tasks_filter is None or it.task in tasks_filter)
                and (s := store.get_or_none(pid, it.embed_time)) is not None
                and s.split == args.query_split
                for it in store.items_for_patient(pid)
            )
        ]
        logger.info("Derived %d patients with %s-split items from patients file", len(pool_ids), args.query_split)

    retriever = None
    if context == "lumia":
        retriever = TaskAwareRetriever(store, candidate_split=args.candidate_split)

    requests: list[dict] = []
    example_prompt_path = args.output_dir / f"example_prompt_{context_name}.txt"
    example_prompt_saved = False

    for pid in pool_ids:
        tasks_filter = set(args.tasks) if args.tasks else None
        items = [
            it for it in store.items_for_patient(pid)
            if it.label != -1 and (tasks_filter is None or it.task in tasks_filter)
        ]
        for item in items:
            state = store.get_or_none(pid, item.embed_time)
            if state is None or state.split != args.query_split:
                continue

            xml_path = args.corpus_dir / f"{pid}.xml"
            if not xml_path.exists():
                logger.warning("Missing XML for %s, skipping", pid)
                continue

            neighbors = []
            if retriever is not None:
                raw = retriever.retrieve(
                    query_vignette=state.vignette,
                    task=item.task,
                    top_k=args.top_k * 4,
                    exclude_pid=pid,
                )
                neighbors = select_balanced_neighbors(raw, args.top_k)

            prompt = build_lumia_prompt(
                context=context,
                pid=pid,
                embed_time=item.embed_time,
                task=item.task,
                corpus_dir=args.corpus_dir,
                max_chars_per_patient=args.max_chars_per_patient,
                neighbors=neighbors,
            )

            if not example_prompt_saved:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                example_prompt_path.write_text(
                    f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n"
                    f"=== USER PROMPT ===\n{prompt}\n",
                    encoding="utf-8",
                )
                example_prompt_saved = True
                logger.info("Saved example prompt to %s", example_prompt_path)

            requests.append({
                "request": {
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 16,
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
                "_meta": json.dumps({
                    "context": context_name,
                    "person_id": pid,
                    "embed_time": item.embed_time,
                    "task": item.task,
                    "task_group": item.task_group,
                    "question": item.question,
                    "label": int(item.label),
                    "label_description": item.label_description,
                    "top_k": args.top_k if context == "lumia" else None,
                    "similar_patient_ids": [n.patient.person_id for n in neighbors],
                    "similar_labels": [label_to_yesno(n.item.label) for n in neighbors],
                    "similar_scores": [n.score for n in neighbors],
                    "max_chars_per_patient": args.max_chars_per_patient,
                }),
            })

    logger.info("Built %d requests for context=%s", len(requests), context)

    in_bucket, in_path = _parse_gcs_uri(args.vertex_input_uri)
    storage_client = storage.Client()
    storage_client.bucket(in_bucket).blob(in_path).upload_from_string(
        "\n".join(json.dumps(r) for r in requests).encode("utf-8"),
        content_type="application/jsonl",
    )
    logger.info("Uploaded %d requests to %s", len(requests), args.vertex_input_uri)

    vertexai.init(project=args.vertex_project, location=args.vertex_location)
    job = BatchPredictionJob.submit(
        source_model=args.model,
        input_dataset=args.vertex_input_uri,
        output_uri_prefix=args.vertex_output_prefix,
        job_display_name=f"lumia_experiment_{context_name}_{int(time.time())}",
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
    out_path = args.output_dir / f"experiment_results_{context_name}.jsonl"
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
                    "max_chars_per_patient": meta.get("max_chars_per_patient"),
                    "model": args.model,
                    "temperature": 0.0,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows += 1
    logger.info("Wrote %d rows to %s", rows, out_path)


if __name__ == "__main__":
    main()
