#!/usr/bin/env python3
"""
Batch inference runner for EHRSHOT and VISTA rubricified datasets.

Submits one Vertex AI batch prediction job per invocation.  Run 4 times to
cover both datasets and both contexts:

  python run_batch_full.py --dataset ehrshot --context baseline_vignette
  python run_batch_full.py --dataset ehrshot --context vignette --top-k 3
  python run_batch_full.py --dataset vista   --context baseline_vignette
  python run_batch_full.py --dataset vista   --context vignette --top-k 3

Key fixes vs the previous run_experiment_batch.py
--------------------------------------------------
* thinkingBudget=0  — Gemini 2.5 Flash burns its thinking tokens before
  generating output; with maxOutputTokens=8 nothing is left for the actual
  "Yes"/"No".  Disabling thinking drops the failure rate from ~80% to ~0%.
* maxOutputTokens raised to 16 as an extra safety margin.
* Empty-parts responses (finishReason=MAX_TOKENS with no text) are caught and
  recorded as pred=None rather than silently dropped.
* Both val and test splits are queried by default (--query-splits both).
  Train is always used as the retrieval candidate pool.

Dataset configs
---------------
  ehrshot : LRRL_MEDS/data/rubric/rubricified/       task questions from config/tasks.py
  vista   : LRRL_MEDS/data/vista_rubric/rubricified/  task questions from config/vista_tasks.json

GCS output bucket  :  gs://vista_bench/temp/pinnacle_templated_summaries/
Vertex AI project  :  som-nero-plevriti-deidbdf
Vertex AI location :  us-central1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from google.cloud import storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import vertexai
from vertexai.batch_prediction import BatchPredictionJob

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

LRRL_MEDS        = Path.home() / "LRRL_MEDS"
OUTPUT_DIR       = Path.home() / "meds-mcp/experiments/fewshot_with_labels/outputs"

GCS_BUCKET       = "vista_bench"
GCS_PREFIX       = "temp/pinnacle_templated_summaries"
VERTEX_PROJECT   = "som-nero-plevriti-deidbdf"
VERTEX_LOCATION  = "us-central1"
VERTEX_MODEL     = "gemini-2.5-flash"

TOP_K_DEFAULT    = 3
TRAIN_SPLIT      = "train"
POLL_INTERVAL    = 60  # seconds

DATASET_CONFIGS = {
    "ehrshot": {
        "rubric_base":     LRRL_MEDS / "data/rubric/rubricified",
        "task_questions":  LRRL_MEDS / "config/tasks.py",  # loads TASKS dict
        "label_type":      "bool",  # labels are True/False strings
    },
    "vista": {
        "rubric_base":     LRRL_MEDS / "data/vista_rubric/rubricified",
        "task_questions":  LRRL_MEDS / "config/vista_tasks.json",
        "label_type":      "bool",
    },
}

SYSTEM_PROMPT = (
    "You are a clinical research AI assistant analyzing de-identified patient records "
    "for medical research purposes. "
    "Answer the prediction question using only the evidence provided in the user message. "
    "Respond with exactly one word and nothing else: Yes or No. "
    "Do not include punctuation, reasoning, or any other text."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def label_to_int(label_val) -> int:
    if isinstance(label_val, bool):
        return int(label_val)
    s = str(label_val).strip().lower()
    return 1 if s in ("true", "1", "yes") else 0


def label_to_yesno(label_val) -> str:
    return "Yes" if label_to_int(label_val) == 1 else "No"


def parse_yes_no(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    first = text.strip().split()[0].strip(".,;:!?\"'`)")
    lower = first.lower()
    if lower in ("yes", "y", "1", "true"):
        return "Yes"
    if lower in ("no", "n", "0", "false"):
        return "No"
    m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if m:
        return "Yes" if m.group(1).lower() == "yes" else "No"
    return None


def load_task_questions(path: Path, dataset: str) -> dict[str, str]:
    """Load task->question mapping from either a .json or tasks.py file."""
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    # tasks.py: exec the file and pull out TASKS dict
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    tasks = ns.get("TASKS")
    if not tasks:
        raise ValueError(f"No TASKS dict found in {path}")
    return tasks


# ---------------------------------------------------------------------------
# Phase 1: Load rubricified data
# ---------------------------------------------------------------------------

def _stratified_sample(
    records: list[dict],
    n_per_label: int,
    seed: int,
) -> list[dict]:
    """Sample n_per_label records per binary label class from query split.

    Only the query (non-train) split records are stratified; train records
    are always returned in full so the retrieval index is complete.
    """
    groups: dict[int, list[dict]] = {}
    for r in records:
        lbl = label_to_int(r["label"])
        groups.setdefault(lbl, []).append(r)

    rng = random.Random(seed)
    chosen: list[dict] = []
    for lbl in sorted(groups):
        pool = groups[lbl]
        take = min(n_per_label, len(pool))
        if take < n_per_label:
            _log(
                f"  [WARN] Label={lbl}: only {len(pool)} records available "
                f"(requested {n_per_label}); taking all."
            )
        chosen.extend(rng.sample(pool, take))
    _log(
        f"  Stratified sample: "
        + ", ".join(f"label={k} n={min(n_per_label, len(v))}" for k, v in sorted(groups.items()))
        + f" → {len(chosen)} total"
    )
    return chosen


def load_rubricified(
    rubric_base: Path,
    query_splits: list[str],
    limit: Optional[int],
    tasks_filter: Optional[set[str]] = None,
    n_per_label: Optional[int] = None,
    seed: int = 42,
) -> dict[str, dict[str, list[dict]]]:
    """Returns {task: {split: [records]}}.

    query_splits controls which splits are loaded as query candidates;
    train is always loaded for retrieval.
    tasks_filter, when set, restricts which tasks are loaded.
    n_per_label, when set, stratifies each query split to n_per_label records
    per binary label class (50/50 by default). Train split is never subsampled.
    """
    import random as _random
    data: dict[str, dict[str, list[dict]]] = {}
    for task_dir in sorted(rubric_base.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        if tasks_filter and task not in tasks_filter:
            continue
        data[task] = {}
        for split_file in sorted(task_dir.iterdir()):
            split = split_file.stem
            with open(split_file) as f:
                recs = json.load(f)
            if split in query_splits or split == TRAIN_SPLIT:
                if split != TRAIN_SPLIT:
                    if n_per_label is not None:
                        recs = _stratified_sample(recs, n_per_label, seed)
                    elif limit is not None:
                        recs = recs[:limit]
                data[task][split] = recs

    n_tasks   = len(data)
    n_query   = sum(
        len(recs)
        for splits in data.values()
        for sp, recs in splits.items()
        if sp in query_splits
    )
    n_train   = sum(
        len(splits.get(TRAIN_SPLIT, []))
        for splits in data.values()
    )
    _log(f"Loaded {n_tasks} tasks | query records: {n_query:,} | train (retrieval): {n_train:,}")
    return data


# ---------------------------------------------------------------------------
# Phase 2: TF-IDF indices (only needed for vignette context)
# ---------------------------------------------------------------------------

class TaskIndex:
    """TF-IDF cosine similarity over one task's train rubricified texts."""

    def __init__(self, records: list[dict]) -> None:
        self._records = records
        texts = [r["rubricified_text"] for r in records]
        self._vec = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
        self._matrix = self._vec.fit_transform(texts)

    def search(
        self,
        query: str,
        top_k: int,
        exclude_pid: Optional[str] = None,
    ) -> list[dict]:
        q_vec = self._vec.transform([query])
        scores: np.ndarray = cosine_similarity(q_vec, self._matrix)[0]
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            rec = self._records[int(idx)]
            if exclude_pid and str(rec["patient_id"]) == str(exclude_pid):
                continue
            results.append({**rec, "score": float(scores[int(idx)])})
            if len(results) >= top_k:
                break
        return results


def build_indices(
    data: dict[str, dict[str, list[dict]]],
) -> dict[str, TaskIndex]:
    indices: dict[str, TaskIndex] = {}
    for task, splits in data.items():
        train = splits.get(TRAIN_SPLIT, [])
        if train:
            indices[task] = TaskIndex(train)
    _log(f"Built TF-IDF indices for {len(indices)} tasks")
    return indices


# ---------------------------------------------------------------------------
# Phase 3: Build batch requests
# ---------------------------------------------------------------------------

def build_requests(
    data: dict[str, dict[str, list[dict]]],
    indices: dict[str, TaskIndex],
    task_questions: dict[str, str],
    context: str,
    query_splits: list[str],
    top_k: int,
    example_prompt_path: Optional[Path] = None,
) -> list[dict]:
    is_fewshot = context == "vignette"
    requests = []

    for task, splits in sorted(data.items()):
        question = task_questions.get(task, f"Does the patient satisfy: {task}?")
        idx = indices.get(task) if is_fewshot else None

        for split in query_splits:
            for rec in splits.get(split, []):
                pid        = str(rec["patient_id"])
                query_text = rec["rubricified_text"]
                label_int  = label_to_int(rec["label"])

                query_block    = f"QUERY PATIENT INFORMATION:\n{query_text}\n"
                question_block = f"QUESTION:\n{question}\n"

                neighbor_blocks: list[str]   = []
                similar_pids:    list[str]   = []
                similar_labels:  list[str]   = []
                similar_scores:  list[float] = []

                if is_fewshot and idx is not None:
                    for sim in idx.search(query_text, top_k=top_k, exclude_pid=pid):
                        answer = label_to_yesno(sim["label"])
                        neighbor_blocks.append(
                            f"SIMILAR PATIENT INFORMATION:\n{sim['rubricified_text']}\nANSWER: {answer}"
                        )
                        similar_pids.append(str(sim["patient_id"]))
                        similar_labels.append(answer)
                        similar_scores.append(sim["score"])

                if neighbor_blocks:
                    prompt = (
                        f"{query_block}\n{question_block}\n"
                        + "\n\n".join(neighbor_blocks)
                    )
                else:
                    prompt = f"{query_block}\n{question_block}"

                if example_prompt_path is not None and not requests:
                    example_prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    example_prompt_path.write_text(
                        f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n"
                        f"=== USER PROMPT ===\n{prompt}\n",
                        encoding="utf-8",
                    )

                requests.append({
                    "request": {
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": 16,
                            # Disable thinking: Gemini 2.5 Flash thinking tokens
                            # count against maxOutputTokens — with budget=0 and a
                            # short cap, the model would exhaust all tokens on thinking
                            # and emit an empty response.
                            "thinkingConfig": {"thinkingBudget": 0},
                        },
                    },
                    "_meta": json.dumps({
                        "context":             context,
                        "patient_id":          pid,
                        "prediction_time":     rec.get("prediction_time", ""),
                        "task":                task,
                        "split":               split,
                        "label":               label_int,
                        "similar_patient_ids": similar_pids,
                        "similar_labels":      similar_labels,
                        "similar_scores":      similar_scores,
                        "top_k":               top_k if is_fewshot else None,
                    }),
                })

    n_query = sum(
        len(splits.get(sp, []))
        for splits in data.values()
        for sp in query_splits
    )
    _log(
        f"Built {len(requests):,} requests  "
        f"({n_query:,} query records × 1 context [{context}])"
    )
    return requests


# ---------------------------------------------------------------------------
# Dry-run validation (no API calls)
# ---------------------------------------------------------------------------

def dry_run_validate(
    data: dict[str, dict[str, list[dict]]],
    task_questions: dict[str, str],
    dataset: str,
    context: str,
    query_splits: list[str],
    top_k: int,
) -> None:
    _log("=== DRY RUN — validating data coverage (no API calls) ===")
    is_fewshot = context == "vignette"

    all_ok = True
    for task, splits in sorted(data.items()):
        q = task_questions.get(task)
        train_n = len(splits.get(TRAIN_SPLIT, []))
        query_counts = {sp: len(splits.get(sp, [])) for sp in query_splits}
        total_query  = sum(query_counts.values())

        has_question = q is not None
        has_train    = train_n > 0 if is_fewshot else True
        has_query    = total_query > 0
        ok = has_question and has_train and has_query

        status = "OK " if ok else "ERR"
        if not ok:
            all_ok = False
        parts = [f"train={train_n}"] + [f"{sp}={query_counts[sp]}" for sp in query_splits]
        _log(
            f"  [{status}] {task:<50s}  "
            + "  ".join(parts)
            + ("  [MISSING QUESTION]" if not has_question else "")
            + ("  [MISSING TRAIN DATA — can't retrieve]" if not has_train else "")
            + ("  [NO QUERY RECORDS]" if not has_query else "")
        )

    # tasks in question file but not in rubricified dir
    rubric_tasks = set(data.keys())
    question_tasks = set(task_questions.keys())
    extra = question_tasks - rubric_tasks
    if extra:
        _log(f"  [WARN] Tasks in question file with no rubricified data: {sorted(extra)}")

    if all_ok:
        _log("All tasks validated OK")
    else:
        _log("ERROR: Some tasks failed validation — fix before running the full job")
        sys.exit(1)

    # Print sample prompt for first task/record
    first_task = next(iter(sorted(data.keys())))
    first_split = query_splits[0]
    recs = data[first_task].get(first_split, [])
    if recs:
        rec = recs[0]
        q = task_questions.get(first_task, "?")
        _log(f"\nSample prompt ({first_task} / {first_split} / patient {rec['patient_id']}):")
        sample = (
            f"QUERY PATIENT INFORMATION:\n{rec['rubricified_text'][:300]}...\n\n"
            f"QUESTION:\n{q}\n"
        )
        print(sample)
        if is_fewshot:
            _log("  (fewshot) — similar-patient blocks would follow in real run")

    total_requests = sum(
        len(splits.get(sp, []))
        for splits in data.values()
        for sp in query_splits
    )
    _log(f"\nDry run complete. Would submit {total_requests:,} requests to Vertex AI.")


# ---------------------------------------------------------------------------
# Phase 4: Upload input JSONL to GCS
# ---------------------------------------------------------------------------

def upload_input(requests: list[dict], dataset: str, run_id: str) -> str:
    client = storage.Client()
    blob_path = f"{GCS_PREFIX}/input/{dataset}_{run_id}.jsonl"
    blob = client.bucket(GCS_BUCKET).blob(blob_path)
    content = "\n".join(json.dumps(r, default=str) for r in requests)
    blob.upload_from_string(content.encode(), content_type="application/jsonl")
    uri = f"gs://{GCS_BUCKET}/{blob_path}"
    _log(f"Uploaded {len(requests):,} lines → {uri}")
    return uri


# ---------------------------------------------------------------------------
# Phase 5 & 6: Submit and wait
# ---------------------------------------------------------------------------

def submit_job(
    input_uri: str,
    output_prefix: str,
    job_name: str,
) -> BatchPredictionJob:
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    job = BatchPredictionJob.submit(
        source_model=VERTEX_MODEL,
        input_dataset=input_uri,
        output_uri_prefix=output_prefix,
        job_display_name=job_name,
    )
    _log(f"Submitted: {job.resource_name}")
    console_url = (
        f"https://console.cloud.google.com/ai/platform/locations/"
        f"{VERTEX_LOCATION}/batch-predictions/"
        f"{job.resource_name.split('/')[-1]}?project={VERTEX_PROJECT}"
    )
    _log(f"Console: {console_url}")
    return job


def wait_for_job(job: BatchPredictionJob) -> str:
    _log(f"Polling every {POLL_INTERVAL}s …")
    while not job.has_ended:
        time.sleep(POLL_INTERVAL)
        job.refresh()
        _log(f"  State: {job.state}")
    if not job.has_succeeded:
        raise RuntimeError(f"Batch job failed: {job.error}")
    output_location = job.output_location
    _log(f"Job complete. Output: {output_location}")
    return output_location


# ---------------------------------------------------------------------------
# Phase 7: Parse output → write results
# ---------------------------------------------------------------------------

def parse_and_write(
    output_gcs: str,
    dataset: str,
    context: str,
    run_id: str,
    out_dir: Path,
) -> dict:
    bucket_name = output_gcs.replace("gs://", "").split("/")[0]
    prefix      = "/".join(output_gcs.replace("gs://", "").split("/")[1:])

    gcs   = storage.Client()
    blobs = [
        b for b in gcs.bucket(bucket_name).list_blobs(prefix=prefix)
        if b.name.endswith(".jsonl")
    ]
    _log(f"Parsing {len(blobs)} output shard(s)")

    records: list[dict] = []
    n_api_errors    = 0  # non-empty status field
    n_empty_content = 0  # finishReason=MAX_TOKENS with no text parts
    n_no_candidates = 0  # response had no candidates at all

    for blob in blobs:
        for line in blob.download_as_text().strip().split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)

            # API-level failure
            if obj.get("status") not in ("", None):
                n_api_errors += 1
                continue

            meta = obj.get("_meta", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    n_api_errors += 1
                    continue

            label_int  = meta.get("label", -1)
            true_yesno = label_to_yesno(label_int)

            # Extract model text — handle thinking-mode empty-parts gracefully
            candidates = obj.get("response", {}).get("candidates", [])
            if not candidates:
                n_no_candidates += 1
                raw  = None
                pred = None
            else:
                cand  = candidates[0]
                parts = cand.get("content", {}).get("parts", [])
                if parts:
                    raw = parts[0].get("text")
                else:
                    # finishReason=MAX_TOKENS with empty content — thinking ate all tokens
                    raw = None
                    n_empty_content += 1
                pred = parse_yes_no(raw)

            correct = pred is not None and pred == true_yesno

            records.append({
                "context":             context,
                "dataset":             dataset,
                "person_id":           meta.get("patient_id"),
                "embed_time":          meta.get("prediction_time"),
                "task":                meta.get("task"),
                "split":               meta.get("split"),
                "label":               label_int,
                "true_yes_no":         true_yesno,
                "pred":                pred,
                "correct":             correct,
                "raw":                 raw,
                "similar_patient_ids": meta.get("similar_patient_ids", []),
                "similar_labels":      meta.get("similar_labels", []),
                "similar_scores":      meta.get("similar_scores", []),
                "top_k":               meta.get("top_k"),
                "model":               VERTEX_MODEL,
                "temperature":         0.0,
                "run_id":              run_id,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"experiment_results_{context}_{run_id}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_parsed  = sum(1 for r in records if r["pred"] is not None)
    n_correct = sum(1 for r in records if r["correct"])
    acc = n_correct / n_parsed * 100 if n_parsed else 0.0

    _log(
        f"  {dataset}/{context}: {len(records):,} rows  |  "
        f"acc={acc:.1f}%  (parsed={n_parsed}/{len(records)})  →  {out_path}"
    )
    if n_api_errors or n_empty_content or n_no_candidates:
        _log(
            f"  Issues: api_errors={n_api_errors}  "
            f"empty_content(thinking)={n_empty_content}  "
            f"no_candidates={n_no_candidates}"
        )

    summary = {
        "dataset":       dataset,
        "context":       context,
        "run_id":        run_id,
        "n_rows":        len(records),
        "n_parsed":      n_parsed,
        "n_correct":     n_correct,
        "accuracy":      round(acc, 2),
        "n_api_errors":  n_api_errors,
        "n_empty_content": n_empty_content,
        "n_no_candidates": n_no_candidates,
    }
    meta_path = out_dir / f"experiment_meta_{context}_{run_id}.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"Meta written: {meta_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",  required=True, choices=list(DATASET_CONFIGS.keys()),
                   help="Which dataset to run (ehrshot or vista)")
    p.add_argument("--context",  required=True, choices=["baseline_vignette", "vignette"],
                   help="baseline_vignette = no similar patients; vignette = top-k few-shot")
    p.add_argument("--query-splits", default="both", choices=["val", "test", "both"],
                   help="Which splits to use as query set (default: both = val + test)")
    p.add_argument("--top-k",    type=int, default=TOP_K_DEFAULT,
                   help="Number of similar patients to inject (vignette context only)")
    p.add_argument("--limit",    type=int, default=None,
                   help="Limit query records per task per split (for smoke tests)")
    p.add_argument("--tasks", nargs="+", default=None, metavar="TASK",
                   help="Only run these tasks (e.g. guo_readmission). Default: all tasks.")
    p.add_argument("--n-per-label", type=int, default=None, metavar="N",
                   help=(
                       "Stratified sampling: take N records per binary label class "
                       "from each query split. Only test/val splits are subsampled; "
                       "train is always used in full for retrieval."
                   ))
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for stratified sampling (default: 42)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Validate data coverage and print sample prompt; do not call Vertex AI")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    query_splits: list[str] = (
        ["val", "test"] if args.query_splits == "both"
        else [args.query_splits]
    )

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"{args.dataset}_{args.context}_{run_id}"
    cfg      = DATASET_CONFIGS[args.dataset]
    out_dir  = OUTPUT_DIR / args.dataset

    _log(f"=== Batch inference: dataset={args.dataset}  context={args.context}  run_id={run_id} ===")
    _log(f"Query splits: {query_splits}  |  top_k={args.top_k if args.context == 'vignette' else 'N/A'}")
    _log(f"Model: {VERTEX_MODEL}  |  thinkingBudget=0  |  maxOutputTokens=16")

    tasks_filter = set(args.tasks) if args.tasks else None
    if tasks_filter:
        _log(f"Task filter: {sorted(tasks_filter)}")
    if args.n_per_label:
        _log(f"Stratified sampling: {args.n_per_label} per label class from query splits (train kept in full)")

    _log("=== Phase 1: Loading rubricified data ===")
    data = load_rubricified(
        cfg["rubric_base"],
        query_splits,
        args.limit,
        tasks_filter=tasks_filter,
        n_per_label=args.n_per_label,
        seed=args.seed,
    )

    _log("=== Loading task questions ===")
    task_questions = load_task_questions(cfg["task_questions"], args.dataset)
    _log(f"  {len(task_questions)} task questions loaded")

    if args.dry_run:
        dry_run_validate(data, task_questions, args.dataset, args.context, query_splits, args.top_k)
        return

    indices: dict[str, TaskIndex] = {}
    if args.context == "vignette":
        _log("=== Phase 2: Building TF-IDF retrieval indices ===")
        indices = build_indices(data)

    _log("=== Phase 3: Building prompts ===")
    example_prompt_path = out_dir / f"example_prompt_{args.context}.txt"
    requests = build_requests(
        data, indices, task_questions, args.context, query_splits, args.top_k,
        example_prompt_path=example_prompt_path,
    )

    _log("=== Phase 4: Uploading input JSONL to GCS ===")
    input_uri = upload_input(requests, job_name, run_id)

    _log("=== Phase 5: Submitting Vertex AI batch job ===")
    output_prefix = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/output/{job_name}"
    job = submit_job(input_uri, output_prefix, job_name)

    _log("=== Phase 6: Waiting for completion ===")
    output_location = wait_for_job(job)

    _log("=== Phase 7: Parsing output and writing results ===")
    parse_and_write(output_location, args.dataset, args.context, run_id, out_dir)

    _log("=== Done ===")


if __name__ == "__main__":
    main()
