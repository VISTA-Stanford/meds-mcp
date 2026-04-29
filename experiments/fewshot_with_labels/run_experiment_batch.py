#!/usr/bin/env python3
"""
Batch-inference runner for the fewshot_with_labels experiment.

Reads rubricified texts from LRRL_MEDS/data/vista_rubric/rubricified/ (already
generated), builds prompts for two contexts, submits them to a single Vertex AI
batch prediction job, then parses the output into experiment_results_*.jsonl.

Contexts run in one job:
  baseline_vignette -- query rubricified text + question only (no similar patients)
  vignette          -- same, plus top-3 similar train patients (TF-IDF cosine retrieval
                       on rubricified texts for the same task)

GCS output lands under:  gs://vista_bench/temp/pinnacle_templated_summaries/
Local results land in:   experiments/fewshot_with_labels/outputs/
"""

from __future__ import annotations

import json
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
# Config
# ---------------------------------------------------------------------------

RUBRIC_BASE    = Path.home() / "LRRL_MEDS/data/vista_rubric/rubricified"
TASK_QUESTIONS = Path.home() / "LRRL_MEDS/config/vista_tasks.json"
OUTPUT_DIR     = Path.home() / "meds-mcp/experiments/fewshot_with_labels/outputs"
LOG_DIR        = OUTPUT_DIR  # log written here alongside results

GCS_BUCKET     = "vista_bench"
GCS_PREFIX     = "temp/pinnacle_templated_summaries"

VERTEX_PROJECT  = "som-nero-plevriti-deidbdf"
VERTEX_LOCATION = "us-central1"
VERTEX_MODEL    = "gemini-2.5-flash"   # short name — no 'google/' prefix

TOP_K          = 3
QUERY_SPLIT    = "val"
TRAIN_SPLIT    = "train"
CONTEXTS       = ["baseline_vignette", "vignette"]
POLL_INTERVAL  = 60   # seconds

SYSTEM_PROMPT = (
    "You are a clinical-prediction assistant. Answer the user's question about the query "
    "patient using only the evidence in the user message. Respond with exactly one word "
    "and nothing else: Yes or No. Do not include punctuation, reasoning, or any other text."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def label_to_int(label_val) -> int:
    """'True'/'False' strings or bool/int → 0/1."""
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


# ---------------------------------------------------------------------------
# Phase 1: Load rubricified data
# ---------------------------------------------------------------------------

def load_rubricified() -> dict[str, dict[str, list[dict]]]:
    """Returns {task: {split: [records]}}."""
    data: dict[str, dict[str, list[dict]]] = {}
    for task_dir in sorted(RUBRIC_BASE.iterdir()):
        task = task_dir.name
        data[task] = {}
        for split_file in sorted(task_dir.iterdir()):
            split = split_file.stem
            with open(split_file) as f:
                data[task][split] = json.load(f)
    total = sum(len(recs) for splits in data.values() for recs in splits.values())
    _log(f"Loaded {len(data)} tasks, {total:,} total records")
    return data


# ---------------------------------------------------------------------------
# Phase 2: Build per-task TF-IDF retrieval indices
# ---------------------------------------------------------------------------

class TaskIndex:
    """TF-IDF cosine similarity index over one task's train rubricified texts."""

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
            if exclude_pid and rec["patient_id"] == exclude_pid:
                continue
            results.append({**rec, "score": float(scores[int(idx)])})
            if len(results) >= top_k:
                break
        return results


def build_indices(data: dict[str, dict[str, list[dict]]]) -> dict[str, TaskIndex]:
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
) -> list[dict]:
    requests = []

    for task, splits in sorted(data.items()):
        question = task_questions.get(task, f"Does the patient satisfy: {task}?")
        val_records = splits.get(QUERY_SPLIT, [])
        idx = indices.get(task)

        for rec in val_records:
            pid           = rec["patient_id"]
            query_text    = rec["rubricified_text"]
            label_int     = label_to_int(rec["label"])

            for context in CONTEXTS:
                query_block    = f"QUERY PATIENT INFORMATION:\n{query_text}\n"
                question_block = f"QUESTION:\n{question}\n"

                neighbor_blocks: list[str] = []
                similar_pids: list[str]    = []
                similar_labels: list[str]  = []
                similar_scores: list[float] = []

                if context == "vignette" and idx is not None:
                    for sim in idx.search(query_text, top_k=TOP_K, exclude_pid=pid):
                        answer = label_to_yesno(sim["label"])
                        neighbor_blocks.append(
                            f"SIMILAR PATIENT INFORMATION:\n{sim['rubricified_text']}\nANSWER: {answer}"
                        )
                        similar_pids.append(sim["patient_id"])
                        similar_labels.append(answer)
                        similar_scores.append(sim["score"])

                if neighbor_blocks:
                    prompt = (
                        f"{query_block}\n{question_block}\n"
                        + "\n\n".join(neighbor_blocks)
                    )
                else:
                    prompt = f"{query_block}\n{question_block}"

                requests.append({
                    "request": {
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": 8,
                        },
                    },
                    # Vertex batch only allows scalar top-level fields → serialize _meta as str
                    "_meta": json.dumps({
                        "context":             context,
                        "patient_id":          pid,
                        "prediction_time":     rec["prediction_time"],
                        "task":                task,
                        "split":               QUERY_SPLIT,
                        "label":               label_int,
                        "label_str":           str(rec["label"]),
                        "similar_patient_ids": similar_pids,
                        "similar_labels":      similar_labels,
                        "similar_scores":      similar_scores,
                        "top_k":               TOP_K if context == "vignette" else None,
                    }),
                })

    val_total = sum(len(s.get(QUERY_SPLIT, [])) for s in data.values())
    _log(f"Built {len(requests):,} requests  ({val_total:,} val records × {len(CONTEXTS)} contexts)")
    return requests


# ---------------------------------------------------------------------------
# Phase 4: Upload input JSONL to GCS
# ---------------------------------------------------------------------------

def upload_input(requests: list[dict], run_id: str) -> str:
    client = storage.Client()
    blob_path = f"{GCS_PREFIX}/input/fewshot_vignette_{run_id}.jsonl"
    blob = client.bucket(GCS_BUCKET).blob(blob_path)
    content = "\n".join(json.dumps(r, default=str) for r in requests)
    blob.upload_from_string(content.encode(), content_type="application/jsonl")
    uri = f"gs://{GCS_BUCKET}/{blob_path}"
    _log(f"Uploaded {len(requests):,} lines → {uri}")
    return uri


# ---------------------------------------------------------------------------
# Phase 5: Submit BatchPredictionJob
# ---------------------------------------------------------------------------

def submit_job(input_uri: str, output_prefix: str, run_id: str) -> BatchPredictionJob:
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    job = BatchPredictionJob.submit(
        source_model=VERTEX_MODEL,
        input_dataset=input_uri,
        output_uri_prefix=output_prefix,
        job_display_name=f"fewshot_vignette_{run_id}",
    )
    _log(f"Submitted: {job.resource_name}")
    _log(f"State: {job.state}")
    console_url = (
        f"https://console.cloud.google.com/ai/platform/locations/"
        f"{VERTEX_LOCATION}/batch-predictions/"
        f"{job.resource_name.split('/')[-1]}?project={VERTEX_PROJECT}"
    )
    _log(f"Console: {console_url}")
    return job


# ---------------------------------------------------------------------------
# Phase 6: Poll until complete
# ---------------------------------------------------------------------------

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
# Phase 7: Parse output → write experiment_results_{context}.jsonl
# ---------------------------------------------------------------------------

def parse_and_write(output_gcs: str, run_id: str) -> None:
    bucket_name = output_gcs.replace("gs://", "").split("/")[0]
    prefix      = "/".join(output_gcs.replace("gs://", "").split("/")[1:])

    gcs   = storage.Client()
    blobs = [b for b in gcs.bucket(bucket_name).list_blobs(prefix=prefix)
             if b.name.endswith(".jsonl")]
    _log(f"Parsing {len(blobs)} output shard(s)")

    buffers: dict[str, list[dict]] = defaultdict(list)
    failed = 0

    for blob in blobs:
        for line in blob.download_as_text().strip().split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)
            meta = obj.get("_meta", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    failed += 1
                    continue

            if obj.get("status") not in ("", None):
                failed += 1
                continue

            try:
                raw = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                failed += 1
                continue

            label_int   = meta.get("label", -1)
            true_yesno  = label_to_yesno(label_int)
            pred        = parse_yes_no(raw)
            correct     = pred is not None and pred == true_yesno
            context     = meta.get("context", "unknown")

            buffers[context].append({
                "context":             context,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context_summary = {}
    for context, records in sorted(buffers.items()):
        out_path = OUTPUT_DIR / f"experiment_results_{context}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_valid   = sum(1 for r in records if r["pred"] is not None)
        n_correct = sum(1 for r in records if r["correct"])
        acc = n_correct / n_valid * 100 if n_valid else 0
        _log(
            f"  {context}: {len(records)} rows  |  acc={acc:.1f}%  "
            f"(parsed={n_valid}/{len(records)})  →  {out_path}"
        )
        context_summary[context] = {
            "n_rows": len(records),
            "n_parsed": n_valid,
            "n_correct": n_correct,
            "accuracy": round(acc, 2),
        }

    if failed:
        _log(f"  WARNING: {failed} failed/unparseable rows")

    meta_path = OUTPUT_DIR / f"experiment_meta_batch_{run_id}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "run_id":          run_id,
            "contexts":        CONTEXTS,
            "top_k":           TOP_K,
            "model":           VERTEX_MODEL,
            "query_split":     QUERY_SPLIT,
            "candidate_split": TRAIN_SPLIT,
            "rubric_base":     str(RUBRIC_BASE),
            "n_failed":        failed,
            "results":         context_summary,
        }, f, indent=2)
    _log(f"Meta written: {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log(f"=== fewshot_with_labels batch experiment  |  run_id={run_id} ===")
    _log(f"Contexts: {CONTEXTS}  |  top_k={TOP_K}  |  model={VERTEX_MODEL}")

    _log("=== Phase 1: Loading rubricified data ===")
    data = load_rubricified()

    with open(TASK_QUESTIONS) as f:
        task_questions: dict[str, str] = json.load(f)

    _log("=== Phase 2: Building retrieval indices ===")
    indices = build_indices(data)

    _log("=== Phase 3: Building prompts ===")
    requests = build_requests(data, indices, task_questions)

    _log("=== Phase 4: Uploading to GCS ===")
    input_uri = upload_input(requests, run_id)

    _log("=== Phase 5: Submitting batch job ===")
    output_prefix = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/output/fewshot_{run_id}"
    job = submit_job(input_uri, output_prefix, run_id)

    _log("=== Phase 6: Waiting for completion ===")
    output_location = wait_for_job(job)

    _log("=== Phase 7: Parsing output and writing results ===")
    parse_and_write(output_location, run_id)

    _log("=== Done ===")


if __name__ == "__main__":
    main()
