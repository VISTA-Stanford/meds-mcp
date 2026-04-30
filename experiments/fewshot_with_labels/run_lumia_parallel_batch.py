#!/usr/bin/env python3
"""
Submit 4 parallel Vertex AI batch jobs for the LUMIA context-length token sweep.
Token caps: 4K / 8K / 16K / 32K  →  char caps: 16384 / 32768 / 65536 / 131072.
Each token bucket gets its own GCS input/output path so jobs never collide.
All 4 jobs are submitted simultaneously; we poll them together and download
results as each one finishes.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
OUT_DIR      = Path("/home/Ayeeshi/meds-mcp/experiments/fewshot_with_labels/outputs/ehrshot")
CORPUS_DIR   = Path("/home/Ayeeshi/meds-mcp/data/ehrshot_lumia/meds_corpus")
PATIENTS     = OUT_DIR / "patients.jsonl"
ITEMS_FILE   = OUT_DIR / "items.jsonl"
POOL_FILE    = OUT_DIR / "pool_test_100.json"
MODEL        = "gemini-2.5-flash"
PROJECT      = "som-nero-plevriti-deidbdf"
LOCATION     = "us-central1"
GCS_BASE     = "gs://vista_bench/temp/pinnacle_templated_summaries"
TASK_FILTER  = {"guo_readmission"}
QUERY_SPLIT  = "test"

# token-cap → char-cap mapping
TOKEN_CAPS = [4096, 8192, 16384, 32768]
CHAR_CAPS  = {t: t * 4 for t in TOKEN_CAPS}   # 16384, 32768, 65536, 131072

CONTEXT_NAMES = {t: f"baseline_lumia_{t // 1024}kt" for t in TOKEN_CAPS}
# e.g. baseline_lumia_4kt, baseline_lumia_8kt, baseline_lumia_16kt, baseline_lumia_32kt


# ── XML helpers ───────────────────────────────────────────────────────────────
def _tree_bytes(tree: ET.ElementTree) -> bytes:
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def get_filtered_xml(pid: str, embed_time_str: str, max_chars: int) -> str:
    cutoff = (datetime.fromisoformat(embed_time_str) + timedelta(days=1)).strftime("%Y-%m-%d")
    xml_path = CORPUS_DIR / f"{pid}.xml"
    tree = filter_xml_by_date(str(xml_path), cutoff)
    root = tree.getroot()
    encounters = root.findall("encounter")
    while encounters:
        if len(_tree_bytes(tree)) <= max_chars:
            break
        root.remove(encounters.pop(0))
    ET.indent(tree)
    return _tree_bytes(tree).decode("utf-8")


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    s = uri.replace("gs://", "", 1)
    b, _, p = s.partition("/")
    return b, p


# ── build requests for one context length ────────────────────────────────────
def build_requests(token_cap: int, store: CohortStore, pool_ids: list[str]) -> list[dict]:
    context_name = CONTEXT_NAMES[token_cap]
    max_chars = CHAR_CAPS[token_cap]
    task_desc = TASK_DESCRIPTIONS.get("guo_readmission", "")

    requests: list[dict] = []
    example_saved = False

    for pid in pool_ids:
        items = [
            it for it in store.items_for_patient(pid)
            if it.label != -1 and it.task in TASK_FILTER
        ]
        for item in items:
            state = store.get_or_none(pid, item.embed_time)
            if state is None or state.split != QUERY_SPLIT:
                continue
            xml_path = CORPUS_DIR / f"{pid}.xml"
            if not xml_path.exists():
                logger.warning("Missing XML for %s", pid)
                continue

            prompt = (
                f"TASK:\n{task_desc}\n\n"
                f"PATIENT TIMELINE:\n"
                f"{get_filtered_xml(pid, item.embed_time, max_chars)}\n"
            )

            if not example_saved:
                out_path = OUT_DIR / f"example_prompt_{context_name}.txt"
                out_path.write_text(
                    f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n"
                    f"=== USER PROMPT ===\n{prompt}\n",
                    encoding="utf-8",
                )
                example_saved = True
                logger.info("[%s] Saved example prompt", context_name)

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
                    "top_k": None,
                    "similar_patient_ids": [],
                    "similar_labels": [],
                    "similar_scores": [],
                    "max_chars_per_patient": max_chars,
                    "token_cap": token_cap,
                }),
            })

    logger.info("[%s] Built %d requests (max_chars=%d)", context_name, len(requests), max_chars)
    return requests


# ── one worker: upload + submit + poll + download ─────────────────────────────
def run_one(token_cap: int, store: CohortStore, pool_ids: list[str]) -> str:
    from google.cloud import storage
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob

    context_name = CONTEXT_NAMES[token_cap]
    gcs_input  = f"{GCS_BASE}/input/ehrshot_lumia_{token_cap // 1024}kt_batch.jsonl"
    gcs_output = f"{GCS_BASE}/output/ehrshot_lumia_{token_cap // 1024}kt_batch"

    requests = build_requests(token_cap, store, pool_ids)

    # upload
    in_bucket, in_path = _parse_gcs_uri(gcs_input)
    storage.Client().bucket(in_bucket).blob(in_path).upload_from_string(
        "\n".join(json.dumps(r) for r in requests).encode("utf-8"),
        content_type="application/jsonl",
    )
    logger.info("[%s] Uploaded %d requests to %s", context_name, len(requests), gcs_input)

    # submit
    vertexai.init(project=PROJECT, location=LOCATION)
    job = BatchPredictionJob.submit(
        source_model=MODEL,
        input_dataset=gcs_input,
        output_uri_prefix=gcs_output,
        job_display_name=f"lumia_{context_name}_{int(time.time())}",
    )
    logger.info("[%s] Submitted job: %s", context_name, job.resource_name)

    # poll
    while not job.has_ended:
        time.sleep(60)
        job.refresh()
        logger.info("[%s] state=%s", context_name, job.state)

    if not job.has_succeeded:
        raise RuntimeError(f"[{context_name}] Job failed: {job.error}")

    # download
    out_bucket, out_prefix = _parse_gcs_uri(job.output_location)
    storage_client = storage.Client()
    blobs = [
        b for b in storage_client.bucket(out_bucket).list_blobs(prefix=out_prefix)
        if b.name.endswith(".jsonl")
    ]

    out_path = OUT_DIR / f"experiment_results_{context_name}.jsonl"
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
                    "query_split": QUERY_SPLIT,
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
                    "similar_patient_ids": [],
                    "similar_labels": [],
                    "similar_scores": [],
                    "top_k": None,
                    "max_chars_per_patient": meta.get("max_chars_per_patient"),
                    "token_cap": meta.get("token_cap"),
                    "model": MODEL,
                    "temperature": 0.0,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows += 1

    logger.info("[%s] Wrote %d rows to %s", context_name, rows, out_path)
    return str(out_path)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    store = CohortStore.load(PATIENTS, ITEMS_FILE)
    pool_ids = [str(x) for x in json.load(open(POOL_FILE))]

    logger.info("Submitting 4 parallel Vertex batch jobs: %s",
                " | ".join(CONTEXT_NAMES.values()))

    results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_one, t, store, pool_ids): t for t in TOKEN_CAPS}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                path = fut.result()
                results[t] = path
                logger.info("DONE [%s] → %s", CONTEXT_NAMES[t], path)
            except Exception as e:
                logger.error("FAILED [%s]: %s", CONTEXT_NAMES[t], e)

    if len(results) == len(TOKEN_CAPS):
        logger.info("All 4 jobs succeeded. Running analyze_results...")
        import subprocess
        subprocess.run(
            ["uv", "run", "python",
             "experiments/fewshot_with_labels/analyze_results.py",
             "--input-dir", str(OUT_DIR),
             "--output-dir", str(OUT_DIR)],
            cwd=str(_REPO_ROOT), check=True,
        )
        logger.info("Done. Results in %s", OUT_DIR)
    else:
        logger.warning("Only %d/%d jobs succeeded", len(results), len(TOKEN_CAPS))


if __name__ == "__main__":
    main()
