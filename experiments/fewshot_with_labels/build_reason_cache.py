#!/usr/bin/env python3
"""
Generate and cache few-shot example reasons for (person_id, embed_time, task).

Supports:
  - online inference (secure-llm client)
  - Vertex AI batch inference (Gemini)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.server.llm import extract_response_content, get_default_generation_config, get_llm_client
from meds_mcp.similarity import CohortStore
from experiments.fewshot_with_labels import _paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT = (
    "You write constrained rationales for binary clinical examples. "
    "Use only explicit evidence from the provided summary. "
    "Do not speculate, do not use outside knowledge, and do not contradict the provided gold label. "
    "Return strict JSON only."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key_for(person_id: str, embed_time: str, task: str) -> str:
    return f"{person_id}|{embed_time}|{task}"


@dataclass(frozen=True)
class ReasonRow:
    person_id: str
    embed_time: str
    task: str
    reason: str
    label: int
    label_yes_no: str
    model: str
    created_at: str


def label_to_yes_no(label: int) -> str:
    return "Yes" if int(label) == 1 else "No"


def normalize_reason(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "Evidence is limited in the provided summary."
    # Preferred path: strict JSON payload.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            raw_reason = str(obj.get("reason", "")).strip()
            if raw_reason:
                return _clamp_reason(raw_reason)
    except Exception:
        pass
    # Backward-compatible fallback.
    if s.lower().startswith("reason:"):
        s = s.split(":", 1)[1].strip()
    return _clamp_reason(s)


def _clamp_reason(reason: str, max_words: int = 45) -> str:
    text = " ".join(reason.split())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Evidence is limited in the provided summary."
    words = text.split(" ")
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:")
        if not text.endswith("."):
            text += "."
    return text


def build_reason_user_prompt(vignette: str, question: str, gold_answer: str) -> str:
    return (
        "Given:\n"
        "1) Patient summary\n"
        "2) Task question\n"
        "3) Gold answer\n\n"
        "Task: write a short rationale (1-2 sentences, <=45 words) that supports the gold answer.\n"
        "Hard constraints:\n"
        "- Use only facts explicitly present in the summary.\n"
        "- Do not invent values, diagnoses, staging, or treatments.\n"
        "- Do not hedge with possibilities.\n"
        "- If evidence is sparse, say that evidence is limited.\n"
        "- Output MUST be valid JSON with exactly one key: {\"reason\": \"...\"}\n\n"
        f"PATIENT SUMMARY:\n{vignette}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWER: {gold_answer}\n"
    )


def load_existing(path: Path) -> dict[str, ReasonRow]:
    out: dict[str, ReasonRow] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                row = ReasonRow(
                    person_id=str(obj["person_id"]),
                    embed_time=str(obj["embed_time"]),
                    task=str(obj["task"]),
                    reason=str(obj["reason"]),
                    label=int(obj.get("label", 0)),
                    label_yes_no=str(obj.get("label_yes_no", "")),
                    model=str(obj.get("model", "")),
                    created_at=str(obj.get("created_at", "")),
                )
                out[key_for(row.person_id, row.embed_time, row.task)] = row
            except Exception:
                continue
    return out


def write_rows(path: Path, rows: list[ReasonRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def collect_todo(
    store: CohortStore,
    existing: dict[str, ReasonRow],
    split: str,
    force: bool,
) -> list[tuple[str, str, str, str, int]]:
    todo: list[tuple[str, str, str, str, int]] = []
    for item in store.items():
        if item.split != split or item.label == -1:
            continue
        state = store.get_or_none(item.person_id, item.embed_time)
        if state is None or not state.vignette.strip():
            continue
        k = key_for(item.person_id, item.embed_time, item.task)
        if (not force) and (k in existing) and existing[k].reason.strip():
            continue
        todo.append((item.person_id, item.embed_time, item.task, item.question, int(item.label)))
    return todo


def run_online(
    store: CohortStore,
    todo: list[tuple[str, str, str, str, int]],
    existing: dict[str, ReasonRow],
    output: Path,
    model: str,
    delay_seconds: float,
) -> None:
    client = get_llm_client(model)
    for idx, (pid, et, task, question, label) in enumerate(todo, start=1):
        state = store.get_or_none(pid, et)
        if state is None:
            continue
        gold = label_to_yes_no(label)
        user_prompt = build_reason_user_prompt(state.vignette, question, gold)
        raw = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            **get_default_generation_config({"temperature": 0.0, "max_tokens": 128}),
        )
        reason = normalize_reason(extract_response_content(raw))
        existing[key_for(pid, et, task)] = ReasonRow(
            person_id=pid,
            embed_time=et,
            task=task,
            reason=reason,
            label=label,
            label_yes_no=gold,
            model=model,
            created_at=utc_now_iso(),
        )
        if idx % 50 == 0:
            logger.info("Generated %d/%d reasons", idx, len(todo))
            write_rows(output, list(existing.values()))
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    write_rows(output, list(existing.values()))


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    s = uri.replace("gs://", "", 1)
    bucket, _, prefix = s.partition("/")
    return bucket, prefix


def run_vertex_batch(
    store: CohortStore,
    todo: list[tuple[str, str, str, str, int]],
    existing: dict[str, ReasonRow],
    output: Path,
    model: str,
    project: str,
    location: str,
    input_uri: str,
    output_uri_prefix: str,
) -> None:
    from google.cloud import storage
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob

    requests: list[dict[str, Any]] = []
    for pid, et, task, question, label in todo:
        state = store.get_or_none(pid, et)
        if state is None:
            continue
        gold = label_to_yes_no(label)
        user_prompt = build_reason_user_prompt(state.vignette, question, gold)
        requests.append(
            {
                "request": {
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 128,
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
                "_meta": json.dumps(
                    {
                        "person_id": pid,
                        "embed_time": et,
                        "task": task,
                        "label": int(label),
                    }
                ),
            }
        )

    in_bucket, in_path = _parse_gcs_uri(input_uri)
    storage_client = storage.Client()
    storage_client.bucket(in_bucket).blob(in_path).upload_from_string(
        "\n".join(json.dumps(r) for r in requests).encode("utf-8"),
        content_type="application/jsonl",
    )
    logger.info("Uploaded %d reason requests to %s", len(requests), input_uri)

    vertexai.init(project=project, location=location)
    job = BatchPredictionJob.submit(
        source_model=model,
        input_dataset=input_uri,
        output_uri_prefix=output_uri_prefix,
        job_display_name=f"fewshot_reason_cache_{int(time.time())}",
    )
    logger.info("Submitted vertex batch job: %s", job.resource_name)
    while not job.has_ended:
        time.sleep(30)
        job.refresh()
        logger.info("Batch state: %s", job.state)
    if not job.has_succeeded:
        raise RuntimeError(f"Reason-cache batch job failed: {job.error}")

    out_bucket, out_prefix = _parse_gcs_uri(job.output_location)
    blobs = [
        b for b in storage_client.bucket(out_bucket).list_blobs(prefix=out_prefix)
        if b.name.endswith(".jsonl")
    ]
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
                continue
            pid = str(meta["person_id"])
            et = str(meta["embed_time"])
            task = str(meta["task"])
            label = int(meta.get("label", 0))
            existing[key_for(pid, et, task)] = ReasonRow(
                person_id=pid,
                embed_time=et,
                task=task,
                reason=normalize_reason(raw),
                label=label,
                label_yes_no=label_to_yes_no(label),
                model=model,
                created_at=utc_now_iso(),
            )
    write_rows(output, list(existing.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached reasons for few-shot examples.")
    parser.add_argument("--patients", type=Path, default=_paths.patients_jsonl())
    parser.add_argument("--items", type=Path, default=_paths.items_jsonl())
    parser.add_argument("--output", type=Path, default=_paths.outputs_dir() / "reason_cache.jsonl")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--mode", choices=("online", "vertex_batch"), default="online")
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    parser.add_argument("--vertex-project", type=str, default="som-nero-plevriti-deidbdf")
    parser.add_argument("--vertex-location", type=str, default="us-central1")
    parser.add_argument(
        "--vertex-input-uri",
        type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/input/reason_cache.jsonl",
    )
    parser.add_argument(
        "--vertex-output-prefix",
        type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/output/reason_cache",
    )
    args = parser.parse_args()

    store = CohortStore.load(args.patients, args.items)
    existing = load_existing(args.output)
    todo = collect_todo(store, existing, split=args.split, force=args.force)
    logger.info("Reason-cache rows to generate: %d", len(todo))
    if not todo:
        logger.info("Nothing to do.")
        return
    if args.mode == "online":
        run_online(
            store=store,
            todo=todo,
            existing=existing,
            output=args.output,
            model=args.model,
            delay_seconds=args.delay_seconds,
        )
    else:
        run_vertex_batch(
            store=store,
            todo=todo,
            existing=existing,
            output=args.output,
            model=args.model,
            project=args.vertex_project,
            location=args.vertex_location,
            input_uri=args.vertex_input_uri,
            output_uri_prefix=args.vertex_output_prefix,
        )
    logger.info("Wrote reason cache to %s", args.output)


if __name__ == "__main__":
    main()

