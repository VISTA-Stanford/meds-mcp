#!/usr/bin/env python3
"""
Vertex batch precompute for patient vignettes.

Builds summarization requests from timeline linearization and updates patients.jsonl
with the resulting vignettes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import CohortStore, DeterministicTimelineLinearizationGenerator, demographics_block
from meds_mcp.similarity.llm_secure_adapter import SecureLLMSummarizer, load_vignette_prompt
from experiments.fewshot_with_labels import _paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    s = uri.replace("gs://", "", 1)
    b, _, p = s.partition("/")
    return b, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute vignettes via Vertex batch inference.")
    parser.add_argument("--patients", type=Path, default=_paths.patients_jsonl())
    parser.add_argument("--items", type=Path, default=_paths.items_jsonl())
    parser.add_argument("--corpus-dir", type=Path, default=_paths.corpus_dir())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-encounters", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--vignette-prompt",
        type=Path,
        default=None,
        help="Path to a custom vignette system prompt file. Overrides the default vignette_prompt.txt.",
    )
    parser.add_argument(
        "--person-ids-file",
        type=Path,
        default=None,
        help="JSON file containing a list of person_id strings. When set, only generate vignettes for those patients.",
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--vertex-project", type=str, default="som-nero-plevriti-deidbdf")
    parser.add_argument("--vertex-location", type=str, default="us-central1")
    parser.add_argument(
        "--vertex-input-uri",
        type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/input/vignette_batch.jsonl",
    )
    parser.add_argument(
        "--vertex-output-prefix",
        type=str,
        default="gs://vista_bench/temp/pinnacle_templated_summaries/output/vignette_batch",
    )
    args = parser.parse_args()

    from google.cloud import storage
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob

    store = CohortStore.load(args.patients, args.items)
    linearizer = DeterministicTimelineLinearizationGenerator(str(args.corpus_dir))
    if args.vignette_prompt is not None:
        system_prompt = args.vignette_prompt.read_text(encoding="utf-8").strip()
        logger.info("Using custom vignette prompt from %s", args.vignette_prompt)
    else:
        system_prompt = load_vignette_prompt() or SecureLLMSummarizer._default_prompt()

    pid_filter: set[str] | None = None
    if args.person_ids_file is not None:
        import json as _json
        pid_filter = {str(x) for x in _json.loads(args.person_ids_file.read_text())}
        logger.info("Restricting vignette generation to %d person_ids from %s", len(pid_filter), args.person_ids_file)

    todo = [
        p for p in store.patient_states()
        if (args.force or not p.vignette.strip())
        and p.embed_time
        and (pid_filter is None or p.person_id in pid_filter)
    ]
    if args.limit is not None:
        todo = todo[: args.limit]
    logger.info("States to summarize via batch: %d", len(todo))
    if not todo:
        return

    requests: list[dict] = []
    for p in todo:
        try:
            timeline = linearizer.generate(
                patient_id=p.person_id,
                cutoff_date=p.embed_time,
                n_encounters=args.n_encounters,
            )
        except Exception as exc:
            logger.warning("Skipping %s@%s (timeline failure): %s", p.person_id, p.embed_time, exc)
            continue
        demos = demographics_block(
            xml_dir=str(args.corpus_dir),
            patient_id=p.person_id,
            cutoff_date=p.embed_time,
        )
        user_text = (demos + "\n" + timeline) if demos else timeline
        requests.append(
            {
                "request": {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024,
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
                "_meta": json.dumps({"person_id": p.person_id, "embed_time": p.embed_time}),
            }
        )

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
        job_display_name=f"fewshot_vignette_batch_{int(time.time())}",
    )
    logger.info("Submitted batch job: %s", job.resource_name)
    while not job.has_ended:
        time.sleep(30)
        job.refresh()
        logger.info("Batch state: %s", job.state)
    if not job.has_succeeded:
        raise RuntimeError(f"Batch failed: {job.error}")

    out_bucket, out_prefix = _parse_gcs_uri(job.output_location)
    blobs = [
        b for b in storage_client.bucket(out_bucket).list_blobs(prefix=out_prefix)
        if b.name.endswith(".jsonl")
    ]
    updates = 0
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
                vignette = obj["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                continue
            pid = str(meta["person_id"])
            et = str(meta["embed_time"])
            state = store.get_or_none(pid, et)
            if state is None:
                continue
            store.update_patient(replace(state, vignette=vignette))
            updates += 1

    store.save(args.patients, args.items)
    logger.info("Updated %d vignettes in %s", updates, args.patients)


if __name__ == "__main__":
    main()

