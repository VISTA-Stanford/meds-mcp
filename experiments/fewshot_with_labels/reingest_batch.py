#!/usr/bin/env python3
"""
Re-ingest a completed Vertex AI batch job into patients.jsonl.

Handles the fact that Vertex AI batch output is NOT in input order — matches
each output line back to its manifest entry using content (user_msg + sys_msg).

Usage:
  uv run python experiments/fewshot_with_labels/reingest_batch.py \
    --batch-dir experiments/fewshot_with_labels/batch_jobs/20260501_072048 \
    --patients data/vista/patients.jsonl

  uv run python experiments/fewshot_with_labels/reingest_batch.py \
    --batch-dir experiments/fewshot_with_labels/batch_jobs/20260501_072113 \
    --patients data/ehrshot/patients.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_text(candidates: list) -> str:
    parts = candidates[0].get("content", {}).get("parts", [])
    return next(
        (p.get("text", "").strip() for p in parts if not p.get("thought", False)),
        ""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, required=True,
                        help="Path to the batch job directory (contains batch_input.jsonl, batch_manifest.jsonl, output/)")
    parser.add_argument("--patients", type=Path, required=True,
                        help="Path to patients.jsonl to update")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing vignettes")
    args = parser.parse_args()

    input_path = args.batch_dir / "batch_input.jsonl"
    manifest_path = args.batch_dir / "batch_manifest.jsonl"

    with open(input_path) as f:
        input_requests = [json.loads(l) for l in f]
    with open(manifest_path) as f:
        manifest = [json.loads(l) for l in f]

    logger.info("Loaded %d manifest entries from %s", len(manifest), args.batch_dir)

    # Collect all output lines
    output_lines: list[str] = []
    for jsonl_file in sorted(args.batch_dir.rglob("output/**/*.jsonl")):
        with open(jsonl_file) as f:
            output_lines.extend(f.readlines())
    logger.info("Output lines: %d", len(output_lines))

    # Build content-based lookup: (user_msg, sys_msg) -> manifest entry
    content_key_to_meta: dict[tuple, dict] = {}
    for inp_req, meta in zip(input_requests, manifest):
        req = inp_req.get("request", inp_req)
        user_msg = req["contents"][0]["parts"][0]["text"]
        sys_msg = req["system_instruction"]["parts"][0]["text"]
        key = (user_msg, sys_msg)
        if key in content_key_to_meta:
            logger.warning("Duplicate content key for %s/%s", meta["person_id"], meta["task"])
        content_key_to_meta[key] = meta

    # Parse output lines — match by content
    vignette_map: dict[tuple, str] = {}
    n_ok = n_fail = n_no_match = 0

    for out_idx, raw_line in enumerate(output_lines):
        try:
            out = json.loads(raw_line)
            status = out.get("status", "").lower()
            if status not in ("ok", "succeeded", "success", ""):
                logger.warning("Output %d: status=%s", out_idx, out.get("status"))
                n_fail += 1
                continue

            # Match by _request_id if present, else by content
            request_id = out.get("_request_id")
            if request_id is not None and 0 <= request_id < len(manifest):
                meta = manifest[request_id]
            else:
                out_req = out.get("request", {})
                out_user = out_req.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
                out_sys = out_req.get("system_instruction", {}).get("parts", [{}])[0].get("text", "")
                meta = content_key_to_meta.get((out_user, out_sys))
                if meta is None:
                    logger.warning("Output %d: no match found", out_idx)
                    n_no_match += 1
                    n_fail += 1
                    continue

            candidates = out.get("response", {}).get("candidates", [])
            if not candidates:
                logger.warning("Output %d (%s/%s): no candidates", out_idx, meta["person_id"], meta["task"])
                n_fail += 1
                continue

            text = extract_text(candidates)
            if not text:
                logger.warning("Output %d (%s/%s): empty text", out_idx, meta["person_id"], meta["task"])
                n_fail += 1
                continue

            key = (meta["person_id"], meta["embed_time"], meta["task"])
            vignette_map[key] = text
            n_ok += 1
        except Exception as e:
            logger.warning("Error on output line %d: %s", out_idx, e)
            n_fail += 1

    logger.info("Parsed: ok=%d  fail=%d  no_match=%d", n_ok, n_fail, n_no_match)

    # Update patients.jsonl
    updated = []
    n_updated = 0
    with open(args.patients) as f:
        for line in f:
            p = json.loads(line)
            pid = p["person_id"]
            et = str(p.get("embed_time", ""))
            task_vigs = dict(p.get("task_vignettes") or {})
            changed = False
            for (v_pid, v_et, v_task), text in vignette_map.items():
                if v_pid == pid and v_et == et:
                    if args.force or not task_vigs.get(v_task, "").strip():
                        task_vigs[v_task] = text
                        changed = True
            if changed:
                p["task_vignettes"] = task_vigs
                n_updated += 1
            updated.append(p)

    tmp = args.patients.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for p in updated:
            f.write(json.dumps(p) + "\n")
    tmp.replace(args.patients)
    logger.info("Wrote %s — %d states updated", args.patients, n_updated)


if __name__ == "__main__":
    main()
