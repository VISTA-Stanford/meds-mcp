#!/usr/bin/env python3
"""
Vertex AI Gemini batch prediction for vignette generation.

Builds a JSONL batch request file from the patients/items store, uploads it
to GCS, submits a Vertex AI batch prediction job, waits for completion, then
ingests results back into patients.jsonl — all without hitting per-request
rate limits.

Usage:
  # VISTA — one state per patient (latest embed_time), all tasks:
  uv run python experiments/fewshot_with_labels/batch_predict_vignettes.py \
    --person-ids 136106947 135933487 135939930 135982243 135958004 \
                 135919570 135934485 136046496 136091788 135908791 \
    --patients data/vista/patients.jsonl \
    --items    data/vista/items.jsonl \
    --corpus-dir /home/Ayeeshi/data/vista_bench_lumia \
    --gcs-staging gs://su_vista_scratch/vignette_batch \
    --vertex-project som-nero-plevriti-deidbdf \
    --latest-per-patient

  # EHRSHOT — one state per patient (latest embed_time), all tasks:
  uv run python experiments/fewshot_with_labels/batch_predict_vignettes.py \
    --person-ids 115967277 115972632 115971016 115973824 115973395 \
                 115971216 115970314 115972547 115970265 115973647 \
    --patients data/ehrshot/patients.jsonl \
    --items    data/ehrshot/items.jsonl \
    --corpus-dir /home/Ayeeshi/data/ehrshot_lumia/meds_corpus \
    --gcs-staging gs://su_vista_scratch/vignette_batch \
    --vertex-project som-nero-plevriti-deidbdf \
    --latest-per-patient
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import types
from dataclasses import replace
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_module(dotted: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(dotted, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


for _pkg, _path in [
    ("meds_mcp", _REPO_ROOT / "src/meds_mcp"),
    ("meds_mcp.similarity", _REPO_ROOT / "src/meds_mcp/similarity"),
    ("meds_mcp.experiments", _REPO_ROOT / "src/meds_mcp/experiments"),
]:
    if _pkg not in sys.modules:
        _ns = types.ModuleType(_pkg)
        _ns.__path__ = [str(_path)]
        sys.modules[_pkg] = _ns

_load_module("meds_mcp.similarity.vignette_base", _REPO_ROOT / "src/meds_mcp/similarity/vignette_base.py")
_cohort = _load_module("meds_mcp.similarity.cohort", _REPO_ROOT / "src/meds_mcp/similarity/cohort.py")
_det = _load_module(
    "meds_mcp.similarity.deterministic_linearization",
    _REPO_ROOT / "src/meds_mcp/similarity/deterministic_linearization.py",
)
_tc = _load_module("meds_mcp.experiments.task_config", _REPO_ROOT / "src/meds_mcp/experiments/task_config.py")

CohortStore = _cohort.CohortStore
DeterministicTimelineLinearizationGenerator = _det.DeterministicTimelineLinearizationGenerator
demographics_block = _det.demographics_block
TASK_DESCRIPTIONS = _tc.TASK_DESCRIPTIONS
TASK_QUESTIONS = _tc.TASK_QUESTIONS
BINARY_TASKS = _tc.BINARY_TASKS

from experiments.fewshot_with_labels._tokens import count_tokens, truncate_to_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts"
_EHRSHOT_TASKS = frozenset(BINARY_TASKS)

_TEMPLATE_FILES = {
    "ehrshot": _PROMPTS_DIR / "vignette_prompt_EHRSHOT.txt",
    "vista":   _PROMPTS_DIR / "vignette_prompt_VISTA.txt",
    "generic": _PROMPTS_DIR / "vignette_prompt_generic.txt",
}


def _load_template(task: str) -> str:
    if task in _EHRSHOT_TASKS:
        key = "ehrshot"
    elif task in TASK_DESCRIPTIONS and task not in _EHRSHOT_TASKS:
        key = "vista"
    else:
        key = "generic"
    path = _TEMPLATE_FILES[key]
    if not path.exists():
        raise SystemExit(f"Prompt template not found: {path}")
    return path.read_text().strip()


def _render_system_prompt(task: str, task_question: str) -> str:
    template = _load_template(task)
    task_focus = TASK_DESCRIPTIONS.get(task, "").strip()
    return template.format(TASK_QUESTION=task_question, TASK_FOCUS=task_focus)


def _build_user_message(person_id: str, embed_time, corpus_dir: Path, max_tokens: int) -> tuple[str, int, bool]:
    """Generate and truncate the LUMIA linearization for a patient state.

    Returns (user_message, original_token_count, was_truncated).
    """
    gen = DeterministicTimelineLinearizationGenerator(str(corpus_dir))
    timeline = gen.generate(person_id, cutoff_date=embed_time)
    demos = demographics_block(xml_dir=str(corpus_dir), patient_id=person_id, cutoff_date=embed_time)
    raw = (demos + "\n" + timeline) if demos else timeline
    return truncate_to_tokens(raw, max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--gcs-staging", type=str, default="gs://su_vista_scratch/vignette_batch",
                        help="GCS prefix for batch I/O. A timestamped subfolder is created automatically.")
    parser.add_argument("--vertex-project", type=str, default="som-nero-plevriti-deidbdf")
    parser.add_argument("--vertex-location", type=str, default="us-central1")
    parser.add_argument("--vertex-model", type=str, default="gemini-2.5-flash",
                        help="Model ID for Vertex AI batch prediction (full publisher path or short name).")
    parser.add_argument("--person-ids", type=str, nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate even if vignette exists")
    parser.add_argument("--latest-per-patient", action="store_true",
                        help="Keep only the latest embed_time per person_id.")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=115904,
                        help="Token budget for LUMIA user message (truncated if larger).")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Restrict to these task names.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the JSONL and manifest locally but do not submit the batch job.")
    args = parser.parse_args()

    # ── 1. Load store ────────────────────────────────────────────────────────
    logger.info("Loading store from %s / %s", args.patients, args.items)
    store = CohortStore.load(args.patients, args.items)

    # ── 2. Build per_task_plan ───────────────────────────────────────────────
    allowed_tasks = set(args.tasks) if args.tasks else None
    per_task_plan: dict = {}
    for it in store.items():
        if not it.embed_time:
            continue
        if allowed_tasks and it.task not in allowed_tasks:
            continue
        state = store.get_or_none(it.person_id, it.embed_time)
        if state is None:
            continue
        existing = state.task_vignettes.get(it.task) or ""
        if not args.force and existing.strip():
            continue
        per_task_plan.setdefault(state.key, []).append((it.task, it.question))

    todo = [store.get(pid, et) for (pid, et) in per_task_plan.keys()]

    if args.person_ids is not None:
        allowed_pids = set(args.person_ids)
        todo = [p for p in todo if p.person_id in allowed_pids]

    # Filter states whose XML exists
    missing_xml = [p for p in todo if not (args.corpus_dir / f"{p.person_id}.xml").exists()]
    if missing_xml:
        logger.warning("%d states skipped: XML missing", len(missing_xml))
    todo = [p for p in todo if (args.corpus_dir / f"{p.person_id}.xml").exists()]

    if args.latest_per_patient:
        best: dict = {}
        for p in todo:
            et = str(p.embed_time)
            if p.person_id not in best or et > str(best[p.person_id].embed_time):
                best[p.person_id] = p
        todo = list(best.values())
        logger.info("--latest-per-patient: %d states", len(todo))

    if not todo:
        logger.info("Nothing to do.")
        return

    # ── 3. Build batch requests ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = _REPO_ROOT / "experiments" / "fewshot_with_labels" / "batch_jobs" / timestamp
    local_dir.mkdir(parents=True, exist_ok=True)

    input_path = local_dir / "batch_input.jsonl"
    manifest_path = local_dir / "batch_manifest.jsonl"

    requests: list[dict] = []   # one per (state, task)
    manifest: list[dict] = []

    n_total = sum(len(tasks) for tasks in per_task_plan.values())
    logger.info("Building %d requests for %d states ...", n_total, len(todo))

    for p in todo:
        tasks_for_state = per_task_plan.get(p.key, [])
        try:
            user_msg, orig_tokens, truncated = _build_user_message(
                p.person_id, p.embed_time, args.corpus_dir, args.max_input_tokens
            )
        except Exception as e:
            logger.warning("Timeline failed for %s@%s: %s — skipping", p.person_id, p.embed_time, e)
            continue

        if not user_msg.strip():
            logger.warning("Empty timeline for %s@%s — skipping", p.person_id, p.embed_time)
            continue

        if truncated:
            logger.info("Truncated %s@%s: %d tokens -> %d", p.person_id, p.embed_time, orig_tokens, args.max_input_tokens)

        for task_name, task_question in tasks_for_state:
            task_q = TASK_QUESTIONS.get(task_name, "") or task_question.strip()
            system_prompt = _render_system_prompt(task_name, task_q)

            request_id = len(requests)  # stable index for matching output back to manifest
            request_obj = {
                "_request_id": request_id,  # passed through in output's request field
                "request": {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": args.max_output_tokens,
                    },
                }
            }
            requests.append(request_obj)
            manifest.append({
                "person_id": p.person_id,
                "embed_time": str(p.embed_time),
                "task": task_name,
            })

    logger.info("Writing %d requests to %s", len(requests), input_path)
    with open(input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    if args.dry_run:
        logger.info("--dry-run: stopping after local file generation. Input: %s", input_path)
        return

    # ── 4. Upload to GCS ─────────────────────────────────────────────────────
    import subprocess
    gcs_prefix = args.gcs_staging.rstrip("/") + "/" + timestamp
    gcs_input = gcs_prefix + "/batch_input.jsonl"
    gcs_output_prefix = gcs_prefix + "/output"

    logger.info("Uploading batch input to %s ...", gcs_input)
    result = subprocess.run(
        ["gsutil", "cp", str(input_path), gcs_input],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"gsutil cp failed: {result.stderr}")
    logger.info("Upload complete.")

    # ── 5. Submit batch job ───────────────────────────────────────────────────
    import vertexai
    from vertexai.preview.batch_prediction import BatchPredictionJob

    vertexai.init(project=args.vertex_project, location=args.vertex_location)

    model_name = args.vertex_model
    if not model_name.startswith("publishers/"):
        model_name = f"publishers/google/models/{model_name}"

    logger.info("Submitting batch job: model=%s input=%s output=%s", model_name, gcs_input, gcs_output_prefix)
    job = BatchPredictionJob.submit(
        source_model=model_name,
        input_dataset=gcs_input,
        output_uri_prefix=gcs_output_prefix,
    )
    logger.info("Batch job submitted: %s", job.name)
    logger.info("Job state: %s", job.state)

    # ── 6. Wait for completion ────────────────────────────────────────────────
    poll_interval = 30
    while not job.has_ended:
        time.sleep(poll_interval)
        job.refresh()
        logger.info("Job state: %s  (elapsed: check GCP console for %s)", job.state, job.name)

    if not job.has_succeeded:
        raise RuntimeError(f"Batch job did not succeed. State: {job.state}\nJob: {job.name}")

    logger.info("Batch job completed successfully.")

    # ── 7. Download output from GCS ───────────────────────────────────────────
    output_local = local_dir / "output"
    output_local.mkdir(exist_ok=True)
    logger.info("Downloading output from %s ...", gcs_output_prefix)
    result = subprocess.run(
        ["gsutil", "-m", "cp", "-r", gcs_output_prefix + "/*", str(output_local)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"gsutil download failed: {result.stderr}")

    # ── 8. Parse results and ingest into patients.jsonl ───────────────────────
    # NOTE: Vertex AI batch does NOT preserve input order — output lines can be
    # in any order. We match each output back to its manifest entry by:
    #   1. _request_id field (if present — set by this script going forward)
    #   2. Fallback: exact match on (user_msg, sys_msg) content
    output_lines: list[str] = []
    for jsonl_file in sorted(output_local.rglob("*.jsonl")):
        with open(jsonl_file) as f:
            output_lines.extend(f.readlines())

    logger.info("Output lines: %d  Manifest entries: %d", len(output_lines), len(manifest))

    # Build content-based lookup for fallback matching
    content_key_to_meta: dict[tuple, dict] = {}
    with open(input_path) as f:
        input_requests = [json.loads(l) for l in f]
    for inp_req, meta in zip(input_requests, manifest):
        req = inp_req.get("request", inp_req)  # handle top-level or nested
        user_msg = req["contents"][0]["parts"][0]["text"]
        sys_msg = req["system_instruction"]["parts"][0]["text"]
        content_key_to_meta[(user_msg, sys_msg)] = meta

    vignette_map: dict[tuple, str] = {}  # (person_id, embed_time, task) -> vignette text
    n_ok = n_fail = n_no_match = 0
    for out_idx, raw_line in enumerate(output_lines):
        try:
            out = json.loads(raw_line)
            status = out.get("status", "").lower()
            if status not in ("ok", "succeeded", "success", ""):
                logger.warning("Output %d failed: status=%s", out_idx, out.get("status"))
                n_fail += 1
                continue

            # Identify which manifest entry this output corresponds to
            out_req = out.get("request", {})
            request_id = out.get("_request_id")  # present in new-format batches
            if request_id is not None and 0 <= request_id < len(manifest):
                meta = manifest[request_id]
            else:
                # Fallback: match by full content
                out_user = out_req.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
                out_sys = out_req.get("system_instruction", {}).get("parts", [{}])[0].get("text", "")
                meta = content_key_to_meta.get((out_user, out_sys))
                if meta is None:
                    logger.warning("Output %d: could not match to any manifest entry", out_idx)
                    n_no_match += 1
                    n_fail += 1
                    continue

            resp = out.get("response", {})
            candidates = resp.get("candidates", [])
            if not candidates:
                logger.warning("Output %d (%s/%s): no candidates", out_idx, meta["person_id"], meta["task"])
                n_fail += 1
                continue

            # Skip thought parts; take the first non-thought text part
            parts = candidates[0].get("content", {}).get("parts", [])
            text = next(
                (p.get("text", "").strip() for p in parts if not p.get("thought", False)),
                ""
            )
            if not text:
                logger.warning("Output %d (%s/%s): empty text", out_idx, meta["person_id"], meta["task"])
                n_fail += 1
                continue

            key = (meta["person_id"], meta["embed_time"], meta["task"])
            vignette_map[key] = text
            n_ok += 1
        except Exception as e:
            logger.warning("Error parsing output line %d: %s", out_idx, e)
            n_fail += 1

    if n_no_match:
        logger.warning("%d output lines could not be matched to manifest entries", n_no_match)

    logger.info("Parsed: ok=%d  fail=%d", n_ok, n_fail)

    # Update patients.jsonl
    updated_patients = []
    with open(args.patients) as f:
        for line in f:
            p = json.loads(line)
            pid = p["person_id"]
            et = str(p.get("embed_time", ""))
            task_vigs = dict(p.get("task_vignettes") or {})
            changed = False
            for task, vignette in vignette_map.items():
                if task[0] == pid and task[1] == et:
                    task_vigs[task[2]] = vignette
                    changed = True
            if changed:
                p["task_vignettes"] = task_vigs
            updated_patients.append(p)

    # Atomic write
    tmp = args.patients.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for p in updated_patients:
            f.write(json.dumps(p) + "\n")
    tmp.replace(args.patients)
    logger.info("Wrote updated patients.jsonl (%d states)", len(updated_patients))

    logger.info("Done. ok=%d fail=%d  batch_job=%s", n_ok, n_fail, job.name)
    logger.info("Manifest + raw outputs saved to: %s", local_dir)


if __name__ == "__main__":
    main()
