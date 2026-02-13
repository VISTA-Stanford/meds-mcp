#!/usr/bin/env python3
"""
Run vista_bench experiment: LLM only vs LLM + tool for each task.

Context is restricted to single-visit (the encounter containing prediction_time) by default.
Precompute events once for faster API calls: --precompute

For each row in each task CSV:
1. Load single-visit events (precomputed or parsed from XML)
2. Run LLM only (use_tools=False)
3. Run LLM + tool (use_tools=True, task-specific tool only)
4. Record raw and normalized responses with ground truth

Usage:
  python scripts/run_vista_bench_experiment.py --config configs/vista.yaml
  python scripts/run_vista_bench_experiment.py --config configs/vista.yaml --precompute  # precompute events first
  python scripts/run_vista_bench_experiment.py --config configs/vista.yaml --task guo_readmission --limit 5
  python scripts/run_vista_bench_experiment.py --config configs/vista.yaml --resume --output results/run_001.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple

from fastapi import HTTPException

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def initialize_for_experiment(
    config: dict,
    patient_ids: Optional[list] = None,
) -> None:
    """Initialize document store (required for get_all_patient_events).
    When patient_ids is provided, only those patients are loaded (subset from label CSVs).
    """
    from meds_mcp.server.rag.simple_storage import initialize_document_store

    data_dir = config.get("data", {}).get("corpus_dir") or os.getenv(
        "DATA_DIR", "data/collections/vista_bench/thoracic_cohort_lumia"
    )
    cache_dir = config.get("data", {}).get("cache_dir") or os.getenv(
        "CACHE_DIR", "cache"
    )
    load_all_patients = (
        config.get("data", {}).get("load_all_patients", True)
        or os.getenv("LOAD_ALL_PATIENTS", "true").lower() == "true"
    )
    patient_id = config.get("data", {}).get("patient_id") or os.getenv("PATIENT_ID")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {data_dir}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if patient_ids:
        logging.info(f"Initializing document store with {len(patient_ids)} patients (subset from labels): {data_dir}")
    else:
        logging.info(f"Initializing document store: {data_dir}")
    initialize_document_store(
        data_dir,
        str(cache_path),
        load_all_patients,
        patient_id=patient_id,
        patient_ids=patient_ids,
    )
    logging.info("Document store initialized")


def _parse_retry_seconds(exc: Exception) -> int:
    """Parse 'Try again in X seconds' from error message. Default 60."""
    msg = str(exc)
    m = re.search(r"(?:try again in|wait)\s+(\d+)\s*seconds?", msg, re.I)
    if m:
        return max(int(m.group(1)), 10)  # at least 10s
    return 60


def _is_retryable_error(exc: Exception) -> bool:
    """Check if error is retryable (429 rate limit, token limit, etc.)."""
    msg = str(exc).lower()
    if getattr(exc, "status_code", None) == 429:
        return True
    if hasattr(exc, "response") and getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return True
    return "429" in msg or "token limit" in msg or "rate limit" in msg


def _cache_key(
    patient_id: str,
    prediction_time: str,
    task_name: str,
    use_tools: bool,
    model: Optional[str] = None,
) -> tuple:
    """Cache key for API responses. Include model when set so different models don't share cache."""
    if model:
        return (patient_id, prediction_time, task_name, use_tools, model)
    return (patient_id, prediction_time, task_name, use_tools)


def _load_precomputed_events(cache_path: Path) -> dict:
    """Load precomputed single-visit events cache. Keys: (patient_id, prediction_time), Values: list of events."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        out = {}
        for k_str, events in data.items():
            if "|" in k_str:
                pid, pt = k_str.split("|", 1)
                out[(pid, pt)] = events
        return out
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logging.warning(f"Could not load precomputed events from {cache_path}: {e}")
        return {}


def _load_context_cache(cache_path: Path) -> dict:
    """Load precomputed formatted context cache. Keys: 'patient_id|prediction_time|task_name', Values: context string."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logging.warning(f"Could not load context cache from {cache_path}: {e}")
        return {}


def _load_cache(cache_path: Path) -> dict:
    """Load response cache from JSON file. Keys are tuples stored as JSON arrays."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        out = {}
        for k_str, v in data.items():
            try:
                k = tuple(json.loads(k_str))
                out[k] = v
            except (json.JSONDecodeError, TypeError):
                continue
        return out
    except (json.JSONDecodeError, TypeError) as e:
        logging.warning(f"Could not load cache from {cache_path}: {e}")
        return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Save response cache to JSON file. Tuples become JSON arrays."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {json.dumps(list(k)): v for k, v in cache.items()}
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=None)


async def run_single_prediction(
    patient_id: str,
    prediction_time: str,
    task_name: str,
    question: str,
    use_tools: bool,
    max_retries: int = 5,
    cache: Optional[dict] = None,
    precomputed_events: Optional[dict] = None,
    precomputed_context_cache: Optional[dict] = None,
    debug: bool = False,
    model: Optional[str] = None,
) -> str:
    """Run cohort_chat for one (patient, task) with LLM only or LLM+tool.
    Retries on 429/token limit with backoff. Uses cache when provided.
    When precomputed_context_cache is set and has the key, uses precomputed_context_text (formatted string).
    Otherwise uses precomputed_events (event lists) if set.
    """
    from meds_mcp.server.api.cohort_chat import cohort_chat
    from meds_mcp.server.api.cohort_chat import CohortChatRequest

    key = _cache_key(patient_id, prediction_time, task_name, use_tools, model=model)
    if cache is not None and key in cache:
        cached = cache[key]
        if isinstance(cached, dict):
            return cached
        return {"answer": cached, "tool_executions": 0}

    pc = None
    pc_text = None
    context_cache_key = f"{patient_id}|{prediction_time}|{task_name}"
    if precomputed_context_cache is not None and context_cache_key in precomputed_context_cache:
        pc_text = {patient_id: precomputed_context_cache[context_cache_key]}
    elif precomputed_events is not None:
        row_key = (patient_id, prediction_time)
        if row_key in precomputed_events:
            pc = {patient_id: precomputed_events[row_key]}

    payload = CohortChatRequest(
        question=question,
        patient_ids=[patient_id],
        prediction_time=prediction_time,
        task_name=task_name,
        use_tools=use_tools,
        max_events_per_patient=500,
        precomputed_context=pc,
        precomputed_context_text=pc_text,
        debug=debug,
        model=model,
    )
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = await cohort_chat(payload)
            out = {
                "answer": result.answer,
                "tool_executions": getattr(result, "tool_executions", 0),
            }
            if cache is not None:
                cache[key] = out
            return out
        except HTTPException as e:
            last_exc = e
            # 429 may be wrapped as 500 by cohort_chat; check detail for retryable
            is_retryable = e.status_code == 429 or _is_retryable_error(e)
            if is_retryable and attempt < max_retries:
                wait_s = _parse_retry_seconds(e)
                logging.warning(
                    f"Rate limited (429/token limit) for {patient_id}, waiting {wait_s}s before retry {attempt + 1}/{max_retries}"
                )
                await asyncio.sleep(wait_s)
                continue
            logging.error(
                f"HTTP {e.status_code} for {patient_id} {task_name} use_tools={use_tools}: {e.detail}"
            )
            return {"answer": f"[ERROR: {str(e.detail)}]", "tool_executions": 0}
        except Exception as e:
            last_exc = e
            if _is_retryable_error(e) and attempt < max_retries:
                wait_s = _parse_retry_seconds(e)
                logging.warning(
                    f"Retryable error for {patient_id}, waiting {wait_s}s before retry {attempt + 1}/{max_retries}: {e}"
                )
                await asyncio.sleep(wait_s)
                continue
            logging.error(f"Error for {patient_id} {task_name} use_tools={use_tools}: {e}")
            return {"answer": f"[ERROR: {str(e)}]", "tool_executions": 0}
    return {"answer": f"[ERROR: {str(last_exc)}]", "tool_executions": 0}


async def _process_single_row(
    row: dict,
    task_name: str,
    question: str,
    is_binary: bool,
    sem: asyncio.Semaphore,
    delay_seconds: float,
    max_retries: int,
    cache: Optional[dict] = None,
    precomputed_events: Optional[dict] = None,
    precomputed_context_cache: Optional[dict] = None,
    debug: bool = False,
    model: Optional[str] = None,
) -> Tuple[Tuple[str, str, str], dict]:
    """Process one row: LLM-only + LLM+tool (in parallel). Returns (row_key, record)."""
    from meds_mcp.experiments.formatters import ground_truth_to_normalized

    patient_id = row.get("patient_id")
    prediction_time = row.get("prediction_time")
    ground_truth_raw = row.get("label", "")
    ground_truth_norm = ground_truth_to_normalized(ground_truth_raw, is_binary)
    row_key = (patient_id, task_name, prediction_time)

    async with sem:
        result_llm, result_tool = await asyncio.gather(
            run_single_prediction(
                patient_id, prediction_time, task_name, question, use_tools=False,
                max_retries=max_retries, cache=cache, precomputed_events=precomputed_events,
                precomputed_context_cache=precomputed_context_cache, debug=debug, model=model,
            ),
            run_single_prediction(
                patient_id, prediction_time, task_name, question, use_tools=True,
                max_retries=max_retries, cache=cache, precomputed_events=precomputed_events,
                precomputed_context_cache=precomputed_context_cache, debug=debug, model=model,
            ),
        )
        await asyncio.sleep(delay_seconds)

    answer_llm = result_llm["answer"]
    answer_tool = result_tool["answer"]
    tool_executions = result_tool.get("tool_executions", 0)

    from meds_mcp.experiments.formatters import normalize_binary, normalize_categorical
    norm_llm = normalize_binary(answer_llm) if is_binary else normalize_categorical(answer_llm)
    norm_tool = normalize_binary(answer_tool) if is_binary else normalize_categorical(answer_tool)

    record = {
        "patient_id": patient_id,
        "task_name": task_name,
        "prediction_time": prediction_time,
        "ground_truth_raw": ground_truth_raw,
        "ground_truth_normalized": ground_truth_norm,
        "llm_only_raw": answer_llm,
        "llm_only_normalized": norm_llm,
        "llm_plus_tool_raw": answer_tool,
        "llm_plus_tool_normalized": norm_tool,
        "tool_executions": tool_executions,
    }
    return (row_key, record)


async def run_experiment_async(
    args, config, num_patients_loaded: int = 0, precomputed_events: Optional[dict] = None,
    precomputed_context_cache: Optional[dict] = None,
):  # noqa: C901
    """Async experiment runner with progress bar and parallel batching."""
    from meds_mcp.experiments.task_config import (
        ALL_TASKS,
        TASK_QUESTIONS,
        get_csv_path_for_task,
        is_binary_task,
    )
    from meds_mcp.experiments.formatters import ground_truth_to_normalized
    import csv as csv_module

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # fallback to simple progress

    tasks_to_run = [args.task] if args.task else ALL_TASKS
    if args.task and args.task not in ALL_TASKS:
        raise ValueError(f"Unknown task: {args.task}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(
        output_dir / f"vista_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # Precomputed single-visit events: use passed-in or load from path (only when not using context cache)
    if precomputed_events is None and not precomputed_context_cache:
        precomputed_path = (
            Path(args.precomputed_events_path)
            if getattr(args, "precomputed_events_path", None)
            else Path(args.output_dir) / "single_visit_events_cache.json"
        )
        precomputed_events = _load_precomputed_events(precomputed_path)
        if precomputed_events:
            logging.info(f"Loaded {len(precomputed_events)} precomputed single-visit event streams from {precomputed_path}")

    # Response cache: avoid re-calling API for same (patient_id, prediction_time, task_name, use_tools)
    cache: dict = {}
    cache_path = Path(output_path).with_suffix(".cache.json")
    if not args.no_cache:
        cache = _load_cache(cache_path)
        if cache:
            logging.info(f"Loaded {len(cache)} cached responses from {cache_path}")

    completed: Set[Tuple[str, str, str]] = set()
    write_mode = "a" if (args.resume and Path(output_path).exists()) else "w"
    if args.resume and Path(output_path).exists():
        with open(output_path, "r") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed.add((
                        rec.get("patient_id", ""),
                        rec.get("task_name", ""),
                        rec.get("prediction_time", ""),
                    ))
                except json.JSONDecodeError:
                    continue
        logging.info(f"Resuming: {len(completed)} rows already in {output_path}")

    total = 0
    task_summaries = {}
    run_start = time.perf_counter()
    sem = asyncio.Semaphore(args.batch_size)
    model_arg = getattr(args, "model", None)
    if model_arg:
        logging.info("Using model: %s", model_arg)

    with open(output_path, write_mode) as f:
        for task_name in tasks_to_run:
            csv_path = get_csv_path_for_task(task_name)
            if not csv_path.exists():
                logging.warning(f"CSV not found for {task_name}: {csv_path}, skipping")
                continue

            question = TASK_QUESTIONS.get(task_name, "Answer based on the patient timeline.")
            is_binary = is_binary_task(task_name)

            rows = []
            with open(csv_path, "r", encoding="utf-8") as cf:
                reader = csv_module.DictReader(cf)
                for row in reader:
                    rows.append(row)
                    if args.limit and len(rows) >= args.limit:
                        break

            to_process = []
            skipped_not_in_cache = 0
            for row in rows:
                patient_id = row.get("patient_id")
                prediction_time = row.get("prediction_time")
                if not patient_id or not prediction_time:
                    continue
                row_key = (patient_id, task_name, prediction_time)
                if row_key in completed:
                    continue
                # When using context cache, skip rows not in cache
                if precomputed_context_cache:
                    if f"{patient_id}|{prediction_time}|{task_name}" not in precomputed_context_cache:
                        skipped_not_in_cache += 1
                        continue
                # When using events cache (and not context cache), skip rows not in cache
                elif precomputed_events and (patient_id, prediction_time) not in precomputed_events:
                    skipped_not_in_cache += 1
                    continue
                to_process.append(row)

            if skipped_not_in_cache:
                logging.info(f"Task {task_name}: skipped {skipped_not_in_cache} rows (not in cache)")
            if not to_process:
                logging.info(f"Task {task_name}: no rows to process (all done or skipped)")
                continue

            task_start = time.perf_counter()
            task_tool_exposed = len(to_process)
            task_tool_executed = 0
            logging.info(f"Task {task_name}: {len(to_process)} rows (batch_size={args.batch_size})")

            async def process_with_progress(row, pbar):
                result = await _process_single_row(
                    row, task_name, question, is_binary,
                    sem, args.delay_seconds, args.max_retries, cache=cache,
                    precomputed_events=precomputed_events,
                    precomputed_context_cache=precomputed_context_cache,
                    debug=getattr(args, "debug", False),
                    model=model_arg,
                )
                if pbar:
                    pbar.update(1)
                return result

            if tqdm:
                pbar = tqdm(total=len(to_process), desc=task_name, unit="row")
            else:
                pbar = None

            tasks = [process_with_progress(row, pbar) for row in to_process]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            if tqdm:
                pbar.close()

            task_success = 0
            write_buffer: list = []
            for r in results:
                if isinstance(r, Exception):
                    logging.error(f"Row failed: {r}")
                    continue
                row_key, record = r
                write_buffer.append(record)
                total += 1
                task_success += 1
                task_tool_executed += record.get("tool_executions", 0)
                completed.add(row_key)
                if len(write_buffer) >= args.write_batch_size:
                    for rec in write_buffer:
                        f.write(json.dumps(rec) + "\n")
                    f.flush()
                    write_buffer = []
            # Flush remaining records
            for rec in write_buffer:
                f.write(json.dumps(rec) + "\n")
            f.flush()

            task_elapsed = time.perf_counter() - task_start
            task_summaries[task_name] = {
                "num_patients": task_success,
                "elapsed_seconds": round(task_elapsed, 2),
                "tool_exposed_count": task_tool_exposed,
                "tool_executed_count": task_tool_executed,
            }
            logging.info(f"  {task_name}: {len(to_process)} patients in {task_elapsed:.1f}s")

    run_elapsed = time.perf_counter() - run_start
    if not args.no_cache and cache:
        _save_cache(cache, cache_path)
        logging.info(f"Saved {len(cache)} cached responses to {cache_path}")
    summary_path = str(Path(output_path).with_suffix("")) + "_summary.json"
    summary = {
        "output_file": output_path,
        "total_patients": total,
        "num_patients_loaded": num_patients_loaded,
        "total_elapsed_seconds": round(run_elapsed, 2),
        "resumed": args.resume,
        "tasks": task_summaries,
    }
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    logging.info(f"Summary written to {summary_path}")

    return total, output_path, summary_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run vista_bench experiment: LLM vs LLM+tool"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vista.yaml",
        help="Path to server config (for corpus_dir, etc.)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run only this task (e.g. guo_readmission). Default: all tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows per task (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSONL results. Default: results/vista_bench_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Delay between patients (throttle). Default: 2.0",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already-processed rows). Requires --output.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per LLM call on 429/token limit. Default: 5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Max concurrent API calls for parallelization. Default: 5",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=50,
        help="Write records to file in batches of this size. Default: 50",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching (do not load or save cached API responses)",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Run precomputation of single-visit events before experiment (saves to output-dir/single_visit_events_cache.json)",
    )
    parser.add_argument(
        "--precomputed-events-path",
        type=str,
        default=None,
        help="Path to precomputed single-visit events JSON. Default: output-dir/single_visit_events_cache.json",
    )
    parser.add_argument(
        "--context-cache-path",
        type=str,
        default=None,
        help="Path to precomputed formatted context JSON (from precompute_context_cache.py). When set, uses new format cache and skips document store init. Default: output-dir/context_cache.json",
    )
    parser.add_argument(
        "--precompute-context",
        action="store_true",
        help="Run precompute_context_cache.py before experiment (builds context_cache.json from events cache or XML)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full context (cohort block + user prompt) to stdout before each LLM/tool call.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (e.g. apim:gpt-4.1-mini). If not set, uses server default.",
    )
    args = parser.parse_args()

    if args.resume and not args.output:
        logging.error("--resume requires --output (path to existing JSONL file)")
        sys.exit(1)

    config = load_config(args.config)

    if args.precompute:
        import subprocess
        precompute_cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "precompute_single_visit_events.py"),
            "--config", args.config,
            "--output-dir", args.output_dir,
        ]
        if args.task:
            precompute_cmd.extend(["--task", args.task])
        if args.limit:
            precompute_cmd.extend(["--limit", str(args.limit)])
        logging.info("Running precomputation of single-visit events...")
        subprocess.run(precompute_cmd, check=True)

    if getattr(args, "precompute_context", False):
        import subprocess
        events_cache = Path(args.output_dir) / "single_visit_events_cache.json"
        ctx_cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "precompute_context_cache.py"),
            "--output-dir", args.output_dir,
        ]
        if events_cache.exists():
            ctx_cmd.extend(["--events-cache", str(events_cache)])
        else:
            ctx_cmd.extend(["--config", args.config])
        if args.task:
            ctx_cmd.extend(["--task", args.task])
        if args.limit:
            ctx_cmd.extend(["--limit", str(args.limit)])
        logging.info("Running precomputation of formatted context cache...")
        subprocess.run(ctx_cmd, check=True)

    # Collect patient_ids from task CSVs so we only load that subset
    from meds_mcp.experiments.task_config import (
        ALL_TASKS,
        get_patient_ids_from_task_csvs,
    )

    tasks_to_run = [args.task] if args.task else ALL_TASKS
    if args.task and args.task not in ALL_TASKS:
        logging.error(f"Unknown task: {args.task}")
        sys.exit(1)

    patient_ids = get_patient_ids_from_task_csvs(tasks_to_run, args.limit)
    logging.info(f"Found {len(patient_ids)} unique patients across task CSVs (will load subset only)")

    # Context cache (new format: formatted context string per pid|pt|task) takes precedence
    context_cache_path = (
        Path(args.context_cache_path)
        if getattr(args, "context_cache_path", None)
        else Path(args.output_dir) / "context_cache.json"
    )
    precomputed_context_cache = _load_context_cache(context_cache_path)
    if precomputed_context_cache:
        logging.info(
            f"Using context cache ({len(precomputed_context_cache)} entries); skipping document store init. "
            "Rows not in cache will be skipped."
        )

    # Events cache (for backward compat when not using context cache)
    if not precomputed_context_cache:
        precomputed_path = (
            Path(args.precomputed_events_path)
            if getattr(args, "precomputed_events_path", None)
            else Path(args.output_dir) / "single_visit_events_cache.json"
        )
        precomputed_events = _load_precomputed_events(precomputed_path)
        if precomputed_events:
            logging.info(
                f"Using precomputed events cache ({len(precomputed_events)} entries); skipping document store init. "
                "Rows not in cache will be skipped."
            )
        else:
            logging.info("No precomputed cache found; initializing document store (this may take a while)")
            initialize_for_experiment(config, patient_ids=patient_ids)
    else:
        precomputed_events = {}

    # Silence per-request logs during experiment (progress bar only)
    logging.getLogger("meds_mcp.server.api.cohort_chat").setLevel(logging.WARNING)
    for _logger in ("securellm", "securellm.providers", "securellm.providers.apim_provider"):
        logging.getLogger(_logger).setLevel(logging.ERROR)

    total, output_path, summary_path = asyncio.run(
        run_experiment_async(
            args, config,
            num_patients_loaded=len(patient_ids),
            precomputed_events=precomputed_events,
            precomputed_context_cache=precomputed_context_cache,
        )
    )
    logging.info(f"Done. Wrote {total} records to {output_path}, summary to {summary_path}")


if __name__ == "__main__":
    main()
