"""
Minimal LLM-only experiment: single task (lab_thrombocytopenia), no tools.

Calls the secure-llm API directly (no cohort_chat, no tools, no tool loop).
Uses context_cache.json and the same prompt text as cohort_chat for Vista task + use_tools=False.
When context cache is provided, document store is not initialized (rows not in cache are skipped).

Usage:
  uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --context-cache-path results/context_cache.json
  uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --precompute-context --limit 10
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Task is fixed for this minimal experiment
TASK_NAME = "lab_thrombocytopenia"


def _build_llm_only_messages(
    task_name: str,
    prediction_time: str,
    patient_id: str,
    context_text: str,
    question: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build system + user messages matching cohort_chat for a Vista task with use_tools=False.
    Same prompt text as cohort_chat so the experiment is comparable.
    """
    from meds_mcp.experiments.task_config import TASK_PREDICTION_TARGET

    task_sentence = TASK_PREDICTION_TARGET.get(
        task_name, question or "will have the outcome of interest"
    )
    system_prompt = (
        f"You are a clinical prediction model. Your task: Predict whether this patient {task_sentence}.\n\n"
        "You have access to a specialized clinical prediction tool that returns a predicted probability for this task. "
        "You may call this tool to get its estimate.\n\n"
        "IMPORTANT: You must critically evaluate the tool's output against the patient's clinical history. "
        "The tool returns a raw probability — you must decide how to interpret it. If the probability conflicts "
        "with strong clinical evidence in the timeline, use your own clinical judgment. Do NOT blindly trust the tool's output.\n\n"
        "After considering all evidence, respond with ONLY a JSON object with exactly two fields: "
        '{"outcome": "yes" or "no", "reasoning": "brief explanation (2-3 sentences)"}. '
        "Output valid JSON only. No additional text outside the JSON object.\n\n"
        "Answer based on the information in the conversation only. Do not use any tools.\n\n"
    )
    prediction_target = TASK_PREDICTION_TARGET.get(task_name, question) if task_name else (question or "will have the outcome of interest")
    pred_time_str = prediction_time or "(not set)"
    cohort_context_block = f"Patient {patient_id}:\n{context_text}"
    user_prompt = textwrap.dedent(
        f"""
        Prediction time: {pred_time_str}. Consider all events in the patient's medical timeline occurring on or before this time.

        Predict whether the patient {prediction_target}.
        """
    )
    user_prompt += textwrap.dedent(
        f"""
        Patient timeline (most recent 4096 tokens, strictly before prediction time):

        {cohort_context_block}
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


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


def _cache_key(patient_id: str, prediction_time: str, task_name: str, model: Optional[str] = None) -> tuple:
    """Cache key for API responses (LLM-only, so no use_tools in key)."""
    if model:
        return (patient_id, prediction_time, task_name, model)
    return (patient_id, prediction_time, task_name)


def _parse_retry_seconds(exc: Exception) -> int:
    """Parse 'Try again in X seconds' from error message. Default 60."""
    msg = str(exc)
    m = re.search(r"(?:try again in|wait)\s+(\d+)\s*seconds?", msg, re.I)
    if m:
        return max(int(m.group(1)), 10)
    return 60


def _is_retryable_error(exc: Exception) -> bool:
    """Check if error is retryable (429 rate limit, token limit, etc.)."""
    msg = str(exc).lower()
    if getattr(exc, "status_code", None) == 429:
        return True
    if hasattr(exc, "response") and getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return True
    return "429" in msg or "token limit" in msg or "rate limit" in msg


async def run_single_prediction_llm_only(
    patient_id: str,
    prediction_time: str,
    task_name: str,
    question: str,
    precomputed_context_cache: dict,
    max_retries: int = 5,
    cache: Optional[dict] = None,
    debug: bool = False,
    model: Optional[str] = None,
) -> dict:
    """
    Call secure-llm API directly for one (patient, task): no cohort_chat, no tools, no tool loop.
    Uses precomputed_context_cache for context; builds same prompts as cohort_chat (Vista task, use_tools=False).
    """
    from meds_mcp.server.llm import (
        get_llm_client,
        get_default_generation_config,
        extract_response_content,
    )

    key = _cache_key(patient_id, prediction_time, task_name, model=model)
    if cache is not None and key in cache:
        cached = cache[key]
        if isinstance(cached, dict):
            return cached
        return {"answer": cached}

    context_cache_key = f"{patient_id}|{prediction_time}|{task_name}"
    context_text = precomputed_context_cache.get(context_cache_key)
    if not context_text:
        return {"answer": "[ERROR: not in context cache]"}

    messages = _build_llm_only_messages(
        task_name=task_name,
        prediction_time=prediction_time,
        patient_id=patient_id,
        context_text=context_text,
        question=question,
    )
    client = get_llm_client(model)
    model_name = model or "apim:gpt-4.1-mini"
    gen_cfg = get_default_generation_config(None)

    if debug:
        logging.info("Minimal LLM-only: messages built for %s %s (no tools)", patient_id, task_name)

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_name,
                messages=messages,
                **gen_cfg,
            )
            answer = extract_response_content(response)
            out = {"answer": answer}
            if cache is not None:
                cache[key] = out
            return out
        except Exception as e:
            last_exc = e
            if _is_retryable_error(e) and attempt < max_retries:
                wait_s = _parse_retry_seconds(e)
                logging.warning(
                    f"Retryable error for {patient_id}, waiting {wait_s}s before retry {attempt + 1}/{max_retries}: {e}"
                )
                await asyncio.sleep(wait_s)
                continue
            logging.error(f"Error for {patient_id} {task_name}: {e}")
            return {"answer": f"[ERROR: {str(e)}]"}
    return {"answer": f"[ERROR: {str(last_exc)}]"}


async def run_experiment_async(
    args,
    config: dict,
    precomputed_context_cache: dict,
) -> Tuple[int, str, str]:
    """Run minimal LLM-only experiment for lab_thrombocytopenia. Returns (total, output_path, summary_path)."""
    import csv as csv_module
    from meds_mcp.experiments.task_config import (
        get_csv_path_for_task,
        TASK_QUESTIONS,
        is_binary_task,
    )
    from meds_mcp.experiments.formatters import normalize_binary, ground_truth_to_normalized

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    task_name = TASK_NAME
    question = TASK_QUESTIONS.get(task_name, "Answer based on the patient timeline.")
    is_binary = is_binary_task(task_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(
        output_dir / f"minimal_llm_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    cache: dict = {}
    cache_path = Path(output_path).with_suffix(".cache.json")
    if not args.no_cache:
        cache = _load_cache(cache_path)

    csv_path = get_csv_path_for_task(task_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found for {task_name}: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as cf:
        reader = csv_module.DictReader(cf)
        for row in reader:
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

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
        if f"{patient_id}|{prediction_time}|{task_name}" not in precomputed_context_cache:
            skipped_not_in_cache += 1
            continue
        to_process.append(row)

    if skipped_not_in_cache:
        logging.info(f"Skipped {skipped_not_in_cache} rows (not in context cache)")
    if not to_process:
        logging.info("No rows to process (all done or not in cache)")
        return 0, output_path, str(Path(output_path).with_suffix("") + "_summary.json")

    sem = asyncio.Semaphore(args.batch_size)
    model_arg = getattr(args, "model", None)

    async def process_with_progress(row, pbar):
        async with sem:
            patient_id = row.get("patient_id")
            prediction_time = row.get("prediction_time")
            row_key = (patient_id, task_name, prediction_time)
            ground_truth_raw = row.get("ground_truth") or row.get("label", "")
            ground_truth_norm = ground_truth_to_normalized(ground_truth_raw, is_binary=True)

            result = await run_single_prediction_llm_only(
                patient_id=patient_id,
                prediction_time=prediction_time,
                task_name=task_name,
                question=question,
                precomputed_context_cache=precomputed_context_cache,
                max_retries=args.max_retries,
                cache=cache,
                debug=getattr(args, "debug", False),
                model=model_arg,
            )
            await asyncio.sleep(args.delay_seconds)
        if pbar:
            pbar.update(1)

        answer = result["answer"]
        norm_llm = normalize_binary(answer)
        record = {
            "patient_id": patient_id,
            "task_name": task_name,
            "prediction_time": prediction_time,
            "ground_truth_raw": ground_truth_raw,
            "ground_truth_normalized": ground_truth_norm,
            "llm_only_raw": answer,
            "llm_only_normalized": norm_llm,
        }
        return (row_key, record)

    run_start = time.perf_counter()
    if tqdm:
        pbar = tqdm(total=len(to_process), desc=task_name, unit="row")
    else:
        pbar = None

    tasks = [process_with_progress(row, pbar) for row in to_process]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if tqdm:
        pbar.close()

    total = 0
    write_buffer: list = []
    with open(output_path, write_mode) as f:
        for r in results:
            if isinstance(r, Exception):
                logging.error(f"Row failed: {r}")
                continue
            row_key, record = r
            write_buffer.append(record)
            total += 1
            completed.add(row_key)
            if len(write_buffer) >= args.write_batch_size:
                for rec in write_buffer:
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                write_buffer = []
        for rec in write_buffer:
            f.write(json.dumps(rec) + "\n")
        f.flush()

    run_elapsed = time.perf_counter() - run_start
    if not args.no_cache and cache:
        _save_cache(cache, cache_path)
        logging.info(f"Saved {len(cache)} cached responses to {cache_path}")

    summary_path = str(Path(output_path).with_suffix("")) + "_summary.json"
    summary = {
        "output_file": output_path,
        "task": task_name,
        "total_patients": total,
        "total_elapsed_seconds": round(run_elapsed, 2),
        "resumed": args.resume,
        "context_cache_entries_used": len(precomputed_context_cache),
    }
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    logging.info(f"Summary written to {summary_path}")

    return total, output_path, summary_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # run.py is at src/meds_mcp/experiments/minimal_LLM_only/run.py -> repo root is 5 parents up
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    parser = argparse.ArgumentParser(
        description="Minimal LLM-only experiment: lab_thrombocytopenia, no tools. Uses context_cache.json."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vista.yaml",
        help="Path to server config (for labels_dir, etc.).",
    )
    parser.add_argument(
        "--context-cache-path",
        type=str,
        default=None,
        help="Path to precomputed context_cache.json. Default: output-dir/context_cache.json",
    )
    parser.add_argument(
        "--precompute-context",
        action="store_true",
        help="Run precompute_context_cache.py for lab_thrombocytopenia before experiment.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows (for quick testing).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSONL results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Delay between API calls. Default: 0.5",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file. Requires --output.",
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
        default=1,
        help="Concurrent API calls. Default: 1 for minimal experiment.",
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
        help="Disable response caching.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full context before each LLM call.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name. If not set, uses server default.",
    )
    args = parser.parse_args()

    if args.resume and not args.output:
        logging.error("--resume requires --output")
        sys.exit(1)

    config = load_config(args.config)
    if config.get("labels_dir") and not os.environ.get("VISTA_LABELS_DIR"):
        os.environ["VISTA_LABELS_DIR"] = config["labels_dir"]

    if args.precompute_context:
        import subprocess
        ctx_cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "precompute_context_cache.py"),
            "--config", args.config,
            "--task", TASK_NAME,
            "--output-dir", args.output_dir,
        ]
        if args.limit:
            ctx_cmd.extend(["--limit", str(args.limit)])
        logging.info("Running precompute_context_cache.py for lab_thrombocytopenia...")
        subprocess.run(ctx_cmd, check=True)

    context_cache_path = (
        Path(args.context_cache_path)
        if args.context_cache_path
        else Path(args.output_dir) / "context_cache.json"
    )
    precomputed_context_cache = _load_context_cache(context_cache_path)
    if not precomputed_context_cache:
        logging.error(
            f"No context cache found at {context_cache_path}. "
            "Run with --precompute-context or build context_cache.json first (e.g. precompute_context_cache.py --task lab_thrombocytopenia)."
        )
        sys.exit(1)
    logging.info(f"Using context cache: {context_cache_path} ({len(precomputed_context_cache)} entries). Skipping document store init.")

    for _logger in ("securellm", "securellm.providers", "securellm.providers.apim_provider"):
        try:
            logging.getLogger(_logger).setLevel(logging.ERROR)
        except Exception:
            pass

    total, output_path, summary_path = asyncio.run(
        run_experiment_async(args, config, precomputed_context_cache)
    )
    logging.info(f"Done. Wrote {total} records to {output_path}, summary to {summary_path}")


if __name__ == "__main__":
    main()
