#!/usr/bin/env python3
"""
Run three variants on sampled progression_subset rows:
  v1: LLM only (query timeline up to embed_time)
  v2: + similar patients; each similar's context chopped at that patient's embed_time
  v3: + similar patients; each similar's context chopped at query patient's embed_time

Requires precomputed vignettes.jsonl (BM25 over vignettes) and Lumia XML under corpus_dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.server.llm import (
    extract_response_content,
    get_default_generation_config,
    get_llm_client,
)
from meds_mcp.similarity import PatientBM25Index, PatientSimilarityPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_TRI = (
    "You answer clinical prediction questions using only the evidence provided in the user message. "
    "Respond with exactly one token and nothing else: -1, 0, or 1.\n"
    "-1 means insufficient information to answer.\n"
    "0 means no, or the event did not occur.\n"
    "1 means yes, or the event occurred.\n"
    "Do not include reasoning, punctuation, or any other text."
)


def load_vignettes_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["person_id"])] = obj
    return out


def trunc_text(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + "\n...[truncated]...\n" + s[-half:]


def parse_tri_label(text: Optional[str]) -> Optional[str]:
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    if s in ("-1", "0", "1"):
        return s
    m = re.search(r"\b(-1|0|1)\b", s)
    if m:
        return m.group(1)
    return None


def ensure_query_vignette(
    pipeline: PatientSimilarityPipeline,
    embed_time: str,
    n_encounters: int,
    jsonl_vignettes: dict[str, dict[str, Any]],
    pid: str,
    computed: dict[str, str],
) -> str:
    if pid in computed:
        return computed[pid]
    if pid in jsonl_vignettes and jsonl_vignettes[pid].get("vignette"):
        v = jsonl_vignettes[pid]["vignette"]
        computed[pid] = v
        return v
    v = pipeline.generate_vignette(pid, cutoff_date=embed_time, n_encounters=n_encounters)
    computed[pid] = v
    return v


def run_llm(
    client: Any,
    model: str,
    user_prompt: str,
    max_tokens: int = 32,
) -> str:
    gen = get_default_generation_config({"temperature": 0.0, "max_tokens": max_tokens})
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_TRI},
            {"role": "user", "content": user_prompt},
        ],
        **gen,
    )
    return extract_response_content(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run progression subset experiment (v1/v2/v3)")
    parser.add_argument(
        "--sampled-csv",
        type=Path,
        default=_REPO_ROOT / "experiments/progression_subset/outputs/sampled_rows.csv",
    )
    parser.add_argument(
        "--vignettes-jsonl",
        type=Path,
        default=_REPO_ROOT / "experiments/progression_subset/outputs/vignettes.jsonl",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_REPO_ROOT / "data/collections/vista_bench/thoracic_cohort_lumia",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "experiments/progression_subset/outputs",
    )
    parser.add_argument("--n-encounters", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5, help="Similar patients to include")
    parser.add_argument("--max-chars", type=int, default=120_000, help="Max chars per timeline block")
    parser.add_argument("--model", type=str, default="apim:gpt-4.1-mini")
    parser.add_argument("--delay-seconds", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None, help="Max rows from sampled CSV")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vignettes = load_vignettes_jsonl(args.vignettes_jsonl)
    bm25_index = PatientBM25Index.from_vignettes(list(vignettes.values()))
    pipeline = PatientSimilarityPipeline(
        xml_dir=str(args.corpus_dir),
        model=args.model,
        n_encounters=args.n_encounters,
        generation_overrides={"temperature": 0.2, "max_tokens": 1024},
    )
    base_generator = pipeline.base_generator
    client = get_llm_client(args.model)

    rows: list[dict[str, str]] = []
    with open(args.sampled_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    if args.limit:
        rows = rows[: args.limit]

    out_jsonl = args.output_dir / "experiment_results.jsonl"
    t0 = time.perf_counter()
    computed_vignettes: dict[str, str] = {}

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            pid = str(row["person_id"]).strip()
            et = str(row["embed_time"]).strip()
            task = str(row.get("task", "")).strip()
            question = str(row.get("question", "")).strip()
            label = str(row.get("label", "")).strip()
            xml_path = args.corpus_dir / f"{pid}.xml"
            if not xml_path.exists():
                logger.warning("Missing XML for %s, skipping", pid)
                continue

            q_vig = ensure_query_vignette(
                pipeline,
                et,
                args.n_encounters,
                vignettes,
                pid,
                computed_vignettes,
            )
            if not q_vig.strip():
                logger.warning("Empty query vignette for %s", pid)

            query_timeline = trunc_text(
                base_generator.generate(pid, cutoff_date=et), args.max_chars
            )

            sims = (
                bm25_index.search(q_vig, top_k=args.top_k, exclude_person_id=pid)
                if q_vig.strip()
                else []
            )

            blocks_v2: list[str] = []
            blocks_v3: list[str] = []
            for sim in sims:
                sid = sim.person_id
                s_xml = args.corpus_dir / f"{sid}.xml"
                if not s_xml.exists():
                    continue
                s_et = str(vignettes.get(sid, {}).get("embed_time") or "").strip()
                if not s_et:
                    continue
                t2 = trunc_text(base_generator.generate(sid, cutoff_date=s_et), args.max_chars)
                blocks_v2.append(f"Similar patient {sid} (context up to this patient's embed time {s_et}):\n{t2}")
                t3 = trunc_text(base_generator.generate(sid, cutoff_date=et), args.max_chars)
                blocks_v3.append(f"Similar patient {sid} (context up to query embed time {et}):\n{t3}")

            base_user = (
                f"QUESTION:\n{question}\n\n"
                f"Query patient {pid} — prediction time (embed time): {et}\n"
                f"Task: {task}\n\n"
                f"Query patient timeline (events on or before prediction time):\n{query_timeline}\n"
            )

            u1 = base_user
            u2 = base_user
            if blocks_v2:
                u2 += "\n---\nSimilar patients (for reference; timelines truncated if long):\n" + "\n\n".join(
                    blocks_v2
                )
            u3 = base_user
            if blocks_v3:
                u3 += "\n---\nSimilar patients (for reference; timelines truncated if long):\n" + "\n\n".join(
                    blocks_v3
                )

            raw1 = run_llm(client, args.model, u1)
            raw2 = run_llm(client, args.model, u2)
            raw3 = run_llm(client, args.model, u3)
            time.sleep(args.delay_seconds)

            p1 = parse_tri_label(raw1)
            p2 = parse_tri_label(raw2)
            p3 = parse_tri_label(raw3)

            rec = {
                "person_id": pid,
                "embed_time": et,
                "task": task,
                "label": label,
                "question": question,
                "pred_v1": p1,
                "pred_v2": p2,
                "pred_v3": p3,
                "raw_v1": raw1,
                "raw_v2": raw2,
                "raw_v3": raw3,
                "similar_patient_ids": [s.person_id for s in sims],
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            logger.info("Row %d/%d %s task=%s v1=%s v2=%s v3=%s", i + 1, len(rows), pid, task, p1, p2, p3)

    elapsed = time.perf_counter() - t0
    meta = {
        "output": str(out_jsonl),
        "n_rows": len(rows),
        "elapsed_seconds": round(elapsed, 2),
        "model": args.model,
        "top_k": args.top_k,
        "n_encounters": args.n_encounters,
    }
    with open(args.output_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_jsonl} ({len(rows)} rows) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
