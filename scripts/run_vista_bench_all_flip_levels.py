#!/usr/bin/env python3
"""
Run vista_bench experiment sequentially for all 4 label flip levels.

Label sets (by % labels flipped): 0%, 25%, 50%, 100%.
Equivalently, % correct labels: 100%, 75%, 50%, 0%.

Each run uses the corresponding labels dir and writes to a separate output subdir
under --output-base (e.g. results/run_all_flip/0pct_flip/, 25pct_flip/, ...).

Usage:
  # From repo root (meds-mcp/)
  uv run python scripts/run_vista_bench_all_flip_levels.py --config configs/ehrshot.yaml
  uv run python scripts/run_vista_bench_all_flip_levels.py --config configs/ehrshot.yaml --limit 10
  uv run python scripts/run_vista_bench_all_flip_levels.py --config configs/ehrshot.yaml --output-base results/run_all_flip
  uv run python scripts/run_vista_bench_all_flip_levels.py --config configs/ehrshot.yaml --context-cache-path results/context_cache.json  # reuse cache; skip document store init for all 4 runs
  uv run python scripts/run_vista_bench_all_flip_levels.py --config configs/ehrshot.yaml --delay-seconds 1.0  # pass-through to experiment script
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Label dir names under data/collections/ehrshot/labels/ (0% flip -> 100% correct, etc.)
FLIP_LEVELS = [
    ("0pct_flip", "labels_100_0pct_flip"),   # 100% correct
    ("25pct_flip", "labels_100_25pct_flip"), # 75% correct
    ("50pct_flip", "labels_100_50pct_flip"), # 50% correct
    ("100pct_flip", "labels_100_100pct_flip"), # 0% correct
]
# Numeric noise level for --tool-noise-level (for quadrant/Sankey analysis)
FLIP_TO_NOISE_LEVEL = {"0pct_flip": 0.0, "25pct_flip": 0.25, "50pct_flip": 0.5, "100pct_flip": 1.0}


def main():
    parser = argparse.ArgumentParser(
        description="Run vista_bench for all 4 flip levels (0%%, 25%%, 50%%, 100%% flip) sequentially."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ehrshot.yaml",
        help="Path to config (e.g. configs/ehrshot.yaml)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="results/run_all_flip",
        help="Base directory for outputs; each run writes to <output-base>/<0pct_flip|25pct_flip|...>",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default=None,
        help="Root dir for label sets (default: <repo>/data/collections/ehrshot/labels).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run only this task (passed to run_vista_bench_experiment.py)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows per task (passed to run_vista_bench_experiment.py)",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Run precomputation before first experiment only",
    )
    parser.add_argument(
        "--precompute-context",
        action="store_true",
        help="Run precompute_context before first experiment only",
    )
    parser.add_argument(
        "--context-cache-path",
        type=str,
        default="results/context_cache.json",
        help="Path to precomputed context_cache.json; passed to each run (default: results/context_cache.json). Set to '' to use per-run output-dir cache only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not run",
    )
    args, extra = parser.parse_known_args()

    labels_root = Path(args.labels_root) if args.labels_root else _REPO_ROOT / "data" / "collections" / "ehrshot" / "labels"
    output_base = Path(args.output_base)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    script = _REPO_ROOT / "scripts" / "run_vista_bench_experiment.py"
    if not script.exists():
        print(f"Experiment script not found: {script}", file=sys.stderr)
        sys.exit(1)

    for i, (short_name, dir_name) in enumerate(FLIP_LEVELS):
        labels_dir = labels_root / dir_name
        if not labels_dir.exists():
            print(f"Skipping {short_name}: labels dir not found: {labels_dir}")
            continue
        out_dir = output_base / short_name
        out_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["VISTA_LABELS_DIR"] = str(labels_dir.resolve())

        cmd = [
            sys.executable,
            str(script),
            "--config", str(config_path),
            "--output-dir", str(out_dir),
            "--tool-noise-level", str(FLIP_TO_NOISE_LEVEL[short_name]),
        ]
        if args.context_cache_path.strip():
            path = Path(args.context_cache_path.strip())
            if not path.is_absolute():
                path = _REPO_ROOT / path
            cmd.extend(["--context-cache-path", str(path)])
        if args.task:
            cmd.extend(["--task", args.task])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.precompute and i == 0:
            cmd.append("--precompute")
        if args.precompute_context and i == 0:
            cmd.append("--precompute-context")
        cmd.extend(extra)

        print(f"\n[{i+1}/4] {short_name} (labels: {labels_dir})")
        print(f"  Output dir: {out_dir}")
        if args.dry_run:
            print("  Command:", " ".join(cmd))
            print("  Env: VISTA_LABELS_DIR=", env["VISTA_LABELS_DIR"])
            continue
        result = subprocess.run(cmd, env=env, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            print(f"  Failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
        print(f"  Done.")

    if not args.dry_run:
        print(f"\nAll 4 runs finished. Results under {output_base.resolve()}")


if __name__ == "__main__":
    main()
