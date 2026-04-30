#!/usr/bin/env python3
"""
Print the vignette-generation prompt (system + user message) for a single
(person_id, task) pair. Does NOT call the LLM.

Pulls the task question from outputs/items.jsonl and the task focus from
``TASK_DESCRIPTIONS``, then renders the template at
``configs/prompts/vignette_prompt.example.txt`` (or ``vignette_prompt.txt``
if present). The user message is the deterministic linearized timeline
(demographics + events) up to the item's embed_time.

Examples:
  uv run python experiments/fewshot_with_labels/show_vignette_prompt.py \\
    --person-id 135908719 --task has_recurrence_1_yr
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_module(dotted: str, file_path: Path):
    """Import a submodule directly by file path, bypassing package __init__.py.

    The package ``meds_mcp.similarity.__init__`` imports ``bm25_retrieval``,
    which pulls in ``llama_index``. This script doesn't need any of that, so
    we register lightweight namespace packages for ``meds_mcp`` and
    ``meds_mcp.similarity`` and then load the few submodules we actually use.
    """
    spec = importlib.util.spec_from_file_location(dotted, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


# Register namespace packages for `meds_mcp` and `meds_mcp.similarity` so the
# real submodules' relative imports (e.g. `from .vignette_base import ...`)
# resolve without executing the heavyweight __init__.py files.
for _pkg, _path in [
    ("meds_mcp", _REPO_ROOT / "src/meds_mcp"),
    ("meds_mcp.similarity", _REPO_ROOT / "src/meds_mcp/similarity"),
    ("meds_mcp.experiments", _REPO_ROOT / "src/meds_mcp/experiments"),
]:
    if _pkg not in sys.modules:
        _ns = types.ModuleType(_pkg)
        _ns.__path__ = [str(_path)]
        sys.modules[_pkg] = _ns

_load_module(
    "meds_mcp.similarity.vignette_base",
    _REPO_ROOT / "src/meds_mcp/similarity/vignette_base.py",
)
_cohort = _load_module(
    "meds_mcp.similarity.cohort",
    _REPO_ROOT / "src/meds_mcp/similarity/cohort.py",
)
_det = _load_module(
    "meds_mcp.similarity.deterministic_linearization",
    _REPO_ROOT / "src/meds_mcp/similarity/deterministic_linearization.py",
)
_tc = _load_module(
    "meds_mcp.experiments.task_config",
    _REPO_ROOT / "src/meds_mcp/experiments/task_config.py",
)

CohortStore = _cohort.CohortStore
DeterministicTimelineLinearizationGenerator = _det.DeterministicTimelineLinearizationGenerator
demographics_block = _det.demographics_block
TASK_DESCRIPTIONS = _tc.TASK_DESCRIPTIONS

from experiments.fewshot_with_labels import _paths

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts"


def _load_template() -> str:
    """Load the vignette prompt template without importing the LLM client."""
    for candidate in (_PROMPTS_DIR / "vignette_prompt.txt", _PROMPTS_DIR / "vignette_prompt.example.txt"):
        if candidate.exists():
            return candidate.read_text().strip()
    raise SystemExit(f"No vignette prompt template found under {_PROMPTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-id", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--patients", type=Path, default=_paths.patients_jsonl())
    parser.add_argument("--items", type=Path, default=_paths.items_jsonl())
    parser.add_argument("--corpus-dir", type=Path, default=_paths.corpus_dir())
    args = parser.parse_args()

    if args.task not in TASK_DESCRIPTIONS:
        raise SystemExit(
            f"Unknown task {args.task!r}. Known: {sorted(TASK_DESCRIPTIONS)}"
        )

    store = CohortStore.load(args.patients, args.items)
    matches = [
        it
        for it in store.items_for_patient(args.person_id)
        if it.task == args.task
    ]
    if not matches:
        raise SystemExit(
            f"No item for person_id={args.person_id} task={args.task}."
        )
    item = matches[0]

    system_prompt = _load_template().format(
        TASK_QUESTION=item.question.strip(),
        TASK_FOCUS=TASK_DESCRIPTIONS[args.task].strip(),
    )

    gen = DeterministicTimelineLinearizationGenerator(str(args.corpus_dir))
    timeline = gen.generate(args.person_id, cutoff_date=item.embed_time)
    demos = demographics_block(
        xml_dir=str(args.corpus_dir),
        patient_id=args.person_id,
        cutoff_date=item.embed_time,
    )
    user_msg = (demos + "\n" + timeline) if demos else timeline

    print("=" * 80)
    print(f"PID  : {args.person_id}")
    print(f"TASK : {args.task}")
    print(f"EMBED: {item.embed_time}")
    print(f"LABEL: {item.label} ({item.label_description})")
    print("=" * 80)
    print("\n[SYSTEM PROMPT]\n")
    print(system_prompt)
    print("\n" + "=" * 80)
    print(f"[USER MESSAGE]  ({len(user_msg)} chars)\n")
    print(user_msg)


if __name__ == "__main__":
    main()
