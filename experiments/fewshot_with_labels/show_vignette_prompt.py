#!/usr/bin/env python3
"""
Print the vignette-generation prompt and precomputed vignette for a single
(person_id, task) pair.

Shows:
  1. The system prompt (vignette generation template filled with task question
     and focus).
  2. The precomputed vignette stored in patients.jsonl (if present), or
     "[no vignette precomputed]" if the vignette field is empty.

Pass --show-timeline to also print the raw LUMIA linearization that was used
as the LLM user message (requires --corpus-dir).

Examples:
  uv run python experiments/fewshot_with_labels/show_vignette_prompt.py \\
    --person-id 115973549 --task guo_readmission \\
    --patients experiments/fewshot_with_labels/outputs/ehrshot/patients.jsonl \\
    --items experiments/fewshot_with_labels/outputs/ehrshot/items.jsonl

  # Also show the raw LUMIA timeline that was fed to the vignette LLM:
  uv run python experiments/fewshot_with_labels/show_vignette_prompt.py \\
    --person-id 115973549 --task guo_readmission --show-timeline \\
    --corpus-dir data/ehrshot_lumia/meds_corpus
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
TASK_QUESTIONS = _tc.TASK_QUESTIONS

from experiments.fewshot_with_labels import _paths

_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts"


_EHRSHOT_TASKS = frozenset(_tc.BINARY_TASKS)
_VISTA_TASKS = frozenset(t for t in _tc.TASK_DESCRIPTIONS if t not in _EHRSHOT_TASKS)

_TEMPLATE_FILES = {
    "ehrshot": _PROMPTS_DIR / "vignette_prompt_EHRSHOT.txt",
    "vista":   _PROMPTS_DIR / "vignette_prompt_VISTA.txt",
    "generic": _PROMPTS_DIR / "vignette_prompt_generic.txt",
}


def _load_template(task: str, template_path: Path = None) -> str:
    """Return the vignette prompt template for ``task``.

    If ``template_path`` is given it is used unconditionally. Otherwise
    selects EHRSHOT / VISTA / generic based on the task name.
    """
    if template_path is not None:
        if not template_path.exists():
            raise SystemExit(f"Template not found: {template_path}")
        return template_path.read_text().strip()

    if task in _EHRSHOT_TASKS:
        key = "ehrshot"
    elif task in _VISTA_TASKS:
        key = "vista"
    else:
        key = "generic"

    path = _TEMPLATE_FILES[key]
    if not path.exists():
        raise SystemExit(f"Prompt template not found: {path}")
    return path.read_text().strip()


def _render_task(item, store, template: str | None, show_timeline: bool, corpus_dir: Path) -> str:
    """Render the full output block for one (person_id, task) item."""
    cohort_item = store.join(item.person_id, item.task)
    vignette = cohort_item.state.vignette_for_task(item.task)

    resolved_template = template if template is not None else _load_template(item.task)
    task_question = item.question.strip() or TASK_QUESTIONS.get(item.task, "")
    system_prompt = resolved_template.format(
        TASK_QUESTION=task_question,
        TASK_FOCUS=TASK_DESCRIPTIONS[item.task].strip(),
    )

    lines = [
        "=" * 80,
        f"PID  : {item.person_id}",
        f"TASK : {item.task}",
        f"EMBED: {item.embed_time}",
        f"LABEL: {item.label} ({item.label_description})",
        "=" * 80,
        "",
        "[VIGNETTE GENERATION PROMPT]",
        "",
        system_prompt,
        "",
        "=" * 80,
        "",
        "[GENERATED VIGNETTE]",
        "",
        vignette if vignette.strip() else "[no vignette precomputed]",
    ]

    if show_timeline:
        gen = DeterministicTimelineLinearizationGenerator(str(corpus_dir))
        timeline = gen.generate(item.person_id, cutoff_date=item.embed_time)
        demos = demographics_block(
            xml_dir=str(corpus_dir),
            patient_id=item.person_id,
            cutoff_date=item.embed_time,
        )
        user_msg = (demos + "\n" + timeline) if demos else timeline
        lines += [
            "",
            "=" * 80,
            "",
            f"[RAW LUMIA TIMELINE]  ({len(user_msg)} chars)",
            "",
            user_msg,
        ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-id", required=True, type=str)
    parser.add_argument("--task", default=None, type=str,
                        help="Task name. If omitted, all tasks for this patient are written "
                             "to a folder under outputs/patient_<person_id>/.")
    parser.add_argument("--patients", type=Path, default=_paths.patients_jsonl())
    parser.add_argument("--items", type=Path, default=_paths.items_jsonl())
    parser.add_argument("--corpus-dir", type=Path, default=_paths.corpus_dir())
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Path to a vignette prompt template. Defaults to vignette_prompt_EHRSHOT.txt.",
    )
    parser.add_argument(
        "--show-timeline",
        action="store_true",
        help="Also include the raw LUMIA linearization (requires --corpus-dir).",
    )
    args = parser.parse_args()

    if args.task is not None and args.task not in TASK_DESCRIPTIONS:
        raise SystemExit(
            f"Unknown task {args.task!r}. Known: {sorted(TASK_DESCRIPTIONS)}"
        )

    store = CohortStore.load(args.patients, args.items)
    # Template is resolved per-task in all-tasks mode; resolved once here in single-task mode.
    template = _load_template(args.task, args.template) if args.task else None

    if args.task is not None:
        # Single-task mode: print to stdout.
        matches = [it for it in store.items_for_patient(args.person_id) if it.task == args.task]
        if not matches:
            raise SystemExit(f"No item for person_id={args.person_id} task={args.task}.")
        print(_render_task(matches[0], store, template, args.show_timeline, args.corpus_dir))
    else:
        # All-tasks mode: write one file per task into outputs/patient_<pid>/.
        all_items = store.items_for_patient(args.person_id)
        if not all_items:
            raise SystemExit(f"No items found for person_id={args.person_id}.")

        # Deduplicate: one item per task (store.join uses the same dedup logic).
        unique_tasks = sorted({it.task for it in all_items})

        out_dir = _paths.outputs_dir() / f"patient_{args.person_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for task in unique_tasks:
            item = store.join(args.person_id, task).item
            content = _render_task(item, store, template, args.show_timeline, args.corpus_dir)
            out_file = out_dir / f"{task}.txt"
            out_file.write_text(content)
            print(f"Wrote {out_file}")

        print(f"\nAll tasks for patient {args.person_id} written to {out_dir}")


if __name__ == "__main__":
    main()
