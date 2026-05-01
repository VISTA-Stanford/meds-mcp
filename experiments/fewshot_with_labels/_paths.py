"""
Path resolution for the fewshot_with_labels experiment.

All paths can be overridden via environment variables. If unset, they fall
back to the layout used during local development so existing invocations keep
working unchanged.

On import, this module also loads a ``.env`` file from the repo root (if one
exists) via ``python-dotenv``. Any env var already set in the current
process wins over the ``.env`` file — i.e. you can still override on the
command line or in a parent shell.

Env vars:
  VISTA_COHORT_CSV   Full path to the source CSV (see build_cohort.py).
  VISTA_CORPUS_DIR   Directory with per-patient {pid}.xml files.
  VISTA_OUTPUTS_DIR  Directory where patients.jsonl / items.jsonl /
                     experiment_results_*.jsonl live.
  VAULT_SECRET_KEY, and other secrets read directly by securellm.Client()
  — they are NOT referenced here, just loaded into the environment so
  downstream libraries can find them.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: experiments/fewshot_with_labels/_paths.py is 2 dirs deep under the repo.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load .env from the repo root if present. `override=False` means an env var
# already set in the shell wins over the file — standard dotenv convention.
try:
    from dotenv import load_dotenv as _load_dotenv

    _ENV_FILE = REPO_ROOT / ".env"
    if _ENV_FILE.exists():
        _load_dotenv(dotenv_path=_ENV_FILE, override=False)
except ImportError:  # pragma: no cover - python-dotenv is a project dependency
    pass

_LOCAL_CSV = (
    REPO_ROOT
    / "data/collections/vista_bench/bikia_dev-lumia_cohort_progression_tasks-000000000000.csv"
)
_LOCAL_CORPUS = REPO_ROOT / "data/collections/vista_bench/thoracic_cohort_lumia"
_LOCAL_OUTPUTS = REPO_ROOT / "experiments/fewshot_with_labels/outputs"


def _env_path(name: str, fallback: Path) -> Path:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else fallback


def cohort_csv() -> Path:
    """Resolve the cohort CSV path (VISTA_COHORT_CSV or local fallback)."""
    return _env_path("VISTA_COHORT_CSV", _LOCAL_CSV)


def corpus_dir() -> Path:
    """Resolve the XML corpus directory (VISTA_CORPUS_DIR or local fallback)."""
    return _env_path("VISTA_CORPUS_DIR", _LOCAL_CORPUS)


def outputs_dir() -> Path:
    """Resolve the outputs directory (VISTA_OUTPUTS_DIR or local fallback)."""
    return _env_path("VISTA_OUTPUTS_DIR", _LOCAL_OUTPUTS)


def patients_jsonl() -> Path:
    return _env_path("VISTA_PATIENTS_JSONL", outputs_dir() / "patients.jsonl")


def items_jsonl() -> Path:
    return _env_path("VISTA_ITEMS_JSONL", outputs_dir() / "items.jsonl")
