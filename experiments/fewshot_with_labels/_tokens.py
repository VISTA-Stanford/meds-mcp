"""
Token counting + truncation + budget utilities shared by
precompute_vignettes.py and run_experiment.py.

Uses tiktoken's ``cl100k_base`` encoding. The experiments run against
``apim:gpt-4.1-mini`` which technically uses ``o200k_base``; the drift between
the two tokenizers is small and in the SAFE direction (cl100k typically
over-counts clinical text by a few percent), so a budget computed with
cl100k tokens will reserve marginally MORE input headroom than strictly
necessary. This module deliberately fails loudly if tiktoken is unavailable
rather than falling back to a char/4 heuristic, because at the 60K–120K
budgets these experiments now use, a 10–25 % under-count from the heuristic
could push real payloads past the model's context window and silently cause
APIM to return an empty body.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "tiktoken is required for reliable token budgeting in the "
        "fewshot_with_labels experiment but could not be loaded. "
        "Install it (it is already in the project deps via `uv sync`) or "
        "run inside the uv-managed venv. Original error: "
        f"{_e!r}"
    )


def count_tokens(text: Optional[str]) -> int:
    """Number of tokens in ``text`` under cl100k_base."""
    if not text:
        return 0
    return len(_TOKENIZER.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> Tuple[str, int, bool]:
    """Head+tail truncate ``text`` so it fits ``max_tokens``.

    Keeps head and tail halves with an explicit middle-truncation marker, so
    the summarizer still sees the earliest and most recent events — usually
    what you want for oncology-trajectory summaries.

    Returns ``(truncated_text, original_token_count, was_truncated)``.
    """
    if max_tokens <= 0:
        return text, 0, False

    ids = _TOKENIZER.encode(text)
    if len(ids) <= max_tokens:
        return text, len(ids), False
    marker = "\n...[truncated — events in the middle of the timeline omitted]...\n"
    marker_ids = _TOKENIZER.encode(marker)
    budget = max_tokens - len(marker_ids)
    head = budget // 2
    tail = budget - head
    new_ids = ids[:head] + marker_ids + ids[-tail:]
    return _TOKENIZER.decode(new_ids), len(ids), True


def effective_input_budget(
    model_context_tokens: int,
    max_output_tokens: int,
    safety_margin: int,
    system_tokens: int = 0,
) -> int:
    """Compute the maximum input-tokens budget that leaves enough context
    for ``max_output_tokens`` of generation plus a ``safety_margin`` buffer.

    If ``system_tokens`` is provided, it is also subtracted (useful for
    run_experiment.py where the system prompt is a meaningful fraction of
    the total budget).

    Raises ``ValueError`` if the resulting budget is not positive — this
    typically means the caller passed an unreasonably small model context
    or oversized output/margin.
    """
    budget = (
        model_context_tokens
        - max_output_tokens
        - safety_margin
        - max(0, system_tokens)
    )
    if budget <= 0:
        raise ValueError(
            "effective_input_budget is not positive: "
            f"model_context_tokens={model_context_tokens}, "
            f"max_output_tokens={max_output_tokens}, "
            f"safety_margin={safety_margin}, "
            f"system_tokens={system_tokens} -> budget={budget}"
        )
    return budget


def summary_stats(values: list[int]) -> dict:
    """Return {"n","min","max","mean","median","p10","p90","total"} for a list of ints.
    Empty input returns all zeros."""
    n = len(values)
    if n == 0:
        return {"n": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "p10": 0, "p90": 0, "total": 0}
    s = sorted(values)
    total = sum(s)
    return {
        "n": n,
        "min": s[0],
        "max": s[-1],
        "mean": round(total / n, 2),
        "median": s[n // 2],
        "p10": s[max(0, int(n * 0.1) - 1)],
        "p90": s[min(n - 1, int(n * 0.9))],
        "total": total,
    }
