#!/usr/bin/env python3
"""
Minimal test: does secure-llm return logprobs when requested?

Run from repo root:
    uv run python scripts/test_securellm_logprobs.py
    # or with a specific model:
    uv run python scripts/test_securellm_logprobs.py --model apim:gpt-4.1-mini

Uses top-level kwargs (logprobs=True, top_logprobs=5), same as cohort_chat via gen_cfg.
APIM at apim.stanfordhealthcare.org rejects extra_body; if logprobs are missing, the
backend or secure-llm is not forwarding/returning them (check VISTA-Stanford/secure-llm).
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _inspect(response, label: str):
    """Inspect response and return (resp_dict, has_logprobs)."""
    if hasattr(response, "model_dump"):
        resp_dict = response.model_dump()
    elif hasattr(response, "dict"):
        resp_dict = response.dict()
    elif isinstance(response, dict):
        resp_dict = response
    else:
        resp_dict = {"_raw_type": type(response).__name__, "_raw": str(response)[:500]}
    choices = resp_dict.get("choices", []) if isinstance(resp_dict, dict) else []
    c0 = choices[0] if choices and isinstance(choices[0], dict) else (getattr(choices[0], "__dict__", None) if choices else None) or {}
    has_lp = (c0.get("logprobs") if isinstance(c0, dict) else getattr(choices[0], "logprobs", None)) is not None
    return resp_dict, has_lp, choices, c0


def main():
    parser = argparse.ArgumentParser(description="Test if secure-llm returns logprobs")
    parser.add_argument(
        "--model",
        default="apim:gpt-4.1-mini",
        help="Model name for the completion request",
    )
    args = parser.parse_args()

    from meds_mcp.server.llm import get_llm_client
    from meds_mcp.server.api.cohort_chat import _extract_score_positive_from_logprobs

    client = get_llm_client()
    model = args.model

    messages = [
        {
            "role": "user",
            "content": "Answer with exactly one word: yes or no. Is the sky blue?",
        }
    ]

    base_kw = {"model": model, "messages": messages, "max_tokens": 10}

    # Top-level kwargs (same as cohort_chat). APIM rejects extra_body (400), so we only use this.
    print("Calling chat.completions.create with logprobs=True, top_logprobs=5...")
    print(f"Model: {model}\n")
    try:
        response = client.chat.completions.create(
            **base_kw,
            logprobs=True,
            top_logprobs=5,
        )
    except Exception as e:
        print(f"Request failed: {e}\n")
        response = None

    if response is None:
        print("No successful response to inspect.")
        return 1

    resp_dict, has_lp, choices, c0 = _inspect(response, "final")
    if not choices:
        print("Response has no 'choices' or choices is empty.")
        return 1

    print("\nFinal choices[0] keys:", list(c0.keys()) if c0 else "N/A")
    logprobs = c0.get("logprobs") if isinstance(c0, dict) else getattr(choices[0], "logprobs", None)
    if logprobs is None:
        print("\n>>> choices[0].logprobs is None or missing — secure-llm or backend did not return logprobs.")
        print("    This is why llm_only_score / llm_plus_tool_score are null in the experiment.")
        print("    Next: check VISTA-Stanford/secure-llm to see if it forwards logprobs to APIM; APIM may need to enable logprobs for this deployment.")
        return 0

    print("\n>>> choices[0].logprobs is present.")
    if isinstance(logprobs, dict):
        print("    logprobs keys:", list(logprobs.keys()))
        content = logprobs.get("content")
        if content is not None:
            print(f"    logprobs.content length: {len(content)}")
            if content:
                first = content[0]
                print("    first content entry keys:", list(first.keys()) if isinstance(first, dict) else type(first))
                if isinstance(first, dict) and "top_logprobs" in first:
                    print("    first entry has top_logprobs:", len(first.get("top_logprobs") or []))
    else:
        print("    logprobs type:", type(logprobs))

    score = _extract_score_positive_from_logprobs(response)
    print(f"\n_extract_score_positive_from_logprobs(response) => {score!r}")

    if isinstance(resp_dict, dict) and resp_dict.get("choices"):
        sample = {"choices": [resp_dict["choices"][0]]}
        out = json.dumps(sample, indent=2, default=str)
        print("\nSample (choices[0] only) for inspection:")
        print(out[:2000] if len(out) > 2000 else out)
        if len(out) > 2000:
            print("... (truncated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
