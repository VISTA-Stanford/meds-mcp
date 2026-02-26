#!/usr/bin/env python3
"""
Debug: does the backend return content and logprobs when using response_format (structured output)?

Run from repo root:
    uv run python scripts/debug_structured_output_response.py
    uv run python scripts/debug_structured_output_response.py --model apim:gpt-4.1-mini

Prints message.content, choices[0].logprobs, and a sample of the raw response so we can see
why Vista bench with use_structured_output gets empty raw and null scores.
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Debug structured output + logprobs: inspect raw API response"
    )
    parser.add_argument(
        "--model",
        default="apim:gpt-4.1-mini",
        help="Model name",
    )
    args = parser.parse_args()

    from meds_mcp.server.llm import get_llm_client
    from meds_mcp.server.api.cohort_chat import (
        VISTA_LABEL_RESPONSE_FORMAT,
        _extract_score_positive_from_logprobs,
    )

    client = get_llm_client()
    model = args.model

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI system performing binary clinical outcome prediction. "
                "Answer with exactly one word: yes or no. Use the JSON schema provided."
            ),
        },
        {
            "role": "user",
            "content": "Is 2 greater than 1? Reply using the required JSON format.",
        },
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 32,
        "response_format": VISTA_LABEL_RESPONSE_FORMAT,
        "logprobs": True,
        "top_logprobs": 20,
    }

    print("Request: response_format (yes_no_answer schema) + logprobs=True, top_logprobs=20")
    print(f"Model: {model}\n")

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"Request failed: {e}")
        return 1

    # Normalize to dict for inspection
    resp_dict = _to_dict(response)
    choices = resp_dict.get("choices", [])
    if not choices:
        print("Response has no 'choices'.")
        print("Full response (truncated):", json.dumps(resp_dict, indent=2, default=str)[:1500])
        return 1

    c0 = _to_dict(choices[0])
    message = c0.get("message", c0)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    print("--- message.content ---")
    if content is None:
        print("  None")
    elif content == "":
        print("  (empty string)")
    else:
        print(f"  {content!r}")

    logprobs_obj = c0.get("logprobs")
    if logprobs_obj is None and hasattr(choices[0], "logprobs"):
        logprobs_obj = choices[0].logprobs
    if logprobs_obj is not None and not isinstance(logprobs_obj, dict):
        logprobs_obj = _to_dict(logprobs_obj) if hasattr(logprobs_obj, "__dict__") else None

    print("\n--- choices[0].logprobs ---")
    if logprobs_obj is None:
        print("  None (backend did not return logprobs with response_format)")
    else:
        print("  Present. Keys:", list(logprobs_obj.keys()) if isinstance(logprobs_obj, dict) else type(logprobs_obj))
        lp_content = logprobs_obj.get("content") if isinstance(logprobs_obj, dict) else None
        if lp_content is not None:
            print(f"  logprobs.content length: {len(lp_content)}")
            if lp_content:
                first = lp_content[0] if isinstance(lp_content[0], dict) else _to_dict(lp_content[0])
                print("  first entry keys:", list(first.keys()) if isinstance(first, dict) else first)
                top = (first.get("top_logprobs") or []) if isinstance(first, dict) else []
                print(f"  first entry top_logprobs count: {len(top)}")
                if top and isinstance(top[0], dict):
                    print("  first top_logprob sample:", top[0])
                elif top:
                    print("  first top_logprob sample:", _to_dict(top[0]))

    score = _extract_score_positive_from_logprobs(response, use_aggregate_yes_no=False)
    print(f"\n--- _extract_score_positive_from_logprobs(response) => {score!r} ---")

    # Sample of full response for debugging
    sample = {"choices": [c0]}
    out = json.dumps(sample, indent=2, default=str)
    print("\n--- choices[0] sample (for inspection) ---")
    print(out[:2500] if len(out) > 2500 else out)
    if len(out) > 2500:
        print("... (truncated)")

    print("\n--- Note ---")
    print("If Vista bench shows null raw/scores, the difference is likely long prompt (backend may return empty content or no logprobs for large requests). Run: uv run python scripts/run_vista_bench_experiment.py --config configs/ehrshot.yaml --limit 1 --debug")

    return 0


if __name__ == "__main__":
    sys.exit(main())
