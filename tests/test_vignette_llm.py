# tests/test_vignette_llm.py

import os
import sys

from meds_mcp.similarity.deterministic_linearization import (
    DeterministicTimelineLinearizationGenerator,
)
from meds_mcp.similarity.vignette_llm import LLMVignetteGenerator
from meds_mcp.similarity.llm_secure_adapter import SecureLLMSummarizer


# Skip if secure LLM key not configured
if not os.getenv("VAULT_SECRET_KEY"):
    print("Skipping LLM vignette test: VAULT_SECRET_KEY not set")
    sys.exit(0)


base_vg = DeterministicTimelineLinearizationGenerator(xml_dir="data/collections/dev-corpus")
secure_llm = SecureLLMSummarizer(
    model="apim:gpt-4.1-mini",  # or any secure-llm model
    generation_overrides={"max_tokens": 512, "temperature": 0.1},
)

vg = LLMVignetteGenerator(
    base_generator=base_vg,
    llm=secure_llm,
)

out = vg.generate(
    "115969130",
    start_date="2021-10-01",
    end_date="2023-10-31",
    temporal_weighting=True,
)

print(out)


