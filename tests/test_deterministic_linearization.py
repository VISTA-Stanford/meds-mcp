# tests/test_vignette.py

import sys
from pathlib import Path

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from meds_mcp.similarity.deterministic_linearization import (
    DeterministicTimelineLinearizationGenerator,
)

vg = DeterministicTimelineLinearizationGenerator(xml_dir="data/collections/dev-corpus")

text = vg.generate(
    "115969130",
    start_date="2020-01-01",
    end_date="2024-01-01",
    temporal_weighting=False,
)

print(text[:1000])
