import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml

from sentence_transformers import SentenceTransformer

from meds_mcp.similarity.deterministic_linearization import (
    DeterministicTimelineLinearizationGenerator,
)
from meds_mcp.similarity.vignette_llm import LLMVignetteGenerator
from meds_mcp.similarity.llm_secure_adapter import SecureLLMSummarizer


# -----------------------------
# Config
# -----------------------------
# Match server default: prefer config data.corpus_dir, then DATA_DIR env, then dev-corpus fallback
DEFAULT_CONFIG = Path("configs/medalign.yaml")

def resolve_corpus_dir():
    # 1) config file
    if DEFAULT_CONFIG.exists():
        try:
            with open(DEFAULT_CONFIG, "r") as f:
                cfg = yaml.safe_load(f) or {}
            corpus_dir = cfg.get("data", {}).get("corpus_dir")
            if corpus_dir:
                return Path(corpus_dir)
        except Exception:
            pass

    # 2) environment
    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        return Path(env_dir)

    # 3) fallback
    return Path("data/collections/dev-corpus")


XML_DIR = resolve_corpus_dir()
EMBEDDING_DIR = Path("data/embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Temporal settings (MUST match retrieval later)
START_DATE = None
END_DATE = None
TEMPORAL_WEIGHTING = True

USE_LLM = False

EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load vignette generator
# -----------------------------
base_vg = DeterministicTimelineLinearizationGenerator(xml_dir=XML_DIR)

if USE_LLM:
    secure_llm = SecureLLMSummarizer(
        model="apim:gpt-4.1-mini",
        generation_overrides={"temperature": 0.1, "max_tokens": 512},
    )
    vg = LLMVignetteGenerator(base_vg, secure_llm)
else:
    vg = base_vg


# -----------------------------
# Load embedding model
# -----------------------------
print(f"Loading embedding model: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME)


# -----------------------------
# Iterate XML files
# -----------------------------
xml_files = sorted(XML_DIR.glob("*.xml"))
if not xml_files:
    raise RuntimeError(f"No XML files found in {XML_DIR}")

print(f"Found {len(xml_files)} XML files")


for xml_path in tqdm(xml_files, desc="Generating embeddings"):
    patient_id = xml_path.stem
    out_path = EMBEDDING_DIR / f"{patient_id}.npy"

    if out_path.exists():
        continue  # safe resume

    try:
        vignette = vg.generate(
            patient_id,
            start_date=START_DATE,
            end_date=END_DATE,
            temporal_weighting=TEMPORAL_WEIGHTING,
        )

        if not vignette.strip():
            print(f"⚠️ Empty vignette for {patient_id}, skipping")
            continue

        embedding = embedder.encode(
            vignette,
            normalize_embeddings=True,
        )

        np.save(out_path, embedding.astype("float32"))

    except Exception as e:
        print(f"❌ Failed for {patient_id}: {e}")


print("✅ Embedding generation complete")
