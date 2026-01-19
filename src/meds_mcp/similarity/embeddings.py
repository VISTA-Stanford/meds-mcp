# src/meds_mcp/similarity/embeddings.py

"""
Embedding abstraction with cached vectors
"""

from pathlib import Path
from typing import Dict, Iterable
import numpy as np


class CachedEmbeddingStore:
    def __init__(self, embedding_dir: str, embedder=None):
        """
        embedder must expose:
          embed(texts: list[str]) -> list[list[float]]
        """
        self.dir = Path(embedding_dir)
        self.embedder = embedder

    def _path(self, pid: str) -> Path:
        return self.dir / f"{pid}.npy"

    def get(self, pid: str, text: str | None = None) -> np.ndarray:
        p = self._path(pid)
        if p.exists():
            return np.load(p)

        if self.embedder is None:
            raise RuntimeError(f"Missing embedding for {pid}")

        vec = np.array(self.embedder.embed([text])[0], dtype=np.float32)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, vec)
        return vec

    def batch_get(
        self,
        patient_ids: Iterable[str],
        texts: Dict[str, str],
    ) -> Dict[str, np.ndarray]:
        return {pid: self.get(pid, texts.get(pid)) for pid in patient_ids}

    def exists(self, patient_id: str) -> bool:
        return (self.dir / f"{patient_id}.npy").exists()