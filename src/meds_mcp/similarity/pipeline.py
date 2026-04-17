"""Patient similarity pipeline: timeline -> vignette -> BM25 retrieval."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .bm25_retrieval import PatientBM25Index, SimilarPatient
from .deterministic_linearization import DeterministicTimelineLinearizationGenerator
from .llm_secure_adapter import SecureLLMSummarizer
from .vignette_base import BaseVignetteGenerator
from .vignette_llm import LLMVignetteGenerator

logger = logging.getLogger(__name__)


@dataclass
class PatientRecord:
    """A patient with optional temporal filters for vignette generation.

    ``cutoff_date=None`` means no landmark filter (entire timeline).
    ``n_encounters=None`` means no encounter cap.
    """

    person_id: str
    cutoff_date: Optional[str] = None  # ISO date or datetime; None = no cutoff
    n_encounters: Optional[int] = None


def load_patient_records_from_csv(
    csv_path: Union[str, Path],
    *,
    person_id_col: str = "person_id",
    cutoff_col: Optional[str] = "embed_time",
    require_all_columns_populated: bool = False,
) -> List[PatientRecord]:
    """Load per-patient landmark dates from a CSV into ``PatientRecord`` objects.

    One record per distinct ``person_id`` (first row wins). A blank or missing
    value in ``cutoff_col`` yields ``cutoff_date=None`` — i.e. that patient's
    vignette will be generated from the full timeline.

    Args:
        csv_path: Path to the CSV file.
        person_id_col: Column containing the patient ID.
        cutoff_col: Column containing the landmark date (ISO ``YYYY-MM-DD`` or
            full ISO datetime). Pass ``None`` to ignore the CSV's cutoff and
            return all patients with ``cutoff_date=None``.
        require_all_columns_populated: If True, any ``person_id`` that appears
            in at least one row with a blank/missing value in any column is
            excluded entirely (matches the behavior originally in
            ``experiments/progression_subset/precompute_vignettes.py``).

    Returns:
        List of ``PatientRecord``. Order follows first-occurrence in the CSV.
    """
    csv_path = Path(csv_path)

    bad_pids: set[str] = set()
    if require_all_columns_populated:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                for col in fieldnames:
                    raw = row.get(col)
                    if raw is None or str(raw).strip() == "":
                        pid = str(row.get(person_id_col, "") or "").strip()
                        if pid:
                            bad_pids.add(pid)
                        break

    seen: set[str] = set()
    records: List[PatientRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get(person_id_col, "") or "").strip()
            if not pid or pid in seen or pid in bad_pids:
                continue
            seen.add(pid)

            cutoff: Optional[str] = None
            if cutoff_col is not None:
                raw_cut = str(row.get(cutoff_col, "") or "").strip()
                if raw_cut:
                    cutoff = raw_cut

            records.append(PatientRecord(person_id=pid, cutoff_date=cutoff))

    return records


class PatientSimilarityPipeline:
    """End-to-end pipeline: XML timelines -> vignettes -> BM25 index -> retrieval.

    Usage::

        pipeline = PatientSimilarityPipeline(
            xml_dir="data/corpus",
            model="apim:gpt-4.1-mini",
            n_encounters=2,
        )
        # Build index from a cohort (each patient filtered by their own cutoff)
        pipeline.build_index(corpus_records)
        # Find similar patients for a query
        results = pipeline.find_similar("patient_123", "2024-01-15", top_k=5)
    """

    def __init__(
        self,
        xml_dir: str,
        model: str = "apim:gpt-4.1-mini",
        n_encounters: Optional[int] = None,
        system_prompt: Optional[str] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
        use_llm_vignettes: bool = True,
    ):
        self._xml_dir = xml_dir
        self._n_encounters = n_encounters
        self._use_llm = use_llm_vignettes

        self._base_generator = DeterministicTimelineLinearizationGenerator(xml_dir)

        if use_llm_vignettes:
            self._summarizer = SecureLLMSummarizer(
                model=model,
                system_prompt=system_prompt,
                generation_overrides=generation_overrides
                or {"temperature": 0.2, "max_tokens": 1024},
            )
            self._generator: BaseVignetteGenerator = LLMVignetteGenerator(
                base_generator=self._base_generator,
                llm=self._summarizer,
            )
        else:
            self._generator = self._base_generator
            self._summarizer = None

        self._index: Optional[PatientBM25Index] = None
        self._vignette_cache: Dict[str, str] = {}

    @property
    def base_generator(self) -> BaseVignetteGenerator:
        """The raw (deterministic) timeline linearizer.

        Exposed so callers can retrieve the pre-LLM timeline text directly,
        e.g. for constructing prompts or diagnostics without re-summarizing.
        """
        return self._base_generator

    @property
    def summarizer(self) -> Optional[SecureLLMSummarizer]:
        """The LLM summarizer, or ``None`` if the pipeline was built with
        ``use_llm_vignettes=False``."""
        return self._summarizer

    def generate_vignette(
        self,
        person_id: str,
        cutoff_date: Optional[str] = None,
        n_encounters: Optional[int] = None,
    ) -> str:
        """Generate a single patient vignette.

        ``cutoff_date=None`` means no landmark filter (entire timeline).
        ``n_encounters=None`` falls back to the pipeline-level default.
        """
        n_enc = n_encounters if n_encounters is not None else self._n_encounters
        return self._generator.generate(
            patient_id=person_id,
            cutoff_date=cutoff_date,
            n_encounters=n_enc,
        )

    def build_index(
        self,
        records: List[PatientRecord],
        precomputed_vignettes: Optional[Dict[str, str]] = None,
    ) -> PatientBM25Index:
        """Build BM25 index from a list of patient records.

        Each patient's timeline is filtered by their own cutoff_date.

        Args:
            records: List of PatientRecord with person_id and cutoff_date.
            precomputed_vignettes: Optional dict of {person_id: vignette}
                to skip generation for already-computed patients.

        Returns:
            The built PatientBM25Index.
        """
        vignette_dicts: List[Dict[str, str]] = []

        for rec in records:
            pid = rec.person_id

            # Use precomputed vignette if available
            if precomputed_vignettes and pid in precomputed_vignettes:
                vig = precomputed_vignettes[pid]
            else:
                try:
                    vig = self.generate_vignette(
                        pid, rec.cutoff_date, rec.n_encounters
                    )
                except Exception:
                    logger.warning("Skipping %s: vignette generation failed", pid, exc_info=True)
                    continue

            if vig and vig.strip():
                vignette_dicts.append({"person_id": pid, "vignette": vig})
                self._vignette_cache[pid] = vig

        self._index = PatientBM25Index.from_vignettes(vignette_dicts)
        logger.info("Built BM25 index with %d patients", self._index.size)
        return self._index

    def find_similar(
        self,
        person_id: str,
        cutoff_date: Optional[str] = None,
        top_k: int = 5,
        n_encounters: Optional[int] = None,
        query_vignette: Optional[str] = None,
    ) -> List[SimilarPatient]:
        """Find top-k similar patients for a query patient.

        Args:
            person_id: Query patient ID.
            cutoff_date: Optional landmark date; ``None`` uses the full timeline.
            top_k: Number of similar patients to return.
            n_encounters: Override encounter limit for query patient.
            query_vignette: Pre-computed query vignette (skips generation).

        Returns:
            List of SimilarPatient with person_id, score, vignette.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if query_vignette is None:
            query_vignette = self.generate_vignette(
                person_id, cutoff_date, n_encounters
            )

        return self._index.search(
            query_vignette=query_vignette,
            top_k=top_k,
            exclude_person_id=person_id,
        )

    @property
    def vignette_cache(self) -> Dict[str, str]:
        """Vignettes generated during build_index, keyed by person_id."""
        return dict(self._vignette_cache)
