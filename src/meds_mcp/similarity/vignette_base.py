"""Base class for vignette generators."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseVignetteGenerator(ABC):
    """Abstract base class for vignette generators."""

    @abstractmethod
    def generate(
        self,
        patient_id: str,
        cutoff_date: Optional[str] = None,
        n_encounters: Optional[int] = None,
    ) -> str:
        """Generate a vignette (text representation) for a patient.

        Args:
            patient_id: The patient ID
            cutoff_date: Optional landmark date (ISO format). Events after
                this date are excluded. YYYY-MM-DD is treated as inclusive
                of the full calendar day.
            n_encounters: Optional limit on the number of most-recent
                qualifying encounters to include.

        Returns:
            String representation of the patient's events
        """
        pass
