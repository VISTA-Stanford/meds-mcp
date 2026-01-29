"""
Base class for vignette generators.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseVignetteGenerator(ABC):
    """Abstract base class for vignette generators."""

    @abstractmethod
    def generate(
        self,
        patient_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        temporal_weighting: bool = False,
    ) -> str:
        """
        Generate a vignette (text representation) for a patient.

        Args:
            patient_id: The patient ID
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            temporal_weighting: Whether to weight events by recency

        Returns:
            String representation of the patient's events
        """
        pass
