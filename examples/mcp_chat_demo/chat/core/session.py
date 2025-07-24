"""
Session state management for the MCP Chat Demo.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from mcp.client.session import ClientSession


class SessionState:
    """Manages the global session state for the demo."""

    def __init__(self):
        self.mcp_session: Optional[ClientSession] = None
        self.current_patient_id: Optional[str] = None
        self.patient_events: List[Dict[str, Any]] = []
        self.node_ids: List[str] = []
        self.token_df: Optional[pd.DataFrame] = None
        self.latest_timestamp: Optional[pd.Timestamp] = None
        self.query_datetime: Optional[pd.Timestamp] = None
        self.timeline_loaded: bool = False

    def reset_patient_data(self):
        """Reset patient-specific data."""
        self.current_patient_id = None
        self.patient_events = []
        self.node_ids = []
        self.token_df = None
        self.latest_timestamp = None
        self.query_datetime = None
        self.timeline_loaded = False


# Global session state instance
session_state = SessionState()
