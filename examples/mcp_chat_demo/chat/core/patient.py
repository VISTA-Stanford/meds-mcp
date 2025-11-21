"""
Patient data management for the MCP Chat Demo.
"""

import asyncio
import logging
from typing import Tuple, Optional

import matplotlib.figure

from .session import session_state
from chat.visualization.timeline import TimelineManager
from chat.mcp_client.client import load_patient_data_simple

logger = logging.getLogger(__name__)

# Constants for status messages
NO_DATA_LOADED = "No data loaded"
NO_TIMELINE_DATA = "No timeline data"


async def load_patient_async(
    patient_id: str, mcp_url: str
) -> Tuple[str, str, Optional[matplotlib.figure.Figure], str, bool, bool]:
    """Load patient data asynchronously.
    
    Returns:
        Tuple containing (patient_id, message, fig, datetime_str, show_timeline, success)
        where success is True when timeline_visible is True (non-error case with timeline data).
    """
    try:
        success, message, events = await load_patient_data_simple(patient_id, mcp_url)

        # Reset session state
        session_state.reset_patient_data()
        session_state.current_patient_id = patient_id

        if success:
            session_state.patient_events = events
            session_state.timeline_loaded = True

            # Compute timeline
            session_state.token_df = TimelineManager.compute_timeline_from_events(
                events
            )

            # Set timestamps
            if session_state.token_df is not None and not session_state.token_df.empty:
                session_state.latest_timestamp = session_state.token_df.index.max()
                session_state.query_datetime = session_state.latest_timestamp

                datetime_str = session_state.latest_timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                fig = TimelineManager.update_timeline_plot()
                timeline_visible = True
                # Success when timeline is visible (non-error case with timeline data)
                load_success = bool(timeline_visible)

                return patient_id, message, fig, datetime_str, timeline_visible, load_success
            else:
                timeline_visible = False
                load_success = bool(timeline_visible)
                return patient_id, message, None, NO_TIMELINE_DATA, timeline_visible, load_success
        else:
            timeline_visible = False
            load_success = bool(timeline_visible)
            return patient_id, message, None, NO_DATA_LOADED, timeline_visible, load_success

    except Exception as e:
        logger.error(f"Error loading patient: {e}")
        timeline_visible = False
        load_success = bool(timeline_visible)
        return patient_id, f"Error: {str(e)}", None, NO_DATA_LOADED, timeline_visible, load_success


def load_patient_sync(
    patient_id: str, mcp_url: str
) -> Tuple[str, str, Optional[matplotlib.figure.Figure], str, bool, bool]:
    """Load patient data synchronously using asyncio.run (for non-async contexts).
    
    Returns:
        Tuple containing (patient_id, message, fig, datetime_str, show_timeline, success)
    """
    try:
        # Check if we're in an async context
        # get_running_loop() raises RuntimeError if no loop is running
        try:
            asyncio.get_running_loop()
            # We have a running loop, so we're in async context - can't use asyncio.run()
            raise ValueError(
                "Cannot use load_patient_sync() in async context. Use load_patient_async() instead."
            )
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(load_patient_async(patient_id, mcp_url))
    except Exception as e:
        logger.error(f"Error loading patient: {e}")
        timeline_visible = False
        load_success = bool(timeline_visible)
        return patient_id, f"Error: {str(e)}", None, NO_DATA_LOADED, timeline_visible, load_success
