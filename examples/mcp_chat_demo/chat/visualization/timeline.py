"""
Timeline visualization and data management for the MCP Chat Demo.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt

from chat.utils.tokens import count_tokens
from .viz import plot_token_timeline
from chat.core.session import session_state

logger = logging.getLogger(__name__)


class TimelineManager:
    """Manages timeline visualization and data."""

    @staticmethod
    def update_timeline_plot(highlight_events: Optional[List[Dict[str, Any]]] = None):
        """Update the timeline plot with optional highlighted events."""
        if session_state.token_df is None:
            return None

        logger.debug(
            f"Updating timeline plot, query_datetime: {session_state.query_datetime}"
        )

        # Convert highlight events to DataFrame if provided
        highlight_df = None
        if highlight_events:
            highlight_times = []
            for event in highlight_events:
                if "timestamp" in event.get("metadata", {}):
                    try:
                        ts = pd.to_datetime(event["metadata"]["timestamp"]).floor("D")
                        highlight_times.append(ts)
                    except (ValueError, TypeError):
                        continue

            if highlight_times:
                highlight_df = pd.DataFrame(
                    data={
                        "token_length": [
                            session_state.token_df.loc[ts, "token_length"]
                            for ts in highlight_times
                        ]
                    },
                    index=pd.DatetimeIndex(highlight_times, name="bin"),
                )

        # Create new figure
        plt.clf()

        # Ensure query_datetime is a proper datetime object
        plot_query_datetime = None
        if session_state.query_datetime is not None:
            if isinstance(session_state.query_datetime, str):
                plot_query_datetime = pd.to_datetime(session_state.query_datetime)
            elif isinstance(session_state.query_datetime, pd.Timestamp):
                plot_query_datetime = session_state.query_datetime
            else:
                logger.warning(
                    f"Unexpected query_datetime type: {type(session_state.query_datetime)}"
                )

        fig = plot_token_timeline(
            session_state.token_df,
            highlight_df=highlight_df,
            bin_size="day",
            fig_width=16,
            highlight_color="#1F78B4",
            min_year_width_ratio=0.05,
            fig_height=1.0,
            year_line_offset_frac=0.15,
            year_line_pad_frac=0.01,
            query_datetime=plot_query_datetime,
        )

        fig.tight_layout(pad=1.0)
        return fig

    @staticmethod
    def compute_timeline_from_events(
        events: List[Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        """Compute token timeline from patient events."""
        if not events:
            return None

        # Create a timeline data structure similar to the original
        timeline_data = []
        for event in events:
            metadata = event.get("metadata", {})
            if "timestamp" in metadata:
                try:
                    timestamp = pd.to_datetime(metadata["timestamp"])
                    content = event.get("content", "")
                    token_count = count_tokens(content)
                    timeline_data.append(
                        {"timestamp": timestamp, "token_length": token_count}
                    )
                except (ValueError, TypeError):
                    continue

        if not timeline_data:
            return None

        # Convert to DataFrame and group by day
        df = pd.DataFrame(timeline_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Group by day and sum tokens
        daily_df = df.groupby(df.index.floor("D"))["token_length"].sum().to_frame()
        daily_df.index.name = "bin"

        return daily_df


def update_query_datetime(datetime_str: str):
    """Update the query datetime."""
    try:
        session_state.query_datetime = pd.to_datetime(datetime_str)
        logger.debug(f"Updated query_datetime to: {session_state.query_datetime}")
        fig = TimelineManager.update_timeline_plot()
        return session_state.query_datetime.strftime("%Y-%m-%d %H:%M:%S"), fig, True  # Timeline visible
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting datetime: {str(e)}")
        current_str = (
            session_state.query_datetime.strftime("%Y-%m-%d %H:%M:%S")
            if session_state.query_datetime
            else "No data loaded"
        )
        return current_str, None, False  # Timeline hidden
