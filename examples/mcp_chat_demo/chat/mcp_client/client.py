"""
MCP Client functions for interacting with the MCP server.
"""

import asyncio
import logging
from typing import List, Dict, Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


async def load_patient_data_simple(patient_id: str, mcp_url: str):
    """Load patient data using MCP - 3 simple steps."""
    try:
        logger.info(f"🔄 Loading patient {patient_id}...")

        async with streamablehttp_client(mcp_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("✅ Connected to MCP server")

                # Step 1: Load patient timeline (ensure it's loaded on server)
                logger.info("📋 Loading patient timeline...")
                result = await session.call_tool(
                    "load_patient_timeline",
                    {"person_id": patient_id, "chunk_element": "event"},
                )
                timeline_result = result.structuredContent.get("result", {})

                if not timeline_result.get("success"):
                    error_msg = timeline_result.get("error", "Unknown error")
                    return False, f"Failed to load timeline: {error_msg}", []

                # Step 2: Get all patient events in one call
                logger.info("📄 Getting all patient events...")
                result = await session.call_tool(
                    "get_all_patient_events", {"person_id": patient_id}
                )
                events = result.structuredContent.get("result", [])

                logger.info(f"✅ Loaded {len(events)} events for patient {patient_id}")
                return True, f"Successfully loaded {len(events)} events", events

    except Exception as e:
        logger.error(f"❌ Error loading patient: {e}")
        return False, f"Error: {str(e)}", []


async def search_patient_events_simple(query: str, patient_id: str, mcp_url: str):
    """Search patient events using MCP."""
    try:
        async with streamablehttp_client(mcp_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "search_patient_events", {"query": query, "person_id": patient_id}
                )
                return result.structuredContent.get("result", [])

    except Exception as e:
        logger.error(f"Error searching patient events: {e}")
        return []


def test_connection_sync(mcp_url: str):
    """Test MCP connection."""
    try:

        async def test_connection():
            async with streamablehttp_client(mcp_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return True

        success = asyncio.run(test_connection())
        return (
            "✅ MCP server connection successful!"
            if success
            else "❌ Connection failed"
        )
    except Exception as e:
        logger.error(f"Connection test error: {e}")
        return f"❌ Connection test failed: {str(e)}"
