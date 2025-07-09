#!/usr/bin/env python3
"""
Test MCP client using the official Python SDK with StreamableHTTP transport.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def test_mcp_server():
    """Test the MCP server using the official SDK."""
    
    print("🧪 Testing MEDS MCP Server with Official SDK")
    print("=" * 60)
    
    try:
        # Connect to the StreamableHTTP server
        async with streamablehttp_client("http://localhost:8000/mcp") as (
            read_stream,
            write_stream,
            _,
        ):
            # Create a session using the client streams
            async with ClientSession(read_stream, write_stream) as session:
                print("✅ Connected to MCP server")
                
                # Initialize the connection
                await session.initialize()
                print("✅ Initialized MCP session")
                
                # List available tools
                tools_response = await session.list_tools()
                print(f"✅ Available tools: {len(tools_response.tools)}")
                
                # Show tool names
                tool_names = [tool.name for tool in tools_response.tools]
                print(f"   Tools: {tool_names}")
                
                # Test load_patient_timeline tool
                if "load_patient_timeline" in tool_names:
                    print("\n📋 Testing load_patient_timeline...")
                    result = await session.call_tool(
                        "load_patient_timeline",
                        {"person_id": "135917824", "chunk_element": "event"}
                    )
                    print(f"✅ Loaded patient timeline: {result}")
                    
                    # Test listing patients
                    print("\n📋 Testing list_patients...")
                    patients_result = await session.call_tool("list_patients", {})
                    print(f"✅ Patients in store: {patients_result}")
                    
                    # Test search functionality
                    print("\n📋 Testing search_patient_events...")
                    search_result = await session.call_tool(
                        "search_patient_events",
                        {"query": "cancer", "person_id": "135917824"}
                    )
                    # Extract actual result from CallToolResult
                    search_data = search_result.structuredContent.get('result', [])
                    print(f"✅ Search results: {len(search_data)} events found")
                    if search_data:
                        print(f"   First result: {search_data[0]['id']}")
                    
                    # Test historical values
                    print("\n📋 Testing get_historical_values...")
                    historical_result = await session.call_tool(
                        "get_historical_values",
                        {"attribute_filters": {"code": "LOINC/8480-6"}, "person_id": "135917824"}
                    )
                    # Extract actual result from CallToolResult
                    historical_data = historical_result.structuredContent.get('result', [])
                    print(f"✅ Historical values: {len(historical_data)} readings found")
                    if historical_data:
                        print(f"   First reading: {historical_data[0]['timestamp']} - {historical_data[0]['value']}")
                    
                else:
                    print("❌ load_patient_timeline tool not found")
                
                print("\n✅ All tests completed successfully!")
                
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_server()) 