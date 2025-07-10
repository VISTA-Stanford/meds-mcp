#!/usr/bin/env python3
"""
Test MCP client using the official Python SDK with StreamableHTTP transport.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

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
                        {"person_id": "135917824", "chunk_element": "event"},
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
                        {"query": "cancer", "person_id": "135917824"},
                    )
                    # Extract actual result from CallToolResult
                    search_data = search_result.structuredContent.get("result", [])
                    print(f"✅ Search results: {len(search_data)} events found")
                    if search_data:
                        print(f"   First result: {search_data[0]['id']}")

                    # Test historical values
                    print("\n📋 Testing get_historical_values...")
                    historical_result = await session.call_tool(
                        "get_historical_values",
                        {
                            "attribute_filters": {"code": "LOINC/8480-6"},
                            "person_id": "135917824",
                        },
                    )
                    # Extract actual result from CallToolResult
                    historical_data = historical_result.structuredContent.get(
                        "result", []
                    )
                    print(
                        f"✅ Historical values: {len(historical_data)} readings found"
                    )
                    if historical_data:
                        print(
                            f"   First reading: {historical_data[0]['timestamp']} - {historical_data[0]['value']}"
                        )
                    # ==============================
                    # Test Athena ontology tools
                    # ==============================
                    print("\n🏥 Testing Athena Ontology Tools...")
                    print("=" * 60)

                    # Define test codes
                    test_codes = [
                        "LOINC/8480-6",  # Systolic blood pressure
                        "SNOMED/363358000",  #  - Malignant tumor of lung
                        "ICD10/A41.9",  # Sepsis, unspecified organism
                    ]

                    async def test_ontology_tools(code: str):
                        """Test all ontology tools for a given code."""
                        print(f"\n📋 Testing code: {code}")
                        print("-" * 40)

                        # Test code metadata
                        print("🔍 Getting code metadata...")
                        metadata_result = await session.call_tool(
                            "get_code_metadata", {"code": code}
                        )
                        metadata = metadata_result.structuredContent.get("result", {})
                        if metadata:
                            print(f"   ✅ Code: {metadata.get('code', 'N/A')}")
                            print(
                                f"   ✅ Description: {metadata.get('description', 'N/A')}"
                            )
                            print(
                                f"   ✅ Vocabulary: {metadata.get('vocabulary', 'N/A')}"
                            )
                        else:
                            print("   ❌ No metadata found")

                        # Test ancestor subgraph
                        print("🔍 Getting ancestor subgraph...")
                        # Extract vocabulary from code (e.g., "LOINC/8480-6" -> "LOINC")
                        vocab = code.split("/")[0] if "/" in code else code
                        ancestor_result = await session.call_tool(
                            "get_ancestor_subgraph",
                            {"code": code, "vocabularies": [vocab]},
                        )
                        ancestor_data = ancestor_result.structuredContent.get(
                            "result", {}
                        )
                        if ancestor_data and "nodes" in ancestor_data:
                            nodes = ancestor_data["nodes"]
                            edges = ancestor_data.get("edges", [])
                            print(
                                f"   ✅ Ancestors: {len(nodes)} nodes, {len(edges)} edges"
                            )
                            if nodes:
                                ancestor_names = [
                                    n.get("name", n.get("code", "N/A"))
                                    for n in nodes[:3]
                                ]
                                print(f"   ✅ Top ancestors: {ancestor_names}")
                        else:
                            print("   ❌ No ancestor data found")

                        # Test descendant subgraph
                        print("🔍 Getting descendant subgraph...")
                        # Extract vocabulary from code (e.g., "LOINC/8480-6" -> "LOINC")
                        vocab = code.split("/")[0] if "/" in code else code
                        descendant_result = await session.call_tool(
                            "get_descendant_subgraph",
                            {"code": code, "vocabularies": [vocab]},
                        )
                        descendant_data = descendant_result.structuredContent.get(
                            "result", {}
                        )
                        if descendant_data and "nodes" in descendant_data:
                            nodes = descendant_data["nodes"]
                            edges = descendant_data.get("edges", [])
                            print(
                                f"   ✅ Descendants: {len(nodes)} nodes, {len(edges)} edges"
                            )
                            if nodes:
                                descendant_names = [
                                    n.get("name", n.get("code", "N/A"))
                                    for n in nodes[-3:]
                                ]
                                print(f"   ✅ Bottom descendants: {descendant_names}")
                        else:
                            print("   ❌ No descendant data found")

                    # Test all codes
                    for code in test_codes:
                        await test_ontology_tools(code)

                else:
                    print("❌ load_patient_timeline tool not found")

                print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
