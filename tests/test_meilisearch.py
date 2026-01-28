import pytest
from unittest.mock import Mock, MagicMock, patch

# Try to import meilisearch, skip test if not available
try:
    from meilisearch import Client
    from meilisearch.errors import MeilisearchCommunicationError
except ImportError:
    pytest.skip("meilisearch package is not installed", allow_module_level=True)


def test_search_with_mock():
    """Unit test using mocked Meilisearch interface."""
    # Create mock search results
    mock_basic_results = {
        "hits": [],
        "estimatedTotalHits": 100,
        "query": "",
        "limit": 10,
        "offset": 0,
        "processingTimeMs": 1
    }
    
    mock_faceted_results = {
        "hits": [
            {
                "patient_id": "12345",
                "gender": "Female",
                "age": 70,
                "age_range": "65+",
                "race": "White",
                "ethnicity": "Not Hispanic",
                "diagnosis_codes": ["E11.9", "I10"],
                "insurance_type": "Medicare"
            }
        ],
        "estimatedTotalHits": 25,
        "query": "",
        "limit": 5,
        "offset": 0,
        "processingTimeMs": 2,
        "facetDistribution": {
            "gender": {"Female": 15, "Male": 10},
            "age_range": {"65+": 20, "45-64": 5},
            "race": {"White": 18, "Black": 5, "Asian": 2},
            "ethnicity": {"Not Hispanic": 22, "Hispanic": 3},
            "diagnosis_codes": {"E11.9": 8, "I10": 12, "E78.5": 5}
        }
    }
    
    mock_filtered_results = {
        "hits": [
            {
                "patient_id": "12345",
                "gender": "Female",
                "age": 70,
                "age_range": "65+"
            }
        ],
        "estimatedTotalHits": 5,
        "query": "",
        "limit": 5,
        "offset": 0,
        "processingTimeMs": 1
    }
    
    # Create mock index
    mock_index = MagicMock()
    mock_index.search = Mock(side_effect=[
        mock_basic_results,
        mock_faceted_results,
        mock_filtered_results
    ])
    
    # Create mock client
    mock_client = MagicMock()
    mock_client.index = Mock(return_value=mock_index)
    
    # Test basic search
    index = mock_client.index("patients")
    results = index.search("")
    assert results["estimatedTotalHits"] == 100
    
    # Test search with facets
    results = index.search(
        "",
        {
            "facets": ["gender", "age_range", "race", "ethnicity", "diagnosis_codes"],
            "limit": 5
        }
    )
    assert results["estimatedTotalHits"] == 25
    assert "facetDistribution" in results
    assert "gender" in results["facetDistribution"]
    assert results["facetDistribution"]["gender"]["Female"] == 15
    assert len(results["hits"]) == 1
    assert results["hits"][0]["patient_id"] == "12345"
    
    # Test search with filters
    filtered_results = index.search(
        "",
        {
            "filter": "gender = 'Female' AND age_range = '65+'",
            "limit": 5
        }
    )
    assert filtered_results["estimatedTotalHits"] == 5
    assert len(filtered_results["hits"]) == 1
    assert filtered_results["hits"][0]["gender"] == "Female"
    assert filtered_results["hits"][0]["age_range"] == "65+"
    
    # Verify mock was called correctly
    assert mock_client.index.call_count == 1  # Called once to get the index
    assert mock_index.search.call_count == 3  # Called 3 times for different searches


def test_mcp_meilisearch_client_with_mock():
    """Test MCPMeiliSearch wrapper class with mocked Meilisearch."""
    from meds_mcp.server.tools.meilisearch_client import MCPMeiliSearch
    
    # Create mock search results
    mock_results = {
        "hits": [{"patient_id": "123", "gender": "Male", "age": 45}],
        "estimatedTotalHits": 10,
        "query": "test",
        "limit": 10
    }
    
    # Create mock index
    mock_index = MagicMock()
    mock_index.search = Mock(return_value=mock_results)
    
    # Create mock client
    mock_client = MagicMock()
    mock_client.index = Mock(return_value=mock_index)
    mock_client.get_index = Mock(return_value=mock_index)
    mock_client.create_index = Mock(return_value=mock_index)
    
    # Patch the Client class
    with patch('meds_mcp.server.tools.meilisearch_client.Client', return_value=mock_client):
        meili = MCPMeiliSearch(host="http://localhost:7700", index_name="patients", reset=False)
        
        # Test search method
        results = meili.search(query="test", limit=10)
        assert results["estimatedTotalHits"] == 10
        assert len(results["hits"]) == 1
        assert results["hits"][0]["patient_id"] == "123"
        
        # Test search with filters
        results = meili.search(query="", filters="gender = 'Male'", facets=["gender"], limit=5)
        assert results["estimatedTotalHits"] == 10


def test_search():
    """Integration test that requires a real Meilisearch server."""
    client = Client("http://localhost:7700")
    index = client.index("patients")
    
    # Check if Meilisearch server is available
    try:
        # Try a simple operation to check connectivity
        index.search("", {"limit": 0})
    except MeilisearchCommunicationError:
        pytest.skip("Meilisearch server is not running on localhost:7700")
    
    # Basic search
    results = index.search("")
    print(f"Total patients: {results['estimatedTotalHits']}")
    
    # Search with facet distribution
    results = index.search(
        "",
        {
            "facets": ["gender", "age_range", "race", "ethnicity", "diagnosis_codes"],
            "limit": 5
        }
    )
    
    print("\nFacet distribution:")
    for facet, values in results["facetDistribution"].items():
        print(f"\n{facet}:")
        for value, count in values.items():
            print(f"  {value}: {count}")
    
    # Show a sample patient
    if results["hits"]:
        print("\nSample patient:")
        for key, value in results["hits"][0].items():
            print(f"  {key}: {value}")
    
    # Search with filters
    filtered_results = index.search(
        "",
        {
            "filter": "gender = 'Female' AND age_range = '65+'",
            "limit": 5
        }
    )
    
    print(f"\nFound {filtered_results['estimatedTotalHits']} elderly female patients")

if __name__ == "__main__":
    test_search()