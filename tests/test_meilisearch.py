import pytest
from meilisearch import Client
from meilisearch.errors import MeilisearchCommunicationError

def test_search():
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