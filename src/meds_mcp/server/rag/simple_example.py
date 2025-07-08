import time
import random
import statistics
from meds_mcp.server.rag.storage import MongoTimelineStore


def test_document_indexing(db: MongoTimelineStore, fpath: str) -> dict:
    """
    Test document indexing performance.

    Args:
        db: MongoTimelineStore instance
        fpath: Path to directory containing XML files

    Returns:
        Dictionary with indexing results and timing
    """
    print("Indexing document collection...")
    start_time = time.time()
    results = db.index_document_collection(fpath)
    indexing_time = time.time() - start_time

    print(f"Indexing completed in {indexing_time:.2f} seconds")
    print(f"Results: {results['successful']} successful, {results['failed']} failed")

    return {"indexing_time": indexing_time, "results": results}


def test_fetch_all_person_ids(db: MongoTimelineStore) -> dict:
    """
    Test fetching all person IDs performance.

    Args:
        db: MongoTimelineStore instance

    Returns:
        Dictionary with fetch results and timing
    """
    print("\nFetching all person IDs...")
    start_time = time.time()
    person_ids = db.get_indexed_person_ids()
    fetch_time = time.time() - start_time

    print(f"Fetched {len(person_ids)} person IDs in {fetch_time:.2f} seconds")

    return {
        "fetch_time": fetch_time,
        "person_ids": person_ids,
        "count": len(person_ids),
    }


def test_random_sampling_performance(
    db: MongoTimelineStore, person_ids: list, sample_size: int = 100
) -> dict:
    """
    Test random sampling performance with statistics.

    Args:
        db: MongoTimelineStore instance
        person_ids: List of available person IDs
        sample_size: Number of documents to sample

    Returns:
        Dictionary with sampling results and performance statistics
    """
    if len(person_ids) < sample_size:
        print(f"\nNot enough documents for sampling (only {len(person_ids)} available)")
        return {
            "error": f"Not enough documents for sampling (only {len(person_ids)} available)"
        }

    print(f"\nRandomly sampling {sample_size} patients...")

    # Randomly sample person_ids
    sampled_ids = random.sample(person_ids, sample_size)

    # Measure individual query times
    query_times = []
    successful_docs = []

    for person_id in sampled_ids:
        start_time = time.time()
        doc = db.get_document(person_id)
        query_time = time.time() - start_time

        query_times.append(query_time)
        if doc:
            successful_docs.append(doc)

    # Calculate statistics
    total_time = sum(query_times)
    mean_time = statistics.mean(query_times)
    min_time = min(query_times)
    max_time = max(query_times)
    median_time = statistics.median(query_times)

    print(f"Sampled {len(successful_docs)} documents in {total_time:.2f} seconds")
    print(f"Performance statistics:")
    print(f"  - Mean time per query: {mean_time:.4f} seconds")
    print(f"  - Median time per query: {median_time:.4f} seconds")
    print(f"  - Min time per query: {min_time:.4f} seconds")
    print(f"  - Max time per query: {max_time:.4f} seconds")

    # Show sample document info
    if successful_docs:
        print(f"\nSample document info:")
        print(f"  - Document ID: {successful_docs[0].doc_id}")
        print(f"  - Content length: {len(successful_docs[0].text)} characters")
        print(f"  - Metadata keys: {list(successful_docs[0].metadata.keys())}")

    return {
        "total_time": total_time,
        "mean_time": mean_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "successful_count": len(successful_docs),
        "query_times": query_times,
    }


def test_individual_document_retrieval(
    db: MongoTimelineStore, test_person_id: str = "135916722"
) -> dict:
    """
    Test individual document retrieval performance.

    Args:
        db: MongoTimelineStore instance
        test_person_id: Person ID to test retrieval for

    Returns:
        Dictionary with retrieval results and timing
    """
    print(f"\nTesting individual document retrieval...")
    start_time = time.time()
    doc = db.get_document(test_person_id)
    individual_fetch_time = time.time() - start_time

    print(f"Individual document fetch took {individual_fetch_time:.4f} seconds")
    print(f"Document found: {doc is not None}")
    print(f"Is person indexed: {db.is_person_indexed(test_person_id)}")

    return {
        "fetch_time": individual_fetch_time,
        "document_found": doc is not None,
        "is_indexed": db.is_person_indexed(test_person_id),
    }


def print_performance_summary(results: dict):
    """
    Print a summary of all performance results.

    Args:
        results: Dictionary containing all test results
    """
    print(f"\n=== Performance Summary ===")
    print(f"Indexing time: {results['indexing']['indexing_time']:.2f} seconds")
    print(f"Fetch all IDs time: {results['fetch_ids']['fetch_time']:.2f} seconds")

    if "sampling" in results and "error" not in results["sampling"]:
        sampling = results["sampling"]
        print(f"Random sampling time: {sampling['total_time']:.2f} seconds")
        print(f"  - Mean query time: {sampling['mean_time']:.4f} seconds")
        print(f"  - Median query time: {sampling['median_time']:.4f} seconds")
        print(f"  - Min query time: {sampling['min_time']:.4f} seconds")
        print(f"  - Max query time: {sampling['max_time']:.4f} seconds")

    print(f"Individual fetch time: {results['individual']['fetch_time']:.4f} seconds")


def print_full_document(db: MongoTimelineStore, person_id: str = None):
    """
    Fetch and print a full document to the console.

    Args:
        db: MongoTimelineStore instance
        person_id: Person ID to fetch. If None, uses the first available ID.
    """
    print(f"\n=== Full Document Display ===")

    # Get a person_id if none provided
    if person_id is None:
        person_ids = db.get_indexed_person_ids()
        if not person_ids:
            print("No documents available to display")
            return
        person_id = person_ids[0]
        print(f"Using first available person_id: {person_id}")

    # Fetch the document
    print(f"Fetching document for person_id: {person_id}")
    doc = db.get_document(person_id)

    if doc is None:
        print(f"Document not found for person_id: {person_id}")
        return

    # Print document details
    print(f"\nDocument ID: {doc.doc_id}")
    print(f"Content length: {len(doc.text)} characters")
    print(f"Metadata: {doc.metadata}")

    # Print the full document content
    print(f"\n=== Full Document Content ===")
    print(doc.text)
    print(f"=== End Document Content ===")


def main():
    """Main function that runs all tests."""
    fpath = "/Users/jfries/Code/lumia/data/collections/dev-corpus/"

    db = MongoTimelineStore(
        mongo_uri="mongodb://localhost:27017/",
        database_name="vista-dev",
        collection_name="starr",
    )

    # Run all tests
    results = {}

    # Test 1: Document indexing
    results["indexing"] = test_document_indexing(db, fpath)

    # Test 2: Fetch all person IDs
    results["fetch_ids"] = test_fetch_all_person_ids(db)

    # Test 3: Random sampling performance
    results["sampling"] = test_random_sampling_performance(
        db, results["fetch_ids"]["person_ids"]
    )

    # Test 4: Individual document retrieval
    results["individual"] = test_individual_document_retrieval(db)

    # Print summary
    print_performance_summary(results)

    # Display a full document
    print_full_document(db)

    # Cleanup
    print(f"\nCleaning up...")
    db.clear_all()
    print("Done")


if __name__ == "__main__":
    main()
