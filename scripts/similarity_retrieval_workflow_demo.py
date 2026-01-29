#!/usr/bin/env python3
"""
Simplified patient similarity workflow demo (generates indices online).

Steps:
1. Build global BM25 index from all patient vignettes (last N encounters → LLM)
2. Extract last N encounters from query patient
3. Generate LLM vignette from those encounters
4. Search global index and return top-k similar patients

Run: python scripts/simplified_workflow_demo.py --patient-id 115969130 --n-encounters 2
Note: Requires VAULT_SECRET_KEY environment variable for LLM access
      Generates indices online, ensuring consistency with n_encounters parameter

Usage:
    python scripts/similarity_retrieval_workflow_demo.py \
        --patient-id 115969130 \
        --corpus-dir data/collections/dev-corpus \
        --n-encounters 2 \
        --top-k 2 \
        --llm-model apim:gpt-4.1-mini
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core.schema import Document
from llama_index.retrievers.bm25 import BM25Retriever

from meds_mcp.similarity.llm_secure_adapter import SecureLLMSummarizer


def load_config():
    """Load corpus directory from config."""
    config_path = Path("configs/medalign.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("data", {}).get("corpus_dir", "data/collections/dev-corpus")
    return os.getenv("DATA_DIR", "data/collections/dev-corpus")


def extract_last_n_encounters(xml_path: Path, n_encounters: int = 2) -> str:
    """Extract events from the last N encounters from XML."""
    from lxml import etree

    root = etree.parse(str(xml_path)).getroot()
    encounters = root.findall("encounter")

    # Get last N encounters
    last_encounters = encounters[-n_encounters:] if len(encounters) >= n_encounters else encounters

    # Linearize only these encounters
    lines = []
    for encounter in last_encounters:
        events_elem = encounter.find("events")
        if events_elem is None:
            continue

        for entry in events_elem.findall("entry"):
            entry_ts = entry.attrib.get("timestamp", "UNK_TIME")
            for event in entry.findall("event"):
                etype = event.attrib.get("type", "")
                code = event.attrib.get("code", "")
                name = event.attrib.get("name", "")
                parts = [p for p in [etype, code, name] if p]
                if parts:
                    lines.append(f"[{entry_ts}] " + " | ".join(parts))

    return "\n".join(lines)


def build_global_vignette_index(
    corpus_path: Path,
    llm_adapter: SecureLLMSummarizer,
    n_encounters: int = 2,
    debug_output_dir: Optional[Path] = None,
) -> tuple[BM25Retriever, Dict[str, str]]:
    """Build global BM25 index from all patient vignettes.

    Args:
        corpus_path: Path to corpus directory with XML files
        llm_adapter: LLM adapter for vignette generation
        n_encounters: Number of last encounters to extract
        debug_output_dir: Optional directory to save vignettes for debugging

    Returns:
        Tuple of (BM25Retriever with all patient vignettes indexed, dict of patient_id -> vignette)
    """
    xml_files = sorted(corpus_path.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files in {corpus_path}")

    print(f"\nBuilding global vignette index from {len(xml_files)} patients...")
    documents = []
    failed_count = 0
    all_vignettes = {}

    for xml_path in tqdm(xml_files, desc="Generating vignettes"):
        patient_id = xml_path.stem

        try:
            # Extract last N encounters
            last_encounters = extract_last_n_encounters(xml_path, n_encounters)

            if not last_encounters.strip():
                failed_count += 1
                continue

            # Generate vignette with LLM
            vignette = llm_adapter.summarize(last_encounters)
            all_vignettes[patient_id] = vignette

            # Create document
            doc = Document(
                text=vignette,
                metadata={
                    "patient_id": patient_id,
                    "source_doc": patient_id,
                    "vignette_length": len(vignette),
                    "n_encounters": n_encounters,
                },
            )
            documents.append(doc)

        except Exception as e:
            failed_count += 1
            tqdm.write(f"  ⚠️  Failed {patient_id}: {e}")

    if not documents:
        raise ValueError(f"No documents generated (failed: {failed_count})")

    print(f"✓ Generated vignettes for {len(documents)} patients ({failed_count} failed)\n")

    # Save vignettes for debugging if output dir specified
    if debug_output_dir:
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        all_vignettes_path = debug_output_dir / "all_vignettes.txt"
        with open(all_vignettes_path, "w") as f:
            for patient_id, vignette in sorted(all_vignettes.items()):
                f.write(f"\n{'='*80}\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"{'='*80}\n")
                f.write(f"{vignette}\n")
        print(f"✓ Saved all vignettes to {all_vignettes_path}")

    # Build BM25 index
    print("Building BM25 retriever...")
    retriever = BM25Retriever.from_defaults(
        nodes=documents,
        similarity_top_k=len(documents),
    )
    print(f"✓ BM25 index built with {len(documents)} patient vignettes\n")

    return retriever, all_vignettes


class SimpleBM25Indexer:
    """BM25 indexer wrapper for search operations."""

    def __init__(self, retriever: Optional[BM25Retriever] = None):
        self.retriever = retriever

    def set_retriever(self, retriever: BM25Retriever):
        """Set the BM25 retriever."""
        self.retriever = retriever

    def search(self, query: str, top_k: int = 5, exclude_patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search and return top-k results, excluding a specific patient ID if provided.

        Args:
            query: Query text for BM25 search
            top_k: Number of top results to return
            exclude_patient_id: Patient ID to exclude from results (e.g., query patient)

        Returns:
            List of top-k results excluding the excluded patient
        """
        if not self.retriever:
            return []

        # BM25Retriever.retrieve() uses similarity_top_k attribute set during persistence
        # We need to temporarily adjust it for this search
        original_top_k = self.retriever.similarity_top_k
        try:
            # Retrieve all results if excluding patient, then filter
            # Otherwise retrieve just top_k
            if exclude_patient_id:
                # Retrieve all available results to account for excluded patient
                self.retriever.similarity_top_k = original_top_k
            else:
                self.retriever.similarity_top_k = top_k

            results = self.retriever.retrieve(query)

            # Filter out excluded patient if needed
            if exclude_patient_id:
                filtered_results = []
                for r in results:
                    patient_id = r.metadata.get("source_doc", r.metadata.get("patient_id", ""))
                    if patient_id == exclude_patient_id:
                        continue
                    filtered_results.append({
                        "patient_id": patient_id,
                        "score": r.score or 0.0,
                        "vignette": r.text,
                    })
                    if len(filtered_results) >= top_k:
                        break
                return filtered_results
            else:
                return [
                    {
                        "patient_id": r.metadata.get("source_doc", r.metadata.get("patient_id", "")),
                        "score": r.score or 0.0,
                        "vignette": r.text,
                    }
                    for r in results
                ]
        finally:
            # Restore original setting
            self.retriever.similarity_top_k = original_top_k


def main():
    parser = argparse.ArgumentParser(
        description="Simplified workflow demo: Build vignette index → Search for similar patients"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="115969130",
        help="Query patient ID",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Corpus directory (defaults to config or DATA_DIR)",
    )
    parser.add_argument(
        "--n-encounters",
        type=int,
        default=2,
        help="Number of last encounters to extract (default: 2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Return top-k similar patients",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="apim:gpt-4.1-mini",
        help="LLM model to use for vignette generation",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="data/vignette_debug",
        help="Directory to save vignettes for debugging (default: data/vignette_debug)",
    )

    args = parser.parse_args()

    # Check VAULT_SECRET_KEY
    if not os.getenv("VAULT_SECRET_KEY"):
        print("❌ VAULT_SECRET_KEY required for LLM vignette generation")
        return

    # Resolve corpus directory
    corpus_dir = args.corpus_dir or load_config()
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_dir}")
        return

    print("\n" + "="*80)
    print("SIMPLIFIED PATIENT SIMILARITY WORKFLOW DEMO")
    print("="*80)
    print(f"Corpus: {corpus_dir}")
    print(f"Query patient: {args.patient_id}")
    print(f"Last N encounters: {args.n_encounters}")
    print(f"LLM model: {args.llm_model}")
    print(f"Top-k results: {args.top_k}")
    print(f"Debug output dir: {args.debug_dir}")
    print("="*80)

    # Create debug directory
    debug_dir = Path(args.debug_dir)

    # Initialize LLM
    try:
        llm_adapter = SecureLLMSummarizer(
            model=args.llm_model,
            generation_overrides={"max_tokens": 512, "temperature": 0.1},
        )
        print(f"\n✓ LLM initialized with {args.llm_model}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return

    # Build global vignette index
    try:
        global_retriever, all_vignettes = build_global_vignette_index(
            corpus_path,
            llm_adapter,
            n_encounters=args.n_encounters,
            debug_output_dir=debug_dir,
        )
    except Exception as e:
        print(f"❌ Failed to build global index: {e}")
        import traceback
        traceback.print_exc()
        return

    indexer = SimpleBM25Indexer(retriever=global_retriever)

    # Step 1: Generate query vignette
    print(f"\n{'-'*80}")
    print(f"STEP 1: Generate Query Vignette ({args.patient_id})")
    print(f"{'-'*80}")

    try:
        xml_path = corpus_path / f"{args.patient_id}.xml"
        if not xml_path.exists():
            print(f"❌ Query patient XML not found: {xml_path}")
            return

        # Extract last N encounters
        print(f"1a. Extracting last {args.n_encounters} encounters...")
        last_encounters = extract_last_n_encounters(xml_path, args.n_encounters)
        print(f"    ✓ Extracted text length: {len(last_encounters)} chars")
        print(f"    Preview (first 300 chars):")
        print(f"    {last_encounters[:300]}...")

        # Generate vignette with LLM
        print(f"\n1b. Generating query vignette with LLM...")
        query_vignette = llm_adapter.summarize(last_encounters)
        print(f"    ✓ Vignette length: {len(query_vignette)} chars")
        print(f"    Preview (first 300 chars):")
        print(f"    {query_vignette[:300]}...")

        # Save query vignette for debugging
        debug_dir.mkdir(parents=True, exist_ok=True)
        query_vignette_path = debug_dir / f"query_vignette_{args.patient_id}.txt"
        with open(query_vignette_path, "w") as f:
            f.write(f"Query Patient ID: {args.patient_id}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{query_vignette}\n")
        print(f"    ✓ Saved query vignette to {query_vignette_path}")

    except Exception as e:
        print(f"❌ Error processing query patient: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Search global index
    print(f"\n{'-'*80}")
    print(f"STEP 2: Search Global Vignette Index")
    print(f"{'-'*80}")

    print(f"Searching for {args.top_k} most similar patients (excluding {args.patient_id})...")
    results = indexer.search(query_vignette, top_k=args.top_k, exclude_patient_id=args.patient_id)

    # Step 3: Results
    print(f"\n{'-'*80}")
    print(f"STEP 3: Results (Top-{args.top_k} Similar Patients)")
    print(f"{'-'*80}\n")

    if results:
        # Save results to file for debugging
        results_path = debug_dir / f"search_results_{args.patient_id}.txt"
        with open(results_path, "w") as f:
            f.write(f"Query Patient: {args.patient_id}\n")
            f.write(f"Number of Results: {len(results)}\n")
            f.write(f"{'='*80}\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"{i}. Patient {result['patient_id']}\n")
                f.write(f"   BM25 Score: {result['score']:.4f}\n")
                f.write(f"   Vignette:\n")
                f.write(f"   {result['vignette']}\n")
                f.write(f"\n{'-'*80}\n\n")

        print(f"✓ Saved search results to {results_path}\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. Patient {result['patient_id']}")
            print(f"   BM25 Score: {result['score']:.4f}")
            print(f"   Vignette:")
            print(f"   {result['vignette']}")
            print()
    else:
        print("No results found")

    print("="*80)
    print("WORKFLOW COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
