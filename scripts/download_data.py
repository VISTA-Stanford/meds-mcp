#!/usr/bin/env python3
"""
Download EHR dataset files and tables from Redivis.

This script supports downloading multiple healthcare datasets including MedAlign,
INSPECT, EHRSHOT, and FactEHR from Redivis with proper authentication.

Usage Examples:
    # List all available datasets
    python scripts/download_data.py --list

    # Download MedAlign files using environment variable for token
    export REDIVIS_ACCESS_TOKEN="your_token_here"
    python scripts/download_data.py medalign --files

    # Download MedAlign files with command line token
    python scripts/download_data.py medalign --files -t your_token_here

    # Download both files and tables for MedAlign
    python scripts/download_data.py medalign --files --tables

    # Download only tables for a dataset
    python scripts/download_data.py medalign --tables

    # Using .env file (recommended)
    echo "REDIVIS_ACCESS_TOKEN=your_token_here" > .env
    python scripts/download_data.py medalign --files

    # After installing with pip install -e .
    download-data medalign --files

Requirements:
    - Redivis access token (set REDIVIS_ACCESS_TOKEN or use -t flag)
    - Required packages: requests, python-dotenv, tqdm (install with: pip install -e .)

Features:
    - Skips download if file already exists
    - Automatically extracts ZIP and TAR.GZ archives after download
    - Checks if archives are already extracted before re-extracting

Output:
    Files are saved to data/meds/{dataset_name}/ directory
    Tables are saved to data/meds/{dataset_name}/tables/ directory
    Archives are automatically extracted to the same directory
"""

import argparse
import os
import sys
import zipfile
import tarfile
from pathlib import Path
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional
from tqdm import tqdm


# Dataset configuration
DATASETS = {
    "medalign": {
        "name": "MedAlign",
        "files": [
            {
                "url": "https://redivis.com/api/v1/rawFiles/6pza-b944ea90v.VgNnlViDr3MA0chpeP6ouQ",
                "filename": "meds_reader_omop_medalign.tar.gz",
                "description": "MedAlign MEDS Reader Extract",
            },
            {
                "url": "https://redivis.com/api/v1/rawFiles/6dt4-fcdrwc636.ODGKZrSdZVdYoTQsHHumxQ",
                "filename": "medalign_instructions_responses_v1_3.zip",
                "description": "MedAlign Instructions + Responses",
            },
        ],
        "tables": [
            # TODO: Add table URLs when available
            # {
            #     "url": "https://redivis.com/api/v1/tables/...",
            #     "filename": "table_name.csv",
            #     "description": "Table description"
            # }
        ],
    },
    # meds_omop_inspect.tar.gz   pmw7-c62y8hnsk.xjtG8l50Q5cGmN_07YkPdQ
    # meds_reader_omop_inspect.tar.gz pmw7-c62y8hnsk.PZeP8tvZMZYdNg4OyvtYnQ
    "inspect": {
        "name": "INSPECT",
        "files": [
            # {
            #    "url": "https://redivis.com/api/v1/rawFiles/pmw7-c62y8hnsk.xjtG8l50Q5cGmN_07YkPdQ",
            #    "filename": "meds_omop_inspect.tar.gz",
            #     "description": "INSPECT MEDS Extract",
            # },
            {
                "url": "https://redivis.com/api/v1/rawFiles/pmw7-c62y8hnsk.PZeP8tvZMZYdNg4OyvtYnQ",
                "filename": "meds_reader_omop_inspect.tar.gz",
                "description": "INSPECT MEDS Reader Extract",
            },
        ],
        "tables": [
            # TODO: Add INSPECT table URLs
        ],
    },
    "ehrshot": {
        "name": "EHRSHOT",
        "files": [
            {
                "url": "https://redivis.com/api/v1/rawFiles/4avd-f4efj6ehd.H-YFM4q--KhxCkq1OwJntA",
                "filename": "meds_reader_omop_ehrshot.tar.gz",
                "description": "EHRSHOT MEDS Reader Extract",
            }
        ],
        "tables": [
            # TODO: Add EHRSHOT table URLs
        ],
    },
    "factehr": {
        "name": "FactEHR",
        "files": [
            # Stanford-only FactEHR dataset
            {
                "url": "https://redivis.com/api/v1/rawFiles/6x2r-90gtsan9p.MSGjyaPBRWeYss0Mo_kVLA",
                "filename": "factehr_stanford.zip",
                "description": "FactEHR-Stanford dataset",
            }
        ],
        "tables": [
            # TODO: Add FactEHR table URLs
        ],
    },
}


def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    for key, dataset in DATASETS.items():
        files_count = len(dataset["files"])
        tables_count = len(dataset["tables"])
        print(
            f"  {key}: {dataset['name']} ({files_count} files, {tables_count} tables)"
        )


def is_archive_decompressed(archive_path: Path) -> bool:
    """Check if an archive file has been decompressed in the same directory."""
    if not archive_path.exists():
        return False
    
    parent_dir = archive_path.parent
    archive_name = archive_path.stem
    
    # For tar.gz files, remove both .tar and .gz extensions
    if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
        archive_name = archive_path.stem[:-4]  # Remove .tar from stem
    
    # Look for extracted content in the same directory
    # Check for common patterns of extracted directories or files
    possible_extracted_names = [
        archive_name,
        archive_name.replace('_', '-'),
        archive_name.replace('-', '_'),
    ]
    
    for name in possible_extracted_names:
        potential_dir = parent_dir / name
        if potential_dir.exists() and potential_dir.is_dir():
            # Check if directory has content
            if any(potential_dir.iterdir()):
                return True
    
    # Also check if there are many files in the parent directory (suggesting extraction)
    if archive_path.suffix.lower() in ['.zip', '.gz'] and archive_path.exists():
        # Count non-archive files in the directory
        non_archive_files = [
            f for f in parent_dir.iterdir() 
            if f.is_file() and f.suffix.lower() not in ['.zip', '.gz', '.tar']
        ]
        # If there are multiple non-archive files, likely extracted
        if len(non_archive_files) > 3:
            return True
    
    return False


def decompress_archive(archive_path: Path) -> bool:
    """Decompress an archive file to the same directory."""
    if not archive_path.exists():
        print(f"Archive file not found: {archive_path}")
        return False
    
    parent_dir = archive_path.parent
    
    try:
        if archive_path.suffix.lower() == '.zip':
            print(f"Extracting ZIP archive: {archive_path.name}")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(parent_dir)
                
        elif archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
            print(f"Extracting TAR.GZ archive: {archive_path.name}")
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(parent_dir)
                
        elif archive_path.suffix.lower() == '.tar':
            print(f"Extracting TAR archive: {archive_path.name}")
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(parent_dir)
                
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
            
        print(f"  ✓ Extracted successfully to: {parent_dir}")
        return True
        
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}", file=sys.stderr)
        return False


def is_archive_file(filename: str) -> bool:
    """Check if a filename represents an archive that should be decompressed."""
    archive_extensions = ['.zip', '.tar.gz', '.tgz', '.tar']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in archive_extensions)


def download_file(url: str, output_path: Path, description: str, token: str) -> bool:
    """Download a single file from Redivis and decompress if needed."""
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    file_downloaded = False
    if output_path.exists():
        print(f"File already exists: {description}")
        print(f"  -> {output_path}")
        print(f"  ✓ Skipping download")
        file_downloaded = True
    else:
        # Download the file
        headers = {"Authorization": f"Bearer {token}"}

        print(f"Downloading: {description}")
        print(f"  -> {output_path}")

        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            # Get total file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=output_path.name,
                    leave=False,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"  ✓ Downloaded successfully")
            file_downloaded = True

        except requests.RequestException as e:
            print(f"  ✗ Download failed: {e}", file=sys.stderr)
            return False
    
    # Check if file is an archive and needs decompression
    if file_downloaded and is_archive_file(output_path.name):
        if is_archive_decompressed(output_path):
            print(f"  ✓ Archive already extracted")
        else:
            print(f"  → Extracting archive...")
            if not decompress_archive(output_path):
                print(f"  ⚠ File downloaded but extraction failed", file=sys.stderr)
                return False
    
    return True


def download_dataset_files(dataset_key: str, token: str) -> bool:
    """Download all files for a dataset."""
    if dataset_key not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_key}'", file=sys.stderr)
        return False

    dataset = DATASETS[dataset_key]

    if not dataset["files"]:
        print(f"No files available for dataset '{dataset['name']}'")
        return True

    print(f"Downloading files for dataset: {dataset['name']}")

    # Create dataset-specific directory
    data_dir = Path("data/meds") / dataset_key

    success = True
    for file_info in dataset["files"]:
        output_path = data_dir / file_info["filename"]
        if not download_file(
            file_info["url"], output_path, file_info["description"], token
        ):
            success = False

    return success


def download_dataset_tables(dataset_key: str, token: str) -> bool:
    """Download all tables for a dataset."""
    if dataset_key not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_key}'", file=sys.stderr)
        return False

    dataset = DATASETS[dataset_key]

    if not dataset["tables"]:
        print(
            f"No tables available for dataset '{dataset['name']}' (feature not implemented yet)"
        )
        return True

    print(f"Downloading tables for dataset: {dataset['name']}")

    # Create dataset-specific directory
    data_dir = Path("data/meds") / dataset_key / "tables"

    success = True
    for table_info in dataset["tables"]:
        output_path = data_dir / table_info["filename"]
        if not download_file(
            table_info["url"], output_path, table_info["description"], token
        ):
            success = False

    return success


def main():
    # Load .env file if it exists
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download EHR-FM dataset files and tables from Redivis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           # List available datasets
  %(prog)s medalign --files                 # Download MedAlign files
  %(prog)s medalign --tables                # Download MedAlign tables
  %(prog)s medalign --files --tables        # Download both files and tables
  %(prog)s inspect --files -t TOKEN         # Download INSPECT files with token
        """,
    )

    # Token argument
    parser.add_argument(
        "-t",
        "--token",
        help="Redivis access token (alternative to REDIVIS_ACCESS_TOKEN env var)",
    )

    # List datasets
    parser.add_argument(
        "--list", action="store_true", help="List available datasets and exit"
    )

    # Dataset selection
    parser.add_argument(
        "dataset", nargs="?", choices=list(DATASETS.keys()), help="Dataset to download"
    )

    # Download type selection
    parser.add_argument("--files", action="store_true", help="Download dataset files")

    parser.add_argument("--tables", action="store_true", help="Download dataset tables")

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_datasets()
        return

    # Validate dataset argument
    if not args.dataset:
        print(
            "Error: Please specify a dataset or use --list to see available datasets",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)

    # Default to files if no download type specified
    if not args.files and not args.tables:
        args.files = True

    # Get access token from command line or environment
    token = args.token or os.getenv("REDIVIS_ACCESS_TOKEN")
    if not token:
        print("Error: No access token provided.", file=sys.stderr)
        print(
            "Set REDIVIS_ACCESS_TOKEN in .env file or use -t option.", file=sys.stderr
        )
        sys.exit(1)

    # Download requested data
    success = True

    if args.files:
        if not download_dataset_files(args.dataset, token):
            success = False

    if args.tables:
        if not download_dataset_tables(args.dataset, token):
            success = False

    if success:
        print(f"\nDownload completed successfully!")
        print(f"Files saved to: data/meds/{args.dataset}/")
    else:
        print(f"\nSome downloads failed. Check the errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
