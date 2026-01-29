#!/usr/bin/env python3
"""
GCP Data Setup and Workflow Runner

USAGE:
    python scripts/setup_gcp_and_run.py --bucket gs://your-bucket --corpus-path data/collections

EXAMPLES:
    # Download all files and run workflow
    python scripts/setup_gcp_and_run.py --bucket gs://vista-data --corpus-path data/collections

    # Debug mode: use only first 10 XML files (for testing)
    python scripts/setup_gcp_and_run.py --bucket gs://vista-data --debug-mode 10

    # Dry run to see what would be done
    python scripts/setup_gcp_and_run.py --bucket gs://vista-data --dry-run

    # Custom settings with debug mode
    python scripts/setup_gcp_and_run.py \\
        --bucket gs://vista-data \\
        --corpus-path data/collections \\
        --patient-id 124594998 \\
        --n-encounters 3 \\
        --top-k 10 \\
        --debug-mode 20

DESCRIPTION:
    This script:
    1. Downloads files from a GCP bucket to the VM
    2. Sets up the required directory structure
    3. Runs the similarity retrieval workflow

    Debug mode (--debug-mode N) downloads only the first N XML files from the corpus.
    This is useful for testing and development on smaller datasets.

PREREQUISITES:
    - Google Cloud SDK installed (gcloud)
    - Authenticated with GCP (gcloud auth login)
    - VPN access for Stanford APIM LLM
    - VAULT_SECRET_KEY environment variable set
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"{Colors.GREEN}{text}{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}\n")


def print_step(step_num: int, text: str):
    """Print a step message."""
    print(f"{Colors.GREEN}Step {step_num}: {text}{Colors.NC}")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def run_command(cmd: list, description: str = "", dry_run: bool = False) -> bool:
    """Run a shell command and return success status."""
    if dry_run:
        print(f"{Colors.BLUE}[DRY RUN] {' '.join(cmd)}{Colors.NC}")
        return True

    if description:
        print(description)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        print_error(str(e))
        return False
    except Exception as e:
        print_error(f"Error running command: {e}")
        return False


def check_gcp_auth(dry_run: bool = False) -> bool:
    """Check if user is authenticated with GCP."""
    if dry_run:
        print(f"{Colors.BLUE}[DRY RUN] gcloud auth list{Colors.NC}")
        return True

    try:
        result = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            account = result.stdout.strip().split('\n')[0]
            print_success(f"Authenticated with GCP as: {account}")
            return True
        else:
            print_error("Not authenticated with GCP. Run: gcloud auth login")
            return False
    except Exception as e:
        print_error(f"Failed to check GCP authentication: {e}")
        return False


def check_bucket_access(bucket: str, path: str, dry_run: bool = False) -> bool:
    """Check if the GCP bucket path is accessible."""
    if dry_run:
        print(f"{Colors.BLUE}[DRY RUN] gsutil ls {bucket}/{path}{Colors.NC}")
        return True

    try:
        result = subprocess.run(
            ["gsutil", "ls", f"{bucket}/{path}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print_success(f"Bucket path is accessible: {bucket}/{path}")
            return True
        else:
            print_error(f"Cannot access bucket path: {bucket}/{path}")
            print_error(result.stderr)
            return False
    except Exception as e:
        print_error(f"Error checking bucket access: {e}")
        return False


def list_xml_files_from_bucket(bucket: str, remote_path: str, max_files: Optional[int] = None) -> list:
    """List XML files from GCP bucket, optionally limiting to first N."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", f"{bucket}/{remote_path}/*.xml"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        if max_files:
            files = files[:max_files]
        return files
    except subprocess.CalledProcessError:
        return []
    except Exception:
        return []


def download_from_bucket(
    bucket: str,
    remote_path: str,
    local_path: str,
    dry_run: bool = False,
    max_files: Optional[int] = None,
) -> bool:
    """Download files from GCP bucket.

    Args:
        bucket: GCP bucket path
        remote_path: Remote path in bucket
        local_path: Local destination path
        dry_run: If True, only show what would be done
        max_files: If set, only download first N XML files (debug mode)
    """
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        if max_files:
            print(f"{Colors.BLUE}[DRY RUN] Download first {max_files} XML files from {bucket}/{remote_path}/{Colors.NC}")
        else:
            print(f"{Colors.BLUE}[DRY RUN] gsutil -m cp -r {bucket}/{remote_path}/* {local_path}/{Colors.NC}")
        return True

    print(f"Downloading from: {bucket}/{remote_path}")
    print(f"Saving to: {local_path}")

    if max_files:
        # Debug mode: download only first N files
        print_warning(f"Debug mode: Downloading only first {max_files} XML files")
        xml_files = list_xml_files_from_bucket(bucket, remote_path, max_files)

        if not xml_files:
            print_error(f"No XML files found in {bucket}/{remote_path}")
            return False

        print(f"Found {len(xml_files)} files to download")

        try:
            for i, file_path in enumerate(xml_files, 1):
                print(f"  [{i}/{len(xml_files)}] Downloading {Path(file_path).name}...")
                cmd = ["gsutil", "cp", file_path, str(local_path) + "/"]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            print_success(f"Downloaded {len(xml_files)} files successfully")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to download file: {e}")
            return False
    else:
        # Normal mode: download all files
        try:
            cmd = ["gsutil", "-m", "cp", "-r", f"{bucket}/{remote_path}/*", str(local_path) + "/"]
            result = subprocess.run(cmd, capture_output=False, text=True, check=True)
            print_success(f"Downloaded {remote_path} successfully")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to download from bucket: {e}")
            return False
        except Exception as e:
            print_error(f"Error during download: {e}")
            return False


def verify_corpus(corpus_path: str) -> bool:
    """Verify that the corpus has XML files."""
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        print_error(f"Corpus path does not exist: {corpus_path}")
        return False

    # Look for dev-corpus subdirectory or XML files directly
    xml_files = list(corpus_path.glob("*.xml"))
    if not xml_files:
        dev_corpus_path = corpus_path / "dev-corpus"
        if dev_corpus_path.exists():
            xml_files = list(dev_corpus_path.glob("*.xml"))
            corpus_path = dev_corpus_path
        else:
            print_error(f"No XML files found in {corpus_path}")
            return False

    file_count = len(xml_files)
    print_success(f"Found {file_count} XML files in {corpus_path}")
    return True


def run_workflow(
    patient_id: str,
    corpus_dir: str,
    n_encounters: int,
    top_k: int,
    output_dir: str,
    dry_run: bool = False,
) -> bool:
    """Run the similarity retrieval workflow."""
    if dry_run:
        cmd_str = f"python scripts/similarity_retrieval_workflow_demo.py \\\n"
        cmd_str += f"  --patient-id {patient_id} \\\n"
        cmd_str += f"  --corpus-dir {corpus_dir} \\\n"
        cmd_str += f"  --n-encounters {n_encounters} \\\n"
        cmd_str += f"  --top-k {top_k} \\\n"
        cmd_str += f"  --debug-dir {output_dir}"
        print(f"{Colors.BLUE}[DRY RUN]\n{cmd_str}{Colors.NC}")
        return True

    cmd = [
        "python",
        "scripts/similarity_retrieval_workflow_demo.py",
        "--patient-id", patient_id,
        "--corpus-dir", corpus_dir,
        "--n-encounters", str(n_encounters),
        "--top-k", str(top_k),
        "--debug-dir", output_dir,
    ]

    print(f"Running workflow with:")
    print(f"  Patient ID: {patient_id}")
    print(f"  Corpus: {corpus_dir}")
    print(f"  Encounters: {n_encounters}")
    print(f"  Top-K: {top_k}")
    print(f"  Output: {output_dir}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print_success("Workflow completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Workflow failed with error code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Error running workflow: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="GCP Data Setup and Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from bucket and run workflow
  python scripts/setup_gcp_and_run.py --bucket gs://vista-data --corpus-path data/collections

  # Debug mode: use only first 10 XML files
  python scripts/setup_gcp_and_run.py --bucket gs://vista-data --debug-mode 10

  # Dry run to see what would be done
  python scripts/setup_gcp_and_run.py --bucket gs://vista-data --dry-run

  # Custom settings
  python scripts/setup_gcp_and_run.py \\
    --bucket gs://vista-data \\
    --corpus-path data/collections \\
    --patient-id 124594998 \\
    --n-encounters 3 \\
    --top-k 10 \\
    --debug-mode 20
        """,
    )

    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCP bucket path (e.g., gs://my-bucket)",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="data/collections",
        help="Corpus location in bucket (default: data/collections)",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="115969130",
        help="Patient ID to query (default: 115969130)",
    )
    parser.add_argument(
        "--n-encounters",
        type=int,
        default=2,
        help="Number of encounters to use (default: 2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar patients to return (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vignette_debug",
        help="Debug output directory (default: data/vignette_debug)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--debug-mode",
        type=int,
        default=None,
        metavar="N",
        help="Debug mode: download only first N XML files (useful for testing)",
    )

    args = parser.parse_args()

    print_header("GCP Data Setup & Workflow Runner")

    # Display configuration
    print("Configuration:")
    print(f"  Bucket: {args.bucket}")
    print(f"  Corpus path: {args.corpus_path}")
    print(f"  Patient ID: {args.patient_id}")
    print(f"  Encounters: {args.n_encounters}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Output dir: {args.output_dir}")
    if args.dry_run:
        print(f"\n  {Colors.YELLOW}DRY RUN MODE{Colors.NC}")
    if args.debug_mode:
        print(f"\n  {Colors.YELLOW}DEBUG MODE: Using only first {args.debug_mode} XML files{Colors.NC}")
    print()

    # Step 1: Check VAULT_SECRET_KEY
    print_step(1, "Checking environment variables")
    if not os.getenv("VAULT_SECRET_KEY"):
        print_warning("VAULT_SECRET_KEY not set. LLM vignette generation will fail.")
        if not args.dry_run:
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print_error("Aborted by user")
                sys.exit(1)
    else:
        print_success("VAULT_SECRET_KEY is set")
    print()

    # Step 2: Check GCP authentication
    print_step(2, "Checking GCP authentication")
    if not check_gcp_auth(args.dry_run):
        print_error("GCP authentication failed")
        sys.exit(1)
    print()

    # Step 3: Check bucket access
    print_step(3, "Checking bucket access")
    if not check_bucket_access(args.bucket, args.corpus_path, args.dry_run):
        print_error("Cannot access bucket")
        sys.exit(1)
    print()

    # Step 4: Create local directories
    print_step(4, "Creating local directories")
    local_corpus_path = Path(args.corpus_path)
    local_output_path = Path(args.output_dir)
    if not args.dry_run:
        local_corpus_path.mkdir(parents=True, exist_ok=True)
        local_output_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {local_corpus_path}")
        print_success(f"Created: {local_output_path}")
    else:
        print(f"{Colors.BLUE}[DRY RUN] mkdir -p {local_corpus_path}{Colors.NC}")
        print(f"{Colors.BLUE}[DRY RUN] mkdir -p {local_output_path}{Colors.NC}")
    print()

    # Step 5: Download from bucket
    print_step(5, "Downloading from GCP bucket")
    if not download_from_bucket(
        args.bucket,
        args.corpus_path,
        args.corpus_path,
        args.dry_run,
        max_files=args.debug_mode,
    ):
        print_error("Failed to download from bucket")
        sys.exit(1)
    print()

    # Step 6: Verify corpus
    print_step(6, "Verifying corpus")
    corpus_check_path = Path(args.corpus_path) / "dev-corpus"
    if corpus_check_path.exists():
        if not verify_corpus(str(corpus_check_path)):
            sys.exit(1)
        corpus_for_workflow = str(corpus_check_path)
    else:
        if not verify_corpus(args.corpus_path):
            sys.exit(1)
        corpus_for_workflow = args.corpus_path
    print()

    # Step 7: Run workflow
    print_step(7, "Running similarity retrieval workflow")
    if not run_workflow(
        args.patient_id,
        corpus_for_workflow,
        args.n_encounters,
        args.top_k,
        args.output_dir,
        args.dry_run,
    ):
        print_error("Workflow failed")
        sys.exit(1)
    print()

    # Summary
    print_header("Setup and Workflow Complete!")
    print("Output files:")
    print(f"  All vignettes: {args.output_dir}/all_vignettes.txt")
    print(f"  Query vignette: {args.output_dir}/query_vignette_{args.patient_id}.txt")
    print(f"  Search results: {args.output_dir}/search_results_{args.patient_id}.txt")
    print()
    print("Next steps:")
    print(f"  1. Review results in: {args.output_dir}/")
    print(f"  2. Copy results back to bucket if needed")
    print()


if __name__ == "__main__":
    main()
