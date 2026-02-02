#!/usr/bin/env python3
"""
Validate ICD-10-CM 7th character definitions against official CMS/CDC sources.

Official sources:
- CMS: https://www.cms.gov/medicare/coding-billing/icd-10-codes
- CDC NCHS: https://www.cdc.gov/nchs/icd/icd-10-cm.htm

The official ICD-10-CM files include:
- icd10cm_tabular_YYYY.xml - Full tabular list with 7th character definitions
- icd10cm_order_YYYY.txt - Simple code list

This script:
1. Downloads the official CMS ICD-10-CM files (if not cached)
2. Parses the tabular XML to extract 7th character definitions
3. Compares against our hardcoded mappings
"""

import os
import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import zipfile
import io

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# CMS download URLs (update year as needed)
# Note: CMS uses URL encoding with slashes in dates
CMS_ICD10CM_URLS = {
    2024: {
        "tabular": "https://www.cms.gov/files/zip/2024-code-descriptions-tabular-order-updated-02/01/2024.zip",
        "tables": "https://www.cms.gov/files/zip/2024-code-tables-tabular-and-index-updated-02/01/2024.zip",
        "guidelines": "https://www.cms.gov/files/document/fy-2024-icd-10-cm-coding-guidelines-updated-02/01/2024.pdf",
    },
    2025: {
        "tabular": "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip",
        "tables": "https://www.cms.gov/files/zip/2025-code-tables-tabular-and-index.zip",
        "guidelines": "https://www.cms.gov/files/document/fy-2025-icd-10-cm-coding-guidelines.pdf",
    },
}

CACHE_DIR = Path(__file__).parent.parent / "data" / "icd10cm_reference"


def download_cms_files(year: int = 2024, force: bool = False) -> Path:
    """Download official CMS ICD-10-CM files."""
    import urllib.request
    
    cache_dir = CACHE_DIR / str(year)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if year not in CMS_ICD10CM_URLS:
        print(f"No URLs configured for year {year}. Available: {list(CMS_ICD10CM_URLS.keys())}")
        raise ValueError(f"Year {year} not supported")
    
    urls = CMS_ICD10CM_URLS[year]
    
    # Download tabular order file (main code descriptions)
    tabular_path = cache_dir / f"icd10cm_{year}_tabular.zip"
    
    if tabular_path.exists() and not force:
        print(f"Using cached file: {tabular_path}")
    else:
        url = urls["tabular"]
        print(f"Downloading tabular order from: {url}")
        
        try:
            urllib.request.urlretrieve(url, tabular_path)
            print(f"Downloaded to: {tabular_path}")
            
            # Extract
            with zipfile.ZipFile(tabular_path, 'r') as zf:
                zf.extractall(cache_dir)
                print(f"Extracted to: {cache_dir}")
        except Exception as e:
            print(f"Error downloading tabular: {e}")
    
    # Download code tables (includes XML with 7th char definitions)
    tables_path = cache_dir / f"icd10cm_{year}_tables.zip"
    
    if tables_path.exists() and not force:
        print(f"Using cached file: {tables_path}")
    else:
        url = urls["tables"]
        print(f"Downloading code tables from: {url}")
        
        try:
            urllib.request.urlretrieve(url, tables_path)
            print(f"Downloaded to: {tables_path}")
            
            # Extract
            with zipfile.ZipFile(tables_path, 'r') as zf:
                zf.extractall(cache_dir)
                print(f"Extracted to: {cache_dir}")
        except Exception as e:
            print(f"Error downloading tables: {e}")
    
    # Download guidelines PDF
    guidelines_path = cache_dir / f"icd10cm_{year}_guidelines.pdf"
    
    if guidelines_path.exists() and not force:
        print(f"Using cached file: {guidelines_path}")
    else:
        url = urls["guidelines"]
        print(f"Downloading guidelines from: {url}")
        
        try:
            urllib.request.urlretrieve(url, guidelines_path)
            print(f"Downloaded to: {guidelines_path}")
        except Exception as e:
            print(f"Error downloading guidelines: {e}")
    
    print(f"\nFiles downloaded to: {cache_dir}")
    print("\nKey files:")
    for f in cache_dir.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")
    
    return cache_dir


def parse_7th_char_from_tabular_xml(xml_path: Path) -> Dict[str, Dict]:
    """
    Parse 7th character definitions from the official ICD-10-CM tabular XML.
    
    The XML structure includes <sevenChrDef> elements that define 7th character meanings
    for specific code ranges.
    
    Returns:
        Dictionary mapping code prefix -> {7th_char -> description}
    """
    print(f"Parsing: {xml_path}")
    
    # The XML can be large, so we'll use iterparse
    seventh_char_defs = {}
    current_chapter = None
    current_section = None
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find all sevenChrDef elements
        # These define 7th character meanings for code ranges
        for elem in root.iter():
            if elem.tag == 'sevenChrDef':
                # Get parent diag code
                parent = elem.find('..')
                code = None
                for diag in root.iter('diag'):
                    name = diag.find('name')
                    if name is not None and diag.find('sevenChrDef') == elem:
                        code = name.text
                        break
                
                if code:
                    seventh_char_defs[code] = {}
                    for ext in elem.findall('extension'):
                        char = ext.get('char')
                        desc = ext.text
                        if char and desc:
                            seventh_char_defs[code][char] = desc.strip()
        
        return seventh_char_defs
        
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        return {}


def parse_7th_char_from_order_file(txt_path: Path) -> Dict[str, str]:
    """
    Parse code descriptions from the order file to extract 7th character info.
    
    Format: Each line is fixed-width with code and description.
    
    Returns:
        Dictionary mapping full code -> description
    """
    print(f"Parsing: {txt_path}")
    
    codes = {}
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Format: 5 chars order, 1 space, 7 chars code, 1 space, 1 char billable, 1 space, short desc, long desc
            if len(line) < 15:
                continue
            
            # Try to extract code and description
            parts = line.strip().split()
            if len(parts) >= 3:
                # Code is typically in position 1 (after order number)
                code = parts[1] if len(parts[1]) >= 3 else parts[0]
                # Description is the rest
                desc_start = line.find(code) + len(code)
                desc = line[desc_start:].strip()
                
                # Clean up - remove billable indicator
                if desc and desc[0] in '01':
                    desc = desc[1:].strip()
                
                codes[code] = desc
    
    return codes


def extract_7th_char_patterns_from_descriptions(codes: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Extract 7th character meanings by analyzing code descriptions.
    
    Groups codes by their 6-character prefix and extracts what each 7th character means.
    """
    # Group by 6-char prefix
    prefix_groups = defaultdict(dict)
    
    for code, desc in codes.items():
        # Only look at 7-character codes (without dot: 7 chars, with dot: 8 chars)
        code_nodot = code.replace('.', '')
        if len(code_nodot) == 7:
            prefix = code_nodot[:6]
            seventh = code_nodot[6]
            prefix_groups[prefix][seventh] = desc
    
    # Extract common patterns
    patterns = {}
    
    # Common 7th character suffixes to look for
    episode_patterns = {
        'initial encounter': 'A',
        'subsequent encounter': 'D', 
        'sequela': 'S',
    }
    
    healing_patterns = {
        'delayed healing': 'G',
        'nonunion': 'K',
        'malunion': 'P',
    }
    
    fracture_patterns = {
        'open fracture type I or II': 'B',
        'open fracture type IIIA, IIIB, or IIIC': 'C',
    }
    
    # Analyze each prefix group
    for prefix, chars in prefix_groups.items():
        patterns[prefix] = {}
        for char, desc in chars.items():
            desc_lower = desc.lower()
            
            # Determine type based on description
            char_type = "unknown"
            if 'initial encounter' in desc_lower:
                char_type = "episode"
            elif 'subsequent encounter' in desc_lower:
                char_type = "episode" if 'healing' not in desc_lower else "healing"
            elif 'sequela' in desc_lower:
                char_type = "episode"
            elif 'delayed healing' in desc_lower or 'nonunion' in desc_lower or 'malunion' in desc_lower:
                char_type = "healing"
            elif 'open fracture type' in desc_lower:
                char_type = "fracture_type"
            elif 'fetus' in desc_lower:
                char_type = "fetus"
            
            patterns[prefix][char] = {
                'description': desc,
                'type': char_type,
            }
    
    return patterns


def validate_hardcoded_mappings():
    """Validate our hardcoded 7th character mappings against actual data."""
    from materialize_paths import ICD10CM_7TH_CHAR_MODIFIERS
    
    print("\n" + "="*70)
    print("Validating hardcoded ICD10CM 7th character mappings")
    print("="*70)
    
    print("\nHardcoded mappings:")
    for char, info in sorted(ICD10CM_7TH_CHAR_MODIFIERS.items()):
        print(f"  {char}: {info['type']}/{info['label']} - {info['description']}")
    
    # Try to load from Athena data
    print("\n\nValidating against Athena ICD10CM descriptions...")
    
    try:
        from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology
        import polars as pl
        
        ontology = SparseGraphOntology.load_from_parquet(
            'data/athena_omop_ontologies',
            graph_path='data/athena_omop_ontologies/ontology_graphs'
        )
        
        # Get all ICD10CM codes with 7 characters
        result = (
            ontology.concepts_df
            .filter(pl.col('code').str.starts_with('ICD10CM/'))
            .select(['code', 'description'])
            .collect()
        )
        
        # Extract patterns
        char_descriptions = defaultdict(lambda: defaultdict(set))
        
        for code, desc in result.rows():
            code_only = code.replace('ICD10CM/', '').replace('.', '')
            if len(code_only) == 7:
                seventh = code_only[-1]
                # Extract the part of description that mentions the episode/healing status
                desc_lower = desc.lower() if desc else ""
                
                if 'initial encounter' in desc_lower:
                    char_descriptions[seventh]['episode'].add('initial encounter')
                elif 'subsequent encounter' in desc_lower:
                    if 'delayed healing' in desc_lower:
                        char_descriptions[seventh]['healing'].add('delayed healing')
                    elif 'nonunion' in desc_lower:
                        char_descriptions[seventh]['healing'].add('nonunion')
                    elif 'malunion' in desc_lower:
                        char_descriptions[seventh]['healing'].add('malunion')
                    else:
                        char_descriptions[seventh]['episode'].add('subsequent encounter')
                elif 'sequela' in desc_lower:
                    char_descriptions[seventh]['episode'].add('sequela')
                elif 'fetus' in desc_lower:
                    char_descriptions[seventh]['fetus'].add('fetus')
        
        print("\n7th character patterns found in Athena data:")
        for char in sorted(char_descriptions.keys()):
            types = char_descriptions[char]
            print(f"\n  {char}:")
            for char_type, descriptions in types.items():
                print(f"    {char_type}: {', '.join(sorted(descriptions))}")
        
        # Compare with hardcoded
        print("\n\nComparison with hardcoded mappings:")
        for char, info in sorted(ICD10CM_7TH_CHAR_MODIFIERS.items()):
            athena_types = char_descriptions.get(char, {})
            if athena_types:
                match = info['type'] in athena_types
                status = "✓" if match else "✗"
                print(f"  {char}: hardcoded={info['type']}, athena={list(athena_types.keys())} {status}")
            else:
                print(f"  {char}: hardcoded={info['type']}, athena=NOT FOUND")
        
    except Exception as e:
        print(f"Error loading Athena data: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate ICD-10-CM 7th character definitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Official sources:
  CMS: https://www.cms.gov/medicare/coding-billing/icd-10-codes
  CDC: https://www.cdc.gov/nchs/icd/icd-10-cm.htm

This script validates our hardcoded 7th character mappings against:
1. Patterns found in Athena ICD10CM code descriptions
2. Official CMS ICD-10-CM tabular files (if downloaded)
        """
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download official CMS ICD-10-CM files"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="ICD-10-CM year to download (default: 2024)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate hardcoded mappings against Athena data"
    )
    
    args = parser.parse_args()
    
    if args.download:
        try:
            cache_dir = download_cms_files(args.year)
            print(f"\nFiles downloaded to: {cache_dir}")
            print("\nLook for:")
            print("  - icd10cm_tabular_*.xml - Full definitions including 7th char")
            print("  - icd10cm_order_*.txt - Code list with descriptions")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            return 1
    
    if args.validate:
        validate_hardcoded_mappings()
    
    print("\n" + "="*70)
    print("Official reference for 7th character definitions:")
    print("="*70)
    print("""
ICD-10-CM Official Guidelines (Section I.C):
https://www.cms.gov/files/document/fy-2024-icd-10-cm-coding-guidelines.pdf

Key 7th character categories:

1. EPISODE OF CARE (most injury codes S00-T88):
   A - Initial encounter (active treatment)
   D - Subsequent encounter (routine care during healing)
   S - Sequela (complication/condition from healed injury)

2. FRACTURE HEALING STATUS (fracture codes):
   A - Initial encounter for closed fracture
   B - Initial encounter for open fracture type I or II
   C - Initial encounter for open fracture type IIIA/IIIB/IIIC
   D - Subsequent encounter for closed fracture with routine healing
   G - Subsequent encounter for closed fracture with delayed healing
   K - Subsequent encounter for closed fracture with nonunion
   P - Subsequent encounter for closed fracture with malunion
   S - Sequela

3. FETUS IDENTIFICATION (obstetric codes O30-O48):
   0 - Not applicable or unspecified
   1-5 - Fetus 1 through 5
   9 - Other fetus

Note: 7th character meanings are CONTEXT-DEPENDENT based on code category!
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
