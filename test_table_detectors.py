#!/usr/bin/env python
"""
Test script to compare table detection across all three backends.

Runs Docling, Camelot, and pdfplumber on a sample PDF and shows results.
"""

import sys
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Add parent to path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from geochem_benchmark.tabledetector import (
    extract_tables_from_pdf,
    TableDetectorBackend,
    get_available_backends,
    get_backend_info,
)


def test_pdf_table_extraction(pdf_path: str | Path) -> None:
    """Test table extraction with all available backends on a single PDF."""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing PDF: {pdf_path.name}")
    print(f"{'='*80}\n")
    
    # Show available backends
    available = get_available_backends()
    backend_info = get_backend_info()
    print(f"Available backends: {', '.join(available)}")
    print()
    
    results = {}
    
    # Test each backend
    for backend in [TableDetectorBackend.DOCLING, TableDetectorBackend.CAMELOT, TableDetectorBackend.PDFPLUMBER]:
        backend_name = backend.value
        
        # Skip if not available
        if not backend_info[backend_name]["available"]:
            print(f"  {backend_name.upper():15} - NOT AVAILABLE")
            continue
        
        print(f"  Testing {backend_name.upper()}...")
        try:
            tables, metrics = extract_tables_from_pdf(
                pdf_path,
                backend=backend,
                force_backend=True,  # Force only this backend
            )
            
            results[backend_name] = {
                "status": "success",
                "tables_found": len(tables),
                "data_tables": metrics.data_tables_found,
                "pages_scanned": metrics.pages_scanned,
                "errors": metrics.errors,
                "table_shapes": [(t.df.shape[0], t.df.shape[1]) for t in tables],
            }
            
            print(f"  [OK] {backend_name.upper():15} - Found {len(tables)} total, {metrics.data_tables_found} data tables")
            if tables:
                print(f"    Table shapes: {results[backend_name]['table_shapes']}")
            if metrics.errors:
                for err in metrics.errors:
                    print(f"    Warning: {err}")
        
        except Exception as exc:
            results[backend_name] = {
                "status": "failed",
                "error": str(exc),
            }
            print(f"  [FAIL] {backend_name.upper():15} - FAILED: {exc}")
        
        print()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(json.dumps(results, indent=2))
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")
    
    successful = {k: v for k, v in results.items() if v["status"] == "success"}
    if successful:
        best_count = max(v["data_tables"] for v in successful.values())
        best_backend = [k for k, v in successful.items() if v["data_tables"] == best_count]
        print(f"Best backend (by data table count): {', '.join(best_backend)} ({best_count} tables)")
    else:
        print("No backends succeeded.")


if __name__ == "__main__":
    # Test with first available PDF or provided path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Find first PDF in data directory
        base_dir = Path(__file__).parent
        data_dir = base_dir / "data"
        pdfs = list(data_dir.glob("*.pdf"))
        if not pdfs:
            print("ERROR: No PDFs found in data/ directory")
            sys.exit(1)
        pdf_path = pdfs[0]
    
    test_pdf_table_extraction(pdf_path)
