#!/usr/bin/env python3
"""
Test multi-source (supplementary + PDF) extraction end-to-end.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

from pipeline import ExtractionPipeline, TableDetectorBackend
from llm_clients import ClaudeClient

def test_multisource_extraction():
    """Test PDF with supplementary files to ensure both sources are used."""
    
    # Use Yuan_et_al_2018 paper (has both PDF and supplementary data)
    pdf_path = Path("data/papers/2018_Yuan_etal.pdf")
    supp_file = Path("data/papers/2018_Yuan_etal.xlsx")
    
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        # Try to locate it
        candidate = list(Path("data").rglob("2018_Yuan_etal.pdf"))
        if candidate:
            pdf_path = candidate[0]
            print(f"Found PDF at: {pdf_path}")
        else:
            print("Could not locate 2018_Yuan_etal.pdf")
            return
    
    if not supp_file.exists():
        candidate = list(Path("data").rglob("2018_Yuan_etal.xlsx"))
        if candidate:
            supp_file = candidate[0]
            print(f"Found supplementary at: {supp_file}")
        else:
            print("Could not locate supplementary file")
            supp_file = None
    
    print(f"\n{'='*70}")
    print(f"Testing multi-source extraction")
    print(f"PDF: {pdf_path}")
    print(f"Supplementary: {supp_file if (supp_file and supp_file.exists()) else 'Not found'}")
    print(f"{'='*70}\n")
    
    try:
        # Initialize LLM client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env")
            return
        
        client = ClaudeClient(api_key=api_key, model="claude-3-5-sonnet-20241022")
        
        # Create pipeline with multi-backend strategy
        pipeline = ExtractionPipeline(
            llm_client=client,
            use_tool_calling=True,
            use_llm_table_filter=True,
            use_self_correction=False,
            use_vision=False,  # Skip vision to save time
            verbose=True,
            table_detector_backend=TableDetectorBackend.AUTO,
        )
        
        # Use supplementary file if found
        supplementary_paths = None
        if supp_file and supp_file.exists():
            supplementary_paths = [supp_file]
            print(f"Using supplementary file: {supp_file}\n")
        
        # Run extraction
        result = pipeline.run(
            pdf_path=str(pdf_path),
            supplementary_paths=supplementary_paths,
        )
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Metadata: {result.metadata.sample_source}")
        print(f"Total samples: {len(result.samples)}")
        if result.samples:
            print(f"  - First sample: {result.samples[0].sample_name}")
            print(f"  - Last sample: {result.samples[-1].sample_name}")
        print(f"\nNotes:")
        for note in result.notes:
            print(f"  - {note}")
        
        if result.errors:
            print(f"\nErrors/Warnings ({len(result.errors)}):")
            for err in result.errors[:3]:  # Show first 3
                print(f"  - {err}")
        
        return len(result.samples)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_multisource_extraction()
