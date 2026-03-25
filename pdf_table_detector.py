"""
pdf_table_detector.py - Advanced PDF table detection and extraction.

Uses multiple backends for robust table extraction from PDF files:
  1. Camelot stream mode — detects borderless/whitespace-aligned tables
  2. pdfplumber — detects grid/bordered tables
  3. PyMuPDF — fallback for bordered tables

Designed for scientific papers where tables often lack clear grid lines.
At scale (millions of papers), this module provides the primary PDF table
extraction without needing expensive LLM vision calls.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Check available backends
# ──────────────────────────────────────────────────────────────────────────────

_HAS_CAMELOT = False
try:
    import camelot
    _HAS_CAMELOT = True
except ImportError:
    pass


@dataclass
class ExtractedTable:
    """A single table extracted from a PDF page."""
    df: pd.DataFrame
    page_number: int          # 1-based page number
    accuracy: float = 0.0     # confidence score (0-100)
    backend: str = ""         # which backend found it
    table_index: int = 0      # index within the page

    @property
    def is_data_table(self) -> bool:
        """Heuristic: is this a geochemical data table (not a caption/header)?"""
        if self.df.shape[0] < 3 or self.df.shape[1] < 3:
            return False
        # Check if any cell contains numeric data
        numeric_count = 0
        for col in self.df.columns:
            for val in self.df[col].head(10):
                try:
                    float(str(val).replace(",", ""))
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass
        return numeric_count >= 5

    def to_text(self, max_rows: int = 200) -> str:
        """Convert to tab-separated text for LLM consumption."""
        df = self.df.head(max_rows)
        rows = []
        for _, row in df.iterrows():
            cells = [str(c) if pd.notna(c) and str(c).strip() else "" for c in row]
            rows.append("\t".join(cells))
        return "\n".join(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ──────────────────────────────────────────────────────────────────────────────

def extract_tables_from_pdf(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    min_rows: int = 3,
    min_cols: int = 3,
) -> list[ExtractedTable]:
    """Extract tables from a PDF using the best available backend.

    Tries Camelot stream mode first (best for borderless scientific tables),
    falling back to pdfplumber.

    Args:
        pdf_path: Path to the PDF file.
        pages: 1-based page numbers to extract from. None = all pages.
        min_rows: Minimum rows for a table to be included.
        min_cols: Minimum columns for a table to be included.

    Returns:
        List of ExtractedTable objects, sorted by page number.
    """
    pdf_path = Path(pdf_path)
    tables: list[ExtractedTable] = []

    if _HAS_CAMELOT:
        tables = _extract_with_camelot(pdf_path, pages, min_rows, min_cols)

    if not tables:
        logger.info("Camelot found no tables, falling back to pdfplumber")
        tables = _extract_with_pdfplumber(pdf_path, pages, min_rows, min_cols)

    # Filter to likely data tables
    data_tables = [t for t in tables if t.is_data_table]
    if data_tables:
        logger.info(
            "Extracted %d data tables from %s (backend: %s)",
            len(data_tables), pdf_path.name,
            data_tables[0].backend if data_tables else "none",
        )
    return data_tables


def extract_tables_as_text(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    max_tables: int = 10,
) -> list[str]:
    """Extract tables from PDF and return as text strings.

    Convenience wrapper that returns tab-separated text suitable for
    LLM prompts. Compatible with the existing tables_text interface.
    """
    tables = extract_tables_from_pdf(pdf_path, pages)
    result = []
    for t in tables[:max_tables]:
        header = f"[Page {t.page_number}, {t.backend}, {t.df.shape[0]}r x {t.df.shape[1]}c]"
        result.append(f"{header}\n{t.to_text()}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Backend: Camelot (stream mode for borderless tables)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_camelot(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
) -> list[ExtractedTable]:
    """Extract tables using Camelot's stream mode."""
    if not _HAS_CAMELOT:
        return []

    # Build page string: "1,2,3" or "all"
    if pages:
        page_str = ",".join(str(p) for p in pages)
    else:
        page_str = "all"

    tables: list[ExtractedTable] = []
    try:
        camelot_tables = camelot.read_pdf(
            str(pdf_path),
            flavor="stream",
            pages=page_str,
            edge_tol=50,      # tolerance for edge detection
            row_tol=10,        # tolerance for row grouping
        )
        for i, ct in enumerate(camelot_tables):
            df = ct.df
            if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                # Clean up empty rows/columns
                df = _clean_table_df(df)
                if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                    tables.append(ExtractedTable(
                        df=df,
                        page_number=ct.page,
                        accuracy=ct.accuracy,
                        backend="camelot-stream",
                        table_index=i,
                    ))
    except Exception as exc:
        logger.warning("Camelot extraction failed: %s", exc)

    return tables


# ──────────────────────────────────────────────────────────────────────────────
# Backend: pdfplumber (lattice/grid tables)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_pdfplumber(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
) -> list[ExtractedTable]:
    """Extract tables using pdfplumber's built-in table detection."""
    try:
        import pdfplumber
    except ImportError:
        return []

    tables: list[ExtractedTable] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            if pages and (page_idx + 1) not in pages:
                continue
            for tbl_idx, tbl in enumerate(page.extract_tables()):
                if not tbl or len(tbl) < min_rows:
                    continue
                # Convert to DataFrame
                df = pd.DataFrame(tbl[1:], columns=tbl[0] if tbl[0] else None)
                if df.shape[1] < min_cols:
                    continue
                df = _clean_table_df(df)
                if df.shape[0] >= min_rows:
                    tables.append(ExtractedTable(
                        df=df,
                        page_number=page_idx + 1,
                        accuracy=100.0,  # pdfplumber doesn't report accuracy
                        backend="pdfplumber",
                        table_index=tbl_idx,
                    ))
    return tables


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty rows and columns from an extracted table."""
    # Drop fully empty rows
    df = df.dropna(how="all")
    # Drop rows where all values are empty strings
    df = df[~df.apply(lambda row: all(str(v).strip() == "" for v in row), axis=1)]
    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.apply(lambda col: all(str(v).strip() == "" for v in col))]
    return df.reset_index(drop=True)
