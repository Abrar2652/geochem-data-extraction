"""
tabledetector.py - Advanced multi-backend PDF table detection and extraction.

Supports three backends with intelligent selection and fallback:
  1. Docling (primary) — state-of-the-art document understanding
  2. Camelot (fallback 1) — stream mode for borderless/whitespace-aligned tables
  3. pdfplumber (fallback 2) — grid/bordered tables

Features:
  - Automatic backend selection based on PDF characteristics
  - Configurable fallback chain: Docling → Camelot → pdfplumber
  - Per-backend performance tracking
  - Supports forcing specific backend via configuration
  - Handles both scientific papers and data-dense tables

Strategy:
  Try Docling first (best for complex layouts). If it finds tables or fails gracefully,
  use those results. Otherwise, fall back to Camelot stream (good for borderless tables).
  If Camelot finds nothing, use pdfplumber (reliable for grid-based tables).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Sequence
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class TableDetectorBackend(str, Enum):
    """Available table detection backends."""
    DOCLING = "docling"
    CAMELOT = "camelot"
    PDFPLUMBER = "pdfplumber"
    AUTO = "auto"  # Automatic selection


# ──────────────────────────────────────────────────────────────────────────────
# Check available backends
# ──────────────────────────────────────────────────────────────────────────────

_HAS_DOCLING = False
try:
    from docling.document_converter import DocumentConverter, FormatOption
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend as _DoclingPdfBackend
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    _HAS_DOCLING = True
except ImportError:
    pass

_HAS_CAMELOT = False
try:
    import camelot
    _HAS_CAMELOT = True
except ImportError:
    pass


@dataclass
class TableDetectionMetrics:
    """Metrics about table detection across backends."""
    backend_used: str
    tables_found: int
    data_tables_found: int
    pages_scanned: int
    time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class ExtractedTable:
    """A single table extracted from a PDF page."""
    df: pd.DataFrame
    page_number: int          # 1-based page number
    accuracy: float = 0.0     # confidence score (0-100) if available
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
                    float(str(val).replace(",", "").replace("<", "").replace(">", ""))
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
# Main extraction function with intelligent backend selection
# ──────────────────────────────────────────────────────────────────────────────

def extract_tables_from_pdf(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    min_rows: int = 3,
    min_cols: int = 3,
    backend: TableDetectorBackend = TableDetectorBackend.AUTO,
    force_backend: bool = False,
) -> tuple[list[ExtractedTable], TableDetectionMetrics]:
    """Extract tables from a PDF using intelligent backend selection.

    Args:
        pdf_path: Path to the PDF file.
        pages: 1-based page numbers to extract from. None = all pages.
        min_rows: Minimum rows for a table to be included.
        min_cols: Minimum columns for a table to be included.
        backend: Which backend to use (AUTO for intelligent selection).
        force_backend: If True and backend is specified, don't fall back.

    Returns:
        Tuple of (list of ExtractedTable objects, metrics).
        
    Strategy:
        - Try Docling first (ML-based, best for complex layouts)
        - Docling has memory-safe retry with OCR disabled fallback
        - If Docling fails/returns nothing: fall back to Camelot
        - If Camelot fails/returns nothing: fall back to pdfplumber
    """
    pdf_path = Path(pdf_path)
    tables: list[ExtractedTable] = []
    metrics = TableDetectionMetrics(backend_used="", tables_found=0, data_tables_found=0, pages_scanned=0)

    # Smart backend selection
    if backend == TableDetectorBackend.AUTO and not force_backend:
        # Try Docling first (best for complex layouts)
        # Memory-safe: automatically retries with OCR disabled on memory errors
        if _HAS_DOCLING:
            tables, metrics = _extract_with_docling(pdf_path, pages, min_rows, min_cols)
            if tables or metrics.errors:
                data_tables = [t for t in tables if t.is_data_table]
                logger.info(
                    "Docling: extracted %d total, %d data tables from %s",
                    len(tables), len(data_tables), pdf_path.name,
                )
                metrics.data_tables_found = len(data_tables)
                return data_tables, metrics

        # Fallback to Camelot (good for borderless tables)
        if _HAS_CAMELOT:
            tables, metrics = _extract_with_camelot(pdf_path, pages, min_rows, min_cols)
            if tables:
                data_tables = [t for t in tables if t.is_data_table]
                logger.info(
                    "Camelot: extracted %d total, %d data tables from %s",
                    len(tables), len(data_tables), pdf_path.name,
                )
                metrics.data_tables_found = len(data_tables)
                return data_tables, metrics

        # Final fallback to pdfplumber
        tables, metrics = _extract_with_pdfplumber(pdf_path, pages, min_rows, min_cols)
        data_tables = [t for t in tables if t.is_data_table]
        logger.info(
            "pdfplumber: extracted %d total, %d data tables from %s",
            len(tables), len(data_tables), pdf_path.name,
        )
        metrics.data_tables_found = len(data_tables)
        return data_tables, metrics

    # Specific backend requested
    else:
        if backend == TableDetectorBackend.DOCLING:
            if not _HAS_DOCLING:
                metrics.errors.append("Docling not installed")
                logger.warning("Docling requested but not available")
                if force_backend:
                    return [], metrics
                # Fall through to Camelot
                backend = TableDetectorBackend.CAMELOT

        if backend == TableDetectorBackend.CAMELOT:
            if not _HAS_CAMELOT:
                metrics.errors.append("Camelot not installed")
                logger.warning("Camelot requested but not available")
                if force_backend:
                    return [], metrics
                # Fall through to pdfplumber
                backend = TableDetectorBackend.PDFPLUMBER

        # Execute specific backend
        if _HAS_DOCLING and backend == TableDetectorBackend.DOCLING:
            tables, metrics = _extract_with_docling(pdf_path, pages, min_rows, min_cols)
        elif _HAS_CAMELOT and backend == TableDetectorBackend.CAMELOT:
            tables, metrics = _extract_with_camelot(pdf_path, pages, min_rows, min_cols)
        else:  # PDFPLUMBER
            tables, metrics = _extract_with_pdfplumber(pdf_path, pages, min_rows, min_cols)

        data_tables = [t for t in tables if t.is_data_table]
        metrics.data_tables_found = len(data_tables)
        return data_tables, metrics


def extract_tables_as_text(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    max_tables: int = 10,
    backend: TableDetectorBackend = TableDetectorBackend.AUTO,
) -> list[str]:
    """Extract tables from PDF and return as text strings.

    Convenience wrapper that returns tab-separated text suitable for
    LLM prompts. Compatible with the existing tables_text interface.
    """
    tables, _ = extract_tables_from_pdf(pdf_path, pages, backend=backend)
    result = []
    for t in tables[:max_tables]:
        header = f"[Page {t.page_number}, {t.backend}, {t.df.shape[0]}r x {t.df.shape[1]}c]"
        result.append(f"{header}\n{t.to_text()}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Backend: Docling (primary) — state-of-the-art document understanding
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_docling(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
) -> tuple[list[ExtractedTable], TableDetectionMetrics]:
    """Extract tables using Docling's document understanding.

    Uses the correct Docling API:
      - DocumentConverter with PdfPipelineOptions for configuration
      - result.document.tables for table iteration
      - table.export_to_dataframe(doc=result.document) for DataFrame export
      - TableFormerMode.ACCURATE for best table structure recognition

    Memory-safe: tries OCR-disabled first, then standard with OCR.
    """
    metrics = TableDetectionMetrics(backend_used="docling", tables_found=0, data_tables_found=0, pages_scanned=0)

    if not _HAS_DOCLING:
        metrics.errors.append("Docling not installed")
        return [], metrics

    tables: list[ExtractedTable] = []

    # Try extraction with progressively more resource-intensive configs
    configs = _get_docling_configs()
    for attempt, config in enumerate(configs, start=1):
        try:
            tables, metrics = _docling_convert_with_config(pdf_path, pages, min_rows, min_cols, config)
            if tables or not metrics.errors:
                logger.info("Docling succeeded on attempt %d with config: %s", attempt, config["name"])
                return tables, metrics
            elif attempt < len(configs):
                logger.info("Docling attempt %d found no tables, trying next config", attempt)
                continue
            else:
                return tables, metrics
        except (MemoryError, OSError, RuntimeError) as mem_err:
            error_msg = str(mem_err)
            if "bad_alloc" in error_msg or "allocate" in error_msg.lower() or isinstance(mem_err, MemoryError):
                logger.warning("Docling memory error on attempt %d (%s): %s", attempt, config["name"], error_msg)
                metrics.errors.append(f"Memory error (attempt {attempt}): {error_msg[:100]}")
                if attempt < len(configs):
                    continue
                else:
                    return tables, metrics
            else:
                raise

    return tables, metrics


def _get_docling_configs() -> list[dict]:
    """Docling converter configs ordered from least to most resource-intensive."""
    return [
        {
            "name": "no_ocr_accurate",
            "do_ocr": False,
            "table_mode": "accurate",
        },
        {
            "name": "with_ocr_accurate",
            "do_ocr": True,
            "table_mode": "accurate",
        },
    ]


def _docling_convert_with_config(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
    config: dict,
) -> tuple[list[ExtractedTable], TableDetectionMetrics]:
    """Convert PDF using Docling with the correct API.

    Uses:
      - PdfPipelineOptions for OCR/table config
      - FormatOption to bind options to InputFormat.PDF
      - result.document.tables for table access
      - table.export_to_dataframe(doc=...) for DataFrame export
    """
    metrics = TableDetectionMetrics(backend_used="docling", tables_found=0, data_tables_found=0, pages_scanned=0)
    tables: list[ExtractedTable] = []

    try:
        # Build pipeline options
        table_mode = (
            TableFormerMode.ACCURATE
            if config.get("table_mode") == "accurate"
            else TableFormerMode.FAST
        )

        pipeline_options = PdfPipelineOptions(
            do_ocr=config.get("do_ocr", False),
            do_table_structure=True,
            force_backend_text=not config.get("do_ocr", False),
        )
        # Set table structure mode
        pipeline_options.table_structure_options.mode = table_mode

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=StandardPdfPipeline,
                    backend=_DoclingPdfBackend,
                ),
            }
        )

        logger.info("Docling converting %s (config: %s)", pdf_path.name, config["name"])
        result = converter.convert(str(pdf_path))

        if result.status != ConversionStatus.SUCCESS:
            metrics.errors.append(f"Docling conversion status: {result.status}")
            # PARTIAL_SUCCESS may still have tables — continue
            if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                return [], metrics

        doc = result.document
        metrics.pages_scanned = doc.num_pages() if hasattr(doc, "num_pages") else 0

        # Use the correct API: result.document.tables
        for table_index, table_item in enumerate(doc.tables):
            try:
                # Get page number from table's location
                page_num = _get_docling_table_page(table_item)

                # Filter to requested pages
                if pages and page_num not in pages:
                    continue

                # Export to DataFrame using the correct API
                df = table_item.export_to_dataframe(doc=doc)

                if df is not None and not df.empty and df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                    df = _clean_table_df(df)
                    if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                        tables.append(ExtractedTable(
                            df=df,
                            page_number=page_num,
                            accuracy=100.0,
                            backend="docling",
                            table_index=table_index,
                        ))
            except Exception as e:
                logger.debug("Failed to export Docling table %d: %s", table_index, e)
                continue

        metrics.tables_found = len(tables)
        if tables:
            logger.info("Docling extracted %d tables from %s", len(tables), pdf_path.name)
        return tables, metrics

    except (MemoryError, OSError) as exc:
        raise
    except Exception as exc:
        metrics.errors.append(f"Docling extraction failed: {exc}")
        logger.warning("Docling extraction failed: %s", exc)
        return [], metrics


def _get_docling_table_page(table_item) -> int:
    """Extract 1-based page number from a Docling TableItem."""
    try:
        # TableItem has prov (provenance) with page_no
        if hasattr(table_item, "prov") and table_item.prov:
            prov = table_item.prov
            if isinstance(prov, list) and prov:
                return getattr(prov[0], "page_no", 1)
            return getattr(prov, "page_no", 1)
    except Exception:
        pass
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Backend: Camelot (fallback 1) — stream mode for borderless tables
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_camelot(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
) -> tuple[list[ExtractedTable], TableDetectionMetrics]:
    """Extract tables using Camelot's stream mode."""
    metrics = TableDetectionMetrics(backend_used="camelot", tables_found=0, data_tables_found=0, pages_scanned=0)

    if not _HAS_CAMELOT:
        metrics.errors.append("Camelot not installed")
        return [], metrics

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

        # Try lattice mode if stream mode fails
        if not camelot_tables:
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                flavor="lattice",
                pages=page_str,
                line_tol=2,
                char_margin=10,
            )

        for i, ct in enumerate(camelot_tables):
            df = ct.df
            if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                df = _clean_table_df(df)
                if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                    tables.append(ExtractedTable(
                        df=df,
                        page_number=ct.page,
                        accuracy=ct.accuracy,
                        backend="camelot",
                        table_index=i,
                    ))

        metrics.tables_found = len(tables)
        return tables, metrics

    except Exception as exc:
        metrics.errors.append(f"Camelot extraction failed: {exc}")
        logger.warning("Camelot extraction failed: %s", exc)
        return [], metrics


# ──────────────────────────────────────────────────────────────────────────────
# Backend: pdfplumber (fallback 2) — grid/bordered tables
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_pdfplumber(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
) -> tuple[list[ExtractedTable], TableDetectionMetrics]:
    """Extract tables using pdfplumber's built-in table detection."""
    metrics = TableDetectionMetrics(backend_used="pdfplumber", tables_found=0, data_tables_found=0, pages_scanned=0)

    try:
        import pdfplumber
    except ImportError:
        metrics.errors.append("pdfplumber not installed")
        return [], metrics

    tables: list[ExtractedTable] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            metrics.pages_scanned = len(pdf.pages)

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
                            accuracy=100.0,
                            backend="pdfplumber",
                            table_index=tbl_idx,
                        ))

        metrics.tables_found = len(tables)
        return tables, metrics

    except Exception as exc:
        metrics.errors.append(f"pdfplumber extraction failed: {exc}")
        logger.warning("pdfplumber extraction failed: %s", exc)
        return [], metrics


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


def get_available_backends() -> list[str]:
    """Return list of available table detection backends."""
    available = []
    if _HAS_DOCLING:
        available.append("docling")
    if _HAS_CAMELOT:
        available.append("camelot")
    available.append("pdfplumber")  # Always available
    return available


def get_backend_info() -> dict:
    """Return info about available backends."""
    return {
        "docling": {
            "available": _HAS_DOCLING,
            "description": "State-of-the-art document understanding (recommended)",
        },
        "camelot": {
            "available": _HAS_CAMELOT,
            "description": "Stream mode for borderless tables",
        },
        "pdfplumber": {
            "available": True,
            "description": "Grid/bordered tables (always available)",
        },
    }
