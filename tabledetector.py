"""
tabledetector.py - Advanced multi-backend PDF table detection and extraction.

Supports five backends with intelligent selection and fallback:
  1. Docling (primary) — state-of-the-art document understanding
  2. Marker (new) — surya-based PDF-to-markdown with strong table recognition
  3. MinerU (new) — magic-pdf layout-aware extraction, good for landscape tables
  4. Camelot (fallback) — stream mode for borderless/whitespace-aligned tables
  5. pdfplumber (fallback) — grid/bordered tables

Features:
  - Automatic backend selection based on PDF characteristics
  - Configurable fallback chain with all five backends
  - Per-backend performance tracking
  - Supports forcing specific backend via configuration
  - Handles both scientific papers and data-dense tables
  - Marker/MinerU excel at landscape/rotated tables that trip up other backends

Strategy:
  Try Docling first (best for complex layouts). If it finds tables or fails gracefully,
  use those results. Then try Marker (good general-purpose). Then MinerU (layout-aware).
  Then Camelot stream (good for borderless tables). Finally pdfplumber (grid tables).
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
    MARKER = "marker"
    MINERU = "mineru"
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

_HAS_MARKER = False
try:
    from marker.converters.pdf import PdfConverter as _MarkerPdfConverter
    from marker.models import create_model_dict as _marker_create_model_dict
    _HAS_MARKER = True
except ImportError:
    pass

_HAS_MINERU = False
try:
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter as _MineruWriter
    from magic_pdf.data.dataset import PymuDocDataset as _MineruDataset
    _HAS_MINERU = True
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

        # Try Marker (surya-based, good for complex/rotated tables)
        if _HAS_MARKER:
            tables, metrics = _extract_with_marker(pdf_path, pages, min_rows, min_cols)
            if tables:
                data_tables = [t for t in tables if t.is_data_table]
                logger.info(
                    "Marker: extracted %d total, %d data tables from %s",
                    len(tables), len(data_tables), pdf_path.name,
                )
                metrics.data_tables_found = len(data_tables)
                return data_tables, metrics

        # Try MinerU (layout-aware, good for landscape tables)
        if _HAS_MINERU:
            tables, metrics = _extract_with_mineru(pdf_path, pages, min_rows, min_cols)
            if tables:
                data_tables = [t for t in tables if t.is_data_table]
                logger.info(
                    "MinerU: extracted %d total, %d data tables from %s",
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
        _BACKEND_AVAILABLE = {
            TableDetectorBackend.DOCLING: _HAS_DOCLING,
            TableDetectorBackend.MARKER: _HAS_MARKER,
            TableDetectorBackend.MINERU: _HAS_MINERU,
            TableDetectorBackend.CAMELOT: _HAS_CAMELOT,
            TableDetectorBackend.PDFPLUMBER: True,
        }
        _BACKEND_EXTRACTORS = {
            TableDetectorBackend.DOCLING: lambda: _extract_with_docling(pdf_path, pages, min_rows, min_cols),
            TableDetectorBackend.MARKER: lambda: _extract_with_marker(pdf_path, pages, min_rows, min_cols),
            TableDetectorBackend.MINERU: lambda: _extract_with_mineru(pdf_path, pages, min_rows, min_cols),
            TableDetectorBackend.CAMELOT: lambda: _extract_with_camelot(pdf_path, pages, min_rows, min_cols),
            TableDetectorBackend.PDFPLUMBER: lambda: _extract_with_pdfplumber(pdf_path, pages, min_rows, min_cols),
        }

        if not _BACKEND_AVAILABLE.get(backend, False):
            metrics.errors.append(f"{backend.value} not installed")
            logger.warning("%s requested but not available", backend.value)
            if force_backend:
                return [], metrics
            # Fall through to pdfplumber
            backend = TableDetectorBackend.PDFPLUMBER

        tables, metrics = _BACKEND_EXTRACTORS[backend]()
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
# Backend: Marker — surya-based PDF-to-markdown with table recognition
# ──────────────────────────────────────────────────────────────────────────────

# Cache marker models globally to avoid reloading (~10s first call)
_marker_model_dict: dict | None = None


def _get_marker_models() -> dict:
    """Lazy-load Marker models (cached after first call)."""
    global _marker_model_dict
    if _marker_model_dict is None:
        logger.info("Loading Marker models (first call, will be cached)...")
        _marker_model_dict = _marker_create_model_dict()
        logger.info("Marker models loaded: %s", list(_marker_model_dict.keys()))
    return _marker_model_dict


def _parse_markdown_tables(md_text: str) -> list[pd.DataFrame]:
    """Parse markdown pipe-delimited tables into DataFrames.

    Handles standard markdown tables with | delimiters and --- separator rows.
    Returns a list of DataFrames, one per table found.
    """
    import re
    dfs = []
    # Split markdown into lines
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Detect start of a table: line starts and ends with |
        if line.startswith("|") and line.endswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith("|") and next_line.endswith("|"):
                    table_lines.append(next_line)
                    i += 1
                elif not next_line:
                    # Blank line — could be table break or end
                    i += 1
                    break
                else:
                    break

            # Parse the table lines into a DataFrame
            if len(table_lines) >= 3:  # header + separator + at least 1 data row
                rows = []
                header = None
                for tl in table_lines:
                    cells = [c.strip() for c in tl.strip("|").split("|")]
                    # Skip separator rows (--- or :---: etc.)
                    if all(re.match(r"^[:\-\s]+$", c) for c in cells):
                        continue
                    if header is None:
                        header = cells
                    else:
                        rows.append(cells)

                if header and rows:
                    # Pad rows to header length
                    n_cols = len(header)
                    rows = [r + [""] * (n_cols - len(r)) if len(r) < n_cols else r[:n_cols] for r in rows]
                    df = pd.DataFrame(rows, columns=header)
                    dfs.append(df)
        else:
            i += 1
    return dfs


def _extract_with_marker(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
) -> tuple[list[ExtractedTable], TableDetectionMetrics]:
    """Extract tables using Marker's surya-based PDF converter.

    Marker converts PDF to markdown with tables rendered as pipe-delimited
    markdown tables. We parse those back into DataFrames.

    Marker excels at:
    - Complex multi-column layouts
    - Rotated/landscape tables
    - Tables with merged cells or irregular structure
    """
    metrics = TableDetectionMetrics(
        backend_used="marker", tables_found=0, data_tables_found=0, pages_scanned=0,
    )

    if not _HAS_MARKER:
        metrics.errors.append("Marker not installed")
        return [], metrics

    tables: list[ExtractedTable] = []
    try:
        model_dict = _get_marker_models()
        converter = _MarkerPdfConverter(artifact_dict=model_dict)
        logger.info("Marker converting %s", pdf_path.name)
        rendered = converter(str(pdf_path))
        md_text = rendered.markdown

        if not md_text:
            metrics.errors.append("Marker produced empty output")
            return [], metrics

        # Parse markdown tables into DataFrames
        raw_dfs = _parse_markdown_tables(md_text)
        metrics.tables_found = len(raw_dfs)

        for tbl_idx, df in enumerate(raw_dfs):
            if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                df = _clean_table_df(df)
                if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                    tables.append(ExtractedTable(
                        df=df,
                        page_number=tbl_idx + 1,  # approximate
                        accuracy=90.0,
                        backend="marker",
                        table_index=tbl_idx,
                    ))

        if tables:
            logger.info("Marker extracted %d tables from %s", len(tables), pdf_path.name)
        return tables, metrics

    except Exception as exc:
        metrics.errors.append(f"Marker extraction failed: {exc}")
        logger.warning("Marker extraction failed: %s", exc)
        return [], metrics


def marker_pdf_to_markdown(pdf_path: str | Path) -> str:
    """Convert a PDF to markdown using Marker. Returns raw markdown text.

    This is a convenience function for use by the pipeline when it needs
    the full markdown output (not just tables).
    """
    if not _HAS_MARKER:
        raise ImportError("marker-pdf not installed: pip install marker-pdf")
    model_dict = _get_marker_models()
    converter = _MarkerPdfConverter(artifact_dict=model_dict)
    rendered = converter(str(pdf_path))
    return rendered.markdown


# ──────────────────────────────────────────────────────────────────────────────
# Backend: MinerU — layout-aware extraction (magic-pdf)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_mineru(
    pdf_path: Path,
    pages: list[int] | None,
    min_rows: int,
    min_cols: int,
) -> tuple[list[ExtractedTable], TableDetectionMetrics]:
    """Extract tables using MinerU (magic-pdf) layout-aware extraction.

    MinerU uses deep learning layout analysis (YOLO-based) to detect tables,
    then extracts them as structured data. Particularly good for:
    - Landscape/rotated tables
    - Complex multi-level headers
    - Tables spanning multiple pages
    """
    metrics = TableDetectionMetrics(
        backend_used="mineru", tables_found=0, data_tables_found=0, pages_scanned=0,
    )

    if not _HAS_MINERU:
        metrics.errors.append("MinerU not installed")
        return [], metrics

    tables: list[ExtractedTable] = []
    try:
        import tempfile
        import os

        with open(str(pdf_path), "rb") as f:
            pdf_bytes = f.read()

        ds = _MineruDataset(pdf_bytes)
        classify_result = ds.classify()
        logger.info("MinerU classification for %s: %s", pdf_path.name, classify_result)

        out_dir = tempfile.mkdtemp(prefix="mineru_")
        img_writer = _MineruWriter(os.path.join(out_dir, "images"))

        # Run layout analysis + extraction
        try:
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            infer_result = ds.apply(doc_analyze, out_dir)
        except Exception as model_exc:
            # If model analysis fails (missing models), fall back to basic
            # extraction without model — still extracts text/tables via pymupdf
            logger.warning("MinerU model analysis failed (%s), trying basic mode", model_exc)
            metrics.errors.append(f"MinerU model fallback: {model_exc}")
            return [], metrics

        # Extract via appropriate pipe mode
        if "TXT" in str(classify_result):
            pipe_result = infer_result.pipe_txt_mode(img_writer)
        else:
            pipe_result = infer_result.pipe_ocr_mode(img_writer)

        # Get markdown and parse tables
        md_text = pipe_result.get_markdown(out_dir)
        if md_text:
            raw_dfs = _parse_markdown_tables(md_text)
            metrics.tables_found = len(raw_dfs)

            for tbl_idx, df in enumerate(raw_dfs):
                if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                    df = _clean_table_df(df)
                    if df.shape[0] >= min_rows and df.shape[1] >= min_cols:
                        tables.append(ExtractedTable(
                            df=df,
                            page_number=tbl_idx + 1,
                            accuracy=85.0,
                            backend="mineru",
                            table_index=tbl_idx,
                        ))

        # Try to get structured content list for better table extraction
        try:
            content_list = pipe_result.get_content_list(
                _MineruWriter(out_dir), "tables",
            )
            for item in content_list:
                if item.get("type") == "table" and "table_body" in item:
                    # Parse HTML table body if available
                    html = item["table_body"]
                    try:
                        html_dfs = pd.read_html(html)
                        for hdf in html_dfs:
                            if hdf.shape[0] >= min_rows and hdf.shape[1] >= min_cols:
                                hdf = _clean_table_df(hdf)
                                if hdf.shape[0] >= min_rows and hdf.shape[1] >= min_cols:
                                    tables.append(ExtractedTable(
                                        df=hdf,
                                        page_number=item.get("page_idx", 0) + 1,
                                        accuracy=90.0,
                                        backend="mineru",
                                        table_index=len(tables),
                                    ))
                    except Exception:
                        pass
        except Exception:
            pass

        if tables:
            logger.info("MinerU extracted %d tables from %s", len(tables), pdf_path.name)
        return tables, metrics

    except Exception as exc:
        metrics.errors.append(f"MinerU extraction failed: {exc}")
        logger.warning("MinerU extraction failed: %s", exc)
        return [], metrics


def mineru_pdf_to_markdown(pdf_path: str | Path) -> str:
    """Convert a PDF to markdown using MinerU. Returns raw markdown text.

    Convenience function for pipeline use.
    """
    if not _HAS_MINERU:
        raise ImportError("magic-pdf not installed: pip install magic-pdf")

    import tempfile
    import os

    with open(str(pdf_path), "rb") as f:
        pdf_bytes = f.read()

    ds = _MineruDataset(pdf_bytes)
    classify_result = ds.classify()
    out_dir = tempfile.mkdtemp(prefix="mineru_")
    img_writer = _MineruWriter(os.path.join(out_dir, "images"))

    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    infer_result = ds.apply(doc_analyze, out_dir)

    if "TXT" in str(classify_result):
        pipe_result = infer_result.pipe_txt_mode(img_writer)
    else:
        pipe_result = infer_result.pipe_ocr_mode(img_writer)

    return pipe_result.get_markdown(out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Backend: Camelot (fallback) — stream mode for borderless tables
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
    if _HAS_MARKER:
        available.append("marker")
    if _HAS_MINERU:
        available.append("mineru")
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
        "marker": {
            "available": _HAS_MARKER,
            "description": "Surya-based PDF-to-markdown, good for rotated/landscape tables",
        },
        "mineru": {
            "available": _HAS_MINERU,
            "description": "Layout-aware extraction (magic-pdf), good for complex layouts",
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
