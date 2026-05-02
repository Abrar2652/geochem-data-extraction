"""
pdf_reader.py - Extract text and tables from research paper PDFs.

Uses pdfplumber for reliable text extraction. Returns both full-document text
and per-section text to help LLMs focus on relevant parts of the paper.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required: pip install pdfplumber")


# ──────────────────────────────────────────────────────────────────────────────
# Section keywords for automatic section detection
# ──────────────────────────────────────────────────────────────────────────────
SECTION_KEYWORDS = {
    "abstract":         ["abstract"],
    "introduction":     ["introduction", "1. introduction", "1 introduction"],
    "geological":       ["geological setting", "geology", "regional geology",
                         "study area", "ore geology", "deposit geology",
                         "tectonic setting", "geologic background",
                         "metallogenic setting", "regional setting"],
    "sampling":         ["sampling", "sample collection", "sample description",
                         "4. sampling", "samples", "sample selection",
                         "sample characterization", "petrography",
                         "sample preparation", "material and method",
                         "materials and method"],
    "analytical":       ["analytical method", "analytical technique",
                         "methods", "5. analytical", "5.1 laser", "la-icp-ms",
                         "icp-ms", "electron microprobe", "epma", "empa",
                         "trace element", "major element", "minor element",
                         "analytical procedure", "analytical condition",
                         "instrument", "experimental method",
                         "la-icpms", "sem-eds", "microprobe analysis",
                         "in situ", "in-situ"],
    "results":          ["results", "6. results", "6.1 minor", "trace element",
                         "geochemistry", "chemical composition",
                         "mineral chemistry", "sulfide chemistry",
                         "ore mineralogy", "major and trace"],
    "discussion":       ["discussion", "7. discussion", "interpretation"],
    "conclusions":      ["conclusion", "8. conclusion", "summary",
                         "concluding remark"],
    "references":       ["references", "bibliography", "literature cited"],
}

# Section heading patterns to detect page breaks between sections
_HEADING_RE = re.compile(
    r"^\s*(\d+\.?\d*\.?\s+)?("
    + "|".join([
        "abstract", "introduction", "geological", "study area", "ore geology",
        "tectonic setting", "metallogenic",
        "sampling", "sample", "petrograph", "material",
        "analytical", "method", "instrument", "la-icp", "epma", "empa",
        "trace element", "major element", "minor element", "in situ", "in-situ",
        "result", "geochemistry", "chemical composition", "mineral chemistry",
        "discussion", "interpretation",
        "conclusion", "summary",
        "reference", "bibliography", "appendix",
    ])
    + r")",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class PDFContent:
    """Structured content extracted from a research paper PDF."""
    full_text: str                         = ""
    pages: list[str]                       = field(default_factory=list)
    sections: dict[str, str]               = field(default_factory=dict)
    tables_text: list[str]                 = field(default_factory=list)
    total_pages: int                       = 0
    metadata: dict                         = field(default_factory=dict)
    marker_markdown: str                   = ""   # Full document via Marker (surya-based)

    @property
    def analytical_section(self) -> str:
        """Returns the analytical methods section, falling back to full text."""
        for key in ("analytical", "methods"):
            if key in self.sections and self.sections[key].strip():
                return self.sections[key]
        return self.full_text

    @property
    def sampling_section(self) -> str:
        for key in ("sampling",):
            if key in self.sections and self.sections[key].strip():
                return self.sections[key]
        return self.full_text

    @property
    def geological_section(self) -> str:
        for key in ("geological",):
            if key in self.sections and self.sections[key].strip():
                return self.sections[key]
        return self.full_text

    def section_summary(self) -> str:
        """Compact multi-section text useful for metadata extraction."""
        priority = ["abstract", "geological", "sampling", "analytical", "conclusions"]
        parts = []
        for sec in priority:
            if sec in self.sections and self.sections[sec].strip():
                parts.append(f"=== {sec.upper()} ===\n{self.sections[sec].strip()}")
        if parts:
            return "\n\n".join(parts)
        # Fallback: first 8000 chars of full text
        return self.full_text[:8000]


def extract_pdf(pdf_path: str | Path, max_chars_per_page: int = 5000) -> PDFContent:
    """Extract structured text from a PDF file.

    Uses pdftext (Marker's fast text layer) for page text extraction when
    available — produces cleaner output with intact table headers (e.g.,
    "Table 3. LA-ICP-MS results... (ppm)"). Falls back to pdfplumber.
    Always uses pdfplumber for structured table extraction.

    Args:
        pdf_path: Path to the PDF file.
        max_chars_per_page: Safety cap on chars extracted per page.

    Returns:
        PDFContent with full text, per-page text, and auto-detected sections.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    content = PDFContent()

    # Try pdftext first for page text (cleaner, faster)
    pdftext_pages = None
    try:
        from pdftext.extraction import paginated_plain_text_output
        pdftext_pages = paginated_plain_text_output(str(pdf_path), sort=True)
    except ImportError:
        pass
    except Exception:
        pass

    with pdfplumber.open(str(pdf_path)) as pdf:
        content.total_pages = len(pdf.pages)
        content.metadata = pdf.metadata or {}

        if pdftext_pages and len(pdftext_pages) == len(pdf.pages):
            # Use pdftext for page text (better quality)
            page_texts = [
                _clean_page_text(p)[:max_chars_per_page]
                for p in pdftext_pages
            ]
        else:
            # Fallback to pdfplumber text extraction
            page_texts = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = _clean_page_text(text)[:max_chars_per_page]
                page_texts.append(text)

        # Always use pdfplumber for structured table extraction
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables:
                tbl_str = _table_to_text(tbl)
                if tbl_str:
                    content.tables_text.append(tbl_str)

        content.pages = page_texts
        content.full_text = "\n\n".join(page_texts)
        content.sections = _detect_sections(content.full_text)

    # Generate Marker full-document markdown (non-blocking: graceful fallback)
    # Marker produces higher-quality markdown with intact table structures,
    # landscape page handling, and proper reading order — superior to pdfplumber
    # for complex geochemistry PDFs.
    try:
        from .tabledetector import marker_pdf_to_markdown
        marker_md = marker_pdf_to_markdown(pdf_path)
        if marker_md and len(marker_md) > 100:
            content.marker_markdown = marker_md
            logger.info("Marker markdown: %d chars (vs pdfplumber: %d chars)",
                       len(marker_md), len(content.full_text))
    except ImportError:
        logger.debug("Marker not available — using pdfplumber text only")
    except Exception as exc:
        logger.debug("Marker markdown extraction failed: %s", exc)

    return content


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_page_text(text: str) -> str:
    """Normalise whitespace and ligature artifacts from PDF extraction."""
    # Fix common PDF text extraction artifacts
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # CamelCase → separate
    text = re.sub(r"\s{3,}", "  ", text)               # Collapse excessive spaces
    text = re.sub(r"-\n", "", text)                    # Merge hyphenated line breaks
    text = text.strip()
    return text


def _table_to_text(table: list[list[Optional[str]]]) -> str:
    """Convert a pdfplumber table (list of rows) to a tab-separated string."""
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(c) if c is not None else "" for c in row]
        rows.append("\t".join(cells))
    return "\n".join(rows)


def _detect_sections(full_text: str) -> dict[str, str]:
    """Split the full paper text into named sections using heading detection.

    Uses a simple line-scan approach: when a line matches a known section
    heading, start a new section.
    """
    sections: dict[str, str] = {}
    current_section = "preamble"
    buffer: list[str] = []

    for line in full_text.splitlines():
        matched_section = _match_section_heading(line)
        if matched_section and matched_section != current_section:
            # Save previous section
            if buffer:
                sections[current_section] = sections.get(current_section, "") + "\n".join(buffer)
            buffer = []
            current_section = matched_section
        else:
            buffer.append(line)

    # Save final section
    if buffer:
        sections[current_section] = sections.get(current_section, "") + "\n".join(buffer)

    return sections


def _match_section_heading(line: str) -> Optional[str]:
    """Return the canonical section name if `line` looks like a section heading.

    Handles:
    - Plain headings: "Abstract", "Analytical Methods"
    - Numbered: "5. Analytical methods", "4.1 LA-ICP-MS analysis"
    - Sub-numbered: "5.2.1 Trace elements"
    """
    stripped = line.strip().lower()
    if not stripped or len(stripped) > 120:
        return None

    # Strip leading section numbers: "5.", "4.1", "5.2.1", etc.
    text = re.sub(r"^\d+(\.\d+)*\.?\s+", "", stripped)

    for section_name, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if text.startswith(kw):
                return section_name
            if stripped.startswith(kw):
                return section_name
    return None


def score_pages_for_data(content: PDFContent) -> list[tuple[int, float]]:
    """Score each page by how likely it contains geochemical data tables.

    Returns a list of (page_index, score) sorted descending by score.
    Only pages that pass minimum thresholds are included.

    Heuristic: pages with element symbols/units AND numeric density,
    excluding reference/bibliography pages.

    Uses two tiers:
    - Strong match: >= 2 data markers AND >= 2 numeric lines (normal tables)
    - Weak match: >= 1 data marker AND >= 1 numeric line (landscape/garbled
      pages where text extraction partially works)
    """
    # Element symbols and units that indicate geochemical data
    _DATA_MARKERS = {
        "ppm", "wt%", "wt.%", "ppb", "mg/kg",
        # common column headers in geochem tables
        "fe", "cu", "zn", "pb", "ag", "au", "as", "sb", "bi", "co",
        "ni", "mn", "cd", "ga", "ge", "se", "te", "sn", "mo",
        "sio2", "tio2", "al2o3", "fe2o3", "feo", "mno", "mgo",
        "cao", "na2o", "k2o", "p2o5", "cr2o3",
        "sample", "spot", "analysis",
        # Table captions commonly seen in geochem papers
        "table", "appendix",
    }
    _REFERENCE_MARKERS = {"doi:", "doi.org", "http", "downloaded from",
                          "bibliography", "et cosmochimica", "geochimica"}

    scored: list[tuple[int, float]] = []
    for i, page_text in enumerate(content.pages):
        lines = page_text.split("\n")
        if not lines:
            continue

        page_lower = page_text.lower()

        # Skip pages that look like references/bibliography
        ref_signals = sum(1 for m in _REFERENCE_MARKERS if m in page_lower)
        if ref_signals >= 3:
            continue

        # Count data markers on this page
        data_marker_count = sum(1 for m in _DATA_MARKERS if m in page_lower)

        # Count lines with significant digit content (excluding DOI-like lines)
        numeric_lines = 0
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["doi:", "http", "downloaded"]):
                continue
            digits = sum(c.isdigit() or c == "." for c in line)
            if digits > 10 and digits / max(len(line), 1) > 0.15:
                numeric_lines += 1

        # Strong match: clearly a data table page
        if data_marker_count >= 2 and numeric_lines >= 2:
            score = data_marker_count * 0.3 + numeric_lines * 0.7
            scored.append((i, score))
        # Weak match: partial signals — landscape/garbled pages where text
        # extraction only partially works, or pages with "Table X" caption
        # but garbled data. Scored lower so they rank after strong matches.
        elif data_marker_count >= 1 and numeric_lines >= 1:
            score = (data_marker_count * 0.3 + numeric_lines * 0.7) * 0.5
            scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def detect_rotated_pages(pdf_path: str | Path) -> list[tuple[int, int]]:
    """Detect pages with rotated content (landscape tables) using PyMuPDF.

    Returns a list of (page_index, rotation_angle) for pages where the
    majority of text is rotated (i.e., landscape tables in portrait pages).

    This is critical for papers like Bonnet et al. where tables are in
    landscape orientation but the page is tagged as portrait.
    """
    import math

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []

    rotated_pages: list[tuple[int, int]] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            blocks = page.get_text("rawdict")["blocks"]

            cos_sum, sin_sum, count = 0.0, 0.0, 0
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        d = span.get("dir", (1, 0))
                        cos_sum += d[0]
                        sin_sum += d[1]
                        count += 1

            if count < 10:
                continue

            avg_cos = cos_sum / count
            avg_sin = sin_sum / count

            # Normal horizontal text: cos~1, sin~0
            # 90 CCW (landscape): cos~0, sin~1
            # 90 CW: cos~0, sin~-1
            if abs(avg_cos) < 0.3:
                angle = round(math.degrees(math.atan2(avg_sin, avg_cos)))
                # Normalize to standard angles
                if abs(angle - 90) < 15:
                    rotated_pages.append((page_idx, 90))
                elif abs(angle + 90) < 15:
                    rotated_pages.append((page_idx, -90))

        doc.close()
    except Exception:
        pass

    return rotated_pages


def extract_rotated_page_text(pdf_path: str | Path, page_idx: int) -> str:
    """Extract text from a rotated page by counter-rotating coordinates.

    Uses PyMuPDF to read text spans with their direction vectors, then
    groups them into lines based on the dominant (rotated) reading order.
    """
    try:
        import fitz
    except ImportError:
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_idx]

        # Re-render with page rotation applied to normalize content
        original_rotation = page.rotation
        # Try setting rotation to 0 and re-extracting
        page.set_rotation(0)
        text = page.get_text("text")

        if not text or len(text.strip()) < 50:
            # Fallback: render as image and get text from the rotated view
            page.set_rotation(original_rotation)
            # Get text with rotation normalization by transforming the matrix
            mat = fitz.Matrix(1, 1).prerotate(-90)
            # Extract text from a pixmap would require OCR, so instead
            # we extract raw dict and re-sort by rotated coordinates
            blocks = page.get_text("rawdict")["blocks"]
            lines_data = []  # (y_rotated, x_rotated, text)
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        d = span.get("dir", (1, 0))
                        bbox = span["bbox"]  # (x0, y0, x1, y1)
                        txt = span["text"]
                        if not txt.strip():
                            continue
                        # For 90-degree rotated text, swap and invert coordinates
                        # to get the "reading order" position
                        if abs(d[1]) > 0.5:  # Rotated text
                            # New x = old y, new y = -old x (for 90 CCW)
                            new_x = bbox[1]  # original y becomes x
                            new_y = -bbox[0]  # original x becomes y (inverted)
                            lines_data.append((round(new_y, 1), round(new_x, 1), txt))
                        else:
                            lines_data.append((round(bbox[1], 1), round(bbox[0], 1), txt))

            # Sort by y (line position) then x (column position)
            lines_data.sort(key=lambda t: (t[0], t[1]))

            # Group by y-position into lines (tolerance of 5 units)
            result_lines = []
            current_y = None
            current_line_parts = []
            for y, x, txt in lines_data:
                if current_y is None or abs(y - current_y) > 5:
                    if current_line_parts:
                        current_line_parts.sort(key=lambda p: p[0])
                        result_lines.append("  ".join(t for _, t in current_line_parts))
                    current_y = y
                    current_line_parts = [(x, txt)]
                else:
                    current_line_parts.append((x, txt))
            if current_line_parts:
                current_line_parts.sort(key=lambda p: p[0])
                result_lines.append("  ".join(t for _, t in current_line_parts))

            text = "\n".join(result_lines)

        doc.close()
        return text
    except Exception:
        return ""


def get_data_pages_text(content: PDFContent, max_chars: int = 15000) -> str:
    """Extract raw text from pages that likely contain geochemical data tables.

    Uses score_pages_for_data() to identify data-dense pages, then returns
    their text concatenated.
    """
    scored = score_pages_for_data(content)
    if not scored:
        return ""

    parts = []
    total = 0
    for page_idx, score in scored:
        text = content.pages[page_idx]
        header = f"\n--- Page {page_idx + 1} ---\n"
        parts.append(header + text)
        total += len(header) + len(text)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def get_paper_text_for_llm(content: PDFContent, max_chars: int = 20000) -> str:
    """Return the most relevant text for LLM metadata extraction.

    Prioritises key sections in order, ensuring critical sections (abstract,
    analytical) are never truncated. Falls back to full text if sections
    not found.
    """
    # Critical sections that should always be included in full
    critical = ["abstract", "analytical"]
    # Additional sections in priority order
    supplementary = ["geological", "sampling", "results", "conclusions"]

    # Keywords that a real analytical section should contain
    _ANALYTICAL_MARKERS = {
        "laser", "ablation", "icp-ms", "icpms", "epma", "microprobe",
        "spot size", "beam", "standard", "calibrat", "carrier gas",
        "dwell time", "repetition", "instrument", "spectrometer",
    }

    parts = []
    total = 0

    # Always include critical sections first (no truncation)
    for sec in critical:
        if sec in content.sections:
            chunk = content.sections[sec].strip()
            if chunk:
                # Validate "analytical" section actually contains methods content
                if sec == "analytical" and len(chunk) > 5000:
                    chunk_lower = chunk[:5000].lower()
                    marker_count = sum(1 for m in _ANALYTICAL_MARKERS if m in chunk_lower)
                    if marker_count < 2:
                        # Mislabeled section — skip it, use keyword extraction below
                        continue
                header = f"\n{'='*60}\n[SECTION: {sec.upper()}]\n{'='*60}\n"
                parts.append(header + chunk)
                total += len(header) + len(chunk)

    # Add supplementary sections up to budget
    for sec in supplementary:
        if sec in content.sections:
            chunk = content.sections[sec].strip()
            if chunk:
                header = f"\n{'='*60}\n[SECTION: {sec.upper()}]\n{'='*60}\n"
                section_len = len(header) + len(chunk)
                if total + section_len <= max_chars:
                    parts.append(header + chunk)
                    total += section_len
                elif total < max_chars:
                    # Include partial section to fill remaining budget
                    remaining = max_chars - total - len(header)
                    if remaining > 200:
                        parts.append(header + chunk[:remaining])
                        total = max_chars
                    break

    # Check if we have analytical methods content — if not, extract via keywords
    has_analytical = any("[SECTION: ANALYTICAL]" in p for p in parts)
    if not has_analytical and content.full_text:
        methods_chunk = _extract_methods_paragraphs(content.full_text)
        if methods_chunk:
            header = f"\n{'='*60}\n[SECTION: ANALYTICAL METHODS]\n{'='*60}\n"
            parts.append(header + methods_chunk)
            total += len(header) + len(methods_chunk)

    if not parts:
        # Prefer Marker markdown when available (better layout, table structure)
        if content.marker_markdown:
            return content.marker_markdown[:max_chars]
        return content.full_text[:max_chars]

    result = "\n".join(parts)

    # Append Marker markdown table sections as supplementary context.
    # Marker often captures tables that pdfplumber misses (landscape, rotated,
    # complex headers). Include Marker's table content in remaining budget.
    if content.marker_markdown and total < max_chars:
        marker_tables = _extract_marker_table_sections(content.marker_markdown)
        if marker_tables:
            remaining = max_chars - total
            marker_header = (
                f"\n{'='*60}\n"
                f"[MARKER DOCUMENT TABLES — higher-fidelity table extraction]\n"
                f"{'='*60}\n"
            )
            marker_content = marker_header + marker_tables
            if len(marker_content) <= remaining:
                result += marker_content
            elif remaining > 500:
                result += marker_content[:remaining]

    return result[:max_chars]


def _extract_marker_table_sections(marker_md: str, max_chars: int = 8000) -> str:
    """Extract table sections from Marker's full-document markdown.

    Marker outputs tables as pipe-delimited markdown. This function extracts
    all table blocks (including their captions/headers) to provide the LLM
    with higher-fidelity table content than pdfplumber.

    Returns concatenated table sections, or empty string if no tables found.
    """
    if not marker_md:
        return ""

    import re

    tables = []
    lines = marker_md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect table start: line with pipe characters (markdown table row)
        if "|" in line and line.strip().startswith("|"):
            # Capture context: 3 lines before table (likely caption/header)
            start = max(0, i - 3)
            context_lines = lines[start:i]

            # Capture all table rows (consecutive lines with |)
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1

            # Capture 1 line after table (footnotes)
            end_context = lines[i:i+1] if i < len(lines) else []

            if len(table_lines) >= 2:  # At least header + 1 data row
                table_block = (
                    "\n".join(context_lines).strip()
                    + "\n" + "\n".join(table_lines)
                    + "\n" + "\n".join(end_context).strip()
                )
                tables.append(table_block.strip())
        i += 1

    if not tables:
        return ""

    result = "\n\n---\n\n".join(tables)
    return result[:max_chars]


def _extract_methods_paragraphs(full_text: str, max_chars: int = 6000) -> str:
    """Extract analytical methods content from full text using keyword anchors.

    When formal section detection fails, this uses keyword proximity to find
    paragraphs containing instrument, standards, and lab details.
    """
    keywords = [
        "LA-ICP-MS", "LA-ICPMS", "laser ablation", "EPMA", "electron microprobe",
        "ICP-MS", "spot size", "beam diameter", "reference material",
        "calibration standard", "NIST SRM", "MASS-1", "carrier gas",
        "operating condition", "analytical condition", "ablation",
        "repetition rate", "pulse rate", "fluence",
    ]
    text_lower = full_text.lower()
    # Find all keyword positions
    positions = []
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        while idx != -1:
            positions.append(idx)
            idx = text_lower.find(kw.lower(), idx + 1)

    if not positions:
        return ""

    # Cluster positions and extract surrounding paragraphs
    positions.sort()
    # Merge nearby positions into ranges (within 500 chars)
    ranges = []
    for pos in positions:
        start = max(0, pos - 200)
        end = min(len(full_text), pos + 400)
        if ranges and start <= ranges[-1][1] + 200:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    # Collect extracted text
    chunks = []
    total = 0
    for start, end in ranges:
        chunk = full_text[start:end].strip()
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk)

    return "\n...\n".join(chunks)
