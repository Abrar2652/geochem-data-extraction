"""
agentic_corrector.py - LLM-powered self-correction for supplementary table parsing.

When rule-based extraction fails (0 rows or very low quality), this module:
  1. Builds a raw preview of the Excel/CSV file for LLM inspection
  2. Asks the LLM to diagnose the failure and return structured parsing hints
  3. Retries extraction with those hints applied
  4. Keeps the best result across all attempts

Bounded to max_attempts (default 2) LLM calls per failing file.
Only invoked when quick_quality_check detects a problem.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import normalise_element_header

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsingHints:
    """Structured hints from LLM diagnosis that override auto-detection in table_reader.

    Fields default to None, meaning "use existing auto-detection."
    Only set fields where the LLM believes auto-detection went wrong.
    """
    header_row: Optional[int] = None
    is_transposed: Optional[bool] = None
    sample_id_col: Optional[str] = None
    unit: Optional[str] = None            # "ppm", "wt%", or "mixed"
    element_label_col: Optional[int] = None  # for transposed: which col has element names
    skip_sheets: Optional[list[str]] = None
    target_sheets: Optional[list[str]] = None
    notes_from_llm: str = ""


@dataclass
class QuickQualityResult:
    """Fast quality assessment (no LLM calls)."""
    passed: bool
    row_count: int
    element_count: int
    quality_score: float        # 0-100
    failure_reasons: list[str]  # human-readable reasons


@dataclass
class CorrectionAttempt:
    """Record of one correction attempt."""
    attempt_number: int
    hints: Optional[ParsingHints] = None
    row_count: int = 0
    element_count: int = 0
    quality_score: float = 0.0
    diagnosis_summary: str = ""
    duration_sec: float = 0.0


@dataclass
class CorrectionMetrics:
    """Full metrics for the self-correction process on one paper."""
    paper_id: str = ""
    initial_row_count: int = 0
    initial_element_count: int = 0
    initial_quality_score: float = 0.0
    attempts: list[CorrectionAttempt] = field(default_factory=list)
    final_row_count: int = 0
    final_element_count: int = 0
    final_quality_score: float = 0.0
    correction_needed: bool = False
    correction_succeeded: bool = False

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "correction_needed": self.correction_needed,
            "correction_succeeded": self.correction_succeeded,
            "initial_rows": self.initial_row_count,
            "initial_elements": self.initial_element_count,
            "initial_quality": self.initial_quality_score,
            "final_rows": self.final_row_count,
            "final_elements": self.final_element_count,
            "final_quality": self.final_quality_score,
            "n_attempts": len(self.attempts),
            "attempt_details": [
                {
                    "attempt": a.attempt_number,
                    "rows": a.row_count,
                    "elements": a.element_count,
                    "quality": a.quality_score,
                    "diagnosis": a.diagnosis_summary,
                    "duration_sec": a.duration_sec,
                }
                for a in self.attempts
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Quick quality check
# ──────────────────────────────────────────────────────────────────────────────

def quick_quality_check(
    supp,  # Optional[SupplementaryTable] — avoid circular import
    min_rows: int = 1,
    min_elements: int = 3,
) -> QuickQualityResult:
    """Fast O(1) quality check on a SupplementaryTable.

    Checks:
      - supp is not None
      - n_samples > 0
      - element_col_map has >= min_elements
      - sample_id_col was detected
      - No suspiciously uniform data (all values identical)

    Returns QuickQualityResult with passed=True if extraction looks viable.
    """
    reasons: list[str] = []

    if supp is None:
        return QuickQualityResult(
            passed=False, row_count=0, element_count=0,
            quality_score=0.0,
            failure_reasons=["Table read returned None (exception during parsing)"],
        )

    n_rows = supp.n_samples
    n_elements = len(supp.element_col_map)

    if n_rows == 0:
        reasons.append("0 rows extracted after filtering")
    if n_elements == 0:
        reasons.append("No element columns detected")
    elif n_elements < min_elements:
        reasons.append(f"Only {n_elements} element columns (expected >= {min_elements})")
    if supp.sample_id_col is None:
        reasons.append("No sample ID column detected")

    # Check for suspiciously uniform data (header row leaked into data)
    if n_rows > 3 and n_elements > 0:
        try:
            elem_cols = list(supp.element_col_map.keys())[:5]  # check first 5 elements
            for col in elem_cols:
                if col in supp.data_df.columns:
                    vals = supp.data_df[col].dropna()
                    if len(vals) > 3 and vals.nunique() == 1:
                        reasons.append(f"Column '{col}' has identical values across all rows (likely header leak)")
                        break
        except Exception:
            pass  # non-critical check

    # Compute a simple quality score (0-100)
    score = 0.0
    if n_rows > 0:
        score += 40.0
    if n_elements >= min_elements:
        score += 30.0
    elif n_elements > 0:
        score += 15.0
    if supp.sample_id_col is not None:
        score += 20.0
    if not reasons:
        score += 10.0  # bonus for no issues

    passed = n_rows >= min_rows and n_elements >= min_elements
    return QuickQualityResult(
        passed=passed,
        row_count=n_rows,
        element_count=n_elements,
        quality_score=score,
        failure_reasons=reasons,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Raw table preview for LLM
# ──────────────────────────────────────────────────────────────────────────────

def build_raw_table_preview(
    paths: list[Path],
    max_rows_per_sheet: int = 25,
    max_chars: int = 5000,
) -> str:
    """Read supplementary files WITHOUT any processing, return text preview for LLM.

    Shows first N rows exactly as they appear, with row numbers.
    For multi-sheet Excel files, shows each sheet.
    """
    parts: list[str] = []

    for path in paths:
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            parts.append(_preview_excel(path, max_rows_per_sheet))
        elif suffix == ".csv":
            parts.append(_preview_csv(path, max_rows_per_sheet))
        else:
            parts.append(f"[{path.name}: unsupported format {suffix}]")

    full = "\n\n".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n... (truncated)"
    return full


def _preview_excel(path: Path, max_rows: int) -> str:
    """Preview all sheets of an Excel file."""
    try:
        if path.suffix.lower() == ".xls":
            import xlrd
            wb = xlrd.open_workbook(str(path))
            sheet_names = wb.sheet_names()
        else:
            import openpyxl
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            sheet_names = wb.sheetnames
            wb.close()
    except Exception as exc:
        return f"[{path.name}: could not open — {exc}]"

    parts = [f"=== FILE: {path.name} ({len(sheet_names)} sheets) ==="]

    for sname in sheet_names:
        try:
            raw = pd.read_excel(path, sheet_name=sname, header=None, dtype=str,
                                nrows=max_rows)
            parts.append(f"\n--- Sheet: '{sname}' ({raw.shape[0]} rows shown, {raw.shape[1]} columns) ---")
            parts.append(_df_to_preview(raw))
        except Exception as exc:
            parts.append(f"\n--- Sheet: '{sname}': read error — {exc} ---")

    return "\n".join(parts)


def _preview_csv(path: Path, max_rows: int) -> str:
    """Preview a CSV file."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = pd.read_csv(path, header=None, dtype=str, nrows=max_rows,
                              encoding=enc)
            parts = [f"=== FILE: {path.name} (CSV, {raw.shape[1]} columns) ==="]
            parts.append(_df_to_preview(raw))
            return "\n".join(parts)
        except (UnicodeDecodeError, Exception):
            continue
    return f"[{path.name}: could not read CSV]"


def _df_to_preview(df: pd.DataFrame) -> str:
    """Convert DataFrame to pipe-delimited text with row indices."""
    lines = []
    for idx in range(len(df)):
        vals = [str(v) if pd.notna(v) else "" for v in df.iloc[idx]]
        lines.append(f"Row {idx:>2} | " + " | ".join(vals))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# LLM diagnosis prompt
# ──────────────────────────────────────────────────────────────────────────────

_DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert geochemistry data parsing specialist. Your job is to diagnose \
why automatic parsing of a supplementary geochemical data table failed, and \
provide structured hints so the parser can retry successfully.

## Common failure modes you should look for:
1. WRONG HEADER ROW: The auto-detector picked a title row instead of the actual \
column headers. Look for the row containing element symbols (Fe, Cu, Zn, Ag, etc.) \
or column labels (Sample, Mineral, Method, etc.).
2. TRANSPOSED TABLE: Elements are in rows (not columns) and samples are in columns. \
The element labels may not be in column 0 — check columns 0-2.
3. UNDETECTED SAMPLE ID: The sample ID column has an unusual name the parser doesn't \
recognize (e.g. "Dot ID", "Analysis Point", "Inclusion No.").
4. WRONG UNIT: Values look like weight percent (0.01-100 range for major elements \
like Fe, Si, Ca) but parser assumed ppm, or vice versa.
5. MULTI-ROW HEADERS: Element symbols in one row, units or metadata labels in the \
next row.
6. NO ELEMENT COLUMNS: The file contains non-geochemical data (isotope ratios, \
spectra, etc.) that the parser cannot handle.

## Your output
Return a single JSON object with parsing hints. Only fill in fields where you \
believe auto-detection went wrong. Use null for fields that should use defaults.

JSON schema:
{
  "header_row": <int or null>,           // 0-indexed row number with element column headers
  "is_transposed": <bool or null>,       // true if elements are in rows, samples in columns
  "sample_id_col": <string or null>,     // exact column name for sample identifiers
  "unit": <"ppm" | "wt%" | "mixed" | null>,  // unit of element concentrations
  "element_label_col": <int or null>,    // for transposed tables: column index with element names
  "skip_sheets": [<string>, ...] or null, // sheet names to skip entirely
  "target_sheets": [<string>, ...] or null, // only process these sheets
  "notes_from_llm": "<one-sentence diagnosis>"
}
"""


def build_diagnosis_prompt(
    raw_preview: str,
    failure_reasons: list[str],
    table_reader_notes: list[str] | None = None,
    previous_hints: ParsingHints | None = None,
) -> tuple[str, str]:
    """Build (system, user) prompt for LLM table structure diagnosis.

    Args:
        raw_preview: Output of build_raw_table_preview().
        failure_reasons: From QuickQualityResult.failure_reasons.
        table_reader_notes: Notes from the SupplementaryTable, if available.
        previous_hints: Hints from a previous failed attempt (for retry context).

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    user_parts = []

    user_parts.append("## RAW TABLE PREVIEW")
    user_parts.append(raw_preview)

    user_parts.append("\n## PARSING FAILURE")
    for reason in failure_reasons:
        user_parts.append(f"- {reason}")

    if table_reader_notes:
        user_parts.append("\n## TABLE READER NOTES (what auto-detection found)")
        for note in table_reader_notes[:10]:  # cap to avoid huge prompts
            user_parts.append(f"- {note}")

    if previous_hints:
        user_parts.append("\n## PREVIOUS ATTEMPT (these hints did NOT fix the problem)")
        prev_dict = {k: v for k, v in previous_hints.__dict__.items()
                     if v is not None and v != ""}
        user_parts.append(json.dumps(prev_dict, indent=2))
        user_parts.append("Please try a DIFFERENT diagnosis. The above hints were applied "
                          "but extraction still failed.")

    user_parts.append("\n## YOUR TASK")
    user_parts.append("Analyze the raw table above and return a JSON object with "
                      "parsing hints. Only set fields where auto-detection went wrong. "
                      "Use null for fields that should use defaults.")

    return _DIAGNOSIS_SYSTEM_PROMPT, "\n".join(user_parts)


def _parse_hints(llm_response: dict | list) -> ParsingHints:
    """Parse LLM JSON response into a ParsingHints object.

    Validates fields against known ParsingHints fields.
    Unknown fields are silently dropped.
    """
    if isinstance(llm_response, list):
        # LLM sometimes wraps in a list
        llm_response = llm_response[0] if llm_response else {}
    if not isinstance(llm_response, dict):
        return ParsingHints(notes_from_llm="LLM returned non-dict response")

    known_fields = set(ParsingHints.__dataclass_fields__.keys())
    filtered = {}
    for k, v in llm_response.items():
        if k in known_fields and v is not None:
            filtered[k] = v

    # Type validation
    if "header_row" in filtered:
        try:
            filtered["header_row"] = int(filtered["header_row"])
        except (TypeError, ValueError):
            del filtered["header_row"]

    if "is_transposed" in filtered:
        if not isinstance(filtered["is_transposed"], bool):
            del filtered["is_transposed"]

    if "element_label_col" in filtered:
        try:
            filtered["element_label_col"] = int(filtered["element_label_col"])
        except (TypeError, ValueError):
            del filtered["element_label_col"]

    if "unit" in filtered:
        if filtered["unit"] not in ("ppm", "wt%", "mixed"):
            del filtered["unit"]

    if "skip_sheets" in filtered:
        if not isinstance(filtered["skip_sheets"], list):
            del filtered["skip_sheets"]

    if "target_sheets" in filtered:
        if not isinstance(filtered["target_sheets"], list):
            del filtered["target_sheets"]

    return ParsingHints(**filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Main correction loop
# ──────────────────────────────────────────────────────────────────────────────

def correction_loop(
    client,  # LLMClient — avoid circular import
    paths: list[Path],
    initial_supp,  # Optional[SupplementaryTable]
    initial_quality: QuickQualityResult,
    this_paper_deposit: Optional[str] = None,
    max_attempts: int = 2,
    quality_threshold: float = 30.0,
    paper_id: str = "",
) -> tuple:
    """Run the agentic self-correction loop.

    Args:
        client: LLM client for diagnosis calls.
        paths: Supplementary file paths.
        initial_supp: SupplementaryTable from the first (failed) attempt, or None.
        initial_quality: QuickQualityResult from initial extraction.
        this_paper_deposit: Deposit name filter.
        max_attempts: Max LLM diagnosis attempts (default 2).
        quality_threshold: Quality score that counts as "good enough" (0-100).
        paper_id: For logging/metrics.

    Returns:
        (best_supp, CorrectionMetrics) — best_supp may be initial_supp if
        correction didn't improve things.
    """
    from .table_reader import read_multiple_supplementary

    metrics = CorrectionMetrics(
        paper_id=paper_id,
        initial_row_count=initial_quality.row_count,
        initial_element_count=initial_quality.element_count,
        initial_quality_score=initial_quality.quality_score,
        correction_needed=True,
    )

    best_supp = initial_supp
    best_quality = initial_quality
    prev_hints: Optional[ParsingHints] = None

    # Collect notes from initial attempt for context
    initial_notes = initial_supp.notes if initial_supp else []

    logger.info("[%s] Self-correction triggered: %s", paper_id, initial_quality.failure_reasons)

    for attempt_num in range(1, max_attempts + 1):
        attempt = CorrectionAttempt(attempt_number=attempt_num)
        start = time.time()

        try:
            # 1. Build raw preview
            raw_preview = build_raw_table_preview(paths)

            # 2. Build diagnosis prompt
            system, user = build_diagnosis_prompt(
                raw_preview=raw_preview,
                failure_reasons=(best_quality.failure_reasons if best_quality else
                                 initial_quality.failure_reasons),
                table_reader_notes=initial_notes,
                previous_hints=prev_hints,
            )

            # 3. Call LLM for diagnosis
            logger.info("[%s] Correction attempt %d: calling LLM for diagnosis",
                        paper_id, attempt_num)
            llm_response = client.complete_json(system=system, user=user, max_tokens=1024)
            hints = _parse_hints(llm_response)
            attempt.hints = hints
            attempt.diagnosis_summary = hints.notes_from_llm
            prev_hints = hints

            logger.info("[%s] LLM diagnosis: %s", paper_id, hints.notes_from_llm)

            # 4. Retry extraction with hints
            new_supp = read_multiple_supplementary(
                paths,
                this_paper_deposit=this_paper_deposit,
                hints=hints,
            )

            # 5. Check quality
            new_quality = quick_quality_check(new_supp)
            attempt.row_count = new_quality.row_count
            attempt.element_count = new_quality.element_count
            attempt.quality_score = new_quality.quality_score

            logger.info("[%s] Attempt %d result: %d rows, %d elements, quality=%.0f%%",
                        paper_id, attempt_num,
                        new_quality.row_count, new_quality.element_count,
                        new_quality.quality_score)

            # 6. Keep best result
            if new_quality.quality_score > best_quality.quality_score:
                best_supp = new_supp
                best_quality = new_quality

            # 7. Early exit conditions
            if new_quality.passed and new_quality.quality_score >= quality_threshold:
                attempt.duration_sec = time.time() - start
                metrics.attempts.append(attempt)
                logger.info("[%s] Correction succeeded on attempt %d",
                            paper_id, attempt_num)
                break

            # No improvement — stop trying
            if (new_quality.quality_score <= initial_quality.quality_score
                    and attempt_num > 1):
                attempt.duration_sec = time.time() - start
                metrics.attempts.append(attempt)
                logger.info("[%s] No improvement on attempt %d — stopping",
                            paper_id, attempt_num)
                break

        except Exception as exc:
            attempt.diagnosis_summary = f"Error: {exc}"
            logger.warning("[%s] Correction attempt %d failed: %s",
                           paper_id, attempt_num, exc)

        attempt.duration_sec = time.time() - start
        metrics.attempts.append(attempt)

    # Final metrics
    metrics.final_row_count = best_quality.row_count
    metrics.final_element_count = best_quality.element_count
    metrics.final_quality_score = best_quality.quality_score
    metrics.correction_succeeded = (
        best_quality.quality_score > initial_quality.quality_score
        and best_quality.passed
    )

    if metrics.correction_succeeded:
        logger.info("[%s] Self-correction improved quality: %.0f%% -> %.0f%% "
                    "(%d -> %d rows)",
                    paper_id,
                    initial_quality.quality_score, best_quality.quality_score,
                    initial_quality.row_count, best_quality.row_count)
    else:
        logger.info("[%s] Self-correction did not improve results", paper_id)

    return best_supp, metrics


# ──────────────────────────────────────────────────────────────────────────────
# PDF extraction self-correction
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PDFExtractionHints:
    """Structured hints from LLM diagnosis for PDF table extraction retry."""
    min_element_cols: int = 3
    try_transposed: bool = False
    target_pages: Optional[list[int]] = None  # 0-indexed page numbers
    unit_override: Optional[str] = None       # "ppm" | "wt%"
    header_row_offset: int = 0                # skip N rows before header
    notes: str = ""


_PDF_DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert geochemistry PDF table extraction specialist. Your job is to \
diagnose why automatic extraction of geochemical data tables from a PDF failed \
or returned too few samples, and provide structured hints for retry.

You are given:
1. Text snippets from the PDF pages that appear to contain data tables
2. What the auto-extraction found (or failed to find)
3. What the paper's analytical methods and expected sample count are

## Common PDF table extraction failures:
1. GARBLED COLUMNS: PDF table reconstruction created misaligned columns
2. MERGED HEADERS: Multi-row headers where element symbols span multiple lines
3. IMAGE-ONLY TABLES: Tables are embedded as images, not text (needs vision)
4. TRANSPOSED LAYOUT: Elements in rows, samples in columns
5. FEW ELEMENT COLS: Table has only 2-3 major elements, below the min threshold
6. WRONG PAGES: Data tables are on unexpected pages
7. UNIT CONFUSION: Values in wt% but auto-detection assumed ppm

## Your output
Return a single JSON object with extraction hints:
{
  "min_element_cols": <int, default 3>,
  "try_transposed": <bool, default false>,
  "target_pages": <list of 0-indexed page numbers, or null>,
  "unit_override": <"ppm" | "wt%" | null>,
  "header_row_offset": <int, rows to skip before header, default 0>,
  "notes": "<one-sentence diagnosis>"
}
"""


def build_pdf_diagnosis_prompt(
    page_texts: list[str],
    backend_table_texts: list[str],
    extraction_summary: str,
    expected_samples: Optional[int],
    expected_elements: Optional[int],
    analytical_methods: list[str],
) -> tuple[str, str]:
    """Build (system, user) prompt for PDF extraction diagnosis."""
    user_parts = []

    if backend_table_texts:
        user_parts.append("## EXTRACTED TABLE TEXT (from PDF backends)")
        for i, tt in enumerate(backend_table_texts[:5]):  # limit to 5 tables
            truncated = tt[:2000] if len(tt) > 2000 else tt
            user_parts.append(f"\n### Table {i+1}\n{truncated}")

    if page_texts:
        user_parts.append("\n## RAW PAGE TEXT (data-dense pages)")
        for i, pt in enumerate(page_texts[:3]):
            truncated = pt[:2000] if len(pt) > 2000 else pt
            user_parts.append(f"\n### Page {i+1}\n{truncated}")

    user_parts.append(f"\n## EXTRACTION RESULT\n{extraction_summary}")

    if expected_samples:
        user_parts.append(f"\nExpected ~{expected_samples} samples from this paper")
    if expected_elements:
        user_parts.append(f"Expected ~{expected_elements} element columns")
    if analytical_methods:
        user_parts.append(f"Analytical methods: {', '.join(analytical_methods)}")

    user_parts.append("\n## YOUR TASK")
    user_parts.append("Diagnose why extraction failed or returned too few samples. "
                      "Return JSON hints for retry.")

    return _PDF_DIAGNOSIS_SYSTEM_PROMPT, "\n".join(user_parts)


def _parse_pdf_hints(llm_response: dict | list) -> PDFExtractionHints:
    """Parse LLM response into PDFExtractionHints."""
    if isinstance(llm_response, list):
        llm_response = llm_response[0] if llm_response else {}
    if not isinstance(llm_response, dict):
        return PDFExtractionHints(notes="LLM returned non-dict response")

    hints = PDFExtractionHints()
    if "min_element_cols" in llm_response:
        try:
            hints.min_element_cols = max(1, int(llm_response["min_element_cols"]))
        except (TypeError, ValueError):
            pass
    if "try_transposed" in llm_response and isinstance(llm_response["try_transposed"], bool):
        hints.try_transposed = llm_response["try_transposed"]
    if "target_pages" in llm_response and isinstance(llm_response["target_pages"], list):
        hints.target_pages = [int(p) for p in llm_response["target_pages"]
                              if isinstance(p, (int, float))]
    if "unit_override" in llm_response and llm_response["unit_override"] in ("ppm", "wt%"):
        hints.unit_override = llm_response["unit_override"]
    if "header_row_offset" in llm_response:
        try:
            hints.header_row_offset = int(llm_response["header_row_offset"])
        except (TypeError, ValueError):
            pass
    if "notes" in llm_response and isinstance(llm_response["notes"], str):
        hints.notes = llm_response["notes"]
    return hints


def pdf_correction_loop(
    client,
    pdf_path,
    pdf_content,
    metadata,
    initial_samples: list,
    backend_table_texts: list[str],
    intelligence,
    errors: list[str],
    extract_fn,  # callable(hints: PDFExtractionHints) -> list[SampleRow]
    max_attempts: int = 1,
    paper_id: str = "",
) -> list:
    """Run agentic self-correction for PDF table extraction.

    When direct extraction returns 0 or far too few samples, this asks the
    LLM to diagnose the PDF table structure and retry with adjusted hints.

    Args:
        client: LLM client.
        pdf_path: Path to the PDF.
        pdf_content: Parsed PDF content.
        metadata: Extracted paper metadata.
        initial_samples: Samples from the initial attempt.
        backend_table_texts: Cached text from multi-backend extraction.
        intelligence: PaperIntelligence with expected counts.
        errors: Error list for logging.
        extract_fn: Callable that takes PDFExtractionHints and returns samples.
        max_attempts: Max LLM diagnosis attempts.
        paper_id: For logging.

    Returns:
        Best sample list (initial or corrected).
    """
    from .pdf_reader import score_pages_for_data

    expected = getattr(intelligence, "expected_sample_count", None)
    n_elements = len(getattr(intelligence, "elements_measured", []))
    methods = getattr(intelligence, "analytical_methods", [])

    extraction_summary = (
        f"Initial extraction found {len(initial_samples)} samples. "
        f"Expected ~{expected or '?'}."
    )

    # Get data-dense page text for diagnosis
    page_texts = []
    if pdf_content and pdf_content.pages:
        try:
            scored = score_pages_for_data(pdf_content)
            for idx, _ in scored[:3]:
                page_texts.append(pdf_content.pages[idx][:2000])
        except Exception:
            pass

    best_samples = initial_samples

    for attempt in range(1, max_attempts + 1):
        try:
            system, user = build_pdf_diagnosis_prompt(
                page_texts=page_texts,
                backend_table_texts=backend_table_texts[:5],
                extraction_summary=extraction_summary,
                expected_samples=expected,
                expected_elements=n_elements,
                analytical_methods=methods,
            )

            logger.info("[%s] PDF correction attempt %d: calling LLM for diagnosis",
                        paper_id, attempt)
            llm_response = client.complete_json(system=system, user=user, max_tokens=1024)
            hints = _parse_pdf_hints(llm_response)
            logger.info("[%s] PDF diagnosis: %s", paper_id, hints.notes)

            new_samples = extract_fn(hints)

            if len(new_samples) > len(best_samples):
                best_samples = new_samples
                logger.info("[%s] PDF correction improved: %d → %d samples",
                            paper_id, len(initial_samples), len(new_samples))
                extraction_summary = (
                    f"Retry {attempt} found {len(new_samples)} samples "
                    f"(improved from {len(initial_samples)}). "
                    f"Expected ~{expected or '?'}."
                )
            else:
                logger.info("[%s] PDF correction attempt %d did not improve", paper_id, attempt)
                break

        except Exception as exc:
            logger.warning("[%s] PDF correction attempt %d failed: %s", paper_id, attempt, exc)
            errors.append(f"PDF correction attempt {attempt} failed: {exc}")
            break

    return best_samples
