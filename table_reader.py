"""
table_reader.py - Read and pre-process supplementary geochemical tables.

Handles Excel (.xlsx, .xls) and CSV files. Detects:
  - Header row location
  - Which rows are sample data vs summary statistics (MEAN, STD, etc.)
  - Which rows belong to "this paper" vs cited references
  - Element column → schema element symbol mapping
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import normalise_element_header, ELEMENT_SYMBOLS
from .agentic_corrector import ParsingHints

# ──────────────────────────────────────────────────────────────────────────────
# Patterns for rows that should be EXCLUDED (summary / comparison data)
# ──────────────────────────────────────────────────────────────────────────────
_SUMMARY_ROW_PATTERNS = re.compile(
    r"^\s*(mean|average|avg|std|s\.d\.?|stdev|standard deviation|"
    r"min|minima|minimum|max|maxima|maximum|median|range|total|sum|"
    r"detection limit|bdl|n\s*=|note:?|notes:?|d\.l\.?|"
    r"concentrations?\s+in|"
    r"\(ppm\)|\(ppb\)|\(wt%?\)|"  # unit-label rows
    r"numbers?\s+after|"  # footnote rows
    r"\*\s*calculated|"  # calculation notes
    r"- not\b|"  # dash-note rows ("- not available")
    r"all\s+values?\s+in\b|"  # "all values in wt%"
    r"sample\s*$|"  # lone "Sample" header leaked through
    r"sp[a-z]\s+n\s*="  # mineral-type group headers ("SpA  n=34")
    r")\b",
    re.IGNORECASE,
)
# Matches summary stats ANYWHERE in the name (e.g., "Py2(n=24)")
_SUMMARY_EMBEDDED_PATTERN = re.compile(
    r"\(n\s*=\s*\d+\)|"            # (n=24) — average of N spots
    r"\bmean\b|\baverage\b|\bmedian\b|\bstd\b|\bstdev\b|"
    r"\bmin\b|\bmax\b|\brange\b",
    re.IGNORECASE,
)

# Reference standard patterns — separate because they need flexible boundaries
# (e.g., "G_NIST610" shouldn't require \b after "nist")
_REFERENCE_STANDARD_PATTERNS = re.compile(
    r"^\s*("
    r"[sg]_?nist\d*|"          # G_NIST610, S_NIST612, NIST610
    r"nist\s*\d+|"             # NIST 610, NIST612
    r"mass-?\d|"               # MASS-1, MASS1
    r"[sg]_?mass[-_]?\d|"      # S_MASS_1, G_MASS-1
    r"[sg]_?ge\d+|"            # S_GE8, G_GE7
    r"ge-?\d+|"                # GE8, GE-7
    r"smr\s*\d+|"              # SMR 610, SMR612
    r"usgs\s*\w+|"             # USGS reference materials
    r"bhvo-?\d|"               # BHVO-2 (basalt reference)
    r"bcr-?\d|"                # BCR-2 (basalt reference)
    r"gse-?\d|"                # GSE-1 (glass reference)
    r"gsd-?\d|"                # GSD-1 (glass reference)
    r"reference\s+standard"    # generic "reference standard"
    r")\s*$",
    re.IGNORECASE,
)


def _is_note_or_description(val: str) -> bool:
    """Check if a sample ID value looks like a note/description rather than
    a legitimate sample identifier."""
    if not val or not val.strip():
        return True
    s = val.strip()
    # Very long text is likely a note/description
    if len(s) > 80:
        return True
    # Contains sentence-like patterns
    if s.endswith(".") and " " in s and len(s) > 30:
        return True
    # Starts with common note markers
    if s.startswith(("Note:", "note:", "*", "†", "‡", "§")):
        return True
    return False

_THIS_PAPER_ALIASES = {
    "this paper", "this study", "this work", "present study", "present work",
    "authors", "own data",
}


@dataclass
class SupplementaryTable:
    """Cleaned and annotated supplementary table."""
    raw_df: pd.DataFrame                        # Full original DataFrame
    data_df: pd.DataFrame                       # Filtered: sample rows only
    element_col_map: dict[str, str]             = field(default_factory=dict)
    # element_col_map: raw_column_name → schema element symbol  (e.g. "Ag" → "ag")
    sample_id_col: Optional[str]                = None
    deposit_col: Optional[str]                  = None
    reference_col: Optional[str]               = None
    mineral_col: Optional[str]                  = None
    method_col: Optional[str]                   = None
    zone_col: Optional[str]                     = None
    inferred_method: Optional[str]              = None  # from filename
    inferred_pub_year: Optional[str]            = None  # from filename
    unit: str                                   = "ppm"
    notes: list[str]                            = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.data_df)

    def to_element_records(self) -> list[dict]:
        """Return one dict per sample row with schema element keys.

        Each dict has: sample_name, sample_local_id, deposit_name, mineral,
        analytical_method, + element_symbol → float/None.

        When duplicate sample_names exist, auto-generates unique
        sample_local_id values (e.g. "YK94-17-1", "YK94-17-2").
        If a "Spot" / "Point" / "No." column exists, uses it to compose
        natural IDs (e.g. "BBJ-1-spot1") instead of sequential counters.
        """
        records = []
        # Track duplicate sample names to generate unique local IDs
        name_counts: dict[str, int] = {}

        # Detect a spot/point numbering column for natural ID composition
        spot_num_col = _detect_spot_number_col(self.data_df, self.sample_id_col)

        for _, row in self.data_df.iterrows():
            rec: dict = {}
            sample_name = None
            if self.sample_id_col:
                sample_name = _safe_str(row.get(self.sample_id_col))
            # Fallback: if sample_id_col gave None, try "sample_name" column directly
            if sample_name is None and "sample_name" in self.data_df.columns:
                sample_name = _safe_str(row.get("sample_name"))
            rec["sample_name"] = sample_name

            if self.deposit_col:
                rec["deposit_name"] = _safe_str(row.get(self.deposit_col))
            elif "deposit_name" in self.data_df.columns:
                rec["deposit_name"] = _safe_str(row.get("deposit_name"))
            if self.mineral_col:
                rec["mineral"] = _safe_str(row.get(self.mineral_col))
            elif "mineral" in self.data_df.columns:
                rec["mineral"] = _safe_str(row.get("mineral"))
            if self.method_col:
                rec["analytical_method"] = _normalize_method(_safe_str(row.get(self.method_col)))
            elif "analytical_method" in self.data_df.columns:
                rec["analytical_method"] = _normalize_method(_safe_str(row.get("analytical_method")))
            elif self.inferred_method:
                rec["analytical_method"] = _normalize_method(self.inferred_method)
            if self.zone_col:
                rec["texture"] = _safe_str(row.get(self.zone_col))
            elif "texture" in self.data_df.columns:
                rec["texture"] = _safe_str(row.get("texture"))
            if self.inferred_pub_year:
                rec["publication_date"] = self.inferred_pub_year
            elif "publication_date" in self.data_df.columns:
                rec["publication_date"] = _safe_str(row.get("publication_date"))
            for raw_col, sym in self.element_col_map.items():
                val = row.get(raw_col)
                rec[f"{sym}_ppm"] = _safe_float(val)

            # Use existing sample_local_id from DataFrame if present
            if "sample_local_id" in self.data_df.columns:
                lid = _safe_str(row.get("sample_local_id"))
                if lid:
                    rec["sample_local_id"] = lid

            # Auto-generate unique sample_local_id for duplicate sample names
            if "sample_local_id" not in rec and sample_name:
                if spot_num_col:
                    spot_val = _safe_str(row.get(spot_num_col))
                    if spot_val:
                        # Strip trailing ".0" from Excel float rendering
                        if spot_val.endswith(".0"):
                            spot_val = spot_val[:-2]
                        prefix = spot_num_col.lower().strip()
                        rec["sample_local_id"] = f"{sample_name}-{prefix}{spot_val}"
                    else:
                        name_counts[sample_name] = name_counts.get(sample_name, 0) + 1
                        rec["sample_local_id"] = f"{sample_name}-{name_counts[sample_name]}"
                else:
                    name_counts[sample_name] = name_counts.get(sample_name, 0) + 1
                    rec["sample_local_id"] = f"{sample_name}-{name_counts[sample_name]}"

            records.append(rec)

        # If all names were unique and no pre-existing local IDs, remove auto-generated
        has_real_lids = "sample_local_id" in self.data_df.columns
        if not has_real_lids and name_counts and max(name_counts.values()) == 1:
            # Only remove if we didn't use spot_num_col (spot IDs are always meaningful)
            if not spot_num_col:
                for rec in records:
                    rec.pop("sample_local_id", None)

        return records

    def table_as_text(self, max_rows: int = 120) -> str:
        """Render the data_df (plus header) as a tab-separated text block.
        Useful for passing to an LLM when the Python mapping is insufficient.
        """
        df_head = self.data_df.head(max_rows)
        lines = ["\t".join(str(c) for c in df_head.columns)]
        for _, row in df_head.iterrows():
            lines.append("\t".join("" if pd.isna(v) else str(v) for v in row))
        return "\n".join(lines)


def read_multiple_supplementary(
    paths: list[str | Path],
    this_paper_deposit: Optional[str] = None,
    hints: Optional[ParsingHints] = None,
) -> SupplementaryTable:
    """Read one or more supplementary files and merge into a single SupplementaryTable.

    Files are merged on sample_name (outer join, first non-null value wins).
    If sample names don't overlap, rows are concatenated instead.
    Useful when a paper distributes major and trace elements across separate files.

    Args:
        paths: One or more paths to .xlsx, .xls, or .csv files.
        this_paper_deposit: Deposit name filter passed to each file's reader.

    Returns:
        A single merged SupplementaryTable.
    """
    if not paths:
        raise ValueError("At least one supplementary path is required.")
    if len(paths) == 1:
        return read_supplementary(paths[0], this_paper_deposit=this_paper_deposit, hints=hints)

    from .knowledge_base import detect_method_from_filename

    supps = []
    for p in paths:
        try:
            s = read_supplementary(p, this_paper_deposit=this_paper_deposit, hints=hints)
        except (ValueError, Exception) as exc:
            # Skip files that can't be parsed (e.g. isotope-only, no element columns)
            supps_notes = [f"File ({Path(p).name}): skipped — {exc}"]
            continue
        # Detect analytical method from filename (e.g., EMPA.xlsx, LAICPMS.xlsx)
        if not s.method_col:
            inferred = detect_method_from_filename(Path(p).name)
            if inferred:
                s.inferred_method = inferred
        supps.append(s)

    if not supps:
        raise ValueError(f"No usable supplementary files among {[Path(p).name for p in paths]}")

    all_notes: list[str] = []
    for i, s in enumerate(supps):
        method_info = f" [method: {s.inferred_method}]" if s.inferred_method else ""
        all_notes.append(f"File {i+1} ({Path(paths[i]).name}): {s.n_samples} rows{method_info}")
        all_notes.extend(s.notes)

    # Build one flat dict per sample from all files
    # Key: sample_name (or row index if no sample name detected)
    merged: dict[str, dict] = {}  # sample_name → merged element dict
    ordered_keys: list[str] = []  # preserve order of first appearance

    for supp in supps:
        for rec in supp.to_element_records():
            # Use the most specific available ID as merge key
            key = str(
                rec.get("sample_local_id")
                or rec.get("sample_name")
                or len(merged)
            )
            if key not in merged:
                merged[key] = {}
                ordered_keys.append(key)
            # First non-null value per field wins
            for k, v in rec.items():
                if k not in merged[key] or merged[key][k] is None:
                    merged[key][k] = v

    rows = [merged[k] for k in ordered_keys]
    data_df = pd.DataFrame(rows)

    # Build element_col_map: column name → element symbol (already canonical)
    all_syms = set()
    for supp in supps:
        all_syms.update(supp.element_col_map.values())
    element_col_map = {f"{sym}_ppm": sym for sym in sorted(all_syms) if f"{sym}_ppm" in data_df.columns}

    all_notes.append(f"Merged: {len(data_df)} rows, {len(element_col_map)} element columns")

    return SupplementaryTable(
        raw_df=supps[0].raw_df,        # use first file's raw_df for reference
        data_df=data_df,
        element_col_map=element_col_map,
        sample_id_col="sample_local_id" if "sample_local_id" in data_df.columns else "sample_name",
        deposit_col="deposit_name" if "deposit_name" in data_df.columns else None,
        reference_col=None,
        unit="ppm",
        notes=all_notes,
    )


def read_supplementary(
    path: str | Path,
    sheet_name: int | str | None = None,
    this_paper_deposit: Optional[str] = None,
    hints: Optional[ParsingHints] = None,
) -> SupplementaryTable:
    """Load an Excel or CSV supplementary file and return a SupplementaryTable.

    For Excel files with multiple sheets, all sheets are read and merged on
    sample_name automatically (first non-null value per field wins).
    Pass an explicit sheet_name (index or string) to read only one sheet.

    Args:
        path: Path to .xlsx, .xls, or .csv file.
        sheet_name: Sheet index or name. None (default) = auto — reads one
            sheet if only one exists, otherwise reads and merges all sheets.
        this_paper_deposit: If provided, filter rows where deposit column
            matches this name (case-insensitive) to isolate "this paper" data.

    Returns:
        SupplementaryTable with cleaned data and element column mapping.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in (".xlsx", ".xls", ".csv"):
        raise ValueError(f"Unsupported file type: {suffix}. Use .xlsx, .xls, or .csv")

    # Detect method from filename for all files
    from .knowledge_base import detect_method_from_filename
    _filename_method = detect_method_from_filename(path.name)

    # Detect publication year from filename (e.g., "2018_Yuan_etal.xlsx" → "2018")
    _filename_year = None
    _year_match = re.match(r"(\d{4})", path.stem)
    if _year_match:
        _filename_year = _year_match.group(1)

    # CSV — no concept of sheets
    if suffix == ".csv":
        csv_df, csv_pre_header, csv_pre_rows = _load_csv(path, hints=hints)
        result = _read_single_sheet(path, raw_df=csv_df,
                                  this_paper_deposit=this_paper_deposit,
                                  sheet_label="(csv)",
                                  pre_header_text=csv_pre_header,
                                  pre_header_rows=csv_pre_rows,
                                  hints=hints)
        if not result.method_col and _filename_method and not result.inferred_method:
            result.inferred_method = _filename_method
        if _filename_year and not result.inferred_pub_year:
            result.inferred_pub_year = _filename_year
        return result

    # Excel — discover available sheets
    if suffix == ".xls":
        # Legacy .xls requires xlrd (openpyxl only supports .xlsx)
        import xlrd
        wb = xlrd.open_workbook(str(path))
        all_sheet_names = wb.sheet_names()
    else:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        all_sheet_names = wb.sheetnames
        wb.close()

    # Explicit sheet requested → read just that one
    if sheet_name is not None:
        raw_df, pre_header_text, pre_header_rows = _load_excel(path, sheet_name, hints=hints)
        return _read_single_sheet(path, raw_df=raw_df,
                                  this_paper_deposit=this_paper_deposit,
                                  sheet_label=str(sheet_name),
                                  pre_header_text=pre_header_text,
                                  pre_header_rows=pre_header_rows,
                                  hints=hints)

    # Single sheet → no merging needed
    if len(all_sheet_names) == 1:
        raw_df, pre_header_text, pre_header_rows = _load_excel(path, 0, hints=hints)
        result = _read_single_sheet(path, raw_df=raw_df,
                                  this_paper_deposit=this_paper_deposit,
                                  sheet_label=all_sheet_names[0],
                                  pre_header_text=pre_header_text,
                                  pre_header_rows=pre_header_rows,
                                  hints=hints)
        if not result.method_col and _filename_method and not result.inferred_method:
            result.inferred_method = _filename_method
        if _filename_year and not result.inferred_pub_year:
            result.inferred_pub_year = _filename_year
        return result

    # Multiple sheets → read each and merge
    result = _read_excel_all_sheets(path, all_sheet_names, this_paper_deposit, hints=hints)
    if not result.method_col and _filename_method and not result.inferred_method:
        result.inferred_method = _filename_method
    if _filename_year and not result.inferred_pub_year:
        result.inferred_pub_year = _filename_year
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Direct DataFrame processing (for PDF-extracted tables)
# ──────────────────────────────────────────────────────────────────────────────

def read_pdf_table(
    df: pd.DataFrame,
    min_element_cols: int = 3,
    label: str = "pdf_table",
    convert_units: bool = True,
) -> Optional[SupplementaryTable]:
    """Process a raw DataFrame from PDF table extraction as if it were a
    supplementary table.  Reuses the same column mapping, unit detection,
    and row-filtering logic that works on Excel/CSV files.

    Unit conversion is enabled by default — all values normalised to ppm.
    (wt% × 10000, ppb ÷ 1000)

    Returns None if the DataFrame does not contain enough recognised element
    columns (< *min_element_cols*).
    """
    if df is None or df.empty:
        return None

    notes: list[str] = [f"PDF table '{label}': {df.shape[0]} rows × {df.shape[1]} cols"]

    # Deduplicate column names (Docling sometimes produces duplicate "Total" etc.)
    cols = list(df.columns)
    seen: dict[str, int] = {}
    new_cols = []
    for c in cols:
        c_str = str(c)
        if c_str in seen:
            seen[c_str] += 1
            new_cols.append(f"{c_str}_{seen[c_str]}")
        else:
            seen[c_str] = 0
            new_cols.append(c_str)
    if new_cols != [str(c) for c in cols]:
        df = df.copy()
        df.columns = new_cols

    # Try promoting first row to header if columns are numeric indices
    raw_df = _maybe_promote_header(df)

    # Detect embedded unit row in first few data rows
    _pdf_pre_header_rows: list[list[str]] = []
    _pdf_unit_detected = None
    for row_idx in range(min(3, len(raw_df))):
        row_vals = [str(v).strip().lower() for v in raw_df.iloc[row_idx] if pd.notna(v)]
        unit_count = sum(1 for v in row_vals if v in ("wt%", "wt.%", "wt %", "ppm", "ppb", "%"))
        if unit_count >= 2 and unit_count >= len(row_vals) * 0.3:
            unit_row_vals = [str(v).strip() if pd.notna(v) else "" for v in raw_df.iloc[row_idx]]
            _pdf_pre_header_rows.append(unit_row_vals)
            has_wt = any(v.lower() in ("wt%", "wt.%", "wt %", "%") for v in unit_row_vals)
            has_ppm = any(v.lower() in ("ppm", "ppb") for v in unit_row_vals)
            if has_wt and has_ppm:
                _pdf_unit_detected = "mixed"
            elif has_wt:
                _pdf_unit_detected = "wt%"
            else:
                _pdf_unit_detected = "ppm"
            raw_df = raw_df.drop(raw_df.index[row_idx]).reset_index(drop=True)
            notes.append(f"Detected embedded unit row: wt%={'yes' if has_wt else 'no'}, ppm={'yes' if has_ppm else 'no'}")
            break

    element_col_map = _map_element_columns(raw_df)
    if len(element_col_map) < min_element_cols:
        return None

    notes.append(f"Detected {len(element_col_map)} element columns: {list(element_col_map.keys())}")

    sample_id_col = _detect_sample_id_col(raw_df)
    mineral_col = _detect_mineral_col(raw_df)
    method_col = _detect_method_col(raw_df)
    deposit_col = _detect_deposit_col(raw_df)
    zone_col = _detect_zone_col(raw_df)

    detected_unit = _pdf_unit_detected if _pdf_unit_detected else _detect_unit_from_headers(raw_df)

    # Heuristic: if header detection didn't find wt%, check for "Total"
    # column with values ~100 (strong EPMA/wt% indicator)
    if detected_unit == "ppm" and element_col_map:
        total_col = None
        for col in raw_df.columns:
            if str(col).strip().lower() == "total":
                total_col = col
                break
        if total_col is not None:
            # Handle duplicate column names — get first matching column as Series
            col_data = raw_df[total_col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            total_vals = pd.to_numeric(col_data, errors="coerce").dropna()
            if len(total_vals) > 0:
                mean_total = total_vals.mean()
                if 85 < mean_total < 110:
                    detected_unit = "wt%"
                    notes.append(f"Detected wt% from Total column (mean={mean_total:.1f})")

    # Filter to sample rows (remove summary/header rows)
    data_df = _filter_sample_rows(
        raw_df,
        sample_id_col=sample_id_col,
        reference_col=None,
        this_paper_deposit=None,
        deposit_col=deposit_col,
    )
    notes.append(f"Filtered from {len(raw_df)} → {len(data_df)} sample rows")

    # Unit conversion — normalise all values to ppm.
    if convert_units and element_col_map:
        if detected_unit in ("wt%", "mixed"):
            if detected_unit == "mixed":
                wt_cols = _detect_wt_pct_columns(raw_df, element_col_map)
            else:
                # Don't blindly convert all columns — use value-range heuristic
                wt_cols = _detect_wt_pct_columns(raw_df, element_col_map)
                if not wt_cols:
                    wt_cols = _detect_wt_pct_by_value_range(data_df, element_col_map)
            if wt_cols:
                data_df = _convert_wt_pct_to_ppm(data_df, wt_cols=wt_cols)
                notes.append(f"Converted {len(wt_cols)}/{len(element_col_map)} wt% columns to ppm (×10000)")
        # Also handle ppb
        ppb_cols = _detect_ppb_columns(data_df, element_col_map)
        if ppb_cols:
            data_df = _convert_ppb_to_ppm(data_df, ppb_cols=ppb_cols)
            notes.append(f"Converted {len(ppb_cols)} ppb columns to ppm (÷1000)")

    if data_df.empty:
        return None

    return SupplementaryTable(
        raw_df=raw_df,
        data_df=data_df,
        element_col_map=element_col_map,
        sample_id_col=sample_id_col,
        deposit_col=deposit_col,
        mineral_col=mineral_col,
        method_col=method_col,
        zone_col=zone_col,
        unit="ppm",
        notes=notes,
    )


def _maybe_promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """If DataFrame columns are integer indices (0, 1, 2, ...), promote
    the first row to column headers — common with Camelot extraction."""
    # Check if columns are default integer range
    if list(df.columns) == list(range(len(df.columns))):
        if len(df) < 2:
            return df
        new_cols = [str(c).strip() if pd.notna(c) else f"col_{i}"
                    for i, c in enumerate(df.iloc[0])]
        new_df = df.iloc[1:].copy()
        new_df.columns = new_cols
        return new_df.reset_index(drop=True)

    # Also check if columns look like "0", "1", "2" (string integers)
    try:
        if all(str(c).isdigit() for c in df.columns):
            if len(df) < 2:
                return df
            new_cols = [str(c).strip() if pd.notna(c) else f"col_{i}"
                        for i, c in enumerate(df.iloc[0])]
            new_df = df.iloc[1:].copy()
            new_df.columns = new_cols
            return new_df.reset_index(drop=True)
    except Exception:
        pass

    return df


def _read_single_sheet(
    path: Path,
    raw_df: "pd.DataFrame",
    this_paper_deposit: Optional[str],
    sheet_label: str,
    pre_header_text: str = "",
    pre_header_rows: list[list[str]] | None = None,
    hints: Optional[ParsingHints] = None,
) -> SupplementaryTable:
    """Build a SupplementaryTable from one already-loaded DataFrame."""
    notes: list[str] = [f"Sheet '{sheet_label}': {len(raw_df)} raw rows"]

    # Detect and pivot transposed tables (rows=elements, cols=samples)
    is_transposed = (hints.is_transposed if hints and hints.is_transposed is not None
                     else _is_transposed(raw_df))
    if is_transposed:
        raw_df = _pivot_transposed(raw_df)
        notes.append(f"Detected transposed table — pivoted to {len(raw_df)} rows × {len(raw_df.columns)} cols")

    # Use hint overrides for column detection where available
    if hints and hints.sample_id_col:
        # Try to match the hint against actual columns (case-insensitive)
        sample_id_col = None
        for col in raw_df.columns:
            if col.lower().strip() == hints.sample_id_col.lower().strip():
                sample_id_col = col
                break
        if sample_id_col is None:
            sample_id_col = _detect_sample_id_col(raw_df)
            notes.append(f"Hint sample_id_col='{hints.sample_id_col}' not found, using auto-detection")
    else:
        sample_id_col = _detect_sample_id_col(raw_df)

    deposit_col     = _detect_deposit_col(raw_df)
    reference_col   = _detect_reference_col(raw_df)
    mineral_col     = _detect_mineral_col(raw_df)
    method_col      = _detect_method_col(raw_df)
    zone_col        = _detect_zone_col(raw_df)
    element_col_map = _map_element_columns(raw_df)

    notes.append(f"Detected {len(element_col_map)} element columns: {list(element_col_map.keys())}")
    if mineral_col:
        notes.append(f"Detected mineral column: '{mineral_col}'")
    if method_col:
        notes.append(f"Detected method column: '{method_col}'")

    # Detect embedded unit row in first few data rows (e.g., "wt%", "ppm", ...)
    # Common pattern: header row has element symbols, next row has units.
    # If found, extract it as a pre_header_row for per-column unit detection
    # and remove it from the data.
    if pre_header_rows is None:
        pre_header_rows = []
    _embedded_unit_row_found = False
    for row_idx in range(min(3, len(raw_df))):
        row_vals = [str(v).strip().lower() for v in raw_df.iloc[row_idx] if pd.notna(v)]
        unit_count = sum(1 for v in row_vals if v in ("wt%", "wt.%", "wt %", "ppm", "ppb", "%"))
        if unit_count >= 2 and unit_count >= len(row_vals) * 0.3:
            # This looks like a unit row — extract and remove
            unit_row_vals = [str(v).strip() if pd.notna(v) else "" for v in raw_df.iloc[row_idx]]
            pre_header_rows.append(unit_row_vals)
            raw_df = raw_df.drop(raw_df.index[row_idx]).reset_index(drop=True)
            _embedded_unit_row_found = True
            # Determine if mixed units from the unit row
            has_wt = any(v.lower() in ("wt%", "wt.%", "wt %", "%") for v in unit_row_vals)
            has_ppm = any(v.lower() in ("ppm", "ppb") for v in unit_row_vals)
            notes.append(f"Detected embedded unit row at data row {row_idx}: "
                         f"wt%={'yes' if has_wt else 'no'}, ppm={'yes' if has_ppm else 'no'}")
            break

    # Detect unit from headers (wt% vs ppm) — hint overrides auto-detection
    if hints and hints.unit:
        detected_unit = hints.unit
        notes.append(f"Using hint unit='{hints.unit}'")
    elif _embedded_unit_row_found:
        # Use the embedded unit row to determine overall unit type
        unit_row = pre_header_rows[-1]
        wt_in_row = any(v.lower() in ("wt%", "wt.%", "wt %", "%") for v in unit_row)
        ppm_in_row = any(v.lower() in ("ppm", "ppb") for v in unit_row)
        if wt_in_row and ppm_in_row:
            detected_unit = "mixed"
        elif wt_in_row:
            detected_unit = "wt%"
        else:
            detected_unit = "ppm"
    else:
        detected_unit = _detect_unit_from_headers(raw_df, pre_header_text=pre_header_text)

    # Forward-fill sparse sample ID column (grouped/merged-cell format)
    if sample_id_col and sample_id_col in raw_df.columns:
        non_null = raw_df[sample_id_col].notna().sum()
        total = len(raw_df)
        has_element_data = sum(
            1 for _, row in raw_df.head(20).iterrows()
            if any(pd.notna(row.get(c)) for c in element_col_map)
        )
        # Forward-fill if the column is sparse (grouped/merged-cell format)
        # AND most sampled rows have element data (not an empty column)
        if non_null < total * 0.5 and non_null >= 2 and has_element_data >= 10:
            raw_df = raw_df.copy()
            raw_df[sample_id_col] = raw_df[sample_id_col].ffill()
            notes.append(f"Forward-filled sparse '{sample_id_col}' column ({non_null} → {raw_df[sample_id_col].notna().sum()} non-null)")

    data_df = _filter_sample_rows(
        raw_df,
        sample_id_col=sample_id_col,
        reference_col=reference_col,
        this_paper_deposit=this_paper_deposit,
        deposit_col=deposit_col,
    )
    notes.append(f"Filtered from {len(raw_df)} → {len(data_df)} sample rows")

    # Unit conversion — normalise all element values to ppm.
    # The "_ppm" column suffix means the value MUST be in ppm.
    #   wt% → ppm: multiply by 10,000
    #   ppb → ppm: divide by 1,000
    # Per-column detection handles mixed-unit tables correctly.
    if detected_unit in ("wt%", "mixed") and element_col_map:
        wt_cols = _detect_wt_pct_columns(data_df, element_col_map, pre_header_rows)
        if detected_unit == "wt%" and not wt_cols:
            # Header-based detection found nothing specific.
            # Use value-range heuristic: only convert columns whose median
            # is in the wt% range (0.01-100). Columns with median >100
            # are likely already in ppm and must NOT be multiplied by 10000.
            wt_cols = _detect_wt_pct_by_value_range(data_df, element_col_map)
            if wt_cols:
                notes.append(f"Detected {len(wt_cols)} wt% columns by value range (median 0.01-100)")
        if wt_cols:
            data_df = _convert_wt_pct_to_ppm(data_df, wt_cols=wt_cols)
            notes.append(f"Converted {len(wt_cols)}/{len(element_col_map)} wt% columns to ppm (×10000)")
    # Also handle ppb columns
    ppb_cols = _detect_ppb_columns(data_df, element_col_map, pre_header_rows)
    if ppb_cols:
        data_df = _convert_ppb_to_ppm(data_df, ppb_cols=ppb_cols)
        notes.append(f"Converted {len(ppb_cols)} ppb columns to ppm (÷1000)")

    return SupplementaryTable(
        raw_df=raw_df,
        data_df=data_df,
        element_col_map=element_col_map,
        sample_id_col=sample_id_col,
        deposit_col=deposit_col,
        reference_col=reference_col,
        mineral_col=mineral_col,
        method_col=method_col,
        zone_col=zone_col,
        unit="ppm",
        notes=notes,
    )


def _read_excel_all_sheets(
    path: Path,
    sheet_names: list[str],
    this_paper_deposit: Optional[str],
    hints: Optional[ParsingHints] = None,
) -> SupplementaryTable:
    """Read every sheet of an Excel file and merge into one SupplementaryTable.

    Each sheet is treated as an independent dataset — they may have different
    sample sets, different elements, or different analytical methods.
    Rows are merged on sample_name; if the same sample appears in multiple
    sheets, the first non-null value per field wins.
    If sample names don't overlap at all, rows are simply concatenated.
    """
    supps: list[SupplementaryTable] = []
    all_notes: list[str] = []

    # Sheet names that indicate non-data content (detection limits, references, etc.)
    _SKIP_SHEET_PATTERNS = (
        "lod", "detection limit", "reference material", "ref material",
        "standard", "certified", "blank",
    )

    # Apply hint-based sheet filtering
    if hints and hints.target_sheets:
        sheet_names = [n for n in sheet_names if n in hints.target_sheets]
        all_notes.append(f"Hint: targeting sheets {hints.target_sheets}")
    if hints and hints.skip_sheets:
        sheet_names = [n for n in sheet_names if n not in hints.skip_sheets]
        all_notes.append(f"Hint: skipping sheets {hints.skip_sheets}")

    for name in sheet_names:
        try:
            # Skip sheets whose names indicate non-sample-data content
            name_lower = name.strip().lower()
            if any(pat in name_lower for pat in _SKIP_SHEET_PATTERNS):
                all_notes.append(f"Sheet '{name}': non-data sheet — skipped")
                continue
            raw_df, pre_header_text, pre_header_rows = _load_excel(path, name, hints=hints)
            # Skip sheets with no element columns (check after potential pivot)
            is_transposed = (hints.is_transposed if hints and hints.is_transposed is not None
                             else _is_transposed(raw_df))
            check_df = _pivot_transposed(raw_df) if is_transposed else raw_df
            if not _map_element_columns(check_df):
                all_notes.append(f"Sheet '{name}': no element columns — skipped")
                continue
            s = _read_single_sheet(path, raw_df=raw_df,
                                   this_paper_deposit=this_paper_deposit,
                                   sheet_label=name,
                                   pre_header_text=pre_header_text,
                                   pre_header_rows=pre_header_rows,
                                   hints=hints)

            # USGS requirement: infer mineral from sheet name when no mineral
            # column is present (e.g., sheets named "Chalcopyrite", "Sphalerite")
            if not s.mineral_col:
                inferred_mineral = _infer_mineral_from_label(name)
                if inferred_mineral:
                    s.inferred_mineral = inferred_mineral
                    all_notes.append(f"Sheet '{name}': inferred mineral='{inferred_mineral}' from sheet name")

            supps.append(s)
            all_notes.append(f"Sheet '{name}': {s.n_samples} sample rows, "
                             f"{len(s.element_col_map)} elements")
        except Exception as exc:
            all_notes.append(f"Sheet '{name}': read error — {exc}")

    if not supps:
        raise ValueError(f"No usable sheets found in {path.name}")
    if len(supps) == 1:
        return supps[0]

    # Merge all sheets on sample_name (same logic as read_multiple_supplementary)
    merged: dict[str, dict] = {}
    ordered_keys: list[str] = []

    for supp in supps:
        for rec in supp.to_element_records():
            key = str(
                rec.get("sample_local_id")
                or rec.get("sample_name")
                or len(merged)
            )
            if key not in merged:
                merged[key] = {}
                ordered_keys.append(key)
            for k, v in rec.items():
                if k not in merged[key] or merged[key][k] is None:
                    merged[key][k] = v

    rows = [merged[k] for k in ordered_keys]
    data_df = pd.DataFrame(rows)

    all_syms: set[str] = set()
    for s in supps:
        all_syms.update(s.element_col_map.values())
    element_col_map = {f"{sym}_ppm": sym for sym in sorted(all_syms)
                       if f"{sym}_ppm" in data_df.columns}

    all_notes.append(f"Merged {len(supps)} sheets → {len(data_df)} rows, "
                     f"{len(element_col_map)} element columns")

    return SupplementaryTable(
        raw_df=supps[0].raw_df,
        data_df=data_df,
        element_col_map=element_col_map,
        sample_id_col="sample_local_id" if "sample_local_id" in data_df.columns else "sample_name",
        deposit_col="deposit_name" if "deposit_name" in data_df.columns else None,
        reference_col=None,
        unit="ppm",
        notes=all_notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _LoadResult:
    """Result of loading an Excel/CSV file with header detection."""
    df: pd.DataFrame
    pre_header_text: str = ""
    pre_header_rows: list[list[str]] = field(default_factory=list)


def _load_excel(
    path: Path,
    sheet_name: int | str,
    hints: Optional[ParsingHints] = None,
) -> tuple[pd.DataFrame, str, list[list[str]]]:
    """Load Excel, trying to auto-detect the real header row.

    Returns (DataFrame, pre_header_text, pre_header_rows) where:
    - pre_header_text: concatenated text from rows before the header
    - pre_header_rows: list of row values (as string lists) for per-column unit detection
    """
    # Read with no header first to inspect
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str)

    # Find the header row — hint overrides auto-detection
    if hints and hints.header_row is not None:
        header_row = hints.header_row
    else:
        header_row = _find_header_row(raw)
        if header_row is None:
            header_row = 0

    # Collect pre-header text and rows for unit detection
    pre_header_text = ""
    pre_header_rows: list[list[str]] = []
    if header_row > 0:
        parts = []
        for i in range(header_row):
            row_vals = raw.iloc[i].astype(str).tolist()
            pre_header_rows.append(row_vals)
            non_nan = [v for v in row_vals if v not in ("nan", "None", "")]
            parts.extend(non_nan)
        pre_header_text = " ".join(parts)

    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, dtype=object)

    # ── Multi-row header fix ─────────────────────────────────────────────
    # If the chosen header row left some columns as "Unnamed:*", the next
    # row may contain metadata column names (e.g. "Mineral", "Sample Type").
    # Promote those values to column names and drop the sub-header row.
    unnamed_idxs = [
        i for i, c in enumerate(df.columns) if str(c).startswith("Unnamed:")
    ]
    if unnamed_idxs and len(df) > 0:
        first_row = df.iloc[0]
        # Count how many unnamed positions have text labels in the first row
        text_count = sum(
            1 for i in unnamed_idxs
            if pd.notna(first_row.iloc[i])
            and isinstance(first_row.iloc[i], str)
            and first_row.iloc[i].strip()
            and normalise_element_header(first_row.iloc[i]) is None  # not an element
        )
        if text_count >= max(1, len(unnamed_idxs) * 0.4):
            new_cols = list(df.columns)
            for i in unnamed_idxs:
                val = first_row.iloc[i]
                if pd.notna(val) and isinstance(val, str) and val.strip():
                    new_cols[i] = val.strip()
            df.columns = new_cols
            df = df.iloc[1:].reset_index(drop=True)
    # ─────────────────────────────────────────────────────────────────────

    df = _clean_dataframe(df)
    return df, pre_header_text, pre_header_rows


def _load_csv(
    path: Path,
    hints: Optional[ParsingHints] = None,
) -> tuple[pd.DataFrame, str, list[list[str]]]:
    """Load CSV, detecting encoding and header row.

    Returns (DataFrame, pre_header_text, pre_header_rows).
    """
    raw = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = pd.read_csv(path, header=None, dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    # Header row — hint overrides auto-detection
    if hints and hints.header_row is not None:
        header_row = hints.header_row
    else:
        header_row = _find_header_row(raw)
        if header_row is None:
            header_row = 0

    # Collect pre-header text and rows
    pre_header_text = ""
    pre_header_rows: list[list[str]] = []
    if header_row > 0 and raw is not None:
        parts = []
        for i in range(header_row):
            row_vals = raw.iloc[i].astype(str).tolist()
            pre_header_rows.append(row_vals)
            non_nan = [v for v in row_vals if v not in ("nan", "None", "")]
            parts.extend(non_nan)
        pre_header_text = " ".join(parts)

    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, header=header_row, dtype=object, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    df = _clean_dataframe(df)
    return df, pre_header_text, pre_header_rows


def _find_header_row(raw: pd.DataFrame) -> Optional[int]:
    """Return the index of the row with the most element-symbol matches.

    Skips title/description rows (rows with very few non-empty cells) and
    searches up to 20 rows deep to handle multi-row headers.
    """
    best_row, best_count = 0, 0
    max_search = min(20, len(raw))
    for idx in range(max_search):
        row = raw.iloc[idx]
        non_empty = sum(1 for v in row if pd.notna(v) and str(v).strip())
        # Skip likely title rows (≤3 non-empty cells in a wide table)
        if non_empty <= 3 and len(raw.columns) > 5:
            continue
        count = sum(
            1 for v in row
            if isinstance(v, str) and normalise_element_header(v) is not None
        )
        if count > best_count:
            best_count = count
            best_row = idx
    return best_row if best_count >= 2 else None


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from headers and string cells."""
    df.columns = [str(c).strip() if c is not None else f"col_{i}"
                  for i, c in enumerate(df.columns)]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Column role detectors
# ──────────────────────────────────────────────────────────────────────────────

def _detect_sample_id_col(df: pd.DataFrame) -> Optional[str]:
    """Find the column most likely to be the sample identifier.

    Uses first-match in column order, but skips columns with very low coverage
    (<30% non-null) when a better-filled candidate exists later.
    """
    candidates = [
        "sample", "sample_id", "sample id", "sample name",
        "sample no", "sample no.", "sample number",
        "spot", "spot_id", "spot id", "spot no", "spot no.", "spot name",
        "analysis", "analysis_id", "analysis id",
        "analysis no", "analysis no.", "analysis number",
        "analythical spot", "analytical spot",
        "dot id", "dot_id", "dot no", "dot no.",
        "id", "label", "name", "point", "grain",
        "number", "no", "no.",
    ]
    # Column names that should NOT be treated as sample IDs
    _EXCLUDE_PATTERNS = {
        "µg/g", "ug/g", "μg/g", "ppm", "ppb", "wt%", "wt %",
        "mg/kg", "g/t", "%", "mean", "2se", "2sd", "sd", "se",
        "detection limit", "lod", "dl",
    }
    # Find all candidate columns with their coverage and uniqueness
    n_rows = len(df)
    found: list[tuple[str, float, float, bool]] = []  # (col, coverage, uniqueness, is_numeric)
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in candidates:
            vals = df[col].dropna().astype(str)
            n_vals = len(vals)
            coverage = n_vals / n_rows if n_rows > 0 else 0.0
            uniqueness = vals.nunique() / n_vals if n_vals > 0 else 0.0
            numeric_frac = sum(1 for v in vals if _safe_float(v) is not None) / n_vals if n_vals > 0 else 0.0
            is_numeric = numeric_frac > 0.8
            found.append((col, coverage, uniqueness, is_numeric))

    if found:
        # If the first match has good coverage (>=30%), use it (original behavior)
        if found[0][1] >= 0.3:
            return found[0][0]
        # First match has low coverage (merged cells). Look for a better alternative:
        # prefer a column with BOTH high coverage AND alphanumeric (non-numeric) values
        best_text = None
        for col, cov, uniq, is_num in found[1:]:
            if cov >= 0.5 and not is_num and uniq > 0.5:
                best_text = col
                break
        if best_text:
            return best_text
        # No good textual alternative — use first match (will be forward-filled later)
        return found[0][0]
    # First column that has mostly unique string values
    for col in df.columns[:5]:
        col_lower = col.lower().strip()
        # Skip columns that look like units or element symbols
        if col_lower in _EXCLUDE_PATTERNS or normalise_element_header(col) is not None:
            continue
        vals = df[col].dropna().astype(str)
        if len(vals) > 0 and vals.nunique() / len(vals) > 0.7:
            if not any(v.lower() in ("nan", "none", "") for v in vals.head(3)):
                return col
    return df.columns[0] if len(df.columns) > 0 else None


def _detect_deposit_col(df: pd.DataFrame) -> Optional[str]:
    """Find the column most likely to be the deposit/location name."""
    candidates = [
        "ore deposit", "deposit", "mine", "location", "area", "district",
        "ore diposit",  # typo in Yuan et al. supplementary
    ]
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


def _detect_reference_col(df: pd.DataFrame) -> Optional[str]:
    """Find the reference/source column."""
    candidates = ["reference", "ref", "source", "citation", "data source"]
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


def _detect_mineral_col(df: pd.DataFrame) -> Optional[str]:
    """Find the column containing mineral names (per-sample metadata)."""
    candidates = [
        "mineral", "mineral1", "mineral_phase", "phase", "mineral phase",
        "mineral name", "mineralogy", "min", "analyzed mineral",
    ]
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


def _detect_method_col(df: pd.DataFrame) -> Optional[str]:
    """Find the column containing analytical method (per-sample metadata)."""
    candidates = [
        "method", "analytical method", "technique", "analysis method",
        "analytical technique", "analysis", "analysis_method",
    ]
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


def _detect_zone_col(df: pd.DataFrame) -> Optional[str]:
    """Find the column containing zone/texture/type info (per-sample metadata)."""
    candidates = [
        "zone", "type", "texture", "crystal zone", "spot type",
        "grain type", "generation",
    ]
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


# ──────────────────────────────────────────────────────────────────────────────
# USGS: Mineral inference from sheet names and analysis_id abbreviations
# ──────────────────────────────────────────────────────────────────────────────

# Common abbreviations found in analysis_id strings (e.g., "5-2002063521cpy1-1.d")
_MINERAL_ABBREVIATIONS: dict[str, str] = {
    "cpy": "chalcopyrite",
    "cp":  "chalcopyrite",
    "ccp": "chalcopyrite",
    "sph": "sphalerite",
    "sp":  "sphalerite",
    "sl":  "sphalerite",
    "gal": "galena",
    "gn":  "galena",
    "py":  "pyrite",
    "pyr": "pyrite",
    "po":  "pyrrhotite",
    "apy": "arsenopyrite",
    "asp": "arsenopyrite",
    "bn":  "bornite",
    "bor": "bornite",
    "cc":  "chalcocite",
    "cv":  "covellite",
    "en":  "enargite",
    "tn":  "tennantite",
    "td":  "tetrahedrite",
    "tt":  "tetrahedrite",
    "mol": "molybdenite",
    "moly": "molybdenite",
    "pn":  "pentlandite",
    "mt":  "magnetite",
    "mag": "magnetite",
    "hem": "hematite",
    "ilm": "ilmenite",
    "cst": "cassiterite",
    "cas": "cassiterite",
    "wf":  "wolframite",
    "sch": "scheelite",
    "qtz": "quartz",
    "qz":  "quartz",
    "cal": "calcite",
    "dol": "dolomite",
    "fl":  "fluorite",
    "brt": "barite",
    "bar": "barite",
}


def _infer_mineral_from_label(label: str) -> Optional[str]:
    """Infer mineral name from a sheet name, table caption, or label string.

    Matches against the MINERAL_TAXONOMY from knowledge_base.py and common
    abbreviations. Case-insensitive.

    Examples:
        "Chalcopyrite" → "chalcopyrite"
        "Sph LA-ICPMS" → "sphalerite"
        "Table 2 - pyrite data" → "pyrite"
    """
    if not label:
        return None
    from .knowledge_base import MINERAL_TAXONOMY
    import re
    label_lower = label.strip().lower()

    # Direct match against known mineral names — longest first to avoid
    # "pyrite" matching before "chalcopyrite" in "chalcopyrite"
    sorted_minerals = sorted(MINERAL_TAXONOMY.keys(), key=len, reverse=True)
    for mineral_name in sorted_minerals:
        # Match as whole word (bounded by non-alpha or string edges)
        pattern = r'(?:^|[^a-z])' + re.escape(mineral_name) + r'(?:[^a-z]|$)'
        if re.search(pattern, label_lower):
            return mineral_name

    # Match against abbreviations
    # Split on non-alpha to get tokens
    tokens = re.split(r'[^a-zA-Z]+', label_lower)
    for token in tokens:
        if token in _MINERAL_ABBREVIATIONS:
            return _MINERAL_ABBREVIATIONS[token]

    return None


def infer_mineral_from_analysis_id(analysis_id: str) -> Optional[str]:
    """Extract mineral from analysis_id string using embedded abbreviations.

    Per USGS protocol, analysis_id strings often contain mineral abbreviations:
        "5-2002063521cpy1-1.d" → "chalcopyrite" (from "cpy")
        "YK94-17-sph-3" → "sphalerite" (from "sph")
        "PY-1-1" → "pyrite" (from "py")

    Returns mineral name or None.
    """
    if not analysis_id:
        return None
    import re
    aid_lower = analysis_id.strip().lower()

    # Look for abbreviation tokens separated by non-alpha characters or at boundaries
    # Try longer abbreviations first to avoid false matches
    for abbrev, mineral in sorted(_MINERAL_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        # Match as a standalone token (bounded by non-alpha or string edges)
        pattern = r'(?:^|[^a-z])' + re.escape(abbrev) + r'(?:[^a-z]|$)'
        if re.search(pattern, aid_lower):
            return mineral

    return None


def _detect_spot_number_col(
    df: pd.DataFrame,
    sample_id_col: Optional[str],
) -> Optional[str]:
    """Find a secondary 'spot number' column for composing natural sample IDs.

    Looks for columns named "Spot", "Point", "No.", "Analysis No." etc. that
    contain sequential/numeric values and are distinct from the primary sample ID.
    Returns the column name if found, else None.
    """
    _SPOT_CANDIDATES = (
        "spot", "spot no", "spot no.", "point", "point no", "point no.",
        "analysis no", "analysis no.",
    )
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower not in _SPOT_CANDIDATES:
            continue
        # Don't use the same column as sample_id
        if col == sample_id_col:
            continue
        # Verify it has mostly numeric values (spot numbers)
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        numeric_count = sum(1 for v in vals if _safe_float(v) is not None)
        if numeric_count / len(vals) >= 0.5:
            return col
    return None


def _map_element_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return {raw_col_name: element_symbol} for all detected element columns."""
    mapping = {}
    for col in df.columns:
        sym = normalise_element_header(col)
        if sym is not None:
            mapping[col] = sym
    return mapping


_WT_INDICATORS = ("wt%", "wt %", "wt.%", "(wt%)", "(wt.%)", "(%)", "mass%", "(mass%)")
_PPM_INDICATORS = ("ppm", "ppb", "µg/g", "ug/g", "μg/g", "mg/kg", "g/t")


def _detect_unit_from_headers(
    df: pd.DataFrame,
    pre_header_text: str = "",
) -> str:
    """Detect the predominant unit from column headers and pre-header text.

    Returns 'wt%' if most element columns contain wt%/% indicators,
    otherwise returns 'ppm' (the default).
    When both wt% and ppm indicators are present (mixed units), returns 'mixed'.
    """
    wt_count = 0
    ppm_count = 0

    for col in df.columns:
        col_lower = str(col).lower().strip()
        if normalise_element_header(col) is None:
            continue
        if any(u in col_lower for u in _WT_INDICATORS):
            wt_count += 1
        elif any(u in col_lower for u in _PPM_INDICATORS):
            ppm_count += 1

    # Check pre-header text for global unit indicators
    if pre_header_text:
        pht = pre_header_text.lower()
        has_wt = any(u in pht for u in _WT_INDICATORS)
        has_ppm = any(u in pht for u in _PPM_INDICATORS)
        # EMPA/EPMA method implies wt% for major elements
        has_empa = any(m in pht for m in ("empa", "epma", "electron microprobe",
                                           "electron probe"))
        if has_wt and has_ppm:
            return "mixed"  # pre-header explicitly mentions both
        if has_empa and not has_ppm and wt_count == 0 and ppm_count == 0:
            # EMPA data without explicit unit → default to wt%
            wt_count += 5
        if has_wt:
            wt_count += 5
        if has_ppm:
            ppm_count += 5

    if wt_count > 0 and ppm_count > 0:
        return "mixed"
    if wt_count > 0:
        return "wt%"
    return "ppm"


def _detect_wt_pct_columns(
    df: pd.DataFrame,
    element_col_map: dict[str, str],
    pre_header_rows: list[list[str]] | None = None,
) -> set[str]:
    """Return the set of raw column names that are in wt% (need ×10000 conversion).

    Checks:
    1. Column header text for wt%/mass% indicators
    2. A dedicated 'unit row' just above or below the header (e.g., a row with
       'wt.%', 'μg/g' values aligned to each column)
    """
    wt_cols: set[str] = set()

    # 1. Check column header text directly
    for raw_col in element_col_map:
        col_lower = raw_col.lower()
        if any(u in col_lower for u in _WT_INDICATORS):
            wt_cols.add(raw_col)

    # 2. Check pre-header rows for a unit row
    if pre_header_rows:
        col_names = list(df.columns)
        for unit_row in pre_header_rows:
            if len(unit_row) != len(col_names):
                continue
            # Check if this looks like a unit row (has wt% or ppm strings)
            unit_values = [str(v).strip().lower() for v in unit_row]
            has_units = sum(
                1 for v in unit_values
                if any(u in v for u in _WT_INDICATORS + _PPM_INDICATORS)
            )
            if has_units < 2:
                continue
            # Map each column to its unit
            for i, col_name in enumerate(col_names):
                if col_name not in element_col_map:
                    continue
                if i < len(unit_values):
                    uv = unit_values[i]
                    if any(u in uv for u in _WT_INDICATORS):
                        wt_cols.add(col_name)

    return wt_cols


def _convert_wt_pct_to_ppm(
    df: pd.DataFrame,
    wt_cols: set[str] | dict[str, str] | None = None,
    element_col_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert element columns from wt% to ppm (multiply by 10000).

    Preserves original values in {col}_original_value and sets
    {col}_original_unit = "wt%" before converting.
    """
    df = df.copy()
    cols_to_convert = wt_cols if wt_cols else (element_col_map or {})
    for raw_col in cols_to_convert:
        if raw_col in df.columns:
            # Preserve original before conversion
            orig_val_col = raw_col.replace("_ppm", "_original_value")
            orig_unit_col = raw_col.replace("_ppm", "_original_unit")
            df[orig_val_col] = pd.to_numeric(df[raw_col], errors="coerce")
            df[orig_unit_col] = "wt%"
            df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce") * 10000
    return df


def _detect_wt_pct_by_value_range(
    df: pd.DataFrame,
    element_col_map: dict[str, str],
) -> set[str]:
    """Detect wt% columns using value-range heuristics when headers are ambiguous.

    Only major elements (Fe, Si, Al, Ca, Mg, Na, K, S, Mn, Ti, P, Cr) are
    candidates for wt% conversion. Trace elements are never in wt%.

    A column is classified as wt% if:
    - The element is a known major element
    - The median positive value is between 0.01 and 100
    - Values >100 ppm would be unusual for wt% (max wt% = 100%)

    This prevents trace elements already in ppm (e.g., Cu=369 ppm) from
    being multiplied by 10,000.
    """
    _MAJOR_ELEMENTS = {"fe", "si", "al", "ca", "mg", "na", "k", "s", "mn", "ti", "p", "cr"}
    wt_cols: set[str] = set()

    for raw_col, sym in element_col_map.items():
        if sym not in _MAJOR_ELEMENTS:
            continue
        if raw_col not in df.columns:
            continue
        vals = pd.to_numeric(df[raw_col], errors="coerce").dropna()
        # Exclude BDL sentinels
        vals = vals[(vals > 0) & (vals != -99999)]
        if len(vals) < 2:
            continue
        median = vals.median()
        # wt% range: 0.01% to 100% (100% = pure element)
        # ppm range for major elements: typically >1000 ppm
        if 0.01 <= median <= 100:
            wt_cols.add(raw_col)

    return wt_cols


_PPB_HEADER_INDICATORS = ("ppb", "(ppb)", "ppb)", "ng/g", "(ng/g)")


def _detect_ppb_columns(
    df: pd.DataFrame,
    element_col_map: dict[str, str],
    pre_header_rows: list[list[str]] | None = None,
) -> set[str]:
    """Return column names that are in ppb (need ÷1000 conversion to ppm)."""
    ppb_cols: set[str] = set()

    # Check column header text
    for raw_col in element_col_map:
        col_lower = raw_col.lower()
        if any(u in col_lower for u in _PPB_HEADER_INDICATORS):
            ppb_cols.add(raw_col)

    # Check pre-header unit rows
    if pre_header_rows:
        col_names = list(df.columns)
        for unit_row in pre_header_rows:
            if len(unit_row) != len(col_names):
                continue
            for i, col_name in enumerate(col_names):
                if col_name not in element_col_map:
                    continue
                if i < len(unit_row):
                    uv = str(unit_row[i]).strip().lower()
                    if any(u in uv for u in _PPB_HEADER_INDICATORS):
                        ppb_cols.add(col_name)

    return ppb_cols


def _convert_ppb_to_ppm(
    df: pd.DataFrame,
    ppb_cols: set[str],
) -> pd.DataFrame:
    """Convert element columns from ppb to ppm (divide by 1000).

    Preserves original values in {col}_original_value and sets
    {col}_original_unit = "ppb" before converting.
    """
    df = df.copy()
    for raw_col in ppb_cols:
        if raw_col in df.columns:
            orig_val_col = raw_col.replace("_ppm", "_original_value")
            orig_unit_col = raw_col.replace("_ppm", "_original_unit")
            df[orig_val_col] = pd.to_numeric(df[raw_col], errors="coerce")
            df[orig_unit_col] = "ppb"
            df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce") / 1000
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Transposed table detection and pivot
# ──────────────────────────────────────────────────────────────────────────────

def _find_transposed_label_col(df: pd.DataFrame) -> Optional[int]:
    """Find which column (0-2) contains element labels in a transposed table.

    Returns the column index, or None if no column qualifies.
    """
    for col_idx in range(min(3, len(df.columns))):
        col_vals = df.iloc[:, col_idx].dropna().astype(str).tolist()
        if not col_vals:
            continue
        matches = sum(1 for v in col_vals if normalise_element_header(v) is not None)
        if matches >= 5 and matches / len(col_vals) >= 0.3:
            return col_idx
    return None


def _is_transposed(df: pd.DataFrame) -> bool:
    """Return True if the table is transposed (rows=elements, columns=samples).

    Detects this by checking whether any of the first 3 columns contain
    element symbol patterns rather than sample IDs. If column headers already
    contain element symbols, the table is in normal orientation.
    """
    if df.empty or len(df.columns) < 2:
        return False

    # If column headers already have element symbols, table is NOT transposed
    col_element_count = sum(
        1 for c in df.columns
        if normalise_element_header(str(c)) is not None
    )
    if col_element_count >= 3:
        return False

    return _find_transposed_label_col(df) is not None


def _pivot_transposed(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a transposed table (rows=elements, columns=samples) into normal form.

    Handles tables like the Denisova LAICPMS/EMPA format where:
      - One of the first 3 columns is the row label (element name or metadata)
      - Remaining columns are individual analyses / spots
      - Some early rows are sample metadata (Spot_id, TS_id, Zone, etc.)
      - One column may be "Average detection limit" — it is dropped
      - Element rows have values like '0.65', 'bdl', '-'

    Returns a normal DataFrame with one row per sample and element columns.
    """
    # Auto-detect which column contains element labels (may not be col 0)
    label_col_idx = _find_transposed_label_col(df) or 0

    # Columns before the label column may contain auxiliary data (e.g. detection limits)
    # Columns after the label column are sample data
    prefix_cols = list(range(label_col_idx))  # cols to drop
    data_start = label_col_idx + 1

    label_col = df.iloc[:, label_col_idx].astype(str).str.strip()

    # Identify which column (if any) is the detection limit column
    # (the Assemblage row often has "Average detection limit" in a nearby column)
    dl_col_idx: Optional[int] = None
    for row_idx in range(min(8, len(df))):
        for col_idx in range(data_start, min(data_start + 4, len(df.columns))):
            cell = str(df.iloc[row_idx, col_idx]).strip().lower()
            if "detection limit" in cell or "average detection" in cell:
                dl_col_idx = col_idx
                break
        if dl_col_idx is not None:
            break

    # Build set of columns to exclude from sample data
    exclude_cols = set(prefix_cols)
    exclude_cols.add(label_col_idx)
    if dl_col_idx is not None:
        exclude_cols.add(dl_col_idx)

    # Data columns = everything not excluded
    data_col_idxs = [i for i in range(len(df.columns)) if i not in exclude_cols]

    # Separate metadata rows from element rows
    sample_meta: dict[str, list] = {}   # metadata_field → [val_col1, val_col2, ...]
    element_rows: dict[str, list] = {}  # element_sym    → [val_col1, val_col2, ...]

    # Sample IDs from column headers
    col_header_ids = [str(df.columns[i]).strip() for i in data_col_idxs]

    _META_LABELS = {
        "spot_id": "sample_local_id",
        "ts_id": "sample_name",
        "ts id": "sample_name",
        "sample_id": "sample_name",
        "sample": "sample_name",
        "zone": "zone",
        "mineral": "mineral",
        "mineral1": "mineral",
        "assemblage": "assemblage",
        "generation": "zone",
    }

    # Check if the label column header contains spot/sample keywords
    label_header = str(df.columns[label_col_idx]).strip().lower()
    if any(kw in label_header for kw in ("spot", "sample", "analysis", "id")):
        sample_meta["sample_local_id"] = col_header_ids

    # Track which elements are in wt% (need ×10000 conversion to ppm)
    wt_pct_elements: set[str] = set()
    # A standalone "wt.%" / "wt%" label row means all elements are in wt%
    global_wt_pct = False

    for i, label in enumerate(label_col):
        label_lower = label.lower().strip()
        vals = [df.iloc[i, c] for c in data_col_idxs]

        # Detect standalone unit label row (e.g. "wt.%", "wt%", "(wt%)")
        if label_lower in ("wt%", "wt.%", "(wt%)", "(wt.%)", "%"):
            global_wt_pct = True
            continue

        sym = normalise_element_header(label)
        if sym is not None:
            element_rows[sym] = vals
            # Detect wt% unit in original label or from global unit marker
            if global_wt_pct or any(u in label_lower for u in ("wt%", "wt %", "(wt%)", "(%)")):
                wt_pct_elements.add(sym)
            continue

        mapped = _META_LABELS.get(label_lower)
        if mapped:
            sample_meta[mapped] = vals
        elif i == 0:
            # First row = sample IDs (fallback) — only if values are mostly unique
            if "sample_local_id" not in sample_meta:
                candidate_ids = [str(df.iloc[0, c]).strip() for c in data_col_idxs]
                non_nan = [v for v in candidate_ids if v.lower() not in ("nan", "none", "")]
                # Require ≥50% non-NaN and ≥30% unique to avoid garbage IDs
                if (len(non_nan) > len(candidate_ids) * 0.5
                        and len(set(non_nan)) > len(non_nan) * 0.3):
                    sample_meta["sample_local_id"] = candidate_ids

    # Build one dict per sample column
    n_samples = len(col_header_ids)
    records = []
    for j in range(n_samples):
        rec: dict = {}
        # Sample metadata
        for field, vals in sample_meta.items():
            rec[field] = _safe_str(vals[j]) if j < len(vals) else None
        # Ensure sample_name falls back to sample_local_id
        if "sample_name" not in rec or rec["sample_name"] is None:
            rec["sample_name"] = rec.get("sample_local_id")
        # Element values — keep in original units (no conversion).
        # The "_ppm" suffix is a column name convention; ground truth stores
        # values as-reported (wt%, ppm, ppb) without normalisation.
        for sym, vals in element_rows.items():
            val = _safe_float(vals[j]) if j < len(vals) else None
            rec[f"{sym}_ppm"] = val
        records.append(rec)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Row filtering
# ──────────────────────────────────────────────────────────────────────────────

def _filter_sample_rows(
    df: pd.DataFrame,
    sample_id_col: Optional[str],
    reference_col: Optional[str],
    this_paper_deposit: Optional[str],
    deposit_col: Optional[str],
) -> pd.DataFrame:
    """Keep only genuine sample-analysis rows from 'this paper'."""
    mask = pd.Series([True] * len(df), index=df.index)

    # 1. Exclude summary/statistics rows based on sample_id content
    if sample_id_col:
        mask &= ~df[sample_id_col].apply(
            lambda v: bool(isinstance(v, str) and _SUMMARY_ROW_PATTERNS.match(v))
        )
        # Also catch summary patterns embedded in names (e.g., "Py2(n=24)")
        mask &= ~df[sample_id_col].apply(
            lambda v: bool(isinstance(v, str) and _SUMMARY_EMBEDDED_PATTERN.search(v))
        )
        # Exclude reference standard rows (NIST, MASS-1, etc.)
        mask &= ~df[sample_id_col].apply(
            lambda v: bool(isinstance(v, str) and _REFERENCE_STANDARD_PATTERNS.match(v.strip()))
        )
        # Exclude note/description rows (sentences, long text)
        mask &= ~df[sample_id_col].apply(
            lambda v: isinstance(v, str) and _is_note_or_description(v)
        )
        # Exclude rows where sample_id looks like a citation/source reference
        _CITATION_PATTERN = re.compile(
            r"Source:\s*|"                     # "Source: Author et al."
            r"\bet\s+al\.?\s*\(\d{4}\)|"      # "et al. (2022)"
            r"\bet\s+al\.?\s*,\s*\d{4}|"      # "et al., 2022"
            r"^\d{4}[a-z]?\s+[A-Z][a-z]",     # "2022 Bertrandsson"
            re.IGNORECASE,
        )
        mask &= ~df[sample_id_col].apply(
            lambda v: bool(isinstance(v, str) and _CITATION_PATTERN.search(v))
        )
        # Exclude rows with no sample ID
        mask &= df[sample_id_col].notna()
        mask &= df[sample_id_col].apply(
            lambda v: pd.notna(v) and str(v).strip() not in ("", "nan", "None", "NaN")
        )

    # 1b. TAG rows by data source instead of removing them.
    # Extract all, let downstream users decide what to keep.
    # Tag: "this_study" | "reference_data" | "cited_study:AuthorYear"
    if "data_source_tag" not in df.columns:
        df["data_source_tag"] = "this_study"  # default

    # Scan text columns for "Source:" pattern → tag as cited_study
    for col in df.columns:
        if col in (sample_id_col, "data_source_tag"):
            continue
        if df[col].dtype == object:
            for idx in df.index:
                val = df.at[idx, col]
                if isinstance(val, str) and val.strip().lower().startswith("source:"):
                    df.at[idx, "data_source_tag"] = f"cited_study:{val.strip()[:50]}"
                elif isinstance(val, str):
                    s = val.strip().lower()
                    if s and s not in _THIS_PAPER_ALIASES and s != "":
                        # Check if it looks like a citation
                        if re.search(r'\bet\s+al\.?\s*[,(]\s*\d{4}', s):
                            df.at[idx, "data_source_tag"] = f"cited_study:{val.strip()[:50]}"

    # 2. TAG by reference column if present (don't filter)
    if reference_col and reference_col in df.columns:
        for idx in df.index:
            val = df.at[idx, reference_col]
            if pd.isna(val):
                continue
            s = str(val).strip().lower()
            if s and s not in _THIS_PAPER_ALIASES:
                df.at[idx, "data_source_tag"] = f"cited_study:{str(val).strip()[:50]}"

    # 3. Filter by deposit name if caller specified one and deposit col exists
    if this_paper_deposit and deposit_col and deposit_col in df.columns:
        target = this_paper_deposit.strip().lower()
        # Use fuzzy matching: exact OR substring containment
        # This handles cases like GT="Daliangzi" vs table="Daliangzi deposit"
        def _deposit_matches(v):
            if not isinstance(v, str):
                return False
            val = v.strip().lower()
            if not val:
                return True  # blank deposit = assume this paper
            return val == target or target in val or val in target
        deposit_mask = df[deposit_col].apply(_deposit_matches)
        # Only apply if the filter keeps a reasonable fraction of rows
        # (avoids wiping all data when deposit names don't match)
        kept = deposit_mask.sum()
        if kept > 0 and kept < len(df):
            mask &= deposit_mask

    return df[mask].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Type helpers
# ──────────────────────────────────────────────────────────────────────────────

# Sentinel value for "below detection limit" — the element WAS measured but
# the concentration was too low to quantify.  This is distinct from None/blank
# which means the element was NOT measured or NOT reported at all.
# Using -99999 loses less information: blank implies "not attempted", while
# -99999 means "attempted, but at very low concentration".
BELOW_DETECTION_SENTINEL = -99999.0

# Strings that mean "below detection limit" (element WAS analysed, too low to quantify).
# USGS convention: "n/a" and "na" mean "not analyzed" = blank, NOT BDL.
_BDL_STRINGS = frozenset({
    "bdl", "b.d.l.", "b.d.l", "bdl.", "<dl", "<d.l.", "nd", "n.d.", "n.d",
    "-", "--", "below detection", "below detection limit",
    "<mdl", "mdl", "<lod", "lod", "b.d.", "bd",
    "below lod", "below dl", "below mdl",
})

# Strings that mean "not analyzed / not applicable" = blank (None).
_NOT_ANALYZED_STRINGS = frozenset({
    "n/a", "na", "n.a.", "n.a", "not analyzed", "not analysed",
    "not measured", "not reported", "not determined", "n.r.", "nr",
})


def _is_below_detection(val) -> bool:
    """Return True if a cell value represents below-detection-limit.

    These are values where the analysis was performed but the element
    concentration was too low to quantify. Distinct from truly missing
    (not measured) values.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _BDL_STRINGS:
            return True
        # Leading < with a number (e.g., "<0.01", "< 5") is below detection
        if s and s[0] == "<":
            return True
    return False


def _extract_detection_limit(val) -> Optional[float]:
    """Extract the numeric detection limit from a BDL cell value.

    Per USGS protocol: if a specific detection limit is reported (e.g., "<0.5"),
    return the negative of that limit (e.g., -0.5). If no specific limit is
    given (e.g., "bdl", "n.d."), return None (caller should use -99999).

    Returns:
        Negative float (e.g., -0.5) if a specific LOD is embedded in the value.
        None if no specific LOD can be extracted.
    """
    if not isinstance(val, str):
        return None
    import re
    s = val.strip()
    # Pattern: "<0.5", "< 0.01", "<0.5 ppm", etc.
    m = re.match(r'^<\s*([0-9]*\.?[0-9]+)', s)
    if m:
        try:
            lod = float(m.group(1))
            if lod > 0:
                return round(-lod, 6)
        except (ValueError, TypeError):
            pass
    return None


def _safe_float(val) -> Optional[float]:
    """Convert a cell value to float.

    USGS below-detection-limit convention:
        - BDL with no specific LOD (e.g., "bdl", "n.d.", "-") → -99999
        - BDL with specific LOD (e.g., "<0.5") → negative of that limit (-0.5)
        - "n/a", "not analyzed" → None (blank = not measured)
        - Normal numeric value → float
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if not s:
            return None
        # "not analyzed" strings → None (blank), NOT BDL
        if s in _NOT_ANALYZED_STRINGS:
            return None
        if s in _BDL_STRINGS:
            return BELOW_DETECTION_SENTINEL
        # Leading < with a number → extract specific LOD as negative value
        if s[0] == "<":
            specific_lod = _extract_detection_limit(val)
            if specific_lod is not None:
                return specific_lod
            return BELOW_DETECTION_SENTINEL
        # Leading > strips the indicator and returns the numeric value
        if s[0] == ">":
            s = s[1:].strip()
            try:
                return round(float(s), 6)
            except (ValueError, TypeError):
                return None
    try:
        # Handle comma-separated thousands (e.g., "50,525" → 50525)
        cleaned = str(val).replace(",", "") if isinstance(val, str) else val
        f = float(cleaned)
        return None if pd.isna(f) else round(f, 6)
    except (ValueError, TypeError):
        return None


def _safe_str(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s if s not in ("", "nan", "None") else None


def _normalize_method(val: Optional[str]) -> Optional[str]:
    """Normalize an analytical method string to its canonical form."""
    if val is None:
        return None
    from .knowledge_base import normalize_method
    return normalize_method(val)


# ──────────────────────────────────────────────────────────────────────────────
# Parse whitespace-separated tabular text from PDF pages into DataFrames
# ──────────────────────────────────────────────────────────────────────────────

def parse_text_tables_from_pages(
    pages: list[str],
    page_indices: list[int] | None = None,
    min_element_cols: int = 3,
    min_data_rows: int = 2,
) -> list[tuple[pd.DataFrame, int]]:
    """Parse whitespace-separated tabular data from PDF page text.

    Scans each page for lines that look like a header row (containing
    element symbols), then collects subsequent lines as data rows.
    Returns a list of (DataFrame, page_index) tuples ready for ``read_pdf_table()``.

    This is the key to lossless numeric extraction from PDF pages where
    pdfplumber/Camelot/Docling fail to detect structured tables but the
    raw text is well-formatted whitespace-aligned data.
    """
    results: list[tuple[pd.DataFrame, int]] = []

    for i, page_text in enumerate(pages):
        pidx = page_indices[i] if page_indices and i < len(page_indices) else i
        dfs = _parse_tables_from_text(page_text, min_element_cols, min_data_rows)
        for df in dfs:
            results.append((df, pidx))

    return results


def _parse_tables_from_text(
    text: str,
    min_element_cols: int = 3,
    min_data_rows: int = 2,
) -> list[pd.DataFrame]:
    """Parse one page of text into DataFrames."""
    lines = text.splitlines()
    tables: list[pd.DataFrame] = []
    i = 0

    while i < len(lines):
        # Look for a header line containing element symbols
        header_cols, element_count = _try_parse_header(lines[i])
        if header_cols and element_count >= min_element_cols:
            n_cols = len(header_cols)
            # If header is ALL elements (no "Spot No." etc.), data rows may
            # have extra leading columns (sample name, subtype).
            # Detect by checking if the first few data lines have more tokens.
            has_sample_col = not all(
                normalise_element_header(c.strip().rstrip("/%").lstrip("(").rstrip(")"))
                is not None for c in header_cols
            )
            extra_lead = 0
            if not has_sample_col:
                # Probe next lines to detect extra leading columns in data rows.
                # Skip blank/short lines (sub-headers, "Spot No." labels, etc.)
                for probe_k in range(1, min(10, len(lines) - i)):
                    probe_line = lines[i + probe_k].strip()
                    if not probe_line:
                        continue
                    probe_tokens = probe_line.split()
                    if len(probe_tokens) > n_cols + 1:
                        extra_lead = max(extra_lead, len(probe_tokens) - n_cols)
                        break  # Found a real data line
                if extra_lead > 0:
                    # Prepend placeholder column names for leading non-element cols
                    leading_names = ["Spot No."] if extra_lead == 1 else \
                        ["Spot No."] + [f"_extra_{k}" for k in range(1, extra_lead)]
                    header_cols = leading_names[:extra_lead] + header_cols
                    n_cols = len(header_cols)

            # Collect data rows.
            # Allow skipping non-data lines between header and first data
            # (sub-headers like "Spot No.\nSphalerite\nsubtype" can span
            # multiple lines). Once data starts, allow only 1-line gaps.
            data_rows = []
            j = i + 1
            found_first_data = False
            while j < len(lines):
                row = _try_parse_data_row(lines[j], n_cols)
                if row is not None:
                    found_first_data = True
                    data_rows.append(row)
                    j += 1
                elif not found_first_data and j < i + 8:
                    # Still looking for first data row — skip sub-headers
                    j += 1
                else:
                    # Allow one non-matching line (e.g. page break text)
                    # then check if data continues
                    if j + 1 < len(lines):
                        row2 = _try_parse_data_row(lines[j + 1], n_cols)
                        if row2 is not None:
                            j += 1
                            continue
                    break

            if len(data_rows) >= min_data_rows:
                df = pd.DataFrame(data_rows, columns=header_cols)
                # Replace below-detection-limit markers with -99999 sentinel.
                # -99999 means "measured but below detection" (not the same as blank/None
                # which means "not measured at all").
                _bdl = BELOW_DETECTION_SENTINEL
                df = df.replace({"-": _bdl, "b.d.l.": _bdl, "bdl": _bdl,
                                 "n.d.": _bdl, "nd": _bdl, "BDL": _bdl,
                                 "N.D.": _bdl, "<LOD": _bdl})
                tables.append(df)
                i = j
            else:
                i += 1
        else:
            i += 1

    return tables


def _try_parse_header(line: str) -> tuple[list[str] | None, int]:
    """Check if a line looks like a table header with element symbols.

    Returns (column_names, element_count) or (None, 0).
    Merges adjacent non-element tokens into multi-word headers
    (e.g. "Spot No." becomes one column).
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 10:
        return None, 0

    # Split by whitespace
    raw_tokens = stripped.split()
    if len(raw_tokens) < 4:
        return None, 0

    # Merge adjacent non-element tokens into multi-word headers
    # e.g. ["Spot", "No.", "As", "Bi"] → ["Spot No.", "As", "Bi"]
    tokens: list[str] = []
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        cleaned = tok.strip().rstrip("/%").lstrip("(").rstrip(")")
        sym = normalise_element_header(cleaned)
        if sym is not None:
            tokens.append(tok)
            i += 1
        else:
            # Non-element token: merge with subsequent non-element tokens
            merged = tok
            j = i + 1
            while j < len(raw_tokens):
                next_tok = raw_tokens[j]
                next_cleaned = next_tok.strip().rstrip("/%").lstrip("(").rstrip(")")
                next_sym = normalise_element_header(next_cleaned)
                if next_sym is not None:
                    break  # Next token is an element, stop merging
                merged += " " + next_tok
                j += 1
            tokens.append(merged)
            i = j

    # Count element symbols in merged tokens
    element_count = 0
    for tok in tokens:
        cleaned = tok.strip().rstrip("/%").lstrip("(").rstrip(")")
        sym = normalise_element_header(cleaned)
        if sym is not None:
            element_count += 1

    if element_count >= 3:
        return tokens, element_count

    return None, 0


def _try_parse_data_row(line: str, expected_cols: int) -> list[str] | None:
    """Try to parse a line as a data row matching the expected column count.

    Returns list of cell values, or None if the line doesn't look like data.
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 5:
        return None

    tokens = stripped.split()

    # Allow some flexibility in column count (±2 for merged/split cells)
    if abs(len(tokens) - expected_cols) > 2:
        return None

    # A data row should have significant numeric content
    numeric_count = 0
    dash_count = 0
    for tok in tokens:
        clean = tok.replace(",", "").replace("<", "").replace(">", "")
        if clean == "-":
            dash_count += 1
            continue
        try:
            float(clean)
            numeric_count += 1
        except ValueError:
            pass

    # At least 30% of tokens should be numeric or dashes
    data_tokens = numeric_count + dash_count
    if data_tokens < len(tokens) * 0.3:
        return None

    # Pad or truncate to match expected columns
    if len(tokens) < expected_cols:
        tokens.extend([None] * (expected_cols - len(tokens)))
    elif len(tokens) > expected_cols:
        tokens = tokens[:expected_cols]

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Utility: render table as plain text for LLM prompts
# ──────────────────────────────────────────────────────────────────────────────

def dataframe_to_text(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Convert a DataFrame to a readable pipe-delimited text table."""
    df_head = df.head(max_rows)
    lines = ["| " + " | ".join(str(c) for c in df_head.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(df_head.columns)) + "|")
    for _, row in df_head.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"... ({len(df) - max_rows} more rows not shown)")
    return "\n".join(lines)
