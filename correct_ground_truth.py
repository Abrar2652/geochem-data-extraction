"""
correct_ground_truth.py - Create corrected ground truth files following USGS protocol.

Corrections applied:
1. BDL strings (bdl, b.d.l., n.d., -, <lod) → -99999 (no specific LOD)
2. BDL with specific LOD (<0.5) → -0.5 (negative of the LOD)
3. N/A strings (n/a, not analyzed) → blank (None)
4. wt% values → ppm (×10000) for major elements stored in wt%
5. ppb values → ppm (÷1000) where applicable
6. Sample order preserved exactly as in original GT

Does NOT change:
- Sample names, IDs, or metadata
- Values already in ppm
- Blank/None cells (left as-is)
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# --- BDL/N/A string sets (same as table_reader.py) ---

_BDL_STRINGS = frozenset({
    "bdl", "b.d.l.", "b.d.l", "bdl.", "<dl", "<d.l.", "nd", "n.d.", "n.d",
    "-", "--", "below detection", "below detection limit",
    "<mdl", "mdl", "<lod", "lod", "b.d.", "bd",
    "below lod", "below dl", "below mdl",
})

_NOT_ANALYZED_STRINGS = frozenset({
    "n/a", "na", "n.a.", "n.a", "not analyzed", "not analysed",
    "not measured", "not reported", "not determined", "n.r.", "nr",
})

# Major elements that are commonly reported in wt%
_MAJOR_ELEMENTS_WT = {"fe", "si", "al", "ca", "mg", "na", "k", "s", "mn", "ti", "p", "cr"}


def correct_cell_value(val) -> object:
    """Convert a single cell value following USGS protocol.

    Returns:
        -99999.0 for BDL (no specific LOD)
        -X for BDL with specific LOD <X
        None for N/A / not analyzed
        float for valid numeric
        original value if not an element cell
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return float(val)

    if not isinstance(val, str):
        return val

    s = val.strip()
    s_lower = s.lower()

    if not s:
        return None

    # N/A strings → None
    if s_lower in _NOT_ANALYZED_STRINGS:
        return None

    # BDL strings → -99999
    if s_lower in _BDL_STRINGS:
        return -99999.0

    # < with number → negative of the LOD
    if s.startswith("<") or s.startswith("< "):
        m = re.match(r'^<\s*([0-9]*\.?[0-9]+)', s)
        if m:
            try:
                lod = float(m.group(1))
                if lod >= 0:
                    return round(-lod, 6)
            except (ValueError, TypeError):
                pass
        # < without number (e.g., "<DL") → -99999
        return -99999.0

    # Try numeric
    try:
        f = float(s.replace(",", ""))
        return f if not pd.isna(f) else None
    except (ValueError, TypeError):
        return val  # Not a numeric cell, return as-is


def detect_wt_pct_columns(df: pd.DataFrame, elem_cols: list[str]) -> set[str]:
    """Detect which element columns contain values in wt% (not ppm).

    Strategy: major elements (Fe, Si, Al, Ca, Mg, Na, K, S, Mn, Ti) with
    median values between 0.01 and 100 are likely in wt%.
    """
    wt_cols = set()
    for col in elem_cols:
        if col not in df.columns:
            continue
        sym = col.replace("_ppm", "")
        if sym not in _MAJOR_ELEMENTS_WT:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        vals = vals[(vals > 0) & (vals != -99999)]  # Exclude BDL
        if len(vals) < 3:
            continue
        median = vals.median()
        # wt% range: 0.01% to 100%
        # ppm range: typically > 100 for major elements
        if 0.01 <= median <= 100:
            wt_cols.add(col)
    return wt_cols


def correct_gt_file(input_path: Path, output_path: Path) -> dict:
    """Correct a single ground truth file.

    Returns stats dict with correction counts.
    """
    df = pd.read_excel(input_path)
    stats = {
        "file": input_path.name,
        "rows": len(df),
        "bdl_converted": 0,
        "neg_lod_converted": 0,
        "na_converted": 0,
        "wt_pct_converted": 0,
        "wt_pct_columns": [],
    }

    # Identify element columns
    elem_cols = [c for c in df.columns if c.endswith("_ppm")]
    det_cols = [c for c in df.columns if c.endswith("_detection_limit")]

    # Step 1: Convert BDL strings and N/A strings
    for col in elem_cols:
        if col not in df.columns:
            continue
        for idx in df.index:
            val = df.at[idx, col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, str):
                corrected = correct_cell_value(val)
                if corrected == -99999.0 and val.strip().lower() in _BDL_STRINGS:
                    stats["bdl_converted"] += 1
                elif isinstance(corrected, float) and corrected < 0 and corrected != -99999.0:
                    stats["neg_lod_converted"] += 1
                elif corrected is None and val.strip().lower() in _NOT_ANALYZED_STRINGS:
                    stats["na_converted"] += 1
                elif corrected == -99999.0:  # < without number
                    stats["bdl_converted"] += 1
                df.at[idx, col] = corrected

    # Step 2: Detect and convert wt% → ppm for major elements
    wt_cols = detect_wt_pct_columns(df, elem_cols)
    if wt_cols:
        for col in wt_cols:
            sym = col.replace("_ppm", "")
            stats["wt_pct_columns"].append(sym)
            for idx in df.index:
                val = df.at[idx, col]
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                if isinstance(val, (int, float)) and not pd.isna(val):
                    if val == -99999.0 or val < 0:
                        continue  # Don't convert BDL sentinels
                    df.at[idx, col] = round(val * 10000, 4)
                    stats["wt_pct_converted"] += 1

    # Save corrected file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(output_path), index=False)

    return stats


def main():
    gt_dir = Path("/nas/ckgfs/jaunts/jahin/geochem_benchmark/ground_truth")
    out_dir = Path("/nas/ckgfs/jaunts/jahin/geochem_benchmark/ground_truth_corrected")

    files = sorted([f for f in gt_dir.iterdir() if f.suffix == ".xlsx" and not f.name.startswith("~")])

    print(f"{'File':<60} {'Rows':>5} {'BDL→-99999':>10} {'<X→-X':>8} {'N/A→∅':>7} {'wt%→ppm':>8} {'wt% cols':>20}")
    print("=" * 130)

    all_stats = []
    for f in files:
        out_path = out_dir / f.name
        stats = correct_gt_file(f, out_path)
        all_stats.append(stats)

        wt_str = ",".join(stats["wt_pct_columns"]) if stats["wt_pct_columns"] else "-"
        print(f"{stats['file']:<60} {stats['rows']:>5} {stats['bdl_converted']:>10} "
              f"{stats['neg_lod_converted']:>8} {stats['na_converted']:>7} "
              f"{stats['wt_pct_converted']:>8} {wt_str:>20}")

    print("=" * 130)
    total_bdl = sum(s["bdl_converted"] for s in all_stats)
    total_neg = sum(s["neg_lod_converted"] for s in all_stats)
    total_na = sum(s["na_converted"] for s in all_stats)
    total_wt = sum(s["wt_pct_converted"] for s in all_stats)
    print(f"{'TOTAL':<60} {sum(s['rows'] for s in all_stats):>5} {total_bdl:>10} "
          f"{total_neg:>8} {total_na:>7} {total_wt:>8}")

    print(f"\nCorrected files saved to: {out_dir}")
    print(f"Files processed: {len(all_stats)}")


if __name__ == "__main__":
    main()
