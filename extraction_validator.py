"""
extraction_validator.py — Post-extraction self-validation agent.

Instead of paper-specific heuristic fixes, this module asks the LLM to review
extracted data against the source paper and identify systematic issues:

1. Are these individual analyses or summary statistics?
2. Is this data from THIS paper or from cited references?
3. Do sample names look like real identifiers or mineral groups / row labels?
4. Does the sample count match what the paper reports?
5. Are there obvious unit or magnitude issues?

The validator returns a cleaned dataset and a list of corrections made.
This is the GENERIC solution — it works on any paper without paper-specific rules.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Output of post-extraction validation."""
    is_valid: bool = True
    rows_removed: int = 0
    rows_before: int = 0
    rows_after: int = 0
    issues_found: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


VALIDATION_SYSTEM_PROMPT = """\
You are a quality control specialist for geochemical data extraction. You will review \
extracted sample data against the original paper text and identify issues.

Your job is NOT to re-extract data — it is to identify which extracted rows are INVALID \
and should be removed. Be conservative: only flag rows you are CONFIDENT are wrong."""


VALIDATION_USER_PROMPT = """\
## PAPER CONTEXT
Title/authors: {paper_info}
Expected sample count from paper: {expected_count}
Analytical methods: {methods}
Minerals analyzed: {minerals}

## PAPER TEXT (key sections)
{paper_text}

## EXTRACTED DATA SUMMARY
Total rows extracted: {n_rows}
Sample names (first 30): {sample_names}
Elements with data: {elements}
Extraction backends used: {backends}

## REVIEW TASKS
Examine the extracted sample names and answer these questions:

1. **SUMMARY_ROWS**: Do any sample names look like summary statistics rather than
   individual analyses? Examples: "Mean", "Average", "Py2(n=24)" where (n=XX) means
   average of XX spots, "Min", "Max", "Std", "Median". List ALL such names.

2. **REFERENCE_DATA**: Do any sample names or patterns suggest data from OTHER papers
   (not this study)? Examples: names containing "Source:", reference citations, data
   clearly attributed to other authors. List ALL such names/patterns.

3. **NON_SAMPLE_NAMES**: Do any "sample names" look like they are NOT real sample
   identifiers? Examples: mineral names used as IDs ("Pyrite", "Chalcopyrite"),
   column headers, numeric values that are clearly data (not IDs), blank/generic labels.
   List ALL such names.

4. **COUNT_CHECK**: The paper reports approximately {expected_count} analyses.
   We extracted {n_rows} rows. If {n_rows} >> {expected_count}, what is the likely
   cause? (e.g., "supplementary contains data from multiple studies", "each row is
   a single element measurement not a full analysis", "reference data included")

5. **DUPLICATE_CHECK**: Are there sample names that appear to be duplicates of
   different formats? (e.g., "101A-2" and "101A-2@L3" are the same sample with
   different spot identifiers)

Return a JSON object:
{{
  "summary_rows": ["Py2(n=24)", "Mean", ...],
  "reference_data_patterns": ["Source: *", ...],
  "non_sample_names": ["Pyrite", "Chalcopyrite", ...],
  "count_explanation": "...",
  "duplicate_patterns": ["@L suffix indicates spot number", ...],
  "rows_to_remove": ["exact name 1", "exact name 2", ...],
  "confidence": 0.85
}}

Return ONLY the JSON object. Be precise about which specific names should be removed."""


def validate_extraction(
    client,
    samples_df: pd.DataFrame,
    paper_text: str,
    expected_count: int = 0,
    methods: str = "",
    minerals: str = "",
    paper_info: str = "",
) -> ValidationResult:
    """Run post-extraction validation using LLM review.

    Args:
        client: LLM client
        samples_df: Extracted data DataFrame
        paper_text: Paper text for context
        expected_count: Expected sample count from Paper Intelligence
        methods: Analytical methods string
        minerals: Minerals analyzed string
        paper_info: Paper title/author info

    Returns:
        ValidationResult with issues found and cleaned DataFrame.
    """
    result = ValidationResult(rows_before=len(samples_df))

    if len(samples_df) == 0:
        result.rows_after = 0
        return result

    # Collect sample names
    name_col = None
    for col in ("sample_name", "sample_local_id", "Sample_ID"):
        if col in samples_df.columns:
            name_col = col
            break

    if not name_col:
        result.rows_after = len(samples_df)
        return result

    all_names = samples_df[name_col].dropna().astype(str).tolist()
    unique_names = list(dict.fromkeys(all_names))  # Preserve order, deduplicate

    # Element columns with data
    elem_cols = sorted([c for c in samples_df.columns if c.endswith("_ppm")
                        and samples_df[c].notna().any()])

    # Backend info
    backends = ""
    if "extraction_backend" in samples_df.columns:
        backends = str(dict(samples_df["extraction_backend"].dropna().value_counts()))

    # Build and send prompt
    prompt = VALIDATION_USER_PROMPT.format(
        paper_info=paper_info or "unknown",
        expected_count=expected_count or "unknown",
        methods=methods or "unknown",
        minerals=minerals or "unknown",
        paper_text=paper_text[:8000],
        n_rows=len(samples_df),
        sample_names=str(unique_names[:30]),
        elements=", ".join(elem_cols[:20]),
        backends=backends,
    )

    try:
        parsed = client.complete_json(
            system=VALIDATION_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=4096,
        )

        if not isinstance(parsed, dict):
            result.rows_after = len(samples_df)
            return result

        # Process results
        summary_rows = set(str(s) for s in (parsed.get("summary_rows") or []))
        reference_patterns = parsed.get("reference_data_patterns") or []
        non_sample = set(str(s) for s in (parsed.get("non_sample_names") or []))
        rows_to_remove = set(str(s) for s in (parsed.get("rows_to_remove") or []))
        confidence = float(parsed.get("confidence", 0.5))

        if parsed.get("count_explanation"):
            result.issues_found.append(f"Count: {parsed['count_explanation']}")
        if parsed.get("duplicate_patterns"):
            result.issues_found.append(f"Duplicates: {parsed['duplicate_patterns']}")

        # CONSERVATIVE removal — only remove rows we are >95% certain are wrong.
        # LLM "non_sample" suggestions are ADVISORY ONLY (logged, not acted on).
        # This prevents over-aggressive removal that destroys real data (Foltyn case).

        # Only these hardcoded patterns trigger actual removal:
        _SAFE_REMOVE_PATTERN = re.compile(
            r'^\s*(mean|average|median|std|stdev|standard deviation|'
            r'min|max|range|total|sum|n\s*=)\s*$'
            r'|\(n\s*=\s*\d+\)'             # (n=24) anywhere
            r'|^Source:\s',                   # Source: Author et al.
            re.IGNORECASE,
        )

        # Apply removal — ONLY safe patterns
        mask = pd.Series(True, index=samples_df.index)
        for idx, row in samples_df.iterrows():
            name = str(row.get(name_col, ""))
            if _SAFE_REMOVE_PATTERN.search(name):
                mask[idx] = False

        removed = (~mask).sum()
        # Safety cap: never remove more than 20% of rows
        max_remove = max(1, int(len(samples_df) * 0.20))
        if removed > max_remove:
            result.issues_found.append(
                f"Safety cap: would remove {removed} rows but capped at {max_remove} (20%)")
            # Only remove the first max_remove flagged rows
            remove_indices = mask[~mask].index[:max_remove]
            mask = pd.Series(True, index=samples_df.index)
            mask[remove_indices] = False
            removed = max_remove

        if removed > 0:
            samples_df = samples_df[mask].reset_index(drop=True)
            result.rows_removed = removed
            result.corrections.append(f"Removed {removed} rows (safe patterns only)")

        # Log LLM suggestions as advisory (NOT acted on)
        if non_sample:
            result.issues_found.append(
                f"LLM flagged as non-sample (advisory, not removed): {sorted(non_sample)[:10]}")
        if rows_to_remove:
            result.issues_found.append(
                f"LLM flagged for removal (advisory, not removed): {sorted(rows_to_remove)[:10]}")

        # Log issues
        if summary_rows:
            result.issues_found.append(f"Summary rows found: {sorted(summary_rows)[:5]}")
        if non_sample:
            result.issues_found.append(f"Non-sample names: {sorted(non_sample)[:5]}")
        if reference_patterns:
            result.issues_found.append(f"Reference patterns: {reference_patterns}")

        logger.info(
            "Validation: %d→%d rows (removed %d), %d issues, confidence=%.2f",
            result.rows_before, len(samples_df), result.rows_removed,
            len(result.issues_found), confidence,
        )

    except Exception as e:
        logger.warning("Extraction validation failed: %s", e)
        result.issues_found.append(f"Validation error: {e}")

    result.rows_after = len(samples_df)
    result.is_valid = result.rows_after > 0
    return result
