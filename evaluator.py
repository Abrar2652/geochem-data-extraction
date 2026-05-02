"""
evaluator.py - Benchmark LLM extraction results against the ground truth.

Evaluation tiers:
  T1 — Metadata accuracy     (categorical / string fields, paper-level)
  T2 — Numerical accuracy    (element concentrations per sample row)
  T3 — Structural accuracy   (row count, sample name matching)
  T4 — NULL accuracy         (correct assignment of null vs non-null)

Overall score = weighted average of T1..T4 scores.
"""

from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .pipeline import ExtractionResult
from .schema import ALL_COLUMNS, ELEMENT_SYMBOLS, PaperMetadata, BELOW_DETECTION_SENTINEL

# ──────────────────────────────────────────────────────────────────────────────
# Weights for overall score
# ──────────────────────────────────────────────────────────────────────────────
TIER_WEIGHTS = {
    "T1_metadata":    0.30,
    "T2_numerical":   0.40,
    "T3_structural":  0.15,
    "T4_null":        0.15,
}

# Metadata fields evaluated in T1
METADATA_EVAL_FIELDS = [
    "deposit_name",
    "deposit_environment",
    "deposit_group",
    "deposit_type",
    "all_commodities",
    "mineral",
    "analytical_method",
    "instrument_type_model",
    "laboratory_location/if reported",
    "operating_conditions/if reported",
    "standards_used/if reported",
    "country",
    "publication_date",
]

# Relative-error tolerance for numerical values (5 %)
NUMERICAL_TOLERANCE = 0.05

# Absolute-error tolerance for very small values
ABSOLUTE_TOLERANCE = 0.001


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldResult:
    """Per-field evaluation result."""
    field_name: str
    predicted: object
    ground_truth: object
    score: float          # 0.0 – 1.0
    tier: str             # T1 / T2 / T3 / T4
    note: str = ""


@dataclass
class SampleMatchResult:
    """Evaluation result for one matched sample pair."""
    sample_name: str
    numerical_results: list[FieldResult] = field(default_factory=list)
    null_results:      list[FieldResult] = field(default_factory=list)

    @property
    def numerical_score(self) -> float:
        if not self.numerical_results:
            return 1.0
        return sum(r.score for r in self.numerical_results) / len(self.numerical_results)

    @property
    def null_score(self) -> float:
        if not self.null_results:
            return 1.0
        return sum(r.score for r in self.null_results) / len(self.null_results)


@dataclass
class BenchmarkReport:
    """Complete benchmark report for one LLM result."""
    model: str
    provider: str

    # Per-tier scores (0–100 %)
    t1_metadata_score:   float = 0.0
    t2_numerical_score:  float = 0.0
    t3_structural_score: float = 0.0
    t4_null_score:       float = 0.0
    overall_score:       float = 0.0

    # Per-field metadata results
    metadata_results:    list[FieldResult] = field(default_factory=list)
    # Per-sample numerical/null results
    sample_results:      list[SampleMatchResult] = field(default_factory=list)

    # Structural stats
    predicted_n_samples:     int = 0
    ground_truth_n_samples:  int = 0
    matched_samples:         int = 0
    missing_samples:         list[str] = field(default_factory=list)
    extra_samples:           list[str] = field(default_factory=list)

    # Precision / Recall / F1 for sample matching
    sample_precision:        float = 0.0  # matched / predicted — are our extractions real?
    sample_recall:           float = 0.0  # matched / gt — did we find everything?
    sample_f1:               float = 0.0  # harmonic mean

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "scores": {
                "T1_metadata_%": round(self.t1_metadata_score * 100, 2),
                "T2_numerical_%": round(self.t2_numerical_score * 100, 2),
                "T3_structural_%": round(self.t3_structural_score * 100, 2),
                "T4_null_%": round(self.t4_null_score * 100, 2),
                "overall_%": round(self.overall_score * 100, 2),
            },
            "structural": {
                "predicted_n": self.predicted_n_samples,
                "ground_truth_n": self.ground_truth_n_samples,
                "matched_n": self.matched_samples,
                "precision_%": round(self.sample_precision * 100, 2),
                "recall_%": round(self.sample_recall * 100, 2),
                "f1_%": round(self.sample_f1 * 100, 2),
                "missing_samples": self.missing_samples[:10],
                "extra_samples": self.extra_samples[:10],
            },
        }

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {self.provider}/{self.model}")
        print(f"{'='*60}")
        print(f"  T1 Metadata    : {self.t1_metadata_score*100:6.1f}%")
        print(f"  T2 Numerical   : {self.t2_numerical_score*100:6.1f}%")
        print(f"  T3 Structural  : {self.t3_structural_score*100:6.1f}%")
        print(f"  T4 Null        : {self.t4_null_score*100:6.1f}%")
        print(f"  {'─'*30}")
        print(f"  Overall        : {self.overall_score*100:6.1f}%")
        print(f"  {'─'*30}")
        print(f"  Precision      : {self.sample_precision*100:6.1f}%")
        print(f"  Recall         : {self.sample_recall*100:6.1f}%")
        print(f"  F1             : {self.sample_f1*100:6.1f}%")
        print(f"  Rows: pred={self.predicted_n_samples} | gt={self.ground_truth_n_samples} | matched={self.matched_samples}")
        print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Compare an ExtractionResult against a ground truth DataFrame."""

    def __init__(
        self,
        ground_truth_path: str | Path,
        metadata_tolerance: float = 0.85,  # fuzzy string match threshold
        numerical_tolerance: float = NUMERICAL_TOLERANCE,
    ):
        """
        Args:
            ground_truth_path: Path to the ground truth Excel file.
            metadata_tolerance: Min similarity score (0–1) for string fields.
            numerical_tolerance: Max relative error for numerical fields (0–1).
        """
        self.gt_df = _load_ground_truth(ground_truth_path)
        self.metadata_tolerance = metadata_tolerance
        self.numerical_tolerance = numerical_tolerance

    def evaluate(self, result: ExtractionResult) -> BenchmarkReport:
        """Evaluate an ExtractionResult against the ground truth."""
        return self.evaluate_dataframe(
            pred_df=result.to_dataframe(),
            model=result.llm_model,
            provider=result.llm_provider,
        )

    def evaluate_dataframe(
        self,
        pred_df: pd.DataFrame,
        model: str,
        provider: str,
    ) -> BenchmarkReport:
        """Evaluate a raw prediction DataFrame against the ground truth.

        Use this when you already have an extracted Excel/CSV file and want to
        score it without re-running the LLM.
        """
        report = BenchmarkReport(model=model, provider=provider)

        # T1: Metadata
        report.metadata_results = self._eval_metadata(pred_df)
        report.t1_metadata_score = _mean_score(report.metadata_results)

        # T3: Structural (must be done before T2/T4 which need matched samples)
        struct = self._eval_structural(pred_df)
        report.t3_structural_score     = struct["score"]
        report.predicted_n_samples     = struct["predicted_n"]
        report.ground_truth_n_samples  = struct["gt_n"]
        report.matched_samples         = struct["matched_n"]
        report.missing_samples         = struct["missing"]
        report.extra_samples           = struct["extra"]

        # Precision / Recall / F1 for sample name matching
        report.sample_precision = (struct["matched_n"] / struct["predicted_n"]
                                   if struct["predicted_n"] > 0 else 0.0)
        report.sample_recall    = (struct["matched_n"] / struct["gt_n"]
                                   if struct["gt_n"] > 0 else 1.0)
        if (report.sample_precision + report.sample_recall) > 0:
            report.sample_f1 = (2 * report.sample_precision * report.sample_recall
                                / (report.sample_precision + report.sample_recall))
        else:
            report.sample_f1 = 0.0

        # T2 + T4: Per-sample numerical and NULL accuracy
        # Strategy: use name-based matching first. If that gives poor coverage
        # (<30% of GT matched), fall back to position-based matching which
        # compares rows by position and validates via element value overlap.
        t2_scores: list[float] = []
        t4_scores: list[float] = []

        name_matched_n = len(struct["matched_sample_names"])
        gt_n = struct["gt_n"]
        use_position_matching = (
            name_matched_n < gt_n * 0.3 and gt_n > 0 and len(pred_df) > 0
        )

        if use_position_matching:
            # Position-based matching: align GT and pred by row order,
            # then validate each pair by checking element value overlap.
            pos_matches = _position_based_matching(self.gt_df, pred_df)
            for gt_idx, pred_idx, match_quality in pos_matches:
                gt_row = self.gt_df.iloc[gt_idx]
                pred_row = pred_df.iloc[pred_idx]
                label = f"pos_{gt_idx}"

                smr = SampleMatchResult(sample_name=label)
                smr.numerical_results = self._eval_numerical_row(pred_row, gt_row)
                smr.null_results      = self._eval_null_row(pred_row, gt_row)
                report.sample_results.append(smr)
                t2_scores.append(smr.numerical_score)
                t4_scores.append(smr.null_score)

            # Update structural metrics to reflect position matching
            report.matched_samples = len(pos_matches)
            report.sample_precision = (len(pos_matches) / len(pred_df)
                                       if len(pred_df) > 0 else 0.0)
            report.sample_recall = (len(pos_matches) / gt_n
                                    if gt_n > 0 else 1.0)
            if (report.sample_precision + report.sample_recall) > 0:
                report.sample_f1 = (2 * report.sample_precision * report.sample_recall
                                    / (report.sample_precision + report.sample_recall))
            else:
                report.sample_f1 = 0.0
            report.t3_structural_score = report.sample_f1

        else:
            # Standard name-based matching
            for sample_name in struct["matched_sample_names"]:
                pred_row = _get_sample_row(pred_df, sample_name)
                gt_row   = _get_sample_row(self.gt_df, sample_name)
                if pred_row is None or gt_row is None:
                    continue

                smr = SampleMatchResult(sample_name=sample_name)
                smr.numerical_results = self._eval_numerical_row(pred_row, gt_row)
                smr.null_results      = self._eval_null_row(pred_row, gt_row)
                report.sample_results.append(smr)
                t2_scores.append(smr.numerical_score)
                t4_scores.append(smr.null_score)

        report.t2_numerical_score = sum(t2_scores) / len(t2_scores) if t2_scores else 0.0
        report.t4_null_score      = sum(t4_scores) / len(t4_scores) if t4_scores else 0.0

        # Overall weighted score
        report.overall_score = (
            TIER_WEIGHTS["T1_metadata"]   * report.t1_metadata_score +
            TIER_WEIGHTS["T2_numerical"]  * report.t2_numerical_score +
            TIER_WEIGHTS["T3_structural"] * report.t3_structural_score +
            TIER_WEIGHTS["T4_null"]       * report.t4_null_score
        )
        return report

    # ── T1: Metadata ──────────────────────────────────────────────────────────

    def _eval_metadata(self, pred_df: pd.DataFrame) -> list[FieldResult]:
        """Compare paper-level metadata fields (uses first row values)."""
        results = []
        pred_row = pred_df.iloc[0] if not pred_df.empty else pd.Series(dtype=object)
        gt_row   = self.gt_df.iloc[0] if not self.gt_df.empty else pd.Series(dtype=object)

        for field_name in METADATA_EVAL_FIELDS:
            pred_val = _clean_val(pred_row.get(field_name))
            gt_val   = _clean_val(gt_row.get(field_name))

            if gt_val is None:
                # Cannot score if ground truth is absent
                continue

            score, note = self._compare_strings(pred_val, gt_val)
            results.append(FieldResult(
                field_name=field_name,
                predicted=pred_val,
                ground_truth=gt_val,
                score=score,
                tier="T1",
                note=note,
            ))
        return results

    def _compare_strings(self, pred: Optional[str], gt: Optional[str]) -> tuple[float, str]:
        """Return (score, note) for a string field comparison."""
        if gt is None:
            return 1.0, "GT absent"
        if pred is None:
            return 0.0, "Predicted absent"

        pred_n = _normalise_str(pred)
        gt_n   = _normalise_str(gt)

        if pred_n == gt_n:
            return 1.0, "Exact match"

        sim = _string_similarity(pred_n, gt_n)
        if sim >= self.metadata_tolerance:
            return sim, f"Fuzzy match ({sim:.2f})"
        return sim, f"Mismatch (sim={sim:.2f})"

    # ── T2: Numerical ─────────────────────────────────────────────────────────

    def _eval_numerical_row(
        self,
        pred_row: pd.Series,
        gt_row: pd.Series,
    ) -> list[FieldResult]:
        """Score element concentration values for one matched sample.

        Handles the below-detection-limit sentinel (-99999):
        - GT=-99999, pred=-99999 → perfect match (both recognised BDL)
        - GT=-99999, pred=None  → penalty (pred missed a BDL marker)
        - GT=-99999, pred=X>0   → partial credit if X is small
        - GT=X>0,   pred=-99999 → wrong (pred says BDL but GT has value)
        """
        results = []
        _BDL = BELOW_DETECTION_SENTINEL
        for sym in ELEMENT_SYMBOLS:
            col = f"{sym}_ppm"
            gt_val  = _safe_float(gt_row.get(col))
            pred_val = _safe_float(pred_row.get(col))

            if gt_val is None:
                continue  # Not measured in this paper — evaluated in T4

            # GT is BDL (-99999)
            if gt_val == _BDL:
                if pred_val is not None and pred_val == _BDL:
                    score, note = 1.0, "Both BDL — correct"
                elif pred_val is None:
                    score, note = 0.5, "GT is BDL, pred is null (missed BDL marker)"
                else:
                    # Pred has a real value but GT says BDL — partial if small value
                    score, note = 0.3, f"GT is BDL, pred={pred_val}"
                results.append(FieldResult(
                    field_name=col, predicted=pred_val, ground_truth=gt_val,
                    score=score, tier="T2", note=note,
                ))
                continue

            # GT has a real value
            if pred_val is None:
                results.append(FieldResult(
                    field_name=col, predicted=None, ground_truth=gt_val,
                    score=0.0, tier="T2", note="Predicted null, GT has value"
                ))
                continue

            if pred_val == _BDL:
                # Pred says BDL but GT has a real value — wrong
                results.append(FieldResult(
                    field_name=col, predicted=pred_val, ground_truth=gt_val,
                    score=0.0, tier="T2", note=f"Pred BDL but GT={gt_val}"
                ))
                continue

            score = _numerical_score(pred_val, gt_val, self.numerical_tolerance)
            rel_err = abs(pred_val - gt_val) / gt_val if gt_val != 0 else 0.0
            results.append(FieldResult(
                field_name=col, predicted=pred_val, ground_truth=gt_val,
                score=score, tier="T2",
                note=f"rel_err={rel_err*100:.2f}%"
            ))
        return results

    # ── T4: NULL accuracy ─────────────────────────────────────────────────────

    def _eval_null_row(
        self,
        pred_row: pd.Series,
        gt_row: pd.Series,
    ) -> list[FieldResult]:
        """Score NULL / non-NULL assignment for elements not in this paper.

        GT=None means the element was NOT measured at all.
        - pred=None  → correct (element wasn't measured)
        - pred=-99999 → wrong (pred claims it was measured-but-BDL, but it wasn't measured)
        - pred=value → wrong (hallucinated a value for an unmeasured element)
        """
        results = []
        _BDL = BELOW_DETECTION_SENTINEL
        for sym in ELEMENT_SYMBOLS:
            col = f"{sym}_ppm"
            gt_val   = _safe_float(gt_row.get(col))
            pred_val = _safe_float(pred_row.get(col))

            if gt_val is not None:
                continue  # Covered by T2

            # GT is null — predict should also be null
            if pred_val is None:
                score, note = 1.0, "Correct null"
            elif pred_val == _BDL:
                # Pred claims BDL but element wasn't measured at all
                score, note = 0.0, "Pred claims BDL (-99999) but GT=null (not measured)"
            else:
                score, note = 0.0, f"Hallucinated value {pred_val} (GT=null)"

            results.append(FieldResult(
                field_name=col, predicted=pred_val, ground_truth=None,
                score=score, tier="T4", note=note,
            ))
        return results

    # ── T3: Structural ────────────────────────────────────────────────────────

    def _eval_structural(self, pred_df: pd.DataFrame) -> dict:
        """Score row count, sample name coverage.

        Tries all combinations of sample-name columns (sample_name,
        sample_local_id, Sample_ID, sample_id) from GT and prediction to find
        the best match.  Uses exact matching first, then bidirectional prefix
        matching for spot-level suffixes like -1, .d, @L1.
        """
        best_key = (-1, -1.0)  # (matched_n, f1) — prefer more matches first
        best_result = None

        for gt_col in _SAMPLE_COLUMNS:
            gt_list = _get_sample_names_from_col(self.gt_df, gt_col)
            if not gt_list:
                continue
            gt_set = set(gt_list)

            for pred_col in _SAMPLE_COLUMNS:
                pred_list = _get_sample_names_from_col(pred_df, pred_col)
                if not pred_list:
                    continue
                pred_set = set(pred_list)

                matched, missing, extra = _match_sample_names(
                    set(gt_set), set(pred_set)
                )

                precision = len(matched) / len(pred_set) if pred_set else 0.0
                recall    = len(matched) / len(gt_set)   if gt_set   else 1.0
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)

                key = (len(matched), f1)
                if key > best_key:
                    best_key = key
                    best_result = {
                        "score":               f1,
                        "predicted_n":         len(pred_df),
                        "gt_n":                len(self.gt_df),
                        "matched_n":           len(matched),
                        "matched_sample_names": sorted(matched),
                        "missing":             sorted(missing),
                        "extra":               sorted(extra),
                    }

        if best_result is None:
            return {
                "score": 0.0, "predicted_n": len(pred_df),
                "gt_n": len(self.gt_df), "matched_n": 0,
                "matched_sample_names": [], "missing": [], "extra": [],
            }
        return best_result


# ──────────────────────────────────────────────────────────────────────────────
# No-ground-truth evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_completeness(
    pred_df: pd.DataFrame,
    model: str,
    provider: str,
) -> dict:
    """Score an extraction without any ground truth.

    Returns a dict with:
      - metadata_completeness_%  : % of key metadata fields that are non-null
      - element_completeness_%   : % of element _ppm columns with ≥1 non-null value
      - schema_compliance_%      : % of the 210 expected columns present in pred_df
      - row_count                : number of rows extracted
      - null_metadata_fields     : list of key metadata fields that were left null
      - present_elements         : list of elements with at least one value
    """
    report: dict = {"model": model, "provider": provider}

    # Schema compliance — are all 210 columns present?
    expected = set(ALL_COLUMNS)
    present  = set(pred_df.columns)
    missing_cols = sorted(expected - present)
    report["schema_compliance_%"] = round(100 * len(present & expected) / len(expected), 2)
    report["missing_columns"]     = missing_cols

    # Metadata completeness
    null_meta = []
    filled = 0
    for f in METADATA_EVAL_FIELDS:
        val = _clean_val(pred_df.iloc[0].get(f)) if not pred_df.empty else None
        if val is not None:
            filled += 1
        else:
            null_meta.append(f)
    report["metadata_completeness_%"] = round(100 * filled / len(METADATA_EVAL_FIELDS), 2)
    report["null_metadata_fields"]    = null_meta

    # Element completeness — how many of the 73 element _ppm columns have any values
    present_elems = []
    for sym in ELEMENT_SYMBOLS:
        col = f"{sym}_ppm"
        if col in pred_df.columns:
            col_vals = pred_df[col].apply(_safe_float).dropna()
            if len(col_vals) > 0:
                present_elems.append(sym)
    report["element_completeness_%"] = round(100 * len(present_elems) / len(ELEMENT_SYMBOLS), 2)
    report["present_elements"]       = present_elems

    report["row_count"] = len(pred_df)
    return report


def evaluate_quality(
    pred_df: pd.DataFrame,
    model: str,
    provider: str,
) -> dict:
    """Domain-aware quality assessment for extractions without ground truth.

    Goes beyond completeness to check scientific plausibility of extracted data.
    Returns a quality report with multiple sub-scores.
    """
    report: dict = {"model": model, "provider": provider}

    if pred_df.empty:
        report.update({
            "row_count": 0,
            "quality_score_%": 0.0,
            "schema_compliance_%": 0.0,
            "metadata_completeness_%": 0.0,
            "element_completeness_%": 0.0,
            "sample_id_quality_%": 0.0,
            "value_plausibility_%": 0.0,
            "internal_consistency_%": 0.0,
            "issues": ["Empty extraction — no rows produced"],
        })
        return report

    issues: list[str] = []

    # 1. Basic completeness (reuse existing logic)
    comp = evaluate_completeness(pred_df, model=model, provider=provider)
    report["row_count"] = comp["row_count"]
    report["schema_compliance_%"] = comp["schema_compliance_%"]
    report["metadata_completeness_%"] = comp["metadata_completeness_%"]
    report["element_completeness_%"] = comp["element_completeness_%"]
    report["present_elements"] = comp["present_elements"]
    report["null_metadata_fields"] = comp["null_metadata_fields"]

    # 2. Sample ID quality — are sample IDs present, unique, and non-trivial?
    sample_id_score = 0.0
    for col in ("sample_name", "sample_local_id", "Sample_ID", "sample_id"):
        if col in pred_df.columns:
            vals = pred_df[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            vals = vals[~vals.str.lower().isin(("nan", "none"))]
            if len(vals) > 0:
                n_unique = vals.nunique()
                n_total = len(vals)
                # Penalize if all same value or too few unique
                uniqueness_ratio = n_unique / n_total if n_total > 0 else 0.0
                coverage = len(vals) / len(pred_df) if len(pred_df) > 0 else 0.0
                sample_id_score = max(sample_id_score,
                                      min(1.0, uniqueness_ratio * 0.5 + coverage * 0.5))
                if uniqueness_ratio < 0.5 and n_total > 5:
                    issues.append(f"Low sample ID uniqueness in {col}: {n_unique}/{n_total}")
                break
    if sample_id_score == 0.0:
        issues.append("No sample identifiers found in any sample ID column")
    report["sample_id_quality_%"] = round(sample_id_score * 100, 2)

    # 3. Value plausibility — are element concentrations physically reasonable?
    plausibility_checks = 0
    plausibility_pass = 0

    # Major elements in wt% typically < 100% when converted to ppm < 1,000,000
    _MAJOR_ELEMENTS = {"fe", "si", "al", "ca", "mg", "na", "k", "ti", "mn", "p", "s"}
    # Trace elements typically < 10,000 ppm (with some exceptions)
    _TRACE_CEILING_PPM = 100000  # generous ceiling

    for sym in ELEMENT_SYMBOLS:
        col = f"{sym}_ppm"
        if col not in pred_df.columns:
            continue
        vals = pred_df[col].apply(_safe_float).dropna()
        if len(vals) == 0:
            continue

        # Exclude below-detection-limit values from plausibility checks.
        # Per USGS protocol, valid BDL representations are:
        #   -99999 = measured but below detection, LOD unknown
        #   Negative values (e.g., -0.5) = measured but below specific LOD of 0.5
        # Both are valid data markers, NOT real negative concentrations.
        vals = vals[(vals != BELOW_DETECTION_SENTINEL) & (vals >= 0)]
        if len(vals) == 0:
            # All values are BDL — element was measured but entirely below detection.
            # This is plausible (not an error), so count as pass.
            plausibility_checks += 1
            plausibility_pass += 1
            continue

        plausibility_checks += 1
        has_issue = False

        # Check: no unexpected negative concentrations
        # (negative LOD values already filtered out above)
        n_negative = (vals < 0).sum()
        if n_negative > 0:
            issues.append(f"{col}: {n_negative} unexpected negative values")
            has_issue = True

        # Check: concentrations not absurdly high
        max_val = vals.max()
        if sym in _MAJOR_ELEMENTS:
            if max_val > 1_000_000:
                issues.append(f"{col}: max={max_val:.0f} ppm (>100 wt%, impossible)")
                has_issue = True
        else:
            if max_val > _TRACE_CEILING_PPM:
                issues.append(f"{col}: max={max_val:.0f} ppm (suspiciously high for trace element)")
                has_issue = True

        if not has_issue:
            plausibility_pass += 1

    value_plausibility = (plausibility_pass / plausibility_checks
                          if plausibility_checks > 0 else 1.0)
    report["value_plausibility_%"] = round(value_plausibility * 100, 2)

    # 4. Internal consistency — do metadata values make sense together?
    consistency_checks = 0
    consistency_pass = 0

    first_row = pred_df.iloc[0]

    # Check: analytical method should be a known method
    method_val = _clean_val(first_row.get("analytical_method"))
    if method_val:
        consistency_checks += 1
        known_methods = {"epma", "la-icpms", "la-icp-ms", "icp-ms", "icpms",
                         "xrf", "sem-eds", "pixe", "inaa", "sims", "empa",
                         "la icpms", "electron microprobe", "laser ablation"}
        if any(km in method_val.lower() for km in known_methods):
            consistency_pass += 1
        else:
            issues.append(f"Unrecognized analytical method: '{method_val}'")

    # Check: mineral should be a known sulfide/oxide mineral
    mineral_val = _clean_val(first_row.get("mineral"))
    if mineral_val:
        consistency_checks += 1
        known_minerals = {"pyrite", "chalcopyrite", "sphalerite", "galena",
                          "arsenopyrite", "pyrrhotite", "magnetite", "hematite",
                          "bornite", "chalcocite", "covellite", "enargite",
                          "tetrahedrite", "tennantite", "molybdenite", "cassiterite",
                          "wolframite", "scheelite", "pentlandite", "millerite",
                          "stannite", "bournonite", "sulfosalt"}
        if any(km in mineral_val.lower() for km in known_minerals):
            consistency_pass += 1
        else:
            # Not necessarily wrong — could be a less common mineral
            issues.append(f"Mineral '{mineral_val}' not in common sulfide/oxide list (may be valid)")
            consistency_pass += 0.5  # partial credit

    # Check: country should be a real country name
    country_val = _clean_val(first_row.get("country"))
    if country_val:
        consistency_checks += 1
        if len(country_val) >= 2 and not country_val.replace(" ", "").isdigit():
            consistency_pass += 1
        else:
            issues.append(f"Suspicious country value: '{country_val}'")

    # Check: publication_date should be a plausible year (1990-2030)
    pub_date = _clean_val(first_row.get("publication_date"))
    if pub_date:
        consistency_checks += 1
        try:
            year = int(re.search(r"\d{4}", pub_date).group())
            if 1990 <= year <= 2030:
                consistency_pass += 1
            else:
                issues.append(f"Unusual publication year: {year}")
        except (ValueError, AttributeError):
            issues.append(f"Cannot parse publication date: '{pub_date}'")

    internal_consistency = (consistency_pass / consistency_checks
                            if consistency_checks > 0 else 0.5)
    report["internal_consistency_%"] = round(internal_consistency * 100, 2)

    # 5. Overall quality score — weighted combination
    quality = (
        0.20 * (comp["metadata_completeness_%"] / 100) +
        0.15 * (comp["element_completeness_%"] / 100) +
        0.20 * sample_id_score +
        0.25 * value_plausibility +
        0.20 * internal_consistency
    )
    report["quality_score_%"] = round(quality * 100, 2)
    report["issues"] = issues

    return report


def completeness_leaderboard(
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare multiple extraction DataFrames on completeness metrics.

    Args:
        results: mapping of model_key → prediction DataFrame
    """
    rows = []
    for key, df in results.items():
        provider, _, model = key.partition("/")
        r = evaluate_completeness(df, model=model or key, provider=provider)
        rows.append({
            "model":                    key,
            "rows":                     r["row_count"],
            "schema_compliance_%":      r["schema_compliance_%"],
            "metadata_completeness_%":  r["metadata_completeness_%"],
            "element_completeness_%":   r["element_completeness_%"],
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("metadata_completeness_%", ascending=False).reset_index(drop=True)
        df_out.index += 1
    return df_out


def cross_model_agreement(results: dict[str, pd.DataFrame]) -> dict:
    """For each metadata field, report % of models that gave the same answer.

    Returns a dict mapping field_name → {"agreement_%": float, "values": {model: value}}
    """
    agreement: dict = {}
    for field_name in METADATA_EVAL_FIELDS:
        values = {}
        for key, df in results.items():
            val = _clean_val(df.iloc[0].get(field_name)) if not df.empty else None
            values[key] = _normalise_str(val) if val else None

        non_null = [v for v in values.values() if v is not None]
        if not non_null:
            pct = 0.0
        else:
            most_common = max(set(non_null), key=non_null.count)
            pct = round(100 * non_null.count(most_common) / len(values), 1)

        agreement[field_name] = {
            "agreement_%": pct,
            "values": {k: v for k, v in values.items()},
        }
    return agreement


# ──────────────────────────────────────────────────────────────────────────────
# Multi-model comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_all_results(
    results: dict[str, ExtractionResult],
    evaluator: Evaluator,
) -> dict[str, BenchmarkReport]:
    """Evaluate multiple extraction results and return a report per model."""
    reports: dict[str, BenchmarkReport] = {}
    for key, result in results.items():
        print(f"\nEvaluating {key} ...")
        report = evaluator.evaluate(result)
        report.print_summary()
        reports[key] = report
    return reports


def leaderboard(reports: dict[str, BenchmarkReport]) -> pd.DataFrame:
    """Return a leaderboard DataFrame sorted by overall score."""
    rows = []
    for key, r in reports.items():
        rows.append({
            "model": key,
            "T1_metadata_%":    round(r.t1_metadata_score * 100, 2),
            "T2_numerical_%":   round(r.t2_numerical_score * 100, 2),
            "T3_structural_%":  round(r.t3_structural_score * 100, 2),
            "T4_null_%":        round(r.t4_null_score * 100, 2),
            "overall_%":        round(r.overall_score * 100, 2),
            "matched_rows":     r.matched_samples,
        })
    df = pd.DataFrame(rows).sort_values("overall_%", ascending=False)
    df = df.reset_index(drop=True)
    df.index += 1  # 1-based rank
    return df


def save_detailed_report(
    report: BenchmarkReport,
    output_path: str | Path,
) -> Path:
    """Save a detailed per-field Excel report for one model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        # Sheet 1: Summary
        summary_df = pd.DataFrame([report.to_dict()["scores"]])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Metadata fields
        meta_rows = [
            {
                "field": r.field_name,
                "predicted": r.predicted,
                "ground_truth": r.ground_truth,
                "score_%": round(r.score * 100, 1),
                "note": r.note,
            }
            for r in report.metadata_results
        ]
        if meta_rows:
            pd.DataFrame(meta_rows).to_excel(writer, sheet_name="Metadata", index=False)

        # Sheet 3: Numerical field summary (mean score per element across all samples)
        elem_scores: dict[str, list[float]] = {}
        for smr in report.sample_results:
            for fr in smr.numerical_results:
                elem_scores.setdefault(fr.field_name, []).append(fr.score)
        elem_rows = [
            {
                "element_field": col,
                "n_evaluated": len(scores),
                "mean_score_%": round(100 * sum(scores) / len(scores), 2),
            }
            for col, scores in sorted(elem_scores.items())
            if scores
        ]
        if elem_rows:
            pd.DataFrame(elem_rows).to_excel(writer, sheet_name="Elements", index=False)

        # Sheet 4: Per-sample scores
        sample_rows = [
            {
                "sample_name": smr.sample_name,
                "numerical_score_%": round(smr.numerical_score * 100, 2),
                "null_score_%": round(smr.null_score * 100, 2),
            }
            for smr in report.sample_results
        ]
        if sample_rows:
            pd.DataFrame(sample_rows).to_excel(writer, sheet_name="Samples", index=False)

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_ground_truth(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(str(path), dtype=object)
    else:
        df = pd.read_csv(str(path), dtype=object)
    # Standardise column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


# Columns to search for sample identifiers (priority order)
_SAMPLE_COLUMNS = ("sample_name", "sample_local_id", "Sample_ID", "sample_id")

_PREFIX_SEPS = ("-", ".", "@", "_", " ", "#")


def _get_sample_names_from_col(df: pd.DataFrame, col: str) -> list[str]:
    """Get non-empty sample names from a specific column.

    Applies light normalisation (strip whitespace, collapse internal spaces)
    and filters out rows that are clearly NOT sample identifiers (notes,
    reference standards, summary labels).
    """
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip().tolist()
    result = []
    for v in vals:
        if not v or v.lower() in ("nan", "none", ""):
            continue
        # Skip obvious non-sample rows (notes, descriptions, reference standards)
        if _is_non_sample_name(v):
            continue
        result.append(v)
    return result


# Patterns that indicate a value is a note/description, not a sample ID
_NON_SAMPLE_RE = re.compile(
    r"^\s*("
    r"note[s:]|"                       # "Notes:", "Note:"
    r"d\.l\.?\s*$|"                    # "D.L." alone
    r"b\.?d\.?l\.?\s*=|"              # "B.D.L. = ..."
    r"concentrations?\s+in\b|"         # "concentrations in ppm"
    r"- not\b|"                        # "- not available"
    r"numbers?\s+after|"               # "numbers after '<' denote..."
    r"\*\s*calculated|"                # "* calculated"
    r"mean|average|std|median|"        # summary labels
    r"min$|max$|minima|maxima|"
    r"detection limit|"
    r"all\s+values?\s+in\b"           # "all values in wt%"
    r")",
    re.IGNORECASE,
)


def _is_non_sample_name(name: str) -> bool:
    """Return True if name looks like a note/description rather than a sample ID."""
    if len(name) > 100:
        return True  # sample IDs are short
    if _NON_SAMPLE_RE.search(name):
        return True
    return False


def _normalise_sample_name(name: str) -> str:
    """Normalise a sample name for matching purposes.

    - Strip leading/trailing whitespace
    - Collapse internal whitespace around dashes: "X - 1" → "X-1"
    - Collapse multiple spaces to one
    """
    s = name.strip()
    # Normalise " - " → "-" (common in Chu_et_al style IDs)
    s = re.sub(r"\s*-\s*", "-", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _match_sample_names(
    gt_names: set[str], pred_names: set[str],
) -> tuple[set, set, set]:
    """Match sample names using exact + normalised + bidirectional prefix matching.

    Returns (matched, missing_from_gt, extra_in_pred).
    """
    matched = gt_names & pred_names
    missing = gt_names - pred_names
    extra   = pred_names - gt_names

    if not missing or not extra:
        return matched, missing, extra

    # Phase 1: Normalised exact match (handles whitespace/dash differences)
    norm_extra = {_normalise_sample_name(p): p for p in extra}
    still_missing = set()
    for gt_name in missing:
        gt_norm = _normalise_sample_name(gt_name)
        if gt_norm in norm_extra:
            matched.add(gt_name)
            extra.discard(norm_extra[gt_norm])
            del norm_extra[gt_norm]
        else:
            still_missing.add(gt_name)
    missing = still_missing

    # Phase 2: Bidirectional prefix matching (on normalised names)
    if missing and extra:
        # Rebuild normalised lookup for remaining extras
        norm_to_orig_extra = {}
        for p in extra:
            norm_to_orig_extra.setdefault(_normalise_sample_name(p), []).append(p)

        still_missing = set()
        for gt_name in missing:
            gt_norm = _normalise_sample_name(gt_name)
            found = False
            # Forward prefix: pred starts with GT name
            for p_norm, p_origs in list(norm_to_orig_extra.items()):
                if (p_norm.startswith(gt_norm) and len(p_norm) > len(gt_norm)
                        and p_norm[len(gt_norm)] in _PREFIX_SEPS):
                    matched.add(gt_name)
                    orig = p_origs[0]
                    extra.discard(orig)
                    p_origs.remove(orig)
                    if not p_origs:
                        del norm_to_orig_extra[p_norm]
                    found = True
                    break
            if found:
                continue
            # Reverse prefix: GT name starts with pred name
            for p_norm, p_origs in list(norm_to_orig_extra.items()):
                if (gt_norm.startswith(p_norm) and len(gt_norm) > len(p_norm)
                        and gt_norm[len(p_norm)] in _PREFIX_SEPS):
                    matched.add(gt_name)
                    orig = p_origs[0]
                    extra.discard(orig)
                    p_origs.remove(orig)
                    if not p_origs:
                        del norm_to_orig_extra[p_norm]
                    found = True
                    break
            if not found:
                still_missing.add(gt_name)
        missing = still_missing

    return matched, missing, extra


def _get_sample_names(df: pd.DataFrame) -> list[str]:
    """Get sample names from the first populated sample column."""
    for col in _SAMPLE_COLUMNS:
        vals = _get_sample_names_from_col(df, col)
        if vals:
            return vals
    return []


def _merge_matching_rows(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Merge multiple matching rows into one, taking the first non-null value per column."""
    matching = df[mask]
    if len(matching) == 1:
        return matching.iloc[0]
    # Merge: first non-null value per column across all matching rows
    result = matching.iloc[0].copy()
    for col in result.index:
        if pd.isna(result[col]) or (isinstance(result[col], str)
                                     and result[col].strip().lower() in ("nan", "none", "")):
            for _, row in matching.iloc[1:].iterrows():
                val = row[col]
                if pd.notna(val) and not (isinstance(val, str)
                                          and val.strip().lower() in ("nan", "none", "")):
                    result[col] = val
                    break
    return result


def _position_based_matching(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> list[tuple[int, int, float]]:
    """Match GT and prediction rows by position + element value validation.

    When sample names don't match (format differences, synthetic IDs),
    this aligns rows by position and validates each pair by checking
    whether their element values overlap.

    Strategy:
    1. If GT and pred have similar row counts, try 1:1 positional alignment
    2. If pred has many more rows (over-extraction), find the best-matching
       window within pred for each GT row using value fingerprinting
    3. Validate each match: at least 3 element values must be within 10%
       relative error for the match to count

    Returns list of (gt_idx, pred_idx, match_quality) tuples.
    """
    from .schema import ELEMENT_SYMBOLS

    # Get element columns present in both
    elem_cols = []
    for sym in ELEMENT_SYMBOLS:
        col = f"{sym}_ppm"
        if col in gt_df.columns and col in pred_df.columns:
            elem_cols.append(col)

    if not elem_cols:
        return []

    def _row_values(row: pd.Series) -> dict[str, float]:
        """Extract non-null, non-BDL numeric element values from a row."""
        vals = {}
        for col in elem_cols:
            v = _safe_float(row.get(col))
            if v is not None and v != BELOW_DETECTION_SENTINEL and v > 0:
                vals[col] = v
        return vals

    def _match_score(gt_vals: dict, pred_vals: dict) -> float:
        """Score how well two rows match based on element value overlap."""
        if not gt_vals or not pred_vals:
            return 0.0
        common = set(gt_vals.keys()) & set(pred_vals.keys())
        if not common:
            return 0.0
        matches = 0
        for col in common:
            gv, pv = gt_vals[col], pred_vals[col]
            if gv == 0:
                continue
            rel_err = abs(pv - gv) / abs(gv)
            if rel_err < 0.10:  # Within 10%
                matches += 1
        return matches / max(len(common), 1)

    gt_n = len(gt_df)
    pred_n = len(pred_df)
    results = []

    if pred_n <= gt_n * 3:
        # Similar size — try direct positional alignment
        # Also try with offset to handle header/skip rows
        best_offset = 0
        best_total_score = 0.0

        for offset in range(min(pred_n, 10)):
            total = 0.0
            for gi in range(min(gt_n, pred_n - offset)):
                pi = gi + offset
                if pi >= pred_n:
                    break
                gt_vals = _row_values(gt_df.iloc[gi])
                pred_vals = _row_values(pred_df.iloc[pi])
                total += _match_score(gt_vals, pred_vals)
            if total > best_total_score:
                best_total_score = total
                best_offset = offset

        # Apply best offset
        for gi in range(min(gt_n, pred_n - best_offset)):
            pi = gi + best_offset
            if pi >= pred_n:
                break
            gt_vals = _row_values(gt_df.iloc[gi])
            pred_vals = _row_values(pred_df.iloc[pi])
            score = _match_score(gt_vals, pred_vals)
            if score >= 0.3:  # At least 30% of shared elements match
                results.append((gi, pi, score))

    else:
        # Pred much larger — build value fingerprints and search
        # For each GT row, find the best matching pred row
        gt_fingerprints = [_row_values(gt_df.iloc[i]) for i in range(gt_n)]
        pred_fingerprints = [_row_values(pred_df.iloc[i]) for i in range(pred_n)]

        used_pred = set()
        for gi, gt_vals in enumerate(gt_fingerprints):
            if not gt_vals:
                continue
            best_pi = -1
            best_score = 0.3  # Minimum threshold
            for pi, pred_vals in enumerate(pred_fingerprints):
                if pi in used_pred:
                    continue
                score = _match_score(gt_vals, pred_vals)
                if score > best_score:
                    best_score = score
                    best_pi = pi
            if best_pi >= 0:
                results.append((gi, best_pi, best_score))
                used_pred.add(best_pi)

    return results


def _get_sample_row(df: pd.DataFrame, sample_name: str) -> Optional[pd.Series]:
    """Find the row matching a sample name, searching all sample columns.

    Priority: exact match > normalised exact > forward prefix > reverse prefix.
    Normalisation handles whitespace/dash differences.

    When multiple rows match the same name (e.g. multi-method data), merges them
    by taking the first non-null value per column.
    """
    name = sample_name.strip()
    name_norm = _normalise_sample_name(name)

    # Pass 1: exact match across all columns
    for col in _SAMPLE_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].astype(str).str.strip()
        mask = vals == name
        if mask.any():
            return _merge_matching_rows(df, mask)

    # Pass 1b: normalised exact match
    for col in _SAMPLE_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].astype(str).str.strip()
        norm_mask = vals.apply(lambda v: _normalise_sample_name(v) == name_norm)
        if norm_mask.any():
            return _merge_matching_rows(df, norm_mask)

    # Pass 2: forward prefix match (normalised)
    for col in _SAMPLE_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].astype(str).str.strip()
        prefix_mask = vals.apply(
            lambda v, n=name_norm: (
                _normalise_sample_name(v).startswith(n)
                and len(_normalise_sample_name(v)) > len(n)
                and _normalise_sample_name(v)[len(n)] in _PREFIX_SEPS
            )
        )
        if prefix_mask.any():
            return _merge_matching_rows(df, prefix_mask)

    # Pass 3: reverse prefix match (normalised)
    for col in _SAMPLE_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].astype(str).str.strip()
        rev_mask = vals.apply(
            lambda v, n=name_norm: (
                n.startswith(_normalise_sample_name(v))
                and len(n) > len(_normalise_sample_name(v))
                and n[len(_normalise_sample_name(v))] in _PREFIX_SEPS
            )
        )
        if rev_mask.any():
            return _merge_matching_rows(df, rev_mask)
    return None


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _clean_val(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s if s not in ("", "nan", "None", "NaN") else None


def _normalise_str(s: str) -> str:
    """Lowercase, collapse spaces, remove punctuation for fuzzy comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _string_similarity(a: str, b: str) -> float:
    """String similarity combining token Jaccard and character sequence matching.

    Uses the max of:
    - Jaccard token overlap (good for multi-word fields)
    - SequenceMatcher ratio (good for abbreviations and short strings)
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Token-level Jaccard
    set_a = set(a.split())
    set_b = set(b.split())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union if union > 0 else 0.0
    # Character-level sequence matching (handles abbreviations like EMPA/EPMA)
    from difflib import SequenceMatcher
    seq_ratio = SequenceMatcher(None, a, b).ratio()
    return max(jaccard, seq_ratio)


def _numerical_score(pred: float, gt: float, tol: float) -> float:
    """Return 1.0 if within tolerance, else partial score based on relative error."""
    if gt == 0:
        return 1.0 if abs(pred) <= ABSOLUTE_TOLERANCE else 0.0
    rel_err = abs(pred - gt) / abs(gt)
    if rel_err <= tol:
        return 1.0
    # Partial score: decay from 1.0 at tol to 0.0 at tol*20
    if rel_err >= tol * 20:
        return 0.0
    return max(0.0, 1.0 - (rel_err - tol) / (tol * 19))


def _mean_score(results: list[FieldResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)
