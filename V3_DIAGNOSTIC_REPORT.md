# V3 Pipeline Diagnostic Report

**Date:** March 27, 2026
**Scope:** Framework-level fixes and diagnostics for PDF-only extraction pipeline
**Batch Run:** `nogt_results_pdf_only_v3/` — 31 non-GT papers, PDF-only mode

---

## Critical Bugs Found and Fixed

### Bug 1: Plausibility Checker Broken by -9999 BDL Sentinel (CRITICAL)

**File:** `evaluator.py` lines 517-559

**Root cause:** The -9999 below-detection-limit sentinel introduced in this session was flagged as "negative concentration" by the plausibility checker, causing every element with BDL values to fail the check. This artificially cratered plausibility scores (e.g., Ye et al dropped from 100% to 30%).

**Fix:** Filter out `BELOW_DETECTION_SENTINEL` values before running negative-value checks. All-BDL elements (entirely below detection) are counted as plausible passes, not failures.

**Impact:** Plausibility scores corrected. Ye et al: 30% -> 100%.

```python
# Before: all -9999 counted as negative
n_negative = (vals < 0).sum()

# After: exclude BDL sentinel from negative check
vals = vals[vals != BELOW_DETECTION_SENTINEL]
n_negative = (vals < 0).sum()
```

---

### Bug 2: T2/T4 Evaluator Didn't Handle -9999 Semantics (CRITICAL)

**File:** `evaluator.py` — `_eval_numerical_row()` and `_eval_null_row()`

**Root cause:** The T2 numerical evaluator compared -9999 values as real concentrations, computing huge relative errors. The T4 null evaluator didn't distinguish between "pred says BDL" vs "pred hallucinated a value" for unmeasured elements.

**Fix:** Added BDL-aware scoring:

| GT | Pred | T2 Score | Rationale |
|----|------|----------|-----------|
| -9999 (BDL) | -9999 (BDL) | 1.0 | Both correctly identified BDL |
| -9999 (BDL) | None | 0.5 | Pred missed the BDL marker |
| -9999 (BDL) | real value | 0.3 | Partial — at least detected presence |
| real value | -9999 (BDL) | 0.0 | Wrong — pred says BDL but GT has value |
| None (unmeasured) | -9999 (BDL) | 0.0 (T4) | Wrong — pred claims BDL for unmeasured element |

---

### Bug 3: Contradictory Unit Conversion Instructions in Prompts (HIGH)

**File:** `prompts.py`

**Root cause:** Three prompt stages gave contradictory instructions about wt% vs ppm conversion:
- Stage 2 (line 315): "Use ppm for ALL values. If the table reports wt%, convert: wt% * 10000 = ppm" **[WRONG]**
- Stage 4 (line 378): "The '_ppm' suffix is just the column name — it does NOT mean you should convert to ppm" **[CORRECT]**
- Vision (line 468): "Keep values in their ORIGINAL units as printed. Do NOT convert" **[CORRECT]**

**Evidence:** Ground truth stores values as-reported without unit normalisation:
- Yuan (LA-ICPMS): Fe max=57,682 → real ppm
- Xia (EPMA): Zn=54-66, S=32-33 → these are wt% stored as-is
- He (LA-ICPMS): Zn max=595,266 → real ppm (sphalerite ~60 wt% Zn)

**Fix:** All prompt stages now consistently say: "Keep values in their ORIGINAL units exactly as printed — do NOT convert between units."

---

### Bug 4: Hardcoded wt%-to-ppm Conversion in Table Reader (HIGH)

**File:** `table_reader.py`

**Root cause:** Two conversion paths multiplied values by 10,000:
1. `read_supplementary()` (line 679): auto-converts detected wt% columns → produced wrong values for supplementary files
2. `_pivot_transposed()` (line 1413): hardcoded `val * 10000` for wt% elements in transposed tables
3. Self-correction path (pipeline.py line 1064): `convert_units=True` when hint says "wt%"

**Fix:**
1. `read_supplementary()`: Disabled auto-conversion. Now logs detected unit but stores values as-reported.
2. `_pivot_transposed()`: Removed `val * 10000` multiplication. Values stored as-reported.
3. Self-correction path: Changed `convert_units=True` to `convert_units=False`.

---

### Bug 5: Cross-Backend Duplicate Samples Not Fully Deduped (MEDIUM)

**File:** `pipeline.py` — `_direct_pdf_table_extraction()` dedup logic

**Root cause:** Dedup key is `(sample_name, analytical_method)`. When Backend A detects method as "EPMA" and Backend B detects method as `None` for the same table, `(name, "EPMA") != (name, None)` — so both are kept as separate samples.

Example: Xia extraction produced 205 rows when GT has 127. ~80 were cross-backend duplicates.

**Fix:** Added fallback dedup: when checking for an existing key, also check `(name, None)` as an alternative key. If a name exists with any method (or no method), the duplicate is caught and the version with more element columns is kept.

```python
# Before: only exact key match
if name and key in seen_name_method:

# After: also check name-only and cross-method matches
existing_key = None
if name and key in seen_name_method:
    existing_key = key
elif name and (name, None) in seen_name_method:
    existing_key = (name, None)
```

---

## V3 Batch Run Results (In Progress)

The batch is running on 31 non-GT papers in PDF-only mode. Results so far:

| # | Paper | Samples | Quality | Meta | Elem | SampleID | Plausibility | Time |
|---|-------|---------|---------|------|------|----------|--------------|------|
| 1 | 2004_Ono_etal | 33 | 88.1% | 100% | 21% | 100% | 100% | 171s |
| 2 | 2011_Ye_etal | 237 | 50.1%* | 92% | 27% | 0% | 30%* | 173s |

*Ye plausibility was 30% under old evaluator. With BDL fix applied, re-evaluation gives 100% plausibility. Quality rises to 67.6%.

### Observed Patterns

**Low SampleID (0%):** Ye et al produced 237 rows with no sample names. Root cause: PDF tables have numeric data without clear sample identifier columns. The `_detect_sample_id_col()` function requires column headers matching patterns like "sample", "spot", "analysis" etc. Tables with generic or missing headers (just element symbols) don't match.

**This is a framework issue:** Many geochemistry papers, especially older ones, don't label sample ID columns clearly. The pipeline needs a fallback strategy: use the first non-element, non-numeric column as sample ID if no named candidate is found.

---

## V3 Batch Results — Re-Evaluated with Fixed Evaluator

**18 papers processed (batch still running), re-evaluated with BDL-aware plausibility:**

| Metric | Value |
|--------|-------|
| Papers processed | 18 |
| Successful | 15 (83%) |
| Failed (0 rows) | 3 |
| Total samples extracted | 1,008 |
| **Mean Quality** | **79.4%** |
| Mean Metadata | 94.4% |
| Mean Elements | 18.8% |
| Mean Sample ID | 75.4% |
| **Mean Plausibility** | **91.2%** |

### Per-Paper Results (Fixed Evaluator)

| # | Paper | Samples | Quality | Meta | Elem | SampleID | Plausibility |
|---|-------|---------|---------|------|------|----------|--------------|
| 1 | 2004_Ono | 33 | 88.1% | 100% | 21% | 100% | 100% |
| 2 | 2011_Ye | 237 | 67.6% | 92% | 27% | 0% | 100% |
| 3 | 2016_Bonnet | 16 | 85.4% | 100% | 16% | 100% | 92% |
| 4 | 2016_George | 11 | 84.9% | 100% | 22% | 91% | 94% |
| 5 | 2019_Maurer | 11 | 81.9% | 92% | 14% | 82% | 100% |
| 6 | 2020_Frenzel | 13 | 81.0% | 77% | 4% | 100% | 100% |
| 7 | 2021_Sun | 57 | 69.1% | 100% | 25% | 91% | 39% |
| 8 | 2022_Culqui | 0 | FAIL | - | - | - | - |
| 9 | 2022_Frenzel | 11 | 82.5% | 100% | 14% | 77% | 100% |
| 10 | 2022_Liu | 16 | 84.2% | 92% | 30% | 81% | 100% |
| 11 | 2022_OrtizBenavente | 0 | FAIL | - | - | - | - |
| 12 | 2023_Fougerouse | 16 | 84.7% | 92% | 8% | 100% | 100% |
| 13 | 2024_Frenzel | 60 | 83.8% | 100% | 25% | 82% | 94% |
| 14 | 2024_Graupner | 29 | 73.2% | 85% | 22% | 95% | 56% |
| 15 | 2024_Marquez-Zavalia | 18 | 66.6% | 100% | 11% | 0% | 100% |
| 16 | 2024_Minhas | 178 | 81.7% | 92% | 22% | 82% | 94% |
| 17 | 2024_MousaviMotlagh | 302 | 76.6% | 92% | 22% | 49% | 100% |
| 18 | 2024_Niu | 0 | FAIL | - | - | - | - |

### Plausibility Fix Impact

Papers where the BDL fix improved plausibility scores:

| Paper | Old Plausibility | Fixed Plausibility | Quality Change |
|-------|------------------|-------------------|----------------|
| 2020_Frenzel | 0% | **100%** | 56% -> 81% |
| 2011_Ye | 30% | **100%** | 50% -> 68% |
| 2016_Bonnet | 67% | **92%** | 79% -> 85% |
| 2024_Minhas | 12% | **94%** | 61% -> 82% |
| 2024_Graupner | 12% | **56%** | 62% -> 73% |

---

## Architecture Notes

### Current Backend Order (all run simultaneously)
1. pdftext (Marker's fast text layer)
2. Text-based whitespace parsing
3. Docling (ML-based)
4. Marker (surya-based)
5. MinerU (YOLO-based)
6. Camelot (stream/lattice)
7. pdfplumber (grid-based)

All results merged, deduplicated by `(sample_name, method)`, highest element-count version kept.

### Key Design Decisions
- **No unit conversion** — values stored as-reported, matching GT convention
- **-9999 for BDL** — below-detection-limit values explicitly marked, not lost as blanks
- **Run all backends** — maximises recall, dedup handles precision
- **No paper-specific hacks** — all fixes are general framework improvements

---

## Files Modified

| File | Change | Severity |
|------|--------|----------|
| `evaluator.py` | BDL-aware plausibility checker, T2/T4 scoring | Critical |
| `prompts.py` | Consistent no-conversion unit instructions | High |
| `table_reader.py` | Removed wt%->ppm conversion, BDL sentinel | High |
| `pipeline.py` | Improved cross-backend dedup, disabled convert_units | Medium |

---

**Report will be updated with full v3 batch results upon completion.**
