# USGS Integration Changelog

**Date:** 2026-04-06
**Source:** USGS instructions from Garth (USGS), CMiO-MIN database standard operating procedure
**Scope:** All PDF extraction — applies universally to every paper, not paper-specific

---

## Summary of Changes

Eight USGS-mandated requirements were audited against the codebase. Three were already implemented, five had gaps. All gaps are now fixed. Changes affect 6 files across the pipeline.

| Requirement | Before | After | File(s) Changed |
|---|---|---|---|
| BDL → -9999 (no LOD) | Implemented | No change | — |
| BDL → negative LOD (e.g., -0.5) | **Missing** — all BDL → -9999 | Extracts numeric LOD from `<0.5` patterns | `table_reader.py` |
| N/A = not analyzed = blank | **Bug** — "n/a" treated as BDL | Fixed: "n/a", "na", "not analyzed" → None | `table_reader.py` |
| `analysis_id` field | **Missing** from schema | Added to `METADATA_COLUMNS`, `SampleRow` | `schema.py`, `pipeline.py` |
| One mineral per row | **Not enforced** | Post-processing splits grouped minerals | `pipeline.py` |
| Mineral from sheet names | **Missing** | Infers from sheet labels (e.g., "Chalcopyrite") | `table_reader.py` |
| Mineral from analysis_id | **Missing** | Parses abbreviations (cpy→chalcopyrite, sph→sphalerite) | `table_reader.py`, `pipeline.py` |
| Hofstra 2021 classification | Taxonomy existed, not enforced | Post-processing validation + prompt enforcement | `pipeline.py`, `prompts.py` |

---

## 1. Below-Detection-Limit: Negative LOD Values

**USGS Rule:** When a specific detection limit is reported (e.g., `<0.5 ppm`), store the NEGATIVE of that limit (-0.5), not -9999.

**Before:**
```python
_safe_float("<0.5")  # → -9999.0  (LOD information lost)
_safe_float("<0.01") # → -9999.0
```

**After:**
```python
_safe_float("<0.5")  # → -0.5   (LOD preserved)
_safe_float("<0.01") # → -0.01  (LOD preserved)
_safe_float("bdl")   # → -9999.0 (no specific LOD given)
_safe_float("n.d.")  # → -9999.0 (no specific LOD given)
```

**New function:** `_extract_detection_limit(val)` parses `<X` patterns and returns `-X`.

**File:** `table_reader.py`, lines 1556-1575

---

## 2. N/A = Not Analyzed = Blank (Critical Bug Fix)

**USGS Rule:** "N/A" and "not analyzed" mean the element was NOT measured — leave blank (None). This is semantically distinct from BDL.

**Bug:** `"n/a"` and `"na"` were in `_BDL_STRINGS`, causing them to be stored as -9999 (implying the element was measured but below detection, which is wrong).

**Before:**
```python
_safe_float("n/a")          # → -9999.0  ← WRONG: implies element was measured
_safe_float("not analyzed") # → None     (not matched, fell through to None)
```

**After:**
```python
_safe_float("n/a")          # → None  ← CORRECT: not analyzed
_safe_float("na")           # → None
_safe_float("not analyzed") # → None
_safe_float("not measured") # → None
_safe_float("n.a.")         # → None
```

**New set:** `_NOT_ANALYZED_STRINGS` — checked before `_BDL_STRINGS` in `_safe_float()`.

**File:** `table_reader.py`, lines 1496-1510

---

## 3. Three-Tier Sample Identification

**USGS Rule:** Three distinct ID fields must be tracked:
- `sample_name` — core identifier (e.g., "201003232")
- `sample_local_id` — local identifier from the paper
- `analysis_id` — full analysis string from supplementary (e.g., "5-2002063521cpy1-1.d")

**Before:** `analysis_id` did not exist in the schema.

**After:**
- Added `analysis_id` to `METADATA_COLUMNS` in `schema.py`
- Added `analysis_id: Optional[str] = None` to `SampleRow` in `schema.py`
- USGS post-processing auto-populates `analysis_id` from `sample_local_id` when not set
- All LLM prompts updated to request `analysis_id` extraction

**Files:** `schema.py`, `pipeline.py`, `prompts.py`

---

## 4. One Mineral Per Row (No Grouped Minerals)

**USGS Rule:** Each line of data can only be assigned to one individual mineral. Grouping minerals (e.g., "chalcopyrite, sphalerite, galena") is strictly prohibited.

**Before:** Grouped mineral strings like `"pyrite, chalcopyrite"` could pass through unchecked.

**After:** `_usgs_postprocess()` in `pipeline.py` detects comma-separated minerals and splits them:
- 1 row with `mineral="pyrite, chalcopyrite"` → 2 rows, one with `mineral="pyrite"`, one with `mineral="chalcopyrite"`
- All element data is duplicated to both rows (same analytical data, different mineral assignment)
- Split count logged in extraction notes

**File:** `pipeline.py`, function `_usgs_postprocess()`

---

## 5. Mineral Inference from Sheet Names

**USGS Rule:** When supplementary files have sheets named after minerals (e.g., tabs named "Chalcopyrite", "Galena", "Sphalerite"), use those names to assign the mineral for all rows in that sheet.

**Before:** `_read_excel_all_sheets()` did not assign mineral from sheet names. If no mineral column existed in the table, the mineral was left blank.

**After:**
- `_infer_mineral_from_label(label)` — new function that matches sheet/table labels against:
  1. Full mineral names from `MINERAL_TAXONOMY` (longest-first to avoid "pyrite" matching in "chalcopyrite")
  2. Common abbreviations from `_MINERAL_ABBREVIATIONS` dict
- Called in `_read_excel_all_sheets()` when no mineral column is detected
- Sets `supp.inferred_mineral` for that sheet's data

**File:** `table_reader.py`, lines 1101-1165

---

## 6. Mineral Inference from Analysis ID Abbreviations

**USGS Rule:** Analysis ID strings often contain mineral abbreviations: "5-2002063521cpy1-1.d" → chalcopyrite (from "cpy").

**New function:** `infer_mineral_from_analysis_id(analysis_id)` with 40+ mineral abbreviations:

| Abbreviation | Mineral |
|---|---|
| cpy, cp, ccp | chalcopyrite |
| sph, sp, sl | sphalerite |
| gal, gn | galena |
| py, pyr | pyrite |
| po | pyrrhotite |
| apy, asp | arsenopyrite |
| bn, bor | bornite |
| mol, moly | molybdenite |
| mt, mag | magnetite |
| hem | hematite |
| ... | (40+ total entries) |

Applied in `_usgs_postprocess()`: when a sample has no mineral assignment (neither from metadata nor table column), the function checks `analysis_id`, `sample_local_id`, and `sample_name` for embedded mineral abbreviations.

**Files:** `table_reader.py` (function definition), `pipeline.py` (applied in post-processing)

---

## 7. Hofstra 2021 Classification Enforcement

**USGS Rule:** `deposit_environment`, `deposit_group`, and `deposit_type` MUST use the CMMI Classification scheme from Hofstra et al. 2021. Alternative classification systems are not permitted.

**Before:** Taxonomy existed in `knowledge_base.py` and was included in prompts, but:
- No post-extraction validation
- Prompts did not explicitly forbid alternative classification systems

**After:**
- `_usgs_postprocess()` validates `deposit_environment` against `DEPOSIT_TAXONOMY` and logs a warning if the value doesn't match any Hofstra 2021 category
- Metadata prompt (Stage 1) now explicitly states: "deposit_environment, deposit_group, deposit_type MUST follow the Hofstra et al. 2021 CMMI classification scheme. Do NOT use any alternative classification system."
- Added instruction 8: "Do NOT hallucinate or assume field values without strong confidence."

**Files:** `pipeline.py`, `prompts.py`

---

## 8. LLM Prompt Updates (All 6 Stages)

All prompt stages updated with USGS protocol:

| Prompt Stage | Key Updates |
|---|---|
| Stage 0 (Paper Intelligence) | No change needed (extracts methodology, not data) |
| Stage 1 (Metadata) | Hofstra 2021 mandate explicit; single mineral per row; anti-hallucination rule |
| Stage 2 (Table Filter) | Three-tier BDL rules; single mineral per row; analysis_id field |
| Stage 3 (One-Shot) | Same as Stage 2 |
| Stage 4 (PDF Table) | USGS BDL protocol with negative LOD; analysis_id; mineral rules |
| Stage 5 (Vision) | USGS BDL protocol with negative LOD; analysis_id; mineral rules |
| Shared (_SAMPLE_FORMAT_INSTRUCTIONS) | Updated for all stages using shared template |

---

## 9. Evaluator Updates

**File:** `evaluator.py`

**Change:** Plausibility checker updated to handle negative LOD values. Previously, only -9999 was excluded from the "no negative concentrations" check. Now all negative values from BDL (both -9999 and specific negative LODs like -0.5) are properly excluded.

**Before:**
```python
vals = vals[vals != BELOW_DETECTION_SENTINEL]  # Only excluded -9999
# → vals like -0.5 would trigger "negative concentration" flag
```

**After:**
```python
vals = vals[(vals != BELOW_DETECTION_SENTINEL) & (vals >= 0)]  # Exclude all BDL representations
# → Both -9999 and -0.5 correctly excluded from plausibility checks
```

---

## Test Results

All 8 integration tests pass:

```
BDL semantics: PASSED
Detection limit: PASSED
Mineral from label: PASSED
Mineral from analysis_id: PASSED
Schema: PASSED
Mineral split: PASSED (1 row -> 2 rows)
Mineral inference: PASSED (mineral=chalcopyrite)
Analysis ID: PASSED (analysis_id=ABC-123)
Hofstra validation: PASSED

=== ALL 8 USGS INTEGRATION TESTS PASSED ===
```

---

## Files Modified

| File | Lines Changed | What |
|---|---|---|
| `table_reader.py` | ~120 lines | BDL semantics, mineral inference functions, N/A fix |
| `schema.py` | ~5 lines | Added `analysis_id` to schema |
| `pipeline.py` | ~80 lines | `_usgs_postprocess()`, imports, integration |
| `prompts.py` | ~60 lines | All 6 stages updated with USGS rules |
| `evaluator.py` | ~10 lines | Negative LOD handling in plausibility |
| `USGS_instructions` | — | Source document (read-only reference) |

---

---

## 10. Proper Unit Conversion (wt% → ppm, ppb → ppm)

**Requirement:** The `_ppm` column must contain values in ppm. If the source reports wt% or ppb, convert:
- wt% → ppm: multiply by 10,000
- ppb → ppm: divide by 1,000

**Before:** Pipeline stored values as-reported regardless of source units. Fe = 5.03 wt% was stored as `fe_ppm: 5.03` (wrong — should be 50,300 ppm).

**After:**
- `_read_single_sheet()` now always converts wt% and ppb to ppm
- `read_pdf_table()` default changed: `convert_units=True`
- Per-column detection handles mixed-unit tables (some cols wt%, some ppm)
- New `_detect_ppb_columns()` and `_convert_ppb_to_ppm()` functions

**Files:** `table_reader.py`

---

## 11. Dual Deposit Classification (Hofstra 2021 + Paper's Original)

**Requirement:** Always use Hofstra et al. 2021 classification. When the paper uses a different scheme, include BOTH — Hofstra as primary, paper's original preserved for reference.

**New schema fields:**
- `deposit_classification_source` — Always "Hofstra et al. 2021". If paper differs, notes it: "Hofstra et al. 2021 (paper uses: <their scheme>)"
- `deposit_type_original` — The deposit type exactly as stated by the paper's authors, verbatim. null if same as Hofstra.

**Files:** `schema.py`, `prompts.py`

---

## 12. Sample Order Preservation

**Requirement:** Samples must be extracted in the same order they appear in the paper, so human evaluators can compare row-by-row.

**Bug found and fixed:** Multi-backend dedup sorted tables by quality (`sorted(... key=element_col_count, reverse=True)`) which shuffled samples from different pages. Fixed to process tables in **page order** instead, with in-place replacement when a higher-quality duplicate is found.

**Changes:**
- `pipeline.py` `_direct_pdf_table_extraction()`: Changed `table_order` from quality-sorted to page-order-sorted. Dedup now replaces in-place (same array position) instead of appending, preserving the first-seen position.
- All 6 LLM prompts: Added explicit "## SAMPLE ORDER (CRITICAL)" section: "Output samples in EXACTLY the same order they appear in the paper tables. Do NOT sort, group, or reorder."
- Supplementary reader: Already preserves DataFrame row order (verified: no `.sort_values()` on sample data)
- Multi-source merge: Preserves supplementary order first, PDF additions appended at end

**Files:** `pipeline.py`, `prompts.py`

---

## 13. Corrected Ground Truth Files

**Location:** `ground_truth_corrected/`
**Script:** `correct_ground_truth.py`

All 28 ground truth files corrected with:

| Correction | Count |
|---|---|
| BDL strings → -9999 | 17,913 |
| Specific LOD strings → negative LOD | 3,712 |
| N/A strings → None | 0 (none found) |
| wt% → ppm conversions | 4,865 |

### Major Element wt% Conversions by Paper

Papers with major elements stored in wt% (now converted to ppm):

| Paper | Elements Converted | Value Count |
|---|---|---|
| He et al. 2024 | Ti, Mg, P, Al, Na, K, Ca | 594 |
| Yang et al. 2022 | Ti, Cr, Mn | 1,013 |
| Sun et al. 2023 | Ti, Cr, Mn | 705 |
| Xia et al. 2024 | S, Ti, Mn, Fe | 389 |
| Bertrandsson Erlandsson et al. 2022 | Cr, Mn | 434 |
| Chu et al. 2022 | Al, Ti, Cr, Mg | 266 |
| Zhao et al. 2024 | Al, Cr, Mn | 355 |
| Zhang et al. 2022 | S, Mn, Fe | 208 |
| ... | ... | ... |

### Negative LOD Papers

| Paper | Count | Examples |
|---|---|---|
| Chu et al. 2022 | 2,535 | `<0.673` → `-0.673` |
| Zhang et al. 2024 | 582 | `<0.184` → `-0.184` |
| Zhao et al. 2024 | 539 | `<0.14` → `-0.14` |
| Zhang et al. 2022 | 44 | `<0.01` → `-0.01` |
| Sun et al. 2024 | 12 | `<0.01` → `-0.01` |

---

## Backward Compatibility

- **Schema change:** 3 new columns added to `ALL_COLUMNS` (now 212 columns): `analysis_id`, `deposit_classification_source`, `deposit_type_original`. Existing results will have these as blank.
- **BDL values:** `<0.5` now stored as `-0.5` instead of `-9999`. More informative, not breaking.
- **N/A fix:** "n/a" now correctly → blank instead of -9999. Fixes semantic error.
- **Unit conversion:** Extraction now converts wt%→ppm and ppb→ppm. Old extractions storing wt% as-is are wrong.
- **Corrected GT:** Use `ground_truth_corrected/` for evaluation. Original `ground_truth/` preserved as-is for reference.
- **Mineral splits:** Grouped minerals produce more rows. Each row is valid.
- **Sample order:** Preserved as-reported — no change from previous behavior.
