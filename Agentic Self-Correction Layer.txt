# Agentic Self-Correction Layer for geochem_benchmark

## Context

The extraction pipeline handles ~85-90% of supplementary tables correctly with rule-based parsing. But ~10-15% fail (0 rows or very low quality) due to unusual formats: wrong header detection, undetected transposition, unfamiliar sample ID columns, unit confusion, etc. Currently a human must diagnose each failure and add new rules. We need an automated LLM-powered correction loop that diagnoses failures and retries with structured hints — enabling the system to scale to millions of papers.

## Architecture

```
pipeline.run()
  → read_multiple_supplementary()           [existing, attempt 1]
  → quick_quality_check()                   [NEW: fast pass/fail]
  → IF failed:
      correction_loop()                     [NEW: max 2 iterations]
        → build_raw_table_preview()         [raw Excel → text for LLM]
        → LLM diagnosis → ParsingHints      [structured JSON output]
        → read_supplementary(..., hints)    [retry with overrides]
        → quick_quality_check()             [re-assess]
        → keep best result
```

## Files to Create

### 1. `geochem_benchmark/agentic_corrector.py` (NEW — ~300 lines)

Core module with:

**Data structures:**
- `ParsingHints` — typed dataclass with override fields:
  - `header_row: Optional[int]` — override `_find_header_row()`
  - `is_transposed: Optional[bool]` — override `_is_transposed()`
  - `sample_id_col: Optional[str]` — override `_detect_sample_id_col()`
  - `unit: Optional[str]` — override `_detect_unit_from_headers()`
  - `element_label_col: Optional[int]` — for transposed tables, which col has element names
  - `skip_sheets: Optional[list[str]]` — sheets to exclude
  - `target_sheets: Optional[list[str]]` — only process these sheets
  - `notes_from_llm: str` — LLM's reasoning
  - All fields default to `None` = use existing auto-detection
- `QuickQualityResult` — fast assessment (passed, row_count, element_count, failure_reasons)
- `CorrectionAttempt` — per-attempt record (hints used, quality achieved, tokens used)
- `CorrectionMetrics` — full tracking (initial vs final quality, all attempts, success flag)

**Functions:**
- `quick_quality_check(supp)` — O(1) checks: rows > 0, elements > 2, has sample IDs, no uniform values
- `build_raw_table_preview(path, max_rows=30)` — reads Excel with `header=None, dtype=str`, renders as pipe-delimited text with row indices, ~4000 chars max. For multi-sheet files, shows each sheet's first 15 rows.
- `build_diagnosis_prompt(raw_preview, failure_reasons, previous_hints)` — returns `(system, user)` prompt tuple
- `_parse_hints(llm_response)` → `ParsingHints` — validates LLM JSON against known fields, drops unknowns
- `correction_loop(client, paths, initial_supp, initial_quality, max_attempts=2, quality_threshold=30.0)` → `(best_supp, metrics)`

**Correction loop logic:**
```
best = initial_supp
for attempt in 1..max_attempts:
    preview = build_raw_table_preview(paths)
    system, user = build_diagnosis_prompt(preview, failures, prev_hints)
    hints = _parse_hints(client.complete_json(system, user))
    new_supp = read_multiple_supplementary(paths, hints=hints)
    new_quality = quick_quality_check(new_supp)
    if new_quality > best_quality: best = new_supp
    if new_quality.passed: break
    if new_quality == prev_quality: break  # no improvement, stop
return best, metrics
```

**Diagnosis prompt design:**
- System: "You are a geochemical data parsing expert. Diagnose why auto-parsing failed."
- User: raw table preview (first 30 rows with row numbers) + failure reasons + table_reader notes + previous hints (if retry) + JSON schema for ParsingHints
- Output: single JSON object with only the fields that need overriding

## Files to Modify

### 2. `table_reader.py` — Add `hints` parameter (backwards-compatible)

Add `hints: Optional[ParsingHints] = None` to:
- `read_supplementary(path, sheet_name, this_paper_deposit, hints)`
- `read_multiple_supplementary(paths, this_paper_deposit, hints)`
- `_read_single_sheet(path, raw_df, ..., hints)`
- `_load_excel(path, sheet_name, header_row_override)` — just the header_row override

Override pattern in `_read_single_sheet`:
```python
# Transposition
is_transposed = (hints.is_transposed if hints and hints.is_transposed is not None
                 else _is_transposed(raw_df))

# Header row (passed through _load_excel)
header_row = hints.header_row if hints and hints.header_row is not None else _find_header_row(raw)

# Sample ID
if hints and hints.sample_id_col:
    sample_id_col = hints.sample_id_col  # use directly
else:
    sample_id_col = _detect_sample_id_col(df)

# Unit
if hints and hints.unit:
    unit = hints.unit
else:
    unit = _detect_unit_from_headers(...)
```

When `hints is None` (default), all behavior is identical to current code.

### 3. `pipeline.py` — Integrate correction loop

- Add to `ExtractionPipeline.__init__`: `use_self_correction=True`, `correction_max_attempts=2`, `correction_quality_threshold=30.0`
- Add `correction_metrics: Optional[CorrectionMetrics] = None` to `ExtractionResult`
- In `run()`, after the existing `read_multiple_supplementary` call (line ~170-186):
  ```python
  # After supp is read (or failed):
  if self.use_self_correction:
      quality = quick_quality_check(supp)
      if not quality.passed:
          corrected, metrics = correction_loop(
              self.client, supplementary_paths,
              initial_supp=supp, initial_quality=quality, ...)
          if corrected: supp = corrected
  ```

### 4. `batch_runner.py` — Pass through flags, display metrics

- Add `use_self_correction` parameter to `run_batch()` and `run_batch_no_gt()`
- Pass through to `ExtractionPipeline`
- Display correction stats in batch summary: papers needing correction, success rate, quality improvement
- Save correction metrics in `batch_metrics.json`

### 5. `main.py` — CLI flags

Add to `_add_common_args()`:
- `--no-self-correction` — disable the loop
- `--correction-max-attempts` (default 2)
- `--correction-threshold` (default 30.0)

Self-correction is ON by default (opt-out with `--no-self-correction`).

## Cost Analysis

- LLM diagnosis: ~2K input + ~500 output tokens per attempt ≈ $0.01-0.02
- Max 2 attempts per failing paper ≈ $0.04 max
- ~10-15% of papers need correction → $0.40-0.60 per 100 papers
- At scale: $4-6 per 1,000 papers — negligible vs the existing Stage 1 metadata call

## Key Design Decisions

1. **Hints, not direct extraction**: LLM outputs ~500-token parsing hints; rule engine does the actual data extraction. This keeps numerical accuracy high.
2. **Bounded retries**: Max 2 attempts — if 2 diagnoses can't fix it, a 3rd rarely helps.
3. **Best-result selection**: Always keeps the highest-quality result across all attempts, not just the latest.
4. **Backwards compatible**: `hints=None` preserves all existing behavior. Self-correction is an additive layer.
5. **Cost-gated**: Only invokes LLM when quick_quality_check fails (0 rows, no elements, etc.).

## Verification Plan

1. Run on the 6 papers we fixed manually this session (Andersen, Benites 2022, Meng 2024, Wang 2025B/D, Yuan 2025) — temporarily revert those fixes and verify self-correction recovers them
2. Run on the 2 unfixable papers (Liu 2023, Das 2024) — verify correction loop exits gracefully after max attempts
3. Run full batch with `--no-self-correction` — verify identical results to current run
4. Run full batch WITH self-correction — verify improved quality for failing papers, no regression for passing papers
5. Check correction metrics in output: correction_rate, success_rate, quality improvement
