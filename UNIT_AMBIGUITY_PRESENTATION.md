# Unit Ambiguity: Column Name vs. Actual Values

**Presenter Script — Technical Deep Dive**  
**Duration:** 10-15 minutes  
**Audience:** Data stewards, ground truth curators, evaluator engineers

---

## OPENING (1 min)

There's a problem hiding in our schema that costs us **5–10 percentage points of accuracy** on papers with mixed analytical methods.

It looks like this:

```
fe_ppm: [2.19, 2190]
```

The question is: are these the same element at different concentrations, or a **unit mismatch**?

Spoiler: it's usually both. And our evaluator can't tell the difference.

---

## SECTION 1: The Schema Convention (2 min)

Our schema uses a **single column name for all element concentrations**, regardless of the actual units:

```
{symbol}_ppm
```

All 73 elements follow this pattern. Fe, Cu, Au, As — all stored in `{symbol}_ppm` columns.

This is intentional. It's domain-agnostic, provides a stable column interface, and works fine when all values in a paper use the same unit.

But here's the problem: **that assumption fails in ~30% of papers.**

---

## SECTION 2: Why Different Units? (2 min)

Different analytical methods naturally report in different units:

| Method | What it measures | Unit | Why |
|---|---|---|---|
| **EPMA** | Electron Microprobe Analysis | wt% | Measures weight fraction of the mineral phase analysed |
| **LA-ICP-MS** | Laser Ablation ICP-MS | ppm | Measures dissolved elemental ions in solution |
| **μ-XRF** | Micro X-Ray Fluorescence | atom% | Counts X-ray photons per element per atom |

All three are *correct*. They're measuring different things.

But when you store them in the **same spreadsheet**, you get:

| Sample | Mineral | Analytical Method | fe_ppm |
|---|---|---|---|
| K21-1 | pyrite | EPMA | 2.19 |
| K21-2 | sphalerite | LA-ICP-MS | 2190 |
| K21-3 | pyrite | μ-XRF | 0.52 |

Same column. Three different units. All legitimate.

---

## SECTION 3: The Evaluation Problem (3 min)

Now imagine this is **ground truth** and extraction produces the same:

```
Extraction:  fe_ppm = [2.19, 2190, 0.52]
Ground Truth: fe_ppm = [2.19, 2190, 0.52]
```

The evaluator compares row-by-row:
- Row 1: 2.19 vs 2.19 → relative error = 0% ✓ **Perfect match**
- Row 2: 2190 vs 2190 → relative error = 0% ✓ **Perfect match**
- Row 3: 0.52 vs 0.52 → relative error = 0% ✓ **Perfect match**

**Result: 100% accuracy on T2 (Numerical).**

This is **numerically correct**. But it's **semantically wrong** — we got lucky because extraction preserved the original units.

### The Real Problem

Now imagine extraction **mistakenly converted** the EPMA value from wt% to ppm:

```
Extraction:  fe_ppm = [21900, 2190, 0.52]  # Converted 2.19 wt% to ~21,900 ppm
Ground Truth: fe_ppm = [2.19, 2190, 0.52]
```

Row-by-row comparison:
- Row 1: 21,900 vs 2.19 → relative error = **99.99%** ✗ **Complete failure**
- Row 2: 2,190 vs 2,190 → relative error = 0% ✓ **Perfect match**
- Row 3: 0.52 vs 0.52 → relative error = 0% ✓ **Perfect match**

**Result: 66% accuracy on T2 (Numerical).**

One row — one **unit confusion** — and accuracy drops from 100% to 66%.

### At Scale

Some papers have 10–50% of rows with analytical method metadata that indicates unit should change. If extraction gets even **30% of those conversions wrong**:

- Paper accuracy drops: 95% → 75% (20 pp loss)
- Batch mean accuracy drops: 73.74% → 68% (5–7 pp loss)

---

## SECTION 4: Why This Happens (2 min)

The confusion arises because:

1. **Prompt says "preserve original units"** — so extraction leaves wt% as-is
2. **GT curator also preserved original units** — so far so good
3. **But evaluator has NO way to know what the original units are**
4. **So it assumes all values in `fe_ppm` are actually in ppm**
5. **Result: unit mismatch goes undetected**

Even worse: if extraction *does* convert, the evaluator can't tell if the conversion was:
- Correct (2.19 wt% → 21,900 ppm, handled properly)
- Incorrect (2.19 wt% → 2.19 ppm, conversion forgotten)
- Accidentally correct (values happen to match for wrong reasons)

---

## SECTION 5: Current Mitigation (2 min)

We have one rule to prevent conversion errors:

**All extraction prompts explicitly state:**

> "Keep values in ORIGINAL units — do NOT convert wt% to ppm or vice versa."

This works because:
- Python table_reader preserves units (no conversion happens)
- LLM extraction is instructed not to convert
- Result: Most papers preserve units correctly

**Trade-off:** This causes ~5–10 pp accuracy loss when GT curator or extraction does a conversion:
- If GT has wt%, extraction has ppm → evaluator sees unit mismatch
- If GT has mixed units, evaluator can't correlate which row uses which unit

---

## SECTION 6: Why Not Just Convert Everything to PPM? (2 min)

**Why not standardize all values to ppm during GT curation?**

Two reasons:

1. **Data loss / Scientific fidelity**
   - EPMA 2.19 wt% ≠ LA-ICP-MS 2.19 ppm
   - Converting wt% to ppm requires knowing the density and composition of the mineral
   - Without that context, the conversion introduces error
   - Example: pyrite has different density than sphalerite — same wt% ≠ same ppm

2. **Workflow complexity**
   - Papers often have ~50 supplementary sheets from different labs
   - Each lab reports in their native unit
   - To standardize, you'd need to track method per row, apply conversion formulas, handle edge cases (merged methods, unknown mineral composition)
   - This is error-prone and manual

**So non-conversion is the safe default.**

---

## SECTION 7: The Recommended Solution (3 min)

**Add a metadata column tracking the actual unit.**

New approach: Add `concentration_unit` column (or per-row column) that explicitly states the unit for each analytical method:

```
Sample | Analytical Method | fe_ppm | concentration_unit
K21-1  | EPMA             | 2.19   | wt%
K21-2  | LA-ICP-MS        | 2190   | ppm
K21-3  | μ-XRF            | 0.52   | atom%
```

**This enables:**

1. **Unit-aware evaluation** — Evaluator skips comparison for different units OR applies conversion rules
2. **Explicit tracking** — Each value's unit is documented
3. **Post-hoc correction** — Can audit which values would fail if conversion had occurred
4. **Error detection** — When extraction reports wt% in a ppm-only paper, it's flagged immediately

**Implementation:**

- Add `concentration_unit` as a categorical column (ppm, wt%, atom%, ppb, mg/kg)
- During evaluation: compare only rows with matching units
- Document valid unit transitions (e.g., EPMA → wt%, LA-ICP-MS → ppm)
- Report accuracy separately per unit to diagnose unit-specific errors

**Cost:** One extra column, minor evaluator logic, clearer error reporting.  
**Benefit:** Eliminates 5–10 pp accuracy loss, enables mid-calculation unit audits.

---

## SECTION 8: What To Do NOW (Best Practice) (2 min)

Until we implement `concentration_unit` tracking, follow this **curation protocol**:

### When Creating Ground Truth

1. **Check the source paper:** What unit does each analytical method report?
   - EPMA tables usually show "wt%" header
   - LA-ICP-MS usually shows "ppm" header
   - μ-XRF usually shows "atom%" header

2. **Inspect supplementary files:** What unit do the columns indicate?
   - Column header: "Fe (wt%)" → store as-is (don't convert)
   - Column header: "Fe (ppm)" → store as-is

3. **When mixing methods in ONE file:**
   - **BEST:** Ensure all columns use the same unit (usually ppm). Check source tables.
   - **ACCEPTABLE:** Keep original units, but document which rows use which units (use a pivot table or filter to verify)
   - **AVOID:** Mixed units with no documentation. This guarantees evaluator confusion.

4. **Test extraction:**
   - Run extraction on your GT paper
   - Check T2 numerical score
   - If <90% on numerical and you expect >95%, investigate unit mismatch
   - Flag in the paper registry notes

### Example: Good Curation

**Source paper:** Yuan et al. 2018 (EPMA only)
- All values in wt%
- Store all rows with `concentration_unit = wt%`
- OR convert standardly to ppm (if conversion formula is known)
- Result: Low unit confusion, high accuracy

### Example: Problematic Curation

**Source paper:** Mixed Chen et al. 2024 (EPMA + LA-ICP-MS)
- EPMA rows in wt%, LA-ICP-MS rows in ppm
- Store both in `fe_ppm` column without unit tracking
- Result: Evaluator confusion, ~5 pp accuracy loss
- **Fix:** Add `concentration_unit` column explicitly

---

## SECTION 9: Impact on Evaluation Metrics (2 min)

### How Unit Ambiguity Affects Each Tier

| Tier | Impact | Example |
|---|---|---|
| **T1 (Metadata)** | No impact | Deposit name, mineral, method are unaffected |
| **T2 (Numerical)** | **HIGH impact** | Unit mismatch → 99% error, fails tier |
| **T3 (Structural)** | Low impact | Sample coverage is structure, not units |
| **T4 (Null)** | Low impact | Null vs non-null unrelated to units |

**Bottom line:** T2 (40% of overall score) takes the full hit.

If T2 drops from 95% to 85% due to one row unit mismatch:
- Overall score: 73.74% → ~70% (3–4 pp loss on sample)

Across 28-paper batch, this adds up.

---

## SECTION 10: Audit and Detection (2 min)

**How to detect unit ambiguity in existing GT files:**

1. **Open ground truth file**
2. **Group by `analytical_method`** (or look at methods manually)
3. **For each method, scan the `{symbol}_ppm` column for outliers:**
   - EPMA data: typically 0.01–100 (wt%), rarely >500
   - LA-ICP-MS data: typically 0.1–10,000 (ppm), rarely <1
   - If EPMA rows have values >500, they might be in ppm (unit error)
   - If LA-ICP-MS rows have values <1, they might be in wt% (unit error)

4. **Cross-check with source paper:**
   - Open supplementary files
   - Read column headers (should say "wt%" or "ppm")
   - Compare to what you stored

5. **Document findings:**
   - If mixing units: note in paper_registry.py
   - If inconsistent: flag for curation correction
   - If suspicious: revert to source and re-enter

**Example audit (pseudocode):**

```python
# Detect unit anomalies
for method in gt_df['analytical_method'].unique():
    subset = gt_df[gt_df['analytical_method'] == method]
    fe_vals = subset['fe_ppm'].dropna()
    
    if method == 'EPMA':
        if (fe_vals > 500).any():
            print(f"⚠️  EPMA with Fe > 500 detected — may be in ppm, not wt%: {fe_vals[fe_vals > 500].values}")
    
    elif method in ['LA-ICP-MS', 'LA-ICPMS']:
        if (fe_vals < 1).any() and len(fe_vals) > 10:
            print(f"⚠️  LA-ICP-MS with Fe < 1 detected — may be in wt%, not ppm: {fe_vals[fe_vals < 1].values}")
```

---

## CLOSING (1 min)

**Summary:**

- **The problem:** Column names (`{symbol}_ppm`) don't encode actual units
- **The impact:** ~5–10 pp accuracy loss on mixed-method papers
- **The cause:** EPMA reports wt%, LA-ICP-MS reports ppm, we can't tell which is which
- **Today's workaround:** Extraction preserves original units, avoid conversion
- **Tomorrow's fix:** Add `concentration_unit` column, enable unit-aware evaluation
- **Your job now:** Audit GT files for unit mismatches, standardize where possible

---

## Q&A

**Q: Why not just always convert to ppm?**  
A: Requires knowing mineral composition and density — introduces error. Plus, it's a curation burden for 100+ papers.

**Q: Does this affect extraction accuracy?**  
A: Only if extraction *converts* units when it shouldn't. Our prompt says "preserve original units," so most extractions are correct. But if an LLM mistakes wt% for ppm and converts, we'd detect it as a big error in evaluation.

**Q: How many papers have mixed units?**  
A: ~30% of our 28 GT papers have both EPMA and LA-ICP-MS data. Not all are in the same file, but ~10 papers have it in one supplementary sheet.

**Q: Will the `concentration_unit` fix slow down evaluation?**  
A: No. It's one extra column comparison. Actually speeds up evaluation because we skip cross-unit comparisons.

**Q: Can we auto-detect the unit from the value magnitude?**  
A: Partially. EPMA is almost always 0.01–100 range. LA-ICP-MS is almost always 0.1–100,000. But there's overlap (both can have values between 1–100), so heuristics fail ~2–5% of the time.

**Q: Should I convert my GT file now?**  
A: Only if you're confident in the conversion formula and know the mineral composition. Otherwise, document which rows use which units and flag for later conversion.
