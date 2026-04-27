# Geochem Benchmark

An LLM benchmarking framework for extracting structured geochemical data from research papers — at scale.

---

## Overview

Research papers in economic geology report trace-element geochemical data across dozens of samples. Curating this data into a standardised database requires reading the full paper — not just copying table numbers, but also comprehending the geological context, analytical methods, deposit classification, and sample metadata scattered throughout the text.

This framework:

1. **Registers** 28 papers with human-curated ground truths via a paper registry
2. **Extracts** structured data from each paper's PDF + supplementary Excel/CSV using LLMs
3. **Evaluates** each extraction against its ground truth across 4 tiers
4. **Aggregates** results into a cross-paper leaderboard with per-paper and mean scores

The target schema is a fixed **209-column** format covering deposit metadata, sample metadata, 73 elements (concentration + detection limit), and geographic/provenance fields.

---

## Input / Output

| | Description |
|---|---|
| **Input 1** | Research paper PDF (full text including methods, sampling, geology sections) |
| **Input 2** | Supplementary data file(s) (`.xlsx`, `.xls`, `.csv`, or `.zip`) with per-sample element concentrations |
| **Output** | Excel/CSV file with one row per analytical spot, conforming to the 209-column schema |
| **Ground truth** | Human-curated Excel file in the same 209-column schema |

### Why both inputs are needed

The **supplementary table** contains the raw numerical measurements (element concentrations per sample) but almost no metadata.

The **paper PDF** contains everything else:
- What deposit type and geological environment the samples came from
- What mineral was analysed (e.g. sphalerite, pyrite, chalcopyrite)
- The full instrument description, laboratory name, operating conditions, and reference standards
- Which rows in the supplementary table belong to *this* paper vs. comparison data from cited references
- Which rows are statistical summaries (MEAN, STD, MINIMA, MAXIMA) that must be excluded

An LLM must comprehend both files to produce a complete, correct output.

---

## Project Structure

```
geochem_benchmark/
├── schema.py              # 209-column schema — pydantic models, element list, column definitions
├── pdf_reader.py          # PDF -> structured text with auto-detected sections + page scoring
├── pdf_vision.py          # Render PDF pages as images for LLM vision API extraction
├── table_reader.py        # Excel/CSV -> cleaned DataFrame with smart row filtering & unit detection
├── knowledge_base.py      # Geochemical ontology (deposit taxonomy, mineral classification, method standardization)
├── prompts.py             # Multi-stage LLM prompt templates with domain knowledge injection
├── llm_clients.py         # Unified Claude / OpenAI / Gemini client interface (text + vision)
├── pipeline.py            # Multi-stage extraction orchestration (LLM metadata + Python tables + vision)
├── agentic_corrector.py   # LLM-powered self-correction for failed table extractions
├── evaluator.py           # 4-tier ground truth comparison and benchmark reports
├── paper_registry.py      # Maps 28 ground truth files to their PDF + supplementary data
├── batch_runner.py        # Batch processing of all papers with aggregate scoring
├── main.py                # CLI entry point
├── requirements.txt
│
├── ground_truth/          # 28 human-curated xlsx files (209-column schema)
├── data/                  # 101 research paper PDFs
│   └── Spreadsheets/      # 127 supplementary data files (xlsx, csv, zip)
└── batch_results/         # Output directory for batch runs
```

---

## Installation

```bash
pip install pdfplumber pandas openpyxl pydantic xlrd

# Vision-based PDF extraction (renders pages as images for LLM vision API):
pip install PyMuPDF>=1.24.0

# Faster PDF text extraction (optional, falls back to pdfplumber):
pip install pdftext

# Install whichever LLM providers you want to benchmark:
pip install anthropic          # Claude
pip install openai             # GPT-5.2
pip install google-genai       # Gemini
```

### API keys

Set API keys as environment variables or in a `.env` file in the project root:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # Claude
export OPENAI_API_KEY="sk-..."           # OpenAI
export GOOGLE_API_KEY="AIza..."          # Gemini
```

---

## Usage

All commands assume you are in the **parent** directory of the package (so `geochem_benchmark/` is a subdirectory). On Windows, set `PYTHONIOENCODING=utf-8` before running.

Every extraction command supports **agentic self-correction** — when table parsing fails (0 rows, missing elements), the LLM automatically diagnoses the issue and retries with structured hints. This is ON by default across all commands. Disable with `--no-self-correction`.

### CLI Commands

#### 1. Discover all papers — see what's available

Scan the entire `data/` directory and show every paper with its PDF, supplementary files, and ground truth status:

```bash
python -m geochem_benchmark.main discover
```

Shows a categorised report:
- **GT + Supplementary** — full benchmark papers (28)
- **GT, no Supplementary** — PDF-only with evaluation possible
- **No GT + Supplementary** — extraction only, quality assessment
- **No GT, no Supplementary** — PDF-only, no evaluation

Add `--verbose` to see file paths for each paper.

#### 2. Run a single paper by ID — the easiest way

Extract any paper by its ID (from `discover`). Auto-resolves PDF, supplementary files, and ground truth. If GT exists, evaluates; otherwise runs quality assessment:

```bash
# Paper with ground truth — extracts + evaluates
python -m geochem_benchmark.main run-paper Yuan_et_al_2018

# Paper without ground truth — extracts + quality assessment
python -m geochem_benchmark.main run-paper 2024_Das_etal

# PDF-only paper (no supplementary) — metadata extraction only
python -m geochem_benchmark.main run-paper 2020_Frenzel_etal

# Partial ID matching works
python -m geochem_benchmark.main run-paper Das

# Use a different provider/model
python -m geochem_benchmark.main run-paper Yuan_et_al_2018 \
  --provider openai --model gpt-4o

# Use Claude for text extraction, Gemini for vision API
python -m geochem_benchmark.main run-paper Xia_et_al_2024 \
  --provider claude --vision-provider gemini

# Use Claude for text, GPT-5.2 for vision with explicit model
python -m geochem_benchmark.main run-paper Xia_et_al_2024 \
  --provider claude --vision-provider openai --vision-model gpt-5.2
```

Outputs saved to `paper_results/` (or `--output-dir`):
- `extraction_{paper_id}.xlsx` — extracted data
- `report_{paper_id}.xlsx` — evaluation report (if GT exists)
- `quality_{paper_id}.json` — quality metrics (if no GT)

Self-correction triggers automatically if table parsing fails.

#### 3. Extract with explicit paths

For papers not in the registry, or to override file paths:

```bash
# With supplementary files
python -m geochem_benchmark.main extract \
  --pdf data/2018_Yuan_etal.pdf \
  --supplementary data/Spreadsheets/2018_Yuan_etal.xlsx \
  --provider claude \
  --output results/yuan_output.xlsx

# Multiple supplementary files (merged on sample name)
python -m geochem_benchmark.main extract \
  --pdf data/2024B_Chen_etal.pdf \
  --supplementary data/Spreadsheets/2024B_Chen_etal_EMPA.xlsx \
                  data/Spreadsheets/2024B_Chen_etal_LAIPCMS.xlsx \
  --provider claude \
  --output results/chen_output.xlsx

# PDF-only (no supplementary) — metadata extraction only
python -m geochem_benchmark.main extract \
  --pdf data/2020_Frenzel_etal.pdf \
  --provider claude \
  --output results/frenzel_output.xlsx
```

#### 4. Batch with ground truth — evaluate all registered papers

```bash
python -m geochem_benchmark.main batch \
  --provider claude \
  --model claude-sonnet-4-6 \
  --output-dir batch_results/
```

Process only specific papers:

```bash
python -m geochem_benchmark.main batch \
  --papers Yuan_et_al_2018 Chen_et_al_2024 \
  --provider claude \
  --output-dir batch_results/
```

Include PDF-only papers alongside those with supplementary:

```bash
python -m geochem_benchmark.main batch \
  --provider claude \
  --include-pdf-only \
  --output-dir batch_results/
```

Process **only** PDF-only papers (skip those with supplementary):

```bash
python -m geochem_benchmark.main batch \
  --provider claude \
  --pdf-only \
  --output-dir batch_results/
```

Outputs in `batch_results/`:

| File | Contents |
|---|---|
| `extraction_{paper_id}.xlsx` | Raw extraction per paper |
| `report_{paper_id}.xlsx` | Per-field evaluation breakdown |
| `batch_summary.xlsx` | Ranked per-paper scores |
| `batch_metrics.json` | Machine-readable aggregate + per-paper metrics |

#### 5. Batch without ground truth — process all discovered papers

Discovers and processes all papers in `data/` that have supplementary files but no ground truth:

```bash
python -m geochem_benchmark.main batch-nogt \
  --provider claude \
  --model claude-sonnet-4-6 \
  --output-dir nogt_results/
```

Include PDF-only papers alongside those with supplementary:

```bash
python -m geochem_benchmark.main batch-nogt \
  --provider claude \
  --include-pdf-only \
  --output-dir nogt_results/
```

Process **only** PDF-only papers (skip those with supplementary):

```bash
python -m geochem_benchmark.main batch-nogt \
  --provider claude \
  --pdf-only \
  --output-dir nogt_results/
```

Outputs include `nogt_batch_summary.xlsx`, `nogt_batch_metrics.json`, per-paper quality scores, and self-correction statistics.

#### 6. Benchmark multiple LLMs on a single paper

```bash
# With ground truth
python -m geochem_benchmark.main benchmark \
  --pdf data/2018_Yuan_etal.pdf \
  --supplementary data/Spreadsheets/2018_Yuan_etal.xlsx \
  --ground-truth ground_truth/Yuan_et_al_2018.xlsx \
  --providers claude openai gemini \
  --output-dir results/

# Without ground truth (completeness + cross-model agreement)
python -m geochem_benchmark.main benchmark \
  --pdf data/2018_Yuan_etal.pdf \
  --supplementary data/Spreadsheets/2018_Yuan_etal.xlsx \
  --providers claude openai gemini \
  --output-dir results/

# PDF-only benchmark (no supplementary)
python -m geochem_benchmark.main benchmark \
  --pdf data/2020_Frenzel_etal.pdf \
  --providers claude openai \
  --output-dir results/
```

#### 7. Evaluate an existing extraction (no LLM call)

```bash
# With ground truth
python -m geochem_benchmark.main eval \
  --prediction results/yuan_output.xlsx \
  --ground-truth ground_truth/Yuan_et_al_2018.xlsx \
  --provider claude \
  --model claude-sonnet-4-6 \
  --output-dir eval_results/

# Completeness-only (no ground truth)
python -m geochem_benchmark.main eval \
  --prediction results/yuan_output.xlsx \
  --provider claude \
  --output-dir eval_results/
```

#### 8. Discovery commands

```bash
# Comprehensive scan of all papers (recommended)
python -m geochem_benchmark.main discover
python -m geochem_benchmark.main discover --verbose

# List only registered papers (with ground truths)
python -m geochem_benchmark.main list-papers

# List only discovered non-GT papers
python -m geochem_benchmark.main list-nogt-papers

# List available LLM models
python -m geochem_benchmark.main models
```

### Common flags

These flags are shared across `extract`, `benchmark`, `batch`, `batch-nogt`, and `run-paper`:

| Flag | Description | Available in |
|---|---|---|
| `--provider` | LLM provider: `claude`, `openai`, or `gemini` | all |
| `--model` | Model ID (uses provider default if omitted) | all |
| `--vision-provider` | Separate LLM provider for vision API calls (default: same as `--provider`) | all |
| `--vision-model` | Model ID for vision API calls (default: provider default) | all |
| `--no-vision` | Disable vision-based PDF page extraction (vision is ON by default) | all |
| `--no-self-correction` | Disable agentic self-correction for failed table extractions | all |
| `--no-tool-calling` | Use plain JSON output instead of Anthropic tool use | all |
| `--llm-table-filter` | Use LLM-assisted table row filtering (slower, for complex tables) | all |
| `--table-detector` | PDF table detection: `auto`, `docling`, `camelot`, `pdfplumber` | all |
| `--verbose` / `-v` | Enable debug logging | all |
| `--include-pdf-only` | Include PDF-only papers alongside those with supplementary | `batch`, `batch-nogt` |
| `--pdf-only` | Process ONLY papers with no supplementary files | `batch`, `batch-nogt` |

### Handling missing supplementary files

Papers without supplementary data files are handled gracefully at every level:

| Scenario | `run-paper` | `extract` | `batch` | `batch-nogt` |
|---|---|---|---|---|
| PDF + Supplementary | Full extraction | Full extraction | Default | Default |
| PDF only (no supp) | Metadata-only extraction | Omit `--supplementary` | `--include-pdf-only` | `--include-pdf-only` |
| Only PDF-only papers | N/A (single paper) | N/A | `--pdf-only` | `--pdf-only` |
| Missing PDF | Error + exit | Error + exit | Skipped | Skipped |

### Python API

```python
from geochem_benchmark.llm_clients import create_client
from geochem_benchmark.pipeline import ExtractionPipeline
from geochem_benchmark.evaluator import Evaluator
from geochem_benchmark.batch_runner import run_batch, run_batch_no_gt
from geochem_benchmark.paper_registry import discover_all_papers, resolve_paper_by_id
from pathlib import Path

# --- Discover all papers ---
papers = discover_all_papers(Path("geochem_benchmark"))
for p in papers:
    print(f"{p.id}: PDF={p.has_pdf}, Supp={p.has_supplementary}, GT={p.has_ground_truth}")

# --- Look up a specific paper by ID ---
info = resolve_paper_by_id(Path("geochem_benchmark"), "Yuan_et_al_2018")
print(f"PDF: {info.pdf_path}, Supp: {info.supplementary_paths}, GT: {info.ground_truth_path}")

# --- Single paper extraction ---
client = create_client(provider="claude", model="claude-sonnet-4-6")
pipeline = ExtractionPipeline(llm_client=client)  # self-correction + vision ON by default

result = pipeline.run(
    pdf_path="data/2018_Yuan_etal.pdf",
    supplementary_paths=["data/Spreadsheets/2018_Yuan_etal.xlsx"],
)

# --- Use a separate vision provider ---
vision = create_client(provider="gemini", model="gemini-3-flash-preview")
pipeline = ExtractionPipeline(llm_client=client, vision_client=vision)

print(f"Extracted {result.n_samples} samples")
result.to_excel("output.xlsx")

# Check if self-correction was used
if result.correction_metrics and result.correction_metrics.correction_needed:
    cm = result.correction_metrics
    print(f"Self-correction: {cm.initial_row_count} -> {cm.final_row_count} rows")

# --- PDF-only extraction (no supplementary) ---
result = pipeline.run(
    pdf_path="data/2020_Frenzel_etal.pdf",
    supplementary_paths=[],  # or omit
)

# --- Evaluation against ground truth ---
evaluator = Evaluator("ground_truth/Yuan_et_al_2018.xlsx")
report = evaluator.evaluate(result)
report.print_summary()

# --- Full batch run (with ground truth) ---
batch = run_batch(
    client=client,
    project_root=Path("geochem_benchmark"),
    output_dir=Path("batch_results"),
)
print(f"Mean overall: {batch.aggregate_scores()['mean_overall_%']:.1f}%")

# --- Non-GT batch run (all papers, quality assessment only) ---
nogt_batch = run_batch_no_gt(
    client=client,
    project_root=Path("geochem_benchmark"),
    output_dir=Path("nogt_results"),
)
print(f"Mean quality: {nogt_batch.aggregate_quality()['mean_quality_score_%']:.1f}%")

# --- Disable self-correction ---
pipeline_no_corr = ExtractionPipeline(
    llm_client=client,
    use_self_correction=False,
)

# --- Control table detector backend ---
from geochem_benchmark.tabledetector import TableDetectorBackend

# Auto-selection with fallback chain
pipeline_auto = ExtractionPipeline(
    llm_client=client,
    table_detector_backend=TableDetectorBackend.AUTO,  # default
)

# Force specific backend
pipeline_docling = ExtractionPipeline(
    llm_client=client,
    table_detector_backend=TableDetectorBackend.DOCLING,
)

result = pipeline_docling.run(
    pdf_path="data/2018_Yuan_etal.pdf",
    supplementary_paths=["data/Spreadsheets/2018_Yuan_etal.xlsx"],
)
# When both supplementary and PDF present, pipeline:
# 1. Extracts from supplementary table
# 2. Extracts from PDF using Docling (all backends if AUTO)
# 3. Merges results: dedup on sample_name, supplementary priority
```

---

## Architecture

### Three-Stage Hybrid Pipeline with Multi-Source Extraction

**Stage 1: PDF Metadata Extraction**
```
PDF text
    |
    v
[pdf_reader.py]
Section detection, full text extraction
    |
    v
[Pipeline Stage 1: LLM]
Metadata extraction: deposit_name, mineral, analytical_method, 
instrument, laboratory, standards, country, publication_date
```

**Stage 2: Supplementary Table Processing**
```
Supplementary file(s)
    |
    v
[table_reader.py]
Auto-detection: headers, sample IDs, element columns, units
Row filtering: exclude MEAN/STD/MINIMA/MAXIMA
Metadata per-row: mineral, method, deposit from columns
    |
    v
[Optional: agentic_corrector.py]
IF parsing failed: LLM diagnoses failure, retry with hints
```

**Stage 3: Multi-Source PDF Table Extraction**
```
PDF + Supplementary both available?
    |
    +---> Extract supplementary (Stage 2 above)
    |
    +---> Extract PDF tables using tiered fallback chain:
    |
    |     [1. pdftext]   -- Marker's fast text layer (~2s, primary)
    |           |
    |           v
    |     [2. Vision LLM] -- Render pages as images, send to vision API
    |           |            (fires when pdftext finds < 85% of expected samples)
    |           |            (uses --vision-provider if set, else same as --provider)
    |           v
    |     [3. Docling/Camelot/pdfplumber] -- Fallback backends
    |           |                           (only if pdftext finds < 5 samples)
    |           v
    |     [4. Text-only LLM fallback] -- Raw page text sent to LLM
    |
    v
[Multi-Source Merge]
Combine supplementary + PDF samples:
  - Cross-table deduplication by (sample_name, method)
  - Supplementary values have priority (more curated)
  - PDF fills gaps in supplementary (more comprehensive)
  - Vision supplements (merges new sample names, doesn't replace)
  - Keep all unique samples (no data loss)
```

**Final Merge and Validation**
```
[pipeline.py]
Merge all sources: LLM metadata + supplementary + PDF tables
Per-row overrides applied
    |
    v
[knowledge_base.py]
Post-validation: taxonomy normalization, mineral classification,
method standardization, enrichment with ontology
    |
    v
209-column output (Excel or CSV)
```

**Why multi-source instead of exclusive?**

Old approach: If supplementary exists, ignore PDF. If supplementary missing, extract from PDF.
- Problem: Supplementary files are often incomplete (missing columns, limited samples)
- Result: Data loss when supplementary lacks what PDF tables contain

New approach: Always extract from BOTH when both available.
- Supplementary provides curated, structured data (highest quality)
- PDF provides comprehensive coverage (all available samples and context)
- Merge intelligently to get best of both: curated accuracy + comprehensive coverage
- Different PDF table formats detected better by different backends — use all three

**Result: +14.5 percentage point accuracy improvement** (v4 59.21% → v5 73.74%)

### Agentic Self-Correction

When rule-based table parsing fails (~10-15% of papers), the `agentic_corrector.py` module automatically diagnoses and recovers:

1. **Quick quality check** — O(1) pass/fail: rows > 0, elements >= 3, sample IDs present
2. **Raw preview** — reads the Excel/CSV with `header=None, dtype=str` and renders as text for LLM inspection
3. **LLM diagnosis** — an expert prompt asks the LLM to identify which auto-detection step failed (wrong header row, undetected transposition, unusual sample ID column, wrong unit, etc.)
4. **Structured hints** — the LLM returns a `ParsingHints` JSON object with only the fields that need overriding
5. **Retry** — table_reader re-reads the file with hints applied, overriding specific auto-detection steps
6. **Best-result selection** — keeps the highest-quality result across all attempts

The LLM outputs ~500-token parsing hints, not the data itself — the rule engine does all numerical extraction with perfect fidelity.

| Aspect | Detail |
|---|---|
| Max attempts | 2 LLM calls per failing paper |
| Cost per paper | ~$0.02-0.04 (diagnosis only, not data extraction) |
| Cost at scale | ~$4-6 per 1,000 papers |
| Trigger | Only when `quick_quality_check` fails |
| Override | `--no-self-correction` flag |

Self-correction is **ON by default**. Disable with `--no-self-correction`.

### Ontology-Anchored Extraction

The `knowledge_base.py` module provides a structured geochemical ontology that is:
- **Injected into LLM prompts** — giving the model deposit taxonomy trees, mineral classifications, and method standardization rules
- **Used for post-validation** — filling gaps in LLM output using domain knowledge (e.g., inferring deposit_group from deposit_type) without overwriting LLM-provided values

### Paper Registry

The `paper_registry.py` maps all 28 ground truth files to their corresponding PDF and supplementary data files. This enables:
- Automated batch processing of the full corpus
- Automatic zip extraction for papers with archived supplementary data
- Tracking of papers with known limitations (PDF-only, docx supplements, etc.)

---

## Table Detection Backends

### Overview

When extracting from PDF tables, the framework provides three detection backends with different trade-offs:

| Backend | Approach | Speed | Accuracy | Best For | Requirements |
|---|---|---|---|---|---|
| **Docling** | ML-based document understanding with visual & textual analysis | Slow (~40-50s/PDF) | Highest on complex layouts | Irregular/complex table structures | `docling>=0.18.0`, torch, transformers |
| **Camelot** | Stream or lattice mode for borderless/whitespace-aligned tables | Fast (~1-2s/PDF) | Good for sparse data | Minimal borders, sparse data | Built-in |
| **pdfplumber** | Grid-based table detection for bordered tables | Very fast (<1s/PDF) | Good for clean grids | Clear table borders, structured layouts | Built-in |

### Usage

```bash
# Auto-selection (smart selection with fallback, default)
# Tries backends sequentially, uses first successful result
python -m geochem_benchmark.main extract --pdf data.pdf --table-detector auto

# Force specific backend
python -m geochem_benchmark.main extract --pdf data.pdf --table-detector docling
python -m geochem_benchmark.main extract --pdf data.pdf --table-detector camelot
python -m geochem_benchmark.main extract --pdf data.pdf --table-detector pdfplumber

# Batch with multi-backend extraction (all three simultaneously)
python -m geochem_benchmark.main batch-nogt --table-detector auto --output-dir batch_results/
```

### Multi-Backend Extraction (When supplementary + PDF both present)

When a paper has both supplementary files AND a PDF, the pipeline runs **all three backends simultaneously**:

```python
# Pseudo-code
tables = []
for backend in [Docling, Camelot, pdfplumber]:
    try:
        tables.extend(extract_tables_from_pdf(pdf_path, backend))
    except:
        pass  # Failure in one backend doesn't affect others

# Result: Combined list of all tables detected by all three backends
# Merge with supplementary data: dedup by sample_name, 
# supplementary priority, PDF fills gaps
```

**Why all three?** Different PDF layouts suit different detectors:
- Complex/irregular layouts → Docling's ML excels
- Sparse/borderless data → Camelot's stream mode excels  
- Bordered tables → pdfplumber's grid detection excels

Running all three and combining results maximizes coverage without excluding valid data.

---

## Evaluation Tiers

| Tier | Weight | What is measured | Metric |
|---|---|---|---|
| **T1 -- Metadata** | 30% | Paper-level fields: deposit type, mineral, instrument model, laboratory, operating conditions, standards, country, etc. (13 fields) | String similarity (exact match + Jaccard token overlap) |
| **T2 -- Numerical** | 40% | Element concentrations for all matched samples | Relative error <= 5% = full credit; partial credit with decay beyond that |
| **T3 -- Structural** | 15% | Correct sample coverage: sample names predicted vs ground truth | F1 of predicted vs ground truth unique sample name sets |
| **T4 -- Null** | 15% | Correctly leaving unmeasured elements as NULL (not hallucinating values) | Per-element null accuracy across matched samples |

Overall score = weighted average of T1-T4.

### Sample matching

The evaluator uses a multi-strategy matching approach:
1. **Exact match** across multiple ID columns (`sample_name`, `sample_local_id`)
2. **Bidirectional prefix matching** for spot-level suffixes (e.g., `YK94-17` matches `YK94-17-1`)
3. **Column combination search** — tries all GT column vs pred column combinations, preferring the pairing with the most matched rows
4. **Row merging** — when multiple pred rows match one sample name (e.g., different analytical methods), merges by taking the first non-null value per column

---

## Benchmark Results (v5 — Multi-Source Extraction)

**Overall Mean Accuracy: 73.74%** across 26 papers (multi-source: supplementary + PDF multi-backend extraction)

This represents a **+14.53 percentage point improvement** over v4 baseline (59.21%), validating the multi-source extraction strategy.

### Performance by Tier

| Tier | Weight | Mean Score | Notes |
|---|---|---|---|
| **T1 -- Metadata** | 30% | 53.62% | LLM extracts from paper prose + supplementary columns |
| **T2 -- Numerical** | 40% | **84.28%** | Python reads directly from spreadsheets; PDF fills supplementary gaps |
| **T3 -- Structural** | 15% | 70.75% | Sample coverage: multi-source merging improves detection |
| **T4 -- Null** | 15% | **88.87%** | Low hallucination: Python extraction is conservative |

### What Changed (v4 → v5)

| Change | Mechanism | Impact |
|---|---|---|
| Multi-source extraction | Extract from BOTH supplementary AND PDF simultaneously | Tables that supplementary missed now found in PDF |
| Multi-backend PDF detection | All three backends (Docling, Camelot, pdfplumber) run in parallel | Different table formats detected better by different backends |
| Intelligent sample merging | Dedup on `sample_name`, supplementary takes priority, PDF fills gaps | No data loss, all unique samples retained |
| Changed `elif` to `if` | Supplementary extraction no longer blocks PDF extraction | PDF tables now always attempted, not skipped |
| **Accuracy improvement** | Combined effect of above | **+14.53 pp** (59.21% → 73.74%) |

### Top-Performing Papers (v5)

| Rank | Paper | Score | Samples | T2 (Numerical) |
|---|---|---|---|---|
| 1 | Yuan et al. 2018 | **95.38%** | 85/85 | 99.5% |
| 2 | Sun et al. 2024 | 88.97% | 67/67 | 100.0% |
| 3 | Soster et al. 2023 | 88.30% | 63/63 | 99.5% |
| 4 | He et al. 2024 | 87.93% | 102/102 | 98.9% |
| 5 | Wu et al. 2024 | 87.58% | 266/438 | 99.4% |

### Coverage Statistics

- **Papers processed**: 26 (with both supplementary + ground truth)
- **Total samples extracted**: 8,909
- **Ground truth references**: 5,076
- **Samples matched**: 3,509
- **Extraction coverage**: 69.1%

---

## What Makes This Task Hard for LLMs

| Challenge | Description |
|---|---|
| **Row filtering** | The supplementary table mixes sample data from this paper with comparison data from cited papers. LLMs must include only rows where Reference = "this paper" |
| **Summary row exclusion** | Rows labelled MEAN, STD, MINIMA, MAXIMA are statistical summaries — not samples — and must be excluded |
| **Scattered metadata** | Instrument model may be in methods section, sampling description in another, deposit type in the abstract. All must be linked to every row |
| **NULL discipline** | Only a subset of 73 element columns have data. The rest must remain NULL. LLMs that hallucinate non-zero values are penalised in T4 |
| **Domain knowledge** | Correctly classifying a deposit as "Basin hydrothermal" vs "Magmatic hydrothermal" requires geological expertise |
| **Unit heterogeneity** | Different analytical methods report in different units (EMPA in wt%, LA-ICP-MS in ppm). Conversion must be applied correctly |
| **Per-sample metadata** | Mineral, method, and deposit can vary per row in multi-method papers. Not all rows share the same metadata |
| **Multi-file merging** | Papers may have separate supplementary files for different methods (EMPA + LA-ICP-MS) that must be merged by sample key |
| **Verbatim fidelity** | Operating conditions and instrument descriptions must be copied exactly from the paper, not paraphrased |

---

## Schema Reference

### Metadata columns (45)

| Group | Fields |
|---|---|
| Deposit | `deposit_name`, `deposit_local_id`, `deposit_environment`, `deposit_group`, `deposit_type`, `primary_commodities`, `secondary_commodities`, `all_commodities`, `deposit_source` |
| Sample identity | `sample_uid`, `sample_name`, `sample_local_id`, `feature_type`, `feature_name`, `feature_uid`, `top_depth_m`, `bottom_depth_m` |
| Sample description | `sample_deposit_relation`, `sample_type`, `sampling_method`, `material_class`, `material_class_comments` |
| Stratigraphy | `province`, `strat_unit_name`, `strat_unit_uid`, `strat_grouping` |
| Material | `earth_material_group`, `earth_material_qualifier`, `earth_material`, `metamorphic_grade` |
| Mineralogy | `mineral`, `paragenetic_stage`, `mode_of_occurrence`, `texture`, `color`, `alteration`, `sample_description`, `associated_minerals` |
| Analysis | `sample_preparation`, `analytical_method`, `instrument_type_model`, `laboratory_location/if reported`, `operating_conditions/if reported`, `standards_used/if reported` |

### Element columns (146 = 73 elements x 2)

Each element has `{symbol}_ppm` and `{symbol}_detection_limit`. Elements covered:

`ag al as au b ba be bi br ca cd ce cl co cr cs cu dy er eu f fe ga gd ge hf hg ho in ir k la li lu mg mn mo na nb nd ni os p pb pd pr pt rb re rh ru s sb sc se si sm sn sr ta tb te th ti tl tm u v w y yb zn zr`

### Provenance columns (18)

`analysis_datetime`, `publication_date`, `sample_source`, `submitter`, `country`, `state`, `deposit_longitude_wgs84`, `deposit_latitude_wgs84`, `sample_longitude_wgs84`, `sample_latitude_wgs84`, `sample_easting`, `sample_northing`, `sample_utm_zone`, `sample_location_description`, `location_source`, `location_accuracy`, `comments`, `last_update`

---

## Adding a New Paper

1. Place the PDF in `data/` and supplementary files in `data/Spreadsheets/`
2. Create a ground truth Excel file in `ground_truth/` using the 209-column schema
3. Add an entry to `paper_registry.py`:

```python
{
    "id": "Smith_et_al_2024",
    "ground_truth": "Smith_et_al_2024.xlsx",
    "pdf": ["2024_Smith_etal.pdf"],
    "supplementary": ["2024_Smith_etal_supp.xlsx"],
},
```

4. Run batch evaluation:

```bash
python -m geochem_benchmark.main batch \
  --provider claude \
  --model claude-sonnet-4-6 \
  --paper-ids Smith_et_al_2024 \
  --output-dir batch_results/
```

---

## Vision-Based PDF Extraction

For PDF-only papers (no supplementary data), vision-based extraction renders data-dense pages as images and sends them to an LLM vision API. This unlocks tables that text extraction cannot parse (~65% of PDFs).

### How it works

1. **Page scoring** — `pdf_reader.score_pages_for_data()` ranks pages by geochemical data density (element symbols, units, numeric content)
2. **Rendering** — `pdf_vision.render_data_pages()` renders top 5 pages as PNG images via PyMuPDF (auto-reduces DPI if images too large)
3. **Vision API** — Images sent to the vision LLM with a specialised prompt for table extraction
4. **Merge** — Vision results supplement text extraction (adds new sample names, doesn't replace existing)

### When vision fires

Vision is **not a fallback** — it supplements text extraction:
- When pdfplumber finds no structured tables AND text parsing finds few samples
- When text extraction finds < 85% of expected samples (estimated from paper intelligence)
- Disabled with `--no-vision`

### Separate vision provider

Use `--vision-provider` to route vision calls to a different LLM than text extraction. This is useful because vision-optimised models (Gemini 3 Flash, GPT-5.2) may differ from the best text extraction model:

```bash
# Claude for metadata + text extraction, Gemini for vision
python -m geochem_benchmark.main run-paper Xia_et_al_2024 \
  --provider claude --vision-provider gemini
```

### Cost at scale

- Vision fires only when text extraction fails (~65% of PDF-only papers)
- Smart page selection sends only 2-5 data-dense pages (not all 20+)
- ~$0.02-0.10 per paper for vision calls

---

## Future Directions

Key remaining gaps:

- **Oxide notation mapping**: Tables using oxide formulas (Na2O, SiO2) instead of element symbols (Na, Si) are not yet mapped to schema columns. The self-correction layer correctly identifies these tables as transposed but element mapping fails on oxide names.
- **Domain-trained models**: Fine-tuned or RL-trained transformers that internalise geochemical ontologies and can extract accurately without ground truth at inference time
- **Chain-of-thought extraction**: Multi-step reasoning where the model first understands the paper's context, then plans its extraction strategy, then executes with self-verification
- **Per-sheet hints**: Currently the LLM returns one set of hints applied globally; some papers need different hints per sheet (e.g., one sheet transposed, another not)

The key insight is that each paper is structurally unique — a generic prompt cannot handle the full diversity of table formats, naming conventions, and metadata organisation across the geological literature. The agentic self-correction layer addresses this by dynamically adapting parsing strategy per paper.
