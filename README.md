# Geochem Benchmark

An LLM benchmarking framework for extracting structured geochemical data from research papers — at scale. Targets 100% accuracy for the USGS/CMMI Critical Minerals database.

---

## Overview

Research papers in economic geology report trace-element geochemical data across dozens of samples. Curating this data into a standardised database requires reading the full paper — not just copying table numbers, but also comprehending the geological context, analytical methods, deposit classification, and sample metadata scattered throughout the text.

This framework:

1. **Extracts** structured data from PDFs + supplementary files using 5 parallel backends + LLMs
2. **Classifies** deposit types against the full 189-type CMMI/Hofstra 2021 taxonomy (from DARPA CRITICALMAAS sri-ta2) with confidence scoring
3. **Evaluates** extractions against 28 ground truth papers across 4 tiers with precision/recall/F1
4. **Exports** as flat files (Excel/CSV), MinMod-compatible formats (JSON/JSON-LD/Turtle), and knowledge graphs (JSON/Neo4j/GraphML)

### Architecture

```
PDF Input
  ├─ Marker (surya OCR) ──────────────┐
  ├─ Docling (ML layout) ─────────────┤
  ├─ MinerU (YOLO layout) ────────────┤── 5 backends in parallel
  ├─ Camelot (borderless tables) ─────┤   → dedup by (sample, method)
  └─ pdfplumber (grid tables) ────────┘   → confidence scoring
           ↓
  LLM Metadata Extraction (Claude/GPT/Gemini)
  + Marker full-document markdown as additional context
           ↓
  Two-Pass Deposit Classification (189 CMMI types from DARPA sri-ta2)
           ↓
  USGS Post-Processing (BDL -99999, units→ppm, minerals, analysis_id)
           ↓
  Self-Validation Agent (conservative: removes only safe patterns)
           ↓
  Output (auto-generated on every to_excel() / to_csv() call):
    ├─ extraction_*.xlsx       — flat 364-col schema
    ├─ minmod_*.json           — MinMod CDR JSON   ← NEW (automatic)
    ├─ minmod_*.jsonld         — JSON-LD + MinMod ontology ← NEW (automatic)
    └─ minmod_*.ttl            — Turtle/RDF for SPARQL ← NEW (automatic)
```

### Schema

The target schema is a **364-column** format:
- 85 metadata columns (deposit, sample, analytical method, provenance, CMMI classification)
- 73 × 4 = 292 element columns (`_ppm` + `_detection_limit` + `_original_value` + `_original_unit`)
- 3 provenance columns (`extraction_backend`, `confidence`, `backend_agreement`)
- 1 `data_source_tag` column (`"this_study"` | `"cited_study:AuthorYear"`)

---

## Input / Output

| | Description |
|---|---|
| **Input 1** | Research paper PDF (full text including methods, sampling, geology sections) |
| **Input 2** | Supplementary data file(s) (`.xlsx`, `.xls`, `.csv`, or `.zip`) with per-sample element concentrations |
| **Output (flat)** | Excel/CSV with one row per analytical spot, 364-column schema |
| **Output (MinMod)** | JSON, JSON-LD, Turtle — auto-generated alongside every Excel/CSV |
| **Output (graph)** | Knowledge graph in JSON, Neo4j CSV, or GraphML |
| **Ground truth** | Human-curated Excel in corrected 364-column schema (`ground_truth_corrected/`) |

---

## Project Structure

```
geochem_benchmark/
├── schema.py                # 364-column schema — pydantic models, element list, original unit tracking
├── pdf_reader.py            # PDF → structured text + Marker full-document markdown
├── pdf_vision.py            # Render PDF pages as images for LLM vision API extraction
├── table_reader.py          # Excel/CSV → cleaned DataFrame with unit conversion, BDL handling
├── tabledetector.py         # 5-backend table detection (Docling, Marker, MinerU, Camelot, pdfplumber)
├── knowledge_base.py        # CMMI taxonomy (189 types from sri-ta2), USGS picklist, deposit scoring
├── deposit_classifier.py    # Two-pass LLM deposit classification against 189 CMMI types
├── minmod_exporter.py       # MinMod JSON / JSON-LD / Turtle export (auto-runs on save)
├── graph_extractor.py       # Knowledge graph output (JSON, Neo4j CSV, GraphML)
├── graph_native_pipeline.py # Graph-native 3-pass LLM extraction (experimental)
├── extraction_validator.py  # Post-extraction self-validation agent
├── prompts.py               # Multi-stage LLM prompt templates with USGS BDL/unit rules
├── llm_clients.py           # Unified Claude / OpenAI / Gemini client interface (text + vision)
├── pipeline.py              # 7-stage extraction pipeline
├── agentic_corrector.py     # LLM-powered self-correction for failed table extractions
├── evaluator.py             # 4-tier evaluation + precision/recall/F1 + position-based matching
├── paper_registry.py        # Maps 28 ground truth files to their PDF + supplementary data
├── batch_runner.py          # Batch processing with aggregate scoring
├── correct_ground_truth.py  # Script to generate corrected GT files (BDL→-99999, wt%→ppm)
├── main.py                  # CLI entry point (12 commands)
├── Pickelist.xlsx           # USGS authoritative picklist for all schema fields
│
├── ground_truth/            # 28 original GT files
├── ground_truth_corrected/  # 28 corrected GT files (BDL, units, schema aligned)
├── sri-ta2/                 # DARPA CRITICALMAAS deposit classification (189 types + descriptions)
├── training/                # Custom model training infrastructure (Marker/MinerU fine-tuning)
├── data/                    # Research paper PDFs
│   └── Spreadsheets/        # Supplementary data files (xlsx, csv, zip)
├── minmod_output/           # MinMod export outputs
└── gt_eval_v8/              # Latest GT evaluation results
```

---

## Installation

```bash
pip install pdfplumber pandas openpyxl pydantic xlrd

# Vision-based PDF extraction (renders pages as images for LLM vision API):
pip install PyMuPDF>=1.24.0

# Faster PDF text extraction (optional, falls back to pdfplumber):
pip install pdftext

# Graph output formats (optional):
pip install networkx       # for GraphML export

# Install whichever LLM providers you want to benchmark:
pip install anthropic          # Claude (Opus, Sonnet, Haiku)
pip install openai             # GPT-4o, GPT-5
pip install google-genai       # Gemini 2.5 Flash/Pro
```

### API keys

Set API keys as environment variables or in a `.env` file in the project root:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # Claude
export OPENAI_API_KEY="sk-..."           # OpenAI
export GOOGLE_API_KEY="AIza..."          # Gemini
```

### Supported models

| Provider | Models |
|---|---|
| Claude | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-5` |
| Gemini | `gemini-2.5-flash`, `gemini-2.5-pro` |

---

## Usage

All commands run from the **parent** directory of the package.

### CLI Commands

#### 1. Discover all papers

```bash
python -m geochem_benchmark.main discover
python -m geochem_benchmark.main discover --verbose
```

#### 2. Run a single paper by ID

```bash
# Full extraction + evaluation (if GT exists)
python -m geochem_benchmark.main run-paper Yuan_et_al_2018

# Use Haiku (faster/cheaper)
python -m geochem_benchmark.main run-paper Yuan_et_al_2018 \
  --provider claude --model claude-haiku-4-5-20251001

# Use GPT-4o
python -m geochem_benchmark.main run-paper Yuan_et_al_2018 \
  --provider openai --model gpt-4o

# Use Claude for text, Gemini for vision
python -m geochem_benchmark.main run-paper Xia_et_al_2024 \
  --provider claude --vision-provider gemini
```

Outputs saved to `paper_results/` (or `--output-dir`):

| File | Contents |
|---|---|
| `extraction_{id}.xlsx` | 364-column flat extraction |
| `minmod_extraction_{id}.json` | MinMod CDR JSON **← auto-generated** |
| `minmod_extraction_{id}.jsonld` | JSON-LD with MinMod ontology **← auto-generated** |
| `minmod_extraction_{id}.ttl` | Turtle/RDF for SPARQL **← auto-generated** |
| `report_{id}.xlsx` | Per-field evaluation breakdown (if GT exists) |
| `quality_{id}.json` | Quality metrics (if no GT) |

#### 3. Extract with explicit paths

```bash
python -m geochem_benchmark.main extract \
  --pdf data/2018_Yuan_etal.pdf \
  --supplementary data/Spreadsheets/2018_Yuan_etal.xlsx \
  --provider claude \
  --output results/yuan_output.xlsx
```

MinMod files are auto-generated at `results/minmod_yuan_output.{json,jsonld,ttl}`.

#### 4. Batch with ground truth

```bash
python -m geochem_benchmark.main batch \
  --provider claude --model claude-sonnet-4-6 \
  --output-dir batch_results/
```

Every paper gets 4 output files: extraction Excel + 3 MinMod formats.

#### 5. Batch without ground truth

```bash
python -m geochem_benchmark.main batch-nogt \
  --provider claude --model claude-haiku-4-5-20251001 \
  --output-dir nogt_results/
```

#### 6. Benchmark multiple LLMs

```bash
python -m geochem_benchmark.main benchmark \
  --pdf data/2018_Yuan_etal.pdf \
  --supplementary data/Spreadsheets/2018_Yuan_etal.xlsx \
  --ground-truth ground_truth_corrected/Yuan_et_al_2018.xlsx \
  --providers claude openai gemini \
  --output-dir results/
```

#### 7. Evaluate existing extraction

```bash
python -m geochem_benchmark.main eval \
  --prediction results/yuan_output.xlsx \
  --ground-truth ground_truth_corrected/Yuan_et_al_2018.xlsx \
  --provider claude --output-dir eval_results/
```

#### 8. MinMod export — standalone (from existing extractions)

If you already have extraction files and want to (re-)generate MinMod outputs:

```bash
# Export all extractions in a directory to all 3 MinMod formats
python -m geochem_benchmark.main export-minmod \
  --input-dir gt_eval_v8/ \
  --output-dir minmod_output/ \
  --formats json jsonld turtle

# JSON only
python -m geochem_benchmark.main export-minmod \
  --input-dir gt_eval_v8/ --formats json
```

Output files in `minmod_output/`:

| File | Format | Use |
|---|---|---|
| `mineral_sites_minmod.json` | `{"MineralSite": [...]}` | DARPA CDR ingest |
| `mineral_sites_minmod.jsonld` | JSON-LD + `@context` | Linked Data, semantic web |
| `mineral_sites_minmod.ttl` | Turtle/RDF | SPARQL endpoint at minmod.isi.edu |

#### MinMod MineralSite structure

Each record contains:
- `name`, `record_id`, `source_id` — site identity
- `country`, `province` — location (ISO3 codes)
- `deposit_type_candidate` — ranked classifications with `normalized_uri` (e.g., `https://minmod.isi.edu/resource/Q380`)
- `mineral_inventory` — commodities with MinMod URIs
- `geochemical_measurements` — per-element stats (median, min, max, n_bdl) — CMMI extension
- `analytical_methods` — instrument, lab, standards verbatim from paper
- `publication_year`

#### 9. Graph-based extraction

```bash
# Table-native with graph output (default — same accuracy as flat file)
python -m geochem_benchmark.main graph-extract \
  --pdf data/2024_Xia_etal.pdf \
  --provider claude --format all

# Graph-native extraction (experimental — 3-pass LLM, no table parsing)
python -m geochem_benchmark.main graph-extract \
  --pdf data/2024_Xia_etal.pdf \
  --provider claude --extraction-mode graph-native

# Side-by-side comparison
python -m geochem_benchmark.main graph-extract \
  --pdf data/2024_Xia_etal.pdf \
  --provider claude --extraction-mode both
```

##### Graph output formats

| Format | File | Best for |
|---|---|---|
| **JSON** | `{paper}_graph.json` | Python analysis, APIs, portable |
| **Neo4j CSV** | `{paper}_neo4j/nodes.csv` + `edges.csv` | Graph DB, Cypher queries: "find all sphalerite from MVT with Cd > 1000 ppm" |
| **GraphML** | `{paper}_graph.graphml` | Visual exploration in Gephi/Cytoscape |

##### Graph-native vs table-native

| Metric | Table-native | Graph-native |
|---|---|---|
| Numerical accuracy (T2) | **78%** | 58% |
| No hallucination (T4) | 99% | 99% |
| Multi-method handling | Row duplication | Explicit edges |
| Source attribution | Tagged post-hoc | Native graph edges |
| Cost per paper | ~$0.05 | ~$0.15 |

Table-native wins on accuracy (reads numbers from DataFrames directly). Graph-native wins on structure (separates EPMA vs LA-ICPMS vs Rb-Sr natively). A hybrid approach (LLM structure + DataFrame values) is in development.

#### 10. Discovery commands

```bash
python -m geochem_benchmark.main list-papers
python -m geochem_benchmark.main list-nogt-papers
python -m geochem_benchmark.main models
```

### Common flags

| Flag | Description | Available in |
|---|---|---|
| `--provider` | LLM provider: `claude`, `openai`, or `gemini` | all |
| `--model` | Model ID (uses provider default if omitted) | all |
| `--vision-provider` | Separate provider for vision API calls | all |
| `--vision-model` | Model ID for vision calls | all |
| `--no-vision` | Disable vision-based page extraction | all |
| `--no-self-correction` | Disable agentic self-correction | all |
| `--no-tool-calling` | Use plain JSON instead of Anthropic tool use | all |
| `--llm-table-filter` | LLM-assisted table row filtering | all |
| `--table-detector` | `auto`, `docling`, `camelot`, `pdfplumber` | all |
| `--verbose` / `-v` | Debug logging | all |
| `--include-pdf-only` | Include PDF-only papers | `batch`, `batch-nogt` |
| `--pdf-only` | Process ONLY PDF-only papers | `batch`, `batch-nogt` |
| `--format` | Graph output: `json`, `neo4j`, `graphml`, `all` | `graph-extract` |
| `--extraction-mode` | `table-native`, `graph-native`, `both` | `graph-extract` |
| `--formats` | MinMod formats: `json`, `jsonld`, `turtle` | `export-minmod` |

### USGS/CMMI compliance

All extractions follow the USGS CMiO-MIN database protocol:

| Rule | Implementation |
|---|---|
| BDL without LOD → **-99999** | `_safe_float()` returns `BELOW_DETECTION_SENTINEL = -99999.0` |
| BDL with specific LOD → **negative value** | `<0.5` → `-0.5` (LOD preserved) |
| N/A / not analyzed → **blank** | `_NOT_ANALYZED_STRINGS` set, returns None |
| All values in **ppm** | wt% × 10,000; ppb ÷ 1,000; original preserved in `_original_value`/`_original_unit` |
| One mineral per row | Post-processing splits grouped minerals |
| Hofstra 2021 classification | 189-type CMMI from DARPA CRITICALMAAS sri-ta2 |
| Three-tier sample ID | `sample_name` + `sample_local_id` + `analysis_id` |
| Deposit classification | Two-pass LLM: evidence extraction → 189-type scoring with confidence + reasoning |
| MinMod export | Auto-generated on every `to_excel()` / `to_csv()` call |

### Python API

```python
import os
from geochem_benchmark.llm_clients import ClaudeClient
from geochem_benchmark.pipeline import ExtractionPipeline
from geochem_benchmark.evaluator import Evaluator
from pathlib import Path

# Haiku (fast/cheap), Sonnet (balanced), Opus (best quality)
client = ClaudeClient(model="claude-haiku-4-5-20251001")
# client = ClaudeClient(model="claude-sonnet-4-6")
# client = ClaudeClient(model="claude-opus-4-6")

pipeline = ExtractionPipeline(llm_client=client)
result = pipeline.run(
    pdf_path="data/2018_Yuan_etal.pdf",
    supplementary_paths=["data/Spreadsheets/2018_Yuan_etal.xlsx"],
)

print(f"Extracted {result.n_samples} samples")

# Save flat file — MinMod outputs auto-generated alongside
result.to_excel("output/extraction_Yuan.xlsx")
# Also creates:
#   output/minmod_extraction_Yuan.json
#   output/minmod_extraction_Yuan.jsonld
#   output/minmod_extraction_Yuan.ttl

# Disable MinMod auto-generation if needed
result.to_excel("output/extraction_Yuan.xlsx", export_minmod=False)

# Evaluate against corrected ground truth
evaluator = Evaluator("ground_truth_corrected/Yuan_et_al_2018.xlsx")
report = evaluator.evaluate(result)
report.print_summary()
# Shows: Overall, T1 Metadata, T2 Numerical, T3 Structural, T4 Null
#        + Precision, Recall, F1

# Access deposit classification
df = result.to_dataframe()
print(df["deposit_type"].iloc[0])           # "MVT zinc-lead"
print(df["deposit_type_confidence"].iloc[0]) # 0.92
print(df["deposit_type_reasoning"].iloc[0])  # "Carbonate host, low-T hydrothermal..."

# Access original values before ppm conversion
print(df["fe_ppm"].iloc[0])               # 50300.0 (converted)
print(df["fe_original_value"].iloc[0])    # 5.03 (original wt%)
print(df["fe_original_unit"].iloc[0])     # "wt%"

# Manual MinMod export
from geochem_benchmark.minmod_exporter import extraction_to_mineral_site, export_minmod_json
site = extraction_to_mineral_site(df, "Yuan_et_al_2018",
                                  publication_doi="https://doi.org/10.1016/j.ore.2018.05.020")
export_minmod_json([site], "output/yuan_minmod.json")

# Graph output (table-native)
from geochem_benchmark.graph_extractor import build_graph_from_extraction
graph = build_graph_from_extraction(result)
graph.to_json("output/yuan_graph.json")
graph.to_neo4j_csv("output/yuan_neo4j/")
print(graph.stats)  # {total_nodes: 1966, total_edges: 3481, ...}
```

---

## Architecture

### 7-Stage Pipeline

```
Stage 0: Paper Intelligence
  LLM reads paper → expected elements, sample count, methods

Stage 1: Metadata Extraction
  LLM extracts deposit, mineral, instrument, lab, standards, country
  Uses USGS Picklist (Pickelist.xlsx) as authoritative value reference

Stage 2: Table Extraction (5 backends in parallel)
  Docling + Marker + MinerU + Camelot + pdfplumber → dedup by (sample, method)
  Source tagged: "this_study" or "cited_study:AuthorYear"
  Confidence scored: n_backends_agreed / n_backends_active

Stage 3: USGS Post-Processing
  Split grouped minerals → one mineral per row
  Infer mineral from analysis_id abbreviations (cpy→chalcopyrite)
  Validate Hofstra 2021 classification
  Populate analysis_id from sample_local_id

Stage 4: LLM Self-Correction (if extraction failed)
  Diagnose failure → structured hints → retry with corrections

Stage 5: Two-Pass Deposit Classification
  Pass 1: Extract geological evidence (host rocks, ore minerals, alteration)
  Pass 2: Score against all 189 CMMI types → top-5 with confidence + reasoning

Stage 6: Self-Validation Agent
  Remove only safe patterns: Mean, Average, (n=XX), "Source:"
  LLM suggestions logged but not acted on
  20% safety cap: never removes >20% of rows

Output: extraction_*.xlsx + minmod_*.{json,jsonld,ttl} (auto)
```

### Deposit Classification (DARPA CRITICALMAAS sri-ta2)

The pipeline uses the full 189-type CMMI taxonomy from the DARPA CRITICALMAAS sri-ta2 repository:
- `sri-ta2/minmod/deposit_type.csv` — 189 types with MinMod IDs (Q301–Q489)
- `sri-ta2/taxonomy/cmmi_options_full_description_with_number.csv` — full descriptions
- Classification resolves to `normalized_uri`: `https://minmod.isi.edu/resource/Q{id}`

### MinMod Export (auto on every save)

Every `to_excel()` / `to_csv()` call automatically produces 3 MinMod files:

| Format | Schema reference | Endpoint |
|---|---|---|
| **JSON** | DARPA CDR `{"MineralSite": [...]}` | CDR ingest |
| **JSON-LD** | MinMod ontology `@context` | Linked Data |
| **Turtle** | `mno:MineralSite`, `mno:deposit_type_candidate` | minmod.isi.edu SPARQL |

Deposit types map to official MinMod URIs (`https://minmod.isi.edu/resource/Q380` = MVT zinc-lead). Commodities map from `sri-ta2/minmod/commodity.csv`.

---

## Evaluation

### Tiers

| Tier | Weight | Measures | Metric |
|---|---|---|---|
| **T1 — Metadata** | 30% | deposit_type, mineral, instrument, lab, standards, country | String similarity |
| **T2 — Numerical** | 40% | Element concentrations for matched samples | Relative error ≤ 5% = full credit |
| **T3 — Structural** | 15% | Sample coverage vs GT | F1 of sample name sets |
| **T4 — Null** | 15% | No hallucinated values for unmeasured elements | Per-element null accuracy |

### Sample matching strategies

1. **Exact match** across `sample_name`, `sample_local_id`, `analysis_id`
2. **Normalised exact** (whitespace/dash differences)
3. **Prefix matching** — `K21-01` matches `K21-01@L3`, `K21-01-12`
4. **Position-based matching** — when names don't match at all, aligns rows by element value fingerprints (within 10% relative error). Fixed Wang et al.: F1 7.5% → 100%

### V8 Benchmark Results (28 GT papers, corrected GT)

| Model | Overall | T2 Numerical | F1 |
|---|---|---|---|
| Claude Opus 4.6 | **76.26%** | **76.95%** | — |
| Claude Sonnet 4.6 | 75.98% | 76.55% | 60.1% |
| GPT-4o | 75.93% | 76.57% | — |
| Gemini 2.5 Flash | 75.69% | 76.44% | — |
| Claude Haiku 4.5 | 75.49% | 76.40% | — |

- 5 papers at 100% F1 (Chu, Soster, Lan, Sun 2024, Wang)
- 7 papers with T2 > 95% (near-perfect numerical accuracy)
- Scores tightly clustered across providers (< 1 pp spread)

---

## USGS BDL Reference

| Cell value | Stored as | Meaning |
|---|---|---|
| `bdl`, `n.d.`, `-`, `--` | `-99999` | Measured, below detection, LOD unknown |
| `<0.5` | `-0.5` | Measured, below LOD of 0.5 ppm |
| `n/a`, `not analyzed` | `null` | NOT measured at all |
| `5.03` (wt%) | `50300` ppm | Measured, converted (original: `fe_original_value=5.03, fe_original_unit=wt%`) |
| `123.5` (ppm) | `123.5` ppm | Measured as-is |

---

## Adding a New Paper

1. Place the PDF in `data/` and supplementary files in `data/Spreadsheets/`
2. Create a corrected ground truth Excel in `ground_truth_corrected/` using the 364-column schema (BDL as -99999, all values in ppm)
3. Add an entry to `paper_registry.py`
4. Run evaluation:

```bash
python -m geochem_benchmark.main batch \
  --provider claude --model claude-sonnet-4-6 \
  --papers Smith_et_al_2024 --output-dir batch_results/
```

---

## Future Directions

- **Hybrid graph+table extraction** — LLM for structural understanding (what to extract), DataFrames for values (how to extract). Combines graph-native accuracy with table-native speed.
- **Custom model fine-tuning** — Marker/MinerU fine-tuned on geochem table layouts (training infrastructure in `training/`)
- **ISWC 2026 cross-domain evaluation** — Same framework tested on ChemTables (drug discovery) and DiSCoMaT (materials science)
- **Scale to millions of papers** — Neo4j graph database for cross-paper querying
