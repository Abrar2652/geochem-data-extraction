"""
prompts.py - LLM prompt templates for multi-stage geochemical data extraction.

Design principle: two-stage prompting
  Stage 1 → extract paper-level metadata from PDF text (applies to all rows)
  Stage 2 → verify / supplement table row data with LLM assistance
"""

from __future__ import annotations
import json
from textwrap import dedent

from .schema import PaperMetadata
from .knowledge_base import get_knowledge_base_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Stage 0: Paper Intelligence Blueprint (pre-extraction analysis)
# ──────────────────────────────────────────────────────────────────────────────

PAPER_INTELLIGENCE_SYSTEM_PROMPT = dedent("""\
You are an expert analytical geochemist. Your task is to carefully read a research paper
and extract specific analytical methodology details. Focus on the Methods, Analytical,
and Results sections.

Be precise — copy instrument descriptions, lab names, standards, and conditions VERBATIM
from the paper. Do not paraphrase or fabricate any details.
""")

PAPER_INTELLIGENCE_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT
{paper_text}

## YOUR TASK
Analyze this geochemistry paper and extract analytical intelligence.
Return a JSON object with EXACTLY these fields:

{{
  "elements_measured": ["fe", "cu", "zn"],
  "expected_sample_count": 50,
  "minerals_analyzed": ["pyrite", "sphalerite"],
  "analytical_methods": ["LA-ICPMS"],
  "instrument": "exact instrument description from paper or null",
  "laboratory": "exact lab name and location from paper or null",
  "standards_used": "all standards mentioned, verbatim, or null",
  "operating_conditions": "all conditions mentioned, verbatim, or null"
}}

## FIELD INSTRUCTIONS
1. **elements_measured**: List ALL element symbols (lowercase) that have actual analytical
   data in the paper's tables. Check table headers and results text carefully.
   Valid symbols: fe, cu, zn, pb, ag, au, as, sb, bi, co, ni, mn, cd, in, ga, ge, se, te,
   tl, sn, mo, w, v, cr, ti, sc, ba, sr, rb, cs, li, be, b, p, s, cl, f, br, hg, re,
   si, al, ca, mg, na, k, la, ce, pr, nd, sm, eu, gd, tb, dy, ho, er, tm, yb, lu, y,
   zr, hf, nb, ta, th, u

2. **expected_sample_count**: Approximate number of individual spot analyses/measurements
   in the paper's data tables. Count analytical spots, not hand samples.

3. **minerals_analyzed**: Mineral names (lowercase) that were analyzed (e.g., pyrite,
   sphalerite, chalcopyrite). Use null if whole-rock analysis.

4. **analytical_methods**: Standardized method names used. Common: LA-ICPMS, EPMA, ICP-MS,
   XRF, SEM-EDS, SIMS.

5. **instrument**: Copy the EXACT instrument description from the paper. Include manufacturer,
   model, and attached components (laser system, etc.). Use null if not stated.

6. **laboratory**: Copy the EXACT laboratory name and location. Use null if not stated.

7. **standards_used**: Copy ALL reference/calibration standards mentioned — internal standards,
   external standards, quality control materials. Verbatim. Use null if not stated.

8. **operating_conditions**: Copy ALL analytical conditions — spot size, beam diameter,
   accelerating voltage, beam current, laser frequency, energy density, carrier gas, dwell
   time, repetition rate, etc. Verbatim. Use null if not stated.

## CRITICAL
- For fields 5-8, copy text VERBATIM — do not summarize or paraphrase.
- If multiple methods were used, separate details with " | " (pipe character).
- For elements_measured, check ACTUAL data table headers, not just the methods description.

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Paper-level metadata extraction
# ──────────────────────────────────────────────────────────────────────────────

METADATA_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator with deep knowledge of economic geology,
analytical geochemistry, and mineralogy. Your task is to extract structured metadata from
research papers for a standardized geochemical database.

Extract information EXACTLY as stated in the paper — do not paraphrase or invent values.
Return ONLY valid JSON. If a field cannot be determined from the paper, use null.

{knowledge_base}
""")

METADATA_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT
{paper_text}

## SUPPLEMENTARY TABLE PREVIEW
The supplementary table (first few rows) looks like:
{table_preview}

## YOUR TASK
Extract paper-level metadata from this research paper. This metadata generally applies to
all samples, but note that some fields (mineral, analytical_method, deposit_name) MAY vary
per sample — if so, extract the MOST COMMON or PRIMARY value here; per-row overrides will
be handled separately from the supplementary table columns.

Return a JSON object with EXACTLY these fields (use null if not found in the paper):

```json
{{
  "deposit_name": "CONCISE deposit name only — no commodity metals, no 'deposit/mine' suffix. E.g. 'Bainiuchang' NOT 'Bainiuchang Zn-Sn polymetallic deposit'. If multiple, comma-separated.",
  "deposit_local_id": "Local deposit identifier code, if any",
  "deposit_environment": "MUST be one of: 'Basin hydrothermal', 'Seafloor hydrothermal', 'Magmatic hydrothermal', 'Magmatic', 'Epithermal', 'Metamorphic', 'Sediment-hosted', 'Supergene', 'Surficial', 'unknown'. Infer from deposit type.",
  "deposit_group": "Broad deposit class. MUST use standard form: 'Mississippi Valley-type (MVT)', 'Volcanic-hosted massive sulfide (VMS)', 'Sedimentary Exhalative (SEDEX)', 'Skarn', 'Porphyry', 'Epithermal', 'Orogenic gold', 'Iron oxide copper-gold (IOCG)', 'Carlin-type', 'Stratiform sediment-hosted Cu', 'Magmatic Ni-Cu-PGE', 'Greisen', 'Pegmatite', 'unknown'",
  "deposit_type": "Specific deposit type per Hofstra et al. 2021. E.g. 'MVT zinc-lead', 'Cu-Au porphyry', 'orogenic gold', 'stratiform sediment-hosted Cu-Co'. Use 'unknown' if not determinable.",
  "deposit_classification_source": "Classification scheme used. Always set to 'Hofstra et al. 2021'. If the paper uses a different scheme, note it here: 'Hofstra et al. 2021 (paper uses: <their scheme>)'.",
  "deposit_type_original": "Deposit type EXACTLY as stated by the paper's authors, verbatim. May differ from Hofstra 2021 (e.g., paper says 'SEDEX-type Zn-Pb' but Hofstra says 'Sedimentary Exhalative'). null if same as deposit_type.",
  "primary_commodities": "Primary economic metals, comma-separated",
  "secondary_commodities": "Secondary/byproduct metals, comma-separated",
  "all_commodities": "ALL commodities combined, comma-separated. E.g. 'Zn, Pb' or 'Cu, Au, Mo'",
  "deposit_source": "Short author + year citation. E.g. 'Yuan et al., 2018'",

  "feature_type": "Mine/outcrop type. E.g. 'underground mine', 'open pit', 'drill core', 'surface outcrop'",
  "sample_deposit_relation": "Relationship of sample to deposit. E.g. 'ore material', 'waste', 'wall rock', 'halo'",
  "sample_type": "Sample collection type. E.g. 'drill core', 'grab', 'chip', 'channel', 'polished section'",
  "sampling_method": "How samples were physically collected. E.g. 'grab', 'drill core', 'channel sampling'",
  "material_class": "Broad material class. If specific minerals were analyzed → 'mineral separate'. If whole rock → 'rock'. Others: 'soil', 'sediment'.",
  "earth_material_group": "Material group. For ore minerals → 'mineralisation'. For host rocks → use rock type (igneous/sedimentary/metamorphic).",
  "mineral": "Specific mineral(s) analyzed. E.g. 'sphalerite', 'pyrite', 'chalcopyrite'. Comma-separated if multiple minerals were analyzed in the paper.",
  "associated_minerals": "Other minerals noted to co-occur (not analyzed, but mentioned in the paragenesis), comma-separated",

  "analytical_method": "Standardized method abbreviation. Use the ANALYTICAL METHOD STANDARDIZATION reference above. Common: 'LA-ICPMS', 'EMPA', 'XRF', 'ICP-MS'. If multiple methods were used, list comma-separated.",
  "instrument_type_model": "Instrument manufacturer and model ONLY. E.g. 'Agilent 7900 ICP-MS coupled with RESOlution 193nm ArF excimer laser'. Copy from paper, no extra description.",
  "laboratory_location": "Lab name and city/institution ONLY. E.g. 'State Key Laboratory of Geological Processes and Mineral Resources, China University of Geosciences, Wuhan'",
  "operating_conditions": "Key analytical parameters: spot size, beam energy, repetition rate, carrier gas, dwell time. E.g. '44 µm spot, 80 mJ, 6 Hz, He carrier gas'",
  "standards_used": "All calibration/reference standards with their purpose, as written in the paper. E.g. 'STDGL3 for chalcophile elements, GSD-1G for lithophile elements, MASS-1 as quality check'",
  "sample_preparation": "How samples were prepared (polished thin sections, mounted in epoxy, pressed pellets, etc.)",

  "publication_date": "Year of publication as integer",
  "sample_source": "Full bibliographic citation: Authors (year), Title, Journal, Volume, Pages",
  "country": "ISO 3-letter country code where the deposit is located. Use the ISO COUNTRY CODES reference above.",
  "state": "State/province/region if mentioned in the paper"
}}
```

## CRITICAL GUIDELINES (USGS/CMMI MANDATORY STANDARDS)
1. **deposit_environment, deposit_group, deposit_type MUST follow the Hofstra et al. 2021 CMMI
   classification scheme** (see DEPOSIT CLASSIFICATION REFERENCE above). Do NOT use any alternative
   classification system, even if the original authors use different terminology. Map the author's
   terminology to the closest Hofstra 2021 category.
2. Use the ANALYTICAL METHOD STANDARDIZATION reference for `analytical_method`.
3. `country` must be an ISO 3-letter code (CHN, AUS, USA, CAN, ZAF, SWE, etc.)
4. Do NOT fabricate coordinates, Mindat references, or any information not in the paper.
5. For `sample_source`, construct the full bibliographic citation from the paper header/footer.
6. Copy instrument descriptions, lab locations, and analytical conditions VERBATIM from the paper.
7. If the paper studies multiple deposits, list comma-separated. But for `mineral`, each row
   can only have ONE mineral — if multiple minerals were analyzed, the per-row assignment
   will be handled downstream. List the primary mineral here.
8. Do NOT hallucinate or assume field values without strong confidence.

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: LLM-assisted table verification / ambiguity resolution
# ──────────────────────────────────────────────────────────────────────────────

TABLE_FILTER_SYSTEM_PROMPT = dedent("""\
You are a geochemistry database curator. You will be given a raw supplementary table
from a research paper. Your job is to identify which rows are genuine sample analyses
from THIS paper (not comparison data from other references or summary statistics).
""")

TABLE_FILTER_USER_PROMPT_TEMPLATE = dedent("""\
## SUPPLEMENTARY TABLE
{table_text}

## TASK
1. Identify rows that are GENUINE SAMPLE ANALYSES from this paper.
   - INCLUDE rows where reference/source = "this paper", "this study", or is blank/missing.
   - EXCLUDE rows cited from other papers (e.g., "Ye et al., 2011").
   - EXCLUDE statistical summary rows (MEAN, STD, MINIMA, MAXIMA, AVERAGE, etc.).
   - EXCLUDE rows with no sample identifier.

2. For each valid sample row, extract the sample name and ALL element concentrations.

## ELEMENT COLUMN MAPPING
{column_mapping}

## OUTPUT FORMAT
Return a JSON array. Each element is one valid sample:
[
  {{
    "sample_name": "LA1-1_1",
    "deposit_name": "Daliangzi",
    "ag_ppm": 43.7,
    "as_ppm": 13.94,
    "bi_ppm": 0.0196,
    "cd_ppm": 1479.04,
    "co_ppm": 2.22,
    "fe_ppm": 32998.28,
    ... (all measured elements)
  }},
  ...
]

Rules:
- CRITICAL DISTINCTION between below-detection and not-measured (USGS protocol):
  * If analysis was performed but the value is below detection limit (BDL, b.d.l., n.d., <DL, "-", "--"):
    - If NO specific detection limit is given → use -99999
    - If a specific detection limit IS given (e.g., "<0.5 ppm") → use the NEGATIVE of that limit (e.g., -0.5)
    - Detection limits may be listed in the table or buried in the paper text
  * If "N/A", "not analyzed", "not measured", or "not reported" → use null (blank = not attempted)
  * -99999 means "measured but too low to quantify, LOD unknown". null means "not measured at all".
  * Leaving a below-detection field as null LOSES valuable information.
- Each data row must be assigned to exactly ONE mineral. Never group minerals (e.g., NOT "chalcopyrite, sphalerite").
  If a table mixes minerals, use the mineral column to assign one mineral per row.
- Track three sample ID tiers where available:
  * "sample_name": core identifier (e.g., "201003232")
  * "sample_local_id": local identifier from the paper
  * "analysis_id": full analysis string from supplementary (e.g., "5-2002063521cpy1-1.d")
- UNIT CONVERSION: All values MUST be in ppm. Convert wt% × 10000, ppb ÷ 1000.
- Output samples in the EXACT order they appear in the table — no sorting/reordering.
- Round values to 4 significant figures maximum.
- Do NOT add elements not present in the supplementary table.
- Do NOT hallucinate or assume values — only extract what is explicitly reported.

Return ONLY the JSON array. No explanation.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Combined one-shot prompt (for very capable models / small tables)
# ──────────────────────────────────────────────────────────────────────────────

FULL_EXTRACTION_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator. Extract structured data from a research
paper (PDF text + supplementary table) and output it in the standardized 210-column schema.
Be precise: extract values EXACTLY as reported, using null for any field not in the paper.
""")

FULL_EXTRACTION_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT (key sections)
{paper_text}

## SUPPLEMENTARY TABLE
{table_text}

## YOUR TASK
Extract ALL sample analytical data into the standardized schema. Output a JSON object with:
  1. "metadata": paper-level fields that apply to every row
  2. "samples": array of per-sample rows with element concentrations

## METADATA SCHEMA (same for all rows)
{metadata_schema}

## SAMPLE SCHEMA (one per analytical spot)
Each sample object must have:
- "sample_name": sample identifier from the table
- "deposit_name": deposit name (may differ per row if multiple deposits)
- Element columns: "{element}_ppm" for each measured element (null if not measured)

## FILTERING RULES
- Include ONLY rows from THIS paper (reference = "this paper" / "this study" / blank)
- EXCLUDE statistical summary rows: MEAN, STD, MINIMA, MAXIMA
- EXCLUDE rows cited from other papers

## OUTPUT FORMAT
{{
  "metadata": {{
    "deposit_name": "...",
    "deposit_environment": "...",
    ... (all PaperMetadata fields)
  }},
  "samples": [
    {{
      "sample_name": "...",
      "ag_ppm": null,
      "fe_ppm": 32998.28,
      ... (only include elements present in the table)
    }},
    ...
  ]
}}

Return ONLY valid JSON. No markdown fences, no explanation.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: PDF-only table extraction (no supplementary file)
# ──────────────────────────────────────────────────────────────────────────────

PDF_TABLE_EXTRACTION_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator. You are given the text of a research paper
(including any tables embedded in the PDF). There is NO supplementary spreadsheet available.
Your task is to extract individual sample analytical data (element concentrations) directly
from tables in the PDF text.

Extract information EXACTLY as reported — do not paraphrase, estimate, or fabricate values.

{knowledge_base}
""")

PDF_TABLE_EXTRACTION_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT (relevant sections)
{paper_text}

## TABLES FOUND IN PDF
{pdf_tables}

## YOUR TASK
Extract ALL individual sample analyses from the tables above. Each row should be one
analytical spot/measurement with its element concentrations.

Return a JSON object with:
1. "samples": array of per-sample objects
2. "extraction_notes": brief description of which table(s) you extracted from

## SAMPLE FORMAT
Each sample object must include:
- "sample_name": the sample identifier from the table
- Element concentrations as "{{element_symbol}}_ppm" (e.g., "fe_ppm", "cu_ppm", "zn_ppm")
  * Keep values in their ORIGINAL units exactly as printed — do NOT convert between units
  * The "_ppm" suffix is just the column name convention — it does NOT mean you should convert to ppm
  * If the table says Fe = 2.19 wt%, record "fe_ppm": 2.19
  * If the table says Cu = 123.5 ppm, record "cu_ppm": 123.5
  * If the table says Au = 50 ppb, record "au_ppm": 50
  * USGS BDL protocol:
    - If below detection limit and NO specific LOD given (BDL, b.d.l., n.d., dash) → use -99999
    - If below detection limit and a specific LOD IS given (e.g., "<0.5") → use NEGATIVE of that limit (e.g., -0.5)
    - If "N/A", "not analyzed", "not measured" → use null (blank = not attempted at all)
    - -99999 = "measured, below detection, LOD unknown". null = "not measured".
  * Use null ONLY for elements that were NOT measured / NOT reported at all
- Optional per-sample metadata fields (include only if they vary per sample):
  * "deposit_name": deposit name if it differs per sample
  * "mineral": EXACTLY ONE mineral name per row — never group (e.g., NOT "pyrite, chalcopyrite")
  * "analytical_method": method if it differs per sample
  * "analysis_id": full analysis identifier string if available
  * "texture": texture description if available

## FILTERING RULES
- Include ONLY rows that are actual sample measurements (individual spots/analyses)
- EXCLUDE statistical summary rows: MEAN, AVERAGE, STD, MIN, MAX, MEDIAN
- EXCLUDE rows cited from other studies/references
- EXCLUDE detection limit rows
- EXCLUDE blank or header rows

## UNIT CONVERSION (MANDATORY)
- All element values in the output MUST be in ppm.
- If the table reports values in wt%, convert to ppm: multiply by 10,000
  (e.g., Fe = 2.19 wt% → "fe_ppm": 21900)
- If the table reports values in ppb, convert to ppm: divide by 1,000
  (e.g., Au = 50 ppb → "au_ppm": 0.05)
- If the table reports values in ppm, µg/g, or mg/kg — no conversion needed
- The "_ppm" column suffix means the value MUST be in ppm units

## SAMPLE ORDER (CRITICAL)
- Output samples in EXACTLY the same order they appear in the paper tables
- Do NOT sort, group, or reorder samples by name, mineral, or any other field
- The human evaluator will compare row-by-row with the source table

## ELEMENT SYMBOL REFERENCE
Common elements and their symbols (use lowercase):
  Fe=fe, Cu=cu, Zn=zn, Pb=pb, Ag=ag, Au=au, As=as, Sb=sb, Bi=bi, Co=co,
  Ni=ni, Mn=mn, Cd=cd, In=in, Ga=ga, Ge=ge, Se=se, Te=te, Tl=tl, Sn=sn,
  Mo=mo, W=w, V=v, Cr=cr, Ti=ti, Sc=sc, Ba=ba, Sr=sr, Rb=rb, Cs=cs,
  Li=li, Be=be, B=b, P=p, S=s, Cl=cl, F=f, Br=br, Hg=hg, Re=re,
  Si=si, Al=al, Ca=ca, Mg=mg, Na=na, K=k,
  La=la, Ce=ce, Pr=pr, Nd=nd, Sm=sm, Eu=eu, Gd=gd, Tb=tb, Dy=dy,
  Ho=ho, Er=er, Tm=tm, Yb=yb, Lu=lu, Y=y, Zr=zr, Hf=hf, Nb=nb, Ta=ta,
  Th=th, U=u

## OUTPUT FORMAT
{{
  "samples": [
    {{
      "sample_name": "PY-1-1",
      "fe_ppm": 460000,
      "cu_ppm": 123.5,
      "zn_ppm": 45.2,
      "as_ppm": null,
      ...
    }},
    ...
  ],
  "extraction_notes": "Extracted 45 analyses from Table 2 (LA-ICP-MS trace elements in pyrite)"
}}

If the paper has NO extractable sample-level analytical data in the PDF tables
(e.g., only summary statistics, or tables are about something else entirely),
return: {{"samples": [], "extraction_notes": "No individual sample analyses found in PDF tables"}}

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: Vision-based PDF table extraction (page images)
# ──────────────────────────────────────────────────────────────────────────────

# Shared sample format instructions (used by both text and vision prompts)
_SAMPLE_FORMAT_INSTRUCTIONS = dedent("""\
## SAMPLE FORMAT
Each sample object must include:
- "sample_name": the sample identifier from the table
- Element concentrations as "{{element_symbol}}_ppm" (e.g., "fe_ppm", "cu_ppm", "zn_ppm")
  * Keep values in their ORIGINAL units exactly as printed in the table — do NOT convert between units
  * If the table says Fe = 2.19 wt%, record "fe_ppm": 2.19
  * If the table says Cu = 123.5 ppm, record "cu_ppm": 123.5
  * If the table says Au = 50 ppb, record "au_ppm": 50
  * The "_ppm" suffix is just the column name — it does NOT mean you should convert to ppm
  * USGS BDL protocol:
    - If below detection limit and NO specific LOD given (BDL, b.d.l., n.d., dash) → use -99999
    - If below detection limit and a specific LOD IS given (e.g., "<0.5") → use NEGATIVE of that limit (e.g., -0.5)
    - If "N/A", "not analyzed", "not measured" → use null (blank = not attempted at all)
    - -99999 = "measured, below detection, LOD unknown". null = "not measured".
  * Use null ONLY for elements that were NOT measured / NOT reported at all
- Optional per-sample metadata fields (include only if they vary per sample):
  * "deposit_name": deposit name if it differs per sample
  * "mineral": EXACTLY ONE mineral name per row — never group (e.g., NOT "pyrite, chalcopyrite")
  * "analytical_method": method if it differs per sample
  * "analysis_id": full analysis identifier string if available
  * "texture": texture description if available

## FILTERING RULES
- Include ONLY rows that are actual sample measurements (individual spots/analyses)
- EXCLUDE statistical summary rows: MEAN, AVERAGE, STD, MIN, MAX, MEDIAN
- EXCLUDE rows cited from other studies/references
- EXCLUDE detection limit rows
- EXCLUDE blank or header rows

## UNIT CONVERSION (MANDATORY)
- All element values in the output MUST be in ppm.
- If the table reports values in wt%, convert to ppm: multiply by 10,000
  (e.g., Fe = 2.19 wt% → "fe_ppm": 21900)
- If the table reports values in ppb, convert to ppm: divide by 1,000
  (e.g., Au = 50 ppb → "au_ppm": 0.05)
- If already in ppm, µg/g, or mg/kg — no conversion needed

## SAMPLE ORDER (CRITICAL)
- Output samples in EXACTLY the same order they appear in the paper tables
- Do NOT sort, group, or reorder samples

## ELEMENT SYMBOL REFERENCE
Common elements and their symbols (use lowercase):
  Fe=fe, Cu=cu, Zn=zn, Pb=pb, Ag=ag, Au=au, As=as, Sb=sb, Bi=bi, Co=co,
  Ni=ni, Mn=mn, Cd=cd, In=in, Ga=ga, Ge=ge, Se=se, Te=te, Tl=tl, Sn=sn,
  Mo=mo, W=w, V=v, Cr=cr, Ti=ti, Sc=sc, Ba=ba, Sr=sr, Rb=rb, Cs=cs,
  Li=li, Be=be, B=b, P=p, S=s, Cl=cl, F=f, Br=br, Hg=hg, Re=re,
  Si=si, Al=al, Ca=ca, Mg=mg, Na=na, K=k,
  La=la, Ce=ce, Pr=pr, Nd=nd, Sm=sm, Eu=eu, Gd=gd, Tb=tb, Dy=dy,
  Ho=ho, Er=er, Tm=tm, Yb=yb, Lu=lu, Y=y, Zr=zr, Hf=hf, Nb=nb, Ta=ta,
  Th=th, U=u

## OUTPUT FORMAT
{{
  "samples": [
    {{
      "sample_name": "PY-1-1",
      "fe_ppm": 460000,
      "cu_ppm": 123.5,
      "zn_ppm": 45.2,
      "as_ppm": null,
      ...
    }},
    ...
  ],
  "extraction_notes": "Extracted 45 analyses from Table 2 (LA-ICP-MS trace elements in pyrite)"
}}

If there are NO extractable sample-level analytical data,
return: {{"samples": [], "extraction_notes": "No individual sample analyses found"}}

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


VISION_TABLE_EXTRACTION_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator. You are given IMAGES of pages from a
research paper PDF that contain geochemical data tables. There is NO supplementary
spreadsheet available. Your task is to visually read the tables and extract individual
sample analytical data (element concentrations).

Read the tables EXACTLY as printed — do not estimate, interpolate, or fabricate values.

## TABLE RECOGNITION RULES

**Orientation & Layout:**
- LANDSCAPE (rotated) pages are COMMON in geochem papers — wide tables with many element
  columns are often printed sideways. Read the table in its natural reading direction
  regardless of page rotation. If a page appears rotated 90°, mentally rotate it and
  read normally.
- Tables may span the full page width OR be split into two side-by-side sub-tables.
- Some tables are TRANSPOSED: elements as rows, samples as columns. Detect this and
  extract correctly — each column becomes one sample, each row is an element.

**Headers & Structure:**
- Column headers may contain element symbols (Fe, Cu, Zn), isotope prefixes (55Mn, 59Co),
  oxide notation (SiO2, Al2O3, FeO), or unit suffixes (ppm, wt%, ppb).
- MULTI-ROW headers are common: Row 1 = element symbols, Row 2 = units, Row 3 = detection limits.
  Combine all header rows to identify what each column represents.
- Merged cells: sample name or mineral name may span multiple rows. Apply the merged value
  to ALL rows it spans.

**Continuation tables:**
- "Table X (continued)" or "Table X (cont.)" means this is a continuation from a previous
  page — the column headers may be repeated but the data rows are NEW samples.
- Some continuation tables continue the SAME samples with DIFFERENT elements (horizontal
  continuation). In this case, merge the element values into the same sample row.

**Data values (USGS BDL protocol):**
- Below detection limit with NO specific LOD: bdl, b.d.l., n.d., dash (—, –, -) → use -99999
- Below detection limit WITH a specific LOD: "<0.5" → use -0.5 (NEGATIVE of the reported LOD)
- "N/A", "not analyzed", "not measured" → use null (blank = not attempted at all)
- -99999 means "measured but too low, LOD unknown". Negative values (e.g., -0.5) mean "below LOD of 0.5".
  null/blank means "not measured at all".
- Commas in numbers are THOUSANDS separators (50,525 = 50525), NOT decimal points.
- Keep values in their ORIGINAL units as printed. Do NOT convert wt% to ppm or vice versa.
- Do NOT hallucinate or assume values. Only extract what is explicitly in the table.

**What to EXCLUDE:**
- Summary/statistics rows: Mean, Median, Average, Std Dev, Min, Max, Range, n= → SKIP these.
- Reference/comparison data from OTHER papers cited in the table → only extract data from THIS paper.
- Table captions, footnotes, and annotations → do not extract as data rows.

{knowledge_base}
""")


VISION_TABLE_EXTRACTION_USER_PROMPT = dedent("""\
## IMAGES
The attached {n_pages} image(s) show pages from the PDF that likely contain
geochemical data tables. Read them carefully — these are the actual page renders.
{orientation_hint}

## PAPER CONTEXT (text from the paper for additional context)
{paper_context}

## YOUR TASK
Extract ALL individual sample analyses visible in the table images. Each row should be
one analytical spot/measurement with its element concentrations.

If a page shows NO data table (only text, figures, or maps), return an empty samples
array for that page — do not fabricate data.

If a table is in LANDSCAPE orientation (rotated 90°), mentally rotate the page and read
the table in its natural left-to-right, top-to-bottom direction.

Return a JSON object with:
1. "samples": array of per-sample objects
2. "extraction_notes": brief description of which table(s) you extracted from,
   including table number/caption if visible

{sample_format_instructions}
""")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for building prompts
# ──────────────────────────────────────────────────────────────────────────────

def build_paper_intelligence_prompt(
    paper_text: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage 0 paper intelligence extraction.

    This focused prompt extracts analytical methodology details (instrument,
    lab, standards, conditions) and the list of elements measured — information
    that guides downstream metadata and table extraction.
    """
    user = PAPER_INTELLIGENCE_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:20000],
    )
    return PAPER_INTELLIGENCE_SYSTEM_PROMPT, user


def build_metadata_prompt(
    paper_text: str,
    table_preview: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage 1 metadata extraction."""
    system = METADATA_SYSTEM_PROMPT.format(
        knowledge_base=get_knowledge_base_prompt(),
    )
    user = METADATA_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:20000],
        table_preview=table_preview[:3000],
    )
    return system, user


def build_table_filter_prompt(
    table_text: str,
    column_mapping: dict[str, str],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage 2 table filtering."""
    mapping_lines = "\n".join(
        f'  Column "{raw}" → schema field "{sym}_ppm"'
        for raw, sym in column_mapping.items()
    )
    user = TABLE_FILTER_USER_PROMPT_TEMPLATE.format(
        table_text=table_text[:8000],
        column_mapping=mapping_lines or "  (auto-detected from column names)",
    )
    return TABLE_FILTER_SYSTEM_PROMPT, user


def build_pdf_table_extraction_prompt(
    paper_text: str,
    pdf_tables: list[str],
    data_pages_text: str = "",
    elements_measured: list[str] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for PDF-only table extraction.

    Used when no supplementary file is available — extracts sample data
    directly from tables embedded in the PDF.

    Args:
        paper_text: Prioritised paper text (abstract, methods, results).
        pdf_tables: Structured tables extracted by pdfplumber.
        data_pages_text: Raw text from pages with high numeric density
            (fallback when pdfplumber can't extract structured tables).
        elements_measured: Optional list of element symbols (lowercase)
            from Paper Intelligence — constrains extraction scope.
    """
    system = PDF_TABLE_EXTRACTION_SYSTEM_PROMPT.format(
        knowledge_base=get_knowledge_base_prompt(),
    )

    # Format PDF tables
    if pdf_tables:
        tables_text = "\n\n".join(
            f"--- Table {i+1} ---\n{tbl}"
            for i, tbl in enumerate(pdf_tables)
            if tbl.strip()
        )
        if not tables_text.strip():
            tables_text = ""
    else:
        tables_text = ""

    # When no structured tables, include raw data-dense pages
    if not tables_text and data_pages_text:
        tables_text = (
            "(No structured tables could be extracted from the PDF. "
            "Below are raw text pages that may contain tabular data. "
            "Look for data patterns — numbers aligned in columns, "
            "sample IDs followed by element concentrations, etc.)\n\n"
            + data_pages_text[:10000]
        )
    elif not tables_text:
        tables_text = (
            "(No structured tables could be extracted from the PDF. "
            "Look carefully in the PAPER TEXT above for any inline "
            "sample data — tables rendered as text, data in the results "
            "section, appendices, or any section with element concentrations.)"
        )

    user = PDF_TABLE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:20000],
        pdf_tables=tables_text[:12000],
    )

    # Append element scope constraint from Paper Intelligence
    if elements_measured:
        scope = ", ".join(sorted(elements_measured))
        user += (
            f"\n\n## ELEMENT SCOPE (from paper analysis)\n"
            f"This paper measures these specific elements: {scope}\n"
            f"Focus extraction on ONLY these elements. Use null for all others.\n"
        )

    return system, user


def build_vision_table_extraction_prompt(
    n_pages: int,
    paper_context: str = "",
    elements_measured: list[str] | None = None,
    has_landscape: bool = False,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for vision-based table extraction.

    Used when pdfplumber cannot extract structured tables — the LLM reads
    rendered page images directly to extract tabular data.

    Args:
        n_pages: Number of page images being sent.
        paper_context: Truncated paper text for context (abstract, methods).
        elements_measured: Optional list of element symbols (lowercase)
            from Paper Intelligence — constrains extraction scope.
        has_landscape: Whether any of the pages are landscape-oriented.
    """
    system = VISION_TABLE_EXTRACTION_SYSTEM_PROMPT.format(
        knowledge_base=get_knowledge_base_prompt(),
    )

    orientation_hint = ""
    if has_landscape:
        orientation_hint = (
            "\nIMPORTANT: Some pages are LANDSCAPE-ORIENTED (rotated 90°). "
            "These contain wide data tables with many element columns. "
            "Rotate mentally and read carefully — landscape tables often "
            "contain the bulk of the paper's geochemical data."
        )

    user = VISION_TABLE_EXTRACTION_USER_PROMPT.format(
        n_pages=n_pages,
        paper_context=paper_context[:6000] if paper_context else "(no text context available)",
        sample_format_instructions=_SAMPLE_FORMAT_INSTRUCTIONS,
        orientation_hint=orientation_hint,
    )

    # Append element scope constraint from Paper Intelligence
    if elements_measured:
        scope = ", ".join(sorted(elements_measured))
        user += (
            f"\n\n## ELEMENT SCOPE (from paper analysis)\n"
            f"This paper measures these specific elements: {scope}\n"
            f"Focus extraction on ONLY these elements. Use null for all others.\n"
        )

    return system, user


def build_full_extraction_prompt(
    paper_text: str,
    table_text: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one-shot full extraction."""
    metadata_fields = json.dumps(
        {k: None for k in PaperMetadata.model_fields.keys()},
        indent=2
    )
    user = FULL_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:10000],
        table_text=table_text[:8000],
        metadata_schema=metadata_fields,
    )
    return FULL_EXTRACTION_SYSTEM_PROMPT, user


def build_metadata_tool_schema() -> dict:
    """Return a valid Anthropic tool schema (JSON Schema draft 2020-12) for metadata extraction.

    All fields are Optional so every property is typed as ["<type>", "null"].
    Python type names are mapped to their JSON Schema equivalents:
        str   → "string"
        int   → "integer"
        float → "number"
    """
    _py_to_json = {"str": "string", "int": "integer", "float": "number"}

    # Skip coordinate fields (float, rarely in paper text)
    skip = {"deposit_longitude_wgs84", "deposit_latitude_wgs84",
            "sample_longitude_wgs84", "sample_latitude_wgs84"}

    properties = {}
    for field_name, field_info in PaperMetadata.model_fields.items():
        if field_name in skip:
            continue
        # Resolve the inner type of Optional[X] → get X
        annotation = field_info.annotation
        if hasattr(annotation, "__args__"):
            # Optional[X] is Union[X, None]; take the first non-None arg
            inner = next(
                (a for a in annotation.__args__ if a is not type(None)),
                str,
            )
        else:
            inner = annotation
        json_type = _py_to_json.get(getattr(inner, "__name__", "str"), "string")
        properties[field_name] = {
            "type": [json_type, "null"],
            "description": field_info.description or field_name,
        }

    return {
        "name": "extract_paper_metadata",
        "description": (
            "Extract paper-level geochemical metadata that applies to all samples. "
            "Use null for any field not explicitly stated in the paper."
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": [],
        },
    }
