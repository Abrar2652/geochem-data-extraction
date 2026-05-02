"""
graph_native_pipeline.py — Graph-native extraction: LLM builds a knowledge graph
directly from paper text, instead of parsing tables into flat rows.

Three-pass approach:
  Pass 1 (Entities):    What deposits, samples, minerals, methods exist?
  Pass 2 (Structure):   How are they related? Which sample used which method?
  Pass 3 (Values):      For each analysis, what are the element concentrations?

The graph is then flattened to the standard 364-column schema for evaluation.
This runs ALONGSIDE the table-native pipeline — both produce output for comparison.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import (
    PaperMetadata, SampleRow, ALL_COLUMNS, ELEMENT_SYMBOLS,
    BELOW_DETECTION_SENTINEL,
)
from .graph_extractor import GeochemGraph, GraphNode, GraphEdge, NodeType, EdgeType
from .knowledge_base import get_knowledge_base_prompt

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1: Entity Extraction
# ──────────────────────────────────────────────────────────────────────────────

ENTITY_SYSTEM = """\
You are an expert geochemist reading a research paper. Your task is to identify \
ALL entities (deposits, samples, minerals, methods) mentioned in the paper that \
relate to geochemical analyses performed by the authors.

Focus on data FROM THIS STUDY — not comparison/reference data from cited papers \
(but note those separately). Be exhaustive — every sample ID, every mineral \
analyzed, every method used."""

ENTITY_USER = """\
## PAPER TEXT
{paper_text}

## TASK
Extract all entities related to geochemical analyses in this paper.
Return a JSON object:

{{
  "paper": {{
    "title": "...",
    "authors": "...",
    "year": 2024,
    "journal": "..."
  }},
  "deposits": [
    {{
      "name": "Kangjiawan",
      "type_as_stated": "epithermal Pb-Zn",
      "environment": "...",
      "country": "CHN",
      "latitude": null,
      "longitude": null,
      "commodities": ["Zn", "Pb", "Au"]
    }}
  ],
  "samples": [
    {{
      "name": "K21-01",
      "deposit": "Kangjiawan",
      "type": "drill core",
      "description": "sphalerite from vein"
    }}
  ],
  "minerals_analyzed": ["sphalerite"],
  "methods_used": [
    {{
      "name": "EPMA",
      "instrument": "JEOL JXA-8230",
      "laboratory": "...",
      "conditions": "15kV, 20nA, 1um beam",
      "standards": "...",
      "elements_measured": ["Fe", "Zn", "S", "Mn", "Cu", "Cd"]
    }},
    {{
      "name": "LA-ICPMS",
      "instrument": "...",
      "laboratory": "...",
      "conditions": "44um spot, 6Hz",
      "standards": "...",
      "elements_measured": ["Fe", "Cu", "Pb", "As", "Bi", "Co", "Ni", "Ga", "Ge", "In", "Sb"]
    }}
  ],
  "cited_datasets": [
    {{
      "authors": "Foltyn et al.",
      "year": 2022,
      "what": "sphalerite trace element data for comparison"
    }}
  ],
  "total_analyses_reported": 127,
  "tables_described": [
    {{
      "label": "Table 2",
      "content": "EPMA major element data for sphalerite",
      "method": "EPMA",
      "n_rows_approx": 60
    }},
    {{
      "label": "Table 3",
      "content": "LA-ICPMS trace element data for sphalerite",
      "method": "LA-ICPMS",
      "n_rows_approx": 67
    }}
  ]
}}

CRITICAL:
- List ALL INDIVIDUAL spot/analysis IDs from the data tables, NOT just hand-sample names.
  Example: if Table 2 has rows K21-01-12, K21-01-13, K21-01-14 etc., list EACH ONE.
  Do NOT summarize as "K21-01" — list the actual row-level IDs.
- If you cannot list all individual IDs (too many), set "total_analyses_reported" to the count
  and list the FIRST 20 and LAST 5 IDs as examples.
- For elements_measured in each method, list EVERY element column header from the table.
Return ONLY the JSON object."""


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2: Structure — link samples to analyses and methods
# ──────────────────────────────────────────────────────────────────────────────

STRUCTURE_SYSTEM = """\
You are mapping the analytical structure of a geochemistry paper. Given a list of \
entities (samples, methods, minerals), determine the exact relationships between them.

For each sample, identify every analysis performed on it (which method, which \
mineral was the target). One sample may have multiple analyses (EPMA + LA-ICPMS)."""

STRUCTURE_USER = """\
## ENTITIES (extracted from paper)
{entities_json}

## PAPER TEXT (tables section)
{table_text}

## TASK
For each sample, list every analysis performed. Return a JSON object:

{{
  "analyses": [
    {{
      "sample": "K21-01-12",
      "method": "EPMA",
      "mineral": "sphalerite",
      "table_source": "Table 2",
      "is_this_study": true
    }},
    {{
      "sample": "K21-01-12",
      "method": "LA-ICPMS",
      "mineral": "sphalerite",
      "table_source": "Table 3",
      "is_this_study": true
    }},
    {{
      "sample": "Reference-A1",
      "method": "LA-ICPMS",
      "mineral": "sphalerite",
      "table_source": "Table 4",
      "is_this_study": false,
      "cited_from": "Foltyn et al. 2022"
    }}
  ]
}}

Include ALL analyses — both from this study and cited reference data. \
Mark each with is_this_study. This is critical for distinguishing real \
data from comparison data.
Return ONLY the JSON object."""


# ──────────────────────────────────────────────────────────────────────────────
# Pass 3: Value Extraction — targeted per analysis
# ──────────────────────────────────────────────────────────────────────────────

VALUE_SYSTEM = """\
You are extracting numerical geochemical data from a paper. For each analysis \
(sample + method combination), extract ALL element concentrations reported.

{knowledge_base}

USGS BDL protocol:
- Below detection limit, no specific LOD → -99999
- Below detection limit, specific LOD (e.g., <0.5) → negative of LOD (e.g., -0.5)
- "N/A" / "not analyzed" → null
- Convert all values to ppm: wt% × 10000, ppb ÷ 1000

Preserve original values: for each element report BOTH the ppm value AND the \
original value with its unit."""

VALUE_USER = """\
## PAPER TEXT (including tables)
{paper_text}

## MARKER TABLES (higher-fidelity table content)
{marker_tables}

## ANALYSES TO EXTRACT
{analyses_json}

## TASK
For each analysis listed above, extract all element concentrations.
Return a JSON object:

{{
  "measurements": [
    {{
      "sample": "K21-01-12",
      "method": "EPMA",
      "mineral": "sphalerite",
      "is_this_study": true,
      "elements": {{
        "fe": {{"ppm": 19000, "original": 1.9, "unit": "wt%"}},
        "zn": {{"ppm": 622800, "original": 62.28, "unit": "wt%"}},
        "s":  {{"ppm": 329700, "original": 32.97, "unit": "wt%"}},
        "mn": {{"ppm": 19700, "original": 1.97, "unit": "wt%"}},
        "cu": {{"ppm": -99999, "original": null, "unit": null, "bdl": true}},
        "as": {{"ppm": -99999, "original": null, "unit": null, "bdl": true}},
        "cd": {{"ppm": 600, "original": 0.06, "unit": "wt%"}}
      }}
    }},
    {{
      "sample": "K21-01-12",
      "method": "LA-ICPMS",
      "mineral": "sphalerite",
      "is_this_study": true,
      "elements": {{
        "fe": {{"ppm": 10415, "original": 10415, "unit": "ppm"}},
        "cu": {{"ppm": 123.5, "original": 123.5, "unit": "ppm"}}
      }}
    }}
  ]
}}

CRITICAL RULES:
- Extract EVERY element reported for each analysis — do not skip any
- wt% values: "fe_ppm" = original_wt% × 10000
- BDL values ("-", "bdl", "n.d.") → ppm = -99999, bdl = true
- BDL with LOD ("<0.5") → ppm = -0.5, bdl = true
- null = element was NOT measured at all (different from BDL)
- Maintain EXACT order as they appear in the paper's tables
Return ONLY the JSON object."""


# ──────────────────────────────────────────────────────────────────────────────
# Graph-Native Pipeline
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphNativeResult:
    """Output of graph-native extraction."""
    graph: GeochemGraph
    entities: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)
    measurements: list[dict] = field(default_factory=list)
    flat_samples: list[SampleRow] = field(default_factory=list)
    metadata: Optional[PaperMetadata] = None
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [s.to_schema_dict() for s in self.flat_samples]
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    def to_excel(self, path: str | Path, export_minmod: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_excel(str(path), index=False)
        if export_minmod and len(df) > 0:
            try:
                from .minmod_exporter import extraction_to_mineral_site, export_minmod_json, export_jsonld, export_turtle
                paper_id = path.stem.replace("extraction_", "")
                site = extraction_to_mineral_site(df, paper_id)
                if site:
                    out_dir = path.parent
                    export_minmod_json([site], out_dir / f"minmod_{path.stem}.json")
                    export_jsonld([site], out_dir / f"minmod_{path.stem}.jsonld")
                    export_turtle([site], out_dir / f"minmod_{path.stem}.ttl")
            except Exception:
                pass
        return path


def run_graph_native(
    client,
    pdf_path: str | Path,
    supplementary_paths: list[str | Path] | None = None,
) -> GraphNativeResult:
    """Run the 3-pass graph-native extraction pipeline.

    Pass 1: Entity extraction (deposits, samples, minerals, methods)
    Pass 2: Structure mapping (sample→analysis→method relationships)
    Pass 3: Value extraction (element concentrations per analysis)

    Then converts graph → flat SampleRows for evaluation compatibility.
    """
    from .pdf_reader import extract_pdf, get_paper_text_for_llm

    result = GraphNativeResult(graph=GeochemGraph())

    # ── Read PDF ──────────────────────────────────────────────────────────
    logger.info("Graph-native: reading PDF")
    pdf_content = extract_pdf(pdf_path)
    paper_text = get_paper_text_for_llm(pdf_content, max_chars=25000)

    # Get Marker tables for Pass 3 (higher fidelity)
    marker_tables = ""
    if pdf_content.marker_markdown:
        from .pdf_reader import _extract_marker_table_sections
        marker_tables = _extract_marker_table_sections(pdf_content.marker_markdown, max_chars=12000)

    # ── Pass 1: Entity Extraction ─────────────────────────────────────────
    logger.info("Graph-native Pass 1: extracting entities")
    # Include Marker tables in Pass 1 so LLM can see actual sample IDs
    pass1_text = paper_text[:15000]
    if marker_tables:
        pass1_text += "\n\n## TABLE CONTENT (from Marker OCR)\n" + marker_tables[:10000]

    try:
        entities = client.complete_json(
            system=ENTITY_SYSTEM,
            user=ENTITY_USER.format(paper_text=pass1_text),
            max_tokens=8192,
        )
        if isinstance(entities, dict):
            result.entities = entities
            n_samples = len(entities.get("samples", []))
            n_methods = len(entities.get("methods_used", []))
            result.notes.append(f"Pass 1: {n_samples} samples, {n_methods} methods identified")
            logger.info("Pass 1: %d samples, %d methods, %d tables described",
                       n_samples, n_methods, len(entities.get("tables_described", [])))
    except Exception as e:
        result.errors.append(f"Pass 1 failed: {e}")
        logger.warning("Pass 1 failed: %s", e)
        return result

    # ── Pass 2: Structure Mapping ─────────────────────────────────────────
    logger.info("Graph-native Pass 2: mapping structure")
    try:
        structure = client.complete_json(
            system=STRUCTURE_SYSTEM,
            user=STRUCTURE_USER.format(
                entities_json=json.dumps(result.entities, indent=2)[:8000],
                table_text=paper_text[:12000],
            ),
            max_tokens=8192,
        )
        if isinstance(structure, dict):
            result.structure = structure
            n_analyses = len(structure.get("analyses", []))
            n_this_study = sum(1 for a in structure.get("analyses", []) if a.get("is_this_study", True))
            result.notes.append(f"Pass 2: {n_analyses} analyses ({n_this_study} this study)")
            logger.info("Pass 2: %d analyses (%d this study)", n_analyses, n_this_study)
    except Exception as e:
        result.errors.append(f"Pass 2 failed: {e}")
        logger.warning("Pass 2 failed: %s", e)

    # ── Pass 3: Value Extraction ──────────────────────────────────────────
    # Process in chunks to stay within output token limits
    analyses = result.structure.get("analyses", [])
    if not analyses:
        result.notes.append("Pass 3: no analyses to extract")
        return result

    logger.info("Graph-native Pass 3: extracting values for %d analyses", len(analyses))
    all_measurements = []

    # Chunk analyses into batches of 10 (keep output JSON manageable)
    chunk_size = 10
    for chunk_start in range(0, len(analyses), chunk_size):
        chunk = analyses[chunk_start:chunk_start + chunk_size]
        chunk_label = f"{chunk_start+1}-{min(chunk_start+chunk_size, len(analyses))}"

        try:
            values = client.complete_json(
                system=VALUE_SYSTEM.format(knowledge_base=get_knowledge_base_prompt()),
                user=VALUE_USER.format(
                    paper_text=paper_text[:15000],
                    marker_tables=marker_tables[:10000],
                    analyses_json=json.dumps(chunk, indent=2),
                ),
                max_tokens=32768,
            )
            if isinstance(values, dict):
                batch = values.get("measurements", [])
                all_measurements.extend(batch)
                logger.info("Pass 3 chunk %s: %d measurements", chunk_label, len(batch))
        except Exception as e:
            result.errors.append(f"Pass 3 chunk {chunk_label} failed: {e}")
            logger.warning("Pass 3 chunk %s failed: %s", chunk_label, e)

    result.measurements = all_measurements
    result.notes.append(f"Pass 3: {len(all_measurements)} measurements extracted")

    # ── Build graph + flat samples ────────────────────────────────────────
    _build_graph(result)
    _flatten_to_samples(result)

    logger.info("Graph-native complete: %d graph nodes, %d flat samples",
                len(result.graph.nodes), len(result.flat_samples))

    return result


def _build_graph(result: GraphNativeResult) -> None:
    """Build the knowledge graph from extracted entities + measurements."""
    graph = result.graph
    entities = result.entities

    # Paper node
    paper_info = entities.get("paper", {})
    paper_nid = f"paper:{paper_info.get('title', 'unknown')[:50]}"
    graph.add_node(GraphNode(paper_nid, NodeType.PAPER, paper_info))

    # Deposit nodes
    for dep in entities.get("deposits", []):
        dep_nid = f"deposit:{dep.get('name', 'unknown')}"
        graph.add_node(GraphNode(dep_nid, NodeType.DEPOSIT, dep))
        graph.add_edge(GraphEdge(paper_nid, dep_nid, EdgeType.CONTAINS))

    # Method nodes
    for method in entities.get("methods_used", []):
        method_nid = f"method:{method.get('name', 'unknown')}"
        graph.add_node(GraphNode(method_nid, NodeType.METHOD, method))

    # Mineral nodes
    for mineral in entities.get("minerals_analyzed", []):
        mineral_nid = f"mineral:{mineral}"
        graph.add_node(GraphNode(mineral_nid, NodeType.MINERAL, {"name": mineral}))

    # Sample nodes
    for sample in entities.get("samples", []):
        sample_nid = f"sample:{sample.get('name', 'unknown')}"
        graph.add_node(GraphNode(sample_nid, NodeType.SAMPLE, sample))
        dep_name = sample.get("deposit", entities.get("deposits", [{}])[0].get("name", "unknown") if entities.get("deposits") else "unknown")
        dep_nid = f"deposit:{dep_name}"
        if graph.get_node(dep_nid):
            graph.add_edge(GraphEdge(dep_nid, sample_nid, EdgeType.CONTAINS))

    # Analysis + measurement nodes from Pass 3
    for meas in result.measurements:
        sample_name = meas.get("sample", "unknown")
        method_name = meas.get("method", "unknown")
        mineral_name = meas.get("mineral", "")
        is_this_study = meas.get("is_this_study", True)

        analysis_nid = f"analysis:{sample_name}:{method_name}"
        graph.add_node(GraphNode(analysis_nid, NodeType.ANALYSIS, {
            "sample": sample_name,
            "method": method_name,
            "mineral": mineral_name,
            "is_this_study": is_this_study,
        }))

        # Link analysis → sample
        sample_base = sample_name.split("@")[0].split("-spot")[0]
        sample_nid = f"sample:{sample_base}"
        if not graph.get_node(sample_nid):
            graph.add_node(GraphNode(sample_nid, NodeType.SAMPLE, {"name": sample_base}))
        graph.add_edge(GraphEdge(sample_nid, analysis_nid, EdgeType.ANALYZED_BY))

        # Link analysis → method
        method_nid = f"method:{method_name}"
        if graph.get_node(method_nid):
            graph.add_edge(GraphEdge(analysis_nid, method_nid, EdgeType.USED_METHOD))

        # Link analysis → mineral
        if mineral_name:
            mineral_nid = f"mineral:{mineral_name}"
            if graph.get_node(mineral_nid):
                graph.add_edge(GraphEdge(analysis_nid, mineral_nid, EdgeType.TARGET_MINERAL))

        # Measurement nodes
        elements = meas.get("elements") or {}
        if not isinstance(elements, dict):
            continue
        for sym, val_info in elements.items():
            if not isinstance(val_info, dict):
                continue
            meas_nid = f"meas:{sample_name}:{method_name}:{sym}"
            graph.add_node(GraphNode(meas_nid, NodeType.MEASUREMENT, {
                "element": sym.lower(),
                "value_ppm": val_info.get("ppm"),
                "original_value": val_info.get("original"),
                "original_unit": val_info.get("unit", "ppm"),
                "is_bdl": val_info.get("bdl", False),
            }))
            graph.add_edge(GraphEdge(analysis_nid, meas_nid, EdgeType.MEASURED))


def _flatten_to_samples(result: GraphNativeResult) -> None:
    """Convert graph measurements into flat SampleRow objects for evaluation."""
    entities = result.entities
    deposits = entities.get("deposits", [])
    methods_info = {m.get("name", ""): m for m in entities.get("methods_used", [])}

    # Build metadata from entities
    dep = deposits[0] if deposits else {}
    paper_info = entities.get("paper", {})

    metadata = PaperMetadata(
        deposit_name=dep.get("name"),
        deposit_type_original=dep.get("type_as_stated"),
        deposit_environment=dep.get("environment"),
        country=dep.get("country"),
        all_commodities=", ".join(dep.get("commodities", [])),
        mineral=", ".join(entities.get("minerals_analyzed", [])),
        publication_date=paper_info.get("year"),
        sample_source=f"{paper_info.get('authors', '')} ({paper_info.get('year', '')}), {paper_info.get('title', '')}",
    )

    # Build one SampleRow per measurement entry
    samples = []
    for meas in result.measurements:
        sample_name = meas.get("sample", "")
        method_name = meas.get("method", "")
        mineral_name = meas.get("mineral", "")
        is_this_study = meas.get("is_this_study", True)

        row_data = {
            "sample_name": sample_name,
            "deposit_name": dep.get("name"),
            "deposit_environment": dep.get("environment"),
            "mineral": mineral_name or metadata.mineral,
            "analytical_method": method_name,
            "country": dep.get("country"),
            "publication_date": paper_info.get("year"),
            "sample_source": metadata.sample_source,
            "data_source_tag": "this_study" if is_this_study else f"cited_study:{meas.get('cited_from', '')}",
        }

        # Add method details
        minfo = methods_info.get(method_name, {})
        if minfo:
            row_data["instrument_type_model"] = minfo.get("instrument")
            row_data["laboratory_location"] = minfo.get("laboratory")
            row_data["operating_conditions"] = minfo.get("conditions")
            row_data["standards_used"] = minfo.get("standards")

        # Add element values
        elements = meas.get("elements") or {}
        if not isinstance(elements, dict):
            elements = {}
        for sym, val_info in elements.items():
            if not isinstance(val_info, dict):
                continue
            sym_lower = sym.lower()
            if sym_lower not in ELEMENT_SYMBOLS:
                continue
            ppm_val = val_info.get("ppm")
            if ppm_val is not None:
                row_data[f"{sym_lower}_ppm"] = float(ppm_val)
            orig = val_info.get("original")
            if orig is not None:
                row_data[f"{sym_lower}_original_value"] = float(orig)
            unit = val_info.get("unit")
            if unit:
                row_data[f"{sym_lower}_original_unit"] = unit

        # Build SampleRow
        known = set(SampleRow.model_fields.keys())
        filtered = {k: v for k, v in row_data.items() if k in known or "_original_" in k}
        try:
            samples.append(SampleRow(**filtered))
        except Exception as e:
            logger.debug("Failed to build SampleRow for %s: %s", sample_name, e)

    result.flat_samples = samples
    result.metadata = metadata
    result.notes.append(f"Flattened: {len(samples)} rows from {len(result.measurements)} measurements")
