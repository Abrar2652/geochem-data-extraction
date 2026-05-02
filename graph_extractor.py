"""
graph_extractor.py — Graph-based document representation for geochemistry papers.

Instead of flattening a paper into a 215-column spreadsheet, this represents
the paper as a knowledge graph where relationships are explicit:

  Paper → contains → Deposit(s)
  Deposit → contains → Sample(s)
  Sample → analyzed_by → Analysis(es)
  Analysis → measured → Measurement(s)
  Analysis → used_method → Method
  Analysis → target_mineral → Mineral
  Measurement → element → Element
  Measurement → value → float (in ppm)
  Measurement → original_value → float (as reported)
  Measurement → original_unit → str (wt%, ppm, ppb)
  Measurement → is_bdl → bool

Why this helps:
  1. One sample analyzed by EPMA + LA-ICPMS → two Analysis nodes, same Sample
     (flat file duplicates the entire row or merges losing method distinction)
  2. Reference data → separate Paper node with cited_by edge
     (flat file either removes it or tags it ambiguously)
  3. Hierarchical IDs: Sample → Grain → Spot are natural parent-child nodes
     (flat file collapses these into naming conventions like "K21-01-12@L3")
  4. Multiple minerals per analysis → Analysis→mineral edges
     (flat file forces one mineral per row, duplicating element data)

Output formats:
  - JSON graph (portable, human-readable)
  - NetworkX graph (for analysis in Python)
  - Neo4j-compatible CSV (for graph database import)
  - GraphML (for visualization tools)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Graph node types
# ──────────────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    PAPER = "paper"
    DEPOSIT = "deposit"
    SAMPLE = "sample"
    GRAIN = "grain"           # Physical grain within a sample
    ANALYSIS = "analysis"     # One analytical spot/measurement
    MEASUREMENT = "measurement"  # One element value within an analysis
    MINERAL = "mineral"
    METHOD = "method"
    ELEMENT = "element"
    LABORATORY = "laboratory"
    INSTRUMENT = "instrument"


class EdgeType(str, Enum):
    CONTAINS = "contains"
    ANALYZED_BY = "analyzed_by"
    MEASURED = "measured"
    TARGET_MINERAL = "target_mineral"
    USED_METHOD = "used_method"
    PERFORMED_AT = "performed_at"
    USED_INSTRUMENT = "used_instrument"
    CITES = "cites"
    LOCATED_IN = "located_in"
    CLASSIFIED_AS = "classified_as"
    HAS_STANDARD = "has_standard"
    PART_OF = "part_of"       # grain PART_OF sample


@dataclass
class GraphNode:
    """A node in the geochemistry knowledge graph."""
    id: str
    type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge connecting two nodes."""
    source: str     # Node ID
    target: str     # Node ID
    type: EdgeType
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeochemGraph:
    """Complete knowledge graph for a geochemistry paper."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    # Indexes for fast lookup
    _node_index: dict[str, GraphNode] = field(default_factory=dict, repr=False)

    def add_node(self, node: GraphNode) -> GraphNode:
        """Add a node, deduplicating by ID."""
        if node.id in self._node_index:
            # Merge properties
            existing = self._node_index[node.id]
            for k, v in node.properties.items():
                if v is not None and existing.properties.get(k) is None:
                    existing.properties[k] = v
            return existing
        self.nodes.append(node)
        self._node_index[node.id] = node
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add an edge."""
        self.edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._node_index.get(node_id)

    @property
    def stats(self) -> dict:
        type_counts = {}
        for n in self.nodes:
            type_counts[n.type.value] = type_counts.get(n.type.value, 0) + 1
        edge_counts = {}
        for e in self.edges:
            edge_counts[e.type.value] = edge_counts.get(e.type.value, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": type_counts,
            "edge_types": edge_counts,
        }

    # ── Export formats ────────────────────────────────────────────────────

    def to_json(self, path: Optional[str | Path] = None) -> dict:
        """Export as JSON."""
        data = {
            "nodes": [
                {"id": n.id, "type": n.type.value, "properties": n.properties}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "type": e.type.value,
                 "properties": e.properties}
                for e in self.edges
            ],
            "stats": self.stats,
        }
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Graph exported to %s (%d nodes, %d edges)",
                       path, len(self.nodes), len(self.edges))
        return data

    def to_neo4j_csv(self, output_dir: str | Path) -> None:
        """Export as Neo4j-compatible CSV files (nodes.csv + edges.csv)."""
        import csv
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nodes CSV
        with open(output_dir / "nodes.csv", "w", newline="") as f:
            writer = csv.writer(f)
            # Collect all property keys
            all_keys = set()
            for n in self.nodes:
                all_keys.update(n.properties.keys())
            header = ["id:ID", ":LABEL"] + sorted(all_keys)
            writer.writerow(header)
            for n in self.nodes:
                row = [n.id, n.type.value]
                for k in sorted(all_keys):
                    row.append(str(n.properties.get(k, "")))
                writer.writerow(row)

        # Edges CSV
        with open(output_dir / "edges.csv", "w", newline="") as f:
            writer = csv.writer(f)
            all_keys = set()
            for e in self.edges:
                all_keys.update(e.properties.keys())
            header = [":START_ID", ":END_ID", ":TYPE"] + sorted(all_keys)
            writer.writerow(header)
            for e in self.edges:
                row = [e.source, e.target, e.type.value]
                for k in sorted(all_keys):
                    row.append(str(e.properties.get(k, "")))
                writer.writerow(row)

        logger.info("Neo4j CSV exported to %s", output_dir)

    def to_graphml(self, path: str | Path) -> None:
        """Export as GraphML for visualization."""
        try:
            import networkx as nx
            G = nx.DiGraph()
            for n in self.nodes:
                G.add_node(n.id, type=n.type.value, **{
                    k: str(v) for k, v in n.properties.items() if v is not None
                })
            for e in self.edges:
                G.add_edge(e.source, e.target, type=e.type.value, **{
                    k: str(v) for k, v in e.properties.items() if v is not None
                })
            nx.write_graphml(G, str(path))
            logger.info("GraphML exported to %s", path)
        except ImportError:
            logger.warning("networkx not installed — cannot export GraphML")


# ──────────────────────────────────────────────────────────────────────────────
# Build graph from ExtractionResult
# ──────────────────────────────────────────────────────────────────────────────

def build_graph_from_extraction(result) -> GeochemGraph:
    """Convert a flat ExtractionResult into a knowledge graph.

    This is the bridge between the existing pipeline (flat file output)
    and the graph representation. It parses the flat rows back into
    their natural hierarchical structure.
    """
    from .schema import ELEMENT_SYMBOLS, BELOW_DETECTION_SENTINEL

    graph = GeochemGraph()
    meta = result.metadata

    # ── Paper node ────────────────────────────────────────────────────────
    paper_id = f"paper:{meta.sample_source or meta.deposit_name or 'unknown'}"
    graph.add_node(GraphNode(
        id=paper_id,
        type=NodeType.PAPER,
        properties={
            "title": meta.sample_source,
            "year": meta.publication_date,
            "country": meta.country,
            "state": meta.state,
        },
    ))

    # ── Deposit node ──────────────────────────────────────────────────────
    deposit_id = f"deposit:{meta.deposit_name or 'unknown'}"
    graph.add_node(GraphNode(
        id=deposit_id,
        type=NodeType.DEPOSIT,
        properties={
            "name": meta.deposit_name,
            "environment": meta.deposit_environment,
            "group": meta.deposit_group,
            "type": meta.deposit_type,
            "type_original": getattr(meta, 'deposit_type_original', None),
            "type_confidence": getattr(meta, 'deposit_type_confidence', None),
            "longitude": meta.deposit_longitude_wgs84,
            "latitude": meta.deposit_latitude_wgs84,
            "commodities": meta.all_commodities,
        },
    ))
    graph.add_edge(GraphEdge(paper_id, deposit_id, EdgeType.CONTAINS))

    # ── Method node(s) ────────────────────────────────────────────────────
    if meta.analytical_method:
        for method_str in meta.analytical_method.split(","):
            method_str = method_str.strip()
            if not method_str:
                continue
            method_id = f"method:{method_str}"
            graph.add_node(GraphNode(
                id=method_id,
                type=NodeType.METHOD,
                properties={"name": method_str},
            ))

    # ── Laboratory node ───────────────────────────────────────────────────
    if meta.laboratory_location:
        lab_id = f"lab:{meta.laboratory_location[:50]}"
        graph.add_node(GraphNode(
            id=lab_id,
            type=NodeType.LABORATORY,
            properties={
                "location": meta.laboratory_location,
                "instrument": meta.instrument_type_model,
                "conditions": meta.operating_conditions,
                "standards": meta.standards_used,
            },
        ))

    # ── Process each sample row ───────────────────────────────────────────
    seen_samples: dict[str, str] = {}  # sample_name → sample node ID
    seen_minerals: dict[str, str] = {}

    for row in result.samples:
        sample_name = row.sample_name or "unknown"
        sample_local_id = row.sample_local_id or ""
        analysis_id = getattr(row, 'analysis_id', '') or ""

        # Parse hierarchical ID: sample → grain → spot
        # Convention: "K21-01-12@L3" → sample=K21-01, grain=12, spot=L3
        base_sample = sample_name.split("@")[0].split("-spot")[0]

        # Sample node (deduplicated by base name)
        if base_sample not in seen_samples:
            sample_nid = f"sample:{base_sample}"
            graph.add_node(GraphNode(
                id=sample_nid,
                type=NodeType.SAMPLE,
                properties={
                    "name": base_sample,
                    "local_id": sample_local_id,
                    "deposit_relation": row.sample_deposit_relation,
                    "type": row.sample_type,
                    "source_tag": getattr(row, 'data_source_tag', 'this_study'),
                },
            ))
            graph.add_edge(GraphEdge(deposit_id, sample_nid, EdgeType.CONTAINS))
            seen_samples[base_sample] = sample_nid
        sample_nid = seen_samples[base_sample]

        # Analysis node (one per row — each row is one analytical spot)
        analysis_nid = f"analysis:{sample_name}:{analysis_id or id(row)}"
        row_method = row.analytical_method or meta.analytical_method or ""
        graph.add_node(GraphNode(
            id=analysis_nid,
            type=NodeType.ANALYSIS,
            properties={
                "sample_name": sample_name,
                "analysis_id": analysis_id,
                "method": row_method,
                "backend": getattr(row, 'extraction_backend', None),
                "confidence": getattr(row, 'extraction_confidence', None),
                "texture": row.texture,
                "source_tag": getattr(row, 'data_source_tag', 'this_study'),
            },
        ))
        graph.add_edge(GraphEdge(sample_nid, analysis_nid, EdgeType.ANALYZED_BY))

        # Link to method
        if row_method:
            method_nid = f"method:{row_method.split(',')[0].strip()}"
            if graph.get_node(method_nid):
                graph.add_edge(GraphEdge(analysis_nid, method_nid, EdgeType.USED_METHOD))

        # Mineral node
        mineral = row.mineral or meta.mineral
        if mineral:
            if mineral not in seen_minerals:
                mineral_nid = f"mineral:{mineral}"
                graph.add_node(GraphNode(
                    id=mineral_nid,
                    type=NodeType.MINERAL,
                    properties={"name": mineral},
                ))
                seen_minerals[mineral] = mineral_nid
            graph.add_edge(GraphEdge(
                analysis_nid, seen_minerals[mineral], EdgeType.TARGET_MINERAL,
            ))

        # Measurement nodes (one per element with data)
        for sym in ELEMENT_SYMBOLS:
            ppm_val = getattr(row, f"{sym}_ppm", None)
            if ppm_val is None:
                continue

            is_bdl = (ppm_val == BELOW_DETECTION_SENTINEL or
                      (isinstance(ppm_val, (int, float)) and ppm_val < 0))

            meas_nid = f"meas:{sample_name}:{sym}"
            # Get original value/unit if available
            orig_val = getattr(row, f"{sym}_original_value", None)
            orig_unit = getattr(row, f"{sym}_original_unit", None)
            det_limit = getattr(row, f"{sym}_detection_limit", None)

            graph.add_node(GraphNode(
                id=meas_nid,
                type=NodeType.MEASUREMENT,
                properties={
                    "element": sym,
                    "value_ppm": ppm_val if not is_bdl else None,
                    "is_bdl": is_bdl,
                    "bdl_value": ppm_val if is_bdl else None,
                    "original_value": orig_val,
                    "original_unit": orig_unit or "ppm",
                    "detection_limit": det_limit,
                },
            ))
            graph.add_edge(GraphEdge(analysis_nid, meas_nid, EdgeType.MEASURED))

            # Element node (shared across all measurements)
            elem_nid = f"element:{sym}"
            graph.add_node(GraphNode(
                id=elem_nid,
                type=NodeType.ELEMENT,
                properties={"symbol": sym},
            ))

    logger.info("Graph built: %s", graph.stats)
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# CLI integration
# ──────────────────────────────────────────────────────────────────────────────

def run_graph_extraction(
    pdf_path: str | Path,
    supplementary_paths: list[str | Path],
    client,
    output_dir: str | Path,
    output_format: str = "json",
) -> GeochemGraph:
    """Run the full pipeline and export as a knowledge graph.

    Args:
        pdf_path: Path to PDF
        supplementary_paths: Supplementary file paths
        client: LLM client
        output_dir: Where to save graph files
        output_format: "json", "neo4j", "graphml", or "all"
    """
    from .pipeline import ExtractionPipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run standard extraction
    pipe = ExtractionPipeline(llm_client=client)
    result = pipe.run(pdf_path=pdf_path, supplementary_paths=supplementary_paths)

    # Convert to graph
    graph = build_graph_from_extraction(result)

    # Export
    stem = Path(pdf_path).stem
    if output_format in ("json", "all"):
        graph.to_json(output_dir / f"{stem}_graph.json")
    if output_format in ("neo4j", "all"):
        graph.to_neo4j_csv(output_dir / f"{stem}_neo4j")
    if output_format in ("graphml", "all"):
        graph.to_graphml(output_dir / f"{stem}_graph.graphml")

    # Also save flat file for comparison
    result.to_excel(output_dir / f"{stem}_flat.xlsx")

    print(f"\nGraph extraction complete:")
    print(f"  Nodes: {graph.stats['total_nodes']}")
    print(f"  Edges: {graph.stats['total_edges']}")
    for ntype, count in sorted(graph.stats['node_types'].items()):
        print(f"    {ntype}: {count}")

    return graph
