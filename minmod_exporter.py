"""
minmod_exporter.py — Export geochem extraction results in MinMod-compatible formats.

Formats supported:
  1. MinMod JSON   — {"MineralSite": [...]} as used by DARPA CRITICALMAAS CDR
  2. JSON-LD       — Linked Data version with @context for MinMod ontology
  3. Turtle (RDF)  — For SPARQL endpoint ingestion at minmod.isi.edu

Schema reference:
  https://github.com/DARPA-CRITICALMAAS/cdr_schemas
  https://github.com/DARPA-CRITICALMAAS/sri-ta2/blob/main/output_minmod_json.py

MinMod MineralSite structure:
  {
    "source_id": str,          # Source database URL
    "record_id": str,          # Unique record ID
    "name": str,               # Deposit name
    "site_rank": str,          # Grade/size rank
    "site_type": str,          # Type of site
    "location": {...},         # GeoJSON geometry + CRS
    "deposit_type_candidate": [# Ranked deposit classifications
      {
        "observed_name": str,
        "confidence": float,
        "normalized_uri": str, # https://minmod.isi.edu/resource/Q{id}
        "source": str,
      }
    ],
    "mineral_inventory": [     # Commodities with MinMod URIs
      {
        "commodity": {
          "observed_name": str,
          "normalized_uri": str,
          "source": str,
          "confidence": float,
        },
        "reference": {"document": {"uri": str}},
      }
    ]
  }
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Load MinMod lookup tables from sri-ta2
# ──────────────────────────────────────────────────────────────────────────────

_DEPOSIT_TYPE_MAP: dict[str, str] = {}   # name → minmod_id (Q301...)
_COMMODITY_MAP: dict[str, str] = {}      # commodity_name → minmod_id (Q501...)
_LOOKUPS_LOADED = False

MINMOD_BASE_URI = "https://minmod.isi.edu/resource/"
PIPELINE_SOURCE = "geochem-benchmark extraction pipeline v1, Jahin et al."


def _load_lookups() -> None:
    global _DEPOSIT_TYPE_MAP, _COMMODITY_MAP, _LOOKUPS_LOADED
    if _LOOKUPS_LOADED:
        return
    base = Path(__file__).parent / "sri-ta2"

    # Deposit types
    dt_path = base / "minmod" / "deposit_type.csv"
    if dt_path.exists():
        df = pd.read_csv(dt_path)
        for _, row in df.iterrows():
            name = str(row["Deposit type"]).strip()
            mid = str(row["Minmod ID"]).strip()
            _DEPOSIT_TYPE_MAP[name.lower()] = mid
            _DEPOSIT_TYPE_MAP[name] = mid  # also exact case

    # Commodities
    comm_path = base / "minmod" / "commodity.csv"
    if comm_path.exists():
        df = pd.read_csv(comm_path)
        for _, row in df.iterrows():
            mid = str(row["minmod_id"]).strip()
            for col in ["CommodityinMRDS", "CommodityinGeoKb"]:
                if col in df.columns:
                    name = str(row.get(col, "")).strip()
                    if name and name != "nan":
                        _COMMODITY_MAP[name.lower()] = mid
                        _COMMODITY_MAP[name] = mid

    _LOOKUPS_LOADED = True
    logger.info("MinMod lookups: %d deposit types, %d commodities",
                len(_DEPOSIT_TYPE_MAP) // 2, len(_COMMODITY_MAP) // 2)


def _deposit_type_uri(name: str) -> Optional[str]:
    _load_lookups()
    mid = _DEPOSIT_TYPE_MAP.get(name) or _DEPOSIT_TYPE_MAP.get(name.lower())
    return f"{MINMOD_BASE_URI}{mid}" if mid else None


def _commodity_uri(name: str) -> Optional[str]:
    _load_lookups()
    # Try various forms: "Zinc", "zinc", "Zn"
    mid = (_COMMODITY_MAP.get(name)
           or _COMMODITY_MAP.get(name.lower())
           or _COMMODITY_MAP.get(name.capitalize()))
    return f"{MINMOD_BASE_URI}{mid}" if mid else None


# Common commodity name normalisation (element symbol → MRDS commodity name)
_ELEMENT_TO_COMMODITY = {
    "zn": "Zinc", "pb": "Lead", "cu": "Copper", "ag": "Silver",
    "au": "Gold", "fe": "Iron", "ni": "Nickel", "co": "Cobalt",
    "mn": "Manganese", "mo": "Molybdenum", "w": "Tungsten",
    "sn": "Tin", "bi": "Bismuth", "as": "Arsenic", "sb": "Antimony",
    "in": "Indium", "ga": "Gallium", "ge": "Germanium",
    "se": "Selenium", "te": "Tellurium", "re": "Rhenium",
    "cd": "Cadmium", "hg": "Mercury", "tl": "Thallium",
    "pt": "Platinum", "pd": "Palladium", "rh": "Rhodium",
    "li": "Lithium", "be": "Beryllium", "rb": "Rubidium",
    "cs": "Cesium", "ba": "Barium", "sr": "Strontium",
    "y": "Yttrium", "nb": "Niobium", "ta": "Tantalum",
    "cr": "Chromium", "v": "Vanadium", "ti": "Titanium",
    "sc": "Scandium", "hf": "Hafnium", "zr": "Zirconium",
    "th": "Thorium", "u": "Uranium",
    "la": "Lanthanum", "ce": "Cerium", "nd": "Neodymium",
    "pr": "Praseodymium", "sm": "Samarium", "eu": "Europium",
    "gd": "Gadolinium", "tb": "Terbium", "dy": "Dysprosium",
    "ho": "Holmium", "er": "Erbium", "yb": "Ytterbium", "lu": "Lutetium",
    "s": "Sulfur", "p": "Phosphorus",
}


# ──────────────────────────────────────────────────────────────────────────────
# Build MineralSite record from extraction result
# ──────────────────────────────────────────────────────────────────────────────

def extraction_to_mineral_site(
    extraction_df: pd.DataFrame,
    paper_id: str,
    source_uri: str = "",
    publication_doi: str = "",
) -> dict:
    """Convert a flat extraction DataFrame to a MinMod MineralSite record.

    Args:
        extraction_df: 364-column extraction DataFrame (one row per analysis)
        paper_id: Paper identifier (e.g., "Yuan_et_al_2018")
        source_uri: DOI or URL of the paper (used as document reference)
        publication_doi: DOI if known

    Returns:
        MineralSite dict ready for MinMod JSON serialisation.
    """
    _load_lookups()
    if extraction_df.empty:
        return {}

    row = extraction_df.iloc[0]  # Paper-level fields from first row

    def get(field: str, default="") -> str:
        val = row.get(field, default)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return str(val).strip()

    site = {}

    # ── Identifiers ───────────────────────────────────────────────────────
    site["source_id"] = source_uri or f"geochem-benchmark:{paper_id}"
    site["record_id"] = paper_id
    deposit_name = get("deposit_name")
    if deposit_name:
        site["name"] = deposit_name

    # ── Location ──────────────────────────────────────────────────────────
    lon = get("deposit_longitude_wgs84")
    lat = get("deposit_latitude_wgs84")
    if lon and lat:
        try:
            site["location"] = {
                "crs": "EPSG:4326",
                "geom": f"POINT({float(lon)} {float(lat)})"
            }
        except (ValueError, TypeError):
            pass

    country = get("country")
    state = get("state")
    if country:
        site["country"] = [country]
    if state:
        site["province"] = [state]

    # ── Deposit type candidates ───────────────────────────────────────────
    dep_type = get("deposit_type")
    dep_conf_raw = row.get("deposit_type_confidence")
    dep_conf = float(dep_conf_raw) if dep_conf_raw and not pd.isna(dep_conf_raw) else 0.5
    dep_alts = get("deposit_type_alternatives")
    dep_reasoning = get("deposit_type_reasoning")

    candidates = []
    if dep_type:
        uri = _deposit_type_uri(dep_type)
        cand = {
            "observed_name": dep_type,
            "confidence": round(dep_conf, 4),
            "source": PIPELINE_SOURCE,
        }
        if uri:
            cand["normalized_uri"] = uri
        if dep_reasoning:
            cand["justification"] = dep_reasoning[:200]
        candidates.append(cand)

    # Add alternatives (format: "Type A (0.25) | Type B (0.10)")
    if dep_alts:
        for alt_str in dep_alts.split("|"):
            alt_str = alt_str.strip()
            if not alt_str:
                continue
            # Parse "MVT zinc-lead (0.25)"
            import re
            m = re.match(r"^(.+?)\s*\(([\d.]+)\)$", alt_str)
            if m:
                alt_name = m.group(1).strip()
                alt_conf = float(m.group(2))
                alt_uri = _deposit_type_uri(alt_name)
                alt_cand = {
                    "observed_name": alt_name,
                    "confidence": round(alt_conf, 4),
                    "source": PIPELINE_SOURCE,
                }
                if alt_uri:
                    alt_cand["normalized_uri"] = alt_uri
                candidates.append(alt_cand)

    if candidates:
        site["deposit_type_candidate"] = candidates

    # ── Mineral inventory (commodities) ───────────────────────────────────
    doc_ref = {"document": {"uri": publication_doi or source_uri or f"geochem-benchmark:{paper_id}"}}

    inventory = []
    all_commodities = get("all_commodities")
    if all_commodities:
        for comm_name in all_commodities.split(","):
            comm_name = comm_name.strip()
            if not comm_name:
                continue
            uri = _commodity_uri(comm_name)
            item = {
                "commodity": {
                    "observed_name": comm_name,
                    "confidence": 1.0,
                    "source": PIPELINE_SOURCE,
                },
                "reference": doc_ref,
            }
            if uri:
                item["commodity"]["normalized_uri"] = uri
            inventory.append(item)

    # Also add minerals analyzed (if distinct from commodities)
    mineral = get("mineral")
    if mineral:
        for m in mineral.split(","):
            m = m.strip()
            if m:
                inventory.append({
                    "commodity": {
                        "observed_name": m,
                        "confidence": 1.0,
                        "source": PIPELINE_SOURCE,
                        "commodity_type": "analyzed_mineral",
                    },
                    "reference": doc_ref,
                })

    if inventory:
        site["mineral_inventory"] = inventory

    # ── Geochemical measurements (extended MinMod) ────────────────────────
    # Standard MinMod doesn't have per-sample geochemistry but we add it
    # as an extension for the CMMI mineral chemistry database
    from .schema import ELEMENT_SYMBOLS, BELOW_DETECTION_SENTINEL
    measurements = []
    for sym in ELEMENT_SYMBOLS:
        col = f"{sym}_ppm"
        if col not in extraction_df.columns:
            continue
        vals = extraction_df[col].dropna()
        real_vals = vals[(vals != BELOW_DETECTION_SENTINEL) & (vals >= 0)]
        if len(real_vals) == 0:
            continue

        comm_name = _ELEMENT_TO_COMMODITY.get(sym, sym.upper())
        uri = _commodity_uri(comm_name)
        meas = {
            "commodity": {"observed_name": comm_name},
            "n_analyses": int(len(vals)),
            "unit": "ppm",
            "median_value": round(float(real_vals.median()), 4),
            "min_value": round(float(real_vals.min()), 4),
            "max_value": round(float(real_vals.max()), 4),
            "n_bdl": int((vals == BELOW_DETECTION_SENTINEL).sum()),
            "reference": doc_ref,
        }
        if uri:
            meas["commodity"]["normalized_uri"] = uri
        measurements.append(meas)

    if measurements:
        site["geochemical_measurements"] = measurements

    # ── Analytical metadata ────────────────────────────────────────────────
    method = get("analytical_method")
    instrument = get("instrument_type_model")
    lab = get("laboratory_location/if reported") or get("laboratory_location")
    standards = get("standards_used/if reported") or get("standards_used")

    if method or instrument or lab:
        site["analytical_methods"] = [{
            "method": method,
            "instrument": instrument,
            "laboratory": lab,
            "standards": standards,
            "source": PIPELINE_SOURCE,
        }]

    pub_year = row.get("publication_date")
    if pub_year and not pd.isna(pub_year):
        site["publication_year"] = int(pub_year)

    return site


# ──────────────────────────────────────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────────────────────────────────────

def export_minmod_json(
    sites: list[dict],
    output_path: str | Path,
) -> Path:
    """Export as MinMod JSON: {"MineralSite": [...]}"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"MineralSite": sites}, f, indent=2, default=str)
    logger.info("MinMod JSON: %d sites → %s", len(sites), output_path)
    return output_path


def export_jsonld(
    sites: list[dict],
    output_path: str | Path,
) -> Path:
    """Export as JSON-LD with MinMod ontology context.

    Adds @context so the document is a valid Linked Data graph compatible
    with the MinMod knowledge graph at https://minmod.isi.edu/.
    """
    context = {
        "@vocab": "https://minmod.isi.edu/ontology/",
        "minmod": "https://minmod.isi.edu/resource/",
        "schema": "https://schema.org/",
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
        "MineralSite": {"@id": "https://minmod.isi.edu/ontology/MineralSite"},
        "deposit_type_candidate": {"@id": "https://minmod.isi.edu/ontology/deposit_type_candidate"},
        "mineral_inventory": {"@id": "https://minmod.isi.edu/ontology/mineral_inventory"},
        "observed_name": {"@id": "rdfs:label"},
        "normalized_uri": {"@id": "@id"},
        "confidence": {
            "@id": "https://minmod.isi.edu/ontology/confidence",
            "@type": "xsd:decimal",
        },
        "source_id": {"@id": "schema:url"},
        "record_id": {"@id": "schema:identifier"},
        "name": {"@id": "schema:name"},
        "country": {"@id": "schema:addressCountry"},
        "province": {"@id": "schema:addressRegion"},
        "publication_year": {
            "@id": "schema:datePublished",
            "@type": "xsd:integer",
        },
        "commodity": {"@id": "https://minmod.isi.edu/ontology/commodity"},
        "unit": {"@id": "schema:unitText"},
        "median_value": {"@id": "schema:value", "@type": "xsd:decimal"},
    }

    # Add @id to each site
    sites_ld = []
    for i, site in enumerate(sites):
        site_ld = dict(site)
        record_id = site.get("record_id", f"site_{i}")
        site_ld["@id"] = f"https://minmod.isi.edu/resource/{record_id}"
        site_ld["@type"] = "MineralSite"
        sites_ld.append(site_ld)

    doc = {
        "@context": context,
        "@graph": sites_ld,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    logger.info("JSON-LD: %d sites → %s", len(sites), output_path)
    return output_path


def export_turtle(
    sites: list[dict],
    output_path: str | Path,
) -> Path:
    """Export as RDF Turtle for SPARQL ingestion at minmod.isi.edu.

    Produces a valid Turtle document with MinMod predicates.
    """
    lines = [
        "@prefix minmod: <https://minmod.isi.edu/resource/> .",
        "@prefix mno: <https://minmod.isi.edu/ontology/> .",
        "@prefix schema: <https://schema.org/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> .",
        "",
    ]

    def escape_str(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    for site in sites:
        record_id = str(site.get("record_id", "unknown"))
        # Create a safe URI-safe ID
        safe_id = record_id.replace(" ", "_").replace("/", "_").replace(":", "_")
        subj = f"minmod:{safe_id}"

        lines.append(f"{subj}")
        lines.append(f"    a mno:MineralSite ;")
        lines.append(f'    schema:identifier "{escape_str(record_id)}" ;')

        name = site.get("name", "")
        if name:
            lines.append(f'    rdfs:label "{escape_str(name)}"@en ;')

        source_id = site.get("source_id", "")
        if source_id:
            lines.append(f'    schema:url <{source_id}> ;')

        for country in site.get("country", []):
            lines.append(f'    schema:addressCountry "{escape_str(country)}" ;')

        for province in site.get("province", []):
            lines.append(f'    schema:addressRegion "{escape_str(province)}" ;')

        if "publication_year" in site:
            lines.append(f'    schema:datePublished "{site["publication_year"]}"^^xsd:integer ;')

        # Deposit type candidates as blank nodes
        for i, cand in enumerate(site.get("deposit_type_candidate", [])):
            bn = f"_:deptype_{safe_id}_{i}"
            lines.append(f"    mno:deposit_type_candidate {bn} ;")
            lines.append(f"{bn}")
            lines.append(f'    rdfs:label "{escape_str(cand.get("observed_name", ""))}"@en ;')
            conf = cand.get("confidence", 0)
            lines.append(f'    mno:confidence "{conf}"^^xsd:decimal ;')
            lines.append(f'    mno:source "{escape_str(cand.get("source", ""))}" ;')
            uri = cand.get("normalized_uri", "")
            if uri:
                lines.append(f"    mno:normalized_uri <{uri}> ;")
            lines.append("    .")

        # Mineral inventory as blank nodes
        for i, inv in enumerate(site.get("mineral_inventory", [])):
            comm = inv.get("commodity", {})
            bn = f"_:inv_{safe_id}_{i}"
            lines.append(f"    mno:mineral_inventory {bn} ;")
            comm_name = escape_str(comm.get("observed_name", ""))
            lines.append(f"{bn}")
            lines.append(f'    rdfs:label "{comm_name}"@en ;')
            comm_uri = comm.get("normalized_uri", "")
            if comm_uri:
                lines.append(f"    mno:normalized_uri <{comm_uri}> ;")
            ref_uri = inv.get("reference", {}).get("document", {}).get("uri", "")
            if ref_uri:
                lines.append(f"    schema:url <{ref_uri}> ;")
            lines.append("    .")

        # Close subject
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Turtle: %d sites → %s", len(sites), output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Batch export
# ──────────────────────────────────────────────────────────────────────────────

def export_batch_minmod(
    extraction_dir: str | Path,
    output_dir: str | Path,
    formats: list[str] | None = None,
    source_uri_map: dict[str, str] | None = None,
) -> dict:
    """Export all extraction files in a directory to MinMod formats.

    Args:
        extraction_dir: Directory with extraction_*.xlsx files
        output_dir: Where to save MinMod output files
        formats: ["json", "jsonld", "turtle"] — default all three
        source_uri_map: Optional mapping from paper_id → DOI/URL

    Returns:
        Stats dict.
    """
    extraction_dir = Path(extraction_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["json", "jsonld", "turtle"]
    source_uri_map = source_uri_map or {}

    all_sites = []
    stats = {"papers": 0, "sites": 0, "failed": 0}

    for ex_file in sorted(extraction_dir.glob("extraction_*.xlsx")):
        paper_id = ex_file.stem.replace("extraction_", "")
        try:
            df = pd.read_excel(ex_file)
            if df.empty:
                continue
            source_uri = source_uri_map.get(paper_id, "")
            site = extraction_to_mineral_site(df, paper_id, source_uri)
            if site:
                all_sites.append(site)
                stats["papers"] += 1
        except Exception as e:
            logger.warning("Failed to convert %s: %s", paper_id, e)
            stats["failed"] += 1

    stats["sites"] = len(all_sites)

    if not all_sites:
        logger.warning("No sites to export")
        return stats

    if "json" in formats:
        export_minmod_json(all_sites, output_dir / "mineral_sites_minmod.json")
    if "jsonld" in formats:
        export_jsonld(all_sites, output_dir / "mineral_sites_minmod.jsonld")
    if "turtle" in formats:
        export_turtle(all_sites, output_dir / "mineral_sites_minmod.ttl")

    logger.info("MinMod export: %d papers → %d sites", stats["papers"], stats["sites"])
    return stats
