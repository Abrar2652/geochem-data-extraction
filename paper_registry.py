"""
paper_registry.py - Maps ground truth files to their corresponding PDF + supplementary data.

Each entry defines:
  - id:             Unique paper identifier
  - ground_truth:   Filename in ground_truth/ folder
  - pdf:            PDF filename(s) in data/ folder
  - supplementary:  Supplementary filenames in data/Spreadsheets/ folder
  - notes:          Any special handling notes

The registry only includes papers that have BOTH a ground truth file and
processable supplementary data (xlsx/csv). Papers with zip-only or docx-only
supplements are included but flagged — the pipeline will attempt zip extraction.
"""

from __future__ import annotations
import logging
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Static registry: ground truth → data files mapping
# ──────────────────────────────────────────────────────────────────────────────

PAPER_REGISTRY: list[dict] = [
    {
        "id": "Yuan_et_al_2018",
        "ground_truth": "Yuan_et_al_2018.xlsx",
        "pdf": ["2018_Yuan_etal.pdf"],
        "supplementary": ["2018_Yuan_etal.xlsx"],
    },
    {
        "id": "Tomlinson_et_al_2021",
        "ground_truth": "Tomlinson_et_al_2021.xlsx",
        "pdf": ["2021_Tomlinson_etal.pdf"],
        "supplementary": ["2021_Tomlinson_etal_supp.xlsx"],
    },
    {
        "id": "Chu_et_al_2022",
        "ground_truth": "Chu_et_al_2022.xlsx",
        "pdf": ["2022_Chu_etal.pdf"],
        "supplementary": ["2022_Chu_etal.xlsx"],
    },
    {
        "id": "Yang_et_al_2022",
        "ground_truth": "Yang_et_al_2022.xlsx",
        "pdf": ["2022_Yang_etal.pdf"],
        "supplementary": ["2022_Yang_etal_supp.zip"],
    },
    {
        "id": "Zhang_et_al_2022",
        "ground_truth": "Zhang_et_al_2022.xlsx",
        "pdf": ["2022_Zhang_etal.pdf"],
        "supplementary": ["2022_Zhang_etal_appendix.docx"],
        "notes": "Supplementary is .docx — may need manual conversion",
    },
    {
        "id": "Jiang_et_al_2023",
        "ground_truth": "Jiang_et_al_2023.xlsx",
        "pdf": ["2023_Jiang_etal.pdf"],
        "supplementary": ["2023_Jiang_etal_supp.xlsx"],
    },
    {
        "id": "Soster_et_al_2023",
        "ground_truth": "Soster_et_al_2023.xlsx",
        "pdf": ["2023_Soster_etal1.pdf"],
        "supplementary": ["2023_Soster_etal_appendix.xlsx"],
    },
    {
        "id": "Sun_et_al_2023",
        "ground_truth": "Sun_et_al_2023.xlsx",
        "pdf": ["2023_Sun_etal.pdf"],
        "supplementary": ["2023_Sun_etal_appendix.xlsx"],
    },
    {
        "id": "Chen_et_al_2024",
        "ground_truth": "Chen_et_al_2024.xlsx",
        "pdf": ["2024B_Chen_etal.pdf"],
        "supplementary": [
            "2024B_Chen_etal_EMPA.xlsx",
            "2024B_Chen_etal_LAIPCMS.xlsx",
        ],
    },
    {
        "id": "Denisova_et_al_2024",
        "ground_truth": "Denisová_et_al_2024.xlsx",
        "pdf": ["2024_Denisova_etal.pdf"],
        "supplementary": [
            "2024_Denisova_etal_EMPA.xlsx",
            "2024_Denisova_etal_LAICPMS.xlsx",
        ],
    },
    {
        "id": "He_et_al_2024",
        "ground_truth": "He_et_al_2024.xlsx",
        "pdf": ["2024_He_etal.pdf"],
        "supplementary": ["2024_He_etal_supp.xlsx"],
    },
    {
        "id": "Lan_et_al_2024",
        "ground_truth": "Lan_et_al_2024.xlsx",
        "pdf": ["2024_Lan_etal.pdf"],
        "supplementary": ["2024_Lan_etal_supp.xlsx"],
    },
    {
        "id": "Liu_et_al_2024",
        "ground_truth": "Liu_et_al_2024.xlsx",
        "pdf": ["2024_Liu_etal.pdf"],
        "supplementary": [
            "2024_Liu_etal_EMPA.xlsx",
            "2024_Liu_etal_LAICPMS.xlsx",
        ],
    },
    {
        "id": "Sun_et_al_2024",
        "ground_truth": "Sun_et_al_2024.xlsx",
        "pdf": ["2024_Sun_etal.pdf"],
        "supplementary": ["2024_Sun_etal_supp.xlsx"],
    },
    {
        "id": "Wei_et_al_2024",
        "ground_truth": "Wei_et_al_2024.xlsx",
        "pdf": ["2024_Wei_etal2.pdf"],
        "supplementary": ["2025_Wei_etal.zip"],
    },
    {
        "id": "Wu_et_al_2024",
        "ground_truth": "Wu_et_al_2024.xlsx",
        "pdf": ["2024_Wu_etal1.pdf"],
        "supplementary": [
            "2024_Wu_etal_EMPA.xlsx",
            "2024_Wu_etal_LAICPMS.xlsx",
        ],
    },
    {
        "id": "Xia_et_al_2024",
        "ground_truth": "Xia_et_al_2024.xlsx",
        "pdf": ["2024_Xia_etal.pdf"],
        "supplementary": [],
        "notes": "No supplementary data file found — PDF-only extraction",
    },
    {
        "id": "Zhang_et_al_2024",
        "ground_truth": "Zhang_et_al_2024.xlsx",
        "pdf": ["2024_Zhang_etal.pdf"],
        "supplementary": ["2024_Zhang_etal_supp.xlsx"],
    },
    {
        "id": "Zhao_et_al_2024",
        "ground_truth": "Zhao_et_al_2024.xlsx",
        "pdf": ["2024_Zhao_etal.pdf"],
        "supplementary": ["2024_Zhao_etal_supp.xlsx"],
    },
    {
        "id": "Zhu_et_al_2024",
        "ground_truth": "Zhu_et_al_2024.xlsx",
        "pdf": ["2024_Zhu_etal.pdf"],
        "supplementary": ["2024_Zhu_etal_supp.zip"],
    },
    {
        "id": "Bertrandsson_Erlandsson_et_al_2025",
        "ground_truth": "Bertrandsson_Erlandsson_et_al_2025.xlsx",
        "pdf": ["2025_Erlandssonetal_Bertrandssonetal_cpy-sph-pyrite in stratiform-sed-Cu-Codistricts-2025.pdf"],
        "supplementary": ["2025_ErlandssonetalSupp2.xlsx"],
    },
    {
        "id": "Bertrandsson_Erlandsson_et_al_2022_reprocessed",
        "ground_truth": "Bertrandsson_Erlandsson_et_al_2022_as_reported_in_Bertrandsson_Erlandsson_et_al_2025.xlsx",
        "pdf": ["2025_Erlandssonetal_Bertrandssonetal_cpy-sph-pyrite in stratiform-sed-Cu-Codistricts-2025.pdf"],
        "supplementary": ["2025_ErlandssonetalSupp2.xlsx"],
        "notes": "Data from 2022 paper as reprocessed in 2025 publication",
    },
    {
        "id": "Ciccolella_et_al_2025",
        "ground_truth": "Ciccolella_et_al_2025.xlsx",
        "pdf": ["2025_Ciccolella_etal.pdf"],
        "supplementary": [
            "2025_Ciccolella_etal_S1.xlsx",
            "2025_Ciccolella_etal_S2.xlsx",
            "2025_Ciccolella_etal_S3.xlsx",
            "2025_Ciccolella_etal_S4.xlsx",
            "2025_Ciccolella_etal_S5.xlsx",
            "2025_Ciccolella_etal_S6.xlsx",
            "2025_Ciccolella_etal_S7.xlsx",
        ],
    },
    {
        "id": "Foltyn_et_al_2022_reprocessed",
        "ground_truth": "Foltyn_et_al_2022_as_reported_in_Bertrandsson_Erlandsson_et_al_2025.xlsx",
        "pdf": ["2025_Foltyn_etal.pdf"],
        "supplementary": ["2025_Foltyn_etal.xlsx"],
    },
    {
        "id": "Li_et_al_2025",
        "ground_truth": "Li_et_al_2025.xlsx",
        "pdf": ["2025B_Li_etal.pdf"],
        "supplementary": [
            "2025B_Li_etal_supp1.xlsx",
            "2025B_Li_etal_supp2.xlsx",
        ],
    },
    {
        "id": "Sun_et_al_2025",
        "ground_truth": "Sun_et_al_2025.xlsx",
        "pdf": ["2025_Sun_etal.pdf"],
        "supplementary": [
            "2025_Sun_etal_S1.xlsx",
            "2025_Sun_etal_S2.xlsx",
        ],
    },
    {
        "id": "Tan_et_al_2025",
        "ground_truth": "Tan_et_al_2025.xlsx",
        "pdf": ["2025_Tan_etal.pdf"],
        "supplementary": [
            "2025_Tan_etal_S1.xlsx",
            "2025_Tan_etal_S2.xlsx",
            "2025_Tan_etal_S3.xlsx",
            "2025_Tan_etal_S4.xlsx",
            "2025_Tan_etal_S5.xlsx",
            "2025_Tan_etal_S6.xlsx",
        ],
    },
    {
        "id": "Wang_et_al_2025",
        "ground_truth": "Wang_et_al_2025.xlsx",
        "pdf": ["2025_Wang_etal.pdf"],
        "supplementary": ["2025_Wang_etal_S1.xlsx"],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Resolved paper entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResolvedPaper:
    """A paper entry with all paths resolved to absolute filesystem paths."""
    id: str
    ground_truth_path: Path
    pdf_path: Path
    supplementary_paths: list[Path]
    notes: str = ""
    processable: bool = True
    skip_reason: str = ""


def resolve_registry(
    project_root: Path,
    paper_ids: Optional[list[str]] = None,
) -> list[ResolvedPaper]:
    """Resolve the registry entries to absolute paths, validating file existence.

    Args:
        project_root: Root directory containing ground_truth/, data/, data/Spreadsheets/.
        paper_ids: If specified, only resolve these paper IDs. None = all.

    Returns:
        List of ResolvedPaper objects (including unprocessable ones, flagged).
    """
    gt_dir = project_root / "ground_truth"
    data_dir = project_root / "data"
    supp_dir = data_dir / "Spreadsheets"

    resolved: list[ResolvedPaper] = []

    for entry in PAPER_REGISTRY:
        paper_id = entry["id"]
        if paper_ids and paper_id not in paper_ids:
            continue

        notes = entry.get("notes", "")

        # Resolve ground truth
        gt_path = gt_dir / entry["ground_truth"]
        if not gt_path.exists():
            resolved.append(ResolvedPaper(
                id=paper_id, ground_truth_path=gt_path, pdf_path=Path(),
                supplementary_paths=[], processable=False,
                skip_reason=f"Ground truth not found: {gt_path.name}",
            ))
            continue

        # Resolve PDF (use first available)
        pdf_path = None
        for pdf_name in entry["pdf"]:
            candidate = data_dir / pdf_name
            if candidate.exists():
                pdf_path = candidate
                break
        if not pdf_path:
            resolved.append(ResolvedPaper(
                id=paper_id, ground_truth_path=gt_path, pdf_path=Path(),
                supplementary_paths=[], processable=False,
                skip_reason=f"PDF not found: {entry['pdf']}",
            ))
            continue

        # Resolve supplementary files
        supp_paths: list[Path] = []
        for supp_name in entry.get("supplementary", []):
            candidate = supp_dir / supp_name
            if candidate.exists():
                supp_paths.append(candidate)
            else:
                logger.warning("Supplementary file not found: %s", candidate)

        # Attempt to extract xlsx/csv from zip files
        extracted_paths: list[Path] = []
        non_zip_paths: list[Path] = []
        for p in supp_paths:
            if p.suffix.lower() == ".zip":
                extracted = _extract_xlsx_from_zip(p)
                extracted_paths.extend(extracted)
            elif p.suffix.lower() in (".xlsx", ".xls", ".csv"):
                non_zip_paths.append(p)
            else:
                logger.info("Skipping non-processable supplementary: %s", p.name)

        final_supp = non_zip_paths + extracted_paths

        if not final_supp:
            # Still processable for PDF-only extraction (metadata only)
            resolved.append(ResolvedPaper(
                id=paper_id, ground_truth_path=gt_path, pdf_path=pdf_path,
                supplementary_paths=[], notes=notes,
                processable=True,
                skip_reason="No processable supplementary files (PDF-only extraction)",
            ))
            continue

        resolved.append(ResolvedPaper(
            id=paper_id, ground_truth_path=gt_path, pdf_path=pdf_path,
            supplementary_paths=final_supp, notes=notes,
        ))

    return resolved


def get_processable_papers(
    project_root: Path,
    paper_ids: Optional[list[str]] = None,
    require_supplementary: bool = False,
) -> list[ResolvedPaper]:
    """Return only papers that can be processed (files exist, format supported).

    Args:
        project_root: Root directory of the project.
        paper_ids: Filter to specific paper IDs. None = all.
        require_supplementary: If True, skip papers with no supplementary files.
    """
    all_papers = resolve_registry(project_root, paper_ids)
    result = []
    for p in all_papers:
        if not p.processable:
            logger.info("Skipping %s: %s", p.id, p.skip_reason)
            continue
        if require_supplementary and not p.supplementary_paths:
            logger.info("Skipping %s: no supplementary files", p.id)
            continue
        result.append(p)
    return result


def list_all_paper_ids() -> list[str]:
    """Return all paper IDs in the registry."""
    return [entry["id"] for entry in PAPER_REGISTRY]


# ──────────────────────────────────────────────────────────────────────────────
# Auto-discovery for non-GT papers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscoveredPaper:
    """A paper discovered from the data/ directory without ground truth."""
    id: str
    pdf_path: Path
    supplementary_paths: list[Path]
    notes: str = ""


def _pdf_stem_to_prefix(pdf_stem: str) -> str:
    """Extract the author-year prefix from a PDF filename stem.

    Examples:
        '2024_He_etal'        -> '2024_He_etal'
        '2025B_Wang_etal'     -> '2025B_Wang_etal'
        '2016_Frenzel_etal_supp_guide' -> '2016_Frenzel_etal'
    """
    # Match: year[letter]_Author_etal
    m = re.match(r"(\d{4}[A-Z]?_\w+_etal)", pdf_stem)
    if m:
        return m.group(1)
    return pdf_stem


def discover_non_gt_papers(
    project_root: Path,
    exclude_pdf_only: bool = False,
) -> list[DiscoveredPaper]:
    """Discover all papers in data/ that are NOT in the ground truth registry.

    Auto-maps PDFs to supplementary files by filename prefix matching.

    Args:
        project_root: Root directory containing data/, data/Spreadsheets/.
        exclude_pdf_only: If True, exclude papers with no supplementary files.

    Returns:
        List of DiscoveredPaper objects.
    """
    data_dir = project_root / "data"
    supp_dir = data_dir / "Spreadsheets"

    # Get all registered PDF filenames (already have GT)
    registered_pdfs: set[str] = set()
    for entry in PAPER_REGISTRY:
        for pdf_name in entry["pdf"]:
            registered_pdfs.add(pdf_name)

    # All PDFs in data/
    all_pdfs = sorted(data_dir.glob("*.pdf"))

    # All supplementary files
    all_supp_files = sorted(supp_dir.iterdir()) if supp_dir.exists() else []

    # Processable supplementary extensions
    _SUPP_EXTENSIONS = {".xlsx", ".xls", ".csv", ".zip"}

    # Skip non-paper PDFs
    _SKIP_PDFS = {"Clark1999_MagneticPetrologyIgRocks.pdf"}

    results: list[DiscoveredPaper] = []

    for pdf_path in all_pdfs:
        if pdf_path.name in registered_pdfs or pdf_path.name in _SKIP_PDFS:
            continue

        # Skip supplementary guide PDFs (not actual papers)
        if "supp_guide" in pdf_path.name.lower():
            continue

        prefix = _pdf_stem_to_prefix(pdf_path.stem)
        paper_id = prefix.replace(" ", "_")

        # Find matching supplementary files by prefix
        matched_supp: list[Path] = []
        for sf in all_supp_files:
            if sf.suffix.lower() in _SUPP_EXTENSIONS and sf.stem.startswith(prefix):
                matched_supp.append(sf)

        # Also try broader prefix matching (just year_author)
        if not matched_supp:
            parts = prefix.split("_")
            if len(parts) >= 2:
                short_prefix = f"{parts[0]}_{parts[1]}"
                for sf in all_supp_files:
                    if sf.suffix.lower() in _SUPP_EXTENSIONS and sf.stem.startswith(short_prefix):
                        matched_supp.append(sf)

        # Process zip files: extract xlsx/csv from inside
        final_supp: list[Path] = []
        for sp in matched_supp:
            if sp.suffix.lower() == ".zip":
                extracted = _extract_xlsx_from_zip(sp)
                final_supp.extend(extracted)
            elif sp.suffix.lower() in (".xlsx", ".xls", ".csv"):
                final_supp.append(sp)
            # skip .docx, .doc, .7z, .pdf supplementary for now

        if exclude_pdf_only and not final_supp:
            continue

        notes = ""
        if not final_supp:
            notes = "PDF-only (no processable supplementary found)"
        elif len(final_supp) > 3:
            notes = f"{len(final_supp)} supplementary files"

        results.append(DiscoveredPaper(
            id=paper_id,
            pdf_path=pdf_path,
            supplementary_paths=final_supp,
            notes=notes,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Unified discovery: all papers with status
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperInfo:
    """Comprehensive info about a paper's files and status."""
    id: str
    pdf_path: Optional[Path] = None
    supplementary_paths: list[Path] = field(default_factory=list)
    ground_truth_path: Optional[Path] = None
    has_pdf: bool = False
    has_supplementary: bool = False
    has_ground_truth: bool = False
    notes: str = ""
    source: str = ""  # "registry" or "discovered"


def discover_all_papers(project_root: Path) -> list[PaperInfo]:
    """Discover ALL papers in the project — both registered (GT) and unregistered.

    Returns a unified list with status flags for each paper:
    - has_pdf, has_supplementary, has_ground_truth
    - source: "registry" (in PAPER_REGISTRY) or "discovered" (auto-found)

    Args:
        project_root: Root directory containing ground_truth/, data/, data/Spreadsheets/.

    Returns:
        List of PaperInfo sorted by ID.
    """
    data_dir = project_root / "data"
    supp_dir = data_dir / "Spreadsheets"
    gt_dir = project_root / "ground_truth"

    results: dict[str, PaperInfo] = {}

    # 1. Process registry entries
    for entry in PAPER_REGISTRY:
        paper_id = entry["id"]
        info = PaperInfo(id=paper_id, source="registry", notes=entry.get("notes", ""))

        # Ground truth
        gt_path = gt_dir / entry["ground_truth"]
        if gt_path.exists():
            info.ground_truth_path = gt_path
            info.has_ground_truth = True

        # PDF
        for pdf_name in entry["pdf"]:
            candidate = data_dir / pdf_name
            if candidate.exists():
                info.pdf_path = candidate
                info.has_pdf = True
                break

        # Supplementary
        supp_paths: list[Path] = []
        for supp_name in entry.get("supplementary", []):
            candidate = supp_dir / supp_name
            if candidate.exists():
                if candidate.suffix.lower() == ".zip":
                    extracted = _extract_xlsx_from_zip(candidate)
                    supp_paths.extend(extracted)
                elif candidate.suffix.lower() in (".xlsx", ".xls", ".csv"):
                    supp_paths.append(candidate)
            # If file doesn't exist, we note it
        info.supplementary_paths = supp_paths
        info.has_supplementary = len(supp_paths) > 0

        if not info.has_supplementary and entry.get("supplementary"):
            missing = [s for s in entry["supplementary"]
                       if not (supp_dir / s).exists()]
            if missing:
                info.notes = (info.notes + "; " if info.notes else "") + \
                    f"Missing supp: {', '.join(missing)}"

        results[paper_id] = info

    # 2. Discover non-registry papers
    registered_pdfs: set[str] = set()
    for entry in PAPER_REGISTRY:
        for pdf_name in entry["pdf"]:
            registered_pdfs.add(pdf_name)

    _SKIP_PDFS = {"Clark1999_MagneticPetrologyIgRocks.pdf"}
    _SUPP_EXTENSIONS = {".xlsx", ".xls", ".csv", ".zip"}

    all_pdfs = sorted(data_dir.glob("*.pdf")) if data_dir.exists() else []
    all_supp_files = sorted(supp_dir.iterdir()) if supp_dir.exists() else []

    # Also check for orphan ground truth files
    gt_filenames_in_registry = {entry["ground_truth"] for entry in PAPER_REGISTRY}

    for pdf_path in all_pdfs:
        if pdf_path.name in registered_pdfs or pdf_path.name in _SKIP_PDFS:
            continue
        if "supp_guide" in pdf_path.name.lower():
            continue

        prefix = _pdf_stem_to_prefix(pdf_path.stem)
        paper_id = prefix.replace(" ", "_")

        info = PaperInfo(id=paper_id, source="discovered")
        info.pdf_path = pdf_path
        info.has_pdf = True

        # Match supplementary files by prefix
        matched_supp: list[Path] = []
        for sf in all_supp_files:
            if sf.suffix.lower() in _SUPP_EXTENSIONS and sf.stem.startswith(prefix):
                matched_supp.append(sf)

        # Broader prefix matching
        if not matched_supp:
            parts = prefix.split("_")
            if len(parts) >= 2:
                short_prefix = f"{parts[0]}_{parts[1]}"
                for sf in all_supp_files:
                    if sf.suffix.lower() in _SUPP_EXTENSIONS and sf.stem.startswith(short_prefix):
                        matched_supp.append(sf)

        # Process zips
        final_supp: list[Path] = []
        for sp in matched_supp:
            if sp.suffix.lower() == ".zip":
                extracted = _extract_xlsx_from_zip(sp)
                final_supp.extend(extracted)
            elif sp.suffix.lower() in (".xlsx", ".xls", ".csv"):
                final_supp.append(sp)

        info.supplementary_paths = final_supp
        info.has_supplementary = len(final_supp) > 0

        # Check if a ground truth file exists (might not be in registry)
        for gt_file in gt_dir.glob("*.xlsx") if gt_dir.exists() else []:
            if gt_file.name not in gt_filenames_in_registry:
                # Fuzzy match by author/year
                gt_stem = gt_file.stem.lower().replace(" ", "_")
                if prefix.lower().replace(" ", "_") in gt_stem or \
                   gt_stem in prefix.lower().replace(" ", "_"):
                    info.ground_truth_path = gt_file
                    info.has_ground_truth = True
                    break

        if not info.has_supplementary:
            info.notes = "PDF-only (no processable supplementary found)"

        results[paper_id] = info

    return sorted(results.values(), key=lambda p: p.id)


def resolve_paper_by_id(
    project_root: Path,
    paper_id: str,
) -> Optional[PaperInfo]:
    """Look up a single paper by ID — checks registry first, then discovery.

    Returns PaperInfo or None if not found.
    """
    all_papers = discover_all_papers(project_root)
    for p in all_papers:
        if p.id == paper_id:
            return p
    # Fuzzy match: case-insensitive partial match
    paper_id_lower = paper_id.lower()
    for p in all_papers:
        if paper_id_lower in p.id.lower():
            return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Zip extraction helper
# ──────────────────────────────────────────────────────────────────────────────

_TEMP_DIR = None

def _extract_xlsx_from_zip(zip_path: Path) -> list[Path]:
    """Extract xlsx/csv files from a zip archive to a temp directory."""
    global _TEMP_DIR
    if _TEMP_DIR is None:
        _TEMP_DIR = tempfile.mkdtemp(prefix="geochem_supp_")

    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith((".xlsx", ".xls", ".csv")) and not name.startswith("__MACOSX"):
                    # Extract to temp dir with unique prefix
                    safe_name = name.replace("/", "_").replace("\\", "_")
                    out_path = Path(_TEMP_DIR) / f"{zip_path.stem}_{safe_name}"
                    with open(str(out_path), "wb") as f:
                        f.write(zf.read(name))
                    extracted.append(out_path)
                    logger.info("Extracted from %s: %s", zip_path.name, name)
    except Exception as exc:
        logger.warning("Failed to extract %s: %s", zip_path.name, exc)

    return extracted
