"""
generate_training_data.py - Generate training data for fine-tuning Marker/MinerU on geochem tables.

Creates annotated datasets from our corrected ground truth files paired with
their source PDFs. The training data captures:
  1. Table bounding boxes (for layout model fine-tuning)
  2. Table structure annotations (headers, data rows, merged cells)
  3. Element column mappings (geochemistry-specific header recognition)
  4. BDL/N/A cell type annotations

Output formats:
  - COCO-format JSON for YOLO/MinerU layout model training
  - Marker-format JSON for Marker table recognition training
  - PubTabNet-format for general table structure recognition
"""

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TableAnnotation:
    """Ground truth annotation for a single table in a PDF."""
    pdf_path: str
    page_number: int            # 1-based
    table_id: str               # e.g., "Table_1", "Supp_Table_2"
    bbox: list[float]           # [x0, y0, x1, y1] normalised 0-1
    n_rows: int
    n_cols: int
    header_rows: int = 1
    element_columns: list[str] = field(default_factory=list)   # e.g., ["Fe", "Cu", "Zn"]
    has_bdl_values: bool = False
    has_merged_cells: bool = False
    is_landscape: bool = False
    is_transposed: bool = False
    analytical_method: Optional[str] = None
    mineral: Optional[str] = None


@dataclass
class TrainingExample:
    """One training example: PDF page image + table annotations."""
    image_path: str             # Rendered page image
    pdf_path: str
    page_number: int
    width: int
    height: int
    tables: list[TableAnnotation] = field(default_factory=list)


def render_page_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[dict]:
    """Render PDF pages to images for training.

    Returns list of {page_number, image_path, width, height}.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed: pip install pymupdf")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        image_name = f"{pdf_path.stem}_p{page_num + 1}.png"
        image_path = output_dir / image_name
        pix.save(str(image_path))

        results.append({
            "page_number": page_num + 1,
            "image_path": str(image_path),
            "width": pix.width,
            "height": pix.height,
        })

    doc.close()
    return results


def detect_table_regions(pdf_path: Path) -> list[dict]:
    """Auto-detect table bounding boxes using pdfplumber.

    Returns list of {page_number, bbox, n_rows, n_cols} for each table found.
    """
    import pdfplumber

    tables = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.find_tables()
            for i, tbl in enumerate(page_tables):
                if tbl.bbox:
                    # Normalise bbox to 0-1 range
                    pw, ph = page.width, page.height
                    x0, y0, x1, y1 = tbl.bbox
                    norm_bbox = [x0/pw, y0/ph, x1/pw, y1/ph]

                    # Extract table to count rows/cols
                    extracted = tbl.extract()
                    n_rows = len(extracted) if extracted else 0
                    n_cols = len(extracted[0]) if extracted and extracted[0] else 0

                    tables.append({
                        "page_number": page_num,
                        "table_index": i,
                        "bbox": norm_bbox,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                    })
    return tables


def match_gt_to_tables(
    gt_df: pd.DataFrame,
    table_regions: list[dict],
    paper_id: str,
) -> list[TableAnnotation]:
    """Match ground truth data to detected table regions.

    Uses element column presence and row counts to pair GT with detected tables.
    """
    from geochem_benchmark.schema import ELEMENT_SYMBOLS

    # Determine which elements are in the GT
    gt_elements = []
    for sym in ELEMENT_SYMBOLS:
        col = f"{sym}_ppm"
        if col in gt_df.columns:
            vals = pd.to_numeric(gt_df[col], errors="coerce").dropna()
            vals = vals[vals != -99999]  # Exclude BDL
            if len(vals) > 0:
                gt_elements.append(sym.capitalize())

    # Determine if GT has BDL values
    has_bdl = False
    for sym in ELEMENT_SYMBOLS:
        col = f"{sym}_ppm"
        if col in gt_df.columns:
            if (gt_df[col] == -99999).any():
                has_bdl = True
                break
            if gt_df[col].apply(lambda x: isinstance(x, (int, float)) and x < 0 and x != -99999).any():
                has_bdl = True
                break

    # Determine analytical method and mineral from GT
    method = None
    mineral = None
    if "analytical_method" in gt_df.columns:
        methods = gt_df["analytical_method"].dropna().unique()
        if len(methods) > 0:
            method = str(methods[0])
    if "mineral" in gt_df.columns:
        minerals = gt_df["mineral"].dropna().unique()
        if len(minerals) > 0:
            mineral = str(minerals[0])

    annotations = []
    for region in table_regions:
        # Simple heuristic: tables with >5 columns and >3 rows are likely data tables
        if region["n_cols"] >= 5 and region["n_rows"] >= 3:
            ann = TableAnnotation(
                pdf_path="",
                page_number=region["page_number"],
                table_id=f"Table_{region['page_number']}_{region['table_index']}",
                bbox=region["bbox"],
                n_rows=region["n_rows"],
                n_cols=region["n_cols"],
                element_columns=gt_elements[:region["n_cols"]],
                has_bdl_values=has_bdl,
                analytical_method=method,
                mineral=mineral,
            )
            annotations.append(ann)

    return annotations


def export_coco_format(
    examples: list[TrainingExample],
    output_path: Path,
) -> None:
    """Export training data in COCO format for YOLO/MinerU training.

    COCO format is the standard for object detection model training.
    Categories: table, table_header, table_data_row, merged_cell
    """
    categories = [
        {"id": 1, "name": "table", "supercategory": "document"},
        {"id": 2, "name": "table_header", "supercategory": "table"},
        {"id": 3, "name": "table_data", "supercategory": "table"},
        {"id": 4, "name": "geochem_table", "supercategory": "table"},
    ]

    images = []
    annotations = []
    ann_id = 1

    for img_id, example in enumerate(examples, 1):
        images.append({
            "id": img_id,
            "file_name": os.path.basename(example.image_path),
            "width": example.width,
            "height": example.height,
        })

        for table in example.tables:
            x0, y0, x1, y1 = table.bbox
            # Convert normalised to pixel coordinates
            px0 = x0 * example.width
            py0 = y0 * example.height
            pw = (x1 - x0) * example.width
            ph = (y1 - y0) * example.height

            # Use geochem_table category if element columns detected
            cat_id = 4 if table.element_columns else 1

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [px0, py0, pw, ph],
                "area": pw * ph,
                "iscrowd": 0,
                "attributes": {
                    "n_rows": table.n_rows,
                    "n_cols": table.n_cols,
                    "elements": table.element_columns,
                    "has_bdl": table.has_bdl_values,
                    "is_landscape": table.is_landscape,
                    "is_transposed": table.is_transposed,
                    "method": table.analytical_method,
                    "mineral": table.mineral,
                },
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    logger.info("Exported %d images, %d annotations to %s",
                len(images), len(annotations), output_path)


def export_marker_format(
    examples: list[TrainingExample],
    output_dir: Path,
) -> None:
    """Export training data for Marker fine-tuning.

    Marker uses a specific JSON format for table structure training:
    - Page images paired with expected markdown output
    - Table region annotations with header/data row distinction
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for example in examples:
        for table in example.tables:
            entry = {
                "image": os.path.basename(example.image_path),
                "page_number": example.page_number,
                "table_bbox": table.bbox,
                "table_structure": {
                    "n_rows": table.n_rows,
                    "n_cols": table.n_cols,
                    "header_rows": table.header_rows,
                    "element_columns": table.element_columns,
                },
                "geochem_metadata": {
                    "has_bdl_values": table.has_bdl_values,
                    "is_landscape": table.is_landscape,
                    "is_transposed": table.is_transposed,
                    "analytical_method": table.analytical_method,
                    "mineral": table.mineral,
                },
            }

            entry_name = f"{Path(example.pdf_path).stem}_p{example.page_number}_{table.table_id}.json"
            with open(output_dir / entry_name, "w") as f:
                json.dump(entry, f, indent=2)


def generate_from_benchmark(
    data_dir: Path | None = None,
    gt_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Generate training data from the benchmark paper collection.

    Uses PDFs + corrected ground truth to create annotated training examples.

    Returns stats dict.
    """
    base = Path(__file__).parent.parent
    if data_dir is None:
        data_dir = base / "data"
    if gt_dir is None:
        gt_dir = base / "ground_truth_corrected"
    if output_dir is None:
        output_dir = base / "training" / "output"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    # Import paper registry
    from geochem_benchmark.paper_registry import PAPER_REGISTRY

    all_examples: list[TrainingExample] = []
    stats = {"papers": 0, "pages": 0, "tables": 0, "geochem_tables": 0}

    for paper in PAPER_REGISTRY:
        pdf_path = data_dir / paper.pdf_path
        gt_path = gt_dir / f"{paper.paper_id}.xlsx"

        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            continue
        if not gt_path.exists():
            logger.debug("No corrected GT for %s", paper.paper_id)
            continue

        logger.info("Processing %s", paper.paper_id)
        stats["papers"] += 1

        # Render page images
        page_images = render_page_images(pdf_path, images_dir)
        stats["pages"] += len(page_images)

        # Detect table regions
        table_regions = detect_table_regions(pdf_path)

        # Match GT to tables
        gt_df = pd.read_excel(gt_path)
        annotations = match_gt_to_tables(gt_df, table_regions, paper.paper_id)
        stats["tables"] += len(table_regions)
        stats["geochem_tables"] += len(annotations)

        # Create training examples
        for page_info in page_images:
            page_tables = [
                ann for ann in annotations
                if ann.page_number == page_info["page_number"]
            ]
            for ann in page_tables:
                ann.pdf_path = str(pdf_path)

            example = TrainingExample(
                image_path=page_info["image_path"],
                pdf_path=str(pdf_path),
                page_number=page_info["page_number"],
                width=page_info["width"],
                height=page_info["height"],
                tables=page_tables,
            )
            all_examples.append(example)

    # Export in both formats
    export_coco_format(all_examples, output_dir / "coco_annotations.json")
    export_marker_format(all_examples, output_dir / "marker_annotations")

    logger.info("Training data generation complete: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = generate_from_benchmark()
    print(f"\nTraining data generated:")
    print(f"  Papers: {stats['papers']}")
    print(f"  Pages rendered: {stats['pages']}")
    print(f"  Tables detected: {stats['tables']}")
    print(f"  Geochem tables annotated: {stats['geochem_tables']}")
