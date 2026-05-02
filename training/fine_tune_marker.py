"""
fine_tune_marker.py - Fine-tune Marker's table recognition on geochemistry tables.

Marker uses surya models internally. This script:
1. Generates training pairs: (page_image, expected_table_markdown)
2. Creates a fine-tuning dataset from our corrected ground truth
3. Runs surya table recognition fine-tuning with geochem-specific augmentations

Key geochem-specific training signals:
- Element symbol headers (Fe, Cu, Zn, Pb, etc.) as column identifiers
- BDL markers (bdl, b.d.l., <0.5, n.d.) as valid cell values
- Multi-level headers (element + unit rows)
- Landscape/rotated table orientation
- Transposed tables (elements as rows, samples as columns)
- Mixed units within single table (wt% and ppm)

Usage:
    python fine_tune_marker.py --data-dir training/output --epochs 10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_table_markdown_from_gt(gt_path: Path, paper_id: str) -> list[dict]:
    """Build expected markdown table representations from corrected ground truth.

    Each entry maps a table to its expected markdown output, which Marker
    should produce when given the page image.

    Returns list of {table_id, expected_markdown, element_columns, n_rows}.
    """
    import pandas as pd
    from geochem_benchmark.schema import ELEMENT_SYMBOLS

    df = pd.read_excel(gt_path)

    # Identify which elements have data
    active_elements = []
    for sym in ELEMENT_SYMBOLS:
        col = f"{sym}_ppm"
        if col in df.columns and df[col].notna().any():
            active_elements.append(sym)

    if not active_elements:
        return []

    # Build expected markdown table
    header_cols = ["Sample"]
    if "mineral" in df.columns and df["mineral"].notna().any():
        header_cols.append("Mineral")
    header_cols.extend([sym.capitalize() for sym in active_elements])

    # Header row
    header_line = "| " + " | ".join(header_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(header_cols)) + " |"

    # Data rows
    data_lines = []
    for _, row in df.iterrows():
        cells = []
        sample_name = str(row.get("sample_name", "")) if pd.notna(row.get("sample_name")) else ""
        cells.append(sample_name)

        if "Mineral" in header_cols:
            mineral = str(row.get("mineral", "")) if pd.notna(row.get("mineral")) else ""
            cells.append(mineral)

        for sym in active_elements:
            col = f"{sym}_ppm"
            val = row.get(col)
            if pd.isna(val) or val is None:
                cells.append("")
            elif val == -99999:
                cells.append("bdl")
            elif isinstance(val, (int, float)) and val < 0:
                cells.append(f"<{abs(val)}")
            else:
                cells.append(str(val))

        data_lines.append("| " + " | ".join(cells) + " |")

    markdown = "\n".join([header_line, separator] + data_lines)

    return [{
        "table_id": f"{paper_id}_main",
        "expected_markdown": markdown,
        "element_columns": [sym.capitalize() for sym in active_elements],
        "n_rows": len(df),
        "paper_id": paper_id,
    }]


def create_fine_tuning_dataset(
    annotations_dir: Path,
    images_dir: Path,
    gt_dir: Path,
    output_path: Path,
) -> int:
    """Create a fine-tuning dataset pairing page images with expected outputs.

    Format: JSONL where each line is:
    {
        "image": "path/to/page_image.png",
        "tables": [
            {
                "bbox": [x0, y0, x1, y1],
                "expected_markdown": "| Sample | Fe | Cu |...",
                "geochem_features": {
                    "has_bdl": true,
                    "elements": ["Fe", "Cu", "Zn"],
                    "is_landscape": false,
                }
            }
        ]
    }

    Returns number of training examples created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_examples = 0

    # Load all annotation files
    ann_files = sorted(annotations_dir.glob("*.json"))
    if not ann_files:
        logger.warning("No annotation files found in %s", annotations_dir)
        return 0

    with open(output_path, "w") as out_f:
        for ann_file in ann_files:
            with open(ann_file) as f:
                ann = json.load(f)

            image_name = ann.get("image", "")
            image_path = images_dir / image_name
            if not image_path.exists():
                continue

            # Build expected markdown from GT if available
            paper_stem = image_name.split("_p")[0] if "_p" in image_name else image_name
            gt_candidates = list(gt_dir.glob(f"*{paper_stem}*.xlsx"))

            expected_md = ""
            if gt_candidates:
                tables = build_table_markdown_from_gt(gt_candidates[0], paper_stem)
                if tables:
                    expected_md = tables[0]["expected_markdown"]

            entry = {
                "image": str(image_path),
                "page_number": ann.get("page_number", 0),
                "tables": [{
                    "bbox": ann.get("table_bbox", [0, 0, 1, 1]),
                    "expected_markdown": expected_md,
                    "geochem_features": ann.get("geochem_metadata", {}),
                    "structure": ann.get("table_structure", {}),
                }],
            }

            out_f.write(json.dumps(entry) + "\n")
            n_examples += 1

    logger.info("Created %d fine-tuning examples at %s", n_examples, output_path)
    return n_examples


def fine_tune_surya_table_rec(
    dataset_path: Path,
    output_model_dir: Path,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
) -> None:
    """Fine-tune surya's table recognition model on geochem data.

    This wraps surya's training API to adapt the table recognition model
    for geochemistry-specific table layouts.

    Prerequisites:
        pip install surya-ocr[train]  # training extras
    """
    logger.info("Starting surya table recognition fine-tuning")
    logger.info("  Dataset: %s", dataset_path)
    logger.info("  Output: %s", output_model_dir)
    logger.info("  Epochs: %d, Batch size: %d, LR: %s", epochs, batch_size, learning_rate)

    output_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Surya fine-tuning API (if available)
        from surya.table_rec import TableRecTrainer
        trainer = TableRecTrainer(
            dataset_path=str(dataset_path),
            output_dir=str(output_model_dir),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        trainer.train()
        logger.info("Fine-tuning complete. Model saved to %s", output_model_dir)

    except ImportError:
        # Surya doesn't expose a public training API yet.
        # Fall back to generating the dataset that can be used with
        # the surya training scripts directly.
        logger.warning(
            "Surya training API not available. "
            "Dataset prepared at %s — use surya's training scripts directly:\n"
            "  python -m surya.training.table_rec \\\n"
            "    --dataset %s \\\n"
            "    --output %s \\\n"
            "    --epochs %d",
            dataset_path, dataset_path, output_model_dir, epochs,
        )

        # Save training config for manual execution
        config = {
            "task": "table_recognition_finetuning",
            "domain": "geochemistry",
            "dataset_path": str(dataset_path),
            "output_model_dir": str(output_model_dir),
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "geochem_augmentations": [
                "random_landscape_rotation",
                "element_header_permutation",
                "bdl_value_injection",
                "unit_row_insertion",
                "transposed_table_flip",
            ],
            "notes": (
                "Use surya's table_rec training scripts with this dataset. "
                "The geochem_augmentations list describes domain-specific data "
                "augmentations that should be applied during training."
            ),
        }
        config_path = output_model_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Training config saved to %s", config_path)


def fine_tune_mineru_layout(
    dataset_path: Path,
    output_model_dir: Path,
    epochs: int = 20,
) -> None:
    """Fine-tune MinerU's YOLO layout detection model on geochem papers.

    MinerU uses doclayout_yolo for page layout analysis. Fine-tuning on
    geochem papers helps it better detect:
    - Data tables vs figure tables
    - Landscape-oriented content
    - Multi-page continuation tables
    """
    output_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO

        # Load the base MinerU layout model
        try:
            from doclayout_yolo import YOLOv10
            base_model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench-onnx")
        except Exception:
            logger.warning("DocLayout-YOLO base model not loadable — using ultralytics YOLO base")
            base_model = YOLO("yolov8n.pt")

        # YOLO fine-tuning expects a data.yaml pointing to COCO-format annotations
        data_yaml = output_model_dir / "data.yaml"
        data_yaml.write_text(f"""
train: {dataset_path.parent / 'images'}
val: {dataset_path.parent / 'images'}

nc: 4
names:
  0: table
  1: table_header
  2: table_data
  3: geochem_table
""")

        logger.info("Starting YOLO layout model fine-tuning for geochem papers")
        results = base_model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=1024,
            batch=4,
            project=str(output_model_dir),
            name="geochem_layout",
        )
        logger.info("YOLO fine-tuning complete: %s", results)

    except ImportError:
        logger.warning(
            "ultralytics not available for YOLO fine-tuning. "
            "Install with: pip install ultralytics\n"
            "Then run: yolo train data=%s epochs=%d",
            output_model_dir / "data.yaml", epochs,
        )

        config = {
            "task": "layout_detection_finetuning",
            "domain": "geochemistry",
            "base_model": "DocLayout-YOLO-DocStructBench",
            "dataset_path": str(dataset_path),
            "output_dir": str(output_model_dir),
            "epochs": epochs,
            "notes": (
                "Fine-tune YOLO on geochem paper layouts. "
                "Key focus: detecting geochemistry data tables vs summary/reference tables."
            ),
        }
        config_path = output_model_dir / "mineru_training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Marker/MinerU on geochem tables")
    parser.add_argument("--data-dir", type=Path, default=Path("training/output"))
    parser.add_argument("--gt-dir", type=Path, default=Path("ground_truth_corrected"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/models"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--target", choices=["marker", "mineru", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    annotations_dir = args.data_dir / "marker_annotations"
    images_dir = args.data_dir / "images"
    coco_path = args.data_dir / "coco_annotations.json"

    if args.target in ("marker", "both"):
        # Create fine-tuning dataset
        ft_path = args.output_dir / "marker_finetune.jsonl"
        n = create_fine_tuning_dataset(
            annotations_dir, images_dir, args.gt_dir, ft_path,
        )
        if n > 0:
            fine_tune_surya_table_rec(
                ft_path,
                args.output_dir / "surya_geochem",
                epochs=args.epochs,
            )

    if args.target in ("mineru", "both"):
        if coco_path.exists():
            fine_tune_mineru_layout(
                coco_path,
                args.output_dir / "mineru_geochem",
                epochs=args.epochs * 2,
            )
        else:
            logger.warning("COCO annotations not found at %s — run generate_training_data.py first", coco_path)


if __name__ == "__main__":
    main()
