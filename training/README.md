# Custom Model Training for Geochem Table Extraction

Fine-tune Marker (surya) and MinerU (YOLO) models on geochemistry-specific table layouts.

## Why Custom Models?

Generic document layout models struggle with:
- **Landscape/rotated tables** — wide tables with 20+ element columns
- **Multi-level headers** — element symbols + isotope mass + unit rows
- **BDL notation** — `bdl`, `b.d.l.`, `<0.5`, `n.d.` are valid cell values, not parsing errors
- **Transposed tables** — elements as rows, samples as columns
- **Mixed units** — wt% and ppm in the same table
- **Continuation tables** — "Table 2 (continued)" spanning multiple pages

## Pipeline

### Step 1: Generate Training Data

```bash
python training/generate_training_data.py
```

This produces:
- `training/output/images/` — Rendered PDF page images (150 DPI)
- `training/output/coco_annotations.json` — COCO-format bounding boxes (for YOLO/MinerU)
- `training/output/marker_annotations/` — Marker-format table structure annotations

### Step 2: Fine-Tune Models

```bash
# Both Marker + MinerU
python training/fine_tune_marker.py --epochs 10 --target both

# Marker only
python training/fine_tune_marker.py --epochs 10 --target marker

# MinerU only
python training/fine_tune_marker.py --epochs 20 --target mineru
```

### Step 3: Use Custom Models

Custom models are saved to `training/models/`. To use them in the pipeline:

```python
# For Marker — set environment variable
export MARKER_MODEL_DIR=training/models/surya_geochem

# For MinerU — set model path
export MINERU_MODEL_DIR=training/models/mineru_geochem
```

## Geochem-Specific Augmentations

During training, these domain-specific augmentations improve model robustness:

1. **Random landscape rotation** — Rotate tables 90° to simulate landscape pages
2. **Element header permutation** — Shuffle element column order (Fe, Cu, Zn → Zn, Fe, Cu)
3. **BDL value injection** — Replace random numeric cells with BDL markers
4. **Unit row insertion** — Add/remove unit rows (wt%, ppm, ppb)
5. **Transposed table flip** — Transpose table orientation (rows ↔ columns)

## Training Data Sources

- 28 benchmark papers with corrected ground truth
- ~500 total pages of geochemistry content
- ~100 geochemical data tables
- Covers: LA-ICPMS, EPMA, ICP-MS, XRF analytical methods
- Minerals: sphalerite, pyrite, chalcopyrite, galena, magnetite, etc.
