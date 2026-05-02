"""
batch_runner.py - Batch processing of all papers with ground truths.

Orchestrates:
  1. Discover papers with ground truth files via paper_registry
  2. For each paper, run ExtractionPipeline (PDF + supplementary → 210-column schema)
  3. Evaluate each extraction against its ground truth
  4. Aggregate all results into a cross-paper leaderboard

Usage:
  python -m geochem_benchmark.main batch \\
      --provider claude \\
      --model claude-sonnet-4-6 \\
      --output-dir batch_results/
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .evaluator import (
    BenchmarkReport,
    Evaluator,
    evaluate_quality,
    save_detailed_report,
)
from .llm_clients import LLMClient
from .paper_registry import (
    DiscoveredPaper,
    ResolvedPaper,
    discover_non_gt_papers,
    get_processable_papers,
)
from .pipeline import ExtractionPipeline, ExtractionResult
from .schema import PaperMetadata
from .tabledetector import TableDetectorBackend

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperResult:
    """Result of processing one paper."""
    paper_id: str
    extraction: Optional[ExtractionResult] = None
    report: Optional[BenchmarkReport] = None
    error: Optional[str] = None
    duration_sec: float = 0.0

    @property
    def success(self) -> bool:
        return self.extraction is not None and self.error is None


@dataclass
class BatchResult:
    """Aggregated results from processing all papers."""
    provider: str
    model: str
    paper_results: list[PaperResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.paper_results)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.paper_results if r.success)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_success

    @property
    def n_evaluated(self) -> int:
        return sum(1 for r in self.paper_results if r.report is not None)

    def aggregate_scores(self) -> dict:
        """Compute mean scores across all evaluated papers."""
        evaluated = [r for r in self.paper_results if r.report is not None]
        if not evaluated:
            return {}
        return {
            "n_papers": len(evaluated),
            "mean_T1_metadata_%": _mean([r.report.t1_metadata_score for r in evaluated]) * 100,
            "mean_T2_numerical_%": _mean([r.report.t2_numerical_score for r in evaluated]) * 100,
            "mean_T3_structural_%": _mean([r.report.t3_structural_score for r in evaluated]) * 100,
            "mean_T4_null_%": _mean([r.report.t4_null_score for r in evaluated]) * 100,
            "mean_overall_%": _mean([r.report.overall_score for r in evaluated]) * 100,
            "total_samples_predicted": sum(r.report.predicted_n_samples for r in evaluated),
            "total_samples_ground_truth": sum(r.report.ground_truth_n_samples for r in evaluated),
            "total_samples_matched": sum(r.report.matched_samples for r in evaluated),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(
    client: LLMClient,
    project_root: Path,
    output_dir: Path,
    paper_ids: Optional[list[str]] = None,
    use_tool_calling: bool = True,
    use_llm_table_filter: bool = False,
    use_self_correction: bool = True,
    use_vision: bool = True,
    vision_client: Optional[LLMClient] = None,
    require_supplementary: bool = False,
    only_pdf_only: bool = False,
    verbose: bool = False,
    table_detector_backend: str = "auto",
) -> BatchResult:
    """Process all papers with ground truths and evaluate against each.

    Args:
        client: LLM client to use for extraction.
        project_root: Root directory containing ground_truth/, data/, data/Spreadsheets/.
        output_dir: Directory to save all output files.
        paper_ids: If specified, only process these paper IDs. None = all.
        use_tool_calling: Enable Claude tool calling for metadata extraction.
        use_llm_table_filter: Use LLM-assisted table row filtering.
        require_supplementary: If True, skip papers with no supplementary files
            (default False — process all GT papers including PDF-only ones).
        only_pdf_only: If True, process ONLY papers with no supplementary files.
        verbose: Enable verbose logging.
        table_detector_backend: PDF table detector backend (auto, docling, camelot, pdfplumber).

    Returns:
        BatchResult with per-paper and aggregated results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover processable papers
    papers = get_processable_papers(
        project_root=project_root,
        paper_ids=paper_ids,
        require_supplementary=require_supplementary,
    )

    # Filter to PDF-only if requested
    if only_pdf_only:
        papers = [p for p in papers if not p.supplementary_paths]

    if not papers:
        logger.error("No processable papers found.")
        return BatchResult(provider=client.provider, model=client.model)

    print(f"\nFound {len(papers)} processable papers:")
    for p in papers:
        supp_info = f"{len(p.supplementary_paths)} supp files" if p.supplementary_paths else "PDF only"
        print(f"  {p.id}: {supp_info}")
    print()

    # Build pipeline
    pipeline = ExtractionPipeline(
        llm_client=client,
        use_tool_calling=use_tool_calling,
        use_llm_table_filter=use_llm_table_filter,
        use_self_correction=use_self_correction,
        use_vision=use_vision,
        vision_client=vision_client,
        verbose=verbose,
        table_detector_backend=TableDetectorBackend(table_detector_backend),
    )

    batch = BatchResult(provider=client.provider, model=client.model)

    # Process each paper
    for i, paper in enumerate(papers, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(papers)}] Processing: {paper.id}")
        print(f"{'='*70}")

        result = _process_one_paper(
            pipeline=pipeline,
            paper=paper,
            output_dir=output_dir,
        )
        batch.paper_results.append(result)

        # Print per-paper summary
        if result.success:
            print(f"  Extracted {result.extraction.n_samples} samples in {result.duration_sec:.1f}s")
            if result.extraction.n_samples == 0 and result.extraction.is_pdf_only:
                print(f"  (PDF-only — metadata extracted, no sample rows)")
                print(result.extraction.metadata_summary())
            if result.report:
                print(f"  Scores: T1={result.report.t1_metadata_score*100:.1f}%  "
                      f"T2={result.report.t2_numerical_score*100:.1f}%  "
                      f"T3={result.report.t3_structural_score*100:.1f}%  "
                      f"T4={result.report.t4_null_score*100:.1f}%  "
                      f"Overall={result.report.overall_score*100:.1f}%")
        else:
            print(f"  FAILED: {result.error}")

    # Save aggregated results
    _save_batch_results(batch, output_dir)

    return batch


def _process_one_paper(
    pipeline: ExtractionPipeline,
    paper: ResolvedPaper,
    output_dir: Path,
) -> PaperResult:
    """Process a single paper: extract + evaluate."""
    start = time.time()
    pr = PaperResult(paper_id=paper.id)

    # Stage 1: Extract
    try:
        extraction = pipeline.run(
            pdf_path=paper.pdf_path,
            supplementary_paths=paper.supplementary_paths,
        )
        pr.extraction = extraction

        # Save extraction output
        out_xlsx = output_dir / f"extraction_{paper.id}.xlsx"
        extraction.to_excel(out_xlsx)
        logger.info("Saved extraction: %s", out_xlsx)

    except Exception as exc:
        pr.error = str(exc)
        logger.error("Extraction failed for %s: %s", paper.id, exc, exc_info=True)
        pr.duration_sec = time.time() - start
        return pr

    # Stage 2: Evaluate against ground truth
    try:
        evaluator = Evaluator(ground_truth_path=paper.ground_truth_path)
        report = evaluator.evaluate(extraction)
        pr.report = report

        # Save detailed report
        report_path = output_dir / f"report_{paper.id}.xlsx"
        save_detailed_report(report, report_path)

    except Exception as exc:
        logger.warning("Evaluation failed for %s: %s", paper.id, exc)
        # Extraction succeeded even if evaluation failed

    pr.duration_sec = time.time() - start
    return pr


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_batch_results(batch: BatchResult, output_dir: Path) -> None:
    """Save aggregated batch results to files."""
    # Per-paper summary table
    rows = []
    for pr in batch.paper_results:
        row = {
            "paper_id": pr.paper_id,
            "status": "success" if pr.success else "failed",
            "n_samples": pr.extraction.n_samples if pr.extraction else 0,
            "duration_sec": round(pr.duration_sec, 1),
        }
        if pr.report:
            row.update({
                "T1_metadata_%": round(pr.report.t1_metadata_score * 100, 2),
                "T2_numerical_%": round(pr.report.t2_numerical_score * 100, 2),
                "T3_structural_%": round(pr.report.t3_structural_score * 100, 2),
                "T4_null_%": round(pr.report.t4_null_score * 100, 2),
                "overall_%": round(pr.report.overall_score * 100, 2),
                "matched_samples": pr.report.matched_samples,
                "gt_samples": pr.report.ground_truth_n_samples,
            })
        if pr.error:
            row["error"] = pr.error
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "overall_%" in df.columns:
        df = df.sort_values("overall_%", ascending=False).reset_index(drop=True)
        df.index += 1

    # Save summary
    summary_path = output_dir / "batch_summary.xlsx"
    df.to_excel(str(summary_path), index=True)
    print(f"\nBatch summary saved → {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  BATCH RESULTS: {batch.provider}/{batch.model}")
    print(f"  {batch.n_success}/{batch.n_total} papers processed, {batch.n_evaluated} evaluated")
    print(f"{'='*80}")
    if not df.empty:
        # Print a compact view
        display_cols = ["paper_id", "n_samples", "overall_%", "T1_metadata_%",
                        "T2_numerical_%", "T3_structural_%", "T4_null_%"]
        available = [c for c in display_cols if c in df.columns]
        print(df[available].to_string())
    print(f"{'='*80}")

    # Aggregated scores
    agg = batch.aggregate_scores()
    if agg:
        print(f"\n  AGGREGATE SCORES ({agg['n_papers']} papers):")
        print(f"    Mean T1 Metadata    : {agg['mean_T1_metadata_%']:6.1f}%")
        print(f"    Mean T2 Numerical   : {agg['mean_T2_numerical_%']:6.1f}%")
        print(f"    Mean T3 Structural  : {agg['mean_T3_structural_%']:6.1f}%")
        print(f"    Mean T4 Null        : {agg['mean_T4_null_%']:6.1f}%")
        print(f"    Mean Overall        : {agg['mean_overall_%']:6.1f}%")
        print(f"    Total samples matched: {agg['total_samples_matched']} / {agg['total_samples_ground_truth']}")

        # Save aggregated metrics
        metrics_path = output_dir / "batch_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            # Include per-paper scores in JSON
            per_paper = {}
            for pr in batch.paper_results:
                if pr.report:
                    per_paper[pr.paper_id] = pr.report.to_dict()
            json.dump({
                "model": batch.model,
                "provider": batch.provider,
                "aggregate": agg,
                "per_paper": per_paper,
            }, f, indent=2, default=str)
        print(f"\n  Metrics saved → {metrics_path}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Non-GT batch runner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NoGTPaperResult:
    """Result of processing one paper without ground truth."""
    paper_id: str
    extraction: Optional[ExtractionResult] = None
    quality: Optional[dict] = None
    error: Optional[str] = None
    duration_sec: float = 0.0

    @property
    def success(self) -> bool:
        return self.extraction is not None and self.error is None


@dataclass
class NoGTBatchResult:
    """Aggregated results from processing non-GT papers."""
    provider: str
    model: str
    paper_results: list[NoGTPaperResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.paper_results)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.paper_results if r.success)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_success

    def aggregate_quality(self) -> dict:
        """Compute mean quality scores across all successfully processed papers."""
        evaluated = [r for r in self.paper_results if r.quality is not None]
        if not evaluated:
            return {}
        return {
            "n_papers": len(evaluated),
            "mean_quality_score_%": _mean([r.quality["quality_score_%"] for r in evaluated]),
            "mean_metadata_completeness_%": _mean([r.quality["metadata_completeness_%"] for r in evaluated]),
            "mean_element_completeness_%": _mean([r.quality["element_completeness_%"] for r in evaluated]),
            "mean_sample_id_quality_%": _mean([r.quality["sample_id_quality_%"] for r in evaluated]),
            "mean_value_plausibility_%": _mean([r.quality["value_plausibility_%"] for r in evaluated]),
            "mean_internal_consistency_%": _mean([r.quality["internal_consistency_%"] for r in evaluated]),
            "total_rows_extracted": sum(r.quality["row_count"] for r in evaluated),
        }


def run_batch_no_gt(
    client: LLMClient,
    project_root: Path,
    output_dir: Path,
    exclude_pdf_only: bool = True,
    only_pdf_only: bool = False,
    use_tool_calling: bool = True,
    use_llm_table_filter: bool = False,
    use_self_correction: bool = True,
    use_vision: bool = True,
    vision_client: Optional[LLMClient] = None,
    verbose: bool = False,
    table_detector_backend: str = "auto",
) -> NoGTBatchResult:
    """Process all non-GT papers (extraction + quality assessment, no evaluation).

    Args:
        client: LLM client to use for extraction.
        project_root: Root directory containing data/, data/Spreadsheets/.
        output_dir: Directory to save all output files.
        exclude_pdf_only: If True, skip papers with no supplementary files.
        only_pdf_only: If True, process ONLY papers with no supplementary files.
        use_tool_calling: Enable Claude tool calling for metadata extraction.
        use_llm_table_filter: Use LLM-assisted table row filtering.
        verbose: Enable verbose logging.
        table_detector_backend: PDF table detector backend (auto, docling, camelot, pdfplumber).

    Returns:
        NoGTBatchResult with per-paper quality metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover non-GT papers
    if only_pdf_only:
        # Get all non-GT papers including PDF-only, then filter to only those
        papers = discover_non_gt_papers(
            project_root=project_root,
            exclude_pdf_only=False,
        )
        papers = [p for p in papers if not p.supplementary_paths]
    else:
        papers = discover_non_gt_papers(
            project_root=project_root,
            exclude_pdf_only=exclude_pdf_only,
        )

    if not papers:
        logger.error("No non-GT papers found.")
        return NoGTBatchResult(provider=client.provider, model=client.model)

    print(f"\nFound {len(papers)} non-GT papers to process:")
    for p in papers:
        supp_info = f"{len(p.supplementary_paths)} supp files" if p.supplementary_paths else "PDF only"
        print(f"  {p.id}: {supp_info}")
    print()

    # Build pipeline
    pipeline = ExtractionPipeline(
        llm_client=client,
        use_tool_calling=use_tool_calling,
        use_llm_table_filter=use_llm_table_filter,
        use_self_correction=use_self_correction,
        use_vision=use_vision,
        vision_client=vision_client,
        verbose=verbose,
        table_detector_backend=TableDetectorBackend(table_detector_backend),
    )

    batch = NoGTBatchResult(provider=client.provider, model=client.model)

    # Process each paper
    for i, paper in enumerate(papers, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(papers)}] Processing: {paper.id}")
        print(f"{'='*70}")

        result = _process_one_nogt_paper(
            pipeline=pipeline,
            paper=paper,
            output_dir=output_dir,
        )
        batch.paper_results.append(result)

        # Print per-paper summary
        if result.success:
            q = result.quality
            print(f"  Extracted {result.extraction.n_samples} samples in {result.duration_sec:.1f}s")
            if result.extraction.n_samples == 0 and result.extraction.is_pdf_only:
                print(f"  (PDF-only — metadata extracted, no sample rows)")
                print(result.extraction.metadata_summary())
            if q:
                print(f"  Quality: {q['quality_score_%']:.1f}%  "
                      f"Meta={q['metadata_completeness_%']:.0f}%  "
                      f"Elem={q['element_completeness_%']:.0f}%  "
                      f"SampleID={q['sample_id_quality_%']:.0f}%  "
                      f"Plausibility={q['value_plausibility_%']:.0f}%")
                if q.get("issues"):
                    for issue in q["issues"][:3]:
                        print(f"    ! {issue}")
            # Show self-correction info
            cm = result.extraction.correction_metrics
            if cm and cm.correction_needed:
                status = "succeeded" if cm.correction_succeeded else "failed"
                print(f"  Self-correction: {status} after {len(cm.attempts)} attempt(s) "
                      f"({cm.initial_row_count}->{cm.final_row_count} rows, "
                      f"quality {cm.initial_quality_score:.0f}%->{cm.final_quality_score:.0f}%)")
        else:
            print(f"  FAILED: {result.error}")

    # Save aggregated results
    _save_nogt_batch_results(batch, output_dir)

    return batch


def _process_one_nogt_paper(
    pipeline: ExtractionPipeline,
    paper: DiscoveredPaper,
    output_dir: Path,
) -> NoGTPaperResult:
    """Process a single non-GT paper: extract + quality assess."""
    start = time.time()
    pr = NoGTPaperResult(paper_id=paper.id)

    # Extract
    try:
        extraction = pipeline.run(
            pdf_path=paper.pdf_path,
            supplementary_paths=paper.supplementary_paths,
        )
        pr.extraction = extraction

        # Save extraction output
        out_xlsx = output_dir / f"extraction_{paper.id}.xlsx"
        extraction.to_excel(out_xlsx)
        logger.info("Saved extraction: %s", out_xlsx)

    except Exception as exc:
        pr.error = str(exc)
        logger.error("Extraction failed for %s: %s", paper.id, exc, exc_info=True)
        pr.duration_sec = time.time() - start
        return pr

    # Quality assessment (no GT needed)
    try:
        pred_df = extraction.to_dataframe()
        quality = evaluate_quality(
            pred_df=pred_df,
            model=extraction.llm_model,
            provider=extraction.llm_provider,
        )
        pr.quality = quality
    except Exception as exc:
        logger.warning("Quality assessment failed for %s: %s", paper.id, exc)

    pr.duration_sec = time.time() - start
    return pr


def _save_nogt_batch_results(batch: NoGTBatchResult, output_dir: Path) -> None:
    """Save aggregated non-GT batch results to files."""
    # Per-paper summary table
    rows = []
    for pr in batch.paper_results:
        row = {
            "paper_id": pr.paper_id,
            "status": "success" if pr.success else "failed",
            "n_samples": pr.extraction.n_samples if pr.extraction else 0,
            "duration_sec": round(pr.duration_sec, 1),
        }
        if pr.quality:
            row.update({
                "quality_score_%": pr.quality["quality_score_%"],
                "metadata_completeness_%": pr.quality["metadata_completeness_%"],
                "element_completeness_%": pr.quality["element_completeness_%"],
                "sample_id_quality_%": pr.quality["sample_id_quality_%"],
                "value_plausibility_%": pr.quality["value_plausibility_%"],
                "internal_consistency_%": pr.quality["internal_consistency_%"],
            })
        if pr.error:
            row["error"] = pr.error
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "quality_score_%" in df.columns:
        df = df.sort_values("quality_score_%", ascending=False).reset_index(drop=True)
        df.index += 1

    # Save summary
    summary_path = output_dir / "nogt_batch_summary.xlsx"
    df.to_excel(str(summary_path), index=True)
    print(f"\nBatch summary saved -> {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  NON-GT BATCH RESULTS: {batch.provider}/{batch.model}")
    print(f"  {batch.n_success}/{batch.n_total} papers processed")
    print(f"{'='*80}")
    if not df.empty:
        display_cols = ["paper_id", "n_samples", "quality_score_%",
                        "metadata_completeness_%", "element_completeness_%",
                        "value_plausibility_%"]
        available = [c for c in display_cols if c in df.columns]
        print(df[available].to_string())
    print(f"{'='*80}")

    # Aggregated quality scores
    agg = batch.aggregate_quality()
    if agg:
        print(f"\n  AGGREGATE QUALITY ({agg['n_papers']} papers):")
        print(f"    Mean Quality Score      : {agg['mean_quality_score_%']:6.1f}%")
        print(f"    Mean Metadata Complete  : {agg['mean_metadata_completeness_%']:6.1f}%")
        print(f"    Mean Element Complete   : {agg['mean_element_completeness_%']:6.1f}%")
        print(f"    Mean Sample ID Quality  : {agg['mean_sample_id_quality_%']:6.1f}%")
        print(f"    Mean Value Plausibility : {agg['mean_value_plausibility_%']:6.1f}%")
        print(f"    Mean Internal Consistency: {agg['mean_internal_consistency_%']:6.1f}%")
        print(f"    Total rows extracted    : {agg['total_rows_extracted']}")

        # Save aggregated metrics
        metrics_path = output_dir / "nogt_batch_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            per_paper = {}
            for pr in batch.paper_results:
                if pr.quality:
                    per_paper[pr.paper_id] = pr.quality
            # Collect self-correction stats
            correction_stats = {
                "papers_needing_correction": 0,
                "corrections_succeeded": 0,
                "total_correction_attempts": 0,
                "per_paper_corrections": [],
            }
            for pr in batch.paper_results:
                cm = pr.extraction.correction_metrics if pr.extraction else None
                if cm and cm.correction_needed:
                    correction_stats["papers_needing_correction"] += 1
                    correction_stats["total_correction_attempts"] += len(cm.attempts)
                    if cm.correction_succeeded:
                        correction_stats["corrections_succeeded"] += 1
                    correction_stats["per_paper_corrections"].append(cm.to_dict())

            json.dump({
                "model": batch.model,
                "provider": batch.provider,
                "aggregate": agg,
                "per_paper": per_paper,
                "self_correction": correction_stats,
            }, f, indent=2, default=str)
        print(f"\n  Metrics saved -> {metrics_path}")

        # Print self-correction summary
        corr_needed = sum(
            1 for pr in batch.paper_results
            if pr.extraction and pr.extraction.correction_metrics
            and pr.extraction.correction_metrics.correction_needed
        )
        if corr_needed > 0:
            corr_success = sum(
                1 for pr in batch.paper_results
                if pr.extraction and pr.extraction.correction_metrics
                and pr.extraction.correction_metrics.correction_succeeded
            )
            print(f"\n  SELF-CORRECTION STATS:")
            print(f"    Papers needing correction : {corr_needed}")
            print(f"    Corrections succeeded     : {corr_success}")
            print(f"    Success rate              : {corr_success/corr_needed*100:.0f}%")
