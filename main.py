"""
main.py - CLI entry point for the geochemical data extraction benchmarking system.

Usage examples:

  # Extract with a single LLM (Claude):
  python -m geochem_benchmark.main extract \\
      --pdf Friday/2018_Yuan_etal.pdf \\
      --supplementary Friday/2018_Yuan_etal.xlsx \\
      --provider claude \\
      --model claude-sonnet-4-6 \\
      --output results/claude_output.xlsx

  # Extract with multiple supplementary files (merged on sample name):
  python -m geochem_benchmark.main extract \\
      --pdf paper.pdf \\
      --supplementary supp1_major.xlsx supp2_trace.xlsx \\
      --provider claude \\
      --output results/claude_output.xlsx

  # Benchmark all providers against ground truth:
  python -m geochem_benchmark.main benchmark \\
      --pdf Friday/2018_Yuan_etal.pdf \\
      --supplementary Friday/2018_Yuan_etal.xlsx \\
      --ground-truth Friday/2018_Yuan_ground_truth.xlsx \\
      --providers claude openai gemini \\
      --output-dir results/

  # Evaluate an already-extracted file against ground truth (no LLM call):
  python -m geochem_benchmark.main eval \\
      --prediction results/claude_output.xlsx \\
      --ground-truth Friday/2018_Yuan_ground_truth.xlsx \\
      --provider claude \\
      --model claude-sonnet-4-6 \\
      --output-dir results/

  # List available models:
  python -m geochem_benchmark.main models

API keys are read from environment variables:
  ANTHROPIC_API_KEY   - for Claude
  OPENAI_API_KEY      - for GPT models
  GOOGLE_API_KEY      - for Gemini models
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Load .env file if present (keeps API keys out of source code)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    # Check for .env relative to this file, then the project root, then cwd
    _pkg_dir = Path(__file__).resolve().parent
    _candidates = [
        _pkg_dir / ".env",
        _pkg_dir.parent / ".env",
        Path.cwd() / ".env",
    ]
    _loaded = False
    for _env_path in _candidates:
        if _env_path.exists():
            load_dotenv(str(_env_path))
            print(f"Loaded API keys from {_env_path}", file=sys.stderr)
            _loaded = True
            break
    if not _loaded:
        load_dotenv()  # also checks current working directory
except ImportError:
    pass  # dotenv optional; fall back to system environment variables

# ──────────────────────────────────────────────────────────────────────────────
# Add parent directory to sys.path if running as script
# ──────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from geochem_benchmark.batch_runner import run_batch, run_batch_no_gt
from geochem_benchmark.evaluator import (
    Evaluator,
    compare_all_results,
    completeness_leaderboard,
    cross_model_agreement,
    evaluate_completeness,
    leaderboard,
    save_detailed_report,
)
from geochem_benchmark.llm_clients import create_client, list_available_models
from geochem_benchmark.paper_registry import list_all_paper_ids, discover_all_papers, resolve_paper_by_id
from geochem_benchmark.pipeline import ExtractionPipeline, run_all_models
from geochem_benchmark.tabledetector import TableDetectorBackend

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _make_vision_client(args: argparse.Namespace):
    """Create a separate vision LLM client if --vision-provider is specified."""
    vp = getattr(args, "vision_provider", None)
    if not vp:
        return None
    vm = getattr(args, "vision_model", None)
    api_key = getattr(args, "api_key", None)
    client = create_client(provider=vp, model=vm or None, api_key=api_key or None)
    logger.info("Vision client: %s", client)
    return client


# ──────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> None:
    """Run extraction with a single LLM and save results."""
    client = create_client(
        provider=args.provider,
        model=args.model or None,
        api_key=args.api_key or None,
    )
    pipeline = ExtractionPipeline(
        llm_client=client,
        use_tool_calling=not args.no_tool_calling,
        use_llm_table_filter=args.llm_table_filter,
        use_self_correction=not args.no_self_correction,
        use_vision=not args.no_vision,
        vision_client=_make_vision_client(args),
        verbose=args.verbose,
        table_detector_backend=TableDetectorBackend(args.table_detector),
    )
    supp_paths = args.supplementary or []
    result = pipeline.run(
        pdf_path=args.pdf,
        supplementary_paths=supp_paths,
        this_paper_deposit=args.deposit or None,
    )

    print(f"\nExtraction summary:")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    # Show self-correction info if it was used
    cm = result.correction_metrics
    if cm and cm.correction_needed:
        status = "succeeded" if cm.correction_succeeded else "partially improved"
        print(f"\n  Self-correction: {status} after {len(cm.attempts)} attempt(s)")
        print(f"    Rows: {cm.initial_row_count} -> {cm.final_row_count}")
        print(f"    Quality: {cm.initial_quality_score:.0f}% -> {cm.final_quality_score:.0f}%")

    # Show metadata detail when few/no sample rows (common for PDF-only)
    if result.n_samples == 0:
        print(f"\n  No sample rows extracted.")
        if result.is_pdf_only:
            print(f"  (PDF-only extraction — no supplementary spreadsheet provided)")
        print(f"\n  Extracted metadata:")
        print(result.metadata_summary())

    out_path = Path(args.output)
    if out_path.suffix.lower() == ".csv":
        saved = result.to_csv(out_path)
    else:
        saved = result.to_excel(out_path)
    print(f"\nSaved {result.n_samples} rows → {saved}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate an already-extracted file. Uses ground truth if provided, else completeness metrics."""
    pred_path = Path(args.prediction)
    if not pred_path.exists():
        print(f"ERROR: Prediction file not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    # Load prediction DataFrame
    if pred_path.suffix.lower() in (".xlsx", ".xls"):
        pred_df = pd.read_excel(str(pred_path), dtype=object)
    else:
        pred_df = pd.read_csv(str(pred_path), dtype=object)
    pred_df.columns = [str(c).strip() for c in pred_df.columns]
    print(f"Loaded prediction: {len(pred_df)} rows from {pred_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_key = f"{args.provider}_{args.model}".replace("/", "_").replace(" ", "_")

    if args.ground_truth:
        # Full evaluation against ground truth
        evaluator = Evaluator(ground_truth_path=args.ground_truth)
        report = evaluator.evaluate_dataframe(
            pred_df=pred_df,
            model=args.model or "unknown",
            provider=args.provider or "unknown",
        )
        report.print_summary()

        report_path = output_dir / f"report_{safe_key}.xlsx"
        save_detailed_report(report, report_path)
        print(f"\nDetailed report saved → {report_path}")

        metrics_path = output_dir / f"metrics_{safe_key}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"Metrics JSON saved → {metrics_path}")

    else:
        # Completeness-only evaluation (no ground truth needed)
        comp = evaluate_completeness(pred_df, model=args.model or "unknown", provider=args.provider or "unknown")
        print(f"\n{'='*60}")
        print(f"  Completeness Report: {args.provider}/{args.model}")
        print(f"{'='*60}")
        print(f"  Rows extracted          : {comp['row_count']}")
        print(f"  Schema compliance       : {comp['schema_compliance_%']}%")
        print(f"  Metadata completeness   : {comp['metadata_completeness_%']}%")
        print(f"  Element completeness    : {comp['element_completeness_%']}%  ({len(comp['present_elements'])}/73 elements)")
        if comp["null_metadata_fields"]:
            print(f"  Null metadata fields    : {', '.join(comp['null_metadata_fields'])}")
        print(f"{'='*60}")

        metrics_path = output_dir / f"completeness_{safe_key}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2, default=str)
        print(f"\nCompleteness report saved → {metrics_path}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run extraction across multiple LLMs and optionally score against ground truth."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build LLM clients
    clients = []
    for provider in args.providers:
        model = args.model or None
        api_key = args.api_key or None
        try:
            client = create_client(provider=provider, model=model, api_key=api_key)
            clients.append(client)
            print(f"  Registered: {client}")
        except Exception as exc:
            print(f"  WARNING: Could not create client for '{provider}': {exc}", file=sys.stderr)

    if not clients:
        print("ERROR: No LLM clients could be initialised.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning extraction with {len(clients)} model(s)...")

    # Run all models
    supp_paths = args.supplementary or []
    results = run_all_models(
        pdf_path=args.pdf,
        supplementary_paths=supp_paths,
        clients=clients,
        this_paper_deposit=args.deposit or None,
        use_tool_calling=not getattr(args, "no_tool_calling", False),
        use_llm_table_filter=args.llm_table_filter,
        use_self_correction=not getattr(args, "no_self_correction", False),
        use_vision=not getattr(args, "no_vision", False),
        vision_client=_make_vision_client(args),
        verbose=getattr(args, "verbose", False),
        table_detector_backend=TableDetectorBackend(args.table_detector),
    )

    # Save raw extraction outputs
    for key, result in results.items():
        safe_key = key.replace("/", "_").replace(" ", "_")
        out_xlsx = output_dir / f"extraction_{safe_key}.xlsx"
        result.to_excel(out_xlsx)
        print(f"  Saved extraction: {out_xlsx}")

    # Evaluate against ground truth (optional)
    if not args.ground_truth:
        # Completeness + cross-model agreement (no ground truth needed)
        pred_dfs = {}
        for key, result in results.items():
            safe_key = key.replace("/", "_").replace(" ", "_")
            xlsx = output_dir / f"extraction_{safe_key}.xlsx"
            if xlsx.exists():
                pred_dfs[key] = pd.read_excel(str(xlsx), dtype=object)

        lb = completeness_leaderboard(pred_dfs)
        print("\n" + "=" * 70)
        print("  COMPLETENESS LEADERBOARD  (no ground truth)")
        print("=" * 70)
        print(lb.to_string())
        print("=" * 70)

        lb_path = output_dir / "completeness_leaderboard.xlsx"
        lb.to_excel(str(lb_path), index=True)
        print(f"\nLeaderboard saved → {lb_path}")

        if len(pred_dfs) > 1:
            agreement = cross_model_agreement(pred_dfs)
            agree_path = output_dir / "cross_model_agreement.json"
            with open(agree_path, "w", encoding="utf-8") as f:
                json.dump(agreement, f, indent=2, default=str)
            print(f"Cross-model agreement saved → {agree_path}")

            # Print per-field agreement summary
            print("\n  Cross-model metadata agreement:")
            for field, info in agreement.items():
                print(f"    {field:<45} {info['agreement_%']:5.1f}%")
        return

    print(f"\nEvaluating against ground truth: {args.ground_truth}")
    evaluator = Evaluator(ground_truth_path=args.ground_truth)
    reports = compare_all_results(results, evaluator)

    # Save detailed per-model reports
    for key, report in reports.items():
        safe_key = key.replace("/", "_").replace(" ", "_")
        report_path = output_dir / f"report_{safe_key}.xlsx"
        save_detailed_report(report, report_path)
        print(f"  Saved report: {report_path}")

    # Print leaderboard
    lb = leaderboard(reports)
    print("\n" + "=" * 70)
    print("  LEADERBOARD")
    print("=" * 70)
    print(lb.to_string())
    print("=" * 70)

    # Save leaderboard
    lb_path = output_dir / "leaderboard.xlsx"
    lb.to_excel(str(lb_path), index=True)
    print(f"\nLeaderboard saved → {lb_path}")

    # Save JSON metrics
    metrics_path = output_dir / "metrics.json"
    metrics_data = {key: report.to_dict() for key, report in reports.items()}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, default=str)
    print(f"Metrics JSON saved → {metrics_path}")


def cmd_batch(args: argparse.Namespace) -> None:
    """Run extraction + evaluation across all papers with ground truths."""
    client = create_client(
        provider=args.provider,
        model=args.model or None,
        api_key=args.api_key or None,
    )

    project_root = Path(__file__).resolve().parent
    paper_ids = args.papers if hasattr(args, "papers") and args.papers else None

    pdf_only = getattr(args, "pdf_only", False)
    include_pdf_only = getattr(args, "include_pdf_only", False)

    batch_result = run_batch(
        client=client,
        project_root=project_root,
        output_dir=Path(args.output_dir),
        paper_ids=paper_ids,
        use_tool_calling=not args.no_tool_calling,
        use_llm_table_filter=args.llm_table_filter,
        use_self_correction=not args.no_self_correction,
        use_vision=not args.no_vision,
        vision_client=_make_vision_client(args),
        require_supplementary=not include_pdf_only and not pdf_only,
        only_pdf_only=pdf_only,
        verbose=args.verbose,
        table_detector_backend=args.table_detector,
    )

    print(f"\nBatch complete: {batch_result.n_success}/{batch_result.n_total} papers succeeded")


def cmd_batch_nogt(args: argparse.Namespace) -> None:
    """Run extraction on all papers WITHOUT ground truth (quality assessment only)."""
    client = create_client(
        provider=args.provider,
        model=args.model or None,
        api_key=args.api_key or None,
    )

    project_root = Path(__file__).resolve().parent

    pdf_only = getattr(args, "pdf_only", False)
    include_pdf_only = getattr(args, "include_pdf_only", False)

    batch_result = run_batch_no_gt(
        client=client,
        project_root=project_root,
        output_dir=Path(args.output_dir),
        exclude_pdf_only=not include_pdf_only and not pdf_only,
        only_pdf_only=pdf_only,
        use_tool_calling=not args.no_tool_calling,
        use_llm_table_filter=args.llm_table_filter,
        use_self_correction=not args.no_self_correction,
        use_vision=not args.no_vision,
        vision_client=_make_vision_client(args),
        verbose=args.verbose,
        table_detector_backend=args.table_detector,
    )

    print(f"\nBatch complete: {batch_result.n_success}/{batch_result.n_total} papers succeeded")


def cmd_list_papers(_args: argparse.Namespace) -> None:
    """List all papers in the registry."""
    print("\nRegistered papers with ground truths:\n")
    from geochem_benchmark.paper_registry import PAPER_REGISTRY
    for entry in PAPER_REGISTRY:
        supp_count = len(entry.get("supplementary", []))
        notes = entry.get("notes", "")
        note_str = f" ({notes})" if notes else ""
        print(f"  {entry['id']}: {supp_count} supp file(s){note_str}")
    print(f"\nTotal: {len(PAPER_REGISTRY)} papers")


def cmd_list_nogt_papers(_args: argparse.Namespace) -> None:
    """List all discovered non-GT papers."""
    from geochem_benchmark.paper_registry import discover_non_gt_papers
    project_root = Path(__file__).resolve().parent
    papers = discover_non_gt_papers(project_root, exclude_pdf_only=False)

    with_supp = [p for p in papers if p.supplementary_paths]
    pdf_only = [p for p in papers if not p.supplementary_paths]

    print(f"\nDiscovered {len(papers)} papers without ground truth:\n")
    print(f"  --- With supplementary data ({len(with_supp)}) ---")
    for p in with_supp:
        print(f"  {p.id}: {len(p.supplementary_paths)} supp file(s)")
    print(f"\n  --- PDF-only ({len(pdf_only)}) ---")
    for p in pdf_only:
        print(f"  {p.id}")
    print(f"\nTotal: {len(papers)} ({len(with_supp)} with supp, {len(pdf_only)} PDF-only)")


def cmd_discover(args: argparse.Namespace) -> None:
    """Scan the project and show all papers with their file associations."""
    project_root = Path(__file__).resolve().parent
    papers = discover_all_papers(project_root)

    if not papers:
        print("No papers found. Check that data/ and data/Spreadsheets/ directories exist.")
        return

    # Categorize
    gt_with_supp = [p for p in papers if p.has_ground_truth and p.has_supplementary]
    gt_no_supp = [p for p in papers if p.has_ground_truth and not p.has_supplementary]
    nogt_with_supp = [p for p in papers if not p.has_ground_truth and p.has_supplementary]
    nogt_no_supp = [p for p in papers if not p.has_ground_truth and not p.has_supplementary]

    print(f"\n{'='*80}")
    print(f"  PAPER DISCOVERY REPORT")
    print(f"{'='*80}")
    print(f"  Total papers found: {len(papers)}")
    print(f"    With ground truth + supplementary : {len(gt_with_supp)}  (full benchmark)")
    print(f"    With ground truth, no supplementary: {len(gt_no_supp)}  (PDF-only + eval)")
    print(f"    No ground truth + supplementary   : {len(nogt_with_supp)}  (extraction only)")
    print(f"    No ground truth, no supplementary : {len(nogt_no_supp)}  (PDF-only, no eval)")
    print()

    def _print_section(title: str, items: list, show_details: bool = False):
        if not items:
            return
        print(f"  --- {title} ({len(items)}) ---")
        for p in items:
            supp_count = len(p.supplementary_paths)
            parts = [f"  {p.id}"]
            if supp_count:
                parts.append(f"{supp_count} supp")
            flags = []
            if p.has_pdf:
                flags.append("PDF")
            if p.has_supplementary:
                flags.append("SUPP")
            if p.has_ground_truth:
                flags.append("GT")
            parts.append(f"[{'+'.join(flags)}]")
            if p.notes:
                parts.append(f"({p.notes})")
            print("    ".join(parts))

            if show_details or getattr(args, "verbose", False):
                if p.pdf_path:
                    print(f"        PDF: {p.pdf_path.name}")
                for sp in p.supplementary_paths:
                    print(f"        Supp: {sp.name}")
                if p.ground_truth_path:
                    print(f"        GT: {p.ground_truth_path.name}")
        print()

    _print_section("Ground truth + Supplementary (full benchmark)", gt_with_supp)
    _print_section("Ground truth, no Supplementary (PDF-only + eval)", gt_no_supp)
    _print_section("No ground truth + Supplementary (extraction only)", nogt_with_supp)
    _print_section("No ground truth, no Supplementary (PDF-only)", nogt_no_supp)

    print(f"{'='*80}")
    print(f"  Use 'run-paper <PAPER_ID>' to extract any paper above.")
    print(f"  Use '--verbose' to see file paths for each paper.")
    print(f"{'='*80}\n")


def cmd_run_paper(args: argparse.Namespace) -> None:
    """Extract a paper by ID — auto-resolves paths, evaluates if GT exists."""
    project_root = Path(__file__).resolve().parent
    paper_id = args.paper_id

    # Look up the paper
    info = resolve_paper_by_id(project_root, paper_id)
    if info is None:
        print(f"ERROR: Paper '{paper_id}' not found.", file=sys.stderr)
        print("Use 'discover' command to see all available papers.", file=sys.stderr)
        sys.exit(1)

    print(f"\nPaper: {info.id}")
    print(f"  PDF: {'YES' if info.has_pdf else 'MISSING'}" +
          (f" ({info.pdf_path.name})" if info.pdf_path else ""))
    print(f"  Supplementary: {'YES' if info.has_supplementary else 'NONE'}" +
          (f" ({len(info.supplementary_paths)} files)" if info.supplementary_paths else ""))
    print(f"  Ground truth: {'YES' if info.has_ground_truth else 'NONE'}" +
          (f" ({info.ground_truth_path.name})" if info.ground_truth_path else ""))
    if info.notes:
        print(f"  Notes: {info.notes}")

    if not info.has_pdf:
        print(f"\nERROR: No PDF found for '{info.id}'. Cannot extract.", file=sys.stderr)
        sys.exit(1)

    # Create client and pipeline
    client = create_client(
        provider=args.provider,
        model=args.model or None,
        api_key=args.api_key or None,
    )
    pipeline = ExtractionPipeline(
        llm_client=client,
        use_tool_calling=not args.no_tool_calling,
        use_llm_table_filter=args.llm_table_filter,
        use_self_correction=not args.no_self_correction,
        use_vision=not args.no_vision,
        vision_client=_make_vision_client(args),
        verbose=args.verbose,
        table_detector_backend=TableDetectorBackend(args.table_detector),
    )

    # Extract
    print(f"\nExtracting with {client}...")
    supp_paths = info.supplementary_paths if info.has_supplementary else []
    result = pipeline.run(
        pdf_path=info.pdf_path,
        supplementary_paths=supp_paths,
    )

    print(f"\nExtraction summary:")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    # Show self-correction info
    cm = result.correction_metrics
    if cm and cm.correction_needed:
        status = "succeeded" if cm.correction_succeeded else "partially improved"
        print(f"\n  Self-correction: {status} after {len(cm.attempts)} attempt(s)")
        print(f"    Rows: {cm.initial_row_count} -> {cm.final_row_count}")
        print(f"    Quality: {cm.initial_quality_score:.0f}% -> {cm.final_quality_score:.0f}%")

    # Show metadata detail when few/no sample rows
    if result.n_samples == 0:
        print(f"\n  No sample rows extracted.")
        if result.is_pdf_only:
            print(f"  (PDF-only extraction — no supplementary spreadsheet available)")
        print(f"\n  Extracted metadata:")
        print(result.metadata_summary())

    # Save extraction
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = output_dir / f"extraction_{info.id}.xlsx"
    result.to_excel(out_xlsx)
    print(f"\nSaved {result.n_samples} rows -> {out_xlsx}")

    # Evaluate if ground truth exists
    if info.has_ground_truth:
        print(f"\nEvaluating against ground truth: {info.ground_truth_path.name}")
        from geochem_benchmark.evaluator import Evaluator, save_detailed_report
        evaluator = Evaluator(ground_truth_path=info.ground_truth_path)
        report = evaluator.evaluate(result)
        report.print_summary()

        report_path = output_dir / f"report_{info.id}.xlsx"
        save_detailed_report(report, report_path)
        print(f"Detailed report saved -> {report_path}")
    else:
        # Quality assessment only
        print(f"\nNo ground truth — running quality assessment...")
        from geochem_benchmark.evaluator import evaluate_quality
        pred_df = result.to_dataframe()
        quality = evaluate_quality(
            pred_df=pred_df,
            model=result.llm_model,
            provider=result.llm_provider,
        )
        print(f"  Quality score: {quality['quality_score_%']:.1f}%")
        print(f"  Metadata completeness: {quality['metadata_completeness_%']:.0f}%")
        print(f"  Element completeness: {quality['element_completeness_%']:.0f}%")
        print(f"  Sample ID quality: {quality['sample_id_quality_%']:.0f}%")
        print(f"  Value plausibility: {quality['value_plausibility_%']:.0f}%")
        if quality.get("issues"):
            for issue in quality["issues"][:5]:
                print(f"    ! {issue}")

        # Save quality metrics
        metrics_path = output_dir / f"quality_{info.id}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2, default=str)
        print(f"  Quality metrics saved -> {metrics_path}")


def cmd_models(_args: argparse.Namespace) -> None:
    """List all available models by provider."""
    print("\nAvailable models:\n")
    for provider, models in list_available_models().items():
        print(f"  {provider}:")
        for m in models:
            print(f"    - {m}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geochem_benchmark",
        description="Geochemical data extraction & LLM benchmarking tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── extract ──────────────────────────────────────────────────────────────
    p_extract = sub.add_parser("extract", help="Extract data using a single LLM.")
    _add_input_args(p_extract)
    p_extract.add_argument("--provider", required=True,
                           choices=["claude", "openai", "gemini"],
                           help="LLM provider.")
    p_extract.add_argument("--model", default=None,
                           help="Model ID (uses provider default if omitted).")
    p_extract.add_argument("--output", default="extraction_output.xlsx",
                           help="Output path (.xlsx or .csv). Default: extraction_output.xlsx")
    _add_common_args(p_extract)

    # ── benchmark ────────────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Benchmark multiple LLMs vs ground truth.")
    _add_input_args(p_bench)
    p_bench.add_argument("--ground-truth", default=None,
                         help="Path to ground truth Excel file (optional). "
                              "If omitted, extractions are saved but no scores are computed.")
    p_bench.add_argument("--providers", nargs="+",
                         default=["claude", "openai", "gemini"],
                         choices=["claude", "openai", "gemini"],
                         help="LLM providers to benchmark.")
    p_bench.add_argument("--model", default=None,
                         help="Use the same model for all providers (overrides defaults).")
    p_bench.add_argument("--output-dir", default="benchmark_results/",
                         help="Directory for all output files. Default: benchmark_results/")
    _add_common_args(p_bench)

    # ── eval ─────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "eval",
        help="Score an already-extracted file against ground truth (no LLM call).",
    )
    p_eval.add_argument("--prediction", required=True,
                        help="Path to the extracted output (.xlsx or .csv).")
    p_eval.add_argument("--ground-truth", default=None,
                        help="Path to ground truth Excel file (optional). "
                             "If omitted, completeness metrics are reported instead.")
    p_eval.add_argument("--provider", default="unknown",
                        help="Provider label for the report (e.g. claude).")
    p_eval.add_argument("--model", default="unknown",
                        help="Model label for the report (e.g. claude-sonnet-4-6).")
    p_eval.add_argument("--output-dir", default="eval_results/",
                        help="Directory for report and metrics files. Default: eval_results/")

    # ── batch ─────────────────────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch",
        help="Run extraction + evaluation across all papers with ground truths.",
    )
    p_batch.add_argument("--provider", default="claude",
                         choices=["claude", "openai", "gemini"],
                         help="LLM provider. Default: claude")
    p_batch.add_argument("--model", default=None,
                         help="Model ID (uses provider default if omitted).")
    p_batch.add_argument("--output-dir", default="batch_results/",
                         help="Directory for all output files. Default: batch_results/")
    p_batch.add_argument("--papers", nargs="+", default=None,
                         metavar="PAPER_ID",
                         help="Specific paper IDs to process (default: all). "
                              "Use 'list-papers' to see available IDs.")
    p_batch_supp = p_batch.add_mutually_exclusive_group()
    p_batch_supp.add_argument("--include-pdf-only", action="store_true",
                              help="Include papers with no supplementary files alongside those with supp.")
    p_batch_supp.add_argument("--pdf-only", action="store_true",
                              help="Process ONLY papers with no supplementary files (PDF-only extraction).")
    _add_common_args(p_batch)

    # ── batch-nogt ─────────────────────────────────────────────────────────
    p_batch_nogt = sub.add_parser(
        "batch-nogt",
        help="Run extraction on all papers WITHOUT ground truth (quality assessment only).",
    )
    p_batch_nogt.add_argument("--provider", default="claude",
                              choices=["claude", "openai", "gemini"],
                              help="LLM provider. Default: claude")
    p_batch_nogt.add_argument("--model", default=None,
                              help="Model ID (uses provider default if omitted).")
    p_batch_nogt.add_argument("--output-dir", default="nogt_results/",
                              help="Directory for all output files. Default: nogt_results/")
    p_nogt_supp = p_batch_nogt.add_mutually_exclusive_group()
    p_nogt_supp.add_argument("--include-pdf-only", action="store_true",
                              help="Include papers with no supplementary files alongside those with supp.")
    p_nogt_supp.add_argument("--pdf-only", action="store_true",
                              help="Process ONLY papers with no supplementary files (PDF-only extraction).")
    _add_common_args(p_batch_nogt)

    # ── run-paper ─────────────────────────────────────────────────────────────
    p_run_paper = sub.add_parser(
        "run-paper",
        help="Extract a single paper by ID (auto-resolves paths, evaluates if GT exists).",
    )
    p_run_paper.add_argument("paper_id",
                             help="Paper ID (use 'discover' to see all available IDs). "
                                  "Supports partial matching.")
    p_run_paper.add_argument("--provider", default="claude",
                             choices=["claude", "openai", "gemini"],
                             help="LLM provider. Default: claude")
    p_run_paper.add_argument("--model", default=None,
                             help="Model ID (uses provider default if omitted).")
    p_run_paper.add_argument("--output-dir", default="paper_results/",
                             help="Directory for output files. Default: paper_results/")
    _add_common_args(p_run_paper)

    # ── discover ──────────────────────────────────────────────────────────────
    p_discover = sub.add_parser(
        "discover",
        help="Scan all papers in data/ — show PDFs, supplementary files, and ground truths.",
    )
    p_discover.add_argument("--verbose", "-v", action="store_true",
                            help="Show file paths for each paper.")

    # ── list-papers ───────────────────────────────────────────────────────────
    sub.add_parser("list-papers", help="List all registered papers with ground truths.")

    # ── list-nogt-papers ──────────────────────────────────────────────────────
    sub.add_parser("list-nogt-papers", help="List all discovered papers without ground truth.")

    # ── models ───────────────────────────────────────────────────────────────
    sub.add_parser("models", help="List all available model IDs by provider.")

    return parser


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pdf", required=True, help="Path to the paper PDF.")
    p.add_argument("--supplementary", nargs="*", default=None,
                   metavar="FILE",
                   help="One or more supplementary data files (.xlsx, .xls, .csv). "
                        "Multiple files are merged on sample name. "
                        "If omitted, runs PDF-only extraction (metadata only).")
    p.add_argument("--deposit", default=None,
                   help="Deposit name to filter supplementary rows (optional).")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-key", default=None,
                   help="API key (prefers environment variable if not set).")
    p.add_argument("--no-tool-calling", action="store_true",
                   help="Disable tool calling for Claude (use plain JSON output instead).")
    p.add_argument("--llm-table-filter", action="store_true",
                   help="Use LLM to filter/parse the supplementary table rows "
                        "(slower but handles complex table structures).")
    p.add_argument("--table-detector", default="auto",
                   choices=["auto", "docling", "camelot", "pdfplumber"],
                   help="PDF table detector backend: auto (smart selection, default), "
                        "docling (state-of-the-art), camelot (borderless), pdfplumber (grid). "
                        "Used for PDF-only extraction and fallback detection.")
    p.add_argument("--no-self-correction", action="store_true",
                   help="Disable agentic self-correction for failed table extractions.")
    p.add_argument("--no-vision", action="store_true",
                   help="Disable vision-based PDF page extraction "
                        "(vision is ON by default for PDF-only papers).")
    p.add_argument("--vision-provider", default=None,
                   choices=["claude", "openai", "gemini"],
                   help="Separate LLM provider for vision API calls. "
                        "If omitted, uses the same provider as --provider.")
    p.add_argument("--vision-model", default=None,
                   help="Model ID for vision API calls. "
                        "If omitted, uses the provider default.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose logging.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)

    dispatch = {
        "extract":          cmd_extract,
        "benchmark":        cmd_benchmark,
        "eval":             cmd_eval,
        "batch":            cmd_batch,
        "batch-nogt":       cmd_batch_nogt,
        "run-paper":        cmd_run_paper,
        "discover":         cmd_discover,
        "list-papers":      cmd_list_papers,
        "list-nogt-papers": cmd_list_nogt_papers,
        "models":           cmd_models,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
