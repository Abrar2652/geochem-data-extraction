"""
pipeline.py - Main extraction pipeline orchestrating PDF + table → schema rows.

Strategy (two-stage hybrid):
  Stage 1: LLM extracts paper-level metadata from PDF text
  Stage 2: Python parses supplementary table (fast, reliable for numbers)
  Stage 3: Merge metadata + numerical rows into final SampleRow objects
  (Optional) Stage 2b: LLM assistance for ambiguous/complex table structures
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .llm_clients import LLMClient, ClaudeClient
from .pdf_reader import PDFContent, extract_pdf, get_paper_text_for_llm, get_data_pages_text
from .pdf_vision import render_data_pages, has_pdfplumber_tables
from .tabledetector import extract_tables_as_text, extract_tables_from_pdf, TableDetectorBackend
from .prompts import (
    build_paper_intelligence_prompt,
    build_metadata_prompt,
    build_table_filter_prompt,
    build_metadata_tool_schema,
    build_pdf_table_extraction_prompt,
    build_vision_table_extraction_prompt,
)
from .schema import PaperMetadata, SampleRow, ALL_COLUMNS, ELEMENT_SYMBOLS
from .table_reader import (
    SupplementaryTable, read_supplementary, read_multiple_supplementary,
    dataframe_to_text, read_pdf_table, parse_text_tables_from_pages,
)
from .knowledge_base import (
    validate_and_enrich_metadata,
    detect_method_from_filename,
)
from .agentic_corrector import (
    quick_quality_check,
    correction_loop,
    pdf_correction_loop,
    PDFExtractionHints,
    QuickQualityResult,
    CorrectionMetrics,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Output of one extraction run."""
    metadata: PaperMetadata
    samples: list[SampleRow]
    llm_model: str
    llm_provider: str
    pdf_path: str
    supplementary_paths: list[str]
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    correction_metrics: Optional[CorrectionMetrics] = None

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all sample rows as a DataFrame with the 210-column schema."""
        rows = [row.to_schema_dict() for row in self.samples]
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    def to_excel(self, output_path: str | Path) -> Path:
        """Save extraction results to Excel."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_excel(str(output_path), index=False)
        logger.info("Saved %d rows to %s", len(df), output_path)
        return output_path

    def to_csv(self, output_path: str | Path) -> Path:
        """Save extraction results to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(str(output_path), index=False, encoding="utf-8-sig")
        logger.info("Saved %d rows to %s", len(df), output_path)
        return output_path

    @property
    def is_pdf_only(self) -> bool:
        """True if extraction ran without supplementary files."""
        return not self.supplementary_paths

    def summary(self) -> dict:
        return {
            "model": self.llm_model,
            "provider": self.llm_provider,
            "n_samples": self.n_samples,
            "deposit": self.metadata.deposit_name,
            "mineral": self.metadata.mineral,
            "method": self.metadata.analytical_method,
            "notes": self.notes,
            "errors": self.errors,
        }

    def metadata_summary(self) -> str:
        """Human-readable summary of extracted metadata fields."""
        meta = self.metadata.model_dump()
        lines = []
        for k, v in meta.items():
            if v is not None and str(v).strip():
                lines.append(f"    {k}: {v}")
        if not lines:
            return "    (no metadata extracted)"
        return "\n".join(lines)


@dataclass
class PaperIntelligence:
    """Pre-extraction analysis of the paper's analytical methodology.

    Extracted before metadata and table extraction to guide downstream steps:
    - elements_measured constrains PDF extraction scope (reduces T4 hallucination)
    - instrument/laboratory/standards/conditions fill T1 metadata gaps
    """
    elements_measured: list[str] = field(default_factory=list)
    expected_sample_count: Optional[int] = None
    minerals_analyzed: list[str] = field(default_factory=list)
    analytical_methods: list[str] = field(default_factory=list)
    instrument: Optional[str] = None
    laboratory: Optional[str] = None
    standards_used: Optional[str] = None
    operating_conditions: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """Orchestrates two-stage extraction: metadata (LLM) + numbers (Python)."""

    def __init__(
        self,
        llm_client: LLMClient,
        use_tool_calling: bool = True,
        use_llm_table_filter: bool = False,
        use_self_correction: bool = True,
        use_vision: bool = True,
        vision_client: LLMClient | None = None,
        correction_max_attempts: int = 2,
        correction_quality_threshold: float = 30.0,
        verbose: bool = False,
        table_detector_backend: TableDetectorBackend = TableDetectorBackend.AUTO,
    ):
        """
        Args:
            llm_client: An LLMClient instance (Claude, OpenAI, Gemini).
            use_tool_calling: If True and client is ClaudeClient, use tool
                calling for guaranteed-JSON metadata extraction.
            use_llm_table_filter: If True, also run LLM-based row filtering
                (useful when Python heuristics are insufficient).
            use_self_correction: If True, invoke LLM-powered diagnosis and
                retry when initial extraction fails or produces low quality.
            use_vision: If True, render PDF pages as images and send to
                the LLM vision API when text extraction fails.
            vision_client: Separate LLM client for vision API calls. If None,
                falls back to llm_client.
            correction_max_attempts: Max LLM diagnosis attempts per paper.
            correction_quality_threshold: Quality score below which self-
                correction triggers (0-100).
            verbose: Enable verbose logging.
            table_detector_backend: Which PDF table detector to use
                (auto, docling, camelot, pdfplumber).
        """
        self.client = llm_client
        self.vision_client = vision_client or llm_client
        self.use_tool_calling = use_tool_calling
        self.use_llm_table_filter = use_llm_table_filter
        self.use_self_correction = use_self_correction
        self.use_vision = use_vision
        self.correction_max_attempts = correction_max_attempts
        self.correction_quality_threshold = correction_quality_threshold
        self.verbose = verbose
        self.table_detector_backend = table_detector_backend
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        pdf_path: str | Path,
        supplementary_paths: str | Path | list[str | Path],
        this_paper_deposit: Optional[str] = None,
    ) -> ExtractionResult:
        """Run the full extraction pipeline.

        Args:
            pdf_path: Path to the research paper PDF.
            supplementary_paths: One or more supplementary Excel/CSV files.
                Pass a single path or a list. Multiple files are merged on
                sample name (useful when elements are split across files).
            this_paper_deposit: Optional deposit name to filter supplementary
                rows. If None, uses the reference column heuristic.

        Returns:
            ExtractionResult with metadata + per-sample rows.
        """
        if isinstance(supplementary_paths, (str, Path)):
            supplementary_paths = [supplementary_paths]
        supplementary_paths = [Path(p) for p in supplementary_paths if p]
        # Filter to only existing files
        supplementary_paths = [p for p in supplementary_paths if p.exists()]

        notes: list[str] = []
        errors: list[str] = []

        # ── Step 1: Read inputs ───────────────────────────────────────────────
        logger.info("Reading PDF: %s", pdf_path)
        pdf_content: PDFContent = extract_pdf(pdf_path)
        notes.append(f"PDF: {pdf_content.total_pages} pages, {len(pdf_content.full_text)} chars")

        supp = None
        if supplementary_paths:
            logger.info("Reading supplementary: %s", supplementary_paths)
            try:
                # Detect analytical method from filenames before reading
                file_methods = {}
                for sp in supplementary_paths:
                    method = detect_method_from_filename(sp.name)
                    if method:
                        file_methods[str(sp)] = method
                        notes.append(f"Inferred method '{method}' from filename: {sp.name}")

                supp: SupplementaryTable = read_multiple_supplementary(
                    supplementary_paths,
                    this_paper_deposit=this_paper_deposit,
                )

                # If only one method inferred from filenames and no method column
                # in the table, set it as the inferred method for all rows
                if file_methods and not supp.method_col:
                    unique_methods = set(file_methods.values())
                    if len(unique_methods) == 1:
                        supp.inferred_method = unique_methods.pop()

                notes.extend(supp.notes)
                notes.append(f"Supplementary: {supp.n_samples} sample rows after filtering")
            except Exception as exc:
                errors.append(f"Supplementary read error: {exc}")
                supp = None

            # ── Self-correction: if extraction quality is poor, diagnose and retry
            if self.use_self_correction and supplementary_paths:
                initial_quality = quick_quality_check(supp)
                if not initial_quality.passed:
                    notes.append(f"Self-correction triggered: {initial_quality.failure_reasons}")
                    logger.info("Self-correction triggered: %s", initial_quality.failure_reasons)
                    try:
                        corrected_supp, corr_metrics = correction_loop(
                            client=self.client,
                            paths=supplementary_paths,
                            initial_supp=supp,
                            initial_quality=initial_quality,
                            this_paper_deposit=this_paper_deposit,
                            max_attempts=self.correction_max_attempts,
                            quality_threshold=self.correction_quality_threshold,
                        )
                        _correction_metrics = corr_metrics
                        # Use corrected result if it's any improvement over initial
                        improved = (
                            corrected_supp is not None
                            and corr_metrics.final_quality_score > corr_metrics.initial_quality_score
                        )
                        if improved:
                            supp = corrected_supp
                            status = "succeeded" if corr_metrics.correction_succeeded else "partially improved"
                            notes.append(
                                f"Self-correction {status}: {corr_metrics.initial_row_count} -> "
                                f"{corr_metrics.final_row_count} rows after "
                                f"{len(corr_metrics.attempts)} attempt(s)"
                            )
                        else:
                            notes.append(
                                f"Self-correction attempted but did not improve results "
                                f"({len(corr_metrics.attempts)} attempt(s))"
                            )
                    except Exception as corr_exc:
                        logger.warning("Self-correction failed: %s", corr_exc)
                        errors.append(f"Self-correction error: {corr_exc}")
                        _correction_metrics = None
                else:
                    _correction_metrics = None
            else:
                _correction_metrics = None
        else:
            notes.append("No supplementary files — PDF-only extraction")
            _correction_metrics = None

        # ── Step 1.5: Paper Intelligence Blueprint ─────────────────────────────
        logger.info("Stage 0: Extracting paper intelligence with %s", self.client.model)
        intelligence = self._extract_paper_intelligence(pdf_content, errors)
        if intelligence.elements_measured:
            notes.append(
                f"Intelligence: {len(intelligence.elements_measured)} elements, "
                f"~{intelligence.expected_sample_count} samples, "
                f"methods={intelligence.analytical_methods}"
            )
        else:
            notes.append("Intelligence: no elements detected (will use unconstrained extraction)")

        # ── Step 2: LLM metadata extraction ──────────────────────────────────
        logger.info("Stage 1: Extracting paper-level metadata with %s", self.client.model)
        paper_text = get_paper_text_for_llm(pdf_content, max_chars=20000)
        table_preview = (
            dataframe_to_text(supp.data_df.head(5), max_rows=5)
            if supp else "(supplementary table unavailable)"
        )

        metadata = self._extract_metadata(paper_text, table_preview, errors)

        # ── Step 2.5: Enhance metadata with intelligence (fill gaps only) ────
        _intel_fills = []
        if intelligence.instrument and not metadata.instrument_type_model:
            metadata.instrument_type_model = intelligence.instrument
            _intel_fills.append("instrument")
        if intelligence.laboratory and not metadata.laboratory_location:
            metadata.laboratory_location = intelligence.laboratory
            _intel_fills.append("laboratory")
        if intelligence.standards_used and not metadata.standards_used:
            metadata.standards_used = intelligence.standards_used
            _intel_fills.append("standards")
        if intelligence.operating_conditions and not metadata.operating_conditions:
            metadata.operating_conditions = intelligence.operating_conditions
            _intel_fills.append("conditions")
        if _intel_fills:
            notes.append(f"Intelligence filled metadata gaps: {', '.join(_intel_fills)}")

        notes.append(f"Metadata: deposit='{metadata.deposit_name}', mineral='{metadata.mineral}'")

        # ── Step 3: Build sample rows ─────────────────────────────────────────
        logger.info("Stage 2: Building sample rows from available sources")
        samples: list[SampleRow] = []
        pdf_samples: list[SampleRow] = []

        # Step 3a: Extract from supplementary files if available
        if supp:
            logger.info("Extracting %d samples from supplementary table", supp.n_samples)
            if self.use_llm_table_filter:
                # LLM-assisted row extraction (for complex table structures)
                samples = self._llm_table_extraction(supp, metadata, errors)
                notes.append("Used LLM-assisted table extraction (supplementary)")
            else:
                # Pure Python extraction (fast, reliable for well-structured tables)
                samples = self._python_table_extraction(supp, metadata)
                notes.append("Used Python table extraction (supplementary)")

        # Step 3b: Extract from PDF tables — fallback chain prioritises
        # direct DataFrame extraction (lossless numbers) over LLM-based.
        # Skip entirely when supplementary extraction already succeeded
        # (avoids expensive Docling calls for papers with good suppl data).
        cached_backend_texts: list[str] = []
        text_extracted_pages: set[int] = set()
        supp_is_sufficient = supp and supp.n_samples >= 5
        if not supp_is_sufficient and (pdf_content.tables_text or pdf_content.full_text):
            logger.info("Attempting PDF table extraction (supplementary: %s)", "yes" if supp else "no")

            # Step 3b-1: DIRECT extraction — read numbers straight from
            # DataFrames produced by Docling/Camelot/pdfplumber + text parsing.
            # Numbers never pass through an LLM → zero hallucination.
            # Also caches table text from all backends for LLM fallback.
            pdf_samples, cached_backend_texts, text_extracted_pages = self._direct_pdf_table_extraction(
                pdf_path, pdf_content, metadata, errors,
            )
            if pdf_samples:
                notes.append(f"PDF table extraction (direct): {len(pdf_samples)} samples")
                logger.info("Direct PDF extraction yielded %d samples", len(pdf_samples))

            # Steps 3b-1.5 through 3b-4 are more aggressive extraction
            # strategies. Only run them when direct extraction found nothing.
            # (supp_is_sufficient already gated the outer block)
            need_aggressive_pdf = not pdf_samples

            # Step 3b-1.5: PDF self-correction — when direct extraction yields
            # 0 samples but backends found tables, ask LLM to diagnose the
            # table structure and retry with adjusted parameters.
            if need_aggressive_pdf and cached_backend_texts and self.use_self_correction:
                def _retry_with_hints(hints: PDFExtractionHints) -> list[SampleRow]:
                    return self._direct_pdf_table_with_hints(
                        pdf_path, pdf_content, metadata, errors, hints,
                    )

                pdf_samples = pdf_correction_loop(
                    client=self.client,
                    pdf_path=pdf_path,
                    pdf_content=pdf_content,
                    metadata=metadata,
                    initial_samples=[],
                    backend_table_texts=cached_backend_texts,
                    intelligence=intelligence,
                    errors=errors,
                    extract_fn=_retry_with_hints,
                    max_attempts=1,
                    paper_id=metadata.deposit_name or "unknown",
                )
                if pdf_samples:
                    notes.append(f"PDF correction loop: {len(pdf_samples)} samples")

            # Step 3b-2: LLM-based extraction from table text
            # (needed when column headers are garbled / unrecognisable)
            # Use cached multi-backend text (richer than pdfplumber alone);
            # fall back to pdfplumber's tables_text if no cached text.
            if need_aggressive_pdf and not pdf_samples:
                fallback_texts = cached_backend_texts or pdf_content.tables_text or []
                if fallback_texts:
                    pdf_samples = self._pdf_table_extraction_with_tables(
                        pdf_content, fallback_texts, metadata, errors,
                        elements_measured=intelligence.elements_measured or None,
                    )
                    if pdf_samples:
                        notes.append(f"PDF table extraction (LLM): {len(pdf_samples)} samples")
                        logger.info("LLM table extraction yielded %d samples", len(pdf_samples))

            # Step 3b-3: Vision extraction on data-dense page images.
            # Run when: (a) no samples found, OR (b) significantly fewer
            # samples than expected (garbled continuation pages).
            expected = intelligence.expected_sample_count or 0
            vision_needed = (
                (need_aggressive_pdf and not pdf_samples)
                or (pdf_samples and expected > 0
                    and len(pdf_samples) < expected * 0.85)
            )
            if vision_needed and self.use_vision:
                logger.info(
                    "Vision extraction: have %d samples, expected ~%d",
                    len(pdf_samples) if pdf_samples else 0, expected,
                )
                vision_samples = self._vision_table_extraction(
                    pdf_path, pdf_content, metadata, errors,
                    elements_measured=intelligence.elements_measured or None,
                    expected_samples=expected,
                    skip_pages=text_extracted_pages,
                )
                if vision_samples:
                    if pdf_samples:
                        # Merge: vision supplements existing direct extraction.
                        # Add vision rows for sample names not already present.
                        existing_names = {s.sample_name for s in pdf_samples if s.sample_name}
                        new_from_vision = [
                            s for s in vision_samples
                            if s.sample_name and s.sample_name not in existing_names
                        ]
                        if new_from_vision:
                            pdf_samples.extend(new_from_vision)
                            notes.append(f"Vision supplemented: +{len(new_from_vision)} new samples")
                    else:
                        pdf_samples = vision_samples
                        notes.append(f"PDF table extraction (vision): {len(pdf_samples)} samples")

            # Step 3b-4: Text-only fallback (data pages or raw text)
            if need_aggressive_pdf and not pdf_samples:
                logger.info("No samples from table/vision extraction — trying text-only fallback")
                pdf_samples = self._pdf_table_extraction(
                    pdf_content, metadata, errors,
                    elements_measured=intelligence.elements_measured or None,
                )
                if pdf_samples:
                    notes.append(f"PDF table extraction (text fallback): {len(pdf_samples)} samples")

            if not pdf_samples:
                notes.append("PDF table extraction: no sample data found")

            # Step 3b-5: Completeness gate — if we extracted far fewer samples
            # than Paper Intelligence expects, retry direct extraction with
            # relaxed thresholds (min_element_cols=2) to catch partial tables.
            if pdf_samples and intelligence.expected_sample_count:
                expected = intelligence.expected_sample_count
                actual = len(pdf_samples)
                if actual < expected * 0.5 and actual < expected - 5:
                    logger.info(
                        "Completeness gate: got %d samples but expected ~%d — "
                        "retrying direct extraction with relaxed thresholds",
                        actual, expected,
                    )
                    retry_samples = self._direct_pdf_table_extraction_relaxed(
                        pdf_path, pdf_content, metadata, errors,
                    )
                    if len(retry_samples) > actual:
                        pdf_samples = retry_samples
                        notes.append(
                            f"Completeness retry improved: {actual} → {len(retry_samples)} samples"
                        )
                        logger.info(
                            "Completeness retry: %d → %d samples",
                            actual, len(retry_samples),
                        )

        # Step 3c: Merge supplementary and PDF samples
        if pdf_samples:
            samples = self._merge_supplementary_and_pdf_samples(samples, pdf_samples)
            if supp:
                notes[-1] = f"Multi-source extraction: {len(samples)} total samples"

        # Step 4: Value verification — spot-check extracted numbers against
        # source text. If LLM-extracted values don't appear anywhere in the
        # PDF text, the extraction may be hallucinated.
        if pdf_samples and not supp:
            # Only verify PDF-only extractions (supplementary data is trusted)
            verification = self._verify_extracted_values(
                samples, pdf_content, cached_backend_texts,
            )
            if verification:
                notes.append(verification)

        result = ExtractionResult(
            metadata=metadata,
            samples=samples,
            llm_model=self.client.model,
            llm_provider=self.client.provider,
            pdf_path=str(pdf_path),
            supplementary_paths=[str(p) for p in supplementary_paths],
            notes=notes,
            errors=errors,
            correction_metrics=_correction_metrics,
        )
        logger.info(
            "Extraction complete: %d samples, %d errors",
            result.n_samples,
            len(errors),
        )
        return result

    # ── Stage 0: Paper Intelligence Blueprint ────────────────────────────────

    def _extract_paper_intelligence(
        self,
        pdf_content: PDFContent,
        errors: list[str],
    ) -> PaperIntelligence:
        """Extract analytical intelligence from the paper before main extraction.

        This focused LLM call extracts:
        - Which elements are measured (for element scope constraint)
        - Instrument/lab/standards/conditions (for T1 metadata gap-filling)
        - Expected sample count and minerals (for downstream validation)

        Gracefully returns empty PaperIntelligence on failure.
        """
        paper_text = get_paper_text_for_llm(pdf_content, max_chars=20000)
        system, user = build_paper_intelligence_prompt(paper_text)

        try:
            parsed = self.client.complete_json(
                system=system, user=user, max_tokens=2048,
            )
            if not isinstance(parsed, dict):
                raise ValueError("Paper intelligence did not return a dict")

            # Validate elements against known symbols
            valid_symbols = set(ELEMENT_SYMBOLS)
            raw_elements = parsed.get("elements_measured", [])
            elements = [
                e.lower().strip()
                for e in raw_elements
                if isinstance(e, str) and e.lower().strip() in valid_symbols
            ]

            intelligence = PaperIntelligence(
                elements_measured=elements,
                expected_sample_count=parsed.get("expected_sample_count"),
                minerals_analyzed=[
                    m.lower().strip()
                    for m in parsed.get("minerals_analyzed", [])
                    if isinstance(m, str)
                ],
                analytical_methods=[
                    m.strip()
                    for m in parsed.get("analytical_methods", [])
                    if isinstance(m, str)
                ],
                instrument=parsed.get("instrument"),
                laboratory=parsed.get("laboratory"),
                standards_used=parsed.get("standards_used"),
                operating_conditions=parsed.get("operating_conditions"),
            )
            logger.info(
                "Paper intelligence: %d elements, ~%s samples, methods=%s",
                len(intelligence.elements_measured),
                intelligence.expected_sample_count,
                intelligence.analytical_methods,
            )
            return intelligence

        except Exception as exc:
            logger.warning("Paper intelligence extraction failed: %s", exc)
            errors.append(f"Paper intelligence failed: {exc}")
            return PaperIntelligence()

    # ── Stage 1: Metadata extraction ─────────────────────────────────────────

    def _extract_metadata(
        self,
        paper_text: str,
        table_preview: str,
        errors: list[str],
    ) -> PaperMetadata:
        """Extract paper-level metadata using the LLM."""
        system, user = build_metadata_prompt(paper_text, table_preview)

        raw_dict: dict = {}
        try:
            # Primary: tool calling for Claude (guaranteed JSON schema output)
            if self.use_tool_calling and isinstance(self.client, ClaudeClient):
                tool_schema = build_metadata_tool_schema()
                raw_dict = self.client.complete_with_tool(
                    system=system,
                    user=user,
                    tool_schema=tool_schema,
                    max_tokens=2048,
                )
        except Exception as tool_exc:
            logger.warning("Tool calling failed, falling back to plain JSON: %s", tool_exc)

        # Fallback / non-Claude path: plain JSON completion
        if not raw_dict:
            try:
                raw_dict = self.client.complete_json(
                    system=system,
                    user=user,
                    max_tokens=2048,
                )
            except Exception as exc:
                errors.append(f"Metadata extraction failed: {exc}")
                logger.warning("Metadata extraction failed: %s", exc)
                return PaperMetadata()

        try:
            raw_dict = _normalise_metadata_keys(raw_dict)
            # Post-validation: enrich with domain knowledge (fills gaps only)
            raw_dict = validate_and_enrich_metadata(raw_dict)
            metadata = PaperMetadata(**{
                k: v for k, v in raw_dict.items()
                if k in PaperMetadata.model_fields
            })
            return metadata
        except Exception as exc:
            errors.append(f"Metadata parse failed: {exc}")
            logger.warning("Metadata parse failed: %s", exc)
            return PaperMetadata()

    # ── Stage 2a: Multi-backend PDF table extraction ────────────────────────────

    def _extract_all_backends_from_pdf(
        self,
        pdf_path: str | Path,
    ) -> list[str]:
        """Extract tables from PDF using all three backends simultaneously.
        
        Returns combined list of all table texts from all three backends
        (Docling, Camelot, pdfplumber).
        """
        all_tables = []
        backends_tried = []

        # Try each backend and collect all results
        for backend in [TableDetectorBackend.DOCLING, TableDetectorBackend.CAMELOT, TableDetectorBackend.PDFPLUMBER]:
            try:
                tables = extract_tables_as_text(str(pdf_path), backend=backend)
                if tables:
                    all_tables.extend(tables)
                    backends_tried.append(f"{backend.value}({len(tables)})")
                    logger.info("%s found %d data tables", backend.value, len(tables))
            except Exception as exc:
                logger.debug("%s extraction failed: %s", backend.value, exc)

        if all_tables:
            logger.info("Multi-backend extraction: %s → total %d tables", ", ".join(backends_tried), len(all_tables))

        return all_tables

    # ── Stage 2a-direct: Direct DataFrame extraction (no LLM for numbers) ────

    def _direct_pdf_table_extraction(
        self,
        pdf_path: str | Path,
        pdf_content: "PDFContent",
        metadata: PaperMetadata,
        errors: list[str],
    ) -> tuple[list[SampleRow], list[str], set[int]]:
        """Extract sample rows directly from PDF tables WITHOUT passing
        numbers through an LLM.

        Two approaches tried in order:
        1. Multi-backend DataFrames (Docling/Camelot/pdfplumber)
        2. Text-based parsing of whitespace-separated tabular data from
           data-dense pages

        Numbers are read directly from DataFrames — zero LLM hallucination.

        Returns:
            (samples, backend_table_texts, extracted_pages) — samples from
            direct extraction, text versions of ALL extracted tables (for LLM
            fallback), and the set of page indices that yielded samples
            (so vision can skip them and focus on failed pages).
        """
        from .tabledetector import extract_tables_from_pdf, TableDetectorBackend
        from .pdf_reader import score_pages_for_data

        all_supp_tables: list[tuple[SupplementaryTable, str]] = []  # (table, label)
        # Cache text versions of all backend tables for potential LLM fallback
        backend_table_texts: list[str] = []
        # Track which page indices yielded successful extraction
        # (vision will skip these and focus on landscape/failed pages)
        extracted_pages: set[int] = set()

        # Approach 1: pdftext — Marker's fast text extraction layer (~2s).
        # Produces cleaner text than pdfplumber with intact table headers
        # (e.g., "Table 3. LA-ICP-MS results... (ppm)") enabling method detection.
        try:
            from pdftext.extraction import paginated_plain_text_output
            pdftext_pages = paginated_plain_text_output(str(pdf_path), sort=True)
            if pdftext_pages:
                # Score pages using pdftext output for better data page detection
                pdftext_content = PDFContent(
                    pages=pdftext_pages,
                    full_text="\n\n".join(pdftext_pages),
                    total_pages=len(pdftext_pages),
                )
                scored = score_pages_for_data(pdftext_content)
                # Only use pages with meaningful data scores to avoid
                # false positives from references or other non-table text
                data_page_indices = [idx for idx, sc in scored[:8] if sc >= 20]
                data_pages = [pdftext_pages[idx] for idx in data_page_indices]

                text_results = parse_text_tables_from_pages(
                    data_pages, page_indices=data_page_indices,
                )
                for tdf, pidx in text_results:
                    # Include surrounding text (table caption) in label for method detection
                    page_header = pdftext_pages[pidx][:300] if 0 <= pidx < len(pdftext_pages) else ""
                    tbl_label = f"pdftext_p{pidx + 1}|{page_header}"

                    # Skip summary/statistics tables (Mean, Median, quantile).
                    # Check both caption and first-column values.
                    _SUMMARY_KW = {"summary", "mean", "median", "average",
                                   "quantile", "std dev", "min-max", "range"}
                    caption_lower = page_header.lower()
                    first_col_vals = (
                        tdf.iloc[:, 0].astype(str).str.lower().tolist()
                        if len(tdf) > 0 else []
                    )
                    is_summary = (
                        "summary" in caption_lower
                        or any(
                            any(sk in v for sk in _SUMMARY_KW)
                            for v in first_col_vals[:5]
                        )
                    )
                    if is_summary:
                        logger.info("Skipping summary table on page %d", pidx + 1)
                        continue

                    supp = read_pdf_table(
                        tdf,
                        min_element_cols=3,
                        label=tbl_label,
                    )
                    if supp and supp.n_samples > 0:
                        all_supp_tables.append((supp, tbl_label))
                        extracted_pages.add(pidx)
                        logger.info(
                            "pdftext parsing p%d → %d samples, %d elements",
                            pidx + 1, supp.n_samples, len(supp.element_col_map),
                        )
                        # Also cache for LLM fallback
                        text_ver = dataframe_to_text(tdf, max_rows=200)
                        if text_ver and len(text_ver.strip()) > 20:
                            backend_table_texts.append(
                                f"[pdftext page {pidx + 1}]\n{text_ver}"
                            )
        except ImportError:
            logger.debug("pdftext not installed — skipping pdftext extraction")
        except Exception as exc:
            logger.debug("pdftext extraction failed: %s", exc)

        # Approach 2: Marker — layout-aware ML pipeline.
        # Handles rotated content, complex layouts, and multi-column tables
        # that pdftext misses. Runs with disable_ocr=True (digital PDFs).
        try:
            from .pdf_vision import extract_tables_with_marker, marker_tables_to_dataframes
            marker_tables = extract_tables_with_marker(pdf_path)
            if marker_tables:
                marker_dfs = marker_tables_to_dataframes(marker_tables)
                for tdf, pidx in marker_dfs:
                    tbl_label = f"marker_p{pidx + 1}"
                    supp = read_pdf_table(
                        tdf,
                        min_element_cols=3,
                        label=tbl_label,
                    )
                    if supp and supp.n_samples > 0:
                        all_supp_tables.append((supp, tbl_label))
                        extracted_pages.add(pidx)
                        logger.info(
                            "Marker → %d samples, %d elements",
                            supp.n_samples, len(supp.element_col_map),
                        )
                        text_ver = dataframe_to_text(tdf, max_rows=200)
                        if text_ver and len(text_ver.strip()) > 20:
                            backend_table_texts.append(
                                f"[marker table]\n{text_ver}"
                            )
        except Exception as exc:
            logger.debug("Marker extraction failed: %s", exc)

        # Approach 3: Text-based parsing of pdfplumber raw page text
        # Fallback when pdftext unavailable or backends fail to extract
        # structured tables but raw text has clear whitespace-aligned columns
        if not all_supp_tables and pdf_content and pdf_content.pages:
            try:
                scored = score_pages_for_data(pdf_content)
                data_page_indices = [idx for idx, _ in scored[:8]]
                data_pages = [pdf_content.pages[idx] for idx in data_page_indices]

                text_results = parse_text_tables_from_pages(
                    data_pages, page_indices=data_page_indices,
                )
                for tdf, pidx in text_results:
                    tbl_label = f"text_p{pidx + 1}"
                    supp = read_pdf_table(
                        tdf,
                        min_element_cols=3,
                        label=tbl_label,
                    )
                    if supp and supp.n_samples > 0:
                        all_supp_tables.append((supp, tbl_label))
                        extracted_pages.add(pidx)
                        logger.info(
                            "Text parsing → %d samples, %d elements",
                            supp.n_samples, len(supp.element_col_map),
                        )
            except Exception as exc:
                logger.debug("Text-based table parsing failed: %s", exc)

        # Approach 3: Multi-backend DataFrame extraction (Docling/Camelot/pdfplumber).
        # Always run all backends — different backends excel at different table
        # formats (landscape, borderless, grid). Cross-table dedup by
        # (sample_name, method) keeps the highest-quality version.
        pdftext_samples = sum(s.n_samples for s, _ in all_supp_tables)
        logger.info("pdftext found %d samples — running Docling/Camelot/pdfplumber backends", pdftext_samples)
        for backend in [TableDetectorBackend.DOCLING, TableDetectorBackend.CAMELOT, TableDetectorBackend.PDFPLUMBER]:
            try:
                tables, metrics = extract_tables_from_pdf(
                    str(pdf_path), backend=backend, force_backend=True,
                )
                for et in tables:
                    # Always cache text version for LLM fallback
                    text_ver = dataframe_to_text(et.df, max_rows=200)
                    if text_ver and len(text_ver.strip()) > 20:
                        backend_table_texts.append(
                            f"[{backend.value} page {et.page_number}]\n{text_ver}"
                        )
                    tbl_label = f"{backend.value}_p{et.page_number}_t{et.table_index}"
                    supp = read_pdf_table(
                        et.df,
                        min_element_cols=3,
                        label=tbl_label,
                    )
                    if supp and supp.n_samples > 0:
                        all_supp_tables.append((supp, tbl_label))
                        if et.page_number > 0:
                            extracted_pages.add(et.page_number - 1)  # 0-based
                        logger.info(
                            "Direct extraction: %s page %d → %d samples, %d elements",
                            backend.value, et.page_number, supp.n_samples,
                            len(supp.element_col_map),
                        )
            except Exception as exc:
                logger.debug("Direct %s extraction failed: %s", backend.value, exc)

        if not all_supp_tables:
            logger.info("Direct extraction: 0 usable tables, %d text tables cached for LLM fallback",
                         len(backend_table_texts))
            return [], backend_table_texts, extracted_pages

        # Detect analytical method per table from page context.
        # Papers often have separate tables for EPMA and LA-ICP-MS data.
        table_methods: list[str | None] = []
        for supp, tbl_label in all_supp_tables:
            method = _detect_method_from_table(supp, pdf_content, tbl_label)
            table_methods.append(method)
            if method:
                logger.info("Table '%s' detected as %s", tbl_label, method)

        # Merge all SupplementaryTable results and build SampleRows.
        # Allow same-name rows from DIFFERENT methods (e.g., EPMA table and
        # LA-ICP-MS table both have "K21-01-01" but with different values).
        # Deduplicate same name + same method across tables (multiple backends
        # may extract the same table), keeping the version with more elements.
        # Sort tables by element column count (descending) so highest-quality
        # tables populate first.
        table_order = sorted(
            range(len(all_supp_tables)),
            key=lambda i: len(all_supp_tables[i][0].element_col_map),
            reverse=True,
        )

        samples: list[SampleRow] = []
        # (name, method) → (index_in_samples, n_elements)
        seen_name_method: dict[tuple[str, str | None], tuple[int, int]] = {}

        for table_idx in table_order:
            supp, _tbl_label = all_supp_tables[table_idx]
            seen_within_table: set[str] = set()
            records = supp.to_element_records()
            table_method = table_methods[table_idx]
            for rec in records:
                name = rec.get("sample_name", "")
                # Skip duplicates within the same table
                if name and name in seen_within_table:
                    continue
                if name:
                    seen_within_table.add(name)

                row_data = _metadata_to_row_dict(metadata)
                # Override analytical_method with per-table method if detected
                if table_method:
                    row_data["analytical_method"] = table_method
                for k, v in rec.items():
                    if v is not None:
                        row_data[k] = v

                # Count non-null element values for quality comparison
                n_elements = sum(
                    1 for k, v in rec.items()
                    if k.endswith("_ppm") and v is not None
                )
                key = (name, table_method)
                if name and key in seen_name_method:
                    prev_idx, prev_n = seen_name_method[key]
                    if n_elements > prev_n:
                        # Replace with higher-quality version
                        samples[prev_idx] = _dict_to_sample_row(row_data)
                        seen_name_method[key] = (prev_idx, n_elements)
                    # else skip — existing version has more elements
                    continue

                if name:
                    seen_name_method[key] = (len(samples), n_elements)
                samples.append(_dict_to_sample_row(row_data))

        logger.info("Direct PDF table extraction: %d samples from %d tables (pages: %s)",
                     len(samples), len(all_supp_tables), sorted(extracted_pages))
        return samples, backend_table_texts, extracted_pages

    def _direct_pdf_table_extraction_relaxed(
        self,
        pdf_path: str | Path,
        pdf_content: "PDFContent",
        metadata: PaperMetadata,
        errors: list[str],
    ) -> list[SampleRow]:
        """Retry direct extraction with relaxed thresholds.

        Called by completeness gate when initial extraction yielded far fewer
        samples than expected. Uses min_element_cols=2 to catch tables with
        fewer recognized element headers (e.g., only major elements).
        """
        from .tabledetector import extract_tables_from_pdf, TableDetectorBackend

        all_supp_tables: list[SupplementaryTable] = []

        for backend in [TableDetectorBackend.DOCLING, TableDetectorBackend.CAMELOT, TableDetectorBackend.PDFPLUMBER]:
            try:
                tables, _ = extract_tables_from_pdf(
                    str(pdf_path), backend=backend, force_backend=True,
                )
                for et in tables:
                    supp = read_pdf_table(
                        et.df,
                        min_element_cols=2,  # relaxed from 3
                        label=f"relaxed_{backend.value}_p{et.page_number}",
                    )
                    if supp and supp.n_samples > 0:
                        all_supp_tables.append(supp)
            except Exception:
                pass

        if not all_supp_tables:
            return []

        samples: list[SampleRow] = []
        seen_names: set[str] = set()
        for supp in all_supp_tables:
            for rec in supp.to_element_records():
                name = rec.get("sample_name", "")
                if name and name in seen_names:
                    continue
                if name:
                    seen_names.add(name)
                row_data = _metadata_to_row_dict(metadata)
                for k, v in rec.items():
                    if v is not None:
                        row_data[k] = v
                samples.append(_dict_to_sample_row(row_data))
        return samples

    def _direct_pdf_table_with_hints(
        self,
        pdf_path: str | Path,
        pdf_content: "PDFContent",
        metadata: PaperMetadata,
        errors: list[str],
        hints: PDFExtractionHints,
    ) -> list[SampleRow]:
        """Retry direct PDF extraction using LLM-diagnosed hints.

        Applies adjustments from PDFExtractionHints:
        - min_element_cols: lower threshold for element column detection
        - try_transposed: transpose DataFrames before processing
        - header_row_offset: skip rows before the header
        """
        from .tabledetector import extract_tables_from_pdf, TableDetectorBackend

        all_supp_tables: list[SupplementaryTable] = []

        for backend in [TableDetectorBackend.DOCLING, TableDetectorBackend.CAMELOT, TableDetectorBackend.PDFPLUMBER]:
            try:
                tables, _ = extract_tables_from_pdf(
                    str(pdf_path), backend=backend, force_backend=True,
                )
                for et in tables:
                    df = et.df

                    # Apply header_row_offset hint
                    if hints.header_row_offset > 0 and len(df) > hints.header_row_offset:
                        df = df.iloc[hints.header_row_offset:].reset_index(drop=True)

                    # Apply transposition hint
                    if hints.try_transposed:
                        df = df.T.reset_index(drop=True)

                    supp = read_pdf_table(
                        df,
                        min_element_cols=hints.min_element_cols,
                        label=f"hints_{backend.value}_p{et.page_number}",
                        convert_units=(hints.unit_override == "wt%"),
                    )
                    if supp and supp.n_samples > 0:
                        all_supp_tables.append(supp)
            except Exception:
                pass

        if not all_supp_tables:
            return []

        samples: list[SampleRow] = []
        seen_names: set[str] = set()
        for supp in all_supp_tables:
            for rec in supp.to_element_records():
                name = rec.get("sample_name", "")
                if name and name in seen_names:
                    continue
                if name:
                    seen_names.add(name)
                row_data = _metadata_to_row_dict(metadata)
                for k, v in rec.items():
                    if v is not None:
                        row_data[k] = v
                samples.append(_dict_to_sample_row(row_data))
        return samples

    def _verify_extracted_values(
        self,
        samples: list[SampleRow],
        pdf_content: "PDFContent",
        cached_texts: list[str],
    ) -> str:
        """Spot-check extracted numerical values against source text.

        Takes up to 5 random non-null element values from extracted samples
        and checks if they appear verbatim in the PDF text or backend table
        text. Returns a verification note string, or "" if all checks pass.

        This is a lightweight O(n) string-matching check — no LLM calls.
        """
        import random

        # Build searchable corpus from PDF text + cached backend tables
        corpus_parts = []
        if pdf_content.full_text:
            corpus_parts.append(pdf_content.full_text)
        for tt in (pdf_content.tables_text or []):
            corpus_parts.append(tt)
        for ct in cached_texts:
            corpus_parts.append(ct)
        corpus = "\n".join(corpus_parts)
        if not corpus:
            return ""

        # Collect non-null element values from samples
        element_cols = [c for c in ALL_COLUMNS if c.endswith("_ppm")]
        value_checks: list[tuple[str, str, float]] = []  # (sample_name, element, value)
        for s in samples:
            for col in element_cols:
                val = getattr(s, col, None)
                if val is not None and val != 0.0:
                    value_checks.append((s.sample_name or "?", col, val))

        if not value_checks:
            return ""

        # Sample up to 10 values to check
        check_sample = random.sample(value_checks, min(10, len(value_checks)))

        found = 0
        not_found = 0
        for sample_name, element, value in check_sample:
            # Format value to check — try exact and rounded versions
            val_str = f"{value:.2f}" if abs(value) < 1000 else f"{value:.1f}"
            val_str_int = str(int(round(value))) if abs(value) >= 10 else None

            if val_str in corpus:
                found += 1
            elif val_str_int and val_str_int in corpus:
                found += 1
            elif f"{value:.4f}" in corpus:
                found += 1
            elif f"{value}" in corpus:
                found += 1
            else:
                not_found += 1

        total = found + not_found
        pct = found / total * 100 if total > 0 else 0

        if pct >= 70:
            return f"Value verification: {found}/{total} spot-checked values found in source ({pct:.0f}%)"
        elif pct >= 40:
            return f"Value verification WARNING: only {found}/{total} values found in source ({pct:.0f}%) — some may be hallucinated"
        else:
            return f"Value verification FAILED: only {found}/{total} values found in source ({pct:.0f}%) — likely hallucinated"

    def _merge_supplementary_and_pdf_samples(
        self,
        supp_samples: list[SampleRow],
        pdf_samples: list[SampleRow],
    ) -> list[SampleRow]:
        """Intelligently merge supplementary + PDF samples.

        Strategy:
        - Keep all unique samples by sample_name
        - For duplicates: supplementary values take priority (likely more complete)
        - PDF samples fill any gaps left by supplementary
        
        Returns merged list with no duplicates on sample_name.
        """
        if not pdf_samples:
            return supp_samples
        if not supp_samples:
            return pdf_samples

        # Index supplementary samples by name
        supp_by_name: dict[str, SampleRow] = {}
        for sample in supp_samples:
            name = sample.sample_name
            if name:
                supp_by_name[name] = sample

        # Merge PDF samples
        merged = list(supp_samples)  # Start with all supplementary
        
        for pdf_sample in pdf_samples:
            if pdf_sample.sample_name and pdf_sample.sample_name in supp_by_name:
                # Duplicate: merge by filling gaps in supplementary with PDF values
                supp_sample = supp_by_name[pdf_sample.sample_name]
                merged_data = supp_sample.model_dump()
                
                # Fill only None fields from PDF sample
                for field, pdf_value in pdf_sample.model_dump().items():
                    if merged_data.get(field) is None and pdf_value is not None:
                        merged_data[field] = pdf_value
                
                # Replace the supplementary sample with merged version
                supp_sample_idx = next(i for i, s in enumerate(merged) if s.sample_name == pdf_sample.sample_name)
                merged[supp_sample_idx] = _dict_to_sample_row(merged_data)
            else:
                # New sample from PDF: add it
                merged.append(pdf_sample)
        
        initial_count = len(supp_samples)
        final_count = len(merged)
        added_count = final_count - initial_count
        if added_count > 0:
            logger.info("Multi-source merge: supplementary %d + PDF %d → total %d samples (added %d new)",
                       initial_count, len(pdf_samples), final_count, added_count)
        
        return merged

    # ── Stage 2a: Python-based table extraction ───────────────────────────────

    def _python_table_extraction(
        self,
        supp: SupplementaryTable,
        metadata: PaperMetadata,
    ) -> list[SampleRow]:
        """Build SampleRow objects by combining metadata with numeric table data.

        Per-row values from the supplementary table (mineral, analytical_method,
        deposit_name, texture) override paper-level metadata when available.
        """
        records = supp.to_element_records()
        samples = []

        for rec in records:
            # Start from metadata (paper-level fields)
            row_data = _metadata_to_row_dict(metadata)
            # Per-row overrides from supplementary table take priority
            # (deposit_name, mineral, analytical_method, texture come from
            # to_element_records() when the table has those columns)
            for k, v in rec.items():
                if v is not None:
                    row_data[k] = v
            samples.append(_dict_to_sample_row(row_data))

        return samples

    # ── Stage 2b: PDF-only table extraction (no supplementary) ────────────────

    def _pdf_table_extraction(
        self,
        pdf_content: "PDFContent",
        metadata: PaperMetadata,
        errors: list[str],
        elements_measured: list[str] | None = None,
    ) -> list[SampleRow]:
        """Extract sample rows from tables embedded in the PDF text.

        Used when no supplementary spreadsheet is available. The LLM reads
        the paper text + embedded tables and returns structured sample data.
        Falls back to raw data-dense pages when pdfplumber can't extract
        structured tables.
        """
        paper_text = get_paper_text_for_llm(pdf_content, max_chars=20000)

        # When pdfplumber found no tables, extract raw text from
        # pages that look like they contain tabular numeric data
        data_pages = ""
        if not pdf_content.tables_text:
            data_pages = get_data_pages_text(pdf_content, max_chars=10000)

        system, user = build_pdf_table_extraction_prompt(
            paper_text=paper_text,
            pdf_tables=pdf_content.tables_text,
            data_pages_text=data_pages,
            elements_measured=elements_measured,
        )

        try:
            parsed = self.client.complete_json(
                system=system, user=user, max_tokens=32768,
            )
            if not isinstance(parsed, dict):
                raise ValueError("PDF table extraction did not return a dict")

            raw_samples = parsed.get("samples", [])
            extraction_notes = parsed.get("extraction_notes", "")
            if extraction_notes:
                logger.info("PDF extraction notes: %s", extraction_notes)

            if not raw_samples:
                return []

            samples = []
            for item in raw_samples:
                row_data = _metadata_to_row_dict(metadata)
                row_data.update({k: v for k, v in item.items() if v is not None})
                samples.append(_dict_to_sample_row(row_data))
            return samples

        except Exception as exc:
            errors.append(f"PDF table extraction failed: {exc}")
            logger.warning("PDF table extraction failed: %s", exc)
            return []

    # ── Stage 2b-c: Camelot-enhanced table extraction ───────────────────────────

    def _pdf_table_extraction_with_tables(
        self,
        pdf_content: "PDFContent",
        table_texts: list[str],
        metadata: PaperMetadata,
        errors: list[str],
        elements_measured: list[str] | None = None,
    ) -> list[SampleRow]:
        """Extract sample rows using tables detected by Camelot or other backends.

        Similar to _pdf_table_extraction but uses externally-detected tables
        instead of pdfplumber's tables_text.
        """
        paper_text = get_paper_text_for_llm(pdf_content, max_chars=20000)

        system, user = build_pdf_table_extraction_prompt(
            paper_text=paper_text,
            pdf_tables=table_texts,
            data_pages_text="",
            elements_measured=elements_measured,
        )

        try:
            parsed = self.client.complete_json(
                system=system, user=user, max_tokens=32768,
            )
            if not isinstance(parsed, dict):
                raise ValueError("PDF table extraction did not return a dict")

            raw_samples = parsed.get("samples", [])
            extraction_notes = parsed.get("extraction_notes", "")
            if extraction_notes:
                logger.info("Camelot extraction notes: %s", extraction_notes)

            if not raw_samples:
                return []

            samples = []
            for item in raw_samples:
                row_data = _metadata_to_row_dict(metadata)
                row_data.update({k: v for k, v in item.items() if v is not None})
                samples.append(_dict_to_sample_row(row_data))
            return samples

        except Exception as exc:
            errors.append(f"Camelot table extraction failed: {exc}")
            logger.warning("Camelot table extraction failed: %s", exc)
            return []

    # ── Stage 2b-v: Vision-based table extraction ──────────────────────────────

    def _vision_table_extraction(
        self,
        pdf_path: str | Path,
        pdf_content: PDFContent,
        metadata: PaperMetadata,
        errors: list[str],
        elements_measured: list[str] | None = None,
        expected_samples: int = 0,
        skip_pages: set[int] | None = None,
    ) -> list[SampleRow]:
        """Extract sample rows from PDF page images using LLM vision.

        Renders data-dense pages as images, sends them to the LLM along with
        a vision-specific prompt, and parses the structured JSON response.

        Pages are processed in chunks of 2 to avoid output token limits
        when papers have large tables spanning many pages.

        Handles edge cases:
        - Landscape/rotated pages (common for wide data tables)
        - Adaptive page count based on expected sample volume
        - Multi-page continuation tables
        - Transposed tables (elements as rows)
        - Skips pages already successfully extracted by text backends
        """
        try:
            page_images = render_data_pages(
                pdf_path=pdf_path,
                content=pdf_content,
                max_pages=8,
                dpi=150,
                expected_samples=expected_samples,
            )
        except Exception as exc:
            logger.warning("PDF rendering failed: %s", exc)
            errors.append(f"PDF rendering failed: {exc}")
            return []

        if not page_images:
            logger.info("No data pages rendered — skipping vision extraction")
            return []

        # Skip pages that text backends already successfully extracted.
        # Vision is expensive — only use it on pages where text failed
        # (landscape, garbled, image-embedded tables).
        if skip_pages:
            before = len(page_images)
            page_images = [
                pi for pi in page_images
                if pi["page_index"] not in skip_pages
            ]
            skipped = before - len(page_images)
            if skipped:
                logger.info(
                    "Vision: skipping %d pages already extracted by text backends (pages %s)",
                    skipped, sorted(skip_pages),
                )
            if not page_images:
                logger.info("All data pages already extracted by text — skipping vision")
                return []

        # Paper context: abstract + analytical methods for deposit/mineral context
        paper_context = get_paper_text_for_llm(pdf_content, max_chars=6000)

        # Process pages in chunks of 2 to stay within output token limits
        CHUNK_SIZE = 2
        all_samples: list[SampleRow] = []
        all_notes: list[str] = []

        for chunk_start in range(0, len(page_images), CHUNK_SIZE):
            chunk = page_images[chunk_start:chunk_start + CHUNK_SIZE]
            images = [
                {"image_bytes": pi["image_bytes"], "media_type": pi["media_type"]}
                for pi in chunk
            ]
            page_indices = [pi["page_index"] for pi in chunk]
            logger.info("Vision extraction: processing pages %s (%d/%d)",
                        page_indices, chunk_start + len(chunk), len(page_images))

            # Check if any page in this chunk is landscape
            chunk_has_landscape = any(pi.get("is_landscape", False) for pi in chunk)

            system, user = build_vision_table_extraction_prompt(
                n_pages=len(chunk),
                paper_context=paper_context,
                elements_measured=elements_measured,
                has_landscape=chunk_has_landscape,
            )

            try:
                logger.info("Vision using %s (%s)", self.vision_client.model, self.vision_client.provider)
                parsed = self.vision_client.complete_json(
                    system=system,
                    user=user,
                    max_tokens=32768,
                    images=images,
                )
                if not isinstance(parsed, dict):
                    continue

                raw_samples = parsed.get("samples", [])
                extraction_notes = parsed.get("extraction_notes", "")
                if extraction_notes:
                    all_notes.append(extraction_notes)
                    logger.info("Vision chunk notes: %s", extraction_notes)

                for item in raw_samples:
                    row_data = _metadata_to_row_dict(metadata)
                    row_data.update({k: v for k, v in item.items() if v is not None})
                    all_samples.append(_dict_to_sample_row(row_data))

            except Exception as exc:
                logger.warning("Vision chunk (pages %s) failed: %s", page_indices, exc)

        if all_notes:
            logger.info("Vision extraction notes: %s", "; ".join(all_notes))

        return all_samples

    # ── Stage 2c: LLM-assisted table extraction ───────────────────────────────

    def _llm_table_extraction(
        self,
        supp: SupplementaryTable,
        metadata: PaperMetadata,
        errors: list[str],
    ) -> list[SampleRow]:
        """Use LLM to parse/filter the supplementary table rows.

        Falls back to Python extraction on failure.
        """
        system, user = build_table_filter_prompt(
            table_text=supp.table_as_text(max_rows=150),
            column_mapping=supp.element_col_map,
        )
        try:
            parsed = self.client.complete_json(system=system, user=user, max_tokens=32768)
            if not isinstance(parsed, list):
                raise ValueError("LLM table extraction did not return a list")

            samples = []
            for item in parsed:
                row_data = _metadata_to_row_dict(metadata)
                row_data.update({k: v for k, v in item.items() if v is not None})
                samples.append(_dict_to_sample_row(row_data))
            return samples

        except Exception as exc:
            errors.append(f"LLM table extraction failed, falling back to Python: {exc}")
            logger.warning("LLM table extraction failed: %s", exc)
            return self._python_table_extraction(supp, metadata)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _detect_method_from_table(
    supp: "SupplementaryTable",
    pdf_content: "PDFContent | None" = None,
    table_label: str = "",
) -> str | None:
    """Detect analytical method from table context, column names, or page text.

    Returns normalized method name (e.g., 'EPMA', 'LA-ICPMS') or None.
    """
    from .knowledge_base import normalize_method

    # Method keywords ordered by specificity (longer/more specific first)
    _METHOD_PATTERNS = [
        ("LA-ICP-MS", "LA-ICPMS"),
        ("LA-ICPMS", "LA-ICPMS"),
        ("LA ICP-MS", "LA-ICPMS"),
        ("LAICPMS", "LA-ICPMS"),
        ("laser ablation", "LA-ICPMS"),
        ("EPMA", "EPMA"),
        ("electron microprobe", "EPMA"),
        ("EMP ", "EPMA"),
        ("WDS", "EPMA"),
        ("ICP-MS", "ICP-MS"),
        ("ICP-OES", "ICP-OES"),
        ("XRF", "XRF"),
        ("SIMS", "SIMS"),
        ("PIXE", "PIXE"),
    ]

    # Build search text from table notes, label, and page context
    search_text = " ".join(supp.notes) + " " + table_label

    if pdf_content and pdf_content.pages:
        # Extract page number from label like "text_p7", "pdftext_p7|...", "camelot_p6_t0"
        page_match = re.search(r'_p(\d+)', table_label)
        if page_match:
            page_idx = int(page_match.group(1)) - 1  # labels are 1-based
            if 0 <= page_idx < len(pdf_content.pages):
                page_text = pdf_content.pages[page_idx][:500]
                search_text += " " + page_text
                # For continuation tables ("Cont."), check previous pages
                # for the original table caption with method info
                is_continuation = "cont" in page_text[:200].lower()
                if page_idx > 0:
                    prev_page = pdf_content.pages[page_idx - 1]
                    if is_continuation:
                        # Full previous page — caption may be anywhere
                        search_text += " " + prev_page
                        # Also check 2 pages back for multi-page tables
                        if page_idx > 1:
                            search_text += " " + pdf_content.pages[page_idx - 2][-500:]
                    else:
                        search_text += " " + prev_page[-300:]

    # Also include column names
    col_names = " ".join(str(c) for c in supp.data_df.columns)
    all_text = f"{search_text} {col_names}".upper()

    for pattern, method in _METHOD_PATTERNS:
        if pattern.upper() in all_text:
            return normalize_method(method)

    # Heuristic: if columns include "Total" and values sum near 100, it's EPMA (wt%)
    if "TOTAL" in col_names.upper() or "WT%" in all_text or "WT.%" in all_text:
        return normalize_method("EPMA")

    return None


def _metadata_to_row_dict(meta: PaperMetadata) -> dict:
    """Convert PaperMetadata → dict with SampleRow field names."""
    d = meta.model_dump()
    # Some fields need remapping (PaperMetadata uses shorter names)
    return d


def _dict_to_sample_row(data: dict) -> SampleRow:
    """Build a SampleRow from a merged data dict, ignoring unknown keys."""
    known = set(SampleRow.model_fields.keys())
    filtered = {k: v for k, v in data.items() if k in known}
    return SampleRow(**filtered)


def _normalise_metadata_keys(d: dict) -> dict:
    """Normalise common LLM key variations to match PaperMetadata field names."""
    renames = {
        "lab":                      "laboratory_location",
        "laboratory":               "laboratory_location",
        "lab_location":             "laboratory_location",
        "conditions":               "operating_conditions",
        "analytical_conditions":    "operating_conditions",
        "standard":                 "standards_used",
        "reference_standard":       "standards_used",
        "calibration_standard":     "standards_used",
        "method":                   "analytical_method",
        "instrument":               "instrument_type_model",
        "instrument_model":         "instrument_type_model",
        "year":                     "publication_date",
        "pub_year":                 "publication_date",
        "citation":                 "sample_source",
        "full_citation":            "sample_source",
        "reference":                "sample_source",
        "country_code":             "country",
        "iso_country":              "country",
        "mineral_analyzed":         "mineral",
        "minerals":                 "mineral",
        "commodity":                "all_commodities",
        "commodities":              "all_commodities",
        "deposit_classification":   "deposit_group",
        "classification":           "deposit_group",
        "ore_type":                 "deposit_type",
    }
    result = {}
    for k, v in d.items():
        canonical = renames.get(k.lower().replace(" ", "_"), k)
        result[canonical] = v
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: run multiple LLMs and return all results
# ──────────────────────────────────────────────────────────────────────────────

def run_all_models(
    pdf_path: str | Path,
    supplementary_paths: str | Path | list[str | Path],
    clients: list[LLMClient],
    this_paper_deposit: Optional[str] = None,
    use_tool_calling: bool = True,
    use_llm_table_filter: bool = False,
    use_self_correction: bool = True,
    use_vision: bool = True,
    vision_client: LLMClient | None = None,
    verbose: bool = False,
    table_detector_backend: TableDetectorBackend = TableDetectorBackend.AUTO,
) -> dict[str, ExtractionResult]:
    """Run extraction with multiple LLM clients in sequence.

    Returns:
        Dict mapping "provider/model" → ExtractionResult.
    """
    if isinstance(supplementary_paths, (str, Path)):
        supplementary_paths = [supplementary_paths]

    results: dict[str, ExtractionResult] = {}

    for client in clients:
        key = f"{client.provider}/{client.model}"
        logger.info("Running extraction with %s", key)
        try:
            pipeline = ExtractionPipeline(
                llm_client=client,
                use_tool_calling=use_tool_calling,
                use_llm_table_filter=use_llm_table_filter,
                use_self_correction=use_self_correction,
                use_vision=use_vision,
                vision_client=vision_client,
                verbose=verbose,
                table_detector_backend=table_detector_backend,
            )
            result = pipeline.run(
                pdf_path=pdf_path,
                supplementary_paths=supplementary_paths,
                this_paper_deposit=this_paper_deposit,
            )
            results[key] = result
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", key, exc)
            results[key] = ExtractionResult(
                metadata=PaperMetadata(),
                samples=[],
                llm_model=client.model,
                llm_provider=client.provider,
                pdf_path=str(pdf_path),
                supplementary_paths=[str(p) for p in supplementary_paths],
                errors=[str(exc)],
            )

    return results
