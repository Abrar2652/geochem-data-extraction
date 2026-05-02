"""
deposit_classifier.py — Two-pass LLM deposit classification using CMMI 189-type taxonomy.

Approach (superior to sri-ta2's single-pass):
  Pass 1: Extract geological evidence from paper (deposit characteristics, host rocks,
           ore minerals, alteration, tectonic setting, USGS model indicators)
  Pass 2: Score all 189 CMMI deposit types against extracted evidence with chain-of-thought
           reasoning, returning top-5 ranked candidates with confidence and justification.

Why two passes?
  - Separating evidence extraction from classification reduces hallucination
  - The reasoning chain can be verified by humans
  - Structured evidence enables both LLM scoring AND deterministic rule matching
  - If LLM classification is uncertain, the extracted evidence still has value

Source: DARPA CRITICALMAAS sri-ta2 / Hofstra et al. 2021 CMMI taxonomy (189 types)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DepositEvidence:
    """Geological evidence extracted from a paper for deposit classification."""
    host_rocks: list[str] = field(default_factory=list)
    ore_minerals: list[str] = field(default_factory=list)
    gangue_minerals: list[str] = field(default_factory=list)
    alteration_types: list[str] = field(default_factory=list)
    tectonic_setting: Optional[str] = None
    geological_age: Optional[str] = None
    ore_textures: list[str] = field(default_factory=list)
    deposit_morphology: Optional[str] = None   # vein, massive, disseminated, etc.
    associated_igneous_rocks: list[str] = field(default_factory=list)
    metal_associations: list[str] = field(default_factory=list)  # e.g., Cu-Au, Zn-Pb-Ag
    depth_of_formation: Optional[str] = None   # epithermal, mesothermal, hypothermal
    fluid_characteristics: Optional[str] = None
    usgs_model_mentioned: Optional[str] = None  # If paper cites a USGS model number
    author_classification: Optional[str] = None  # What the authors call it
    key_indicators: list[str] = field(default_factory=list)  # Other distinctive features


@dataclass
class DepositClassification:
    """Ranked deposit type classification result."""
    deposit_type: str
    deposit_environment: str
    deposit_group: str
    minmod_id: str
    confidence: float           # 0.0-1.0
    reasoning: str              # Why this type was chosen
    evidence_match: list[str]   # Which evidence items matched


@dataclass
class ClassificationResult:
    """Full classification output for one paper."""
    evidence: DepositEvidence = field(default_factory=DepositEvidence)
    classifications: list[DepositClassification] = field(default_factory=list)
    top_type: Optional[str] = None
    top_environment: Optional[str] = None
    top_group: Optional[str] = None
    top_confidence: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Load CMMI taxonomy
# ──────────────────────────────────────────────────────────────────────────────

_CMMI_CACHE: list[dict] = []


def _load_cmmi() -> list[dict]:
    """Load the 189-type CMMI taxonomy with descriptions."""
    global _CMMI_CACHE
    if _CMMI_CACHE:
        return _CMMI_CACHE
    try:
        import pandas as pd
        base = Path(__file__).parent
        dt_path = base / "sri-ta2" / "minmod" / "deposit_type.csv"
        desc_path = base / "sri-ta2" / "taxonomy" / "cmmi_options_full_description_with_number.csv"

        dt = pd.read_csv(dt_path)
        descriptions = {}
        if desc_path.exists():
            desc_df = pd.read_csv(desc_path)
            for _, r in desc_df.iterrows():
                descriptions[r["Deposit type"]] = str(r["Description"])

        for _, row in dt.iterrows():
            _CMMI_CACHE.append({
                "name": row["Deposit type"],
                "environment": row["Deposit environment"],
                "group": row["Deposit group"],
                "minmod_id": row["Minmod ID"],
                "description": descriptions.get(row["Deposit type"], ""),
            })
    except Exception as e:
        logger.warning("Failed to load CMMI taxonomy: %s", e)
    return _CMMI_CACHE


def _build_options_text() -> str:
    """Build the compact 189-type options listing for the LLM prompt."""
    cmmi = _load_cmmi()
    lines = []
    for entry in cmmi:
        desc = entry["description"][:200] if entry["description"] else ""
        lines.append(
            f'{entry["minmod_id"]}: {entry["name"]} '
            f'[{entry["environment"]} > {entry["group"]}] — {desc}'
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1: Evidence extraction
# ──────────────────────────────────────────────────────────────────────────────

EVIDENCE_SYSTEM_PROMPT = """\
You are an expert economic geologist. Extract deposit characterization evidence \
from a research paper to enable mineral deposit classification per Hofstra et al. 2021.

Focus on geological features that distinguish deposit types — NOT the analytical \
chemistry data. Read the geological setting, introduction, and discussion sections carefully.

Return ONLY valid JSON. Use null for uncertain/unstated fields. Use [] for empty lists."""

EVIDENCE_USER_PROMPT = """\
## PAPER TEXT
{paper_text}

## EXTRACTED METADATA (from prior pipeline stages)
Deposit name: {deposit_name}
Minerals analyzed: {minerals}
Analytical method: {method}
Commodities: {commodities}

## YOUR TASK
Extract the geological evidence needed to classify this deposit per CMMI/Hofstra 2021.
Return a JSON object with EXACTLY these fields:

{{
  "host_rocks": ["granite", "limestone"],
  "ore_minerals": ["pyrite", "sphalerite", "galena"],
  "gangue_minerals": ["quartz", "calcite"],
  "alteration_types": ["propylitic", "sericitic"],
  "tectonic_setting": "continental arc" or null,
  "geological_age": "Cretaceous" or null,
  "ore_textures": ["disseminated", "vein-hosted", "massive"],
  "deposit_morphology": "vein" or "massive sulfide" or "disseminated" or null,
  "associated_igneous_rocks": ["granite", "diorite"],
  "metal_associations": ["Cu-Au", "Zn-Pb-Ag"],
  "depth_of_formation": "epithermal" or "mesothermal" or "hypothermal" or null,
  "fluid_characteristics": "low salinity CO2-bearing" or null,
  "usgs_model_mentioned": "17" or null,
  "author_classification": "porphyry Cu-Au" or "MVT" or null,
  "key_indicators": ["fluid inclusion data", "sulfur isotopes"]
}}

## CRITICAL
- Extract what the paper STATES, do not infer deposit type yet
- If the authors explicitly name a deposit type or USGS model, capture it in author_classification/usgs_model_mentioned
- ore_minerals = minerals that carry economic metals (pyrite, chalcopyrite, sphalerite, galena, etc.)
- gangue_minerals = non-economic minerals in the ore (quartz, calcite, barite, fluorite, etc.)
- metal_associations = which metals occur together (e.g., "Cu-Au", "Zn-Pb-Ag", "Ni-Cu-PGE")

Return ONLY the JSON object."""


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2: Classification
# ──────────────────────────────────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = """\
You are an expert mineral deposit classifier using the Hofstra et al. 2021 CMMI \
classification scheme (189 deposit types). You will be given geological evidence \
extracted from a research paper, plus the full list of 189 deposit types with descriptions.

Your task: rank the TOP 5 most likely deposit types with confidence scores and reasoning.

Classification principles:
1. Match evidence to deposit type DESCRIPTIONS, not just names
2. Consider the full geological context — host rocks + alteration + metals + tectonic setting
3. If the author explicitly names a deposit type/USGS model, weight it heavily but verify
4. Confidence should reflect how well the evidence fits — partial matches get lower scores
5. Multiple plausible classifications are expected for ambiguous deposits"""

CLASSIFY_USER_PROMPT = """\
## GEOLOGICAL EVIDENCE (extracted from the paper)
{evidence_json}

## ALL 189 CMMI DEPOSIT TYPES (Hofstra et al. 2021)
{options_text}

## YOUR TASK
Based on the geological evidence above, rank the TOP 5 most likely deposit types.

Return a JSON object with:
{{
  "classifications": [
    {{
      "minmod_id": "Q380",
      "deposit_type": "Porphyry copper ± gold",
      "confidence": 0.85,
      "reasoning": "Paper describes Cu-Au mineralization in a porphyry system with propylitic alteration, quartz stockwork veins, in a continental arc setting. Host rocks are diorite-granodiorite. All consistent with porphyry Cu-Au.",
      "evidence_match": ["Cu-Au metals", "propylitic alteration", "continental arc", "porphyry host"]
    }},
    {{
      "minmod_id": "Q376",
      "deposit_type": "Skarn copper",
      "confidence": 0.10,
      "reasoning": "Some carbonate host rock present, but alteration and texture don't match typical skarn.",
      "evidence_match": ["carbonate host"]
    }},
    ... (3 more)
  ]
}}

## SCORING GUIDELINES
- 0.8-1.0: Strong match — multiple lines of evidence converge on this type
- 0.5-0.8: Good match — most evidence fits but some ambiguity
- 0.2-0.5: Possible — some evidence matches but significant uncertainty
- 0.0-0.2: Unlikely — only marginal evidence
- Top 5 confidence scores should sum to roughly 1.0 (they represent a probability distribution)

Return ONLY the JSON object. No explanation outside the JSON."""


# ──────────────────────────────────────────────────────────────────────────────
# Main classifier
# ──────────────────────────────────────────────────────────────────────────────

def classify_deposit(
    client,
    paper_text: str,
    deposit_name: Optional[str] = None,
    minerals: Optional[list[str]] = None,
    analytical_method: Optional[str] = None,
    commodities: Optional[str] = None,
) -> ClassificationResult:
    """Two-pass deposit classification using LLM + CMMI 189-type taxonomy.

    Args:
        client: LLMClient instance (Claude, OpenAI, etc.)
        paper_text: Full or prioritized paper text
        deposit_name: Deposit name from metadata extraction
        minerals: List of analyzed minerals
        analytical_method: Method used (LA-ICPMS, EPMA, etc.)
        commodities: Commodity string (e.g., "Cu, Au")

    Returns:
        ClassificationResult with evidence + ranked classifications
    """
    result = ClassificationResult(evidence=DepositEvidence())

    # ── Pass 1: Extract geological evidence ──────────────────────────────
    logger.info("Deposit classification Pass 1: extracting geological evidence")
    try:
        evidence_prompt = EVIDENCE_USER_PROMPT.format(
            paper_text=paper_text[:15000],
            deposit_name=deposit_name or "unknown",
            minerals=", ".join(minerals) if minerals else "unknown",
            method=analytical_method or "unknown",
            commodities=commodities or "unknown",
        )

        evidence_raw = client.complete_json(
            system=EVIDENCE_SYSTEM_PROMPT,
            user=evidence_prompt,
            max_tokens=2048,
        )

        if isinstance(evidence_raw, dict):
            result.evidence = DepositEvidence(
                host_rocks=evidence_raw.get("host_rocks") or [],
                ore_minerals=evidence_raw.get("ore_minerals") or [],
                gangue_minerals=evidence_raw.get("gangue_minerals") or [],
                alteration_types=evidence_raw.get("alteration_types") or [],
                tectonic_setting=evidence_raw.get("tectonic_setting"),
                geological_age=evidence_raw.get("geological_age"),
                ore_textures=evidence_raw.get("ore_textures") or [],
                deposit_morphology=evidence_raw.get("deposit_morphology"),
                associated_igneous_rocks=evidence_raw.get("associated_igneous_rocks") or [],
                metal_associations=evidence_raw.get("metal_associations") or [],
                depth_of_formation=evidence_raw.get("depth_of_formation"),
                fluid_characteristics=evidence_raw.get("fluid_characteristics"),
                usgs_model_mentioned=evidence_raw.get("usgs_model_mentioned"),
                author_classification=evidence_raw.get("author_classification"),
                key_indicators=evidence_raw.get("key_indicators") or [],
            )
            logger.info(
                "Evidence extracted: %d ore minerals, %d host rocks, author_class=%s",
                len(result.evidence.ore_minerals),
                len(result.evidence.host_rocks),
                result.evidence.author_classification,
            )
    except Exception as e:
        logger.warning("Pass 1 (evidence extraction) failed: %s", e)

    # ── Pass 2: Classify against 189 types ───────────────────────────────
    logger.info("Deposit classification Pass 2: scoring against 189 CMMI types")
    try:
        # Build evidence JSON for the classifier
        evidence_dict = {
            "host_rocks": result.evidence.host_rocks,
            "ore_minerals": result.evidence.ore_minerals,
            "gangue_minerals": result.evidence.gangue_minerals,
            "alteration_types": result.evidence.alteration_types,
            "tectonic_setting": result.evidence.tectonic_setting,
            "geological_age": result.evidence.geological_age,
            "ore_textures": result.evidence.ore_textures,
            "deposit_morphology": result.evidence.deposit_morphology,
            "associated_igneous_rocks": result.evidence.associated_igneous_rocks,
            "metal_associations": result.evidence.metal_associations,
            "depth_of_formation": result.evidence.depth_of_formation,
            "fluid_characteristics": result.evidence.fluid_characteristics,
            "usgs_model_mentioned": result.evidence.usgs_model_mentioned,
            "author_classification": result.evidence.author_classification,
            "key_indicators": result.evidence.key_indicators,
            "deposit_name": deposit_name,
            "commodities": commodities,
        }

        classify_prompt = CLASSIFY_USER_PROMPT.format(
            evidence_json=json.dumps(evidence_dict, indent=2),
            options_text=_build_options_text(),
        )

        classify_raw = client.complete_json(
            system=CLASSIFY_SYSTEM_PROMPT,
            user=classify_prompt,
            max_tokens=4096,
        )

        if isinstance(classify_raw, dict):
            raw_classes = classify_raw.get("classifications", [])
            cmmi_lookup = {e["minmod_id"]: e for e in _load_cmmi()}

            for rc in raw_classes[:5]:
                mid = rc.get("minmod_id", "")
                cmmi_entry = cmmi_lookup.get(mid, {})

                classification = DepositClassification(
                    deposit_type=rc.get("deposit_type", cmmi_entry.get("name", "")),
                    deposit_environment=cmmi_entry.get("environment", ""),
                    deposit_group=cmmi_entry.get("group", ""),
                    minmod_id=mid,
                    confidence=float(rc.get("confidence", 0)),
                    reasoning=rc.get("reasoning", ""),
                    evidence_match=rc.get("evidence_match", []),
                )
                result.classifications.append(classification)

            if result.classifications:
                top = result.classifications[0]
                result.top_type = top.deposit_type
                result.top_environment = top.deposit_environment
                result.top_group = top.deposit_group
                result.top_confidence = top.confidence

            logger.info(
                "Classification complete: top='%s' (%.2f), %d candidates",
                result.top_type, result.top_confidence, len(result.classifications),
            )

    except Exception as e:
        logger.warning("Pass 2 (classification) failed: %s", e)

    return result


def apply_classification_to_metadata(
    result: ClassificationResult,
    metadata,
) -> list[str]:
    """Apply classification results to PaperMetadata fields.

    Updates metadata in-place. Returns list of notes.
    """
    notes = []
    if not result.classifications:
        return notes

    top = result.classifications[0]

    # Only override if confidence is reasonable
    if top.confidence >= 0.3:
        # Preserve original before overriding
        if metadata.deposit_type and metadata.deposit_type != top.deposit_type:
            metadata.deposit_type_original = metadata.deposit_type

        metadata.deposit_type = top.deposit_type
        metadata.deposit_environment = top.deposit_environment
        metadata.deposit_group = top.deposit_group
        metadata.deposit_classification_source = "Hofstra et al. 2021 (CMMI 189-type, LLM classified)"
        metadata.deposit_type_confidence = top.confidence
        metadata.deposit_type_reasoning = top.reasoning

        # Format alternatives
        alts = [
            f"{c.deposit_type} ({c.confidence:.2f})"
            for c in result.classifications[1:5]
        ]
        metadata.deposit_type_alternatives = " | ".join(alts) if alts else None

        notes.append(
            f"Deposit classified: '{top.deposit_type}' "
            f"({top.deposit_environment} > {top.deposit_group}) "
            f"conf={top.confidence:.2f}"
        )
    else:
        notes.append(
            f"Deposit classification uncertain: top='{top.deposit_type}' "
            f"conf={top.confidence:.2f} — below 0.3 threshold"
        )

    # Capture author's classification if available
    if result.evidence.author_classification:
        if not metadata.deposit_type_original:
            metadata.deposit_type_original = result.evidence.author_classification
        notes.append(f"Author classifies as: '{result.evidence.author_classification}'")

    return notes
