"""
schema.py - Ground truth schema definition for geochemical data extraction.

Defines the 210-column standardized schema, pydantic models, and element mappings
that are constant across all papers being benchmarked.

Below-detection-limit convention:
  -99999 = element WAS analysed but concentration was below detection limit
  blank  = element was NOT analysed / NOT reported (no measurement attempted)
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

# Sentinel value: below detection limit (measured but too low to quantify)
BELOW_DETECTION_SENTINEL = -99999.0

# ──────────────────────────────────────────────────────────────────────────────
# All 73 elements tracked in the schema (alphabetical symbol order)
# ──────────────────────────────────────────────────────────────────────────────
ELEMENT_SYMBOLS = [
    "ag", "al", "as", "au", "b",  "ba", "be", "bi", "br", "ca",
    "cd", "ce", "cl", "co", "cr", "cs", "cu", "dy", "er", "eu",
    "f",  "fe", "ga", "gd", "ge", "hf", "hg", "ho", "in", "ir",
    "k",  "la", "li", "lu", "mg", "mn", "mo", "na", "nb", "nd",
    "ni", "os", "p",  "pb", "pd", "pr", "pt", "rb", "re", "rh",
    "ru", "s",  "sb", "sc", "se", "si", "sm", "sn", "sr", "ta",
    "tb", "te", "th", "ti", "tl", "tm", "u",  "v",  "w",  "y",
    "yb", "zn", "zr",
]

# ──────────────────────────────────────────────────────────────────────────────
# Metadata columns (non-numerical, paper/sample level)
# ──────────────────────────────────────────────────────────────────────────────
METADATA_COLUMNS = [
    "Uploaded",
    "deposit_name",
    "deposit_local_id",
    "deposit_environment",
    "deposit_group",
    "deposit_type",
    "deposit_classification_source",
    "deposit_type_original",
    "deposit_type_confidence",
    "deposit_type_reasoning",
    "deposit_type_alternatives",
    "primary_commodities",
    "secondary_commodities",
    "all_commodities",
    "deposit_source",
    "sample_uid",
    "sample_name",
    "sample_local_id",
    "analysis_id",
    "feature_type",
    "feature_name",
    "feature_uid",
    "top_depth_m",
    "bottom_depth_m",
    "sample_deposit_relation",
    "sample_type",
    "sampling_method",
    "material_class",
    "material_class_comments",
    "province",
    "strat_unit_name",
    "strat_unit_uid",
    "strat_grouping",
    "earth_material_group",
    "earth_material_qualifier",
    "earth_material",
    "metamorphic_grade",
    "mineral",
    "paragenetic_stage",
    "mode_of_occurrence",
    "texture",
    "color",
    "alteration",
    "sample_description",
    "associated_minerals",
    "sample_preparation",
    "analytical_method",
    "instrument_type_model",
    "laboratory_location/if reported",
    "operating_conditions/if reported",
    "standards_used/if reported",
]

# ──────────────────────────────────────────────────────────────────────────────
# Element columns: each element has _ppm + _detection_limit + _original_value + _original_unit
# _ppm: value converted to ppm (canonical)
# _detection_limit: detection limit if reported
# _original_value: value as originally reported in the paper (before conversion)
# _original_unit: unit as reported (e.g., "wt%", "ppm", "ppb")
# ──────────────────────────────────────────────────────────────────────────────
ELEMENT_COLUMNS: list[str] = []
for _sym in ELEMENT_SYMBOLS:
    ELEMENT_COLUMNS.append(f"{_sym}_ppm")
    ELEMENT_COLUMNS.append(f"{_sym}_detection_limit")
    ELEMENT_COLUMNS.append(f"{_sym}_original_value")
    ELEMENT_COLUMNS.append(f"{_sym}_original_unit")

# ──────────────────────────────────────────────────────────────────────────────
# Provenance / location columns
# ──────────────────────────────────────────────────────────────────────────────
PROVENANCE_COLUMNS = [
    "analysis_datetime",
    "publication_date",
    "sample_source",
    "submitter",
    "country",
    "state",
    "deposit_longitude_wgs84",
    "deposit_latitude_wgs84",
    "sample_longitude_wgs84",
    "sample_latitude_wgs84",
    "sample_easting",
    "sample_northing",
    "sample_utm_zone",
    "sample_location_description",
    "location_source",
    "location_accuracy",
    "comments",
    "last_update",
    "extraction_backend",
    "extraction_confidence",
    "backend_agreement",
    "data_source_tag",
]

# Full ordered column list (matches ground truth Excel exactly)
ALL_COLUMNS = METADATA_COLUMNS + ELEMENT_COLUMNS + PROVENANCE_COLUMNS


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class PaperMetadata(BaseModel):
    """Paper-level metadata extracted from the full PDF text.

    Applies uniformly to every sample row in that paper.
    Field descriptions aligned with CMMI Mineral Gx database schema v0.
    """
    deposit_name:               Optional[str]   = Field(None, description="Name of mineral deposit to which the sample is associated")
    deposit_local_id:           Optional[str]   = Field(None, description="Identifier for the deposit in local/national database")
    deposit_environment:        Optional[str]   = Field(None, description="Geological environment in which the deposit formed, per Hofstra et al. 2021 classification")
    deposit_group:              Optional[str]   = Field(None, description="Restricted grouping of deposits with similar characteristics, per Hofstra et al. 2021")
    deposit_type:               Optional[str]   = Field(None, description="Detailed deposit classification: subgroup + group + principal commodity(s), per Hofstra et al. 2021")
    deposit_classification_source: Optional[str] = Field(None, description="Classification scheme used: 'Hofstra et al. 2021' or paper's original if different")
    deposit_type_original:      Optional[str]   = Field(None, description="Deposit type as originally stated by the paper's authors (may differ from Hofstra 2021)")
    deposit_type_confidence:    Optional[float] = Field(None, description="0.0-1.0 confidence score for deposit_type classification")
    deposit_type_reasoning:     Optional[str]   = Field(None, description="Why this deposit type was chosen (keyword matches, description overlap)")
    deposit_type_alternatives:  Optional[str]   = Field(None, description="Top alternative deposit type candidates with scores, pipe-separated")
    primary_commodities:        Optional[str]   = Field(None, description="Aggregated comma-separated list of primary commodity codes")
    secondary_commodities:      Optional[str]   = Field(None, description="Aggregated comma-separated list of secondary commodity codes")
    all_commodities:            Optional[str]   = Field(None, description="All commodity codes: 'Primary commodities (secondary commodities)'")
    deposit_source:             Optional[str]   = Field(None, description="Name/description of the organisation or database from which deposit info was sourced")

    feature_type:               Optional[str]   = Field(None, description="Sampling feature type: borehole, field site (underground/surface), measured section, or unknown")
    sample_deposit_relation:    Optional[str]   = Field(None, description="Sample's relation to the mineral deposit: high grade ore, low grade ore, gangue, alteration halo, host rock, unknown")
    sample_type:                Optional[str]   = Field(None, description="Type or form of the sample: half core, outcrop specimen, drill chips, float specimen, unknown")
    sampling_method:            Optional[str]   = Field(None, description="Method used to collect the sample: core drilling, RC drilling, outcrop sampling, channel sampling, unknown")
    material_class:             Optional[str]   = Field(None, description="Broad categorisation of sample material type: rock, regolith, sediment")
    material_class_comments:    Optional[str]   = Field(None, description="Additional comments about the sample material class/type")
    earth_material_group:       Optional[str]   = Field(None, description="Dominant lithological grouping: acid intrusive, sedimentary siliciclastic, sedimentary carbonate, mineralisation, unknown")
    earth_material_qualifier:   Optional[str]   = Field(None, description="Qualifying term(s) before lithology name: graphitic, fine grained, foliated")
    earth_material:             Optional[str]   = Field(None, description="Aggregated lithological names: sandstone, siltstone, granite")
    mineral:                    Optional[str]   = Field(None, description="Target mineral of the mineral chemistry analysis, matching IMA Mineral List")
    paragenetic_stage:          Optional[str]   = Field(None, description="Paragenetic stage of the sample under analysis, as described in the paper")
    mode_of_occurrence:         Optional[str]   = Field(None, description="Mode of occurrence: fracture fill, vein, replacement, xenolith")
    texture:                    Optional[str]   = Field(None, description="Mineral textures: foliated, vuggy, colloform, massive")
    associated_minerals:        Optional[str]   = Field(None, description="Significant minerals in association with the target mineral, comma-separated, tied to IMA List")
    sample_preparation:         Optional[str]   = Field(None, description="Description of sample preparation: thin section, mount, crushing, milling")

    analytical_method:          Optional[str]   = Field(None, description="Analytical method for mineral chemistry: LA-ICPMS, EMPA, ICP-MS, SEM-EDS, SIMS")
    instrument_type_model:      Optional[str]   = Field(None, description="Instrument type and model exactly as stated in paper")
    laboratory_location:        Optional[str]   = Field(None, description="Laboratory location formatted CITY / STATE / COUNTRY, exactly as stated")
    operating_conditions:       Optional[str]   = Field(None, description="Operating conditions for the analysis exactly as stated")
    standards_used:             Optional[str]   = Field(None, description="Any reported standards for the mineral chemistry analysis")

    publication_date:           Optional[int]   = Field(None, description="Year of publication as integer (YYYY)")
    sample_source:              Optional[str]   = Field(None, description="Citation of published/unpublished article or report for the sample analysis")
    country:                    Optional[str]   = Field(None, description="Three letter country code (ISO 3166): AUS, CAN, USA, CHN")
    state:                      Optional[str]   = Field(None, description="Letter code for state/province (ISO 3166-2): NSW, AB")
    deposit_longitude_wgs84:    Optional[float] = Field(None, description="Deposit location longitude in WGS84 decimal degrees")
    deposit_latitude_wgs84:     Optional[float] = Field(None, description="Deposit location latitude in WGS84 decimal degrees")


class SampleRow(BaseModel):
    model_config = {"extra": "allow"}  # Accept _original_value/_original_unit fields dynamically
    """One analytical measurement row — metadata + per-element values."""

    # --- sample identity (three-tier per USGS protocol) ---
    sample_name:     Optional[str]   = None
    sample_local_id: Optional[str]   = None
    analysis_id:     Optional[str]   = None

    # --- inherited from PaperMetadata (flattened) ---
    deposit_name:               Optional[str]   = None
    deposit_local_id:           Optional[str]   = None
    deposit_environment:        Optional[str]   = None
    deposit_group:              Optional[str]   = None
    deposit_type:               Optional[str]   = None
    deposit_classification_source: Optional[str] = None
    deposit_type_original:      Optional[str]   = None
    deposit_type_confidence:    Optional[float] = None
    deposit_type_reasoning:     Optional[str]   = None
    deposit_type_alternatives:  Optional[str]   = None
    primary_commodities:        Optional[str]   = None
    secondary_commodities:      Optional[str]   = None
    all_commodities:            Optional[str]   = None
    deposit_source:             Optional[str]   = None
    feature_type:               Optional[str]   = None
    sample_deposit_relation:    Optional[str]   = None
    sample_type:                Optional[str]   = None
    sampling_method:            Optional[str]   = None
    material_class:             Optional[str]   = None
    material_class_comments:    Optional[str]   = None
    earth_material_group:       Optional[str]   = None
    earth_material_qualifier:   Optional[str]   = None
    earth_material:             Optional[str]   = None
    mineral:                    Optional[str]   = None
    paragenetic_stage:          Optional[str]   = None
    mode_of_occurrence:         Optional[str]   = None
    texture:                    Optional[str]   = None
    associated_minerals:        Optional[str]   = None
    sample_preparation:         Optional[str]   = None
    analytical_method:          Optional[str]   = None
    instrument_type_model:      Optional[str]   = None
    laboratory_location:        Optional[str]   = None
    operating_conditions:       Optional[str]   = None
    standards_used:             Optional[str]   = None
    publication_date:           Optional[int]   = None
    sample_source:              Optional[str]   = None
    country:                    Optional[str]   = None
    state:                      Optional[str]   = None
    deposit_longitude_wgs84:    Optional[float] = None
    deposit_latitude_wgs84:     Optional[float] = None
    sample_longitude_wgs84:     Optional[float] = None
    sample_latitude_wgs84:      Optional[float] = None

    # --- extraction provenance ---
    extraction_backend:         Optional[str]   = None   # Which backend extracted this row (e.g., "docling", "marker", "camelot")
    extraction_confidence:      Optional[float] = None   # 0.0-1.0 confidence score
    backend_agreement:          Optional[str]   = None   # Cross-backend agreement summary (e.g., "3/5 backends agree")
    data_source_tag:            Optional[str]   = None   # "this_study" | "supplementary" | "reference_data" | "cited_study:Author2022"

    # --- elements (73 × 2 fields) ---
    ag_ppm: Optional[float] = None;  ag_detection_limit: Optional[float] = None
    al_ppm: Optional[float] = None;  al_detection_limit: Optional[float] = None
    as_ppm: Optional[float] = None;  as_detection_limit: Optional[float] = None
    au_ppm: Optional[float] = None;  au_detection_limit: Optional[float] = None
    b_ppm:  Optional[float] = None;  b_detection_limit:  Optional[float] = None
    ba_ppm: Optional[float] = None;  ba_detection_limit: Optional[float] = None
    be_ppm: Optional[float] = None;  be_detection_limit: Optional[float] = None
    bi_ppm: Optional[float] = None;  bi_detection_limit: Optional[float] = None
    br_ppm: Optional[float] = None;  br_detection_limit: Optional[float] = None
    ca_ppm: Optional[float] = None;  ca_detection_limit: Optional[float] = None
    cd_ppm: Optional[float] = None;  cd_detection_limit: Optional[float] = None
    ce_ppm: Optional[float] = None;  ce_detection_limit: Optional[float] = None
    cl_ppm: Optional[float] = None;  cl_detection_limit: Optional[float] = None
    co_ppm: Optional[float] = None;  co_detection_limit: Optional[float] = None
    cr_ppm: Optional[float] = None;  cr_detection_limit: Optional[float] = None
    cs_ppm: Optional[float] = None;  cs_detection_limit: Optional[float] = None
    cu_ppm: Optional[float] = None;  cu_detection_limit: Optional[float] = None
    dy_ppm: Optional[float] = None;  dy_detection_limit: Optional[float] = None
    er_ppm: Optional[float] = None;  er_detection_limit: Optional[float] = None
    eu_ppm: Optional[float] = None;  eu_detection_limit: Optional[float] = None
    f_ppm:  Optional[float] = None;  f_detection_limit:  Optional[float] = None
    fe_ppm: Optional[float] = None;  fe_detection_limit: Optional[float] = None
    ga_ppm: Optional[float] = None;  ga_detection_limit: Optional[float] = None
    gd_ppm: Optional[float] = None;  gd_detection_limit: Optional[float] = None
    ge_ppm: Optional[float] = None;  ge_detection_limit: Optional[float] = None
    hf_ppm: Optional[float] = None;  hf_detection_limit: Optional[float] = None
    hg_ppm: Optional[float] = None;  hg_detection_limit: Optional[float] = None
    ho_ppm: Optional[float] = None;  ho_detection_limit: Optional[float] = None
    in_ppm: Optional[float] = None;  in_detection_limit: Optional[float] = None
    ir_ppm: Optional[float] = None;  ir_detection_limit: Optional[float] = None
    k_ppm:  Optional[float] = None;  k_detection_limit:  Optional[float] = None
    la_ppm: Optional[float] = None;  la_detection_limit: Optional[float] = None
    li_ppm: Optional[float] = None;  li_detection_limit: Optional[float] = None
    lu_ppm: Optional[float] = None;  lu_detection_limit: Optional[float] = None
    mg_ppm: Optional[float] = None;  mg_detection_limit: Optional[float] = None
    mn_ppm: Optional[float] = None;  mn_detection_limit: Optional[float] = None
    mo_ppm: Optional[float] = None;  mo_detection_limit: Optional[float] = None
    na_ppm: Optional[float] = None;  na_detection_limit: Optional[float] = None
    nb_ppm: Optional[float] = None;  nb_detection_limit: Optional[float] = None
    nd_ppm: Optional[float] = None;  nd_detection_limit: Optional[float] = None
    ni_ppm: Optional[float] = None;  ni_detection_limit: Optional[float] = None
    os_ppm: Optional[float] = None;  os_detection_limit: Optional[float] = None
    p_ppm:  Optional[float] = None;  p_detection_limit:  Optional[float] = None
    pb_ppm: Optional[float] = None;  pb_detection_limit: Optional[float] = None
    pd_ppm: Optional[float] = None;  pd_detection_limit: Optional[float] = None
    pr_ppm: Optional[float] = None;  pr_detection_limit: Optional[float] = None
    pt_ppm: Optional[float] = None;  pt_detection_limit: Optional[float] = None
    rb_ppm: Optional[float] = None;  rb_detection_limit: Optional[float] = None
    re_ppm: Optional[float] = None;  re_detection_limit: Optional[float] = None
    rh_ppm: Optional[float] = None;  rh_detection_limit: Optional[float] = None
    ru_ppm: Optional[float] = None;  ru_detection_limit: Optional[float] = None
    s_ppm:  Optional[float] = None;  s_detection_limit:  Optional[float] = None
    sb_ppm: Optional[float] = None;  sb_detection_limit: Optional[float] = None
    sc_ppm: Optional[float] = None;  sc_detection_limit: Optional[float] = None
    se_ppm: Optional[float] = None;  se_detection_limit: Optional[float] = None
    si_ppm: Optional[float] = None;  si_detection_limit: Optional[float] = None
    sm_ppm: Optional[float] = None;  sm_detection_limit: Optional[float] = None
    sn_ppm: Optional[float] = None;  sn_detection_limit: Optional[float] = None
    sr_ppm: Optional[float] = None;  sr_detection_limit: Optional[float] = None
    ta_ppm: Optional[float] = None;  ta_detection_limit: Optional[float] = None
    tb_ppm: Optional[float] = None;  tb_detection_limit: Optional[float] = None
    te_ppm: Optional[float] = None;  te_detection_limit: Optional[float] = None
    th_ppm: Optional[float] = None;  th_detection_limit: Optional[float] = None
    ti_ppm: Optional[float] = None;  ti_detection_limit: Optional[float] = None
    tl_ppm: Optional[float] = None;  tl_detection_limit: Optional[float] = None
    tm_ppm: Optional[float] = None;  tm_detection_limit: Optional[float] = None
    u_ppm:  Optional[float] = None;  u_detection_limit:  Optional[float] = None
    v_ppm:  Optional[float] = None;  v_detection_limit:  Optional[float] = None
    w_ppm:  Optional[float] = None;  w_detection_limit:  Optional[float] = None
    y_ppm:  Optional[float] = None;  y_detection_limit:  Optional[float] = None
    yb_ppm: Optional[float] = None;  yb_detection_limit: Optional[float] = None
    zn_ppm: Optional[float] = None;  zn_detection_limit: Optional[float] = None
    zr_ppm: Optional[float] = None;  zr_detection_limit: Optional[float] = None

    def to_schema_dict(self) -> dict:
        """Return a dict keyed by ALL_COLUMNS (210 column names)."""
        d = self.model_dump()
        # Remap internal field names to schema column names
        renames = {
            "laboratory_location":  "laboratory_location/if reported",
            "operating_conditions": "operating_conditions/if reported",
            "standards_used":       "standards_used/if reported",
        }
        for src, dst in renames.items():
            if src in d:
                d[dst] = d.pop(src)
        # Ensure all 210 columns are present (fill missing with None)
        return {col: d.get(col) for col in ALL_COLUMNS}


# ──────────────────────────────────────────────────────────────────────────────
# Common element-name aliases for fuzzy table-header matching
# ──────────────────────────────────────────────────────────────────────────────
ELEMENT_ALIASES: dict[str, str] = {
    # symbol → canonical symbol (all lowercase)
    "silver":      "ag", "aluminum": "al", "aluminium": "al",
    "arsenic":     "as", "gold":     "au", "boron":     "b",
    "barium":      "ba", "beryllium":"be", "bismuth":   "bi",
    "bromine":     "br", "calcium":  "ca", "cadmium":   "cd",
    "cerium":      "ce", "chlorine": "cl", "cobalt":    "co",
    "chromium":    "cr", "caesium":  "cs", "cesium":    "cs",
    "copper":      "cu", "dysprosium":"dy","erbium":    "er",
    "europium":    "eu", "fluorine": "f",  "iron":      "fe",
    "gallium":     "ga", "gadolinium":"gd","germanium": "ge",
    "hafnium":     "hf", "mercury":  "hg", "holmium":   "ho",
    "indium":      "in", "iridium":  "ir", "potassium": "k",
    "lanthanum":   "la", "lithium":  "li", "lutetium":  "lu",
    "magnesium":   "mg", "manganese":"mn", "molybdenum":"mo",
    "sodium":      "na", "niobium":  "nb", "neodymium": "nd",
    "nickel":      "ni", "osmium":   "os", "phosphorus":"p",
    "lead":        "pb", "palladium":"pd", "praseodymium":"pr",
    "platinum":    "pt", "rubidium": "rb", "rhenium":   "re",
    "rhodium":     "rh", "ruthenium":"ru", "sulfur":    "s",
    "sulphur":     "s",  "antimony": "sb", "scandium":  "sc",
    "selenium":    "se", "silicon":  "si", "samarium":  "sm",
    "tin":         "sn", "strontium":"sr", "tantalum":  "ta",
    "terbium":     "tb", "tellurium":"te", "thorium":   "th",
    "titanium":    "ti", "thallium": "tl", "thulium":   "tm",
    "uranium":     "u",  "vanadium": "v",  "tungsten":  "w",
    "yttrium":     "y",  "ytterbium":"yb", "zinc":      "zn",
    "zirconium":   "zr",
}


def normalise_element_header(raw: str) -> Optional[str]:
    """Map a raw supplementary-table header to a canonical element symbol.

    Handles cases like 'Ag', 'AG', 'Silver', 'Fe2O3' (returns None for oxides),
    'Pb (ppm)', 'V51_ppm_mean' (isotope notation), etc.

    Returns lowercase symbol string or None if not an element column.
    """
    if not raw:
        return None
    import re as _re
    cleaned = raw.strip().lower()

    # Skip detection-limit / LOD paired columns (e.g. Mn55_ppm_LOD, Fe57_ppb_LOD)
    if _re.search(r"_pp[mb]_(lod|dl|det_lim)$", cleaned):
        return None

    # Strip isotope_ppm_content suffix (e.g. Mn55_ppm_content → mn55)
    cleaned = _re.sub(r"_pp[mb]_(content|conc)$", "", cleaned)

    # Strip isotope-mass suffixes like _ppm_m55, _ppm_m57, _ppb_m208
    cleaned = _re.sub(r"_pp[mb]_m\d+", "", cleaned)

    # Strip statistical suffixes (mean, 2se, 2sd, median, etc.)
    cleaned = _re.sub(r"[_\s]+(mean|2se|2sd|sd|se|median|avg|rsd|1sd|1se)$", "", cleaned)

    # Strip unit suffixes in various forms
    for suffix in [
        " (ppm)", " (ppb)", " (wt%)", " wt%", "_wt%", "(wt%)", " (wt.%)", "(wt.%)",
        "(mass%)", "(mass %)", " (mass%)", " (mass %)", " mass%",
        " (%)", " %",
        " µg/g", " ug/g", " μg/g", " mg/kg", " g/t",
        "_ppm", "_ppb", " ppm", " ppb",
    ]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    cleaned = cleaned.strip().rstrip("_").strip()

    # Direct match to symbol
    if cleaned in ELEMENT_SYMBOLS:
        return cleaned
    # Alias lookup
    if cleaned in ELEMENT_ALIASES:
        return ELEMENT_ALIASES[cleaned]

    # Strip trailing isotope mass numbers (e.g., v51 → v, cr52 → cr, mn55 → mn)
    # Only for short strings (≤6 chars) to avoid false positives like "B291152" → "b"
    # Require ≥2 digit mass numbers to avoid "B6"→"b", "F4"→"f" false matches
    if len(cleaned) <= 6:
        stripped = _re.sub(r"\d+$", "", cleaned)
        digits = cleaned[len(stripped):]
        if stripped and digits and len(digits) >= 2:
            if stripped in ELEMENT_SYMBOLS:
                return stripped
            if stripped in ELEMENT_ALIASES:
                return ELEMENT_ALIASES[stripped]

    # Strip leading isotope mass numbers (e.g., 57fe → fe, 208pb → pb, 34s → s)
    if len(cleaned) <= 6:
        stripped_leading = _re.sub(r"^\d+", "", cleaned)
        digits = cleaned[: len(cleaned) - len(stripped_leading)]
        if stripped_leading and digits and len(digits) >= 2:
            if stripped_leading in ELEMENT_SYMBOLS:
                return stripped_leading
            if stripped_leading and stripped_leading in ELEMENT_ALIASES:
                return ELEMENT_ALIASES[stripped_leading]

    return None
