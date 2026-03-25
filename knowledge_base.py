"""
knowledge_base.py - Geochemical domain ontology for ontology-anchored extraction.

Provides curated taxonomies that are injected into LLM prompts to guide accurate
classification of deposit types, minerals, analytical methods, and material classes.
Also provides post-extraction validation to catch inconsistencies.

Technique: Ontology-Anchored Hierarchical Extraction
  1. Domain taxonomy is injected into the system prompt as reference material
  2. LLM uses taxonomy during extraction for consistent classification
  3. Post-extraction validation checks consistency against the ontology
"""

from __future__ import annotations
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Deposit Classification Taxonomy
# Maps deposit_type keywords → (deposit_environment, deposit_group)
# ──────────────────────────────────────────────────────────────────────────────

DEPOSIT_TAXONOMY: dict[str, tuple[str, str]] = {
    # Mississippi Valley-type
    "mvt":                  ("Basin hydrothermal",      "Mississippi Valley-type (MVT)"),
    "mississippi valley":   ("Basin hydrothermal",      "Mississippi Valley-type (MVT)"),
    # SEDEX
    "sedex":                ("Seafloor hydrothermal",   "Sedimentary Exhalative (SEDEX)"),
    "sedimentary exhalative": ("Seafloor hydrothermal", "Sedimentary Exhalative (SEDEX)"),
    # VMS / VHMS
    "vms":                  ("Seafloor hydrothermal",   "Volcanic-hosted massive sulfide (VMS)"),
    "vhms":                 ("Seafloor hydrothermal",   "Volcanic-hosted massive sulfide (VMS)"),
    "volcanogenic massive": ("Seafloor hydrothermal",   "Volcanic-hosted massive sulfide (VMS)"),
    "kuroko":               ("Seafloor hydrothermal",   "Volcanic-hosted massive sulfide (VMS)"),
    # Porphyry
    "porphyry":             ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry cu":          ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry cu-mo":       ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry cu-au":       ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry mo":          ("Magmatic hydrothermal",   "Porphyry"),
    # Skarn
    "skarn":                ("Magmatic hydrothermal",   "Skarn"),
    "exoskarn":             ("Magmatic hydrothermal",   "Skarn"),
    "endoskarn":            ("Magmatic hydrothermal",   "Skarn"),
    # Epithermal
    "epithermal":           ("Epithermal",              "Epithermal"),
    "high-sulfidation":     ("Epithermal",              "Epithermal"),
    "low-sulfidation":      ("Epithermal",              "Epithermal"),
    "intermediate-sulfidation": ("Epithermal",          "Epithermal"),
    # Orogenic
    "orogenic":             ("Metamorphic",             "Orogenic"),
    "orogenic gold":        ("Metamorphic",             "Orogenic gold"),
    "lode gold":            ("Metamorphic",             "Orogenic gold"),
    # IOCG
    "iocg":                 ("Magmatic hydrothermal",   "Iron oxide copper-gold (IOCG)"),
    "iron oxide copper":    ("Magmatic hydrothermal",   "Iron oxide copper-gold (IOCG)"),
    # Carlin-type
    "carlin":               ("Sediment-hosted",         "Carlin-type"),
    # Stratiform Cu
    "stratiform":           ("Basin hydrothermal",      "Stratiform sediment-hosted Cu"),
    "sediment-hosted cu":   ("Basin hydrothermal",      "Stratiform sediment-hosted Cu"),
    "sediment-hosted copper": ("Basin hydrothermal",    "Stratiform sediment-hosted Cu"),
    "kupferschiefer":       ("Basin hydrothermal",      "Stratiform sediment-hosted Cu"),
    # Magmatic Ni-Cu
    "magmatic ni":          ("Magmatic",                "Magmatic Ni-Cu-PGE"),
    "magmatic sulfide":     ("Magmatic",                "Magmatic Ni-Cu-PGE"),
    "ni-cu-pge":            ("Magmatic",                "Magmatic Ni-Cu-PGE"),
    # Greisen
    "greisen":              ("Magmatic hydrothermal",   "Greisen"),
    # Pegmatite
    "pegmatite":            ("Magmatic",                "Pegmatite"),
    # Carbonatite
    "carbonatite":          ("Magmatic",                "Carbonatite"),
    # Laterite
    "laterite":             ("Supergene",               "Laterite"),
    # Placer
    "placer":               ("Surficial",               "Placer"),
    # BIF
    "bif":                  ("Chemical sedimentary",    "Banded iron formation (BIF)"),
    "banded iron":          ("Chemical sedimentary",    "Banded iron formation (BIF)"),
    # Replacement
    "carbonate replacement": ("Magmatic hydrothermal",  "Carbonate replacement deposit (CRD)"),
    "crd":                  ("Magmatic hydrothermal",   "Carbonate replacement deposit (CRD)"),
    "manto":                ("Magmatic hydrothermal",   "Carbonate replacement deposit (CRD)"),
    # Five-element veins
    "five-element":         ("Metamorphic",             "Five-element vein"),
    "five element":         ("Metamorphic",             "Five-element vein"),
    # Intrusion-related
    "intrusion-related":    ("Magmatic hydrothermal",   "Intrusion-related"),
    "reduced intrusion":    ("Magmatic hydrothermal",   "Intrusion-related"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Mineral Classification
# Maps mineral name → (material_class, earth_material_group)
# ──────────────────────────────────────────────────────────────────────────────

MINERAL_TAXONOMY: dict[str, tuple[str, str]] = {
    # Sulfide ore minerals
    "sphalerite":       ("mineral separate", "mineralisation"),
    "pyrite":           ("mineral separate", "mineralisation"),
    "chalcopyrite":     ("mineral separate", "mineralisation"),
    "galena":           ("mineral separate", "mineralisation"),
    "pyrrhotite":       ("mineral separate", "mineralisation"),
    "arsenopyrite":     ("mineral separate", "mineralisation"),
    "marcasite":        ("mineral separate", "mineralisation"),
    "bornite":          ("mineral separate", "mineralisation"),
    "chalcocite":       ("mineral separate", "mineralisation"),
    "covellite":        ("mineral separate", "mineralisation"),
    "enargite":         ("mineral separate", "mineralisation"),
    "tennantite":       ("mineral separate", "mineralisation"),
    "tetrahedrite":     ("mineral separate", "mineralisation"),
    "stannite":         ("mineral separate", "mineralisation"),
    "molybdenite":      ("mineral separate", "mineralisation"),
    "pentlandite":      ("mineral separate", "mineralisation"),
    "millerite":        ("mineral separate", "mineralisation"),
    "cobaltite":        ("mineral separate", "mineralisation"),
    "linnaeite":        ("mineral separate", "mineralisation"),
    "siegenite":        ("mineral separate", "mineralisation"),
    "carrollite":       ("mineral separate", "mineralisation"),
    "cubanite":         ("mineral separate", "mineralisation"),
    "bismuthinite":     ("mineral separate", "mineralisation"),
    "realgar":          ("mineral separate", "mineralisation"),
    "orpiment":         ("mineral separate", "mineralisation"),
    "cinnabar":         ("mineral separate", "mineralisation"),
    "stibnite":         ("mineral separate", "mineralisation"),
    "acanthite":        ("mineral separate", "mineralisation"),
    "argentite":        ("mineral separate", "mineralisation"),
    "proustite":        ("mineral separate", "mineralisation"),
    "polybasite":       ("mineral separate", "mineralisation"),
    "electrum":         ("mineral separate", "mineralisation"),
    "native gold":      ("mineral separate", "mineralisation"),
    "native copper":    ("mineral separate", "mineralisation"),
    "native silver":    ("mineral separate", "mineralisation"),
    "cassiterite":      ("mineral separate", "mineralisation"),
    "wolframite":       ("mineral separate", "mineralisation"),
    "scheelite":        ("mineral separate", "mineralisation"),
    "magnetite":        ("mineral separate", "mineralisation"),
    "hematite":         ("mineral separate", "mineralisation"),
    "ilmenite":         ("mineral separate", "mineralisation"),
    "chromite":         ("mineral separate", "mineralisation"),
    "columbite":        ("mineral separate", "mineralisation"),
    "tantalite":        ("mineral separate", "mineralisation"),
    # Sulfosalts
    "bournonite":       ("mineral separate", "mineralisation"),
    "jamesonite":       ("mineral separate", "mineralisation"),
    "geocronite":       ("mineral separate", "mineralisation"),
    "boulangerite":     ("mineral separate", "mineralisation"),
    # Tellurides
    "hessite":          ("mineral separate", "mineralisation"),
    "calaverite":       ("mineral separate", "mineralisation"),
    "sylvanite":        ("mineral separate", "mineralisation"),
    # Gangue / rock-forming minerals
    "quartz":           ("mineral separate", "gangue"),
    "calcite":          ("mineral separate", "gangue"),
    "dolomite":         ("mineral separate", "gangue"),
    "fluorite":         ("mineral separate", "gangue"),
    "barite":           ("mineral separate", "gangue"),
    "ankerite":         ("mineral separate", "gangue"),
    "siderite":         ("mineral separate", "gangue"),
    "muscovite":        ("mineral separate", "gangue"),
    "biotite":          ("mineral separate", "gangue"),
    "chlorite":         ("mineral separate", "gangue"),
    "feldspar":         ("mineral separate", "gangue"),
    "garnet":           ("mineral separate", "gangue"),
    "epidote":          ("mineral separate", "gangue"),
    "tourmaline":       ("mineral separate", "gangue"),
    "amphibole":        ("mineral separate", "gangue"),
    "pyroxene":         ("mineral separate", "gangue"),
    "olivine":          ("mineral separate", "gangue"),
    "apatite":          ("mineral separate", "gangue"),
    "titanite":         ("mineral separate", "gangue"),
    "zircon":           ("mineral separate", "gangue"),
    "rutile":           ("mineral separate", "gangue"),
    "monazite":         ("mineral separate", "gangue"),
    "xenotime":         ("mineral separate", "gangue"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Analytical Method Standardization
# ──────────────────────────────────────────────────────────────────────────────

METHOD_STANDARDIZATION: dict[str, str] = {
    # LA-ICPMS variants
    "la-icpms":                     "LA-ICPMS",
    "la-icp-ms":                    "LA-ICPMS",
    "la icpms":                     "LA-ICPMS",
    "la icp ms":                    "LA-ICPMS",
    "laser ablation icp-ms":        "LA-ICPMS",
    "laser ablation icpms":         "LA-ICPMS",
    "laser ablation inductively coupled plasma mass spectrometry": "LA-ICPMS",
    "laser ablation":               "LA-ICPMS",
    "laicpms":                      "LA-ICPMS",
    "la-icp ms":                    "LA-ICPMS",
    # EMPA / EPMA variants
    "empa":                         "EMPA",
    "epma":                         "EMPA",
    "electron microprobe":          "EMPA",
    "electron microprobe analysis": "EMPA",
    "electron probe microanalysis": "EMPA",
    "electron probe micro-analysis": "EMPA",
    "electron probe":               "EMPA",
    "microprobe":                   "EMPA",
    # XRF
    "xrf":                          "XRF",
    "x-ray fluorescence":           "XRF",
    # ICP-MS (solution)
    "icp-ms":                       "ICP-MS",
    "icpms":                        "ICP-MS",
    "icp ms":                       "ICP-MS",
    "inductively coupled plasma mass spectrometry": "ICP-MS",
    # ICP-OES / ICP-AES
    "icp-oes":                      "ICP-OES",
    "icp-aes":                      "ICP-OES",
    "icp oes":                      "ICP-OES",
    # Others
    "inaa":                         "INAA",
    "tims":                         "TIMS",
    "mc-icpms":                     "MC-ICPMS",
    "mc-icp-ms":                    "MC-ICPMS",
    "shrimp":                       "SHRIMP",
    "sims":                         "SIMS",
    "pixe":                         "PIXE",
    "pige":                         "PIGE",
    "sem-eds":                      "SEM-EDS",
    "eds":                          "SEM-EDS",
    "wds":                          "WDS",
    "aas":                          "AAS",
    "atomic absorption":            "AAS",
}


# ──────────────────────────────────────────────────────────────────────────────
# Country / Region → ISO 3-letter code hints
# ──────────────────────────────────────────────────────────────────────────────

COUNTRY_HINTS: dict[str, str] = {
    "china":        "CHN", "chinese":      "CHN", "prc":          "CHN",
    "australia":    "AUS", "australian":   "AUS",
    "usa":          "USA", "united states": "USA", "american":    "USA",
    "canada":       "CAN", "canadian":     "CAN",
    "mexico":       "MEX", "mexican":      "MEX",
    "brazil":       "BRA", "brazilian":    "BRA",
    "argentina":    "ARG", "argentine":    "ARG",
    "chile":        "CHL", "chilean":      "CHL",
    "peru":         "PER", "peruvian":     "PER",
    "south africa": "ZAF", "south african": "ZAF",
    "namibia":      "NAM",
    "zambia":       "ZMB",
    "congo":        "COD",
    "morocco":      "MAR",
    "iran":         "IRN", "iranian":      "IRN",
    "turkey":       "TUR", "turkish":      "TUR", "türkiye": "TUR",
    "india":        "IND", "indian":       "IND",
    "japan":        "JPN", "japanese":     "JPN",
    "south korea":  "KOR", "korea":        "KOR",
    "russia":       "RUS", "russian":      "RUS",
    "sweden":       "SWE", "swedish":      "SWE",
    "norway":       "NOR", "norwegian":    "NOR",
    "finland":      "FIN", "finnish":      "FIN",
    "spain":        "ESP", "spanish":      "ESP",
    "portugal":     "PRT", "portuguese":   "PRT",
    "italy":        "ITA", "italian":      "ITA",
    "germany":      "DEU", "german":       "DEU",
    "france":       "FRA", "french":       "FRA",
    "uk":           "GBR", "united kingdom": "GBR", "britain": "GBR",
    "ireland":      "IRL",
    "greece":       "GRC", "greek":        "GRC",
    "poland":       "POL", "polish":       "POL",
    "czech":        "CZE", "czech republic": "CZE", "czechia": "CZE",
    "slovakia":     "SVK",
    "austria":      "AUT",
    "switzerland":  "CHE",
    "romania":      "ROU",
    "bulgaria":     "BGR",
    "serbia":       "SRB",
    "mongolia":     "MNG", "mongolian":    "MNG",
    "kazakhstan":   "KAZ",
    "uzbekistan":   "UZB",
    "tajikistan":   "TJK",
    "papua new guinea": "PNG",
    "indonesia":    "IDN", "indonesian":   "IDN",
    "philippines":  "PHL",
    "vietnam":      "VNM",
    "thailand":     "THA",
    "myanmar":      "MMR",
    "laos":         "LAO",
    "new zealand":  "NZL",
    "bolivia":      "BOL",
    "colombia":     "COL",
    "ecuador":      "ECU",
    "cuba":         "CUB",
    "dominican republic": "DOM",
}


# ──────────────────────────────────────────────────────────────────────────────
# Method detection from supplementary filenames
# ──────────────────────────────────────────────────────────────────────────────

FILENAME_METHOD_PATTERNS: dict[str, str] = {
    "empa":     "EPMA",
    "epma":     "EPMA",
    "laicpms":  "LA-ICPMS",
    "laipcms":  "LA-ICPMS",   # common typo
    "la-icpms": "LA-ICPMS",
    "la_icpms": "LA-ICPMS",
    "xrf":      "XRF",
    "icpms":    "ICP-MS",
    "icp-ms":   "ICP-MS",
    "inaa":     "INAA",
    "sims":     "SIMS",
}


def detect_method_from_filename(filename: str) -> Optional[str]:
    """Detect analytical method from a supplementary filename.

    E.g., '2024_Wu_etal_EMPA.xlsx' → 'EMPA'
         '2024_Wu_etal_LAICPMS.xlsx' → 'LA-ICPMS'
    """
    stem = filename.rsplit(".", 1)[0].lower()
    # Check for method patterns in the filename (order by length descending for specificity)
    for pattern, method in sorted(FILENAME_METHOD_PATTERNS.items(),
                                  key=lambda x: len(x[0]), reverse=True):
        if pattern in stem:
            return method
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Method string normalization
# ──────────────────────────────────────────────────────────────────────────────

_METHOD_NORMALIZATION: dict[str, str] = {
    "la-icp-ms": "LA-ICPMS",
    "la icp ms": "LA-ICPMS",
    "la-icpms": "LA-ICPMS",
    "laicpms": "LA-ICPMS",
    "laipcms": "LA-ICPMS",
    "laser ablation icp-ms": "LA-ICPMS",
    "laser ablation icpms": "LA-ICPMS",
    "empa": "EPMA",
    "epma": "EPMA",
    "emp": "EPMA",
    "electron microprobe": "EPMA",
    "electron probe microanalysis": "EPMA",
    "electron probe micro-analysis": "EPMA",
    "icp-ms": "ICP-MS",
    "icpms": "ICP-MS",
}


def normalize_method(method: str) -> str:
    """Normalize an analytical method string to its canonical form."""
    if not method:
        return method
    key = method.strip().lower()
    return _METHOD_NORMALIZATION.get(key, method.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Prompt injection: format knowledge base for LLM consumption
# ──────────────────────────────────────────────────────────────────────────────

def get_deposit_taxonomy_prompt() -> str:
    """Format deposit classification rules as LLM reference text."""
    # Group by environment for cleaner presentation
    env_groups: dict[str, list[str]] = {}
    seen = set()
    for kw, (env, group) in DEPOSIT_TAXONOMY.items():
        key = (env, group)
        if key not in seen:
            seen.add(key)
            env_groups.setdefault(env, []).append(group)

    lines = ["DEPOSIT CLASSIFICATION REFERENCE:"]
    for env, groups in sorted(env_groups.items()):
        lines.append(f"  Environment: '{env}'")
        for g in sorted(set(groups)):
            lines.append(f"    → {g}")
    return "\n".join(lines)


def get_method_standardization_prompt() -> str:
    """Format method standardization rules as LLM reference text."""
    # Group by standardized method
    std_groups: dict[str, list[str]] = {}
    for raw, std in METHOD_STANDARDIZATION.items():
        std_groups.setdefault(std, []).append(raw)

    lines = ["ANALYTICAL METHOD STANDARDIZATION:"]
    for std, raws in sorted(std_groups.items()):
        examples = ", ".join(f'"{r}"' for r in sorted(raws)[:3])
        lines.append(f"  {examples} → '{std}'")
    return "\n".join(lines)


def get_mineral_classification_prompt() -> str:
    """Format mineral classification rules as LLM reference text."""
    lines = [
        "MINERAL CLASSIFICATION RULES:",
        "  When the analyzed material is a specific mineral (e.g., sphalerite, pyrite):",
        "    material_class = 'mineral separate'",
        "    earth_material_group = 'mineralisation' (for ore minerals)",
        "    earth_material_group = 'gangue' (for non-ore minerals like quartz, calcite)",
        "  When the analyzed material is whole rock:",
        "    material_class = 'rock'",
        "    earth_material_group = infer from rock type (igneous/sedimentary/metamorphic)",
    ]
    return "\n".join(lines)


def get_knowledge_base_prompt() -> str:
    """Return the full knowledge base section for injection into LLM prompts."""
    sections = [
        "## GEOCHEMICAL DOMAIN KNOWLEDGE REFERENCE",
        "",
        get_deposit_taxonomy_prompt(),
        "",
        get_method_standardization_prompt(),
        "",
        get_mineral_classification_prompt(),
        "",
        "ISO COUNTRY CODES (common):",
        "  China=CHN, Australia=AUS, USA=USA, Canada=CAN, Brazil=BRA,",
        "  South Africa=ZAF, Sweden=SWE, Chile=CHL, Peru=PER, Mexico=MEX,",
        "  Iran=IRN, Turkey=TUR, India=IND, Japan=JPN, Russia=RUS,",
        "  Argentina=ARG, Czech Republic=CZE, Poland=POL, Greece=GRC",
    ]
    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────────────────────
# Post-extraction validation
# ──────────────────────────────────────────────────────────────────────────────

def standardize_method(raw_method: Optional[str]) -> Optional[str]:
    """Standardize an analytical method string using the taxonomy."""
    if not raw_method:
        return raw_method
    key = raw_method.strip().lower()
    return METHOD_STANDARDIZATION.get(key, raw_method)


def infer_deposit_environment(deposit_type: Optional[str]) -> Optional[str]:
    """Infer deposit_environment from deposit_type using the taxonomy."""
    if not deposit_type:
        return None
    key = deposit_type.strip().lower()
    for pattern, (env, _) in DEPOSIT_TAXONOMY.items():
        if pattern in key:
            return env
    return None


def infer_deposit_group(deposit_type: Optional[str]) -> Optional[str]:
    """Infer deposit_group from deposit_type using the taxonomy."""
    if not deposit_type:
        return None
    key = deposit_type.strip().lower()
    for pattern, (_, group) in DEPOSIT_TAXONOMY.items():
        if pattern in key:
            return group
    return None


def infer_mineral_class(mineral: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Infer (material_class, earth_material_group) from mineral name."""
    if not mineral:
        return None, None
    key = mineral.strip().lower()
    # Handle comma-separated minerals — use the first one
    if "," in key:
        key = key.split(",")[0].strip()
    if key in MINERAL_TAXONOMY:
        return MINERAL_TAXONOMY[key]
    return None, None


def _normalize_deposit_name(name: str) -> str:
    """Strip common verbose suffixes from deposit names.

    LLMs often return 'Bainiuchang Zn-Sn polymetallic deposit' when GT expects 'Bainiuchang'.
    """
    import re
    if not name:
        return name
    # Remove trailing commodity-deposit pattern: "Cu-Au deposit", "Zn-Pb mine", etc.
    # Pattern: optional commodity metals + deposit/mine/prospect/occurrence
    cleaned = re.sub(
        r'\s+(?:(?:[A-Z][a-z]?[-–])*[A-Z][a-z]?\s+)?'
        r'(?:polymetallic\s+)?'
        r'(?:deposit|mine|prospect|occurrence|ore\s+field|orefield|mining\s+district|district)\s*$',
        '', name, flags=re.IGNORECASE
    ).strip()
    # Remove trailing parenthetical commodity info: "Bainiuchang (Zn-Sn)"
    cleaned = re.sub(r'\s*\([A-Z][a-z]?(?:[-–][A-Z][a-z]?)*\)\s*$', '', cleaned).strip()
    return cleaned if cleaned else name


def _normalize_standards(standards: str) -> str:
    """Extract just standard reference material codes from verbose LLM output.

    LLM: 'USGS sulfide reference material MASS-1 (Wilson et al., 2002), NIST SRM610'
    GT:  'MASS-1, NIST SRM610'
    """
    import re
    if not standards:
        return standards
    # Common reference material patterns
    _STD_PATTERNS = [
        r'MASS-\d+[a-z]?',
        r'NIST\s*SRM\s*\d+[a-z]?',
        r'SRM\s*\d+[a-z]?',
        r'BHVO-\d+[a-z]?',
        r'BCR-\d+[a-z]?',
        r'BIR-\d+[a-z]?',
        r'GSE-\d+[a-z]?',
        r'GSD-\d+[a-z]?',
        r'STDGL\d+[a-z]*-?\d*',
        r'StdGl\d+[a-z]*-?\d*',
        r'NBS\s*\d+[a-z]?',
        r'USGS\s+[A-Z]+[-\d]+',
        r'[A-Z]+-\d+[a-z]?(?:\s*[A-Z])?',  # Generic CODE-NUMBER pattern
        r'FeS2?(?:\s*\([^)]+\))?',
        r'ZnS(?:\s*\([^)]+\))?',
        r'CuFeS2?(?:\s*\([^)]+\))?',
        r'PO-\d+',
        r'PS-\d+',
        r'S-[A-Z]+\d+[a-z]?',
    ]
    # Try to extract standard codes; if found, return them comma-separated
    found = []
    for pat in _STD_PATTERNS:
        matches = re.findall(pat, standards, re.IGNORECASE)
        found.extend(matches)
    if found:
        # Deduplicate preserving order
        seen = set()
        unique = []
        for s in found:
            s_norm = s.strip()
            if s_norm.lower() not in seen:
                seen.add(s_norm.lower())
                unique.append(s_norm)
        return ', '.join(unique)
    return standards


def validate_and_enrich_metadata(metadata: dict) -> dict:
    """Post-process extracted metadata: fill gaps using domain knowledge.

    Also normalizes verbose LLM outputs to concise forms matching GT conventions.
    """
    result = dict(metadata)

    # ── Normalize verbose LLM outputs to concise forms ──
    if result.get("deposit_name"):
        result["deposit_name"] = _normalize_deposit_name(result["deposit_name"])

    # Note: standards_used is kept verbatim — GT uses full descriptions, not just codes

    # Standardize analytical method
    if result.get("analytical_method"):
        std = standardize_method(result["analytical_method"])
        if std:
            result["analytical_method"] = std

    # Infer deposit_environment from deposit_type if missing
    if not result.get("deposit_environment") and result.get("deposit_type"):
        env = infer_deposit_environment(result["deposit_type"])
        if env:
            result["deposit_environment"] = env

    # Infer deposit_group from deposit_type if missing
    if not result.get("deposit_group") and result.get("deposit_type"):
        group = infer_deposit_group(result["deposit_type"])
        if group:
            result["deposit_group"] = group

    # Infer material_class and earth_material_group from mineral if missing
    mineral = result.get("mineral")
    if mineral:
        mat_class, mat_group = infer_mineral_class(mineral)
        if mat_class and not result.get("material_class"):
            result["material_class"] = mat_class
        if mat_group and not result.get("earth_material_group"):
            result["earth_material_group"] = mat_group

    return result
