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

# ──────────────────────────────────────────────────────────────────────────────
# Deposit Classification — loaded from DARPA CRITICALMAAS sri-ta2 / Hofstra 2021
# 189 deposit types, 14 environments, 53 groups
# ──────────────────────────────────────────────────────────────────────────────

_CMMI_DEPOSIT_TYPES: list[dict] = []  # Loaded lazily from sri-ta2 CSV
_CMMI_LOADED = False


def _load_cmmi_deposit_types() -> list[dict]:
    """Load the full 189-type CMMI hierarchy from sri-ta2 data."""
    global _CMMI_DEPOSIT_TYPES, _CMMI_LOADED
    if _CMMI_LOADED:
        return _CMMI_DEPOSIT_TYPES
    try:
        import pandas as pd
        from pathlib import Path
        base = Path(__file__).parent

        # Try sri-ta2 CSV first (authoritative)
        csv_path = base / "sri-ta2" / "minmod" / "deposit_type.csv"
        desc_path = base / "sri-ta2" / "taxonomy" / "cmmi_options_full_description_with_number.csv"

        if csv_path.exists():
            dt = pd.read_csv(csv_path)
            descriptions = {}
            if desc_path.exists():
                desc_df = pd.read_csv(desc_path)
                for _, r in desc_df.iterrows():
                    descriptions[r["Deposit type"]] = str(r["Description"])

            for _, row in dt.iterrows():
                _CMMI_DEPOSIT_TYPES.append({
                    "name": row["Deposit type"],
                    "environment": row["Deposit environment"],
                    "group": row["Deposit group"],
                    "minmod_id": row["Minmod ID"],
                    "description": descriptions.get(row["Deposit type"], ""),
                })

        # Fallback: load from JSON
        elif (base / "deposit_classification_cmmi.json").exists():
            import json
            with open(base / "deposit_classification_cmmi.json") as f:
                data = json.load(f)
            for env, groups in data["hierarchy"].items():
                for grp, types in groups.items():
                    for t in types:
                        _CMMI_DEPOSIT_TYPES.append({
                            "name": t["name"],
                            "environment": env,
                            "group": grp,
                            "minmod_id": t.get("minmod_id", ""),
                            "description": t.get("description", ""),
                        })
    except Exception:
        pass
    _CMMI_LOADED = True
    return _CMMI_DEPOSIT_TYPES


# Legacy keyword → (environment, group) mapping for backward compatibility
# AND for fast keyword-based deposit type inference from paper text
DEPOSIT_TAXONOMY: dict[str, tuple[str, str]] = {
    # These are keyword shortcuts — the full 189-type CMMI is authoritative
    "mvt":                  ("Basin hydrothermal",      "Mississippi Valley- type (MVT)"),
    "mississippi valley":   ("Basin hydrothermal",      "Mississippi Valley- type (MVT)"),
    "sedex":                ("Basin hydrothermal",      "Sediment-hosted"),
    "sedimentary exhalative": ("Basin hydrothermal",    "Sediment-hosted"),
    "vms":                  ("Volcanic basin hydrothermal", "Volcanogenic massive sulfide (VMS)"),
    "vhms":                 ("Volcanic basin hydrothermal", "Volcanogenic massive sulfide (VMS)"),
    "volcanogenic massive": ("Volcanic basin hydrothermal", "Volcanogenic massive sulfide (VMS)"),
    "kuroko":               ("Volcanic basin hydrothermal", "Volcanogenic massive sulfide (VMS)"),
    "porphyry":             ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry cu":          ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry cu-mo":       ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry cu-au":       ("Magmatic hydrothermal",   "Porphyry"),
    "porphyry mo":          ("Magmatic hydrothermal",   "Porphyry"),
    "skarn":                ("Magmatic hydrothermal",   "Skarn"),
    "exoskarn":             ("Magmatic hydrothermal",   "Skarn"),
    "endoskarn":            ("Magmatic hydrothermal",   "Skarn"),
    "epithermal":           ("Magmatic hydrothermal",   "Epithermal"),
    "high-sulfidation":     ("Magmatic hydrothermal",   "Epithermal"),
    "low-sulfidation":      ("Magmatic hydrothermal",   "Epithermal"),
    "intermediate-sulfidation": ("Magmatic hydrothermal", "Epithermal"),
    "orogenic":             ("Metamorphic hydrothermal", "Orogenic"),
    "orogenic gold":        ("Metamorphic hydrothermal", "Orogenic"),
    "lode gold":            ("Metamorphic hydrothermal", "Orogenic"),
    "iocg":                 ("Regional metasomatic",    "IOCG"),
    "iron oxide copper":    ("Regional metasomatic",    "IOCG"),
    "carlin":               ("Magmatic hydrothermal",   "Carlin-type"),
    "stratiform":           ("Basin hydrothermal",      "Sediment-hosted"),
    "sediment-hosted cu":   ("Basin hydrothermal",      "Sediment-hosted"),
    "sediment-hosted copper": ("Basin hydrothermal",    "Sediment-hosted"),
    "kupferschiefer":       ("Basin hydrothermal",      "Sediment-hosted"),
    "magmatic ni":          ("Magmatic",                "Ultramafic and (or) mafic intrusion"),
    "magmatic sulfide":     ("Magmatic",                "Ultramafic and (or) mafic intrusion"),
    "ni-cu-pge":            ("Magmatic",                "Ultramafic and (or) mafic intrusion"),
    "greisen":              ("Magmatic hydrothermal",   "Greisen"),
    "pegmatite":            ("Magmatic",                "Pegmatite"),
    "carbonatite":          ("Magmatic",                "Carbonatite"),
    "laterite":             ("Supergene",               "Laterite"),
    "placer":               ("Erosional",               "Placer"),
    "bif":                  ("Basin chemical",          "Iron formation"),
    "banded iron":          ("Basin chemical",          "Iron formation"),
    "carbonate replacement": ("Magmatic hydrothermal",  "Replacement"),
    "crd":                  ("Magmatic hydrothermal",   "Replacement"),
    "manto":                ("Magmatic hydrothermal",   "Replacement"),
    "five-element":         ("Basin hydrothermal",      "Five-element"),
    "five element":         ("Basin hydrothermal",      "Five-element"),
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
    """Format the full CMMI 189-type deposit classification for LLM prompts.

    Uses the authoritative sri-ta2/Hofstra 2021 hierarchy.
    Falls back to the keyword taxonomy if sri-ta2 data unavailable.
    """
    cmmi = _load_cmmi_deposit_types()

    if cmmi:
        # Group by environment → group → types
        hierarchy: dict[str, dict[str, list[str]]] = {}
        for entry in cmmi:
            env = entry["environment"]
            grp = entry["group"]
            name = entry["name"]
            hierarchy.setdefault(env, {}).setdefault(grp, []).append(name)

        lines = [
            "DEPOSIT CLASSIFICATION REFERENCE (Hofstra et al. 2021 CMMI — 189 types):",
            "Source: DARPA CRITICALMAAS sri-ta2. Use EXACTLY these names.",
            "",
        ]
        for env in sorted(hierarchy.keys()):
            groups = hierarchy[env]
            n_types = sum(len(v) for v in groups.values())
            lines.append(f"  Environment: '{env}' ({n_types} types)")
            for grp in sorted(groups.keys()):
                types = groups[grp]
                lines.append(f"    Group: '{grp}'")
                for t in sorted(types):
                    lines.append(f"      - {t}")
        return "\n".join(lines)

    # Fallback to keyword taxonomy
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


def score_deposit_types(
    paper_text: str,
    deposit_name: Optional[str] = None,
    minerals: Optional[list[str]] = None,
    commodities: Optional[list[str]] = None,
) -> list[dict]:
    """Score all 189 CMMI deposit types against paper context.

    Returns top-N scored deposit types with confidence and reasoning.
    Uses keyword matching against deposit type descriptions from sri-ta2.

    Each result: {name, environment, group, score, reason}
    """
    cmmi = _load_cmmi_deposit_types()
    if not cmmi:
        return []

    text_lower = paper_text.lower() if paper_text else ""
    deposit_lower = deposit_name.lower() if deposit_name else ""
    mineral_set = {m.lower() for m in (minerals or [])}
    commodity_set = {c.lower() for c in (commodities or [])}

    scored = []
    for entry in cmmi:
        score = 0.0
        reasons = []
        name_lower = entry["name"].lower()
        desc_lower = entry["description"].lower() if entry["description"] else ""
        grp_lower = entry["group"].lower()
        env_lower = entry["environment"].lower()

        # 1. Direct name match in paper text (strongest signal)
        if name_lower in text_lower:
            score += 0.4
            reasons.append(f"deposit type '{entry['name']}' mentioned in text")

        # 2. Group name match
        if grp_lower in text_lower:
            score += 0.2
            reasons.append(f"group '{entry['group']}' in text")

        # 3. Environment match
        if env_lower in text_lower:
            score += 0.1
            reasons.append(f"environment '{entry['environment']}' in text")

        # 4. Keyword overlap between description and paper text
        if desc_lower:
            desc_words = set(desc_lower.split())
            # Filter to meaningful words (>4 chars, not common)
            _COMMON = {"that", "this", "with", "from", "have", "been", "also", "type",
                       "they", "their", "which", "where", "these", "those", "into",
                       "deposit", "deposits", "mineral", "known", "typically"}
            desc_keywords = {w for w in desc_words if len(w) > 4 and w not in _COMMON}
            if desc_keywords:
                overlap = sum(1 for kw in desc_keywords if kw in text_lower)
                kw_score = min(0.2, overlap / len(desc_keywords) * 0.3)
                if kw_score > 0.05:
                    score += kw_score
                    reasons.append(f"{overlap}/{len(desc_keywords)} description keywords match")

        # 5. Commodity match
        for comm in commodity_set:
            if comm in name_lower or comm in desc_lower:
                score += 0.1
                reasons.append(f"commodity '{comm}' matches")
                break

        # 6. Deposit name similarity
        if deposit_lower:
            if any(w in deposit_lower for w in name_lower.split() if len(w) > 3):
                score += 0.1
                reasons.append("deposit name overlap")

        if score > 0.05:
            scored.append({
                "name": entry["name"],
                "environment": entry["environment"],
                "group": entry["group"],
                "minmod_id": entry["minmod_id"],
                "score": round(min(1.0, score), 3),
                "reason": "; ".join(reasons),
            })

    scored.sort(key=lambda x: -x["score"])
    return scored[:10]  # Top 10


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


# ──────────────────────────────────────────────────────────────────────────────
# USGS Picklist — authoritative valid values for schema fields
# Loaded from Pickelist.xlsx at import time; used as LLM reference
# ──────────────────────────────────────────────────────────────────────────────

_USGS_PICKLIST: dict[str, list[str]] = {}
_PICKLIST_LOADED = False


def _load_usgs_picklist() -> dict[str, list[str]]:
    """Load the USGS picklist from Pickelist.xlsx (one-time, cached)."""
    global _USGS_PICKLIST, _PICKLIST_LOADED
    if _PICKLIST_LOADED:
        return _USGS_PICKLIST
    try:
        import pandas as pd
        from pathlib import Path
        picklist_path = Path(__file__).parent / "Pickelist.xlsx"
        if not picklist_path.exists():
            _PICKLIST_LOADED = True
            return _USGS_PICKLIST
        df = pd.read_excel(picklist_path, sheet_name="Sheet1")
        for col in df.columns:
            vals = sorted(df[col].dropna().unique().astype(str).tolist())
            if vals:
                _USGS_PICKLIST[col] = vals
        _PICKLIST_LOADED = True
    except Exception:
        _PICKLIST_LOADED = True
    return _USGS_PICKLIST


def get_usgs_picklist_prompt() -> str:
    """Format the USGS picklist as a reference section for LLM prompts.

    Includes the most relevant fields (deposit, mineral, method, material).
    Large fields (country, state, earth_material) are summarised to save tokens.
    """
    picklist = _load_usgs_picklist()
    if not picklist:
        return ""

    # Fields to include in full (compact enough for prompt)
    full_fields = [
        "deposit_environment", "deposit_group", "feature_type",
        "sample_deposit_relation", "sample_type", "sampling_method",
        "material_class", "Earth_material_group", "Metamorphic_grade",
        "Mode of Occurrence", "color", "alteration", "analyzed_material",
        "analytical_method", "location_source",
    ]
    # Fields to include as sample (too large for full listing)
    sample_fields = {
        "deposit_type": 30,
        "Mineral": 50,
        "earth_material": 20,
    }

    lines = [
        "USGS PICKLIST REFERENCE (authoritative valid values from CMMI database):",
        "When extracting metadata, prefer values from this list. Other values are",
        "acceptable if the paper uses specific terminology not covered here.",
        "",
    ]

    for field in full_fields:
        if field in picklist:
            vals = picklist[field]
            lines.append(f"  {field} ({len(vals)} valid):")
            lines.append(f"    {', '.join(vals)}")
            lines.append("")

    for field, limit in sample_fields.items():
        if field in picklist:
            vals = picklist[field]
            shown = vals[:limit]
            lines.append(f"  {field} ({len(vals)} valid, showing {len(shown)}):")
            lines.append(f"    {', '.join(shown)}")
            if len(vals) > limit:
                lines.append(f"    ... and {len(vals) - limit} more")
            lines.append("")

    return "\n".join(lines)


def validate_against_picklist(field_name: str, value: str) -> tuple[bool, Optional[str]]:
    """Check if a value is in the USGS picklist for a given field.

    Returns (is_valid, suggestion) where suggestion is the closest match
    if the value is not exactly in the list.
    """
    picklist = _load_usgs_picklist()
    # Map schema field names to picklist column names
    field_map = {
        "deposit_environment": "deposit_environment",
        "deposit_group": "deposit_group",
        "deposit_type": "deposit_type",
        "feature_type": "feature_type",
        "sample_deposit_relation": "sample_deposit_relation",
        "sample_type": "sample_type",
        "sampling_method": "sampling_method",
        "material_class": "material_class",
        "earth_material_group": "Earth_material_group",
        "metamorphic_grade": "Metamorphic_grade",
        "mineral": "Mineral",
        "mode_of_occurrence": "Mode of Occurrence",
        "color": "color",
        "alteration": "alteration",
        "analytical_method": "analytical_method",
        "country": "country",
    }

    picklist_col = field_map.get(field_name)
    if not picklist_col or picklist_col not in picklist:
        return True, None  # No picklist for this field

    valid_values = picklist[picklist_col]
    value_lower = value.strip().lower()

    # Exact match (case-insensitive)
    for v in valid_values:
        if v.lower() == value_lower:
            return True, v  # Return canonical casing

    # Fuzzy match — find closest
    best_match = None
    best_score = 0.0
    for v in valid_values:
        # Simple containment check
        v_lower = v.lower()
        if value_lower in v_lower or v_lower in value_lower:
            score = len(min(value_lower, v_lower, key=len)) / len(max(value_lower, v_lower, key=len))
            if score > best_score:
                best_score = score
                best_match = v

    if best_match and best_score > 0.5:
        return False, best_match
    return False, None


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
        get_usgs_picklist_prompt(),
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
