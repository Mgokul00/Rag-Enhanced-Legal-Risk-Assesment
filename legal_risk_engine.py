"""
legal_risk_engine.py
────────────────────
Two new features for the Legal Due Diligence RAG System:


FEATURE 1 — Threat Inventory with Document References
  • Counts concrete threats per risk category
  • Locates each threat to a page number AND line span in the original document
  • Produces a machine-readable dict suitable for CLI printing AND REST API JSON


FEATURE 2 — Legal Risk Index (LRI) via Weighted Aggregation
  • Algorithm (no LLM scoring):
      LRI = Σ  w_i × P_i   for each risk category i
      P_i is derived from three objective signals:
        a) keyword_density   – fraction of category keywords found in the deed text
        b) rag_similarity    – normalised cosine-distance score returned by ChromaDB
        c) threat_density    – threats-per-1000-words for the category
      These three sub-scores are combined with fixed intra-category weights.
  • Domain weights w_i reflect legal severity (title > litigation > tax > construction > govt)
  • Output: LRI ∈ [0, 100], band label (Low / Medium / High / Critical)
"""


from __future__ import annotations


import re
import math
import json
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple




# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────


# Domain weights  w_i  (must sum to 1.0)
CATEGORY_WEIGHTS: Dict[str, float] = {
    "Ownership Disputes":       0.30,
    "Litigation History":       0.25,
    "Tax Obligations":          0.20,
    "Construction Compliance":  0.15,
    "Government Notifications": 0.10,
}


# Intra-category sub-score weights  (must sum to 1.0)
# gemini_severity is added as a 4th signal when categorized_risks is available.
# When it is NOT available the remaining three are renormalised automatically.
SUBSCORE_WEIGHTS = {
    "keyword_density":   0.30,   # fraction of risk keywords found in deed text
    "rag_similarity":    0.25,   # ChromaDB cosine similarity to known risk provisions
    "threat_density":    0.15,   # threats per 1000 words (normalised)
    "gemini_severity":   0.30,   # severity level extracted from Gemini's own analysis
}


# Weights used when Gemini analysis is NOT yet available (3-signal fallback)
SUBSCORE_WEIGHTS_NO_GEMINI = {
    "keyword_density":  0.45,
    "rag_similarity":   0.35,
    "threat_density":   0.20,
}


# Keyword lists per category (from LegalDueDiligenceSystem.RISK_CATEGORIES)
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Ownership Disputes": [
        "unclear title", "title chain", "multiple claimants", "succession",
        "partition", "joint ownership", "co-owner", "disputed", "heirship",
        "intestate", "testamentary", "probate", "gift deed", "nominee",
    ],
    "Litigation History": [
        "pending case", "court order", "injunction", "stay order",
        "criminal proceeding", "civil suit", "decree", "writ petition",
        "arbitration", "encumbrance", "lis pendens", "attachment",
    ],
    "Construction Compliance": [
        "building plan", "occupancy certificate", "completion certificate",
        "unauthorized construction", "setback", "FSI", "FAR", "floor area",
        "RERA", "deviation", "compounding", "building bye-law",
        "planning permission", "demolition notice",
    ],
    "Tax Obligations": [
        "property tax", "stamp duty", "registration charge",
        "capital gains", "TDS", "tax arrear", "dues", "penalty",
        "assessment", "surcharge", "municipal tax", "wealth tax",
    ],
    "Government Notifications": [
        "acquisition", "land acquisition", "demolition order",
        "land use change", "town planning", "road widening",
        "reservation", "public purpose", "notification", "gazette",
        "compulsory acquisition", "development plan",
    ],
}


# Threat-indicator patterns — regex phrases that signal a concrete threat
THREAT_PATTERNS: Dict[str, List[str]] = {
    "Ownership Disputes": [
        r"title\s+(?:not\s+)?(?:clear|verified|established)",
        r"(?:multiple|joint|disputed)\s+owner",
        r"succession\s+(?:dispute|issue|not\s+resolved)",
        r"partition\s+(?:pending|not\s+complete)",
        r"heirship\s+(?:not|un)(?:settled|resolved|established)",
        r"gift\s+deed\s+(?:challenged|disputed|invalid)",
        r"probate\s+(?:not|pending)",
    ],
    "Litigation History": [
        r"(?:case|suit|petition)\s+(?:filed|pending|active)",
        r"injunction\s+(?:order|granted|in\s+force)",
        r"stay\s+order\s+(?:granted|operative|in\s+place)",
        r"encumbrance\s+(?:found|detected|noted|present)",
        r"lis\s+pendens",
        r"attachment\s+(?:order|levied)",
        r"criminal\s+(?:case|proceeding|complaint)\s+(?:filed|pending)",
    ],
    "Construction Compliance": [
        r"(?:building\s+plan|plan\s+approval)\s+(?:not|missing|absent|unavailable)",
        r"occupancy\s+certificate\s+(?:not|missing|absent|not\s+obtained)",
        r"completion\s+certificate\s+(?:not|missing|absent)",
        r"unauthorized\s+(?:construction|floor|addition|structure)",
        r"(?:setback|FSI|FAR)\s+(?:violation|non.?complian|exceed)",
        r"RERA\s+(?:not\s+registered|violation|non.?complian)",
        r"demolition\s+(?:notice|order|threat)",
    ],
    "Tax Obligations": [
        r"property\s+tax\s+(?:arrear|due|unpaid|outstanding)",
        r"stamp\s+duty\s+(?:short|under.?paid|not\s+paid|deficien)",
        r"TDS\s+(?:not\s+deducted|not\s+deposited|default)",
        r"capital\s+gains\s+(?:tax\s+)?(?:not\s+paid|liability|due)",
        r"(?:tax|dues|penalty)\s+(?:outstanding|arrear|unpaid)",
        r"registration\s+(?:charges?\s+)?(?:short|under.?paid|unpaid)",
    ],
    "Government Notifications": [
        r"(?:land\s+)?acquisition\s+(?:notice|notification|proceeding)",
        r"demolition\s+order\s+(?:issued|pending|served)",
        r"land\s+use\s+(?:change|conversion)\s+(?:pending|not\s+sanctioned)",
        r"road\s+widening\s+(?:proposed|affected|notified)",
        r"town\s+planning\s+(?:scheme|reservation|notification)",
        r"gazette\s+(?:notification|notif)",
        r"compulsory\s+acquisition",
    ],
}


LRI_BANDS = [
    (0,   20,  "Low",      "green",  "Minimal legal risk. Safe to proceed with standard due diligence."),
    (20,  45,  "Medium",   "yellow", "Moderate concerns identified. Address flagged items before proceeding."),
    (45,  70,  "High",     "orange", "Significant risks present. Legal remediation required before registration."),
    (70,  100, "Critical", "red",    "Severe legal exposure. Do NOT proceed without complete legal resolution."),
]




# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ThreatReference:
    """A single located threat instance inside the sale deed."""
    threat_id: str                  # e.g. "OD-001"
    category: str
    pattern_matched: str            # human-readable description of what was found
    matched_text: str               # the actual snippet from the document (≤120 chars)
    page_number: int                # 1-based
    line_start: int                 # 1-based, absolute line in full text
    line_end: int                   # inclusive
    char_offset_start: int          # byte offset for deep-linking
    char_offset_end: int
    context_snippet: str            # ±1 sentence for readability


    def to_dict(self) -> dict:
        return asdict(self)




@dataclass
class CategoryThreatSummary:
    category: str
    weight: float                            # w_i
    threat_count: int
    threats: List[ThreatReference] = field(default_factory=list)
    keyword_density: float = 0.0             # P_i sub-score a
    rag_similarity_score: float = 0.0        # P_i sub-score b  (0-1)
    threat_density_score: float = 0.0        # P_i sub-score c  (0-1 normalised)
    gemini_severity_score: float = 0.0       # P_i sub-score d  (parsed from Gemini text)
    gemini_severity_label: str = "N/A"       # raw label: High / Medium / Low / None / N/A
    probability: float = 0.0                # combined P_i
    weighted_contribution: float = 0.0      # w_i × P_i




@dataclass
class LegalRiskIndex:
    lri_score: float                        # 0-100
    band: str
    band_color: str
    band_description: str
    categories: Dict[str, CategoryThreatSummary]
    total_threats: int
    word_count: int
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())


    def to_dict(self) -> dict:
        d = {
            "lri_score": round(self.lri_score, 2),
            "band": self.band,
            "band_color": self.band_color,
            "band_description": self.band_description,
            "total_threats": self.total_threats,
            "word_count": self.word_count,
            "computed_at": self.computed_at,
            "categories": {},
        }
        for cat_name, cat in self.categories.items():
            d["categories"][cat_name] = {
                "weight": cat.weight,
                "threat_count": cat.threat_count,
                "keyword_density": round(cat.keyword_density, 4),
                "rag_similarity_score": round(cat.rag_similarity_score, 4),
                "gemini_severity_score": round(cat.gemini_severity_score, 4),
                "gemini_severity_label": cat.gemini_severity_label,
                "threat_density_score": round(cat.threat_density_score, 4),
                "probability": round(cat.probability, 4),
                "weighted_contribution": round(cat.weighted_contribution, 4),
                "threats": [t.to_dict() for t in cat.threats],
            }
        return d




# ─────────────────────────────────────────────────────────────────────────────
# HELPER: split text into pages and lines
# ─────────────────────────────────────────────────────────────────────────────


def _build_line_index(text: str) -> List[Tuple[int, int, int]]:
    """
    Returns list of (line_number_1based, char_start, char_end) for every line.
    """
    index = []
    pos = 0
    for lineno, line in enumerate(text.splitlines(keepends=True), start=1):
        index.append((lineno, pos, pos + len(line)))
        pos += len(line)
    return index




def _build_page_index(text: str, chars_per_page: int = 3000) -> List[Tuple[int, int, int]]:
    """
    Approximate page boundaries (used when we don't have real PDF page breaks).
    Returns list of (page_number_1based, char_start, char_end).
    """
    pages = []
    total = len(text)
    page = 1
    for start in range(0, total, chars_per_page):
        end = min(start + chars_per_page, total)
        pages.append((page, start, end))
        page += 1
    return pages




def _char_to_line(char_pos: int, line_index: List[Tuple[int, int, int]]) -> int:
    """Binary-search the line index for a character offset."""
    lo, hi = 0, len(line_index) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        lineno, start, end = line_index[mid]
        if char_pos < start:
            hi = mid - 1
        elif char_pos >= end:
            lo = mid + 1
        else:
            return lineno
    return len(line_index)   # fallback: last line




def _char_to_page(
    char_pos: int,
    page_index: List[Tuple[int, int, int]],
    real_page_breaks: Optional[List[int]] = None,
) -> int:
    """Return 1-based page number for a character offset."""
    if real_page_breaks:
        for page_no, break_pos in enumerate(real_page_breaks, start=2):
            if char_pos < break_pos:
                return page_no - 1
        return len(real_page_breaks) + 1


    for page_no, start, end in page_index:
        if start <= char_pos < end:
            return page_no
    return page_index[-1][0]




def _extract_context(text: str, match_start: int, match_end: int, window: int = 150) -> str:
    """Return ±window chars around a match, clipped to sentence boundaries."""
    ctx_start = max(0, match_start - window)
    ctx_end = min(len(text), match_end + window)
    snippet = text[ctx_start:ctx_end].strip()
    # clip to nearest sentence boundary
    if ctx_start > 0:
        first_dot = snippet.find('. ')
        if first_dot != -1:
            snippet = snippet[first_dot + 2:]
    if ctx_end < len(text):
        last_dot = snippet.rfind('. ')
        if last_dot != -1:
            snippet = snippet[:last_dot + 1]
    return snippet.replace('\n', ' ').strip()




# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: Threat Inventory
# ─────────────────────────────────────────────────────────────────────────────


_CATEGORY_PREFIX = {
    "Ownership Disputes":       "OD",
    "Litigation History":       "LH",
    "Construction Compliance":  "CC",
    "Tax Obligations":          "TO",
    "Government Notifications": "GN",
}


_PATTERN_LABELS: Dict[str, Dict[str, str]] = {
    "Ownership Disputes": {
        r"title\s+(?:not\s+)?(?:clear|verified|established)":
            "Title clarity / verification issue",
        r"(?:multiple|joint|disputed)\s+owner":
            "Multiple or disputed ownership",
        r"succession\s+(?:dispute|issue|not\s+resolved)":
            "Unresolved succession dispute",
        r"partition\s+(?:pending|not\s+complete)":
            "Incomplete property partition",
        r"heirship\s+(?:not|un)(?:settled|resolved|established)":
            "Unestablished heirship",
        r"gift\s+deed\s+(?:challenged|disputed|invalid)":
            "Challenged gift deed in chain",
        r"probate\s+(?:not|pending)":
            "Pending or absent probate",
    },
    "Litigation History": {
        r"(?:case|suit|petition)\s+(?:filed|pending|active)":
            "Active legal case / suit",
        r"injunction\s+(?:order|granted|in\s+force)":
            "Injunction order in force",
        r"stay\s+order\s+(?:granted|operative|in\s+place)":
            "Stay order operative",
        r"encumbrance\s+(?:found|detected|noted|present)":
            "Encumbrance detected",
        r"lis\s+pendens":
            "Lis pendens (suit pending on land)",
        r"attachment\s+(?:order|levied)":
            "Attachment order levied",
        r"criminal\s+(?:case|proceeding|complaint)\s+(?:filed|pending)":
            "Criminal proceedings filed",
    },
    "Construction Compliance": {
        r"(?:building\s+plan|plan\s+approval)\s+(?:not|missing|absent|unavailable)":
            "Building plan approval absent",
        r"occupancy\s+certificate\s+(?:not|missing|absent|not\s+obtained)":
            "Occupancy Certificate missing",
        r"completion\s+certificate\s+(?:not|missing|absent)":
            "Completion Certificate absent",
        r"unauthorized\s+(?:construction|floor|addition|structure)":
            "Unauthorized construction / addition",
        r"(?:setback|FSI|FAR)\s+(?:violation|non.?complian|exceed)":
            "Setback / FSI / FAR violation",
        r"RERA\s+(?:not\s+registered|violation|non.?complian)":
            "RERA non-compliance",
        r"demolition\s+(?:notice|order|threat)":
            "Demolition notice issued",
    },
    "Tax Obligations": {
        r"property\s+tax\s+(?:arrear|due|unpaid|outstanding)":
            "Property tax arrears outstanding",
        r"stamp\s+duty\s+(?:short|under.?paid|not\s+paid|deficien)":
            "Stamp duty deficiency",
        r"TDS\s+(?:not\s+deducted|not\s+deposited|default)":
            "TDS not deducted / deposited",
        r"capital\s+gains\s+(?:tax\s+)?(?:not\s+paid|liability|due)":
            "Capital gains tax liability",
        r"(?:tax|dues|penalty)\s+(?:outstanding|arrear|unpaid)":
            "Outstanding tax dues / penalties",
        r"registration\s+(?:charges?\s+)?(?:short|under.?paid|unpaid)":
            "Registration charges under-paid",
    },
    "Government Notifications": {
        r"(?:land\s+)?acquisition\s+(?:notice|notification|proceeding)":
            "Land acquisition notice served",
        r"demolition\s+order\s+(?:issued|pending|served)":
            "Demolition order issued",
        r"land\s+use\s+(?:change|conversion)\s+(?:pending|not\s+sanctioned)":
            "Unsanctioned land use change",
        r"road\s+widening\s+(?:proposed|affected|notified)":
            "Road widening notification",
        r"town\s+planning\s+(?:scheme|reservation|notification)":
            "Town planning reservation",
        r"gazette\s+(?:notification|notif)":
            "Gazette notification present",
        r"compulsory\s+acquisition":
            "Compulsory acquisition proceedings",
    },
}




def scan_threats(
    text: str,
    real_page_breaks: Optional[List[int]] = None,
) -> Dict[str, List[ThreatReference]]:
    """
    Scans the full sale deed text for every threat pattern.
    Returns {category: [ThreatReference, ...]} — de-duplicated by char offset.
    """
    line_index = _build_line_index(text)
    page_index = _build_page_index(text)
    lower_text = text.lower()


    all_threats: Dict[str, List[ThreatReference]] = {
        cat: [] for cat in CATEGORY_KEYWORDS
    }
    counters = {cat: 1 for cat in CATEGORY_KEYWORDS}


    for category, patterns in THREAT_PATTERNS.items():
        seen_spans: List[Tuple[int, int]] = []


        for raw_pattern in patterns:
            label = _PATTERN_LABELS[category].get(raw_pattern, raw_pattern)
            compiled = re.compile(raw_pattern, re.IGNORECASE)


            for m in compiled.finditer(lower_text):
                ms, me = m.start(), m.end()


                # De-duplicate: skip if overlapping with already-recorded threat
                overlap = any(
                    not (me <= ss or ms >= se)
                    for ss, se in seen_spans
                )
                if overlap:
                    continue
                seen_spans.append((ms, me))


                line_start = _char_to_line(ms, line_index)
                line_end   = _char_to_line(me, line_index)
                page_no    = _char_to_page(ms, page_index, real_page_breaks)
                snippet    = text[ms:me]
                context    = _extract_context(text, ms, me)
                prefix     = _CATEGORY_PREFIX[category]
                tid        = f"{prefix}-{counters[category]:03d}"
                counters[category] += 1


                all_threats[category].append(ThreatReference(
                    threat_id=tid,
                    category=category,
                    pattern_matched=label,
                    matched_text=snippet[:120],
                    page_number=page_no,
                    line_start=line_start,
                    line_end=line_end,
                    char_offset_start=ms,
                    char_offset_end=me,
                    context_snippet=context,
                ))


        # Sort by position in document
        all_threats[category].sort(key=lambda t: t.char_offset_start)


    return all_threats




# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: LRI computation
# ─────────────────────────────────────────────────────────────────────────────


def _keyword_density_score(text_lower: str, keywords: List[str]) -> float:
    """
    Fraction of category keywords that appear at least once in the deed text.
    Returns a value in [0, 1].
    """
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)




def _rag_similarity_score(
    category: str,
    vectorizer,
    legal_db,
    rag_queries: List[str],
    n_results: int = 5,
) -> float:
    """
    Uses ChromaDB's returned distances (L2 or cosine) and converts to similarity.
    ChromaDB returns distances in results['distances']; lower = more similar.
    Average the top-n distances across all category queries, then normalise to [0,1].


    Score interpretation: higher => deed text is MORE similar to known risk provisions
    => higher risk probability for this category.
    """
    if legal_db is None or vectorizer is None:
        return 0.0


    raw_distances = []
    for query in rag_queries:
        try:
            results = vectorizer.query_database(legal_db, query, n_results=n_results)
            dists = results.get('distances', [[]])[0]
            raw_distances.extend(dists)
        except Exception:
            pass


    if not raw_distances:
        return 0.0


    # ChromaDB L2 distances are unbounded; we cap at 2.0 for normalisation
    # similarity = 1 - (distance / max_distance)
    MAX_DIST = 2.0
    similarities = [max(0.0, 1.0 - (d / MAX_DIST)) for d in raw_distances]
    return min(1.0, sum(similarities) / len(similarities))




def _threat_density_score(threat_count: int, word_count: int) -> float:
    """
    Threats per 1000 words, normalised to [0, 1].
    Cap at 10 threats/1000 words for full score.
    """
    if word_count == 0:
        return 0.0
    rate = (threat_count / word_count) * 1000
    return min(1.0, rate / 10.0)




# Severity label → numeric score mapping (algorithmic, no LLM call)
_SEVERITY_SCORES = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.2,
    "none":   0.0,
}


_SEVERITY_PATTERN = re.compile(
    r'\b(severity|risk|level|rating)\s*(?:level|rating|is|:)?\s*'
    r'(high|medium|moderate|low|minimal|none|no\s+risk)',
    re.IGNORECASE
)
_SEVERITY_LEAD_PATTERN = re.compile(
    r'^\s*(?:overall\s+)?(?:risk\s+)?(?:severity\s+)?(?:level\s+)?'
    r'(?:is\s+)?(?:rated?\s+(?:as\s+)?)?'
    r'(high|medium|moderate|low|minimal|none)',
    re.IGNORECASE | re.MULTILINE
)




def _gemini_severity_score(analysis_text: str) -> Tuple[float, str]:
    """
    Parses Gemini's written risk analysis for a severity label entirely via
    regex — no LLM call.  Returns (numeric_score ∈ [0,1], raw_label).


    Strategy (in priority order):
      1. Look for explicit "severity level: High" style phrases.
      2. Look for the label at the very start of the first paragraph
         (Gemini is instructed to state severity in the first sentence).
      3. Count High/Medium/Low mentions and take the dominant one.
    """
    if not analysis_text:
        return 0.0, "N/A"


    text = analysis_text.strip()


    # Strategy 1: explicit phrase
    m = _SEVERITY_PATTERN.search(text)
    if m:
        raw = m.group(2).lower()
        raw = raw.replace("moderate", "medium").replace("minimal", "low").replace("no risk", "none")
        return _SEVERITY_SCORES.get(raw, 0.0), raw.capitalize()


    # Strategy 2: first sentence / paragraph lead
    first_para = text.split('\n\n')[0][:300]
    m = _SEVERITY_LEAD_PATTERN.search(first_para)
    if m:
        raw = m.group(1).lower()
        raw = raw.replace("moderate", "medium").replace("minimal", "low")
        return _SEVERITY_SCORES.get(raw, 0.0), raw.capitalize()


    # Strategy 3: dominant mention count
    counts = {}
    for label in ("high", "medium", "low", "none"):
        counts[label] = len(re.findall(r'\b' + label + r'\b', text, re.IGNORECASE))
    dominant = max(counts, key=counts.get)
    if counts[dominant] > 0:
        return _SEVERITY_SCORES.get(dominant, 0.0), dominant.capitalize()


    return 0.0, "N/A"




def _combine_subscores(
    keyword_density: float,
    rag_similarity: float,
    threat_density: float,
    gemini_severity: Optional[float] = None,
) -> float:
    """
    Weighted combination → P_i in [0,1].
    Uses 4-signal weights when gemini_severity is provided,
    falls back to 3-signal weights otherwise.
    """
    if gemini_severity is not None:
        return (
            SUBSCORE_WEIGHTS["keyword_density"]  * keyword_density  +
            SUBSCORE_WEIGHTS["rag_similarity"]   * rag_similarity   +
            SUBSCORE_WEIGHTS["threat_density"]   * threat_density   +
            SUBSCORE_WEIGHTS["gemini_severity"]  * gemini_severity
        )
    else:
        return (
            SUBSCORE_WEIGHTS_NO_GEMINI["keyword_density"]  * keyword_density  +
            SUBSCORE_WEIGHTS_NO_GEMINI["rag_similarity"]   * rag_similarity   +
            SUBSCORE_WEIGHTS_NO_GEMINI["threat_density"]   * threat_density
        )




def _lri_band(score: float) -> Tuple[str, str, str]:
    """Return (band_label, color, description) for a given LRI score."""
    for lo, hi, label, color, desc in LRI_BANDS:
        if lo <= score < hi:
            return label, color, desc
    return "Critical", "red", LRI_BANDS[-1][4]




def compute_lri(
    text: str,
    threat_map: Dict[str, List[ThreatReference]],
    vectorizer=None,
    legal_db=None,
    rag_queries_per_category: Optional[Dict[str, List[str]]] = None,
    categorized_risks: Optional[Dict] = None,
) -> LegalRiskIndex:
    """
    Computes the Legal Risk Index using the formula:
        LRI = Σ  w_i × P_i × 100


    P_i is algorithmically derived from up to 4 signals:
      a) keyword_density   — fraction of category risk-keywords found in deed text
      b) rag_similarity    — ChromaDB cosine proximity to known risk provisions
      c) threat_density    — threats per 1000 words, normalised
      d) gemini_severity   — severity label (High/Medium/Low/None) extracted by
                             regex from Gemini's written analysis; parsed here
                             algorithmically, NOT by another LLM call.


    When categorized_risks is None (Gemini not yet run), signal (d) is omitted
    and weights fall back to the 3-signal set.
    """
    text_lower = text.lower()
    word_count = len(text.split())


    if rag_queries_per_category is None:
        rag_queries_per_category = {
            "Ownership Disputes": [
                "title deed ownership chain verification",
                "property succession inheritance law India",
            ],
            "Litigation History": [
                "property litigation pending court cases",
                "injunction stay order property transaction",
            ],
            "Construction Compliance": [
                "building plan approval RERA compliance",
                "occupancy certificate completion certificate",
            ],
            "Tax Obligations": [
                "property tax stamp duty registration charges",
                "capital gains tax TDS property sale",
            ],
            "Government Notifications": [
                "government land acquisition notice property",
                "town planning scheme road widening",
            ],
        }


    categories: Dict[str, CategoryThreatSummary] = {}
    lri_accumulator = 0.0
    total_threats = 0


    for category in CATEGORY_WEIGHTS:
        weight = CATEGORY_WEIGHTS[category]
        keywords = CATEGORY_KEYWORDS.get(category, [])
        threats = threat_map.get(category, [])
        threat_count = len(threats)
        total_threats += threat_count


        # Sub-score a: keyword density
        kd = _keyword_density_score(text_lower, keywords)


        # Sub-score b: RAG similarity (0 if no DB)
        qs = rag_queries_per_category.get(category, [])
        rs = _rag_similarity_score(category, vectorizer, legal_db, qs)


        # Sub-score c: threat density
        td = _threat_density_score(threat_count, word_count)


        # Sub-score d: Gemini severity (only when analysis is available)
        gs: Optional[float] = None
        gs_label = "N/A"
        if categorized_risks and category in categorized_risks:
            analysis_text = categorized_risks[category].get("analysis", "")
            gs, gs_label = _gemini_severity_score(analysis_text)


        # Combined probability P_i
        pi = _combine_subscores(kd, rs, td, gs)


        # Weighted contribution w_i × P_i
        wi_pi = weight * pi
        lri_accumulator += wi_pi


        categories[category] = CategoryThreatSummary(
            category=category,
            weight=weight,
            threat_count=threat_count,
            threats=threats,
            keyword_density=kd,
            rag_similarity_score=rs,
            threat_density_score=td,
            gemini_severity_score=gs if gs is not None else 0.0,
            gemini_severity_label=gs_label,
            probability=pi,
            weighted_contribution=wi_pi,
        )


    # Scale to 0-100
    lri_score = lri_accumulator * 100.0
    band_label, band_color, band_desc = _lri_band(lri_score)


    return LegalRiskIndex(
        lri_score=lri_score,
        band=band_label,
        band_color=band_color,
        band_description=band_desc,
        categories=categories,
        total_threats=total_threats,
        word_count=word_count,
    )




# ─────────────────────────────────────────────────────────────────────────────
# CLI PRINTER
# ─────────────────────────────────────────────────────────────────────────────


_BAND_CHARS = {
    "Low":      "░",
    "Medium":   "▒",
    "High":     "▓",
    "Critical": "█",
}


_BAND_ANSI = {
    "Low":      "\033[92m",   # green
    "Medium":   "\033[93m",   # yellow
    "High":     "\033[33m",   # dark-yellow / orange
    "Critical": "\033[91m",   # red
}
_RESET = "\033[0m"




def _bar(value: float, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    filled = round(value * width)
    return fill * filled + empty * (width - filled)




def print_lri_cli(lri: LegalRiskIndex) -> None:
    """
    Pretty-prints the LRI result and threat inventory to stdout.
    Designed to look clean in both terminal and captured log output.
    """
    w = 72
    sep = "─" * w
    bold = "\033[1m"
    dim  = "\033[2m"
    R    = _RESET


    band_color = _BAND_ANSI.get(lri.band, "")
    fill_char  = _BAND_CHARS.get(lri.band, "█")


    print()
    print("═" * w)
    print(f"{bold}  LEGAL RISK INDEX (LRI) REPORT{R}")
    print(f"  Generated: {lri.computed_at[:19].replace('T', ' ')}")
    print("═" * w)


    # LRI score bar
    bar_filled = round((lri.lri_score / 100) * 50)
    bar_str = fill_char * bar_filled + "░" * (50 - bar_filled)
    print()
    print(f"  {bold}Overall LRI Score{R}")
    print(f"  [{bar_str}]  {band_color}{bold}{lri.lri_score:.1f} / 100{R}")
    print(f"  Risk Band : {band_color}{bold}{lri.band.upper()}{R}")
    print(f"  Assessment: {lri.band_description}")
    print(f"  Total Threats Found : {bold}{lri.total_threats}{R}")
    print(f"  Document Word Count : {lri.word_count:,}")
    print()
    print(sep)
    print(f"  {bold}CATEGORY BREAKDOWN{R}   (LRI = Σ w_i × P_i)")
    print(sep)
    print(f"  {'Category':<28} {'w_i':>5}  {'P_i':>6}  {'Contribution':>12}  {'Threats':>7}")
    print(sep)


    for cat_name, cat in lri.categories.items():
        short = cat_name[:27]
        bar_p = _bar(cat.probability, width=14)
        print(
            f"  {short:<28} {cat.weight:>5.2f}  {cat.probability:>5.1%}  "
            f"{cat.weighted_contribution:>10.4f}    {cat.threat_count:>4}"
        )


    print(sep)
    print(f"  {'TOTAL LRI (×100)':>54}  {lri.lri_score:>7.2f}")
    print()


    # Sub-score detail
    print(sep)
    print(f"  {bold}SUB-SCORE DETAIL{R}  "
          f"(KD=keyword density · RS=RAG similarity · TD=threat density · GS=Gemini severity)")
    print(sep)
    print(f"  {'Category':<28}  {'KD':>6}  {'RS':>6}  {'TD':>6}  {'GS':>6}  {'Label':<8}  {'P_i':>6}")
    print(sep)
    for cat_name, cat in lri.categories.items():
        short = cat_name[:27]
        print(
            f"  {short:<28}  "
            f"{cat.keyword_density:>5.1%}  "
            f"{cat.rag_similarity_score:>5.1%}  "
            f"{cat.threat_density_score:>5.1%}  "
            f"{cat.gemini_severity_score:>5.1%}  "
            f"{cat.gemini_severity_label:<8}  "
            f"{cat.probability:>5.1%}"
        )
    print()


    # Per-category threat detail
    print("═" * w)
    print(f"{bold}  THREAT INVENTORY — DOCUMENT REFERENCES{R}")
    print("═" * w)


    for cat_name, cat in lri.categories.items():
        print()
        print(f"  {bold}▶ {cat_name}{R}  — {cat.threat_count} threat(s) found")
        print(f"  {'─'*68}")


        if not cat.threats:
            print(f"  {dim}  No threat indicators detected for this category.{R}")
            continue


        for t in cat.threats:
            print(f"  [{t.threat_id}]  {bold}{t.pattern_matched}{R}")
            print(f"         Page : {t.page_number}   "
                  f"Lines : {t.line_start}–{t.line_end}   "
                  f"Chars : {t.char_offset_start}–{t.char_offset_end}")
            print(f"         Matched : \"{t.matched_text}\"")
            wrapped = textwrap.fill(
                t.context_snippet, width=62,
                initial_indent="         Context : ",
                subsequent_indent="                   "
            )
            print(wrapped)
            print()


    print("═" * w)
    print(f"  {bold}Formula:{R}  LRI = Σ (w_i × P_i) × 100")
    print(f"  {dim}  P_i = 0.30×KD + 0.25×RS + 0.15×TD + 0.30×GS  (when Gemini analysis available){R}")
    print(f"  {dim}  P_i = 0.45×KD + 0.35×RS + 0.20×TD             (fallback, no Gemini){R}")
    print(f"  {dim}  Domain weights: Ownership 30% · Litigation 25% · Tax 20% · Construction 15% · Government 10%{R}")
    print("═" * w)
    print()




# ─────────────────────────────────────────────────────────────────────────────
# API / JSON OUTPUT  (for web backend integration)
# ─────────────────────────────────────────────────────────────────────────────


def lri_to_api_response(lri: LegalRiskIndex) -> dict:
    """
    Returns a structured dict ready to be serialised as JSON by a web framework
    (FastAPI, Flask, Django REST, etc.).


    Shape designed for a dashboard with:
      - summary card (lri_score, band, total_threats)
      - radial / bar chart (categories[].probability, weighted_contribution)
      - threat table (categories[].threats[])
      - sub-score radar (keyword_density, rag_similarity_score, threat_density_score)
    """
    d = lri.to_dict()


    # Add a flat list of all threats for easy table rendering on the frontend
    all_threats = []
    for cat_name, cat_data in d["categories"].items():
        for threat in cat_data["threats"]:
            all_threats.append({
                **threat,
                "category": cat_name,
            })
    all_threats.sort(key=lambda t: t["char_offset_start"])
    d["all_threats_flat"] = all_threats


    # Summary metrics the frontend might want at the top level
    d["summary"] = {
        "lri_score": d["lri_score"],
        "band": d["band"],
        "band_color": d["band_color"],
        "band_description": d["band_description"],
        "total_threats": d["total_threats"],
        "word_count": d["word_count"],
        "computed_at": d["computed_at"],
        "category_scores": {
            cat: {
                "threat_count": vals["threat_count"],
                "probability_pct": round(vals["probability"] * 100, 1),
                "weighted_contribution_pct": round(vals["weighted_contribution"] * 100, 1),
            }
            for cat, vals in d["categories"].items()
        },
    }


    return d




def lri_to_json(lri: LegalRiskIndex, indent: int = 2) -> str:
    """Serialise the full LRI result to a JSON string."""
    return json.dumps(lri_to_api_response(lri), indent=indent, ensure_ascii=False)




# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT — called from legal_due_diligence_rag.py
# ─────────────────────────────────────────────────────────────────────────────


def run_risk_engine(
    text: str,
    real_page_breaks: Optional[List[int]] = None,
    vectorizer=None,
    legal_db=None,
    rag_queries_per_category: Optional[Dict[str, List[str]]] = None,
    categorized_risks: Optional[Dict] = None,
    print_cli: bool = True,
    return_json: bool = False,
):
    """
    Main entry point.


    Parameters
    ----------
    text                    : full sale deed text (plain string)
    real_page_breaks        : char offsets where PDF pages begin (optional)
    vectorizer              : PDFVectorizer instance (for RAG similarity scoring)
    legal_db                : ChromaDB collection (for RAG similarity scoring)
    rag_queries_per_category: override default RAG queries per category
    categorized_risks       : output of LegalDueDiligenceSystem.classify_risks()
                              — Gemini's per-category written analysis. When
                              supplied, severity labels are parsed algorithmically
                              and used as the 4th sub-score in P_i.
    print_cli               : if True, print the formatted CLI report
    return_json             : if True, also return the JSON string


    Returns
    -------
    (LegalRiskIndex, optional[str])
    """
    print("\n[LRI] Scanning document for threats...")
    threat_map = scan_threats(text, real_page_breaks=real_page_breaks)
    for cat, threats in threat_map.items():
        print(f"  [{cat}]  {len(threats)} threat indicator(s) located")


    print("\n[LRI] Computing Legal Risk Index...")
    lri = compute_lri(
        text=text,
        threat_map=threat_map,
        vectorizer=vectorizer,
        legal_db=legal_db,
        rag_queries_per_category=rag_queries_per_category,
        categorized_risks=categorized_risks,
    )


    if print_cli:
        print_lri_cli(lri)


    if return_json:
        return lri, lri_to_json(lri)
    return lri, None

