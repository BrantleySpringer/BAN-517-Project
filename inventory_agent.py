"""Supply Chain Analytics Inventory Decision Agent.

Implements the 7-stage decision pipeline:
    Customer Question
        -> Identify Model
        -> Extract Inputs
        -> Select Python Tool
        -> Calculate
        -> Verify
        -> Explain

By default, Model Identification and Input Extraction use a deterministic
multi-signal NLP parser with zero external dependencies or API keys. When
`parser_mode="llm"` (or `"auto"` with an available LLM client and API key),
an LLM can perform Model Identification and Input Extraction into strict JSON,
while ALL numerical calculations and verification always run in Python
(`inventory_tools.py`).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from inventory_tools import TOOL_REGISTRY, get_tool, verify_calculation


# ---------------------------------------------------------------------------
# Structured Pipeline Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ModelIdentification:
    model_id: Optional[str]
    model_name: Optional[str]
    confidence: float
    matched_signals: List[str]
    scores: Dict[str, float]
    reasoning: str
    method: str = "rule_based"


@dataclass
class InputExtraction:
    params: Dict[str, Any]
    missing_required: List[str]
    assumptions: List[str] = field(default_factory=list)
    unit_conversions: List[str] = field(default_factory=list)
    raw_matches: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    method: str = "rule_based"


@dataclass
class ToolSelection:
    model_id: str
    model_name: str
    tool_name: str
    arguments: Dict[str, Any]
    ready: bool
    reason: str


@dataclass
class VerificationReport:
    passed: bool
    checks: List[Dict[str, Any]]


@dataclass
class AgentResponse:
    question: str
    status: str  # "completed", "needs_clarification", "missing_inputs", "error"
    identification: ModelIdentification
    extraction: Optional[InputExtraction]
    tool_selection: Optional[ToolSelection]
    calculation: Optional[Dict[str, Any]]
    verification: Optional[VerificationReport]
    explanation: str
    pipeline_trace: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Numeric & Time-Unit Parsing Helpers
# ---------------------------------------------------------------------------

_TIME_UNIT_ALIASES = {
    "day": "day",
    "days": "day",
    "daily": "day",
    "week": "week",
    "weeks": "week",
    "weekly": "week",
    "wk": "week",
    "wks": "week",
    "month": "month",
    "months": "month",
    "monthly": "month",
    "mo": "month",
    "mos": "month",
    "year": "year",
    "years": "year",
    "yearly": "year",
    "annual": "year",
    "annually": "year",
    "yr": "year",
    "yrs": "year",
}


def _to_float(num_str: str) -> float:
    cleaned = num_str.replace(",", "").replace("$", "").strip()
    multiplier = 1.0
    if cleaned.lower().endswith("k") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
        multiplier = 1000.0
    return float(cleaned) * multiplier


def _normalize_time_unit(raw_unit: Optional[str], default: str = "day") -> str:
    if not raw_unit:
        return default
    key = raw_unit.strip().lower()
    return _TIME_UNIT_ALIASES.get(key, default)


def _periods_per_year(unit: str, working_days_per_year: float = 365.0) -> float:
    if unit == "day":
        return working_days_per_year
    if unit == "week":
        return 52.0
    if unit == "month":
        return 12.0
    if unit == "year":
        return 1.0
    return 1.0


def _convert_time_length(
    value: float,
    from_unit: str,
    to_unit: str,
    working_days_per_year: float = 365.0,
) -> float:
    if from_unit == to_unit:
        return value
    days_map = {
        "day": 1.0,
        "week": 7.0,
        "month": working_days_per_year / 12.0,
        "year": working_days_per_year,
    }
    in_days = value * days_map[from_unit]
    return in_days / days_map[to_unit]


def _convert_demand_rate(
    mean_val: float,
    from_unit: str,
    to_unit: str,
    working_days_per_year: float = 365.0,
) -> float:
    if from_unit == to_unit:
        return mean_val
    annual = mean_val * _periods_per_year(from_unit, working_days_per_year)
    return annual / _periods_per_year(to_unit, working_days_per_year)


def _convert_demand_std(
    std_val: float,
    from_unit: str,
    to_unit: str,
    working_days_per_year: float = 365.0,
) -> float:
    """Convert independent-period demand std dev via variance scaling."""
    if from_unit == to_unit:
        return std_val
    ratio = _periods_per_year(from_unit, working_days_per_year) / _periods_per_year(
        to_unit, working_days_per_year
    )
    return std_val * math.sqrt(ratio)


def _extract_working_days(text: str) -> float:
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:working|operating|business)?\s*days\s*(?:per|a|in\s+a|/)\s*year",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    return 365.0


def _parse_rate_or_prob(val_str: str, has_pct: bool) -> float:
    val = float(val_str.replace(",", ""))
    if has_pct or val > 1.0:
        return val / 100.0
    return val


# ---------------------------------------------------------------------------
# Stage 2: Deterministic Model Identification
# ---------------------------------------------------------------------------

def _count_price_break_tiers(text: str) -> int:
    range_matches = re.findall(
        r"(?:\d[\d,]*\s*(?:-|to|–)\s*\d[\d,]*|\d[\d,]*\s*\+|\d[\d,]*\s+or\s+more|under\s+\d[\d,]*|less\s+than\s+\d[\d,]*|<\s*\d[\d,]*|>=\s*\d[\d,]*)"
        r"[^.;\n]{0,45}?\$\s*\d[\d,]*(?:\.\d+)?",
        text,
        re.IGNORECASE,
    )
    discount_pct_matches = re.findall(
        r"\d[\d,]*(?:\s*\+|\s+or\s+more)[^.;\n]{0,40}?\d+(?:\.\d+)?\s*%\s*discount",
        text,
        re.IGNORECASE,
    )
    return max(len(range_matches), len(discount_pct_matches))


def identify_model_rule_based(question: str) -> ModelIdentification:
    """Score the natural-language question against all 5 inventory decision models."""
    q = question.strip()
    ql = q.lower()

    scores: Dict[str, float] = {m: 0.0 for m in TOOL_REGISTRY}
    signals: Dict[str, List[str]] = {m: [] for m in TOOL_REGISTRY}

    def add(model: str, weight: float, label: str) -> None:
        scores[model] += weight
        signals[model].append(label)

    tier_count = _count_price_break_tiers(q)

    # --- 1. EOQ ---
    if re.search(r"\beoq\b|economic\s+order\s+quantity", ql):
        add("eoq", 4.5, "explicit EOQ keyword")
    if re.search(r"optimal\s+order\s+quantity|order\s+size|how\s+many\s+units\s+should\s+(?:we|i)\s+order\s+each\s+time", ql):
        add("eoq", 2.0, "order quantity objective")
    has_ordering = bool(
        re.search(
            r"(?:ordering(?:\s*/\s*setup)?|setup(?:\s*/\s*ordering)?|order|setup|set-up)\s+(?:cost|expense)s?"
            r"|per\s+order|cost\s+(?:of\s+placing|to\s+place)\s+(?:an?|each)\s+order",
            ql,
        )
    )
    has_holding = bool(re.search(r"(?:holding|carrying|storage)\s+(?:cost|rate|charge|expense)", ql))
    if has_ordering and has_holding:
        add("eoq", 3.0, "ordering + holding cost structure")
    elif has_ordering or has_holding:
        add("eoq", 1.2, "partial EOQ cost cue")
    if re.search(r"annual\s+demand|(?:per|a|each)\s+year|annually", ql):
        add("eoq", 0.8, "annual demand horizon")

    # --- 2. Quantity Discount ---
    for pat, label, wt in [
        (r"quantity\s+discount", "quantity discount mention", 4.5),
        (r"price\s+break", "price break mention", 4.5),
        (r"quantity\s+break", "quantity break mention", 4.5),
        (r"(?:bulk|volume|tier(?:ed)?)\s+discount", "volume/tier discount mention", 4.0),
        (r"discount\s+(?:schedule|table|tier)", "discount schedule", 4.0),
        (r"all[- ]units|incremental\s+discount", "discount regime specified", 4.0),
    ]:
        if re.search(pat, ql):
            add("quantity_discount", wt, label)
    if tier_count >= 2:
        add("quantity_discount", 5.5, f"detected {tier_count} price-break tiers")
        add("eoq", -2.5, "penalized due to multiple price tiers")
    elif tier_count == 1 and re.search(
        r"discount|quantity\s+break|price\s+break|base\s+(?:unit\s+)?(?:purchase\s+)?(?:cost|price)",
        ql,
    ):
        add("quantity_discount", 3.5, "discount break tier detected")
        add("eoq", -2.0, "penalized due to price-break tier")

    # --- 3. Safety Stock & Reorder Point ---
    for pat, label, wt in [
        (r"safety\s+stock", "safety stock mention", 4.0),
        (r"reorder\s+(?:point|level)|\brop\b", "reorder point mention", 4.0),
        (r"continuous\s+review|q[- ]model", "continuous review / Q-model", 3.5),
        (r"when\s+(?:should|do)\s+(?:we|i)\s+reorder", "reorder trigger question", 3.5),
        (r"lead[- ]time\s+demand|demand\s+during\s+(?:the\s+)?lead\s+time", "lead-time demand", 3.5),
    ]:
        if re.search(pat, ql):
            add("safety_stock_rop", wt, label)
    if re.search(r"lead\s+time", ql):
        add("safety_stock_rop", 1.5, "lead time mentioned")
    if re.search(r"service\s+level|in[- ]stock\s+probability|stockout\s+(?:probability|risk)|fill\s+rate|z[- ]?score", ql):
        add("safety_stock_rop", 1.5, "service level / stockout metric")
    if re.search(r"std|standard\s+deviation|\bsd\b", ql) and re.search(r"lead\s+time", ql):
        add("safety_stock_rop", 1.2, "demand/lead-time variability")

    # --- 4. Fixed-Period Review ---
    for pat, label, wt in [
        (r"fixed[- ]period", "fixed-period mention", 4.5),
        (r"periodic\s+review|p[- ]model", "periodic review / P-model", 4.5),
        (r"order[- ]up[- ]to|target\s+(?:inventory\s+)?level", "order-up-to target level", 4.5),
        (r"review\s+(?:period|interval)", "review period/interval", 4.5),
        (r"(?:review(?:ed)?|check(?:ed)?|inspect(?:ed)?|count(?:ed)?|restock(?:ed)?)\s+(?:inventory\s+|stock\s+)?every\s+\d+", "periodic inspection cadence", 4.0),
        (r"\bevery\s+\d+(?:\.\d+)?\s*(?:days?|weeks?|months?)\b", "periodic every-N-periods cadence", 2.8),
        (r"protection\s+(?:period|interval)", "protection interval", 3.5),
    ]:
        if re.search(pat, ql):
            add("fixed_period_review", wt, label)
    if re.search(r"on[- ]hand|on[- ]order|scheduled\s+receipt|inventory\s+position|backorder", ql):
        add("fixed_period_review", 1.8, "inventory position state")
    if scores["fixed_period_review"] >= 3.5 and not re.search(r"reorder\s+(?:point|level)|\brop\b", ql):
        add("safety_stock_rop", -1.5, "periodic review cadence overrides continuous ROP")

    # --- 5. Newsvendor ---
    for pat, label, wt in [
        (r"newsvendor|newsboy", "newsvendor mention", 5.0),
        (r"single[- ]period|one[- ]time\s+order", "single-period horizon", 4.5),
        (r"critical\s+ratio|underage|overage", "critical ratio / underage-overage", 4.5),
        (r"salvage|unsold|leftover|scrap\s+value|disposal\s+(?:cost|fee)|worthless|thrown?\s+away|discard", "salvage / unsold disposition", 3.5),
        (r"donat(?:e|ed)|end\s+of\s+(?:the\s+)?(?:day|season)|perishable|markdown", "perishable / end-of-period signal", 2.5),
    ]:
        if re.search(pat, ql):
            add("newsvendor", wt, label)
    has_sell_price = bool(re.search(r"selling\s+price|retail\s+price|sells?\s+for|retails?\s+for", ql))
    has_unit_cost = bool(re.search(r"unit\s+cost|cost\s+to\s+(?:make|bake|buy|produce)|(?:make|bake|buy|purchase)s?\s+.{0,20}\$", ql))
    if has_sell_price and has_unit_cost and not has_ordering and not has_holding:
        add("newsvendor", 2.5, "selling price + unit cost without ongoing ordering/holding")
    if re.search(r"uniform(?:ly)?\s+distributed|probability\s+of|prob\s*\d", ql) and (has_sell_price or has_unit_cost):
        add("newsvendor", 1.5, "single-period demand distribution")
    if has_ordering and has_holding and re.search(r"annual", ql):
        add("newsvendor", -2.5, "annual EOQ structure penalizes single-period")

    # Determine best model
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_score = ranked[0]
    second_id, second_score = ranked[1]

    if best_score < 2.0:
        return ModelIdentification(
            model_id=None,
            model_name=None,
            confidence=0.0,
            matched_signals=[],
            scores=scores,
            reasoning=(
                "Unable to confidently match the question to one of the 5 supported inventory "
                "models (EOQ, Quantity Discount, Safety Stock & ROP, Fixed-Period Review, Newsvendor)."
            ),
        )

    margin = max(0.0, best_score - max(0.0, second_score))
    confidence = min(0.99, round(0.55 + 0.06 * best_score + 0.05 * margin, 2))
    model_name = TOOL_REGISTRY[best_id]["name"]
    reasoning = (
        f"Selected {model_name} (score {best_score:.1f} vs runner-up "
        f"{TOOL_REGISTRY[second_id]['name']} {second_score:.1f}) based on: "
        + ", ".join(signals[best_id])
        + "."
    )
    return ModelIdentification(
        model_id=best_id,
        model_name=model_name,
        confidence=confidence,
        matched_signals=signals[best_id],
        scores={k: round(v, 2) for k, v in scores.items()},
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Stage 3: Deterministic Input Extraction
# ---------------------------------------------------------------------------

def _extract_annual_demand_and_costs(
    text: str,
    assumptions: List[str],
    conversions: List[str],
    raw_matches: Dict[str, Any],
) -> Dict[str, Any]:
    """Shared extractor for EOQ and Quantity Discount parameters."""
    params: Dict[str, Any] = {}
    working_days = _extract_working_days(text)
    if working_days != 365.0:
        params["working_days_per_year"] = working_days
        raw_matches["working_days_per_year"] = working_days

    # Annual demand (explicit annual first, then scaled daily/weekly/monthly, then symbolic D)
    annual_patterns = [
        r"(?:annual|yearly)\s+(?:demand|usage|sales|consumption|requirement)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?k?)\s*(?:units?|items?|pieces?|backpacks?|widgets?)?",
        r"(?:demand|usage|sales|consumption|requirement)\s*(?:is|of|=|:|averages?)?\s*([\d,]+(?:\.\d+)?k?)\s*(?:[a-z]+\s+){0,2}(?:per\s+year|a\s+year|each\s+year|annually|/year|/yr)",
        r"(?:sell|sells|selling|buy|buys|buying|use|uses|consume|consumes|need|needs)\s*(?:about\s+|approximately\s+)?([\d,]+(?:\.\d+)?k?)\s*(?:[a-z]+\s+){0,3}(?:per\s+year|a\s+year|each\s+year|annually|/year|/yr)",
        r"\bD\s*=\s*([\d,]+(?:\.\d+)?k?)\b",
    ]
    for pat in annual_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            params["annual_demand"] = _to_float(m.group(1))
            raw_matches["annual_demand"] = m.group(0)
            break

    if "annual_demand" not in params:
        period_m = re.search(
            r"(?:demand\s*(?:is|of|averages?)?|sell(?:s|ing)?|buy(?:s|ing)?|use(?:s)?)\s*(?:about\s+)?([\d,]+(?:\.\d+)?k?)\s*(?:[a-z]+\s+){0,2}(?:per|a|each|/)\s*(month|week|day)",
            text,
            re.IGNORECASE,
        )
        if not period_m:
            period_m = re.search(
                r"(daily|weekly|monthly)\s+(?:demand|usage|sales)\s*(?:is|of|averages?|=|:)?\s*([\d,]+(?:\.\d+)?k?)",
                text,
                re.IGNORECASE,
            )
            if period_m:
                unit = _normalize_time_unit(period_m.group(1))
                val = _to_float(period_m.group(2))
                factor = _periods_per_year(unit, working_days)
                params["annual_demand"] = val * factor
                conversions.append(
                    f"Converted {unit} demand ({val:g}) to annual demand: {val:g} * {factor:g} = {params['annual_demand']:g} units/year"
                )
                raw_matches["annual_demand"] = period_m.group(0)
        else:
            val = _to_float(period_m.group(1))
            unit = _normalize_time_unit(period_m.group(2))
            factor = _periods_per_year(unit, working_days)
            params["annual_demand"] = val * factor
            conversions.append(
                f"Converted {unit} demand ({val:g}) to annual demand: {val:g} * {factor:g} = {params['annual_demand']:g} units/year"
            )
            raw_matches["annual_demand"] = period_m.group(0)

    # Ordering / setup cost (all standard EOQ synonyms + reversed order + symbolic S)
    ord_patterns = [
        r"(?:ordering\s*/\s*setup|setup\s*/\s*ordering|ordering|order|setup|set-up|order\s+placement|fixed\s+order)\s*(?:cost|expense|fee)s?\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"(?:cost\s+(?:of\s+placing|to\s+place)\s+(?:an?|each)\s+order|cost\s+per\s+order)\s*(?:is|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:cost\s+per\s+order|cost\s+(?:of\s+placing|to\s+place)\s+(?:an?|each)\s+order|per\s+order|/order|to\s+place\s+(?:an?|each)\s+order|(?:ordering\s*/\s*setup|setup\s*/\s*ordering|ordering|order|setup|set-up)\s*(?:cost|expense|fee))",
        r"(?:each\s+order\s+)?costs?\s*\$\s*([\d,]+(?:\.\d+)?)\s*(?:to\s+place(?:\s+an\s+order)?|per\s+order|each\s+time)",
        r"\bS\s*=\s*\$?\s*([\d,]+(?:\.\d+)?)\b",
    ]
    for pat in ord_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            params["ordering_cost"] = _to_float(m.group(1))
            raw_matches["ordering_cost"] = m.group(0)
            break

    # Percentage holding cost / carrying rate (% or explicit decimal fraction)
    rate_patterns = [
        r"(?:annual\s+)?(?:percentage|percent|fractional)?\s*(?:holding|carrying|inventory\s+carrying|storage)\s*(?:cost\s*)?(?:rate|charge|factor|percentage|percent)?\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*(%|percent)",
        r"(\d+(?:\.\d+)?)\s*(%|percent)\s*(?:of\s+(?:the\s+)?(?:unit|item|purchase)?\s*(?:cost|price|value))",
        r"(?:percentage\s+)?(?:holding|carrying|inventory\s+carrying)\s*(?:cost\s*)?(?:rate|charge|factor)\s*(?:is|of|=|:)?\s*(0?\.\d+)",
        r"\bi\s*=\s*(?:(\d+(?:\.\d+)?)\s*(%|percent)|(0?\.\d+))",
    ]
    for idx, pat in enumerate(rate_patterns):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            if idx == 2:
                params["holding_rate"] = float(m.group(1))
            elif idx == 3:
                if m.group(1):
                    params["holding_rate"] = float(m.group(1)) / 100.0
                else:
                    params["holding_rate"] = float(m.group(3))
            else:
                params["holding_rate"] = float(m.group(1)) / 100.0
            raw_matches["holding_rate"] = m.group(0)
            break

    # Fixed holding cost ($/unit/period) - only if percentage holding rate was not specified
    if "holding_rate" not in params:
        hold_patterns = [
            r"(?:fixed\s+|annual\s+)?(?:holding|carrying|storage)\s*(?:cost|expense)\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)(?:\s*(?:per\s*unit|/unit))?(?:\s*(?:per\s*(year|month|week|day)|annually|/year|/yr|/month|/week|/day))?",
            r"(?:fixed\s+|annual\s+)?(?:holding|carrying|storage)\s*(?:cost|expense)\s*(?:is|of|=|:)\s*([\d,]+(?:\.\d+)?)\b(?!\s*%)",
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s*unit|/unit)\s*(?:per\s*(year|month|week|day)|annually|/year|/yr|/month|/week|/day)?\s*(?:to\s+hold|for\s+holding|(?:fixed\s+)?(?:holding|carrying|storage)\s*(?:cost|expense))?",
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:fixed\s+)?(?:holding|carrying|storage)\s*(?:cost|expense)",
            r"\bH\s*=\s*\$?\s*([\d,]+(?:\.\d+)?)\b",
        ]
        for pat in hold_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                h_val = _to_float(m.group(1))
                raw_unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                if raw_unit:
                    unit = _normalize_time_unit(raw_unit)
                    if unit != "year":
                        factor = _periods_per_year(unit, working_days)
                        h_annual = h_val * factor
                        conversions.append(
                            f"Annualized holding cost from ${h_val:g}/{unit} to ${h_annual:g}/year"
                        )
                        h_val = h_annual
                params["holding_cost"] = h_val
                raw_matches["holding_cost"] = m.group(0)
                break

    # Unit purchase price / unit cost (useful for EOQ when holding_rate is given)
    unit_cost_patterns = [
        r"(?:base\s+|regular\s+|standard\s+)?(?:unit|purchase|item)\s*(?:purchase\s*)?(?:cost|price)\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"(?:at\s+a\s+(?:unit\s+)?(?:cost|price)\s+of|costs?)\s*\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit|each|/unit)",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:unit|purchase|item)\s*(?:cost|price)",
        r"\bC\s*=\s*\$?\s*([\d,]+(?:\.\d+)?)\b",
    ]
    for pat in unit_cost_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            params["unit_cost"] = _to_float(m.group(1))
            raw_matches["unit_cost"] = m.group(0)
            break

    return params


def _extract_quantity_discount_tiers(text: str) -> List[Dict[str, Any]]:
    """Parse price-break tiers from varied natural-language tier descriptions."""
    tiers: List[Dict[str, Any]] = []
    seen_spans: List[Tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= a or s >= b) for a, b in seen_spans)

    # 1. Range: "1-499 units: $12.00" or "200 to 599, price is $28"
    for m in re.finditer(
        r"(?<!\d)([\d,]+)\s*(?:-|to|–)\s*([\d,]+)\s*(?:units?|items?)?[^.;\n$\d]{0,40}?\$\s*([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    ):
        if overlaps(m.start(), m.end()):
            continue
        seen_spans.append((m.start(), m.end()))
        tiers.append(
            {
                "min_qty": _to_float(m.group(1)),
                "max_qty": _to_float(m.group(2)),
                "max_exclusive": False,
                "unit_price": _to_float(m.group(3)),
            }
        )

    # 2. Lower open-ended: "under 200 units = $30" / "less than 500: $20" / "< 100: $50" / "up to 499: $20"
    for m in re.finditer(
        r"(?:(under|less\s+than|below|<)|(up\s+to|<=))\s*([\d,]+)\s*(?:units?|items?)?[^.;\n$\d]{0,40}?\$\s*([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    ):
        if overlaps(m.start(), m.end()):
            continue
        seen_spans.append((m.start(), m.end()))
        is_exclusive = bool(m.group(1))
        tiers.append(
            {
                "min_qty": 0.0,
                "max_qty": _to_float(m.group(3)),
                "max_exclusive": is_exclusive,
                "unit_price": _to_float(m.group(4)),
            }
        )

    # 3. Upper open-ended: "1000+: $11.00" / "600 or more = $26" / "500 units or more cost $18"
    for m in re.finditer(
        r"(?<![\d.$])(?:(>=|at\s+least|over|above)?\s*([\d,]+)\s*(?:units?|items?)?\s*(?:\+|\s+or\s+more|\s+and\s+above|\s+or\s+above)?)\s*(?:units?|items?)?[^.;\n$\d]{0,40}?\$\s*([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    ):
        matched_chunk = m.group(0)
        has_marker = bool(
            m.group(1)
            or re.search(r"\+|or\s+more|and\s+above|or\s+above", matched_chunk, re.IGNORECASE)
        )
        if not has_marker or overlaps(m.start(), m.end()):
            continue
        seen_spans.append((m.start(), m.end()))
        min_q = _to_float(m.group(2))
        if re.search(r"\b(?:over|above)\b", matched_chunk, re.IGNORECASE) and not re.search(
            r"or\s+above|and\s+above", matched_chunk, re.IGNORECASE
        ):
            min_q += 1.0
        tiers.append(
            {
                "min_qty": min_q,
                "max_qty": None,
                "max_exclusive": False,
                "unit_price": _to_float(m.group(3)),
            }
        )

    # 4. Percentage discounts off base/unit purchase price if explicit dollar tiers not found
    if not tiers:
        base_m = re.search(
            r"(?:base|regular|standard|list|unit)?\s*(?:purchase\s*)?(?:price|cost)\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if base_m:
            base_price = _to_float(base_m.group(1))
            disc_matches = list(
                re.finditer(
                    r"([\d,]+)\s*(?:units?\s*)?(?:\+|\s+or\s+more)\s*(?:units?)?[^.;\n]{0,35}?(\d+(?:\.\d+)?)\s*%\s*(?:off|discount)",
                    text,
                    re.IGNORECASE,
                )
            )
            if disc_matches:
                disc_tiers = sorted(
                    [(_to_float(dm.group(1)), float(dm.group(2))) for dm in disc_matches],
                    key=lambda x: x[0],
                )
                tiers.append(
                    {
                        "min_qty": 0.0,
                        "max_qty": disc_tiers[0][0],
                        "max_exclusive": True,
                        "unit_price": base_price,
                    }
                )
                for idx, (min_q, pct) in enumerate(disc_tiers):
                    max_q = disc_tiers[idx + 1][0] if idx + 1 < len(disc_tiers) else None
                    tiers.append(
                        {
                            "min_qty": min_q,
                            "max_qty": max_q,
                            "max_exclusive": max_q is not None,
                            "unit_price": base_price * (1.0 - pct / 100.0),
                        }
                    )

    if not tiers:
        return []

    # Sort and resolve any open max_qty on lower tiers using next tier's min_qty
    tiers.sort(key=lambda t: t["min_qty"])
    for i in range(len(tiers) - 1):
        if tiers[i]["max_qty"] is None:
            tiers[i]["max_qty"] = tiers[i + 1]["min_qty"]
            tiers[i]["max_exclusive"] = True
    return tiers


def _extract_service_level_and_z(
    text: str,
    params: Dict[str, Any],
    raw_matches: Dict[str, Any],
) -> None:
    fill_m = re.search(
        r"fill\s*rate\s*(?:of|is|=|:|target)?\s*(\d+(?:\.\d+)?)\s*(%|percent)?",
        text,
        re.IGNORECASE,
    )
    if fill_m:
        params["service_level"] = _parse_rate_or_prob(fill_m.group(1), bool(fill_m.group(2)))
        params["service_level_type"] = "fill_rate"
        raw_matches["service_level"] = fill_m.group(0)
        return

    csl_patterns = [
        r"(?:target\s*)?(?:cycle\s*)?(?:service\s*level|csl|in[- ]stock\s*probability|probability\s*of\s*not\s*stocking\s*out)\s*(?:target\s*)?(?:of|is|=|:|at)?\s*(\d+(?:\.\d+)?)\s*(%|percent)?",
        r"(\d+(?:\.\d+)?)\s*(%|percent)\s*(?:target\s*)?(?:cycle\s*)?(?:service\s*level|csl|in[- ]stock\s*probability)",
    ]
    for pat in csl_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            params["service_level"] = _parse_rate_or_prob(m.group(1), bool(m.group(2)))
            raw_matches["service_level"] = m.group(0)
            return

    stockout_m = re.search(
        r"(?:stockout\s*(?:probability|risk)|probability\s*of\s*(?:a\s*)?stockout)\s*(?:of|is|no\s*more\s*than|at\s*most|<=|<|=|:)?\s*(\d+(?:\.\d+)?)\s*(%|percent)?",
        text,
        re.IGNORECASE,
    )
    if stockout_m:
        so_prob = _parse_rate_or_prob(stockout_m.group(1), bool(stockout_m.group(2)))
        params["service_level"] = 1.0 - so_prob
        raw_matches["service_level"] = f"1 - ({stockout_m.group(0)})"
        return

    z_m = re.search(r"\bz[- ]?(?:score|value)?\s*(?:of|is|=|:)\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if z_m:
        params["z_score"] = float(z_m.group(1))
        raw_matches["z_score"] = z_m.group(0)


def _extract_demand_and_lead_time_stats(
    text: str,
    conversions: List[str],
    raw_matches: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract demand mean/std, lead time (+ std), and align time units cleanly."""
    out: Dict[str, Any] = {}
    working_days = _extract_working_days(text)

    # Check direct lead-time demand distribution first
    direct_lt = re.search(
        r"(?:lead[- ]time\s+demand|demand\s+during\s+(?:the\s+)?lead[- ]time)[^.;\n]{0,50}?(?:mean|average|avg)\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)[^.;\n]{0,50}?(?:std(?:\s*dev)?|standard\s+deviation|sd)\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if direct_lt:
        out["lead_time_demand_mean"] = _to_float(direct_lt.group(1))
        out["lead_time_demand_std"] = _to_float(direct_lt.group(2))
        raw_matches["lead_time_demand"] = direct_lt.group(0)
        return out

    # Lead time (average / mean / expected lead time)
    lt_val: Optional[float] = None
    lt_unit: Optional[str] = None
    lt_patterns = [
        r"(?:(?:average|mean|expected|avg|delivery|replenishment)\s+)?lead[- ]time\s*(?:is\s+(?:constant|fixed|deterministic)\s+at|averages?|is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
        r"\bL\s*=\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
    ]
    for pat in lt_patterns:
        lt_m = re.search(pat, text, re.IGNORECASE)
        if lt_m:
            lt_val = _to_float(lt_m.group(1))
            lt_unit = _normalize_time_unit(lt_m.group(2), default="") if lt_m.group(2) else None
            raw_matches["lead_time"] = lt_m.group(0)
            break

    # Lead time std dev (both "lead-time std dev is X" and "std dev of lead time is X" and sigma_L)
    lt_std_val: Optional[float] = None
    lt_std_unit: Optional[str] = None
    lt_std_patterns = [
        r"lead[- ]time\s*(?:std(?:\s*dev)?|standard\s+deviation|sd)\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
        r"(?:std(?:\s*dev)?|standard\s+deviation|sd)\s*(?:of|for)\s*(?:the\s+)?lead[- ]time\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
        r"lead[- ]time[^.?!\n]{0,60}?(?:std(?:\s*dev)?|standard\s+deviation|sd)\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
        r"(?:sigma_L|σ_L)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
    ]
    for pat in lt_std_patterns:
        lt_std_m = re.search(pat, text, re.IGNORECASE)
        if lt_std_m:
            lt_std_val = _to_float(lt_std_m.group(1))
            lt_std_unit = (
                _normalize_time_unit(lt_std_m.group(2), default=lt_unit or "day")
                if lt_std_m.group(2)
                else lt_unit
            )
            raw_matches["lead_time_std"] = lt_std_m.group(0)
            break

    # Per-period average/mean demand
    d_mean: Optional[float] = None
    d_mean_unit: Optional[str] = None
    d_mean_patterns = [
        r"(?:(?:average|mean|expected|avg)\s+)?(daily|weekly|monthly|annual|yearly)\s+demand\s*(?:is\s+(?:constant|fixed|deterministic)\s+at|averages?|has\s+(?:a\s+)?mean(?:\s+of)?|is|of|=|:)?\s*([\d,]+(?:\.\d+)?k?)",
        r"(?:mean|average|avg|expected)\s+(daily|weekly|monthly|annual|yearly)\s+demand\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?k?)",
        r"(?:(?:average|mean|expected|avg)\s+)?demand\s*(?:is\s+(?:constant|fixed|deterministic)\s+at|averages?|has\s+(?:a\s+)?mean(?:\s+of)?|is|of|=|:)?\s*([\d,]+(?:\.\d+)?k?)\s*(?:units?\s*)?(?:per|a|each|/)\s*(day|week|month|year)",
        r"(?:(?:average|mean|expected|avg)\s+)?demand\s*(?:is\s+(?:constant|fixed|deterministic)\s+at|averages?|has\s+(?:a\s+)?mean(?:\s+of)?|is|of|=|:)\s*([\d,]+(?:\.\d+)?k?)",
        r"\b(?:d|mu_d)\s*=\s*([\d,]+(?:\.\d+)?k?)(?:\s*(?:units?)?\s*(?:per|/)\s*(day|week|month|year))?",
    ]
    for idx, pat in enumerate(d_mean_patterns):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            if idx in (0, 1):
                d_mean_unit = _normalize_time_unit(m.group(1))
                d_mean = _to_float(m.group(2))
            elif idx == 2:
                d_mean = _to_float(m.group(1))
                d_mean_unit = _normalize_time_unit(m.group(2))
            elif idx == 4:
                d_mean = _to_float(m.group(1))
                if m.group(2):
                    d_mean_unit = _normalize_time_unit(m.group(2))
            else:
                d_mean = _to_float(m.group(1))
            raw_matches["demand_mean"] = m.group(0)
            break

    # Per-period demand std dev (strip all lead-time std dev clauses first so they never collide)
    d_std: Optional[float] = None
    d_std_unit: Optional[str] = None
    text_no_lt = text
    for pat in lt_std_patterns:
        text_no_lt = re.sub(pat, " ", text_no_lt, flags=re.IGNORECASE)

    d_std_patterns = [
        r"(daily|weekly|monthly|annual|yearly)\s+(?:demand\s+)?(?:std(?:\s*dev)?|standard\s+deviation|sd)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
        r"demand\s+(?:std(?:\s*dev)?|standard\s+deviation|sd)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)(?:\s*(?:units?\s*)?(?:per|/|a)\s*(day|week|month|year))?",
        r"(?:std(?:\s*dev)?|standard\s+deviation|sd)\s*(?:of\s+(?:(daily|weekly|monthly|annual|yearly)\s+)?demand)?\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)(?:\s*(?:units?\s*)?(?:per|/|a)\s*(day|week|month|year))?",
        r"(?:sigma_d|σ_d)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)(?:\s*(?:units?\s*)?(?:per|/|a)\s*(day|week|month|year))?",
    ]
    for idx, pat in enumerate(d_std_patterns):
        m = re.search(pat, text_no_lt, re.IGNORECASE)
        if m:
            if idx == 0:
                d_std_unit = _normalize_time_unit(m.group(1))
                d_std = _to_float(m.group(2))
            elif idx == 2:
                if m.group(1):
                    d_std_unit = _normalize_time_unit(m.group(1))
                d_std = _to_float(m.group(2))
                if m.group(3):
                    d_std_unit = _normalize_time_unit(m.group(3))
            else:
                d_std = _to_float(m.group(1))
                if m.lastindex and m.lastindex >= 2 and m.group(2):
                    d_std_unit = _normalize_time_unit(m.group(2))
            raw_matches["demand_std"] = m.group(0)
            break

    # Determine unified base time unit (prefer lead_time unit, else demand_mean unit, else demand_std unit)
    base_unit = lt_unit or d_mean_unit or d_std_unit or "period"
    if base_unit != "period":
        if d_mean_unit is None:
            d_mean_unit = base_unit
        if d_std_unit is None:
            d_std_unit = d_mean_unit or base_unit
        if lt_unit is None:
            lt_unit = base_unit
        if lt_std_unit is None:
            lt_std_unit = lt_unit

    if lt_val is not None:
        if base_unit != "period" and lt_unit and lt_unit != base_unit:
            lt_conv = _convert_time_length(lt_val, lt_unit, base_unit, working_days)
            conversions.append(f"Converted lead time {lt_val:g} {lt_unit}(s) -> {lt_conv:g} {base_unit}(s)")
            lt_val = lt_conv
        out["lead_time"] = lt_val

    if lt_std_val is not None:
        if base_unit != "period" and lt_std_unit and lt_std_unit != base_unit:
            lt_std_conv = _convert_time_length(lt_std_val, lt_std_unit, base_unit, working_days)
            conversions.append(
                f"Converted lead time std {lt_std_val:g} {lt_std_unit}(s) -> {lt_std_conv:g} {base_unit}(s)"
            )
            lt_std_val = lt_std_conv
        out["lead_time_std"] = lt_std_val

    if d_mean is not None:
        if base_unit != "period" and d_mean_unit and d_mean_unit != base_unit:
            d_mean_conv = _convert_demand_rate(d_mean, d_mean_unit, base_unit, working_days)
            conversions.append(
                f"Converted demand mean {d_mean:g}/{d_mean_unit} -> {d_mean_conv:g}/{base_unit}"
            )
            d_mean = d_mean_conv
        out["demand_mean"] = d_mean

    if d_std is not None:
        if base_unit != "period" and d_std_unit and d_std_unit != base_unit:
            d_std_conv = _convert_demand_std(d_std, d_std_unit, base_unit, working_days)
            conversions.append(
                f"Converted demand std {d_std:g}/{d_std_unit} -> {d_std_conv:g}/{base_unit}"
            )
            d_std = d_std_conv
        out["demand_std"] = d_std

    out["time_unit"] = base_unit
    return out


def _extract_newsvendor_inputs(
    text: str,
    assumptions: List[str],
    raw_matches: Dict[str, Any],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    # 1. Salvage / disposal / unsold clauses FIRST so "sold at end of day for $X" never collides with selling_price
    clean_text = text
    salvage_patterns = [
        r"(?:salvage\s*value|salvage|residual\s*value|resale\s*value|scrap\s*value|liquidation\s*value|markdown\s*price)\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+)?(?:salvage\s*value|residual\s*value|resale\s*value|scrap\s*value|liquidation\s*value)",
        r"(?:unsold|leftover|remaining|excess)\s+[^.?!\n]{0,45}?(?:donated?|liquidated?|resold|sold|salvaged?)\s+(?:for|at)\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"(?:donated?|liquidated?|salvaged?)\s+(?:for|at)\s*\$\s*([\d,]+(?:\.\d+)?)(?:\s*(?:salvage\s*value|each|per\s*unit))?",
        r"(?:sold|resold)\s+(?:at\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(?:day|season|period)|as\s+(?:scrap|leftover|salvage))\s+(?:for|at)\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"(?:^|[,\s;(])s\s*=\s*\$\s*([\d,]+(?:\.\d+)?)",
    ]
    for pat in salvage_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            params["salvage_value"] = _to_float(m.group(1))
            raw_matches["salvage_value"] = m.group(0).strip()
            clean_text = clean_text.replace(m.group(0), " ")
            break

    if "salvage_value" not in params:
        disp_m = re.search(
            r"disposal\s*(?:cost|fee)\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
            clean_text,
            re.IGNORECASE,
        )
        if disp_m:
            params["salvage_value"] = -_to_float(disp_m.group(1))
            raw_matches["salvage_value"] = f"-({disp_m.group(0)})"
            clean_text = clean_text.replace(disp_m.group(0), " ")
        elif re.search(
            r"no\s+salvage(?:\s+value)?|zero\s+salvage(?:\s+value)?|worthless|thrown?\s+away|discarded",
            text,
            re.IGNORECASE,
        ):
            params["salvage_value"] = 0.0
            raw_matches["salvage_value"] = "0.0 (explicit zero salvage)"

    # 2. Goodwill/penalty and direct underage/overage costs BEFORE generic price/cost patterns
    pen_m = re.search(
        r"(?:goodwill\s*(?:cost|penalty|loss)|(?:shortage|stockout)\s*penalty(?:\s*cost)?)\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
        clean_text,
        re.IGNORECASE,
    ) or re.search(
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+)?(?:goodwill\s*(?:cost|penalty|loss)|(?:shortage|stockout)\s*penalty(?:\s*cost)?)",
        clean_text,
        re.IGNORECASE,
    )
    if pen_m:
        params["shortage_penalty"] = _to_float(pen_m.group(1))
        raw_matches["shortage_penalty"] = pen_m.group(0).strip()
        clean_text = clean_text.replace(pen_m.group(0), " ")

    cu_m = (
        re.search(
            r"(?:underage|shortage|stockout|lost\s*margin|unmet\s*demand)\s*cost\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
            clean_text,
            re.IGNORECASE,
        )
        or re.search(
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+)?(?:underage|shortage|stockout|lost\s*margin)\s*cost",
            clean_text,
            re.IGNORECASE,
        )
        or re.search(
            r"\bC_?u\s*(?:=|is|of)\s*\$?\s*([\d,]+(?:\.\d+)?)",
            clean_text,
            re.IGNORECASE,
        )
    )
    if cu_m:
        clean_text = clean_text.replace(cu_m.group(0), " ")

    co_m = (
        re.search(
            r"(?:overage|excess|overstock)\s*cost\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
            clean_text,
            re.IGNORECASE,
        )
        or re.search(
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+)?(?:overage|excess|overstock)\s*cost",
            clean_text,
            re.IGNORECASE,
        )
        or re.search(
            r"\bC_?o\s*(?:=|is|of)\s*\$?\s*([\d,]+(?:\.\d+)?)",
            clean_text,
            re.IGNORECASE,
        )
    )
    if co_m:
        clean_text = clean_text.replace(co_m.group(0), " ")

    # 3. Selling price (on cleaned text so salvage/Cu/Co clauses never collide)
    sell_patterns = [
        r"(?:selling\s*price|retail\s*price|sale\s*price|unit\s*selling\s*price|revenue\s*per\s*unit|(?<!purchase\s)(?<!wholesale\s)(?<!unit\s)(?<!salvage\s)(?<!liquidation\s)(?<!resale\s)(?<!markdown\s)\bprice)\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+)?(?:selling\s*price|retail\s*price|sale\s*price)",
        r"(?:sells?|sold|retails?|retail|priced)\s+(?:[a-z]+\s+){0,3}?(?:for|at)\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"(?:^|[,\s;(])p\s*=\s*\$?\s*([\d,]+(?:\.\d+)?)",
    ]
    for pat in sell_patterns:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            params["selling_price"] = _to_float(m.group(1))
            raw_matches["selling_price"] = m.group(0).strip()
            clean_text = clean_text.replace(m.group(0), " ")
            break

    # 4. Purchase / unit cost (on cleaned text)
    cost_patterns = [
        r"(?:purchase\s*cost|purchase\s*price|unit\s*cost|wholesale\s*(?:cost|price)|acquisition\s*cost|procurement\s*cost|production\s*cost|cost\s+per\s+unit|cost\s+to\s+(?:make|bake|produce|buy|purchase|acquire))\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+)?(?:purchase\s*cost|purchase\s*price|unit\s*cost|wholesale\s*cost|production\s*cost)",
        r"(?:makes?|bakes?|buys?|purchases?|produces?|acquires?|made|baked|bought|purchased|produced|acquired)\s+[^.?!\n$]{0,35}?(?:for|at)\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"costs?\s*(?:is|of|=|:)?\s*\$\s*([\d,]+(?:\.\d+)?)",
        r"(?:^|[,\s;(])c\s*=\s*\$?\s*([\d,]+(?:\.\d+)?)",
    ]
    for pat in cost_patterns:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            params["unit_cost"] = _to_float(m.group(1))
            raw_matches["unit_cost"] = m.group(0).strip()
            break

    # Resolve direct underage / overage costs vs shortage penalty when price & cost also exist
    if cu_m:
        if (
            "selling_price" in params
            and "unit_cost" in params
            and not co_m
            and "shortage_penalty" not in params
            and re.search(r"shortage|stockout", cu_m.group(0), re.IGNORECASE)
        ):
            params["shortage_penalty"] = _to_float(cu_m.group(1))
            raw_matches["shortage_penalty"] = cu_m.group(0).strip()
        else:
            params["underage_cost"] = _to_float(cu_m.group(1))
            raw_matches["underage_cost"] = cu_m.group(0).strip()
    if co_m:
        params["overage_cost"] = _to_float(co_m.group(1))
        raw_matches["overage_cost"] = co_m.group(0).strip()

    if (
        "selling_price" in params
        and "unit_cost" in params
        and "salvage_value" not in params
        and not ("underage_cost" in params and "overage_cost" in params)
    ):
        params["salvage_value"] = 0.0
        assumptions.append("Assumed salvage_value = $0.00 since no salvage value was mentioned.")

    # 5. Demand distribution: Uniform -> Discrete -> Normal
    uni_m = (
        re.search(
            r"uniform(?:ly)?(?:\s+distribution)?(?:\s*distributed)?(?:\s*(?:between|from|on))?\s*([\d,]+(?:\.\d+)?)\s*(?:and|to|-)\s*([\d,]+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"uniform\s+distribution\s+(?:with\s+)?(?:min(?:imum)?|a)\s*(?:=|is|of)?\s*([\d,]+(?:\.\d+)?)\s*(?:,|and)\s*(?:max(?:imum)?|b)\s*(?:=|is|of)?\s*([\d,]+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"\bU\s*\(\s*([\d,]+(?:\.\d+)?)\s*,\s*([\d,]+(?:\.\d+)?)\s*\)",
            text,
        )
    )
    if uni_m:
        params["distribution"] = "uniform"
        params["demand_min"] = _to_float(uni_m.group(1))
        params["demand_max"] = _to_float(uni_m.group(2))
        raw_matches["distribution"] = uni_m.group(0)
        return params

    if re.search(r"\buniform(?:ly)?\b", text, re.IGNORECASE):
        params["distribution"] = "uniform"
        raw_matches["distribution"] = "uniform"
        return params

    # Discrete demand outcomes
    disc_matches = re.findall(
        r"([\d,]+(?:\.\d+)?)\s*(?:units?\s*)?(?:\(\s*(?:prob(?:ability)?\s*(?:=|is|of)?\s*)?|:\s*|with\s+prob(?:ability)?\s*(?:of\s*)?)(\d+(?:\.\d+)?)\s*(%|percent)?\s*\)?",
        text,
        re.IGNORECASE,
    )
    if len(disc_matches) >= 2:
        outcomes = [
            (_to_float(val_s), _parse_rate_or_prob(prob_s, bool(pct_s)))
            for val_s, prob_s, pct_s in disc_matches
        ]
        if abs(sum(p for _, p in outcomes) - 1.0) <= 0.02:
            params["distribution"] = "discrete"
            params["demand_outcomes"] = outcomes
            raw_matches["distribution"] = f"discrete ({len(outcomes)} outcomes)"
            return params

    # Normal distribution
    params["distribution"] = "normal"
    if re.search(r"\bnormal(?:ly)?\b", text, re.IGNORECASE):
        raw_matches["distribution"] = "normal"

    norm_paren = re.search(
        r"(?:\b[Nn]ormal|\bN)\s*\(\s*(?:mean\s*=\s*|μ\s*=\s*|mu\s*=\s*)?([\d,]+(?:\.\d+)?)\s*,\s*(?:sd\s*=\s*|std\s*=\s*|sigma\s*=\s*|σ\s*=\s*)?([\d,]+(?:\.\d+)?)\s*\)",
        text,
    )
    n_pair = re.search(
        r"(?:mean|average|avg|expected)\s*(?:demand)?\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:units?\s*)?(?:and|,)?\s*(?:a\s+)?(?:demand\s+)?(?:std(?:\.?\s*dev\.?)?|standard\s+deviation|sd)\s*(?:of\s+(?:the\s+)?demand\s*)?(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if norm_paren:
        params["demand_mean"] = _to_float(norm_paren.group(1))
        params["demand_std"] = _to_float(norm_paren.group(2))
        raw_matches["distribution"] = norm_paren.group(0)
    elif n_pair:
        params["demand_mean"] = _to_float(n_pair.group(1))
        params["demand_std"] = _to_float(n_pair.group(2))
        raw_matches["distribution"] = n_pair.group(0)
    else:
        mu_patterns = [
            r"(?:mean|average|avg|expected)\s*(?:daily|weekly|monthly|seasonal|period)?\s*demand\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"demand\s+(?:has\s+(?:a\s+)?|(?:is\s+)?normally\s+distributed\s+with\s+(?:a\s+)?)?(?:mean|average|avg|expected\s+value)\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"(?:mean|average|avg)\s*(?:of|is|=|:)\s*([\d,]+(?:\.\d+)?)",
            r"(?:μ|mu)\s*(?:=|is|of)\s*([\d,]+(?:\.\d+)?)",
        ]
        for pat in mu_patterns:
            mu_m = re.search(pat, text, re.IGNORECASE)
            if mu_m:
                params["demand_mean"] = _to_float(mu_m.group(1))
                raw_matches["demand_mean"] = mu_m.group(0).strip()
                break

        sd_patterns = [
            r"(?:(?:daily|weekly|monthly|seasonal|period)\s+)?(?:demand\s+[-]?)?(?:std\.?\s*dev\.?|standard\s+deviation|sd)\s*(?:of\s+(?:the\s+)?(?:daily|weekly|monthly|seasonal|period\s+)?demand\s*)?(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
            r"(?:σ|sigma)\s*(?:=|is|of)\s*([\d,]+(?:\.\d+)?)",
        ]
        for pat in sd_patterns:
            sd_m = re.search(pat, text, re.IGNORECASE)
            if sd_m:
                params["demand_std"] = _to_float(sd_m.group(1))
                raw_matches["demand_std"] = sd_m.group(0).strip()
                break

    return params


def _apply_explicit_constant_flags(
    question: str,
    params: Dict[str, Any],
    raw_matches: Dict[str, Any],
) -> None:
    if "demand_std" not in params and re.search(
        r"(?:constant|deterministic|fixed)\s+(?:daily|weekly|monthly|annual)?\s*demand"
        r"|demand\s+(?:is\s+)?(?:constant|deterministic|fixed)"
        r"|no\s+demand\s+(?:variability|uncertainty)",
        question,
        re.IGNORECASE,
    ):
        params["demand_std"] = 0.0
        raw_matches["demand_std"] = "0.0 (explicit constant demand)"
    if "lead_time_std" not in params and re.search(
        r"(?:constant|deterministic|fixed)\s+lead\s*time"
        r"|lead\s*time\s+(?:is\s+)?(?:constant|deterministic|fixed)"
        r"|no\s+lead[- ]time\s+(?:variability|uncertainty)",
        question,
        re.IGNORECASE,
    ):
        params["lead_time_std"] = 0.0
        raw_matches["lead_time_std"] = "0.0 (explicit constant lead time)"


def extract_inputs_rule_based(question: str, model_id: str) -> InputExtraction:
    """Extract structured tool inputs and missing-field diagnostics for the identified model."""
    assumptions: List[str] = []
    conversions: List[str] = []
    raw_matches: Dict[str, Any] = {}
    warnings: List[str] = []
    missing: List[str] = []

    if model_id == "eoq":
        params = _extract_annual_demand_and_costs(question, assumptions, conversions, raw_matches)
        if "annual_demand" not in params:
            missing.append("annual_demand")
        if "ordering_cost" not in params:
            missing.append("ordering_cost")
        has_h = "holding_cost" in params or (
            "holding_rate" in params and "unit_cost" in params
        )
        if not has_h:
            missing.append("holding_cost (or holding_rate + unit_cost)")

    elif model_id == "quantity_discount":
        params = _extract_annual_demand_and_costs(question, assumptions, conversions, raw_matches)
        params.pop("unit_cost", None)  # Price comes from tiers
        tiers = _extract_quantity_discount_tiers(question)
        if tiers:
            params["tiers"] = tiers
            raw_matches["tiers"] = f"{len(tiers)} tiers"
        else:
            missing.append("tiers (quantity ranges and unit prices)")
        if re.search(r"incremental", question, re.IGNORECASE):
            params["discount_type"] = "incremental"
        else:
            params["discount_type"] = "all_units"
        if "annual_demand" not in params:
            missing.append("annual_demand")
        if "ordering_cost" not in params:
            missing.append("ordering_cost")
        if "holding_rate" not in params and "holding_cost" not in params:
            missing.append("holding_rate (or holding_cost)")

    elif model_id == "safety_stock_rop":
        params = _extract_demand_and_lead_time_stats(question, conversions, raw_matches)
        _extract_service_level_and_z(question, params, raw_matches)
        _apply_explicit_constant_flags(question, params, raw_matches)

        oq_m = re.search(
            r"(?:order\s*quantity|eoq|\bQ\b)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            question,
            re.IGNORECASE,
        )
        if oq_m:
            params["order_quantity"] = _to_float(oq_m.group(1))
            raw_matches["order_quantity"] = oq_m.group(0)

        has_direct_lt = "lead_time_demand_mean" in params and "lead_time_demand_std" in params
        is_deterministic = (
            params.get("demand_std") == 0.0 and params.get("lead_time_std") == 0.0
        ) or (has_direct_lt and params.get("lead_time_demand_std") == 0.0)
        if not has_direct_lt:
            if "demand_mean" not in params:
                missing.append("demand_mean")
            if "lead_time" not in params:
                missing.append("lead_time")
            if "demand_std" not in params and "lead_time_std" not in params:
                missing.append("demand_std (or lead_time_std)")
        if not is_deterministic and "service_level" not in params and "z_score" not in params:
            missing.append("service_level (or z_score)")

    elif model_id == "fixed_period_review":
        params = _extract_demand_and_lead_time_stats(question, conversions, raw_matches)
        _extract_service_level_and_z(question, params, raw_matches)
        _apply_explicit_constant_flags(question, params, raw_matches)
        working_days = _extract_working_days(question)

        rev_m = (
            re.search(
                r"(?:review[- ](?:period|interval|cycle)|inspection[- ](?:period|interval)|order\s*cycle)\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:reviewed|check(?:ed)?|order(?:ed)?|restock(?:ed)?|replenish(?:ed)?)\s*every\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"\bevery\s+([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:^|[,\s;(])[TP]\s*=\s*([\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)?",
                question,
            )
        )
        if rev_m:
            rev_val = _to_float(rev_m.group(1))
            rev_unit = _normalize_time_unit(rev_m.group(2), default=params.get("time_unit", "period"))
            base_unit = params.get("time_unit", "period")
            if base_unit == "period" and rev_unit != "period":
                base_unit = rev_unit
                params["time_unit"] = base_unit
            elif base_unit != "period" and rev_unit != "period" and rev_unit != base_unit:
                conv_rev = _convert_time_length(rev_val, rev_unit, base_unit, working_days)
                conversions.append(
                    f"Converted review period {rev_val:g} {rev_unit}(s) -> {conv_rev:g} {base_unit}(s)"
                )
                rev_val = conv_rev
            params["review_period"] = rev_val
            raw_matches["review_period"] = rev_m.group(0).strip()

        # Explicit safety stock (optional alternative to std dev + service level)
        ss_m = (
            re.search(
                r"(?:fixed\s+|target\s+|desired\s+|required\s+)?safety\s*stock\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)(?!\s*(?:%|percent))(?:\s*units?)?",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"([\d,]+(?:\.\d+)?)\s*(?:units?\s+)?(?:of\s+|in\s+)?safety\s*stock\b",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"\bSS\s*(?:=|is|of)\s*([\d,]+(?:\.\d+)?)(?!\s*(?:%|percent))",
                question,
                re.IGNORECASE,
            )
        )
        if ss_m:
            params["safety_stock"] = _to_float(ss_m.group(1))
            raw_matches["safety_stock"] = ss_m.group(0).strip()

        # Inventory position fields
        oh_m = (
            re.search(
                r"([\d,]+(?:\.\d+)?)\s*(?:units?\s*)?(?:currently\s+)?(?:on[- ]hand|in\s*stock|stock\s*on[- ]hand)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:on[- ]hand(?:\s*(?:inventory|stock))?|stock\s+on[- ]hand|current\s*(?:stock|inventory)|in\s*stock|available\s+(?:stock|inventory))\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"\bOH\s*(?:=|is|of)\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
        )
        if oh_m:
            params["on_hand_inventory"] = _to_float(oh_m.group(1))
            raw_matches["on_hand_inventory"] = oh_m.group(0).strip()
        elif re.search(
            r"\b(?:no|zero)\s+(?:inventory|stock|units?)\s+on[- ]hand|\bzero\s+on[- ]hand",
            question,
            re.IGNORECASE,
        ):
            params["on_hand_inventory"] = 0.0
            raw_matches["on_hand_inventory"] = "0 (explicit zero)"

        oo_m = (
            re.search(
                r"([\d,]+(?:\.\d+)?)\s*(?:units?\s*)?(?:currently\s+)?(?:on[- ]order|in\s*transit|scheduled\s*receipts?|open\s*orders?|pending\s*orders?)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:on[- ]order(?:\s*(?:inventory|units?|stock))?|scheduled\s*receipts?|in[- ]transit(?:\s*(?:inventory|units?))?|open\s*orders?|pending\s*orders?)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"\bOO\s*(?:=|is|of)\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
        )
        if oo_m:
            params["on_order_inventory"] = _to_float(oo_m.group(1))
            raw_matches["on_order_inventory"] = oo_m.group(0).strip()
        elif re.search(
            r"\b(?:no|zero)\s+(?:units?\s+)?(?:on[- ]order|in\s*transit|scheduled\s*receipts?|open\s*orders?|pending\s*orders?)",
            question,
            re.IGNORECASE,
        ):
            params["on_order_inventory"] = 0.0
            raw_matches["on_order_inventory"] = "0 (explicit zero)"

        bo_m = (
            re.search(
                r"([\d,]+(?:\.\d+)?)\s*(?:units?\s*)?(?:in\s+|on\s+|of\s+)?(?:backorders?|backordered|unfilled\s*orders?)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:backorders?|backordered(?:\s*units?)?|unfilled\s*orders?)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"\bBO\s*(?:=|is|of)\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
        )
        if bo_m:
            params["backorders"] = _to_float(bo_m.group(1))
            raw_matches["backorders"] = bo_m.group(0).strip()
        elif re.search(
            r"\b(?:no|zero)\s+(?:units?\s+)?(?:in\s+|on\s+)?(?:backorders?|backordered|unfilled\s*orders?)",
            question,
            re.IGNORECASE,
        ):
            params["backorders"] = 0.0
            raw_matches["backorders"] = "0 (explicit zero)"

        ip_m = (
            re.search(
                r"(?:net\s+)?inventory\s*position\s*(?:is|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"([\d,]+(?:\.\d+)?)\s*(?:units?\s*)?(?:net\s+)?inventory\s*position",
                question,
                re.IGNORECASE,
            )
            or re.search(
                r"\bIP\s*(?:=|is|of)\s*([\d,]+(?:\.\d+)?)",
                question,
                re.IGNORECASE,
            )
        )
        if ip_m:
            params["inventory_position"] = _to_float(ip_m.group(1))
            raw_matches["inventory_position"] = ip_m.group(0).strip()

        if "review_period" not in params:
            missing.append("review_period")
        if "lead_time" not in params:
            missing.append("lead_time")
        if "demand_mean" not in params:
            missing.append("demand_mean")
        if "safety_stock" not in params:
            if "demand_std" not in params and "lead_time_std" not in params:
                missing.append("demand_std (or lead_time_std or safety_stock)")
            is_deterministic = params.get("demand_std") == 0.0 and params.get("lead_time_std", 0.0) == 0.0
            if not is_deterministic and "service_level" not in params and "z_score" not in params:
                missing.append("service_level (or z_score)")

    elif model_id == "newsvendor":
        params = _extract_newsvendor_inputs(question, assumptions, raw_matches)
        has_direct_cu_co = "underage_cost" in params and "overage_cost" in params
        has_price_cost = "selling_price" in params and "unit_cost" in params
        if not has_direct_cu_co and not has_price_cost:
            if "underage_cost" in params or "overage_cost" in params:
                if "underage_cost" not in params:
                    missing.append("underage_cost")
                if "overage_cost" not in params:
                    missing.append("overage_cost")
            elif "selling_price" in params and "unit_cost" not in params:
                missing.append("unit_cost")
            elif "unit_cost" in params and "selling_price" not in params:
                missing.append("selling_price")
            else:
                missing.append("selling_price and unit_cost (or underage_cost and overage_cost)")
        dist = params.get("distribution", "normal")
        if dist == "normal":
            if "demand_mean" not in params:
                missing.append("demand_mean")
            if "demand_std" not in params:
                missing.append("demand_std")
        elif dist == "uniform":
            if "demand_min" not in params or "demand_max" not in params:
                missing.append("demand_min and demand_max")
        elif dist == "discrete":
            if "demand_outcomes" not in params:
                missing.append("demand_outcomes")
    else:
        raise KeyError(f"Unsupported model_id: {model_id!r}")

    return InputExtraction(
        params=params,
        missing_required=missing,
        assumptions=assumptions,
        unit_conversions=conversions,
        raw_matches=raw_matches,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Optional LLM NLU Adapter (only used if enabled and API key is configured)
# ---------------------------------------------------------------------------

def _llm_identify_and_extract(question: str) -> Optional[Tuple[ModelIdentification, InputExtraction]]:
    """Use OpenAI or Google GenAI ONLY for model identification and JSON extraction.

    Returns None if no API key or SDK is available, allowing seamless fallback
    to the deterministic rule-based parser. Never performs calculations.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not openai_key and not gemini_key:
        return None

    schema_prompt = (
        "You are a Supply Chain Analytics NLU parser. DO NOT compute any math results.\n"
        "Classify the user's inventory question into ONE model_id from:\n"
        "  - eoq\n  - quantity_discount\n  - safety_stock_rop\n  - fixed_period_review\n  - newsvendor\n"
        "Extract numeric input parameters into JSON matching these field names:\n"
        "  eoq: annual_demand, ordering_cost, holding_cost, holding_rate, unit_cost, working_days_per_year\n"
        "  quantity_discount: annual_demand, ordering_cost, holding_rate, holding_cost, discount_type, "
        "tiers=[{min_qty, max_qty, max_exclusive, unit_price}]\n"
        "  safety_stock_rop: demand_mean, demand_std, lead_time, lead_time_std, lead_time_demand_mean, "
        "lead_time_demand_std, service_level, z_score, service_level_type, order_quantity, holding_cost, time_unit\n"
        "  fixed_period_review: review_period, lead_time, demand_mean, demand_std, lead_time_std, "
        "service_level, z_score, safety_stock, on_hand_inventory, on_order_inventory, backorders, inventory_position, time_unit\n"
        "  newsvendor: selling_price, unit_cost, salvage_value, shortage_penalty, underage_cost, overage_cost, "
        "distribution ('normal'|'uniform'|'discrete'), demand_mean, demand_std, demand_min, demand_max, demand_outcomes\n"
        "Return strict JSON: {model_id, confidence, reasoning, params, missing_required, assumptions}."
    )

    raw_json: Optional[str] = None
    try:
        if openai_key:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": schema_prompt},
                    {"role": "user", "content": question},
                ],
            )
            raw_json = resp.choices[0].message.content
        elif gemini_key:
            from google import genai  # type: ignore

            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=f"{schema_prompt}\n\nQuestion:\n{question}",
            )
            raw_json = resp.text
    except Exception:
        return None

    if not raw_json:
        return None

    try:
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
        data = json.loads(cleaned)
        model_id = data.get("model_id")
        if model_id not in TOOL_REGISTRY:
            return None
        ident = ModelIdentification(
            model_id=model_id,
            model_name=TOOL_REGISTRY[model_id]["name"],
            confidence=float(data.get("confidence", 0.9)),
            matched_signals=["LLM semantic classification"],
            scores={model_id: float(data.get("confidence", 0.9))},
            reasoning=str(data.get("reasoning", "Identified by LLM NLU parser.")),
            method="llm",
        )
        extraction = InputExtraction(
            params=dict(data.get("params") or {}),
            missing_required=list(data.get("missing_required") or []),
            assumptions=list(data.get("assumptions") or []),
            method="llm",
        )
        return ident, extraction
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 7: Explanation Builder
# ---------------------------------------------------------------------------

def _fmt_num(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val:,.{decimals}f}"


def _ceil_units(val: Optional[float]) -> Optional[int]:
    """Round up to the next whole unit for indivisible-unit operational recommendations.

    Guards against floating-point representation noise (e.g. 500.00000000000006 -> 500).
    """
    if val is None:
        return None
    v = float(val)
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return int(math.ceil(v))


def _is_fractional_unit(val: Optional[float]) -> bool:
    """Return True when a positive inventory quantity has a non-negligible fractional component."""
    if val is None:
        return False
    v = float(val)
    return abs(v - round(v)) >= 1e-9


def build_explanation(
    question: str,
    ident: ModelIdentification,
    extraction: InputExtraction,
    selection: ToolSelection,
    result: Dict[str, Any],
    verification: VerificationReport,
) -> str:
    """Create an educational, student-verifiable explanation with all 8 required calculation sections."""
    lines: List[str] = [
        f"### Model Selected: {ident.model_name}",
        f"**Why:** {ident.reasoning}",
        "",
        f"**Python Tool:** `{selection.tool_name}`",
        "",
        "### 1. Inputs Extracted",
    ]

    inputs_to_show = result.get("inputs_extracted") or result.get("inputs") or selection.arguments
    for key, val in inputs_to_show.items():
        if val is None:
            continue
        if key == "tiers" and isinstance(val, list):
            lines.append("- **tiers:**")
            for t in val:
                hi = "∞" if t.get("max_qty") is None else f"{t['max_qty']:g}"
                lines.append(f"  - Qty {t['min_qty']:g} to {hi}: ${_fmt_num(t['unit_price'])}/unit")
        else:
            lines.append(f"- **{key}:** `{val}`")

    if extraction.unit_conversions:
        lines.append("")
        lines.append("**Unit Conversions Applied:**")
        for conv in extraction.unit_conversions:
            lines.append(f"- {conv}")

    # --- 2. Variable Definitions ---
    lines.extend(["", "### 2. Variable Definitions"])
    var_defs = result.get("variable_definitions", {})
    if var_defs:
        for sym, defn in var_defs.items():
            lines.append(f"- **`{sym}`:** {defn}")
    else:
        lines.append("- Standard inventory decision variables applied.")

    # --- 3. Formula ---
    lines.extend(["", "### 3. Formula"])
    formulas = result.get("formula", {})
    if formulas:
        for f_name, f_expr in formulas.items():
            title = f_name.replace("_", " ").title()
            lines.append(f"- **{title}:** `{f_expr}`")
    else:
        lines.append("- Deterministic closed-form equations applied.")

    # --- 4. Formula with Substituted Values ---
    lines.extend(["", "### 4. Formula with Substituted Values"])
    sub_formulas = result.get("formula_with_substituted_values", {})
    if sub_formulas:
        for s_name, s_val in sub_formulas.items():
            title = s_name.replace("_", " ").title()
            if isinstance(s_val, list):
                lines.append(f"- **{title}:**")
                for item in s_val:
                    lines.append(f"  - `{item}`")
            else:
                lines.append(f"- **{title}:** `{s_val}`")
    else:
        lines.append("- Evaluated with extracted input parameters.")

    # --- 5. Intermediate Calculations ---
    lines.extend(["", "### 5. Intermediate Calculations"])
    m = ident.model_id
    inter = result.get("intermediate_calculations", {})

    if m == "eoq":
        cycle_detail = (
            f"{_fmt_num(inter.get('cycle_time_days'), 1)} operating days"
            if inter.get("cycle_time_days") is not None
            else f"{_fmt_num(inter.get('cycle_time_years'), 4)} years"
        )
        lines.extend(
            [
                f"- **Holding Cost Basis:** `{inter.get('holding_cost_basis', 'N/A')}` (effective H = ${_fmt_num(inter.get('effective_holding_cost'), 4)}/unit/year)",
                f"- **Numerator (`2 * D * S`):** {_fmt_num(inter.get('numerator_2DS'))}",
                f"- **Ratio (`(2 * D * S) / H`):** {_fmt_num(inter.get('ratio_2DS_over_H'))}",
                f"- **Replenishment Orders per Year (`D / Q*`):** {_fmt_num(inter.get('orders_per_year'))} orders/year",
                f"- **Order Cycle Time (`Q* / D`):** {cycle_detail}",
                f"- **Average Cycle Inventory (`Q* / 2`):** {_fmt_num(inter.get('average_inventory'))} units",
                f"- **Annual Ordering Cost:** ${_fmt_num(inter.get('annual_ordering_cost'))}",
                f"- **Annual Holding Cost:** ${_fmt_num(inter.get('annual_holding_cost'))}",
            ]
        )
        if inter.get("annual_purchase_cost") is not None:
            lines.append(f"- **Annual Purchase Cost:** ${_fmt_num(inter.get('annual_purchase_cost'))}")

    elif m == "quantity_discount":
        lines.extend(
            [
                f"- **Discount Structure:** `{inter.get('holding_cost_mode', 'N/A')}`",
                f"- **Orders per Year:** {_fmt_num(inter.get('orders_per_year'))} orders/year",
                f"- **Annual Purchase Cost:** ${_fmt_num(inter.get('annual_purchase_cost'))}",
                f"- **Annual Ordering Cost:** ${_fmt_num(inter.get('annual_ordering_cost'))}",
                f"- **Annual Holding Cost:** ${_fmt_num(inter.get('annual_holding_cost'))}",
                "",
                "**Tier-by-Tier Evaluation Summary:**",
            ]
        )
        for row in inter.get("tier_evaluations", []):
            hi = "∞" if row.get("max_qty") is None else f"{row['max_qty']:g}"
            q_str = _fmt_num(row["candidate_quantity"]) if row.get("candidate_quantity") is not None else "Infeasible"
            tc_str = f"${_fmt_num(row['total_annual_cost'])}" if row.get("total_annual_cost") is not None else "N/A"
            lines.append(
                f"- Tier {row['tier_index']} [{row['min_qty']:g}–{hi}, ${_fmt_num(row['unit_price'])}]: "
                f"Candidate Q = {q_str} ({row['feasibility']}), Total Annual Cost = {tc_str}"
            )

    elif m == "safety_stock_rop":
        lines.extend(
            [
                f"- **Variability Case:** `{inter.get('formula_case', 'N/A')}`",
                f"- **Expected Lead-Time Demand (`μ_LT = d * L`):** {_fmt_num(inter.get('expected_lead_time_demand'))} units",
                f"- **Lead-Time Demand Variance:** {_fmt_num(inter.get('lead_time_demand_variance'), 4)}",
                f"- **Lead-Time Demand Std Dev (`σ_LT`):** {_fmt_num(inter.get('lead_time_demand_std'), 4)} units",
                f"- **Safety Factor (`z`):** {_fmt_num(inter.get('z_score'), 4) if inter.get('z_score') is not None else 'N/A (deterministic)'}",
                f"- **Cycle Service Level (CSL):** {_fmt_num((inter.get('cycle_service_level') or 0.0) * 100.0, 2)}%",
                f"- **Stockout Risk (`1 - CSL`):** {_fmt_num((inter.get('stockout_probability') or 0.0) * 100.0, 2)}%",
            ]
        )
        if inter.get("achieved_fill_rate") is not None:
            lines.append(f"- **Achieved Fill Rate:** {_fmt_num(inter['achieved_fill_rate'] * 100.0, 2)}%")
        if inter.get("annual_safety_stock_holding_cost") is not None:
            lines.append(
                f"- **Annual Safety Stock Holding Cost:** ${_fmt_num(inter['annual_safety_stock_holding_cost'])}"
            )

    elif m == "fixed_period_review":
        unit = result.get("time_unit", "period")
        lines.extend(
            [
                f"- **Protection Interval (`T + L`):** {_fmt_num(inter.get('protection_period'), 2)} {unit}(s)",
                f"- **Expected Protection Period Demand (`μ_(T+L)`):** {_fmt_num(inter.get('expected_protection_demand'))} units",
                f"- **Protection Period Std Dev (`σ_(T+L)`):** {_fmt_num(inter.get('protection_demand_std'), 4) if inter.get('protection_demand_std') is not None else 'N/A (explicit fixed safety stock)'}",
                f"- **Safety Factor (`z`):** {_fmt_num(inter.get('z_score'), 4) if inter.get('z_score') is not None else 'N/A'}",
                f"- **Cycle Service Level:** {_fmt_num((inter.get('cycle_service_level') or 0.0) * 100.0, 2) if inter.get('cycle_service_level') is not None else 'N/A'}%",
                f"- **Evaluated Inventory Position (`IP`):** {_fmt_num(inter.get('inventory_position')) if inter.get('inventory_position') is not None else 'N/A'} units",
                f"- **Average Cycle Inventory (`d * T / 2`):** {_fmt_num(inter.get('average_cycle_inventory'))} units",
                f"- **Expected Average Inventory (`d * T / 2 + SS`):** {_fmt_num(inter.get('expected_average_inventory'))} units",
            ]
        )

    elif m == "newsvendor":
        lines.extend(
            [
                f"- **Underage Cost (`Cu`):** ${_fmt_num(inter.get('underage_cost'))}/unit",
                f"- **Overage Cost (`Co`):** ${_fmt_num(inter.get('overage_cost'))}/unit",
                f"- **Critical Ratio (`Cu / (Cu + Co)`):** {_fmt_num((inter.get('critical_ratio') or 0.0) * 100.0, 2)}% ({inter.get('critical_ratio', 0.0):.4f})",
                f"- **In-Stock Probability:** {_fmt_num((inter.get('in_stock_probability') or 0.0) * 100.0, 2)}%",
                f"- **Volumetric Fill Rate:** {_fmt_num((inter.get('fill_rate') or 0.0) * 100.0, 2)}%",
                f"- **Expected Sales:** {_fmt_num(inter.get('expected_sales'))} units",
                f"- **Expected Leftover Inventory:** {_fmt_num(inter.get('expected_leftover'))} units",
                f"- **Expected Lost Sales:** {_fmt_num(inter.get('expected_lost_sales'))} units",
            ]
        )
        if inter.get("expected_profit") is not None:
            lines.append(f"- **Expected Profit:** ${_fmt_num(inter.get('expected_profit'))}")
        else:
            lines.append(f"- **Expected Mismatch Cost:** ${_fmt_num(inter.get('expected_mismatch_cost'))}")

    # --- 6. Final Answer ---
    lines.extend(["", "### 6. Final Answer"])
    ans = result.get("final_answer", {})

    if m == "eoq":
        q_exact = ans.get("optimal_order_quantity")
        q_whole = _ceil_units(q_exact)
        lines.extend(
            [
                f"- **Optimal Order Quantity (EOQ / `Q*`):** **{_fmt_num(q_exact)} units**",
                f"- **Total Relevant Inventory Cost (Ordering + Holding):** **${_fmt_num(ans.get('total_relevant_cost'))}/year**",
            ]
        )
        if inter.get("annual_purchase_cost") is not None and ans.get("total_annual_cost") is not None:
            lines.append(
                f"- **Total Annual Cost (incl. Purchase):** **${_fmt_num(ans.get('total_annual_cost'))}/year**"
            )
        else:
            lines.append(
                "- **Total Annual Cost (incl. Purchase):** *Not calculated (no unit purchase price provided)*"
            )
        lines.extend(
            [
                "",
                f"**Recommendation:** Order **{_fmt_num(q_exact)} units** per replenishment order to minimize total relevant inventory costs.",
            ]
        )
        if q_whole is not None:
            if _is_fractional_unit(q_exact):
                lines.append(
                    f"**Practical Whole-Unit Recommendation (if units are indivisible):** "
                    f"Order **{q_whole:,} units** per replenishment order (rounded up from exact `{_fmt_num(q_exact)}` units)."
                )
            else:
                lines.append(
                    f"**Practical Whole-Unit Recommendation (if units are indivisible):** "
                    f"Order **{q_whole:,} units** per replenishment order (exact whole-unit quantity)."
                )

    elif m == "quantity_discount":
        q_exact = ans.get("optimal_order_quantity")
        q_whole = _ceil_units(q_exact)
        # Ensure rounded-up recommendation stays within the winning tier's feasible upper bound if finite
        win_tier_idx = ans.get("optimal_tier")
        tier_rows = inter.get("tier_evaluations") or result.get("tier_evaluations") or []
        win_tier = next((t for t in tier_rows if t.get("tier_index") == win_tier_idx), None)
        if q_whole is not None and win_tier is not None and win_tier.get("max_qty") is not None:
            max_bound = float(win_tier["max_qty"])
            if win_tier.get("max_exclusive"):
                max_bound = max_bound - 1.0
            if q_whole > max_bound:
                q_whole = int(math.floor(max_bound))

        lines.extend(
            [
                f"- **Optimal Order Quantity (`Q*`):** **{_fmt_num(q_exact)} units**",
                f"- **Winning Price Tier:** **Tier {win_tier_idx}** (@ ${_fmt_num(ans.get('optimal_unit_price'))}/unit)",
                f"- **Minimum Total Annual Cost:** **${_fmt_num(ans.get('total_annual_cost'))}/year**",
                "",
                f"**Recommendation:** Order **{_fmt_num(q_exact)} units** from Tier {win_tier_idx} "
                f"to achieve the lowest total annual cost of **${_fmt_num(ans.get('total_annual_cost'))}/year**.",
            ]
        )
        if q_whole is not None:
            if _is_fractional_unit(q_exact):
                lines.append(
                    f"**Practical Whole-Unit Recommendation (if units are indivisible):** "
                    f"Order **{q_whole:,} units** from Tier {win_tier_idx} (rounded up from exact `{_fmt_num(q_exact)}` units within the feasible tier range)."
                )
            else:
                lines.append(
                    f"**Practical Whole-Unit Recommendation (if units are indivisible):** "
                    f"Order **{q_whole:,} units** from Tier {win_tier_idx} (exact whole-unit quantity)."
                )

    elif m == "safety_stock_rop":
        ss_exact = ans.get("safety_stock")
        rop_exact = ans.get("reorder_point")
        ss_whole = _ceil_units(ss_exact)
        rop_whole = _ceil_units(rop_exact)
        lines.extend(
            [
                f"- **Safety Stock (`SS`):** **{_fmt_num(ss_exact)} units**",
                f"- **Reorder Point (`ROP`):** **{_fmt_num(rop_exact)} units**",
                "",
                f"**Recommendation:** Maintain **{_fmt_num(ss_exact)} units** of safety stock "
                f"and place a replenishment order whenever inventory drops to **{_fmt_num(rop_exact)} units**.",
            ]
        )
        if ss_whole is not None and rop_whole is not None:
            if _is_fractional_unit(ss_exact) or _is_fractional_unit(rop_exact):
                lines.append(
                    f"**Practical Whole-Unit Operational Recommendation (if units are indivisible):** "
                    f"Maintain **{ss_whole:,} units** of safety stock (rounded up from `{_fmt_num(ss_exact)}`) "
                    f"and set the reorder point to **{rop_whole:,} units** (rounded up from `{_fmt_num(rop_exact)}`) "
                    f"to preserve the target cycle service level."
                )
            else:
                lines.append(
                    f"**Practical Whole-Unit Operational Recommendation (if units are indivisible):** "
                    f"Maintain **{ss_whole:,} units** of safety stock and set the reorder point to **{rop_whole:,} units** "
                    f"(exact whole-unit quantities)."
                )

    elif m == "fixed_period_review":
        ss_exact = ans.get("safety_stock")
        target_exact = ans.get("target_inventory_level")
        order_exact = ans.get("order_quantity")
        ss_whole = _ceil_units(ss_exact)
        target_whole = _ceil_units(target_exact)
        order_whole = _ceil_units(order_exact)

        lines.extend(
            [
                f"- **Safety Stock (`SS`):** **{_fmt_num(ss_exact)} units**",
                f"- **Order-Up-To Target Inventory Level (`S`):** **{_fmt_num(target_exact)} units**",
            ]
        )
        if order_exact is not None:
            lines.append(
                f"- **Order Quantity to Place Now (`Q`):** **{_fmt_num(order_exact)} units**"
            )
            rec = (
                f"Set order-up-to level to **{_fmt_num(target_exact)} units** (including "
                f"**{_fmt_num(ss_exact)} units** safety stock) and place an order for "
                f"**{_fmt_num(order_exact)} units** immediately."
            )
            any_frac = (
                _is_fractional_unit(ss_exact)
                or _is_fractional_unit(target_exact)
                or _is_fractional_unit(order_exact)
            )
            if any_frac:
                whole_rec = (
                    f"**Practical Whole-Unit Operational Recommendation (if units are indivisible):** "
                    f"Maintain **{ss_whole:,} units** of safety stock (rounded up from `{_fmt_num(ss_exact)}`), "
                    f"set the order-up-to target level to **{target_whole:,} units** (rounded up from `{_fmt_num(target_exact)}`), "
                    f"and place an order for **{order_whole:,} units** (rounded up from `{_fmt_num(order_exact)}`) "
                    f"to meet or exceed the target service level."
                )
            else:
                whole_rec = (
                    f"**Practical Whole-Unit Operational Recommendation (if units are indivisible):** "
                    f"Maintain **{ss_whole:,} units** of safety stock, set the order-up-to target level to "
                    f"**{target_whole:,} units**, and place an order for **{order_whole:,} units** "
                    f"(exact whole-unit quantities)."
                )
        else:
            rec = (
                f"Set order-up-to target level to **{_fmt_num(target_exact)} units** "
                f"(including **{_fmt_num(ss_exact)} units** safety stock)."
            )
            any_frac = _is_fractional_unit(ss_exact) or _is_fractional_unit(target_exact)
            if any_frac:
                whole_rec = (
                    f"**Practical Whole-Unit Operational Recommendation (if units are indivisible):** "
                    f"Maintain **{ss_whole:,} units** of safety stock (rounded up from `{_fmt_num(ss_exact)}`) "
                    f"and set the order-up-to target level to **{target_whole:,} units** (rounded up from `{_fmt_num(target_exact)}`) "
                    f"to meet or exceed the target service level."
                )
            else:
                whole_rec = (
                    f"**Practical Whole-Unit Operational Recommendation (if units are indivisible):** "
                    f"Maintain **{ss_whole:,} units** of safety stock and set the order-up-to target level to "
                    f"**{target_whole:,} units** (exact whole-unit quantities)."
                )
        lines.extend(["", f"**Recommendation:** {rec}", whole_rec])

    elif m == "newsvendor":
        lines.extend(
            [
                f"- **Critical Ratio:** **{ans.get('critical_ratio', 0.0):.4f}** ({_fmt_num((ans.get('critical_ratio') or 0.0) * 100.0, 2)}%)",
                f"- **Recommended Order Quantity (`Q*`):** **{_fmt_num(ans.get('recommended_order_quantity'))} units** "
                f"(nearest integer: {ans.get('rounded_order_quantity', 0):,} units)",
            ]
        )
        if ans.get("expected_profit") is not None:
            lines.append(f"- **Expected Profit:** **${_fmt_num(ans.get('expected_profit'))}**")
        lines.extend(
            [
                "",
                f"**Recommendation:** Order **{ans.get('rounded_order_quantity', 0):,} units** "
                f"(exact: {_fmt_num(ans.get('recommended_order_quantity'))}) to balance stockout risk and excess inventory cost.",
            ]
        )

    # --- 7. Verification Checks ---
    v_passed = result.get("verification_passed", verification.passed)
    v_checks = result.get("verification_checks") or verification.checks
    lines.extend(["", "### 7. Verification Checks", f"**Status:** {'PASSED' if v_passed else 'FAILED'}"])
    for chk in v_checks:
        badge = "PASS" if chk.get("passed") else "FAIL"
        lines.append(f"- [{badge}] **{chk.get('name')}:** `{chk.get('detail')}`")

    # --- 8. Assumptions ---
    lines.extend(["", "### 8. Assumptions"])
    merged_assumptions = list(
        dict.fromkeys([*extraction.assumptions, *(result.get("assumptions") or [])])
    )
    if merged_assumptions:
        for assump in merged_assumptions:
            lines.append(f"- {assump}")
    else:
        lines.append("- Standard inventory management assumptions apply.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-End 7-Stage Agent Orchestrator
# ---------------------------------------------------------------------------

class InventoryDecisionAgent:
    """Orchestrates Customer Question -> Identify -> Extract -> Select Tool -> Calculate -> Verify -> Explain."""

    def __init__(self, parser_mode: str = "auto") -> None:
        if parser_mode not in {"auto", "rule", "llm"}:
            raise ValueError("parser_mode must be 'auto', 'rule', or 'llm'.")
        self.parser_mode = parser_mode

    def identify_model(self, question: str) -> ModelIdentification:
        if self.parser_mode in {"auto", "llm"}:
            llm_out = _llm_identify_and_extract(question)
            if llm_out is not None:
                return llm_out[0]
        return identify_model_rule_based(question)

    def extract_inputs(self, question: str, model_id: str) -> InputExtraction:
        if self.parser_mode in {"auto", "llm"}:
            llm_out = _llm_identify_and_extract(question)
            if llm_out is not None and llm_out[0].model_id == model_id:
                return llm_out[1]
        return extract_inputs_rule_based(question, model_id)

    def select_tool(
        self, model_id: str, extraction: InputExtraction
    ) -> ToolSelection:
        meta = TOOL_REGISTRY[model_id]
        ready = len(extraction.missing_required) == 0
        reason = (
            f"All required inputs extracted for {meta['tool_name']}."
            if ready
            else f"Missing required inputs: {', '.join(extraction.missing_required)}."
        )
        return ToolSelection(
            model_id=model_id,
            model_name=meta["name"],
            tool_name=meta["tool_name"],
            arguments=dict(extraction.params),
            ready=ready,
            reason=reason,
        )

    def calculate(self, selection: ToolSelection) -> Dict[str, Any]:
        fn = get_tool(selection.model_id)
        return fn(**selection.arguments)

    def verify(
        self, model_id: str, inputs: Dict[str, Any], result: Dict[str, Any]
    ) -> VerificationReport:
        if "verification_checks" in result and "verification_passed" in result:
            return VerificationReport(
                passed=result["verification_passed"],
                checks=result["verification_checks"],
            )
        report = verify_calculation(model_id, inputs, result)
        return VerificationReport(passed=report["passed"], checks=report["checks"])

    def run(self, question: str) -> AgentResponse:
        trace: List[Dict[str, Any]] = []
        cleaned_question = question.strip()
        trace.append({"stage": "1. Customer Question", "status": "ok", "summary": cleaned_question})

        if not cleaned_question:
            empty_ident = ModelIdentification(
                model_id=None,
                model_name=None,
                confidence=0.0,
                matched_signals=[],
                scores={},
                reasoning="Empty question provided.",
            )
            return AgentResponse(
                question=question,
                status="needs_clarification",
                identification=empty_ident,
                extraction=None,
                tool_selection=None,
                calculation=None,
                verification=None,
                explanation="Please enter a supply chain inventory question.",
                pipeline_trace=trace,
            )

        # Stage 2: Identify Model
        ident: Optional[ModelIdentification] = None
        extraction: Optional[InputExtraction] = None
        if self.parser_mode in {"auto", "llm"}:
            llm_out = _llm_identify_and_extract(cleaned_question)
            if llm_out is not None:
                ident, extraction = llm_out
        if ident is None:
            ident = identify_model_rule_based(cleaned_question)

        trace.append(
            {
                "stage": "2. Identify Model",
                "status": "ok" if ident.model_id else "needs_clarification",
                "summary": f"{ident.model_name or 'Unidentified'} (confidence={ident.confidence:.2f}, method={ident.method})",
            }
        )

        if not ident.model_id:
            msg = (
                f"{ident.reasoning}\n\nSupported models:\n"
                + "\n".join(f"- **{meta['name']}**: {meta['description']}" for meta in TOOL_REGISTRY.values())
            )
            return AgentResponse(
                question=cleaned_question,
                status="needs_clarification",
                identification=ident,
                extraction=None,
                tool_selection=None,
                calculation=None,
                verification=None,
                explanation=msg,
                pipeline_trace=trace,
            )

        # Stage 3: Extract Inputs
        if extraction is None:
            extraction = extract_inputs_rule_based(cleaned_question, ident.model_id)

        trace.append(
            {
                "stage": "3. Extract Inputs",
                "status": "ok" if not extraction.missing_required else "incomplete",
                "summary": f"Extracted {len(extraction.params)} parameters; missing={extraction.missing_required}",
            }
        )

        # Stage 4: Select Python Tool
        selection = self.select_tool(ident.model_id, extraction)
        trace.append(
            {
                "stage": "4. Select Python Tool",
                "status": "ok" if selection.ready else "blocked",
                "summary": f"{selection.tool_name} ({selection.reason})",
            }
        )

        if not selection.ready:
            msg = (
                f"Identified **{ident.model_name}**, but cannot run `{selection.tool_name}` yet because "
                f"the following required inputs are missing:\n"
                + "\n".join(f"- `{item}`" for item in extraction.missing_required)
                + f"\n\nExtracted so far: `{extraction.params}`"
            )
            return AgentResponse(
                question=cleaned_question,
                status="missing_inputs",
                identification=ident,
                extraction=extraction,
                tool_selection=selection,
                calculation=None,
                verification=None,
                explanation=msg,
                pipeline_trace=trace,
            )

        # Stage 5: Calculate (strictly in Python)
        try:
            calc_result = self.calculate(selection)
        except Exception as exc:
            trace.append({"stage": "5. Calculate", "status": "error", "summary": str(exc)})
            return AgentResponse(
                question=cleaned_question,
                status="error",
                identification=ident,
                extraction=extraction,
                tool_selection=selection,
                calculation=None,
                verification=None,
                explanation=f"Calculation failed for `{selection.tool_name}`: {exc}",
                pipeline_trace=trace,
            )

        trace.append(
            {
                "stage": "5. Calculate",
                "status": "ok",
                "summary": f"Executed {selection.tool_name} successfully.",
            }
        )

        # Stage 6: Verify
        verification = self.verify(ident.model_id, selection.arguments, calc_result)
        trace.append(
            {
                "stage": "6. Verify",
                "status": "ok" if verification.passed else "warning",
                "summary": f"{sum(1 for c in verification.checks if c['passed'])}/{len(verification.checks)} checks passed",
            }
        )

        # Stage 7: Explain
        explanation = build_explanation(
            cleaned_question, ident, extraction, selection, calc_result, verification
        )
        trace.append({"stage": "7. Explain", "status": "ok", "summary": "Generated explanation."})

        return AgentResponse(
            question=cleaned_question,
            status="completed",
            identification=ident,
            extraction=extraction,
            tool_selection=selection,
            calculation=calc_result,
            verification=verification,
            explanation=explanation,
            pipeline_trace=trace,
        )
