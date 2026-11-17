"""Pure-Python deterministic inventory decision tools and analytical verification.

Supported models and primary functions:
1. Economic Order Quantity (EOQ): `calculate_eoq`
2. Quantity Discount: `calculate_quantity_discount`
3. Safety Stock and Reorder Point: `calculate_rop`
4. Fixed-Period Review: `calculate_periodic_review`
5. Newsvendor: `calculate_newsvendor`

Every tool validates inputs, refuses to invent missing numerical inputs, and
returns a structured dictionary containing:
  - model
  - tool_name
  - inputs
  - intermediate_calculations
  - final_answer
  - assumptions
alongside flat summary fields for direct access.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Normal distribution & standard normal loss helpers (pure Python, machine precision)
# ---------------------------------------------------------------------------

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def normal_pdf(z: float) -> float:
    """Standard normal probability density function phi(z)."""
    return math.exp(-0.5 * z * z) / _SQRT_2PI


def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z)."""
    return 0.5 * (1.0 + math.erf(z / _SQRT_2))


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF Phi^{-1}(p) accurate to ~1e-15.

    Uses Peter Acklam's rational approximation followed by Newton-Raphson
    refinement using math.erf.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
        raise ValueError(f"Probability p must be a finite number in (0, 1), got {p!r}.")
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability p must be strictly between 0 and 1, got {p!r}.")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    for _ in range(3):
        err = normal_cdf(z) - p
        pdf = normal_pdf(z)
        if pdf == 0.0:
            break
        z -= err / pdf
    return z


def normal_loss(z: float) -> float:
    """Standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))."""
    return normal_pdf(z) - z * (1.0 - normal_cdf(z))


def solve_z_from_loss(target_loss: float) -> float:
    """Solve L(z) = target_loss for z via bisection (used for fill-rate service level)."""
    if target_loss <= 0.0:
        raise ValueError("Target standard normal loss must be strictly positive.")
    lo, hi = -8.0, 8.0
    while normal_loss(lo) < target_loss:
        lo -= 4.0
    while normal_loss(hi) > target_loss:
        hi += 2.0
        if hi > 38.0:
            return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if normal_loss(mid) > target_loss:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Strict Input Validation Helpers
# ---------------------------------------------------------------------------

def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric value, got {value!r}.")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return val


def _require_positive(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required input '{name}'.")
    val = _as_finite_float(value, name)
    if val <= 0.0:
        raise ValueError(f"{name} must be > 0, got {val!r}.")
    return val


def _require_non_negative(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required input '{name}'.")
    val = _as_finite_float(value, name)
    if val < 0.0:
        raise ValueError(f"{name} must be >= 0, got {val!r}.")
    return val


def _resolve_holding_cost(
    holding_cost: Optional[float],
    holding_rate: Optional[float],
    unit_cost: Optional[float],
) -> Tuple[float, str]:
    if holding_cost is not None:
        return _require_positive(holding_cost, "holding_cost"), "fixed_per_unit"
    if holding_rate is not None and unit_cost is not None:
        rate = _require_positive(holding_rate, "holding_rate")
        cost = _require_positive(unit_cost, "unit_cost")
        return rate * cost, "percentage_of_unit_cost"
    raise ValueError(
        "Missing holding cost input: provide 'holding_cost' or both 'holding_rate' and 'unit_cost'."
    )


def _resolve_z_and_csl(
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
) -> Tuple[float, float]:
    if service_level is not None and z_score is not None:
        raise ValueError("Provide either 'service_level' or 'z_score', not both.")
    if service_level is not None:
        csl = _as_finite_float(service_level, "service_level")
        if not (0.0 < csl < 1.0):
            raise ValueError(f"service_level must be strictly in (0, 1), got {csl!r}.")
        return normal_ppf(csl), csl
    if z_score is not None:
        z = _as_finite_float(z_score, "z_score")
        return z, normal_cdf(z)
    raise ValueError("Missing service target: provide either 'service_level' or 'z_score'.")


def _fmt_val(val: Optional[float], decimals: int = 2) -> str:
    """Format numeric values cleanly for substituted formula strings and student verification."""
    if val is None:
        return "N/A"
    if not math.isfinite(val):
        return str(val)
    if abs(val - round(val)) < 1e-9 and abs(val) < 1e12:
        return f"{int(round(val)):,}"
    return f"{val:,.{decimals}f}"


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------

def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: Optional[float] = None,
    *,
    holding_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    working_days_per_year: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate Economic Order Quantity: Q = sqrt(2 * D * S / H)."""
    d = _require_positive(annual_demand, "annual_demand")
    s = _require_positive(ordering_cost, "ordering_cost")
    h, holding_basis = _resolve_holding_cost(holding_cost, holding_rate, unit_cost)
    days_per_year = (
        _require_positive(working_days_per_year, "working_days_per_year")
        if working_days_per_year is not None
        else None
    )

    numerator_2ds = 2.0 * d * s
    q_opt = math.sqrt(numerator_2ds / h)
    orders_per_year = d / q_opt
    cycle_time_years = q_opt / d
    cycle_time_days = cycle_time_years * days_per_year if days_per_year is not None else None
    avg_inventory = q_opt / 2.0

    annual_ordering_cost = (d / q_opt) * s
    annual_holding_cost = (q_opt / 2.0) * h
    total_relevant_cost = annual_ordering_cost + annual_holding_cost

    if unit_cost is not None:
        c = _require_non_negative(unit_cost, "unit_cost")
        annual_purchase_cost: Optional[float] = d * c
        total_annual_cost = annual_purchase_cost + total_relevant_cost
    else:
        c = None
        annual_purchase_cost = None
        total_annual_cost = total_relevant_cost

    assumptions = [
        "Demand is known, constant, and continuous over the annual horizon.",
        "Replenishment lead time is constant (or zero) with instantaneous receipt and no shortages allowed.",
        "Ordering cost S per order and unit holding cost H per unit per year are constant with no quantity discounts.",
    ]
    if holding_basis == "percentage_of_unit_cost":
        assumptions.append(
            f"Annual holding cost computed from carrying rate i={holding_rate!r} * unit_cost=${c:,.4f} -> H=${h:,.4f}/unit/year."
        )
    if days_per_year is None:
        assumptions.append(
            "No working_days_per_year provided; cycle time is reported in years without inventing a calendar day count."
        )
    else:
        assumptions.append(f"Cycle time in days uses working_days_per_year={days_per_year:g}.")

    inputs_dict = {
        "annual_demand": d,
        "ordering_cost": s,
        "holding_cost": holding_cost,
        "holding_rate": holding_rate,
        "unit_cost": c,
        "working_days_per_year": days_per_year,
    }

    var_defs: Dict[str, str] = {
        "D": f"Annual demand = {_fmt_val(d)} units/year",
        "S": f"Ordering / setup cost per replenishment order = ${_fmt_val(s)}/order",
        "H": (
            f"Annual holding cost per unit per year = ${_fmt_val(h, 4)}/unit/year"
            + (f" ({holding_rate*100:g}% of unit price ${_fmt_val(c)})" if holding_basis == "percentage_of_unit_cost" else "")
        ),
        "Q*": f"Economic Order Quantity (optimal replenishment order quantity) = {_fmt_val(q_opt)} units",
        "N": f"Order frequency / replenishment orders per year = D / Q* = {_fmt_val(orders_per_year)} orders/year",
        "T": (
            f"Order cycle time = Q* / D = {_fmt_val(cycle_time_years, 4)} years"
            + (f" ({_fmt_val(cycle_time_days, 1)} operating days)" if cycle_time_days is not None else "")
        ),
        "I_avg": f"Average cycle inventory = Q* / 2 = {_fmt_val(avg_inventory)} units",
        "TOC": f"Annual ordering cost = (D / Q*) * S = ${_fmt_val(annual_ordering_cost)}/year",
        "THC": f"Annual holding cost = (Q* / 2) * H = ${_fmt_val(annual_holding_cost)}/year",
        "TRC": f"Total relevant annual inventory cost = TOC + THC = ${_fmt_val(total_relevant_cost)}/year",
    }
    if c is not None:
        var_defs["C"] = f"Unit purchase price = ${_fmt_val(c)}/unit"
        var_defs["TC"] = f"Total annual cost (including purchase) = D * C + TRC = ${_fmt_val(total_annual_cost)}/year"

    formulas: Dict[str, str] = {
        "optimal_order_quantity": "Q* = sqrt((2 * D * S) / H)",
        "orders_per_year": "N = D / Q*",
        "order_cycle_time": "T = Q* / D (years)" + (" or (Q* / D) * working_days_per_year (days)" if days_per_year else ""),
        "average_cycle_inventory": "I_avg = Q* / 2",
        "annual_ordering_cost": "TOC = (D / Q*) * S",
        "annual_holding_cost": "THC = (Q* / 2) * H",
        "total_relevant_cost": "TRC = TOC + THC = (D / Q*) * S + (Q* / 2) * H",
    }
    if c is not None:
        formulas["total_annual_cost"] = "TC = D * C + TRC = D * C + (D / Q*) * S + (Q* / 2) * H"

    cycle_time_sub = (
        f"T = {_fmt_val(q_opt)} / {_fmt_val(d)} = {_fmt_val(cycle_time_years, 4)} years"
        + (f" ({_fmt_val(cycle_time_days, 1)} operating days at {_fmt_val(days_per_year)} days/year)" if cycle_time_days is not None else "")
    )
    sub_formulas: Dict[str, str] = {
        "optimal_order_quantity": (
            f"Q* = sqrt((2 * {_fmt_val(d)} * {_fmt_val(s)}) / {_fmt_val(h, 4)}) "
            f"= sqrt({_fmt_val(numerator_2ds)} / {_fmt_val(h, 4)}) "
            f"= sqrt({_fmt_val(numerator_2ds / h, 4)}) "
            f"= {_fmt_val(q_opt)} units"
        ),
        "orders_per_year": f"N = {_fmt_val(d)} / {_fmt_val(q_opt)} = {_fmt_val(orders_per_year)} orders/year",
        "order_cycle_time": cycle_time_sub,
        "average_cycle_inventory": f"I_avg = {_fmt_val(q_opt)} / 2 = {_fmt_val(avg_inventory)} units",
        "annual_ordering_cost": f"TOC = ({_fmt_val(d)} / {_fmt_val(q_opt)}) * ${_fmt_val(s)} = ${_fmt_val(annual_ordering_cost)}/year",
        "annual_holding_cost": f"THC = ({_fmt_val(q_opt)} / 2) * ${_fmt_val(h, 4)} = ${_fmt_val(annual_holding_cost)}/year",
        "total_relevant_cost": f"TRC = ${_fmt_val(annual_ordering_cost)} + ${_fmt_val(annual_holding_cost)} = ${_fmt_val(total_relevant_cost)}/year",
    }
    if c is not None:
        sub_formulas["total_annual_cost"] = (
            f"TC = ({_fmt_val(d)} * ${_fmt_val(c)}) + ${_fmt_val(total_relevant_cost)} "
            f"= ${_fmt_val(annual_purchase_cost)} + ${_fmt_val(total_relevant_cost)} = ${_fmt_val(total_annual_cost)}/year"
        )

    intermediate = {
        "holding_cost_basis": holding_basis,
        "effective_holding_cost": h,
        "numerator_2DS": numerator_2ds,
        "ratio_2DS_over_H": numerator_2ds / h,
        "sqrt_2DS_over_H": q_opt,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "average_inventory": avg_inventory,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_relevant_cost": total_relevant_cost,
        "total_annual_cost": total_annual_cost,
    }
    final_answer = {
        "optimal_order_quantity": q_opt,
        "total_relevant_cost": total_relevant_cost,
        "total_annual_cost": total_annual_cost,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "average_inventory": avg_inventory,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
    }

    result = {
        "model": "eoq",
        "tool_name": "calculate_eoq",
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": var_defs,
        "formula": formulas,
        "formula_with_substituted_values": sub_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "optimal_order_quantity": q_opt,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "average_inventory": avg_inventory,
        "effective_holding_cost": h,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_relevant_cost": total_relevant_cost,
        "total_annual_cost": total_annual_cost,
    }

    ver_report = verify_calculation("eoq", inputs_dict, result)
    result["verification_checks"] = ver_report["checks"]
    result["verification_passed"] = ver_report["passed"]
    return result


# ---------------------------------------------------------------------------
# 2. Quantity Discount Model
# ---------------------------------------------------------------------------

def _normalize_tiers(tiers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(tiers, (list, tuple)) or not tiers:
        raise ValueError("tiers must be a non-empty list or tuple of price-break tier dicts.")
    normalized: List[Dict[str, Any]] = []
    for idx, tier in enumerate(tiers):
        if not isinstance(tier, dict):
            raise ValueError(f"tiers[{idx}] must be a dict, got {type(tier).__name__}.")
        if "min_qty" not in tier or tier["min_qty"] is None:
            raise ValueError(f"tiers[{idx}] is missing required field 'min_qty'.")
        min_q = _require_non_negative(tier["min_qty"], f"tiers[{idx}].min_qty")
        raw_max = tier.get("max_qty")
        max_q = float("inf") if raw_max is None else _require_positive(raw_max, f"tiers[{idx}].max_qty")
        if max_q < min_q:
            raise ValueError(f"tiers[{idx}].max_qty ({max_q}) cannot be < min_qty ({min_q}).")
        price = _require_positive(tier.get("unit_price"), f"tiers[{idx}].unit_price")
        tier_h = tier.get("holding_cost")
        if tier_h is not None:
            tier_h = _require_positive(tier_h, f"tiers[{idx}].holding_cost")
        normalized.append(
            {
                "tier_index": idx + 1,
                "min_qty": min_q,
                "max_qty": max_q,
                "max_exclusive": bool(tier.get("max_exclusive", False)),
                "unit_price": price,
                "holding_cost": tier_h,
            }
        )
    normalized.sort(key=lambda t: t["min_qty"])
    for i, t in enumerate(normalized, start=1):
        t["tier_index"] = i
        if i < len(normalized):
            next_min = normalized[i]["min_qty"]
            if t["max_qty"] > next_min:
                raise ValueError(
                    f"Overlapping quantity tiers: Tier {i} max_qty ({t['max_qty']}) > Tier {i + 1} min_qty ({next_min})."
                )
    return normalized


def _tier_holding_cost(
    tier: Dict[str, Any],
    holding_rate: Optional[float],
    fixed_holding_cost: Optional[float],
) -> Tuple[float, str]:
    if tier["holding_cost"] is not None:
        return float(tier["holding_cost"]), "tier_specific_fixed"
    if holding_rate is not None:
        return holding_rate * tier["unit_price"], "percentage_of_tier_price"
    if fixed_holding_cost is not None:
        return fixed_holding_cost, "fixed_across_tiers"
    raise ValueError(
        "Missing holding cost input: provide 'holding_rate', 'holding_cost', or per-tier 'holding_cost'."
    )


def _upper_feasible_bound(tier: Dict[str, Any]) -> float:
    max_q = tier["max_qty"]
    if not math.isfinite(max_q):
        return max_q
    if tier["max_exclusive"]:
        return max(tier["min_qty"], max_q - 1e-6)
    return max_q


def _evaluate_all_units_tiers(
    d: float,
    s: float,
    tiers: List[Dict[str, Any]],
    holding_rate: Optional[float],
    fixed_holding_cost: Optional[float],
) -> Tuple[List[Dict[str, Any]], str]:
    evaluated: List[Dict[str, Any]] = []
    holding_mode = "unknown"
    for tier in tiers:
        p = tier["unit_price"]
        h, holding_mode = _tier_holding_cost(tier, holding_rate, fixed_holding_cost)
        eoq = math.sqrt((2.0 * d * s) / h)

        upper = _upper_feasible_bound(tier)
        if eoq < tier["min_qty"]:
            candidate_q = tier["min_qty"]
            feasibility = "adjusted_up_to_min_break"
        elif eoq > upper:
            candidate_q = upper
            feasibility = "adjusted_down_to_max_bound"
        else:
            candidate_q = eoq
            feasibility = "feasible_eoq"

        if candidate_q <= 0.0:
            raise ValueError(
                f"Tier {tier['tier_index']} produced non-positive candidate quantity ({candidate_q})."
            )

        purchase_cost = d * p
        ordering_cost = (d / candidate_q) * s
        holding_cost = (candidate_q / 2.0) * h
        total_cost = purchase_cost + ordering_cost + holding_cost

        evaluated.append(
            {
                "tier_index": tier["tier_index"],
                "min_qty": tier["min_qty"],
                "max_qty": None if not math.isfinite(tier["max_qty"]) else tier["max_qty"],
                "max_exclusive": tier["max_exclusive"],
                "unit_price": p,
                "holding_cost_per_unit": h,
                "unconstrained_eoq": eoq,
                "candidate_quantity": candidate_q,
                "is_eoq_feasible_in_tier": feasibility == "feasible_eoq",
                "feasibility": feasibility,
                "annual_purchase_cost": purchase_cost,
                "annual_ordering_cost": ordering_cost,
                "annual_holding_cost": holding_cost,
                "total_annual_cost": total_cost,
            }
        )
    return evaluated, holding_mode


def _evaluate_incremental_tiers(
    d: float,
    s: float,
    tiers: List[Dict[str, Any]],
    holding_rate: Optional[float],
    fixed_holding_cost: Optional[float],
) -> Tuple[List[Dict[str, Any]], str]:
    if holding_rate is None and fixed_holding_cost is None:
        raise ValueError(
            "Incremental discount requires 'holding_rate' or fixed 'holding_cost'."
        )
    evaluated: List[Dict[str, Any]] = []
    cumulative_fixed = 0.0
    prev_break = 0.0
    holding_mode = "fixed_across_tiers" if fixed_holding_cost is not None else "percentage_of_tier_price"

    for idx, tier in enumerate(tiers):
        p = tier["unit_price"]
        min_q = tier["min_qty"]
        if idx > 0:
            prev_price = tiers[idx - 1]["unit_price"]
            cumulative_fixed += (min_q - prev_break) * prev_price
            prev_break = min_q

        numerator = 2.0 * d * (s + cumulative_fixed - p * min_q)
        if fixed_holding_cost is not None:
            denom = fixed_holding_cost
            eoq = math.sqrt(numerator / denom) if numerator > 0.0 else float("nan")
            h_eff = fixed_holding_cost
        else:
            assert holding_rate is not None
            denom = holding_rate * p
            eoq = math.sqrt(numerator / denom) if numerator > 0.0 else float("nan")
            h_eff = holding_rate * p

        upper = _upper_feasible_bound(tier)
        is_valid = math.isfinite(eoq) and (eoq >= max(min_q, 1e-9)) and (eoq <= upper)
        candidate_q = eoq if is_valid else None

        if candidate_q is not None:
            order_cost_total = cumulative_fixed + p * (candidate_q - min_q)
            avg_unit_cost: Optional[float] = order_cost_total / candidate_q
            purchase_cost: Optional[float] = d * avg_unit_cost
            ordering_cost: Optional[float] = (d / candidate_q) * s
            eff_h = (
                fixed_holding_cost
                if fixed_holding_cost is not None
                else holding_rate * avg_unit_cost  # type: ignore[operator]
            )
            holding_cost: Optional[float] = (candidate_q / 2.0) * eff_h
            total_cost: Optional[float] = purchase_cost + ordering_cost + holding_cost
            feasibility = "feasible_interior_optimum"
        else:
            avg_unit_cost = None
            purchase_cost = None
            ordering_cost = None
            holding_cost = None
            total_cost = None
            feasibility = "out_of_tier_range"

        evaluated.append(
            {
                "tier_index": tier["tier_index"],
                "min_qty": min_q,
                "max_qty": None if not math.isfinite(tier["max_qty"]) else tier["max_qty"],
                "max_exclusive": tier["max_exclusive"],
                "unit_price": p,
                "effective_average_unit_price": avg_unit_cost,
                "holding_cost_per_unit": h_eff,
                "unconstrained_eoq": None if not math.isfinite(eoq) else eoq,
                "candidate_quantity": candidate_q,
                "is_eoq_feasible_in_tier": is_valid,
                "feasibility": feasibility,
                "annual_purchase_cost": purchase_cost,
                "annual_ordering_cost": ordering_cost,
                "annual_holding_cost": holding_cost,
                "total_annual_cost": total_cost,
            }
        )
    return evaluated, holding_mode


def calculate_quantity_discount(
    annual_demand: float,
    ordering_cost: float,
    tiers: Sequence[Dict[str, Any]],
    *,
    holding_rate: Optional[float] = None,
    holding_cost: Optional[float] = None,
    discount_type: str = "all_units",
) -> Dict[str, Any]:
    """Evaluate feasible quantities per price tier and compare total annual costs.

    Supports both fixed holding cost ($/unit/year) and percentage holding cost (i * P).
    """
    d = _require_positive(annual_demand, "annual_demand")
    s = _require_positive(ordering_cost, "ordering_cost")
    h_rate = _require_positive(holding_rate, "holding_rate") if holding_rate is not None else None
    h_fixed = _require_positive(holding_cost, "holding_cost") if holding_cost is not None else None
    if h_rate is not None and h_fixed is not None:
        raise ValueError("Provide either 'holding_rate' or fixed 'holding_cost', not both.")
    norm_tiers = _normalize_tiers(tiers)

    if not isinstance(discount_type, str):
        raise ValueError("discount_type must be a string ('all_units' or 'incremental').")
    dtype = discount_type.lower().strip()
    if dtype not in {"all_units", "incremental"}:
        raise ValueError("discount_type must be 'all_units' or 'incremental'.")

    if dtype == "all_units":
        evaluated, holding_mode = _evaluate_all_units_tiers(d, s, norm_tiers, h_rate, h_fixed)
        feasible_rows = evaluated
    else:
        evaluated, holding_mode = _evaluate_incremental_tiers(d, s, norm_tiers, h_rate, h_fixed)
        feasible_rows = [row for row in evaluated if row["total_annual_cost"] is not None]
        if not feasible_rows:
            raise ValueError("No feasible interior optimum found across incremental tiers.")

    best = min(feasible_rows, key=lambda r: (r["total_annual_cost"], -r["candidate_quantity"]))
    orders_per_year = d / best["candidate_quantity"]
    opt_unit_price = best.get("effective_average_unit_price") or best["unit_price"]

    assumptions = [
        "Demand is constant and deterministic over the annual horizon with no shortages allowed.",
        f"Discount structure is '{dtype}' across {len(norm_tiers)} price tiers.",
        (
            f"Annual holding cost scales as percentage i={h_rate:g} of unit price."
            if h_rate is not None
            else (
                f"Annual holding cost is fixed at H=${h_fixed:g}/unit/year across tiers."
                if h_fixed is not None
                else "Per-tier holding cost was supplied explicitly."
            )
        ),
    ]

    inputs_dict = {
        "annual_demand": d,
        "ordering_cost": s,
        "holding_rate": h_rate,
        "holding_cost": h_fixed,
        "discount_type": dtype,
        "tiers": [
            {
                "tier_index": t["tier_index"],
                "min_qty": t["min_qty"],
                "max_qty": None if not math.isfinite(t["max_qty"]) else t["max_qty"],
                "max_exclusive": t["max_exclusive"],
                "unit_price": t["unit_price"],
            }
            for t in norm_tiers
        ],
    }

    var_defs: Dict[str, str] = {
        "D": f"Annual demand = {_fmt_val(d)} units/year",
        "S": f"Ordering / setup cost per replenishment order = ${_fmt_val(s)}/order",
        "discount_type": f"Discount structure = {dtype}",
        "holding_cost_mode": f"Holding cost basis = {holding_mode}",
        "Q*": f"Optimal order quantity = {_fmt_val(best['candidate_quantity'])} units",
        "Optimal_Tier": f"Tier {best['tier_index']}",
        "Unit_Price": f"Unit price at optimal quantity = ${_fmt_val(opt_unit_price)}/unit",
        "Purchase_Cost": f"Annual purchase cost = ${_fmt_val(best['annual_purchase_cost'])}/year",
        "Ordering_Cost": f"Annual ordering cost = ${_fmt_val(best['annual_ordering_cost'])}/year",
        "Holding_Cost": f"Annual holding cost = ${_fmt_val(best['annual_holding_cost'])}/year",
        "Total_Annual_Cost": f"Total annual cost = ${_fmt_val(best['total_annual_cost'])}/year",
    }

    formulas: Dict[str, str] = {
        "unconstrained_eoq": "Q_j = sqrt((2 * D * S) / H_j) where H_j is the tier-specific holding cost",
        "feasibility_rule": (
            "If min_qty_j <= Q_j <= max_qty_j -> candidate Q = Q_j (feasible in tier); "
            "If Q_j < min_qty_j -> evaluate the lower feasible boundary (candidate Q = min_qty_j); "
            "If Q_j > max_qty_j -> evaluate the upper feasible boundary (candidate Q = max_qty_j); "
            "compare Total Annual Cost across all feasible tier candidates to select Q*."
        ),
        "total_annual_cost": (
            "All-Units: TC(Q) = D * C_j + (D / Q) * S + (Q / 2) * H_j; "
            "Incremental: TC(Q) = PurchaseCost(Q) + (D / Q) * S + HoldingCost(Q)"
        ),
        "optimal_decision": "Q* = argmin_j TC(candidate_Q_j)",
    }

    tier_sub_steps: List[str] = []
    for t in evaluated:
        hi = "inf" if t["max_qty"] is None else _fmt_val(t["max_qty"])
        h_val = t.get("holding_cost_per_unit") if t.get("holding_cost_per_unit") is not None else t.get("holding_cost")
        h_str = f"${_fmt_val(h_val, 4)}"
        q_raw_str = _fmt_val(t["unconstrained_eoq"]) if t.get("unconstrained_eoq") is not None else "N/A"
        cand_str = _fmt_val(t["candidate_quantity"]) if t.get("candidate_quantity") is not None else "Infeasible"
        tc_str = f"${_fmt_val(t['total_annual_cost'])}/year" if t.get("total_annual_cost") is not None else "N/A"
        tier_sub_steps.append(
            f"Tier {t['tier_index']} [{_fmt_val(t['min_qty'])}–{hi}, ${_fmt_val(t['unit_price'])}/unit]: "
            f"H={h_str}/unit/year, "
            f"Q_unconstrained = sqrt(2 * {_fmt_val(d)} * {_fmt_val(s)} / {h_str}) = {q_raw_str} -> "
            f"Candidate Q = {cand_str} ({t['feasibility']}), "
            f"Purchase=${_fmt_val(t.get('annual_purchase_cost'))} + Order=${_fmt_val(t.get('annual_ordering_cost'))} + "
            f"Hold=${_fmt_val(t.get('annual_holding_cost'))} -> Total = {tc_str}"
        )

    sub_formulas: Dict[str, Any] = {
        "tier_evaluations": tier_sub_steps,
        "optimal_decision": (
            f"Lowest Total Cost = ${_fmt_val(best['total_annual_cost'])}/year achieved at Tier {best['tier_index']} "
            f"with Order Quantity Q* = {_fmt_val(best['candidate_quantity'])} units @ ${_fmt_val(opt_unit_price)}/unit."
        ),
    }

    intermediate = {
        "holding_cost_mode": holding_mode,
        "tier_evaluations": evaluated,
        "orders_per_year": orders_per_year,
        "annual_purchase_cost": best["annual_purchase_cost"],
        "annual_ordering_cost": best["annual_ordering_cost"],
        "annual_holding_cost": best["annual_holding_cost"],
        "total_annual_cost": best["total_annual_cost"],
    }
    final_answer = {
        "optimal_order_quantity": best["candidate_quantity"],
        "optimal_tier": best["tier_index"],
        "optimal_unit_price": opt_unit_price,
        "total_annual_cost": best["total_annual_cost"],
        "annual_purchase_cost": best["annual_purchase_cost"],
        "annual_ordering_cost": best["annual_ordering_cost"],
        "annual_holding_cost": best["annual_holding_cost"],
    }

    result = {
        "model": "quantity_discount",
        "tool_name": "calculate_quantity_discount",
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": var_defs,
        "formula": formulas,
        "formula_with_substituted_values": sub_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "discount_type": dtype,
        "holding_cost_mode": holding_mode,
        "optimal_order_quantity": best["candidate_quantity"],
        "optimal_tier": best["tier_index"],
        "optimal_unit_price": opt_unit_price,
        "listed_tier_unit_price": best["unit_price"],
        "annual_purchase_cost": best["annual_purchase_cost"],
        "annual_ordering_cost": best["annual_ordering_cost"],
        "annual_holding_cost": best["annual_holding_cost"],
        "total_annual_cost": best["total_annual_cost"],
        "orders_per_year": orders_per_year,
        "tier_evaluations": evaluated,
    }

    ver_report = verify_calculation("quantity_discount", inputs_dict, result)
    result["verification_checks"] = ver_report["checks"]
    result["verification_passed"] = ver_report["passed"]
    return result


# ---------------------------------------------------------------------------
# 3. Safety Stock & Reorder Point (Continuous Review)
# ---------------------------------------------------------------------------

def calculate_rop(
    *,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    lead_time: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    lead_time_demand_mean: Optional[float] = None,
    lead_time_demand_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    service_level_type: str = "cycle",
    order_quantity: Optional[float] = None,
    holding_cost: Optional[float] = None,
    time_unit: str = "period",
) -> Dict[str, Any]:
    """Calculate Safety Stock and Reorder Point (ROP) for continuous review.

    Supports all four demand / lead-time regimes:
      1. constant_demand_constant_lead_time (demand_std=0, lead_time_std=0)
      2. variable_demand_constant_lead_time (demand_std>0, lead_time_std=0 or omitted)
      3. constant_demand_variable_lead_time (demand_std=0 or omitted, lead_time_std>0)
      4. variable_demand_variable_lead_time:
             sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
             SS = z * sigma_LT
             ROP = d * L + SS
    Also supports direct lead_time_demand_mean + lead_time_demand_std.
    """
    assumptions: List[str] = []

    if lead_time_demand_mean is not None or lead_time_demand_std is not None:
        if lead_time_demand_mean is None or lead_time_demand_std is None:
            raise ValueError(
                "Direct lead-time distribution requires both 'lead_time_demand_mean' and 'lead_time_demand_std'."
            )
        mu_lt = _require_non_negative(lead_time_demand_mean, "lead_time_demand_mean")
        sigma_lt = _require_non_negative(lead_time_demand_std, "lead_time_demand_std")
        var_lt = sigma_lt ** 2
        d_mean_val: Optional[float] = None
        d_std_val: Optional[float] = None
        lt_mean_val: Optional[float] = None
        lt_std_val: Optional[float] = None
        formula_case = "direct_lead_time_distribution"
        assumptions.append("Lead-time demand distribution (mu_LT, sigma_LT) was provided directly.")
    else:
        d_mean_val = _require_non_negative(demand_mean, "demand_mean")
        lt_mean_val = _require_positive(lead_time, "lead_time")

        if demand_std is None and lead_time_std is None:
            raise ValueError(
                "Missing variability inputs: provide 'demand_std' and/or 'lead_time_std' "
                "(pass 0.0 explicitly for constant demand and/or constant lead time)."
            )

        if demand_std is None:
            d_std_val = 0.0
            assumptions.append(
                "No demand_std provided while lead_time_std was supplied; demand is treated as constant (sigma_d = 0)."
            )
        else:
            d_std_val = _require_non_negative(demand_std, "demand_std")

        if lead_time_std is None:
            lt_std_val = 0.0
            assumptions.append(
                "No lead_time_std provided while demand_std was supplied; lead time is treated as constant (sigma_L = 0)."
            )
        else:
            lt_std_val = _require_non_negative(lead_time_std, "lead_time_std")

        mu_lt = d_mean_val * lt_mean_val
        var_lt = lt_mean_val * (d_std_val ** 2) + (d_mean_val ** 2) * (lt_std_val ** 2)
        sigma_lt = math.sqrt(var_lt)

        if d_std_val == 0.0 and lt_std_val == 0.0:
            formula_case = "constant_demand_constant_lead_time"
            assumptions.append("Both demand and lead time are constant (deterministic); no safety stock is needed.")
        elif d_std_val > 0.0 and lt_std_val == 0.0:
            formula_case = "variable_demand_constant_lead_time"
            assumptions.append(
                "Per-period demand is i.i.d. normal and lead time is constant: sigma_LT = sigma_d * sqrt(L)."
            )
        elif d_std_val == 0.0 and lt_std_val > 0.0:
            formula_case = "constant_demand_variable_lead_time"
            assumptions.append(
                "Demand per period is constant and lead time is normally distributed: sigma_LT = d * sigma_L."
            )
        else:
            formula_case = "variable_demand_variable_lead_time"
            assumptions.append(
                "Demand and lead time are independent random variables: sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)."
            )

    sl_type = service_level_type.lower().strip()
    if sl_type not in {"cycle", "fill_rate"}:
        raise ValueError("service_level_type must be 'cycle' or 'fill_rate'.")

    z: Optional[float]
    csl: float
    achieved_fill_rate: Optional[float] = None

    if sigma_lt == 0.0:
        if service_level is not None:
            sl_val = _as_finite_float(service_level, "service_level")
            if not (0.0 < sl_val <= 1.0):
                raise ValueError(f"service_level must be in (0, 1], got {sl_val!r}.")
            z = normal_ppf(sl_val) if sl_val < 1.0 else None
        elif z_score is not None:
            z = _as_finite_float(z_score, "z_score")
        else:
            z = None
        csl = 1.0
        safety_stock = 0.0
        if order_quantity is not None:
            _require_positive(order_quantity, "order_quantity")
            achieved_fill_rate = 1.0
    elif sl_type == "fill_rate":
        if service_level is None or order_quantity is None:
            raise ValueError(
                "fill_rate mode requires both 'service_level' (target fill rate) and 'order_quantity'."
            )
        beta = _as_finite_float(service_level, "service_level")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"Target fill rate must be strictly in (0, 1), got {beta!r}.")
        q_val = _require_positive(order_quantity, "order_quantity")
        target_loss = (q_val * (1.0 - beta)) / sigma_lt
        z = solve_z_from_loss(target_loss)
        csl = normal_cdf(z)
        achieved_fill_rate = beta
        safety_stock = z * sigma_lt
        assumptions.append("Safety factor z solved from Type 2 fill-rate standard normal loss function.")
    else:
        z, csl = _resolve_z_and_csl(service_level=service_level, z_score=z_score)
        safety_stock = z * sigma_lt
        if order_quantity is not None:
            q_val = _require_positive(order_quantity, "order_quantity")
            achieved_fill_rate = 1.0 - (sigma_lt * normal_loss(z)) / q_val
        assumptions.append("Lead-time demand is approximated as normally distributed for Type 1 cycle service level.")

    reorder_point = mu_lt + safety_stock
    ss_holding_cost = (
        safety_stock * _require_non_negative(holding_cost, "holding_cost")
        if holding_cost is not None
        else None
    )

    inputs_dict = {
        "demand_mean": d_mean_val,
        "demand_std": d_std_val,
        "lead_time": lt_mean_val,
        "lead_time_std": lt_std_val,
        "lead_time_demand_mean": lead_time_demand_mean,
        "lead_time_demand_std": lead_time_demand_std,
        "service_level": service_level,
        "z_score": z_score,
        "service_level_type": sl_type,
        "order_quantity": order_quantity,
        "holding_cost": holding_cost,
        "time_unit": time_unit,
    }

    var_defs: Dict[str, str] = {
        "d": f"Average / expected demand rate = {_fmt_val(d_mean_val)} units/{time_unit}" if d_mean_val is not None else "Average demand rate",
        "sigma_d": f"Standard deviation of demand per period = {_fmt_val(d_std_val)} units/{time_unit}" if d_std_val is not None else "Demand standard deviation",
        "L": f"Average replenishment lead time = {_fmt_val(lt_mean_val)} {time_unit}s" if lt_mean_val is not None else "Average lead time",
        "sigma_L": f"Standard deviation of lead time = {_fmt_val(lt_std_val)} {time_unit}s" if lt_std_val is not None else "Lead-time standard deviation",
        "mu_LT": f"Expected demand during lead time = d * L = {_fmt_val(mu_lt)} units",
        "sigma_LT": f"Standard deviation of lead-time demand = {_fmt_val(sigma_lt, 4)} units",
        "z": f"Safety factor (z-score) = {_fmt_val(z, 4)}" if z is not None else "Safety factor (z-score) = N/A (deterministic)",
        "CSL": f"Cycle service level = {_fmt_val(csl * 100.0, 2)}%" if csl is not None else "Cycle service level = N/A",
        "SS": f"Safety stock = z * sigma_LT = {_fmt_val(safety_stock)} units",
        "ROP": f"Reorder point = mu_LT + SS = {_fmt_val(reorder_point)} units",
    }
    if achieved_fill_rate is not None:
        var_defs["Fill_Rate"] = f"Achieved fill rate = {_fmt_val(achieved_fill_rate * 100.0, 2)}%"
    if ss_holding_cost is not None:
        var_defs["SS_Holding_Cost"] = f"Annual safety stock holding cost = ${_fmt_val(ss_holding_cost)}/year"

    formulas: Dict[str, str] = {
        "expected_lead_time_demand": "mu_LT = d * L",
        "lead_time_demand_std": (
            "sigma_LT = 0"
            if formula_case == "constant_demand_constant_lead_time"
            else (
                "sigma_LT = sqrt(L * sigma_d^2) = sigma_d * sqrt(L)"
                if formula_case == "variable_demand_constant_lead_time"
                else (
                    "sigma_LT = sqrt(d^2 * sigma_L^2) = d * sigma_L"
                    if formula_case == "constant_demand_variable_lead_time"
                    else "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
                )
            )
        ),
        "safety_stock": "SS = z * sigma_LT" if sigma_lt > 0.0 else "SS = 0 (deterministic)",
        "reorder_point": "ROP = mu_LT + SS = d * L + SS",
    }

    if formula_case == "constant_demand_constant_lead_time":
        sigma_sub = "sigma_LT = 0 (deterministic; constant demand and constant lead time)"
        ss_sub = "SS = 0 units (no safety stock required)"
    elif formula_case == "variable_demand_constant_lead_time":
        sigma_sub = f"sigma_LT = {_fmt_val(d_std_val)} * sqrt({_fmt_val(lt_mean_val)}) = {_fmt_val(sigma_lt, 4)} units"
        ss_sub = f"SS = {_fmt_val(z, 4)} * {_fmt_val(sigma_lt, 4)} = {_fmt_val(safety_stock)} units"
    elif formula_case == "constant_demand_variable_lead_time":
        sigma_sub = f"sigma_LT = {_fmt_val(d_mean_val)} * {_fmt_val(lt_std_val)} = {_fmt_val(sigma_lt, 4)} units"
        ss_sub = f"SS = {_fmt_val(z, 4)} * {_fmt_val(sigma_lt, 4)} = {_fmt_val(safety_stock)} units"
    else:
        term_d = (lt_mean_val or 0.0) * ((d_std_val or 0.0) ** 2)
        term_l = ((d_mean_val or 0.0) ** 2) * ((lt_std_val or 0.0) ** 2)
        sigma_sub = (
            f"sigma_LT = sqrt({_fmt_val(lt_mean_val)} * ({_fmt_val(d_std_val)})^2 + ({_fmt_val(d_mean_val)})^2 * ({_fmt_val(lt_std_val)})^2) "
            f"= sqrt({term_d:,.2f} + {term_l:,.2f}) = sqrt({var_lt:,.2f}) = {_fmt_val(sigma_lt, 4)} units"
        )
        ss_sub = f"SS = {_fmt_val(z, 4)} * {_fmt_val(sigma_lt, 4)} = {_fmt_val(safety_stock)} units"

    sub_formulas: Dict[str, str] = {
        "expected_lead_time_demand": (
            f"mu_LT = {_fmt_val(d_mean_val)} * {_fmt_val(lt_mean_val)} = {_fmt_val(mu_lt)} units"
            if d_mean_val is not None and lt_mean_val is not None else
            f"mu_LT = {_fmt_val(mu_lt)} units"
        ),
        "lead_time_demand_std": sigma_sub,
        "safety_stock": ss_sub,
        "reorder_point": f"ROP = {_fmt_val(mu_lt)} + {_fmt_val(safety_stock)} = {_fmt_val(reorder_point)} units",
    }

    intermediate = {
        "formula_case": formula_case,
        "expected_lead_time_demand": mu_lt,
        "lead_time_demand_variance": var_lt,
        "lead_time_demand_std": sigma_lt,
        "z_score": z,
        "cycle_service_level": csl,
        "stockout_probability": 1.0 - csl,
        "achieved_fill_rate": achieved_fill_rate,
        "annual_safety_stock_holding_cost": ss_holding_cost,
    }
    final_answer = {
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "expected_lead_time_demand": mu_lt,
        "lead_time_demand_std": sigma_lt,
        "cycle_service_level": csl,
        "z_score": z,
    }

    result = {
        "model": "safety_stock_rop",
        "tool_name": "calculate_rop",
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": var_defs,
        "formula": formulas,
        "formula_with_substituted_values": sub_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "formula_case": formula_case,
        "service_level_type": sl_type,
        "time_unit": time_unit,
        "z_score": z,
        "cycle_service_level": csl,
        "stockout_probability": 1.0 - csl,
        "expected_lead_time_demand": mu_lt,
        "lead_time_demand_variance": var_lt,
        "lead_time_demand_std": sigma_lt,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "achieved_fill_rate": achieved_fill_rate,
        "annual_safety_stock_holding_cost": ss_holding_cost,
    }

    ver_report = verify_calculation("safety_stock_rop", inputs_dict, result)
    result["verification_checks"] = ver_report["checks"]
    result["verification_passed"] = ver_report["passed"]
    return result


# Backwards-compatible alias
calculate_safety_stock_rop = calculate_rop


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review (Periodic Review / P-Model)
# ---------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: float,
    lead_time: float,
    demand_mean: float,
    demand_std: Optional[float] = None,
    *,
    lead_time_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    safety_stock: Optional[float] = None,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: Optional[float] = None,
    backorders: Optional[float] = None,
    inventory_position: Optional[float] = None,
    time_unit: str = "period",
) -> Dict[str, Any]:
    """Calculate Fixed-Period (Periodic) Review target level and order quantity.

    Formulas:
      Protection period = review_period + lead_time
      Target inventory = demand_during_protection_period + safety_stock
      Order quantity = max(0, target_inventory - inventory_position)
    """
    t_rev = _require_positive(review_period, "review_period")
    lt = _require_non_negative(lead_time, "lead_time")
    d_mean = _require_non_negative(demand_mean, "demand_mean")

    assumptions: List[str] = [
        "Inventory position is reviewed at fixed intervals T and replenished up to target level S.",
        "Protection period equals review_period + lead_time (T + L).",
    ]

    protection_period = t_rev + lt
    mu_prot = d_mean * protection_period

    z: Optional[float] = None
    csl: Optional[float] = None
    d_std: Optional[float] = None
    lt_std: Optional[float] = None
    var_prot: Optional[float] = None
    sigma_prot: Optional[float] = None

    if safety_stock is not None:
        ss_val = _require_non_negative(safety_stock, "safety_stock")
        assumptions.append("Safety stock was supplied directly by the user.")
        if demand_std is not None or lead_time_std is not None:
            d_std = _require_non_negative(demand_std, "demand_std") if demand_std is not None else 0.0
            lt_std = _require_non_negative(lead_time_std, "lead_time_std") if lead_time_std is not None else 0.0
            var_prot = protection_period * (d_std ** 2) + (d_mean ** 2) * (lt_std ** 2)
            sigma_prot = math.sqrt(var_prot)
            if sigma_prot > 0.0:
                z = ss_val / sigma_prot
                csl = normal_cdf(z)
    else:
        if demand_std is None and lead_time_std is None:
            raise ValueError(
                "Missing variability inputs: provide 'demand_std' (pass 0.0 for constant demand), "
                "'lead_time_std', or an explicit 'safety_stock'."
            )

        if demand_std is None:
            d_std = 0.0
            assumptions.append(
                "No demand_std provided while lead_time_std was supplied; demand is treated as constant (sigma_d = 0)."
            )
        else:
            d_std = _require_non_negative(demand_std, "demand_std")

        if lead_time_std is None:
            lt_std = 0.0
            assumptions.append("Lead time is treated as constant (sigma_L = 0).")
        else:
            lt_std = _require_non_negative(lead_time_std, "lead_time_std")

        var_prot = protection_period * (d_std ** 2) + (d_mean ** 2) * (lt_std ** 2)
        sigma_prot = math.sqrt(var_prot)

        if sigma_prot == 0.0:
            if service_level is not None:
                sl_val = _as_finite_float(service_level, "service_level")
                if not (0.0 < sl_val <= 1.0):
                    raise ValueError(f"service_level must be in (0, 1], got {sl_val!r}.")
                z = normal_ppf(sl_val) if sl_val < 1.0 else None
            elif z_score is not None:
                z = _as_finite_float(z_score, "z_score")
            else:
                z = None
            csl = 1.0
            ss_val = 0.0
            assumptions.append("Demand and lead time are deterministic (sigma_(T+L) = 0); safety stock is 0.")
        else:
            z, csl = _resolve_z_and_csl(service_level=service_level, z_score=z_score)
            ss_val = z * sigma_prot
            assumptions.append(
                "Demand over the protection period (T + L) is approximated as normally distributed."
            )

    safety_stock = ss_val
    target_inventory_level = mu_prot + safety_stock

    computed_inv_pos: Optional[float] = None
    order_quantity: Optional[float] = None
    if inventory_position is not None:
        computed_inv_pos = _as_finite_float(inventory_position, "inventory_position")
    elif on_hand_inventory is not None:
        oh = _require_non_negative(on_hand_inventory, "on_hand_inventory")
        oo = _require_non_negative(on_order_inventory, "on_order_inventory") if on_order_inventory is not None else 0.0
        bo = _require_non_negative(backorders, "backorders") if backorders is not None else 0.0
        computed_inv_pos = oh + oo - bo
        if on_order_inventory is None and backorders is None:
            assumptions.append(
                "Computed inventory_position from on_hand_inventory assuming zero scheduled receipts and zero backorders."
            )
        elif on_order_inventory is None:
            assumptions.append("Assumed zero on-order scheduled receipts when computing inventory_position.")
        elif backorders is None:
            assumptions.append("Assumed zero backorders when computing inventory_position.")

    if computed_inv_pos is not None:
        order_quantity = max(0.0, target_inventory_level - computed_inv_pos)
    else:
        assumptions.append(
            "No inventory_position or on_hand_inventory provided; order_quantity is omitted."
        )

    avg_cycle_inventory = 0.5 * d_mean * t_rev
    expected_avg_inventory = avg_cycle_inventory + safety_stock

    inputs_dict = {
        "review_period": t_rev,
        "lead_time": lt,
        "demand_mean": d_mean,
        "demand_std": d_std,
        "lead_time_std": lt_std,
        "service_level": service_level,
        "z_score": z_score,
        "safety_stock": safety_stock,
        "on_hand_inventory": on_hand_inventory,
        "on_order_inventory": on_order_inventory,
        "backorders": backorders,
        "inventory_position": inventory_position,
        "time_unit": time_unit,
    }

    var_defs: Dict[str, str] = {
        "T": f"Review period / interval = {_fmt_val(t_rev)} {time_unit}s",
        "L": f"Replenishment lead time = {_fmt_val(lt)} {time_unit}s",
        "T_plus_L": f"Protection interval = T + L = {_fmt_val(protection_period)} {time_unit}s",
        "d": f"Average demand rate per period = {_fmt_val(d_mean)} units/{time_unit}",
        "mu_T_plus_L": f"Expected demand over protection interval = d * (T + L) = {_fmt_val(mu_prot)} units",
        "SS": f"Safety stock = {_fmt_val(safety_stock)} units",
        "S": f"Order-up-to target inventory level = mu_(T+L) + SS = {_fmt_val(target_inventory_level)} units",
    }
    if d_std is not None:
        var_defs["sigma_d"] = f"Demand standard deviation = {_fmt_val(d_std)} units/{time_unit}"
    if sigma_prot is not None:
        var_defs["sigma_T_plus_L"] = f"Protection interval demand standard deviation = {_fmt_val(sigma_prot, 4)} units"
    if z is not None:
        var_defs["z"] = f"Safety factor (z-score) = {_fmt_val(z, 4)}"
    if csl is not None:
        var_defs["CSL"] = f"Cycle service level = {_fmt_val(csl * 100.0, 2)}%"
    if computed_inv_pos is not None:
        var_defs["IP"] = f"Current inventory position = {_fmt_val(computed_inv_pos)} units"
        var_defs["Q"] = f"Replenishment order quantity = max(0, S - IP) = {_fmt_val(order_quantity)} units"

    formulas: Dict[str, str] = {
        "protection_period": "Protection Interval = T + L",
        "expected_protection_demand": "mu_(T+L) = d * (T + L)",
        "target_inventory_level": "S = mu_(T+L) + SS = d * (T + L) + SS",
    }
    if sigma_prot is not None:
        formulas["protection_demand_std"] = (
            "sigma_(T+L) = sqrt((T + L) * sigma_d^2 + d^2 * sigma_L^2)"
            if (lt_std or 0.0) > 0.0 else
            "sigma_(T+L) = sigma_d * sqrt(T + L)"
        )
        formulas["safety_stock"] = "SS = z * sigma_(T+L)"
    else:
        formulas["safety_stock"] = "SS = provided fixed safety stock"
    if computed_inv_pos is not None:
        formulas["inventory_position"] = "IP = On_Hand + On_Order - Backorders"
        formulas["order_quantity"] = "Q = max(0, S - IP)"

    sub_formulas: Dict[str, str] = {
        "protection_period": f"T + L = {_fmt_val(t_rev)} + {_fmt_val(lt)} = {_fmt_val(protection_period)} {time_unit}s",
        "expected_protection_demand": (
            f"mu_(T+L) = {_fmt_val(d_mean)} * ({_fmt_val(t_rev)} + {_fmt_val(lt)}) "
            f"= {_fmt_val(d_mean)} * {_fmt_val(protection_period)} = {_fmt_val(mu_prot)} units"
        ),
    }
    if sigma_prot is not None and z is not None:
        if (lt_std or 0.0) > 0.0:
            sub_formulas["protection_demand_std"] = (
                f"sigma_(T+L) = sqrt({_fmt_val(protection_period)} * ({_fmt_val(d_std)})^2 + ({_fmt_val(d_mean)})^2 * ({_fmt_val(lt_std)})^2) "
                f"= sqrt({var_prot:,.2f}) = {_fmt_val(sigma_prot, 4)} units"
            )
        else:
            sub_formulas["protection_demand_std"] = (
                f"sigma_(T+L) = {_fmt_val(d_std)} * sqrt({_fmt_val(protection_period)}) = {_fmt_val(sigma_prot, 4)} units"
            )
        sub_formulas["safety_stock"] = f"SS = {_fmt_val(z, 4)} * {_fmt_val(sigma_prot, 4)} = {_fmt_val(safety_stock)} units"
    else:
        sub_formulas["safety_stock"] = f"SS = {_fmt_val(safety_stock)} units (fixed/deterministic)"

    sub_formulas["target_inventory_level"] = f"S = {_fmt_val(mu_prot)} + {_fmt_val(safety_stock)} = {_fmt_val(target_inventory_level)} units"
    if computed_inv_pos is not None:
        if on_hand_inventory is not None or on_order_inventory is not None or backorders is not None:
            sub_formulas["inventory_position"] = (
                f"IP = {_fmt_val(on_hand_inventory or 0.0)} (on hand) + {_fmt_val(on_order_inventory or 0.0)} (on order) - "
                f"{_fmt_val(backorders or 0.0)} (backorders) = {_fmt_val(computed_inv_pos)} units"
            )
        else:
            sub_formulas["inventory_position"] = f"IP = {_fmt_val(computed_inv_pos)} units"
        sub_formulas["order_quantity"] = f"Q = max(0, {_fmt_val(target_inventory_level)} - {_fmt_val(computed_inv_pos)}) = {_fmt_val(order_quantity)} units"

    intermediate = {
        "protection_period": protection_period,
        "expected_protection_demand": mu_prot,
        "protection_demand_variance": var_prot,
        "protection_demand_std": sigma_prot,
        "z_score": z,
        "cycle_service_level": csl,
        "stockout_probability": (1.0 - csl) if csl is not None else None,
        "inventory_position": computed_inv_pos,
        "average_cycle_inventory": avg_cycle_inventory,
        "expected_average_inventory": expected_avg_inventory,
    }
    final_answer = {
        "protection_period": protection_period,
        "safety_stock": safety_stock,
        "target_inventory_level": target_inventory_level,
        "order_quantity": order_quantity,
        "inventory_position": computed_inv_pos,
    }

    result = {
        "model": "fixed_period_review",
        "tool_name": "calculate_periodic_review",
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": var_defs,
        "formula": formulas,
        "formula_with_substituted_values": sub_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "time_unit": time_unit,
        "review_period": t_rev,
        "lead_time": lt,
        "protection_period": protection_period,
        "protection_interval": protection_period,
        "z_score": z,
        "cycle_service_level": csl,
        "stockout_probability": (1.0 - csl) if csl is not None else None,
        "expected_protection_demand": mu_prot,
        "protection_demand_variance": var_prot,
        "protection_demand_std": sigma_prot,
        "safety_stock": safety_stock,
        "target_inventory_level": target_inventory_level,
        "inventory_position": computed_inv_pos,
        "order_quantity": order_quantity,
        "average_cycle_inventory": avg_cycle_inventory,
        "expected_average_inventory": expected_avg_inventory,
    }

    ver_report = verify_calculation("fixed_period_review", inputs_dict, result)
    result["verification_checks"] = ver_report["checks"]
    result["verification_passed"] = ver_report["passed"]
    return result


# Backwards-compatible alias
calculate_fixed_period_review = calculate_periodic_review


# ---------------------------------------------------------------------------
# 5. Newsvendor (Single-Period Model)
# ---------------------------------------------------------------------------

def _resolve_newsvendor_costs(
    *,
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    shortage_penalty: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
) -> Dict[str, Any]:
    assumptions: List[str] = [
        "Single-period ordering decision with no replenishment during the selling period."
    ]

    if underage_cost is not None or overage_cost is not None:
        if underage_cost is None or overage_cost is None:
            raise ValueError(
                "When supplying marginal mismatch costs directly, provide both 'underage_cost' and 'overage_cost'."
            )
        cu = _require_positive(underage_cost, "underage_cost")
        co = _require_positive(overage_cost, "overage_cost")
        has_full_economics = selling_price is not None and unit_cost is not None
        s_eff = _as_finite_float(salvage_value, "salvage_value") if salvage_value is not None else 0.0
        g_eff = _require_non_negative(shortage_penalty, "shortage_penalty") if shortage_penalty is not None else 0.0
    elif selling_price is not None and unit_cost is not None:
        p = _require_positive(selling_price, "selling_price")
        c = _require_non_negative(unit_cost, "unit_cost")
        if salvage_value is None:
            s_eff = 0.0
            assumptions.append("No salvage_value provided; assumed zero salvage value (s = $0.00).")
        else:
            s_eff = _as_finite_float(salvage_value, "salvage_value")
        if shortage_penalty is None:
            g_eff = 0.0
        else:
            g_eff = _require_non_negative(shortage_penalty, "shortage_penalty")
        cu = p - c + g_eff
        co = c - s_eff
        if cu <= 0.0:
            raise ValueError(
                f"Underage cost (p - c + g = {cu}) must be > 0 for stocking to be profitable."
            )
        if co <= 0.0:
            raise ValueError(
                f"Overage cost (c - s = {co}) must be > 0 (otherwise infinite stocking is optimal)."
            )
        has_full_economics = True
    else:
        raise ValueError(
            "Missing cost inputs: provide either ('selling_price' and 'unit_cost') or ('underage_cost' and 'overage_cost')."
        )

    critical_ratio = cu / (cu + co)
    return {
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": critical_ratio,
        "has_full_economics": has_full_economics,
        "selling_price": selling_price,
        "unit_cost": unit_cost,
        "salvage_value": s_eff,
        "shortage_penalty": g_eff,
        "assumptions": assumptions,
    }


def calculate_newsvendor(
    *,
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    shortage_penalty: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    distribution: str = "normal",
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    demand_min: Optional[float] = None,
    demand_max: Optional[float] = None,
    demand_outcomes: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity."""
    econ = _resolve_newsvendor_costs(
        selling_price=selling_price,
        unit_cost=unit_cost,
        salvage_value=salvage_value,
        shortage_penalty=shortage_penalty,
        underage_cost=underage_cost,
        overage_cost=overage_cost,
    )
    cu = econ["underage_cost"]
    co = econ["overage_cost"]
    cr = econ["critical_ratio"]
    assumptions: List[str] = list(econ["assumptions"])

    if not isinstance(distribution, str):
        raise ValueError("distribution must be 'normal', 'uniform', or 'discrete'.")
    dist = distribution.lower().strip()

    if dist == "normal":
        mu = _require_non_negative(demand_mean, "demand_mean")
        sigma = _require_positive(demand_std, "demand_std")
        z_opt = normal_ppf(cr)
        q_opt = max(0.0, mu + z_opt * sigma)
        z_eval = (q_opt - mu) / sigma
        expected_lost_sales = sigma * normal_loss(z_eval)
        expected_sales = mu - expected_lost_sales
        expected_leftover = q_opt - expected_sales
        in_stock_prob = normal_cdf(z_eval)
        dist_details: Dict[str, Any] = {
            "demand_mean": mu,
            "demand_std": sigma,
            "z_score": z_eval,
        }
        assumptions.append(f"Demand follows a Normal distribution N(mean={mu:g}, std={sigma:g}).")
    elif dist == "uniform":
        if demand_min is None or demand_max is None:
            raise ValueError("Uniform distribution requires both 'demand_min' and 'demand_max'.")
        a = _require_non_negative(demand_min, "demand_min")
        b = _as_finite_float(demand_max, "demand_max")
        if b <= a:
            raise ValueError(f"demand_max ({b}) must be > demand_min ({a}).")
        mu = 0.5 * (a + b)
        q_opt = a + cr * (b - a)
        expected_leftover = ((q_opt - a) ** 2) / (2.0 * (b - a))
        expected_lost_sales = ((b - q_opt) ** 2) / (2.0 * (b - a))
        expected_sales = q_opt - expected_leftover
        in_stock_prob = (q_opt - a) / (b - a)
        dist_details = {
            "demand_mean": mu,
            "demand_min": a,
            "demand_max": b,
        }
        assumptions.append(f"Demand follows a Uniform distribution U({a:g}, {b:g}).")
    elif dist == "discrete":
        if not demand_outcomes:
            raise ValueError("Discrete distribution requires non-empty 'demand_outcomes'.")
        pairs = sorted(
            (_require_non_negative(val, "demand_outcome_value"), _require_non_negative(prob, "demand_outcome_prob"))
            for val, prob in demand_outcomes
        )
        total_prob = sum(p for _, p in pairs)
        if total_prob <= 0.0:
            raise ValueError("Discrete probabilities must sum to 1.0.")
        if abs(total_prob - 1.0) > 1e-3:
            raise ValueError(f"Discrete probabilities must sum to 1.0, got {total_prob:.6f}.")
        pairs = [(v, p / total_prob) for v, p in pairs]

        merged: List[Tuple[float, float]] = []
        for v, p in pairs:
            if merged and abs(merged[-1][0] - v) < 1e-12:
                merged[-1] = (merged[-1][0], merged[-1][1] + p)
            else:
                merged.append((v, p))

        cdf = 0.0
        q_opt = merged[-1][0]
        in_stock_prob = 1.0
        for v, p in merged:
            cdf += p
            if cdf >= cr - 1e-12:
                q_opt = v
                in_stock_prob = cdf
                break

        mu = sum(v * p for v, p in merged)
        expected_sales = sum(min(v, q_opt) * p for v, p in merged)
        expected_leftover = sum(max(0.0, q_opt - v) * p for v, p in merged)
        expected_lost_sales = sum(max(0.0, v - q_opt) * p for v, p in merged)
        dist_details = {
            "demand_mean": mu,
            "demand_outcomes": merged,
        }
        assumptions.append(f"Demand follows an empirical discrete distribution with {len(merged)} outcomes.")
    else:
        raise ValueError("distribution must be 'normal', 'uniform', or 'discrete'.")

    expected_mismatch_cost = cu * expected_lost_sales + co * expected_leftover
    if econ["has_full_economics"]:
        p_val = float(selling_price)  # type: ignore[arg-type]
        c_val = float(unit_cost)  # type: ignore[arg-type]
        s_val = float(econ["salvage_value"])
        g_val = float(econ["shortage_penalty"])
        expected_profit: Optional[float] = (
            (p_val - s_val) * expected_sales - (c_val - s_val) * q_opt - g_val * expected_lost_sales
        )
    else:
        expected_profit = None

    fill_rate = (expected_sales / mu) if mu > 0.0 else 1.0
    rounded_q = round(q_opt)

    inputs_dict = {
        "selling_price": selling_price,
        "unit_cost": unit_cost,
        "salvage_value": econ["salvage_value"],
        "shortage_penalty": econ["shortage_penalty"],
        "underage_cost": underage_cost,
        "overage_cost": overage_cost,
        "distribution": dist,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
        "demand_min": demand_min,
        "demand_max": demand_max,
        "demand_outcomes": demand_outcomes,
    }

    var_defs: Dict[str, str] = {
        "Cu": f"Underage cost (cost of understocking per unit; lost profit margin) = ${_fmt_val(cu)}/unit",
        "Co": f"Overage cost (cost of overstocking per unit; net loss per unsold unit) = ${_fmt_val(co)}/unit",
        "CR": f"Critical ratio (target in-stock probability) = Cu / (Cu + Co) = {cr:.4f} ({cr * 100.0:.2f}%)",
        "distribution": f"Demand distribution = {dist}",
        "Q*": f"Optimal order quantity = {_fmt_val(q_opt)} units (nearest integer: {rounded_q:,} units)",
        "in_stock_probability": f"Achieved in-stock probability = {_fmt_val(in_stock_prob * 100.0, 2)}%",
        "fill_rate": f"Expected volumetric fill rate = {_fmt_val(fill_rate * 100.0, 2)}%",
        "E_Sales": f"Expected sales = {_fmt_val(expected_sales)} units",
        "E_Leftover": f"Expected leftover inventory = {_fmt_val(expected_leftover)} units",
        "E_Lost_Sales": f"Expected unmet lost sales = {_fmt_val(expected_lost_sales)} units",
    }
    if selling_price is not None:
        var_defs["p"] = f"Unit selling price = ${_fmt_val(selling_price)}/unit"
    if unit_cost is not None:
        var_defs["c"] = f"Unit purchase or production cost = ${_fmt_val(unit_cost)}/unit"
    if econ.get("salvage_value") is not None:
        var_defs["s"] = f"Unit salvage value = ${_fmt_val(econ['salvage_value'])}/unit"
    if econ.get("shortage_penalty") is not None and econ["shortage_penalty"] > 0:
        var_defs["B"] = f"Unit shortage penalty = ${_fmt_val(econ['shortage_penalty'])}/unit"
    if expected_profit is not None:
        var_defs["E_Profit"] = f"Expected profit = ${_fmt_val(expected_profit)}"
    else:
        var_defs["E_Mismatch_Cost"] = f"Expected mismatch cost = ${_fmt_val(expected_mismatch_cost)}"

    formulas: Dict[str, str] = {
        "underage_cost": "Cu = (p - c) + B (or direct underage cost)",
        "overage_cost": "Co = c - s (or direct overage cost)",
        "critical_ratio": "CR = Cu / (Cu + Co)",
        "optimal_order_quantity": (
            "Normal: Q* = mu + z* * sigma where z* = Phi^-1(CR); "
            "Uniform: Q* = a + CR * (b - a); "
            "Discrete: Smallest Q with sum_{x <= Q} P(x) >= CR"
        ),
        "expected_sales": "E[Sales] = mu - E[Lost Sales]",
        "expected_leftover": "E[Leftover] = Q* - E[Sales]",
        "expected_lost_sales": (
            "Normal: E[Lost Sales] = sigma * L(z*); "
            "Uniform / Discrete: sum_{x > Q} (x - Q) * P(x)"
        ),
    }
    if expected_profit is not None:
        formulas["expected_profit"] = "E[Profit] = p * E[Sales] + s * E[Leftover] - c * Q* - B * E[Lost Sales]"
    else:
        formulas["expected_mismatch_cost"] = "E[Mismatch Cost] = Cu * E[Lost Sales] + Co * E[Leftover]"

    p_disp = selling_price
    c_disp = unit_cost
    s_disp = econ.get("salvage_value")
    g_disp = econ.get("shortage_penalty") or 0.0

    cu_sub = (
        f"Cu = (${_fmt_val(p_disp)} - ${_fmt_val(c_disp)}) + ${_fmt_val(g_disp)} = ${_fmt_val(cu)}/unit"
        if (p_disp is not None and c_disp is not None) else
        f"Cu = ${_fmt_val(cu)}/unit (direct)"
    )
    co_sub = (
        f"Co = ${_fmt_val(c_disp)} - ${_fmt_val(s_disp)} = ${_fmt_val(co)}/unit"
        if c_disp is not None else
        f"Co = ${_fmt_val(co)}/unit (direct)"
    )
    cr_sub = f"CR = ${_fmt_val(cu)} / (${_fmt_val(cu)} + ${_fmt_val(co)}) = ${_fmt_val(cu)} / ${_fmt_val(cu + co)} = {cr:.4f} ({cr * 100.0:.2f}%)"

    if dist == "normal":
        z_str = f"{dist_details.get('z_score', 0.0):.4f}"
        q_sub = (
            f"z* = Phi^-1({cr:.4f}) = {z_str}; "
            f"Q* = {_fmt_val(demand_mean)} + ({z_str} * {_fmt_val(demand_std)}) "
            f"= {_fmt_val(demand_mean)} + {(float(z_str) * (demand_std or 0.0)):,.2f} "
            f"= {_fmt_val(q_opt)} units (~{rounded_q:,} units)"
        )
    elif dist == "uniform":
        q_sub = (
            f"Q* = {_fmt_val(demand_min)} + {cr:.4f} * ({_fmt_val(demand_max)} - {_fmt_val(demand_min)}) "
            f"= {_fmt_val(demand_min)} + {cr * ((demand_max or 0.0) - (demand_min or 0.0)):,.2f} "
            f"= {_fmt_val(q_opt)} units (~{rounded_q:,} units)"
        )
    else:
        q_sub = f"Smallest outcome Q where Cumulative Probability >= {cr:.4f} -> Q* = {_fmt_val(q_opt)} units"

    sub_formulas: Dict[str, str] = {
        "underage_cost": cu_sub,
        "overage_cost": co_sub,
        "critical_ratio": cr_sub,
        "optimal_order_quantity": q_sub,
        "expected_sales": f"E[Sales] = {_fmt_val(mu)} - {_fmt_val(expected_lost_sales)} = {_fmt_val(expected_sales)} units",
        "expected_leftover": f"E[Leftover] = {_fmt_val(q_opt)} - {_fmt_val(expected_sales)} = {_fmt_val(expected_leftover)} units",
        "expected_lost_sales": f"E[Lost Sales] = {_fmt_val(expected_lost_sales)} units",
    }
    if expected_profit is not None:
        sub_formulas["expected_profit"] = (
            f"E[Profit] = (${_fmt_val(p_disp)} * {_fmt_val(expected_sales)}) + (${_fmt_val(s_disp)} * {_fmt_val(expected_leftover)}) "
            f"- (${_fmt_val(c_disp)} * {_fmt_val(q_opt)}) = ${_fmt_val(expected_profit)}"
        )
    else:
        sub_formulas["expected_mismatch_cost"] = (
            f"E[Mismatch Cost] = (${_fmt_val(cu)} * {_fmt_val(expected_lost_sales)}) + (${_fmt_val(co)} * {_fmt_val(expected_leftover)}) = ${_fmt_val(expected_mismatch_cost)}"
        )

    intermediate = {
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": cr,
        "in_stock_probability": in_stock_prob,
        "stockout_probability": 1.0 - in_stock_prob,
        "fill_rate": fill_rate,
        "expected_sales": expected_sales,
        "expected_leftover": expected_leftover,
        "expected_lost_sales": expected_lost_sales,
        "expected_mismatch_cost": expected_mismatch_cost,
        "expected_profit": expected_profit,
        **dist_details,
    }
    final_answer = {
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": cr,
        "recommended_order_quantity": q_opt,
        "rounded_order_quantity": rounded_q,
        "expected_profit": expected_profit,
        "expected_mismatch_cost": expected_mismatch_cost,
    }

    result = {
        "model": "newsvendor",
        "tool_name": "calculate_newsvendor",
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": var_defs,
        "formula": formulas,
        "formula_with_substituted_values": sub_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "distribution": dist,
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": cr,
        "recommended_order_quantity": q_opt,
        "optimal_order_quantity": q_opt,
        "rounded_order_quantity": rounded_q,
        "in_stock_probability": in_stock_prob,
        "stockout_probability": 1.0 - in_stock_prob,
        "fill_rate": fill_rate,
        "expected_sales": expected_sales,
        "expected_leftover": expected_leftover,
        "expected_lost_sales": expected_lost_sales,
        "expected_mismatch_cost": expected_mismatch_cost,
        "expected_profit": expected_profit,
        **dist_details,
    }

    ver_report = verify_calculation("newsvendor", inputs_dict, result)
    result["verification_checks"] = ver_report["checks"]
    result["verification_passed"] = ver_report["passed"]
    return result


# ---------------------------------------------------------------------------
# Analytical & Sanity Verification Suite (Stage 6: Verify)
# ---------------------------------------------------------------------------

def _make_check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def verify_calculation(
    model_id: str,
    inputs: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Independently verify mathematical optimality, feasibility, and conservation laws."""
    checks: List[Dict[str, Any]] = []

    if model_id == "eoq":
        q = result["optimal_order_quantity"]
        ord_cost = result["annual_ordering_cost"]
        hold_cost = result["annual_holding_cost"]
        d = inputs["annual_demand"]
        s = inputs["ordering_cost"]
        h = result["effective_holding_cost"]

        checks.append(
            _make_check(
                "Positivity",
                q > 0.0 and ord_cost > 0.0 and hold_cost > 0.0,
                f"Q*={q:.4f} > 0, ordering=${ord_cost:.2f}, holding=${hold_cost:.2f}",
            )
        )
        rel_diff = abs(ord_cost - hold_cost) / max(ord_cost, 1e-12)
        checks.append(
            _make_check(
                "EOQ Cost Balance (Ordering == Holding)",
                rel_diff < 1e-9,
                f"|{ord_cost:.6f} - {hold_cost:.6f}| / {ord_cost:.6f} = {rel_diff:.2e}",
            )
        )
        tc_opt = (d / q) * s + (q / 2.0) * h
        tc_low = (d / (0.98 * q)) * s + (0.98 * q / 2.0) * h
        tc_high = (d / (1.02 * q)) * s + (1.02 * q / 2.0) * h
        checks.append(
            _make_check(
                "Local Cost Minimum",
                tc_opt < tc_low and tc_opt < tc_high,
                f"TC(0.98Q)=${tc_low:.2f} > TC(Q*)=${tc_opt:.2f} < TC(1.02Q)=${tc_high:.2f}",
            )
        )

    elif model_id == "quantity_discount":
        q = result["optimal_order_quantity"]
        opt_tier_idx = result["optimal_tier"]
        tiers = result["tier_evaluations"]
        selected_tier = next(t for t in tiers if t["tier_index"] == opt_tier_idx)
        min_q = selected_tier["min_qty"]
        max_q = selected_tier["max_qty"] if selected_tier["max_qty"] is not None else float("inf")
        tol = 1e-6
        in_range = (q >= min_q - tol) and (
            q < max_q if selected_tier["max_exclusive"] else q <= max_q + tol
        )
        checks.append(
            _make_check(
                "Tier Bound Feasibility",
                in_range and q > 0.0,
                f"Q*={q:.4f} within Tier {opt_tier_idx} [{min_q}, {max_q}]",
            )
        )
        best_cost = result["total_annual_cost"]
        all_costs = [
            t["total_annual_cost"] for t in tiers if t["total_annual_cost"] is not None
        ]
        is_global_min = all(best_cost <= c + 1e-9 for c in all_costs)
        checks.append(
            _make_check(
                "Global Minimum Across All Tiers",
                is_global_min,
                f"Selected total cost ${best_cost:,.2f} <= all {len(all_costs)} tier candidates",
            )
        )
        comp_sum = (
            result["annual_purchase_cost"]
            + result["annual_ordering_cost"]
            + result["annual_holding_cost"]
        )
        checks.append(
            _make_check(
                "Cost Component Reconciliation",
                abs(comp_sum - best_cost) < 1e-6,
                f"Purchase + Ordering + Holding = ${comp_sum:,.2f} vs Total = ${best_cost:,.2f}",
            )
        )

    elif model_id in {"safety_stock_rop", "rop"}:
        z = result["z_score"]
        csl = result["cycle_service_level"]
        mu_lt = result["expected_lead_time_demand"]
        sigma_lt = result["lead_time_demand_std"]
        ss = result["safety_stock"]
        rop = result["reorder_point"]

        if sigma_lt == 0.0:
            checks.append(
                _make_check(
                    "Deterministic Lead-Time Demand Check",
                    abs(ss) < 1e-12 and abs(rop - mu_lt) < 1e-9,
                    f"sigma_LT=0 -> SS={ss:.4f}, ROP={rop:.4f} == mu_LT={mu_lt:.4f}",
                )
            )
        else:
            assert z is not None
            cdf_err = abs(normal_cdf(z) - csl)
            checks.append(
                _make_check(
                    "Normal Quantile Round-Trip",
                    cdf_err < 1e-7,
                    f"|Phi({z:.6f}) - {csl:.6f}| = {cdf_err:.2e}",
                )
            )
            rop_err = abs((mu_lt + z * sigma_lt) - rop)
            checks.append(
                _make_check(
                    "ROP Identity (mu_LT + SS == ROP)",
                    abs((mu_lt + ss) - rop) < 1e-9 and rop_err < 1e-9,
                    f"{mu_lt:.4f} + {ss:.4f} = {rop:.4f}",
                )
            )
            checks.append(
                _make_check(
                    "Service Level / Safety Stock Sign Consistency",
                    (csl >= 0.5 and ss >= -1e-9) or (csl < 0.5 and ss < 0.0),
                    f"CSL={csl:.4f}, Safety Stock={ss:.4f}",
                )
            )

    elif model_id in {"fixed_period_review", "periodic_review"}:
        z = result["z_score"]
        csl = result["cycle_service_level"]
        prot = result["protection_period"]
        mu_p = result["expected_protection_demand"]
        sigma_p = result["protection_demand_std"]
        ss = result["safety_stock"]
        target = result["target_inventory_level"]

        checks.append(
            _make_check(
                "Protection Period (T + L)",
                abs((inputs["review_period"] + inputs["lead_time"]) - prot) < 1e-9,
                f"T + L = {prot:.4f}",
            )
        )
        if sigma_p is None:
            checks.append(
                _make_check(
                    "Order-Up-To Target Identity (mu_(T+L) + SS == S)",
                    abs((mu_p + ss) - target) < 1e-9,
                    f"{mu_p:.4f} + {ss:.4f} = {target:.4f}",
                )
            )
        elif sigma_p == 0.0:
            checks.append(
                _make_check(
                    "Deterministic Protection Demand Check",
                    abs(ss) < 1e-12 and abs(target - mu_p) < 1e-9,
                    f"sigma_(T+L)=0 -> SS={ss:.4f}, Target={target:.4f} == mu_(T+L)={mu_p:.4f}",
                )
            )
        else:
            if z is not None and csl is not None:
                cdf_err = abs(normal_cdf(z) - csl)
                checks.append(
                    _make_check(
                        "Normal Quantile Round-Trip",
                        cdf_err < 1e-7,
                        f"|Phi({z:.6f}) - {csl:.6f}| = {cdf_err:.2e}",
                    )
                )
            checks.append(
                _make_check(
                    "Order-Up-To Target Identity (mu_(T+L) + SS == S)",
                    abs((mu_p + ss) - target) < 1e-9,
                    f"{mu_p:.4f} + {ss:.4f} = {target:.4f}",
                )
            )
        if result["order_quantity"] is not None and result["inventory_position"] is not None:
            expected_q = max(0.0, target - result["inventory_position"])
            checks.append(
                _make_check(
                    "Order Quantity Non-Negativity",
                    abs(result["order_quantity"] - expected_q) < 1e-9 and result["order_quantity"] >= 0.0,
                    f"Q = max(0, {target:.2f} - {result['inventory_position']:.2f}) = {result['order_quantity']:.2f}",
                )
            )

    elif model_id == "newsvendor":
        cr = result["critical_ratio"]
        q = result["optimal_order_quantity"]
        mu = result["demand_mean"]
        exp_sales = result["expected_sales"]
        exp_left = result["expected_leftover"]
        exp_lost = result["expected_lost_sales"]

        checks.append(
            _make_check(
                "Critical Ratio Bounds",
                0.0 < cr < 1.0,
                f"CR = {cr:.6f} in (0, 1)",
            )
        )
        if result["distribution"] == "normal":
            prob_at_q = normal_cdf((q - mu) / result["demand_std"])
            if q > 0.0:
                checks.append(
                    _make_check(
                        "Normal Optimality Condition (Phi((Q*-mu)/sigma) == CR)",
                        abs(prob_at_q - cr) < 1e-7,
                        f"Phi(z*)={prob_at_q:.6f} vs CR={cr:.6f}",
                    )
                )
        elif result["distribution"] == "uniform":
            a = result["demand_min"]
            b = result["demand_max"]
            cdf_q = (q - a) / (b - a)
            checks.append(
                _make_check(
                    "Uniform Optimality Condition (F(Q*) == CR)",
                    abs(cdf_q - cr) < 1e-9,
                    f"F(Q*)={cdf_q:.6f} vs CR={cr:.6f}",
                )
            )
        elif result["distribution"] == "discrete":
            outcomes = result["demand_outcomes"]
            cdf_at_q = sum(p for v, p in outcomes if v <= q + 1e-12)
            below_q = [v for v, _ in outcomes if v < q - 1e-12]
            cdf_prev = sum(p for v, p in outcomes if v <= max(below_q) + 1e-12) if below_q else 0.0
            checks.append(
                _make_check(
                    "Discrete Optimality Condition (F(Q_prev) < CR <= F(Q*))",
                    cdf_prev < cr - 1e-12 and cdf_at_q >= cr - 1e-12,
                    f"F(prev)={cdf_prev:.4f} < CR={cr:.4f} <= F(Q*)={cdf_at_q:.4f}",
                )
            )

        stock_cons = abs((exp_sales + exp_left) - q)
        demand_cons = abs((exp_sales + exp_lost) - mu)
        checks.append(
            _make_check(
                "Supply & Demand Conservation Identities",
                stock_cons < 1e-6 and demand_cons < 1e-6,
                f"E[Sales]+E[Leftover]={exp_sales + exp_left:.4f} (Q*={q:.4f}); "
                f"E[Sales]+E[Lost]={exp_sales + exp_lost:.4f} (mu={mu:.4f})",
            )
        )
    else:
        raise ValueError(f"Unknown model_id for verification: {model_id!r}")

    all_passed = all(c["passed"] for c in checks)
    return {"passed": all_passed, "checks": checks}


# ---------------------------------------------------------------------------
# Tool Registry (Stage 4: Select Python Tool)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "eoq": {
        "name": "Economic Order Quantity (EOQ)",
        "tool_name": "calculate_eoq",
        "callable": calculate_eoq,
        "required_params": ["annual_demand", "ordering_cost"],
        "holding_requirement": "holding_cost OR (holding_rate AND unit_cost)",
        "optional_params": ["holding_cost", "holding_rate", "unit_cost", "working_days_per_year"],
        "description": "Minimizes total annual ordering and holding costs for constant deterministic demand.",
    },
    "quantity_discount": {
        "name": "Quantity Discount Model",
        "tool_name": "calculate_quantity_discount",
        "callable": calculate_quantity_discount,
        "required_params": ["annual_demand", "ordering_cost", "tiers"],
        "holding_requirement": "holding_rate OR holding_cost",
        "optional_params": ["holding_rate", "holding_cost", "discount_type"],
        "description": "Finds the cost-minimizing order quantity across supplier price-break tiers.",
    },
    "safety_stock_rop": {
        "name": "Safety Stock and Reorder Point",
        "tool_name": "calculate_rop",
        "callable": calculate_rop,
        "required_params": [],
        "holding_requirement": "(demand_mean + lead_time + demand_std/lead_time_std) OR (lead_time_demand_mean + lead_time_demand_std), plus service_level/z_score when stochastic",
        "optional_params": [
            "demand_mean",
            "demand_std",
            "lead_time",
            "lead_time_std",
            "lead_time_demand_mean",
            "lead_time_demand_std",
            "service_level",
            "z_score",
            "service_level_type",
            "order_quantity",
            "holding_cost",
            "time_unit",
        ],
        "description": "Determines safety stock buffer and continuous-review reorder point for constant/variable demand and lead time.",
    },
    "fixed_period_review": {
        "name": "Fixed-Period Review (Periodic Review)",
        "tool_name": "calculate_periodic_review",
        "callable": calculate_periodic_review,
        "required_params": ["review_period", "lead_time", "demand_mean"],
        "holding_requirement": "(demand_std/lead_time_std + service_level/z_score) OR explicit safety_stock",
        "optional_params": [
            "demand_std",
            "lead_time_std",
            "service_level",
            "z_score",
            "safety_stock",
            "on_hand_inventory",
            "on_order_inventory",
            "backorders",
            "inventory_position",
            "time_unit",
        ],
        "description": "Determines protection period T+L, target order-up-to level S, safety stock, and order quantity.",
    },
    "newsvendor": {
        "name": "Newsvendor (Single-Period Model)",
        "tool_name": "calculate_newsvendor",
        "callable": calculate_newsvendor,
        "required_params": [],
        "holding_requirement": "(selling_price + unit_cost) OR (underage_cost + overage_cost), plus demand distribution",
        "optional_params": [
            "selling_price",
            "unit_cost",
            "salvage_value",
            "shortage_penalty",
            "underage_cost",
            "overage_cost",
            "distribution",
            "demand_mean",
            "demand_std",
            "demand_min",
            "demand_max",
            "demand_outcomes",
        ],
        "description": "Optimizes a one-time perishable or seasonal order using underage cost, overage cost, and critical ratio.",
    },
}


def get_tool(model_id: str) -> Callable[..., Dict[str, Any]]:
    """Retrieve the Python tool function for a given model identifier."""
    if model_id not in TOOL_REGISTRY:
        raise KeyError(f"Unsupported inventory model: {model_id!r}")
    return TOOL_REGISTRY[model_id]["callable"]
