"""Multicriteria ABCD Inventory Classification Engine (BAN 517 Methodology).

Implements the official BAN 517 Inventory Classification Methodology:
1. Independent Pareto ranking & cumulative percentage classification for Sales ($)
   and Hits, sorted highest-to-lowest with deterministic tie-breaking by Item Number
   (ascending).
2. Continuous, non-overlapping cumulative-percentage boundaries (full precision,
   no pre-rounding):
   - A: cum_pct <= 60.0%
   - B: 60.0% < cum_pct <= 80.0%
   - C: 80.0% < cum_pct <= 90.0%
   - D: cum_pct > 90.0%
3. Numerical score mapping: A = 4, B = 3, C = 2, D = 1.
4. Multicriteria weighted score:
   Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
5. Continuous, non-overlapping final weighted-score boundaries (full precision):
   - A: score >= 3.25
   - B: 2.50 <= score < 3.25
   - C: 1.75 <= score < 2.50
   - D: score < 1.75
6. Full mapping back to original items via Item Number, class-level summary,
   reconciliation audit, and course inventory policy recommendations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Official Methodology Constants
# ---------------------------------------------------------------------------

REQUIRED_ABCD_CSV_COLUMNS: List[str] = [
    "Item Number",
    "Item Name",
    "UoM",
    "Sales ($)",
    "Hits",
]

STANDARD_INVENTORY_CSV_COLUMNS: List[str] = [
    "Product",
    "Category",
    "Units",
    "Unit Cost",
]

ABCD_OUTPUT_COLUMNS: List[str] = [
    "Item Number",
    "Item Name",
    "UoM",
    "Sales ($)",
    "Hits",
    "Cumulative Sales",
    "Cumulative Sales %",
    "Sales Class",
    "Sales Score",
    "Cumulative Hits",
    "Cumulative Hits %",
    "Hits Class",
    "Hits Score",
    "Combined Weighted Score",
    "Final ABCD Class",
]

ABCD_CLASS_ORDER: List[str] = ["A", "B", "C", "D"]

SALES_WEIGHT: float = 0.65
HITS_WEIGHT: float = 0.35

CLASS_SCORE_MAP: Dict[str, int] = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
}

# Official BAN 517 project dataset reconciliation benchmarks (ProjectDataV2_UTF8.csv)
PROJECT_BENCHMARK_ITEMS: int = 205
PROJECT_BENCHMARK_SALES: float = 82_046_150.0
PROJECT_BENCHMARK_HITS: float = 23_252.0

# Module Inventory Policy Recommendations
ABCD_INVENTORY_POLICIES: Dict[str, Dict[str, str]] = {
    "A": {
        "class": "A",
        "cycle_counting": "Weekly cycle counting",
        "cycle_service_level": "95%–98%",
        "safety_stock_guidance": "Maintain high service-level safety stock (95%–98% CSL) once demand variability and lead-time data are available.",
        "action_summary": "Tightest inventory control, frequent review, and priority replenishment.",
    },
    "B": {
        "class": "B",
        "cycle_counting": "Monthly cycle counting",
        "cycle_service_level": "85%–90%",
        "safety_stock_guidance": "Maintain moderate safety stock (85%–90% CSL) once demand variability and lead-time data are available.",
        "action_summary": "Balanced inventory control with routine monthly tracking.",
    },
    "C": {
        "class": "C",
        "cycle_counting": "Annual cycle counting",
        "cycle_service_level": "70%–80%",
        "safety_stock_guidance": "Maintain lower safety stock (70%–80% CSL) once demand variability and lead-time data are available.",
        "action_summary": "Simplified control procedures and low administrative overhead.",
    },
    "D": {
        "class": "D",
        "cycle_counting": "No cycle counting under the course policy",
        "cycle_service_level": "No safety stock",
        "safety_stock_guidance": "Do not hold safety stock; evaluate for discounting, donating, or disposing of remaining inventory.",
        "action_summary": "Consider discounting, donating, or disposing of the inventory.",
    },
}

SAFETY_STOCK_DISCLAIMER: str = (
    "Actual safety stock quantities are NOT calculated from Sales ($) and Hits alone. "
    "Computing numerical safety stock requires additional operational parameters—specifically "
    "demand rate and demand standard deviation per period, replenishment lead time (and lead-time "
    "standard deviation if variable), and a chosen target cycle service level. Once those inputs "
    "are available, use the Safety Stock & Reorder Point tool in this application."
)

METHODOLOGY_NOTES: List[str] = [
    (
        "Cumulative Percentage Thresholds (continuous, non-overlapping interpretation of "
        "displayed whole-number bands 0–60%, 61–80%, 81–90%, 91–100%, evaluated at full precision "
        "without pre-rounding): A: <= 60.0%; B: > 60.0% and <= 80.0%; C: > 80.0% and <= 90.0%; D: > 90.0%."
    ),
    (
        "Weighted Score Formula: Combined Weighted Score = 0.65 × Sales Score + 0.35 × Hits Score, "
        "where A = 4, B = 3, C = 2, D = 1."
    ),
    (
        "Final ABCD Score Boundaries (continuous, non-overlapping interpretation of displayed "
        "ranges 3.25–4.00, 2.50–3.24, 1.75–2.49, Below 1.75, evaluated at full precision): "
        "A: >= 3.25; B: >= 2.50 and < 3.25; C: >= 1.75 and < 2.50; D: < 1.75."
    ),
    (
        "Independent Ranking & Mapping: Sales ($) and Hits are ranked independently from highest to "
        "lowest, cumulative metrics and classes are computed within each independent ranking order, "
        "and results are mapped back to the original dataset via Item Number."
    ),
    (
        "Deterministic Tie-Breaking Assumption: When two or more items share an identical Sales ($) "
        "or Hits value, ties are broken deterministically by sorting Item Number in ascending order "
        "(numeric ascending for integer item identifiers, lexicographic ascending otherwise)."
    ),
]


# ---------------------------------------------------------------------------
# Pure Deterministic Classification Helpers
# ---------------------------------------------------------------------------

def detect_csv_schema(columns: Sequence[str]) -> str:
    """Identify which supported CSV schema matches the uploaded DataFrame columns.

    Returns:
        - "abcd_project": contains all BAN 517 ABCD columns
        - "standard_inventory": contains all original 4-column inventory CSV columns
        - "unknown": missing required columns for both supported formats
    """
    cleaned = {str(c).lstrip("\ufeff").strip() for c in columns}
    if all(col in cleaned for col in REQUIRED_ABCD_CSV_COLUMNS):
        return "abcd_project"
    if all(col in cleaned for col in STANDARD_INVENTORY_CSV_COLUMNS):
        return "standard_inventory"
    return "unknown"


def classify_cumulative_pct(cum_pct: float) -> str:
    """Assign Pareto class A, B, C, or D from a full-precision cumulative percentage (0–100 scale).

    Continuous, non-overlapping interpretation of 0–60%, 61–80%, 81–90%, 91–100%:
      - A: cum_pct <= 60.0
      - B: 60.0 < cum_pct <= 80.0
      - C: 80.0 < cum_pct <= 90.0
      - D: cum_pct > 90.0
    """
    val = float(cum_pct)
    if math.isnan(val) or val < 0.0:
        raise ValueError(f"Cumulative percentage must be a non-negative finite number, got {cum_pct!r}.")
    if val <= 60.0:
        return "A"
    if val <= 80.0:
        return "B"
    if val <= 90.0:
        return "C"
    return "D"


def class_to_score(abcd_class: str) -> int:
    """Convert an ABCD letter grade to its official numeric score (A=4, B=3, C=2, D=1)."""
    key = str(abcd_class).strip().upper()
    if key not in CLASS_SCORE_MAP:
        raise ValueError(f"Invalid ABCD class {abcd_class!r}; expected one of A, B, C, D.")
    return CLASS_SCORE_MAP[key]


def compute_weighted_score(sales_score: int, hits_score: int) -> float:
    """Compute full-precision Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score."""
    return SALES_WEIGHT * float(sales_score) + HITS_WEIGHT * float(hits_score)


def classify_weighted_score(score: float) -> str:
    """Assign Final ABCD Class from a full-precision Combined Weighted Score.

    Continuous, non-overlapping interpretation of 3.25–4.00, 2.50–3.24, 1.75–2.49, Below 1.75:
      - A: score >= 3.25
      - B: 2.50 <= score < 3.25
      - C: 1.75 <= score < 2.50
      - D: score < 1.75
    """
    val = float(score)
    if math.isnan(val):
        raise ValueError(f"Weighted score must be a finite number, got {score!r}.")
    if val >= 3.25:
        return "A"
    if val >= 2.50:
        return "B"
    if val >= 1.75:
        return "C"
    return "D"


def _item_tie_break_key(value: Any) -> Tuple[int, Any]:
    """Deterministic ascending tie-break key for Item Number (numeric when integer, else string)."""
    s = str(value).strip()
    if s.isdigit():
        return (0, int(s))
    try:
        num = float(s)
        if num.is_integer():
            return (0, int(num))
    except (ValueError, TypeError):
        pass
    return (1, s)


def _clean_numeric_value(val: Any) -> float:
    """Parse numeric values safely, stripping optional currency symbols, commas, and whitespace."""
    if val is None:
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip().replace("$", "").replace(",", "")
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Main Pandas Classification & Reconciliation Pipeline
# ---------------------------------------------------------------------------

def classify_abcd_dataframe(df: Any) -> Dict[str, Any]:
    """Perform full BAN 517 Multicriteria ABCD Classification on a pandas DataFrame.

    Args:
        df: pandas.DataFrame loaded from the uploaded CSV.

    Returns:
        Structured dictionary containing:
        - valid (bool), missing_columns, error_message
        - metadata: num_rows, num_columns, column_names
        - totals: total_items, total_sales, total_hits
        - classified_df: pandas.DataFrame with all 15 required columns aligned to original items
        - summary_df: pandas.DataFrame aggregated by final class A, B, C, D
        - reconciliation: dict of all internal & project benchmark verification checks
        - policies: course inventory policies & safety stock disclaimer
        - methodology_notes: documented thresholds, weights, and tie-breaking assumptions
    """
    raw_cols = list(df.columns)
    cleaned_map = {c: str(c).lstrip("\ufeff").strip() for c in raw_cols}
    cleaned_cols = [cleaned_map[c] for c in raw_cols]

    missing = [col for col in REQUIRED_ABCD_CSV_COLUMNS if col not in cleaned_cols]
    num_rows = int(len(df))
    num_cols = int(len(raw_cols))

    if missing:
        return {
            "valid": False,
            "missing_columns": missing,
            "error_message": (
                f"Missing required ABCD classification column(s): {', '.join(missing)}. "
                f"Required columns are: {', '.join(REQUIRED_ABCD_CSV_COLUMNS)}."
            ),
            "num_rows": num_rows,
            "num_columns": num_cols,
            "column_names": cleaned_cols,
        }

    import pandas as pd

    # Preserve original DataFrame without mutating caller's object; normalize header BOM/whitespace
    work_df = df.copy()
    if any(cleaned_map[c] != c for c in raw_cols):
        work_df = work_df.rename(columns=cleaned_map)

    if num_rows == 0:
        return {
            "valid": False,
            "missing_columns": [],
            "error_message": "Uploaded CSV contains 0 data rows; cannot perform ABCD classification.",
            "num_rows": 0,
            "num_columns": num_cols,
            "column_names": cleaned_cols,
        }

    # Check Item Number uniqueness so mapping back to original items is strictly 1-to-1
    item_ids = list(work_df["Item Number"])
    if len(set(item_ids)) != len(item_ids):
        return {
            "valid": False,
            "missing_columns": [],
            "error_message": (
                "Duplicate Item Number values detected. Item Number must uniquely identify "
                "each inventory item for independent Sales/Hits mapping."
            ),
            "num_rows": num_rows,
            "num_columns": num_cols,
            "column_names": cleaned_cols,
        }

    # Parse Sales ($) and Hits cleanly at full precision while preserving original Item Number, Item Name, UoM
    sales_numeric = pd.Series(
        [_clean_numeric_value(v) for v in work_df["Sales ($)"]],
        index=work_df.index,
    )
    hits_numeric = pd.Series(
        [_clean_numeric_value(v) for v in work_df["Hits"]],
        index=work_df.index,
    )

    if any(math.isnan(v) for v in sales_numeric) or any(math.isnan(v) for v in hits_numeric):
        return {
            "valid": False,
            "missing_columns": [],
            "error_message": (
                "Non-numeric or missing values found in 'Sales ($)' or 'Hits'. "
                "All rows must contain valid numeric Sales ($) and Hits values."
            ),
            "num_rows": num_rows,
            "num_columns": num_cols,
            "column_names": cleaned_cols,
        }

    if any(v < 0.0 for v in sales_numeric) or any(v < 0.0 for v in hits_numeric):
        return {
            "valid": False,
            "missing_columns": [],
            "error_message": (
                "Negative values found in 'Sales ($)' or 'Hits'. "
                "Pareto cumulative percentage classification requires non-negative values."
            ),
            "num_rows": num_rows,
            "num_columns": num_cols,
            "column_names": cleaned_cols,
        }

    work_df["Sales ($)"] = sales_numeric
    work_df["Hits"] = hits_numeric

    total_sales = float(work_df["Sales ($)"].sum())
    total_hits = float(work_df["Hits"].sum())

    if total_sales <= 0.0 or total_hits <= 0.0:
        return {
            "valid": False,
            "missing_columns": [],
            "error_message": (
                f"Total Sales (${total_sales:,.2f}) and Total Hits ({total_hits:,.0f}) must both be "
                "strictly positive to compute cumulative percentages."
            ),
            "num_rows": num_rows,
            "num_columns": num_cols,
            "column_names": cleaned_cols,
        }

    # Deterministic secondary sort key (ascending Item Number)
    tie_keys = [_item_tie_break_key(v) for v in work_df["Item Number"]]
    work_df["_tie_key"] = tie_keys

    # ------------------------------------------------------------------
    # Step 2A: Independent Sales ($) Ranking & Cumulative Classification
    # ------------------------------------------------------------------
    sales_ranked = (
        work_df[["Item Number", "Sales ($)", "_tie_key"]]
        .sort_values(by=["Sales ($)", "_tie_key"], ascending=[False, True])
        .reset_index(drop=True)
    )
    sales_ranked["Cumulative Sales"] = sales_ranked["Sales ($)"].cumsum()
    sales_ranked["Cumulative Sales %"] = (sales_ranked["Cumulative Sales"] / total_sales) * 100.0
    sales_ranked["Sales Class"] = sales_ranked["Cumulative Sales %"].apply(classify_cumulative_pct)
    sales_ranked["Sales Score"] = sales_ranked["Sales Class"].apply(class_to_score)

    sales_map_df = sales_ranked[
        ["Item Number", "Cumulative Sales", "Cumulative Sales %", "Sales Class", "Sales Score"]
    ]

    # ------------------------------------------------------------------
    # Step 2B: Independent Hits Ranking & Cumulative Classification
    # ------------------------------------------------------------------
    hits_ranked = (
        work_df[["Item Number", "Hits", "_tie_key"]]
        .sort_values(by=["Hits", "_tie_key"], ascending=[False, True])
        .reset_index(drop=True)
    )
    hits_ranked["Cumulative Hits"] = hits_ranked["Hits"].cumsum()
    hits_ranked["Cumulative Hits %"] = (hits_ranked["Cumulative Hits"] / total_hits) * 100.0
    hits_ranked["Hits Class"] = hits_ranked["Cumulative Hits %"].apply(classify_cumulative_pct)
    hits_ranked["Hits Score"] = hits_ranked["Hits Class"].apply(class_to_score)

    hits_map_df = hits_ranked[
        ["Item Number", "Cumulative Hits", "Cumulative Hits %", "Hits Class", "Hits Score"]
    ]

    # ------------------------------------------------------------------
    # Step 3 & 4: Map Back to Original Items, Compute Weighted Score & Final Class
    # ------------------------------------------------------------------
    base_df = work_df[REQUIRED_ABCD_CSV_COLUMNS].copy()
    merged_df = base_df.merge(sales_map_df, on="Item Number", how="left")
    merged_df = merged_df.merge(hits_map_df, on="Item Number", how="left")

    merged_df["Combined Weighted Score"] = [
        compute_weighted_score(int(s), int(h))
        for s, h in zip(merged_df["Sales Score"], merged_df["Hits Score"])
    ]
    merged_df["Final ABCD Class"] = merged_df["Combined Weighted Score"].apply(classify_weighted_score)

    classified_df = merged_df[ABCD_OUTPUT_COLUMNS].copy()

    # ------------------------------------------------------------------
    # Step 5: Required Summary Table by Final Class (A, B, C, D)
    # ------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    class_col_vals = list(classified_df["Final ABCD Class"])
    sales_col_vals = list(classified_df["Sales ($)"])
    hits_col_vals = list(classified_df["Hits"])

    for cls in ABCD_CLASS_ORDER:
        cls_indices = [i for i, c in enumerate(class_col_vals) if c == cls]
        cls_count = len(cls_indices)
        cls_sales = sum(float(sales_col_vals[i]) for i in cls_indices)
        cls_hits = sum(float(hits_col_vals[i]) for i in cls_indices)
        policy = ABCD_INVENTORY_POLICIES[cls]

        summary_rows.append(
            {
                "Final ABCD Class": cls,
                "Item Count": cls_count,
                "% of Items": (cls_count / num_rows) * 100.0,
                "Total Sales ($)": cls_sales,
                "% of Sales": (cls_sales / total_sales) * 100.0,
                "Total Hits": cls_hits,
                "% of Hits": (cls_hits / total_hits) * 100.0,
                "Cycle Counting Policy": policy["cycle_counting"],
                "Target Cycle Service Level": policy["cycle_service_level"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Step 6: Data Reconciliation & Integrity Checks
    # ------------------------------------------------------------------
    sum_class_items = int(sum(r["Item Count"] for r in summary_rows))
    sum_class_sales = float(sum(r["Total Sales ($)"] for r in summary_rows))
    sum_class_hits = float(sum(r["Total Hits"] for r in summary_rows))

    every_item_classified = (
        len(class_col_vals) == num_rows
        and all(c in ABCD_CLASS_ORDER for c in class_col_vals)
    )
    items_reconcile = sum_class_items == num_rows
    sales_reconcile = math.isclose(sum_class_sales, total_sales, rel_tol=1e-9, abs_tol=1e-6)
    hits_reconcile = math.isclose(sum_class_hits, total_hits, rel_tol=1e-9, abs_tol=1e-6)

    matches_project_benchmark = (
        num_rows == PROJECT_BENCHMARK_ITEMS
        and math.isclose(total_sales, PROJECT_BENCHMARK_SALES, rel_tol=1e-9, abs_tol=1e-6)
        and math.isclose(total_hits, PROJECT_BENCHMARK_HITS, rel_tol=1e-9, abs_tol=1e-6)
    )

    reconciliation: Dict[str, Any] = {
        "all_internal_checks_passed": (
            every_item_classified and items_reconcile and sales_reconcile and hits_reconcile
        ),
        "every_item_single_class": every_item_classified,
        "items_reconcile": items_reconcile,
        "sales_reconcile": sales_reconcile,
        "hits_reconcile": hits_reconcile,
        "total_items": num_rows,
        "sum_class_items": sum_class_items,
        "total_sales": total_sales,
        "sum_class_sales": sum_class_sales,
        "total_hits": total_hits,
        "sum_class_hits": sum_class_hits,
        "matches_project_benchmark": matches_project_benchmark,
        "benchmark_items": PROJECT_BENCHMARK_ITEMS,
        "benchmark_sales": PROJECT_BENCHMARK_SALES,
        "benchmark_hits": PROJECT_BENCHMARK_HITS,
    }

    return {
        "valid": True,
        "missing_columns": [],
        "error_message": None,
        "num_rows": num_rows,
        "num_columns": num_cols,
        "column_names": cleaned_cols,
        "total_items": num_rows,
        "total_sales": total_sales,
        "total_hits": total_hits,
        "classified_df": classified_df,
        "summary_df": summary_df,
        "summary_rows": summary_rows,
        "reconciliation": reconciliation,
        "policies": ABCD_INVENTORY_POLICIES,
        "safety_stock_disclaimer": SAFETY_STOCK_DISCLAIMER,
        "methodology_notes": METHODOLOGY_NOTES,
    }
