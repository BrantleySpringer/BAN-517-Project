"""Application interface for the Supply Chain Analytics Inventory Decision Agent.

Supports two execution modes:
1. Web UI (Streamlit): `streamlit run app.py`
2. CLI / Demo / Interactive REPL (zero external dependencies):
   - `python3 app.py --demo`
   - `python3 app.py --question "Your inventory question..."`
   - `python3 app.py` (interactive prompt)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List

from abcd_classification import (
    ABCD_CLASS_ORDER,
    ABCD_INVENTORY_POLICIES,
    REQUIRED_ABCD_CSV_COLUMNS,
    classify_abcd_dataframe,
    detect_csv_schema,
)
from inventory_agent import AgentResponse, InventoryDecisionAgent
from inventory_tools import TOOL_REGISTRY


SAMPLE_QUESTIONS: Dict[str, str] = {
    "1. Economic Order Quantity (EOQ)": (
        "Our annual demand is 12,000 units. Each order costs $150 to place, and "
        "holding cost is $4 per unit per year. What is the economic order quantity "
        "and total annual relevant inventory cost?"
    ),
    "2. Quantity Discount": (
        "Annual demand is 10,000 units, ordering cost is $80 per order, and annual "
        "holding cost is 20% of unit price. A supplier offers price breaks: "
        "1-499 units at $12.00, 500-999 units at $11.50, and 1000+ units at $11.00. "
        "What order quantity minimizes total annual cost?"
    ),
    "3. Safety Stock & Reorder Point": (
        "Daily demand averages 120 units with a standard deviation of 20 units. "
        "Lead time is 4 days. We want a 95% cycle service level. "
        "What should our safety stock and reorder point be?"
    ),
    "4. Fixed-Period Review": (
        "We review inventory every 14 days and lead time is 3 days. Daily demand "
        "averages 50 units with a standard deviation of 8 units. Target service level "
        "is 95%. Currently we have 220 units on hand and 50 units on order. "
        "What is the order-up-to target level and how many units should we order?"
    ),
    "5. Newsvendor": (
        "A bakery sells fresh loaves for $6.00 that cost $2.40 to bake. Unsold loaves "
        "at the end of the day are donated for $0.50 salvage value. Daily demand is "
        "normally distributed with mean 180 and standard deviation 30. "
        "How many loaves should be baked to maximize expected profit?"
    ),
}


# ---------------------------------------------------------------------------
# CLI Formatting & Execution
# ---------------------------------------------------------------------------

def format_cli_response(resp: AgentResponse) -> str:
    """Format the 7-stage agent response for readable terminal output."""
    sep = "=" * 78
    lines: List[str] = [
        sep,
        f"STATUS: {resp.status.upper()}",
        f"QUESTION: {resp.question}",
        "-" * 78,
        "PIPELINE TRACE:",
    ]
    for step in resp.pipeline_trace:
        lines.append(f"  [{step['status'].upper():>6}] {step['stage']}: {step['summary']}")
    lines.extend(["-" * 78, resp.explanation, sep])
    return "\n".join(lines)


def run_cli(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent (CLI)"
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help="Natural-language inventory question to solve.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in sample questions for all 5 inventory models.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "rule", "llm"],
        default="auto",
        help="NLU parser mode (numerical calculation always runs in Python).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output structured JSON instead of formatted text.",
    )
    args = parser.parse_args(argv)
    agent = InventoryDecisionAgent(parser_mode=args.mode)

    if args.demo:
        outputs = []
        for label, question in SAMPLE_QUESTIONS.items():
            resp = agent.run(question)
            if args.as_json:
                outputs.append({"sample": label, "response": resp.to_dict()})
            else:
                print(f"\n>>> DEMO: {label}")
                print(format_cli_response(resp))
        if args.as_json:
            print(json.dumps(outputs, indent=2))
        return 0

    if args.question:
        resp = agent.run(args.question)
        if args.as_json:
            print(json.dumps(resp.to_dict(), indent=2))
        else:
            print(format_cli_response(resp))
        return 0 if resp.status == "completed" else 1

    # Interactive REPL
    print("Supply Chain Analytics Inventory Decision Agent")
    print("Architecture: Question -> Identify -> Extract -> Select Tool -> Calculate -> Verify -> Explain")
    print("Type 'demo' to run all 5 sample scenarios, or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("Inventory Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if user_input.lower() == "demo":
            for label, q in SAMPLE_QUESTIONS.items():
                print(f"\n>>> {label}")
                print(format_cli_response(agent.run(q)))
            continue
        resp = agent.run(user_input)
        print(format_cli_response(resp))
    return 0


# ---------------------------------------------------------------------------
# Streamlit Web Interface
# ---------------------------------------------------------------------------

def _escape_streamlit_markdown(text: str) -> str:
    """Escape literal dollar signs and math star-variables outside code spans for Streamlit Markdown.

    Streamlit interprets unescaped `$...$` in Markdown strings as KaTeX LaTeX math and
    paired `*` characters (such as `Q*`, `TC(Q*)`, or multiplication `*`) as italics.
    Leaving fenced and inline code spans untouched, this escapes:
      - Unescaped `$` -> `\\$`
      - Single-letter math star variables like `Q*`, `z*`, `TC(Q*)` -> `Q\\*`, `z\\*`, `TC(Q\\*)`
      - Standalone multiplication operators ` * ` -> ` \\* `
    """
    if not text:
        return text
    parts = re.split(r"(```[\s\S]*?```|`[^`]*`)", str(text))
    escaped: List[str] = []
    for idx, seg in enumerate(parts):
        if idx % 2 == 1:
            escaped.append(seg)
        else:
            seg = re.sub(r"(?<!\\)\$", r"\\$", seg)
            seg = re.sub(r"(?<![A-Za-z0-9*])([A-Za-z])\*(?!\*)", r"\1\\*", seg)
            seg = re.sub(r"(?<=\s)\*(?=\s)", r"\\*", seg)
            escaped.append(seg)
    return "".join(escaped)


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        try:
            import streamlit.runtime as rt

            return bool(rt.exists())
        except Exception:
            return False


def _render_key_metrics(st: Any, model_id: str, calc: Dict[str, Any]) -> None:
    if model_id == "eoq":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Order Qty (Q*)", f"{calc['optimal_order_quantity']:,.2f}")
        c2.metric("Orders / Year", f"{calc['orders_per_year']:,.2f}")
        if calc.get("cycle_time_days") is not None:
            c3.metric("Cycle Time (Days)", f"{calc['cycle_time_days']:,.1f}")
        else:
            c3.metric("Cycle Time (Years)", f"{calc['cycle_time_years']:,.3f}")
        c4.metric("Total Relevant Cost", f"${calc['total_relevant_cost']:,.2f}")
    elif model_id == "quantity_discount":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Order Qty (Q*)", f"{calc['optimal_order_quantity']:,.2f}")
        c2.metric("Optimal Tier", f"Tier {calc['optimal_tier']}")
        c3.metric("Unit Price", f"${calc['optimal_unit_price']:,.2f}")
        c4.metric("Total Annual Cost", f"${calc['total_annual_cost']:,.2f}")
    elif model_id == "safety_stock_rop":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Safety Stock (SS)", f"{calc['safety_stock']:,.2f}")
        c2.metric("Reorder Point (ROP)", f"{calc['reorder_point']:,.2f}")
        c3.metric("Lead-Time Mean", f"{calc['expected_lead_time_demand']:,.2f}")
        c4.metric("Service Level (CSL)", f"{calc['cycle_service_level'] * 100:.2f}%")
    elif model_id == "fixed_period_review":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target Level (S)", f"{calc['target_inventory_level']:,.2f}")
        c2.metric("Safety Stock (SS)", f"{calc['safety_stock']:,.2f}")
        c3.metric("Protection Interval", f"{calc['protection_interval']:,.2f} {calc['time_unit']}(s)")
        order_q = (
            f"{calc['order_quantity']:,.2f}"
            if calc["order_quantity"] is not None
            else "N/A"
        )
        c4.metric("Order Quantity Now (Q)", order_q)
    elif model_id == "newsvendor":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Order Qty (Q*)", f"{calc['optimal_order_quantity']:,.2f}")
        c2.metric("Critical Ratio", f"{calc['critical_ratio']:.4f}")
        c3.metric("Expected Sales", f"{calc['expected_sales']:,.2f}")
        if calc["expected_profit"] is not None:
            c4.metric("Expected Profit", f"${calc['expected_profit']:,.2f}")
        else:
            c4.metric("Mismatch Cost", f"${calc['expected_mismatch_cost']:,.2f}")


REQUIRED_INVENTORY_CSV_COLUMNS: List[str] = ["Product", "Category", "Units", "Unit Cost"]


def analyze_inventory_dataframe(df: Any) -> Dict[str, Any]:
    """Validate and analyze an uploaded inventory DataFrame using pandas.

    Requirements:
    1. Validate required columns: Product, Category, Units, Unit Cost.
    2. If missing, return diagnostic dictionary identifying missing columns.
    3. Calculate total Units.
    4. Calculate average Unit Cost.
    5. Identify the Product with the highest Units.
    6. Group by Category and calculate total Units for each Category.
    7. Create new calculated column: Inventory Value = Units * Unit Cost.
    8. Calculate total Inventory Value across all products.
    """
    col_list = list(df.columns)
    missing = [c for c in REQUIRED_INVENTORY_CSV_COLUMNS if c not in col_list]
    num_rows = int(len(df))
    num_cols = int(len(col_list))

    if missing:
        return {
            "valid": False,
            "missing_columns": missing,
            "error_message": (
                f"Missing required column(s): {', '.join(missing)}. "
                f"Required columns are: {', '.join(REQUIRED_INVENTORY_CSV_COLUMNS)}."
            ),
            "num_rows": num_rows,
            "num_columns": num_cols,
            "column_names": col_list,
        }

    import pandas as pd

    analyzed_df = df.copy()
    analyzed_df["Units"] = pd.to_numeric(analyzed_df["Units"], errors="coerce")
    analyzed_df["Unit Cost"] = pd.to_numeric(analyzed_df["Unit Cost"], errors="coerce")

    # 1. Total Units
    total_units = float(analyzed_df["Units"].sum())

    # 2. Average Unit Cost
    avg_unit_cost = float(analyzed_df["Unit Cost"].mean()) if num_rows > 0 else 0.0

    # 3. Product with highest Units
    if num_rows > 0 and not analyzed_df["Units"].dropna().empty:
        max_idx = analyzed_df["Units"].idxmax()
        highest_product = str(analyzed_df.loc[max_idx, "Product"])
        highest_units = float(analyzed_df.loc[max_idx, "Units"])
    else:
        highest_product = "N/A"
        highest_units = 0.0

    # 4. Group by Category and calculate total Units
    units_by_category = (
        analyzed_df.groupby("Category", as_index=False)["Units"]
        .sum()
        .sort_values(by="Units", ascending=False)
        .reset_index(drop=True)
    )

    # 5. Create new calculated column: Inventory Value = Units * Unit Cost
    analyzed_df["Inventory Value"] = analyzed_df["Units"] * analyzed_df["Unit Cost"]

    # 6. Total Inventory Value across all products
    total_inventory_value = float(analyzed_df["Inventory Value"].sum())

    return {
        "valid": True,
        "missing_columns": [],
        "error_message": None,
        "num_rows": num_rows,
        "num_columns": num_cols,
        "column_names": col_list,
        "total_units": total_units,
        "average_unit_cost": avg_unit_cost,
        "highest_units_product": highest_product,
        "highest_units_count": highest_units,
        "units_by_category": units_by_category,
        "total_inventory_value": total_inventory_value,
        "dataframe": analyzed_df,
    }


def _render_standard_inventory_csv_section(st: Any, df: Any) -> None:
    """Validate, analyze, and display the standard 4-column inventory CSV dataset."""
    import pandas as pd

    st.header("Inventory Data Analysis")

    analysis = analyze_inventory_dataframe(df)

    if not analysis["valid"]:
        st.error(analysis["error_message"])
        st.markdown(
            f"**Dataset Dimensions:** {analysis['num_rows']} rows × {analysis['num_columns']} columns  \n"
            f"**Detected Columns:** `{', '.join(analysis['column_names'])}`  \n"
            f"**Required Columns:** `{', '.join(REQUIRED_INVENTORY_CSV_COLUMNS)}`"
        )
        st.subheader("Uploaded CSV Preview")
        st.dataframe(df, use_container_width=True)
        return

    # 1. Dataset Dimensions & Column Names
    st.markdown("#### Dataset Metadata")
    m1, m2, m3 = st.columns(3)
    m1.metric("Number of Rows", f"{analysis['num_rows']:,}")
    m2.metric("Number of Columns", f"{analysis['num_columns']:,}")
    m3.write(f"**Column Names:**\n\n`{', '.join(analysis['column_names'])}`")

    # 2. Executive Key Inventory Metrics / Results
    st.markdown("#### Key Inventory Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Units", f"{analysis['total_units']:,.0f}")
    k2.metric("Average Unit Cost", f"${analysis['average_unit_cost']:,.2f}")
    k3.metric(
        "Product with Highest Units",
        f"{analysis['highest_units_product']}",
        f"{analysis['highest_units_count']:,.0f} units",
    )
    k4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

    # 3. Total Units by Category
    st.markdown("#### Total Units by Category")
    st.dataframe(
        analysis["units_by_category"],
        use_container_width=True,
        hide_index=True,
    )

    # 4. Display Complete Dataframe with Calculated Column & Currency Formatting
    st.markdown("#### Complete Inventory Dataset (with Calculated Inventory Value)")
    df_display = analysis["dataframe"].copy()
    try:
        st.dataframe(
            df_display,
            column_config={
                "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
                "Units": st.column_config.NumberColumn("Units", format="%d"),
            },
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        formatted_df = df_display.copy()
        if "Unit Cost" in formatted_df.columns:
            formatted_df["Unit Cost"] = formatted_df["Unit Cost"].apply(
                lambda v: f"${v:,.2f}" if pd.notnull(v) else ""
            )
        if "Inventory Value" in formatted_df.columns:
            formatted_df["Inventory Value"] = formatted_df["Inventory Value"].apply(
                lambda v: f"${v:,.2f}" if pd.notnull(v) else ""
            )
        if "Units" in formatted_df.columns:
            formatted_df["Units"] = formatted_df["Units"].apply(
                lambda v: f"{v:,.0f}" if pd.notnull(v) else ""
            )
        st.dataframe(formatted_df, use_container_width=True)


def _render_abcd_classification_section(st: Any, df: Any) -> None:
    """Validate, classify, reconcile, and display BAN 517 Multicriteria ABCD Classification."""
    st.header("Multicriteria ABCD Inventory Classification")
    st.caption(
        _escape_streamlit_markdown(
            "Official BAN 517 Methodology: Independent Pareto classification for **Sales ($)** (65% weight) "
            "and **Hits** (35% weight), mapped back to each original item by **Item Number**."
        )
    )

    result = classify_abcd_dataframe(df)
    if not result["valid"]:
        st.error(_escape_streamlit_markdown(result["error_message"]))
        st.markdown(
            f"**Dataset Dimensions:** {result['num_rows']} rows × {result['num_columns']} columns  \n"
            f"**Detected Columns:** `{', '.join(result['column_names'])}`  \n"
            f"**Required Columns:** `{', '.join(REQUIRED_ABCD_CSV_COLUMNS)}`"
        )
        st.subheader("Uploaded CSV Preview")
        st.dataframe(df, use_container_width=True)
        return

    recon = result["reconciliation"]

    # 1. Dataset Overview & Data Reconciliation Checks
    st.markdown("#### Dataset Totals & Reconciliation")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Inventory Items", f"{result['total_items']:,}")
    k2.metric("Total Sales ($)", f"${result['total_sales']:,.2f}")
    k3.metric("Total Hits", f"{result['total_hits']:,.0f}")
    k4.metric(
        "Reconciliation Status",
        "PASSED" if recon["all_internal_checks_passed"] else "CHECK FAILED",
    )

    if recon["all_internal_checks_passed"]:
        benchmark_note = (
            " Matches official BAN 517 project benchmark (205 items, $82,046,150 Sales, 23,252 Hits)."
            if recon["matches_project_benchmark"]
            else ""
        )
        reconciliation_msg = (
            f"All reconciliation checks passed: Class item counts sum to **{recon['sum_class_items']:,}**, "
            f"class Sales sum to **${recon['sum_class_sales']:,.2f}**, class Hits sum to "
            f"**{recon['sum_class_hits']:,.0f}**, and every item received exactly one final ABCD class."
            f"{benchmark_note}"
        )
        st.success(_escape_streamlit_markdown(reconciliation_msg))
    else:
        st.error("One or more data reconciliation checks failed. Please inspect the input dataset.")

    # 2. Required Classification Summary (A, B, C, D)
    st.markdown("#### Classification Summary by Final ABCD Class")
    summary_df = result["summary_df"]
    try:
        st.dataframe(
            summary_df,
            column_config={
                "Item Count": st.column_config.NumberColumn("Item Count", format="%d"),
                "% of Items": st.column_config.NumberColumn("% of Items", format="%.2f%%"),
                "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
                "% of Sales": st.column_config.NumberColumn("% of Sales", format="%.2f%%"),
                "Total Hits": st.column_config.NumberColumn("Total Hits", format="%d"),
                "% of Hits": st.column_config.NumberColumn("% of Hits", format="%.2f%%"),
            },
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(summary_df, use_container_width=True)

    # 3. Course Inventory Policy Recommendations
    st.markdown("#### Inventory Policy Recommendations (BAN 517 Module)")
    policy_cols = st.columns(4)
    for col, cls in zip(policy_cols, ABCD_CLASS_ORDER):
        policy = ABCD_INVENTORY_POLICIES[cls]
        col.markdown(
            f"**Class {cls}**  \n"
            f"- **Cycle Counting:** {policy['cycle_counting']}  \n"
            f"- **Cycle Service Level:** {policy['cycle_service_level']}  \n"
            f"- **Action:** {policy['action_summary']}"
        )
    st.info(_escape_streamlit_markdown(result["safety_stock_disclaimer"]))

    # 4. Detailed 15-Column Sortable Classification Table + CSV Export
    st.markdown("#### Detailed Item-Level ABCD Classification Table")
    classified_df = result["classified_df"]

    detail_col_config: Dict[str, Any] = {
        "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
        "Hits": st.column_config.NumberColumn("Hits", format="%d"),
        "Cumulative Sales": st.column_config.NumberColumn("Cumulative Sales", format="$%.2f"),
        "Cumulative Sales %": st.column_config.NumberColumn("Cumulative Sales %", format="%.4f%%"),
        "Sales Score": st.column_config.NumberColumn("Sales Score", format="%d"),
        "Cumulative Hits": st.column_config.NumberColumn("Cumulative Hits", format="%d"),
        "Cumulative Hits %": st.column_config.NumberColumn("Cumulative Hits %", format="%.4f%%"),
        "Hits Score": st.column_config.NumberColumn("Hits Score", format="%d"),
        "Combined Weighted Score": st.column_config.NumberColumn(
            "Combined Weighted Score", format="%.2f"
        ),
    }
    if all(
        isinstance(v, (int, float)) or str(v).strip().isdigit()
        for v in classified_df["Item Number"]
    ):
        detail_col_config["Item Number"] = st.column_config.NumberColumn("Item Number", format="%d")

    try:
        st.dataframe(
            classified_df,
            column_config=detail_col_config,
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(classified_df, use_container_width=True)

    csv_bytes = classified_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ABCD Classification Results (CSV)",
        data=csv_bytes,
        file_name="abcd_classification_results.csv",
        mime="text/csv",
    )

    # 5. Documented Methodology & Assumptions
    with st.expander("Methodology, Continuous Thresholds & Tie-Breaking Assumptions"):
        for note in result["methodology_notes"]:
            st.markdown(_escape_streamlit_markdown(f"- {note}"))


def _render_csv_analysis_section(st: Any, uploaded_file: Any) -> None:
    """Read an uploaded CSV file and route it to the appropriate inventory analysis workflow."""
    import pandas as pd

    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception as exc:
        st.error(f"Error reading CSV file: {exc}")
        return

    schema = detect_csv_schema(df.columns)
    if schema == "abcd_project":
        _render_abcd_classification_section(st, df)
    elif schema == "standard_inventory":
        _render_standard_inventory_csv_section(st, df)
    else:
        st.header("Inventory Data Analysis")
        col_list = [str(c).lstrip("\ufeff").strip() for c in df.columns]
        missing_std = [c for c in REQUIRED_INVENTORY_CSV_COLUMNS if c not in col_list]
        missing_abcd = [c for c in REQUIRED_ABCD_CSV_COLUMNS if c not in col_list]
        st.error(
            "Uploaded CSV does not match either supported inventory dataset format.\n\n"
            f"- For **Multicriteria ABCD Classification**, missing column(s): `{', '.join(missing_abcd)}` "
            f"(required: `{', '.join(REQUIRED_ABCD_CSV_COLUMNS)}`).\n"
            f"- For **Standard Inventory Data Analysis**, missing column(s): `{', '.join(missing_std)}` "
            f"(required: `{', '.join(REQUIRED_INVENTORY_CSV_COLUMNS)}`)."
        )
        st.markdown(
            f"**Dataset Dimensions:** {len(df)} rows × {len(col_list)} columns  \n"
            f"**Detected Columns:** `{', '.join(col_list)}`"
        )
        st.subheader("Uploaded CSV Preview")
        st.dataframe(df, use_container_width=True)


def render_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("Supply Chain Analytics Inventory Decision Agent")
    st.caption(
        "Architecture: **Customer Question → Identify Model → Extract Inputs → "
        "Select Python Tool → Calculate → Verify → Explain**"
    )

    with st.sidebar:
        st.header("Configuration")
        parser_mode = st.selectbox(
            "NLU Parser Mode",
            options=["auto", "rule", "llm"],
            index=0,
            help=(
                "'auto' uses an LLM if OPENAI_API_KEY / GEMINI_API_KEY is set and falls back "
                "to the deterministic parser. All calculations always run in Python."
            ),
        )
        st.divider()
        st.header("Sample Scenarios")
        selected_sample = st.selectbox(
            "Load an example question:",
            options=["Custom Question"] + list(SAMPLE_QUESTIONS.keys()),
        )
        st.divider()
        st.subheader("Supported Models")
        for meta in TOOL_REGISTRY.values():
            st.markdown(f"- **{meta['name']}**: {meta['description']}")

        # DATA UPLOAD: Last section in the left sidebar (single uploader for all CSV formats)
        st.divider()
        st.header("Data Upload")
        uploaded_csv = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help=(
                "Upload either the BAN 517 ABCD project CSV (Item Number, Item Name, UoM, Sales ($), Hits) "
                "or a standard inventory CSV (Product, Category, Units, Unit Cost)."
            ),
        )

    # If CSV is uploaded, render data processing and analysis at top of main page
    if uploaded_csv is not None:
        _render_csv_analysis_section(st, uploaded_csv)
        st.divider()

    default_q = (
        SAMPLE_QUESTIONS[selected_sample]
        if selected_sample != "Custom Question"
        else SAMPLE_QUESTIONS["1. Economic Order Quantity (EOQ)"]
    )

    question = st.text_area(
        "1. Customer Question",
        value=default_q,
        height=120,
        placeholder="Ask any EOQ, Quantity Discount, Safety Stock/ROP, Fixed-Period Review, or Newsvendor question...",
    )

    if st.button("Analyze & Solve", type="primary", use_container_width=True):
        agent = InventoryDecisionAgent(parser_mode=parser_mode)
        resp = agent.run(question)

        # Pipeline Trace Overview
        st.subheader("Pipeline Execution Trace")
        cols = st.columns(len(resp.pipeline_trace))
        for col, step in zip(cols, resp.pipeline_trace):
            icon = "✅" if step["status"] == "ok" else "⚠️"
            col.markdown(
                _escape_streamlit_markdown(f"**{icon} {step['stage']}**\n\n{step['summary']}")
            )

        st.divider()

        if resp.status != "completed":
            st.warning(_escape_streamlit_markdown(resp.explanation))
            if resp.identification.scores:
                with st.expander("Model Scoring Breakdown"):
                    st.json(resp.identification.scores)
            return

        assert resp.calculation is not None
        assert resp.verification is not None
        assert resp.extraction is not None
        assert resp.tool_selection is not None

        # Key KPIs
        _render_key_metrics(st, resp.identification.model_id or "", resp.calculation)

        tab_explain, tab_stages, tab_verify, tab_raw = st.tabs(
            [
                "Explanation & Recommendation",
                "Model, Inputs & Tool Details",
                "Verification Audit",
                "Full JSON Trace",
            ]
        )

        with tab_explain:
            st.markdown(_escape_streamlit_markdown(resp.explanation))

        with tab_stages:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 2. Identify Model")
                st.write(
                    f"**Model:** {resp.identification.model_name} "
                    f"(`{resp.identification.model_id}`)"
                )
                st.write(f"**Confidence:** {resp.identification.confidence:.2f}")
                st.write(f"**Method:** `{resp.identification.method}`")
                st.write(_escape_streamlit_markdown(resp.identification.reasoning))

                st.markdown("#### 4. Select Python Tool")
                st.write(f"**Tool Function:** `inventory_tools.{resp.tool_selection.tool_name}`")
                st.json(resp.tool_selection.arguments)

            with col_b:
                st.markdown("#### 3. Extract Inputs")
                st.json(resp.extraction.params)
                if resp.extraction.unit_conversions:
                    st.markdown("**Unit Conversions:**")
                    for c in resp.extraction.unit_conversions:
                        st.markdown(_escape_streamlit_markdown(f"- {c}"))
                if resp.extraction.assumptions:
                    st.markdown("**Assumptions:**")
                    for a in resp.extraction.assumptions:
                        st.markdown(_escape_streamlit_markdown(f"- {a}"))

            st.markdown("#### 5. Python Calculation Output")
            if resp.identification.model_id == "quantity_discount":
                st.dataframe(resp.calculation["tier_evaluations"], use_container_width=True)
            st.json(resp.calculation)

        with tab_verify:
            if resp.verification.passed:
                st.success("All analytical and domain verification checks passed.")
            else:
                st.error("One or more verification checks failed.")
            for chk in resp.verification.checks:
                badge = "✅ PASS" if chk["passed"] else "❌ FAIL"
                check_name = _escape_streamlit_markdown(str(chk["name"]))
                st.markdown(f"- **{badge}** — **{check_name}**: `{chk['detail']}`")

        with tab_raw:
            st.json(resp.to_dict())


if __name__ == "__main__":
    if _is_streamlit_runtime():
        render_streamlit_app()
    else:
        sys.exit(run_cli())
