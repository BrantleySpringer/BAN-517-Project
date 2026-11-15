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
import sys
from typing import Any, Dict, List

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
            col.markdown(f"**{icon} {step['stage']}**\n\n{step['summary']}")

        st.divider()

        if resp.status != "completed":
            st.warning(resp.explanation)
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
                "7. Explanation & Recommendation",
                "2–5. Model, Inputs & Tool Details",
                "6. Verification Audit",
                "Full JSON Trace",
            ]
        )

        with tab_explain:
            st.markdown(resp.explanation)

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
                st.write(resp.identification.reasoning)

                st.markdown("#### 4. Select Python Tool")
                st.write(f"**Tool Function:** `inventory_tools.{resp.tool_selection.tool_name}`")
                st.json(resp.tool_selection.arguments)

            with col_b:
                st.markdown("#### 3. Extract Inputs")
                st.json(resp.extraction.params)
                if resp.extraction.unit_conversions:
                    st.markdown("**Unit Conversions:**")
                    for c in resp.extraction.unit_conversions:
                        st.markdown(f"- {c}")
                if resp.extraction.assumptions:
                    st.markdown("**Assumptions:**")
                    for a in resp.extraction.assumptions:
                        st.markdown(f"- {a}")

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
                st.markdown(f"- **{badge}** — **{chk['name']}**: {chk['detail']}")

        with tab_raw:
            st.json(resp.to_dict())


if __name__ == "__main__":
    if _is_streamlit_runtime():
        render_streamlit_app()
    else:
        sys.exit(run_cli())
