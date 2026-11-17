"""Unit and integration tests for inventory_tools.py and inventory_agent.py.

Uses Python standard library unittest so it runs cleanly with both
`python3 -m unittest` and `pytest`.
"""

import math
from typing import Any, Dict
import unittest

from inventory_agent import InventoryDecisionAgent
from inventory_tools import (
    calculate_eoq,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_rop,
    normal_cdf,
    normal_loss,
    normal_ppf,
    verify_calculation,
)


class TestNormalHelpers(unittest.TestCase):
    def test_normal_ppf_round_trip(self) -> None:
        for p in [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.975, 0.99, 0.999]:
            z = normal_ppf(p)
            self.assertAlmostEqual(normal_cdf(z), p, places=12)

    def test_known_z_values(self) -> None:
        self.assertAlmostEqual(normal_ppf(0.5), 0.0, places=12)
        self.assertAlmostEqual(normal_ppf(0.95), 1.6448536269514722, places=10)
        self.assertAlmostEqual(normal_ppf(0.975), 1.959963984540054, places=10)

    def test_normal_loss_at_zero(self) -> None:
        self.assertAlmostEqual(normal_loss(0.0), 1.0 / math.sqrt(2.0 * math.pi), places=12)


class TestInventoryToolsMath(unittest.TestCase):
    def _assert_structured_schema(self, res: dict, expected_tool: str) -> None:
        for key in ("model", "tool_name", "inputs", "intermediate_calculations", "final_answer", "assumptions"):
            self.assertIn(key, res)
        self.assertEqual(res["tool_name"], expected_tool)
        self.assertIsInstance(res["inputs"], dict)
        self.assertIsInstance(res["intermediate_calculations"], dict)
        self.assertIsInstance(res["final_answer"], dict)
        self.assertIsInstance(res["assumptions"], list)
        self.assertGreater(len(res["assumptions"]), 0)

    def test_eoq_basic_and_no_invented_days(self) -> None:
        res = calculate_eoq(annual_demand=12000, ordering_cost=150, holding_cost=4)
        self._assert_structured_schema(res, "calculate_eoq")
        expected_q = math.sqrt((2.0 * 12000.0 * 150.0) / 4.0)
        self.assertAlmostEqual(res["final_answer"]["optimal_order_quantity"], expected_q, places=8)
        self.assertAlmostEqual(
            res["intermediate_calculations"]["annual_ordering_cost"],
            res["intermediate_calculations"]["annual_holding_cost"],
            places=8,
        )
        self.assertIsNone(res["cycle_time_days"])
        ver = verify_calculation(
            "eoq", {"annual_demand": 12000, "ordering_cost": 150, "holding_cost": 4}, res
        )
        self.assertTrue(ver["passed"])

    def test_eoq_holding_rate_and_unit_cost(self) -> None:
        res = calculate_eoq(
            annual_demand=5000,
            ordering_cost=100,
            holding_rate=0.20,
            unit_cost=25,
            working_days_per_year=250,
        )
        self._assert_structured_schema(res, "calculate_eoq")
        self.assertAlmostEqual(res["effective_holding_cost"], 5.0, places=8)
        self.assertAlmostEqual(res["optimal_order_quantity"], math.sqrt(200000.0), places=8)
        self.assertAlmostEqual(res["annual_purchase_cost"], 125000.0, places=8)
        self.assertAlmostEqual(
            res["cycle_time_days"], (res["optimal_order_quantity"] / 5000.0) * 250.0, places=8
        )

    def test_quantity_discount_percentage_holding_cost(self) -> None:
        tiers = [
            {"min_qty": 1, "max_qty": 499, "unit_price": 12.0},
            {"min_qty": 500, "max_qty": 999, "unit_price": 11.5},
            {"min_qty": 1000, "max_qty": None, "unit_price": 11.0},
        ]
        inputs = {
            "annual_demand": 10000,
            "ordering_cost": 80,
            "holding_rate": 0.20,
            "tiers": tiers,
        }
        res = calculate_quantity_discount(**inputs)
        self._assert_structured_schema(res, "calculate_quantity_discount")
        self.assertEqual(res["holding_cost_mode"], "percentage_of_tier_price")
        self.assertEqual(res["final_answer"]["optimal_tier"], 3)
        self.assertAlmostEqual(res["final_answer"]["optimal_order_quantity"], 1000.0, places=6)
        self.assertAlmostEqual(res["final_answer"]["total_annual_cost"], 111900.0, places=6)
        self.assertEqual(len(res["intermediate_calculations"]["tier_evaluations"]), 3)
        ver = verify_calculation("quantity_discount", inputs, res)
        self.assertTrue(ver["passed"])

    def test_quantity_discount_fixed_holding_cost(self) -> None:
        tiers = [
            {"min_qty": 1, "max_qty": 499, "unit_price": 20.0},
            {"min_qty": 500, "max_qty": 1999, "unit_price": 18.0},
            {"min_qty": 2000, "max_qty": None, "unit_price": 17.5},
        ]
        inputs = {
            "annual_demand": 4000,
            "ordering_cost": 100,
            "holding_cost": 5.0,
            "tiers": tiers,
        }
        res = calculate_quantity_discount(**inputs)
        self._assert_structured_schema(res, "calculate_quantity_discount")
        self.assertEqual(res["holding_cost_mode"], "fixed_across_tiers")
        # EOQ = sqrt(2 * 4000 * 100 / 5) = 400 -> feasible in Tier 1 (TC=82000)
        # Tier 2 min=500 -> TC = 72000 + 800 + 1250 = 74050
        # Tier 3 min=2000 -> TC = 70000 + 200 + 5000 = 75200
        self.assertEqual(res["optimal_tier"], 2)
        self.assertAlmostEqual(res["optimal_order_quantity"], 500.0, places=6)
        self.assertAlmostEqual(res["total_annual_cost"], 74050.0, places=6)
        ver = verify_calculation("quantity_discount", inputs, res)
        self.assertTrue(ver["passed"])

    def test_quantity_discount_incremental(self) -> None:
        tiers = [
            {"min_qty": 0, "max_qty": 500, "unit_price": 10.0},
            {"min_qty": 500, "max_qty": None, "unit_price": 9.0},
        ]
        res = calculate_quantity_discount(
            annual_demand=4000,
            ordering_cost=100,
            holding_rate=0.20,
            tiers=tiers,
            discount_type="incremental",
        )
        self._assert_structured_schema(res, "calculate_quantity_discount")
        self.assertEqual(res["discount_type"], "incremental")
        self.assertGreater(res["optimal_order_quantity"], 500.0)
        ver = verify_calculation(
            "quantity_discount",
            {"annual_demand": 4000, "ordering_cost": 100, "holding_rate": 0.20, "tiers": tiers},
            res,
        )
        self.assertTrue(ver["passed"])

    def test_rop_constant_demand_constant_lead_time(self) -> None:
        inputs = {
            "demand_mean": 100,
            "demand_std": 0.0,
            "lead_time": 5,
            "lead_time_std": 0.0,
        }
        res = calculate_rop(**inputs)
        self._assert_structured_schema(res, "calculate_rop")
        self.assertEqual(res["formula_case"], "constant_demand_constant_lead_time")
        self.assertAlmostEqual(res["lead_time_demand_std"], 0.0, places=10)
        self.assertAlmostEqual(res["final_answer"]["safety_stock"], 0.0, places=10)
        self.assertAlmostEqual(res["final_answer"]["reorder_point"], 500.0, places=10)
        ver = verify_calculation("safety_stock_rop", inputs, res)
        self.assertTrue(ver["passed"])

    def test_rop_variable_demand_constant_lead_time(self) -> None:
        inputs = {
            "demand_mean": 120,
            "demand_std": 20,
            "lead_time": 4,
            "lead_time_std": 0.0,
            "service_level": 0.95,
        }
        res = calculate_rop(**inputs)
        self._assert_structured_schema(res, "calculate_rop")
        self.assertEqual(res["formula_case"], "variable_demand_constant_lead_time")
        self.assertAlmostEqual(res["expected_lead_time_demand"], 480.0, places=8)
        self.assertAlmostEqual(res["lead_time_demand_std"], 40.0, places=8)
        self.assertAlmostEqual(res["safety_stock"], normal_ppf(0.95) * 40.0, places=6)
        self.assertAlmostEqual(res["reorder_point"], 480.0 + res["safety_stock"], places=8)
        ver = verify_calculation("safety_stock_rop", inputs, res)
        self.assertTrue(ver["passed"])

    def test_rop_constant_demand_variable_lead_time(self) -> None:
        inputs = {
            "demand_mean": 80,
            "demand_std": 0.0,
            "lead_time": 5,
            "lead_time_std": 1.2,
            "service_level": 0.95,
        }
        res = calculate_rop(**inputs)
        self._assert_structured_schema(res, "calculate_rop")
        self.assertEqual(res["formula_case"], "constant_demand_variable_lead_time")
        expected_sigma = 80.0 * 1.2
        expected_ss = normal_ppf(0.95) * expected_sigma
        self.assertAlmostEqual(res["lead_time_demand_std"], expected_sigma, places=8)
        self.assertAlmostEqual(res["safety_stock"], expected_ss, places=6)
        self.assertAlmostEqual(res["reorder_point"], 400.0 + expected_ss, places=6)
        ver = verify_calculation("safety_stock_rop", inputs, res)
        self.assertTrue(ver["passed"])

    def test_rop_variable_demand_variable_lead_time(self) -> None:
        inputs = {
            "demand_mean": 300,
            "demand_std": 45,
            "lead_time": 3,
            "lead_time_std": 0.5,
            "service_level": 0.98,
        }
        res = calculate_rop(**inputs)
        self._assert_structured_schema(res, "calculate_rop")
        self.assertEqual(res["formula_case"], "variable_demand_variable_lead_time")
        expected_sigma = math.sqrt(3.0 * 45.0**2 + 300.0**2 * 0.5**2)
        expected_ss = normal_ppf(0.98) * expected_sigma
        self.assertAlmostEqual(res["lead_time_demand_std"], expected_sigma, places=8)
        self.assertAlmostEqual(res["safety_stock"], expected_ss, places=6)
        self.assertAlmostEqual(res["reorder_point"], 900.0 + expected_ss, places=6)
        ver = verify_calculation("safety_stock_rop", inputs, res)
        self.assertTrue(ver["passed"])

    def test_rop_fill_rate(self) -> None:
        res = calculate_rop(
            demand_mean=100,
            demand_std=25,
            lead_time=4,
            service_level=0.98,
            service_level_type="fill_rate",
            order_quantity=1000,
        )
        implied_fill = 1.0 - (res["lead_time_demand_std"] * normal_loss(res["z_score"])) / 1000.0
        self.assertAlmostEqual(implied_fill, 0.98, places=8)

    def test_periodic_review_with_inventory_position(self) -> None:
        inputs = {
            "review_period": 14,
            "lead_time": 3,
            "demand_mean": 50,
            "demand_std": 8,
            "service_level": 0.95,
            "on_hand_inventory": 220,
            "on_order_inventory": 50,
        }
        res = calculate_periodic_review(**inputs)
        self._assert_structured_schema(res, "calculate_periodic_review")
        expected_sigma = 8.0 * math.sqrt(17.0)
        expected_ss = normal_ppf(0.95) * expected_sigma
        expected_target = 850.0 + expected_ss
        self.assertAlmostEqual(res["final_answer"]["protection_period"], 17.0, places=8)
        self.assertAlmostEqual(res["intermediate_calculations"]["expected_protection_demand"], 850.0, places=8)
        self.assertAlmostEqual(res["final_answer"]["safety_stock"], expected_ss, places=6)
        self.assertAlmostEqual(res["final_answer"]["target_inventory_level"], expected_target, places=6)
        self.assertAlmostEqual(res["final_answer"]["order_quantity"], expected_target - 270.0, places=6)
        ver = verify_calculation("fixed_period_review", inputs, res)
        self.assertTrue(ver["passed"])

    def test_periodic_review_zero_order_and_omitted_inventory_position(self) -> None:
        res_zero = calculate_periodic_review(
            review_period=7,
            lead_time=2,
            demand_mean=20,
            demand_std=5,
            service_level=0.95,
            inventory_position=500,
        )
        self.assertEqual(res_zero["final_answer"]["order_quantity"], 0.0)

        res_no_pos = calculate_periodic_review(
            review_period=7,
            lead_time=2,
            demand_mean=20,
            demand_std=5,
            service_level=0.95,
        )
        self.assertIsNone(res_no_pos["inventory_position"])
        self.assertIsNone(res_no_pos["final_answer"]["order_quantity"])

    def test_newsvendor_normal(self) -> None:
        inputs = {
            "selling_price": 6.0,
            "unit_cost": 2.4,
            "salvage_value": 0.5,
            "distribution": "normal",
            "demand_mean": 180.0,
            "demand_std": 30.0,
        }
        res = calculate_newsvendor(**inputs)
        self._assert_structured_schema(res, "calculate_newsvendor")
        cr = 3.6 / 5.5
        expected_q = 180.0 + normal_ppf(cr) * 30.0
        self.assertAlmostEqual(res["final_answer"]["underage_cost"], 3.6, places=10)
        self.assertAlmostEqual(res["final_answer"]["overage_cost"], 1.9, places=10)
        self.assertAlmostEqual(res["final_answer"]["critical_ratio"], cr, places=10)
        self.assertAlmostEqual(res["final_answer"]["recommended_order_quantity"], expected_q, places=8)
        self.assertAlmostEqual(res["expected_sales"] + res["expected_leftover"], expected_q, places=8)
        self.assertAlmostEqual(res["expected_sales"] + res["expected_lost_sales"], 180.0, places=8)
        ver = verify_calculation("newsvendor", inputs, res)
        self.assertTrue(ver["passed"])

    def test_newsvendor_uniform_and_discrete(self) -> None:
        uni_inputs = {
            "selling_price": 50.0,
            "unit_cost": 20.0,
            "salvage_value": 5.0,
            "distribution": "uniform",
            "demand_min": 100.0,
            "demand_max": 300.0,
        }
        res_uni = calculate_newsvendor(**uni_inputs)
        self._assert_structured_schema(res_uni, "calculate_newsvendor")
        self.assertAlmostEqual(res_uni["critical_ratio"], 2.0 / 3.0, places=10)
        self.assertAlmostEqual(res_uni["recommended_order_quantity"], 100.0 + 400.0 / 3.0, places=8)
        self.assertTrue(verify_calculation("newsvendor", uni_inputs, res_uni)["passed"])

        disc_inputs = {
            "selling_price": 4.0,
            "unit_cost": 1.5,
            "salvage_value": 0.0,
            "distribution": "discrete",
            "demand_outcomes": [(100, 0.25), (150, 0.45), (200, 0.30)],
        }
        res_disc = calculate_newsvendor(**disc_inputs)
        self._assert_structured_schema(res_disc, "calculate_newsvendor")
        self.assertAlmostEqual(res_disc["critical_ratio"], 0.625, places=10)
        self.assertEqual(res_disc["recommended_order_quantity"], 150.0)
        self.assertTrue(verify_calculation("newsvendor", disc_inputs, res_disc)["passed"])

    def test_validation_never_invents_missing_inputs(self) -> None:
        # EOQ invalid / missing
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=-100, ordering_cost=50, holding_cost=2)
        with self.assertRaises(ValueError):
            calculate_eoq(annual_demand=True, ordering_cost=50, holding_cost=2)  # type: ignore[arg-type]

        # ROP refuses missing variability (neither demand_std nor lead_time_std supplied)
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=100, lead_time=5)

        # Periodic review refuses missing variability
        with self.assertRaises(ValueError):
            calculate_periodic_review(review_period=14, lead_time=3, demand_mean=50, service_level=0.95)

        # Newsvendor refuses unprofitable Cu <= 0 or missing demand
        with self.assertRaises(ValueError):
            calculate_newsvendor(
                selling_price=5, unit_cost=10, distribution="normal", demand_mean=50, demand_std=10
            )
        with self.assertRaises(ValueError):
            calculate_newsvendor(selling_price=20, unit_cost=10, distribution="normal")

    def test_verify_catches_perturbed_result(self) -> None:
        res = calculate_eoq(annual_demand=1000, ordering_cost=50, holding_cost=2)
        bad_res = dict(res)
        bad_res["optimal_order_quantity"] = res["optimal_order_quantity"] + 100.0
        bad_res["annual_holding_cost"] = (bad_res["optimal_order_quantity"] / 2.0) * 2.0
        ver = verify_calculation(
            "eoq", {"annual_demand": 1000, "ordering_cost": 50, "holding_cost": 2}, bad_res
        )
        self.assertFalse(ver["passed"])


class TestInventoryDecisionAgentEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = InventoryDecisionAgent(parser_mode="rule")

    def test_agent_eoq_natural_language(self) -> None:
        q = (
            "We sell 12,000 units annually. Ordering cost is $150 per order, and "
            "holding cost is $4 per unit per year. Find the optimal order quantity."
        )
        resp = self.agent.run(q)
        self.assertEqual(resp.status, "completed")
        self.assertEqual(resp.identification.model_id, "eoq")
        self.assertEqual(resp.tool_selection.tool_name, "calculate_eoq")
        self.assertTrue(resp.verification.passed)
        self.assertAlmostEqual(resp.calculation["optimal_order_quantity"], math.sqrt(900000.0), places=6)

    def test_agent_quantity_discount_natural_language(self) -> None:
        q = (
            "Annual demand is 10,000 units, ordering cost is $80, and annual holding "
            "cost is 20% of unit price. Supplier price breaks: 1-499 units at $12.00, "
            "500-999 units at $11.50, 1000+ units at $11.00."
        )
        resp = self.agent.run(q)
        self.assertEqual(resp.status, "completed")
        self.assertEqual(resp.identification.model_id, "quantity_discount")
        self.assertEqual(resp.tool_selection.tool_name, "calculate_quantity_discount")
        self.assertTrue(resp.verification.passed)
        self.assertEqual(resp.calculation["optimal_tier"], 3)
        self.assertAlmostEqual(resp.calculation["optimal_order_quantity"], 1000.0, places=6)

    def test_agent_rop_natural_language(self) -> None:
        q = (
            "Weekly demand has mean 300 and std dev 45. Lead time averages 3 weeks "
            "with a standard deviation of 0.5 weeks. Target cycle service level is 98%. "
            "Calculate safety stock and reorder point."
        )
        resp = self.agent.run(q)
        self.assertEqual(resp.status, "completed")
        self.assertEqual(resp.identification.model_id, "safety_stock_rop")
        self.assertEqual(resp.tool_selection.tool_name, "calculate_rop")
        self.assertTrue(resp.verification.passed)
        self.assertAlmostEqual(resp.calculation["expected_lead_time_demand"], 900.0, places=6)

    def test_agent_rop_constant_demand_constant_lead_time(self) -> None:
        q = (
            "Daily demand is constant at 100 units and lead time is constant at 5 days. "
            "What is the reorder point and safety stock?"
        )
        resp = self.agent.run(q)
        self.assertEqual(resp.status, "completed")
        self.assertEqual(resp.identification.model_id, "safety_stock_rop")
        self.assertEqual(resp.calculation["formula_case"], "constant_demand_constant_lead_time")
        self.assertAlmostEqual(resp.calculation["safety_stock"], 0.0, places=8)
        self.assertAlmostEqual(resp.calculation["reorder_point"], 500.0, places=8)
        self.assertTrue(resp.verification.passed)

    def test_agent_periodic_review_natural_language(self) -> None:
        q = (
            "We review inventory every 14 days and lead time is 3 days. Daily demand "
            "averages 50 units with standard deviation 8. We need a 95% service level. "
            "We have 220 units on hand and 50 on order. How much should we order?"
        )
        resp = self.agent.run(q)
        self.assertEqual(resp.status, "completed")
        self.assertEqual(resp.identification.model_id, "fixed_period_review")
        self.assertEqual(resp.tool_selection.tool_name, "calculate_periodic_review")
        self.assertTrue(resp.verification.passed)
        self.assertAlmostEqual(resp.calculation["protection_period"], 17.0, places=6)
        self.assertGreater(resp.calculation["order_quantity"], 600.0)

    def test_agent_newsvendor_natural_language(self) -> None:
        q = (
            "A bakery sells fresh loaves for $6.00 that cost $2.40 to bake. Unsold loaves "
            "at the end of the day are donated for $0.50 salvage value. Daily demand is "
            "normal with mean 180 and standard deviation 30. How many should we bake?"
        )
        resp = self.agent.run(q)
        self.assertEqual(resp.status, "completed")
        self.assertEqual(resp.identification.model_id, "newsvendor")
        self.assertEqual(resp.tool_selection.tool_name, "calculate_newsvendor")
        self.assertTrue(resp.verification.passed)
        self.assertAlmostEqual(resp.calculation["critical_ratio"], 3.6 / 5.5, places=8)

    def test_agent_missing_inputs_returns_diagnostic(self) -> None:
        resp = self.agent.run("What is the EOQ when annual demand is 5,000 units?")
        self.assertEqual(resp.status, "missing_inputs")
        self.assertEqual(resp.identification.model_id, "eoq")
        self.assertIn("ordering_cost", resp.extraction.missing_required)

        resp_pr = self.agent.run(
            "We review inventory every 14 days, lead time is 3 days, daily demand averages 50, service level 95%."
        )
        self.assertEqual(resp_pr.status, "missing_inputs")
        self.assertIn("demand_std (or lead_time_std or safety_stock)", resp_pr.extraction.missing_required)

    def test_agent_unrecognized_question_needs_clarification(self) -> None:
        resp = self.agent.run("What is the weather forecast for tomorrow?")
        self.assertEqual(resp.status, "needs_clarification")
        self.assertIsNone(resp.identification.model_id)


class TestNaturalLanguageSynonymsAndExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = InventoryDecisionAgent(parser_mode="rule")

    def test_eoq_all_ordering_cost_synonyms(self) -> None:
        ordering_phrases = [
            "ordering cost is $100",
            "order cost of $100",
            "setup cost = $100",
            "setup expense is $100",
            "cost per order is $100",
            "$100 cost per order",
            "cost of placing an order is $100",
            "ordering/setup cost is $100",
        ]
        for phrase in ordering_phrases:
            with self.subTest(phrase=phrase):
                q = (
                    "What is the EOQ? Holding cost is $5 per unit per year. "
                    f"Annual demand is 2,500 units, and the {phrase}."
                )
                resp = self.agent.run(q)
                self.assertEqual(resp.status, "completed")
                self.assertEqual(resp.identification.model_id, "eoq")
                self.assertAlmostEqual(resp.calculation["inputs"]["ordering_cost"], 100.0, places=6)
                self.assertAlmostEqual(resp.calculation["optimal_order_quantity"], math.sqrt(100000.0), places=6)
                self.assertTrue(resp.verification.passed)

    def test_rop_distinguishes_lead_time_std_and_demand_std(self) -> None:
        questions = [
            (
                "Average demand is 200 units per week with a demand standard deviation of 30 units. "
                "Average lead time is 2 weeks with a lead-time standard deviation of 0.5 weeks. "
                "Target service level is 95%. Find safety stock and reorder point."
            ),
            (
                "Average demand is 200 units per week and standard deviation of demand is 30 units. "
                "Average lead time is 2 weeks and standard deviation of lead time is 0.5 weeks. "
                "Target service level is 95%. Calculate reorder point and safety stock."
            ),
        ]
        for q in questions:
            with self.subTest(question=q[:60]):
                resp = self.agent.run(q)
                self.assertEqual(resp.status, "completed")
                self.assertEqual(resp.identification.model_id, "safety_stock_rop")
                self.assertEqual(resp.calculation["formula_case"], "variable_demand_variable_lead_time")
                self.assertAlmostEqual(resp.calculation["inputs"]["demand_mean"], 200.0, places=6)
                self.assertAlmostEqual(resp.calculation["inputs"]["demand_std"], 30.0, places=6)
                self.assertAlmostEqual(resp.calculation["inputs"]["lead_time"], 2.0, places=6)
                self.assertAlmostEqual(resp.calculation["inputs"]["lead_time_std"], 0.5, places=6)
                self.assertAlmostEqual(resp.calculation["inputs"]["service_level"], 0.95, places=6)
                self.assertTrue(resp.verification.passed)

    def test_quantity_discount_synonyms_fixed_and_percentage_holding(self) -> None:
        q_pct = (
            "Annual demand is 5,000 units with ordering cost of $49 and percentage holding cost "
            "of 20% per year. The base unit purchase price is $5.00, with a quantity break of "
            "1,000 units or more at a unit price of $4.80."
        )
        resp_pct = self.agent.run(q_pct)
        self.assertEqual(resp_pct.status, "completed")
        self.assertEqual(resp_pct.identification.model_id, "quantity_discount")
        self.assertAlmostEqual(resp_pct.calculation["optimal_order_quantity"], 1000.0, places=6)
        self.assertAlmostEqual(resp_pct.calculation["optimal_unit_price"], 4.80, places=6)
        self.assertTrue(resp_pct.verification.passed)

        q_fixed = (
            "Annual demand is 4,000 units, ordering cost is $60, and fixed holding cost is $3 "
            "per unit per year. Supplier tiers: 1-499 at $20, 500 or more at $19."
        )
        resp_fixed = self.agent.run(q_fixed)
        self.assertEqual(resp_fixed.status, "completed")
        self.assertEqual(resp_fixed.identification.model_id, "quantity_discount")
        self.assertAlmostEqual(resp_fixed.calculation["optimal_unit_price"], 19.0, places=6)
        self.assertTrue(resp_fixed.verification.passed)

    def test_periodic_review_explicit_safety_stock_and_inventory_position(self) -> None:
        # Direct tool test with explicit safety_stock
        tool_res = calculate_periodic_review(
            review_period=10,
            lead_time=4,
            demand_mean=25,
            safety_stock=60,
            on_hand_inventory=180,
            on_order_inventory=40,
            backorders=10,
            time_unit="day",
        )
        self.assertAlmostEqual(tool_res["protection_interval"], 14.0, places=8)
        self.assertAlmostEqual(tool_res["expected_protection_demand"], 350.0, places=8)
        self.assertAlmostEqual(tool_res["safety_stock"], 60.0, places=8)
        self.assertAlmostEqual(tool_res["target_inventory_level"], 410.0, places=8)
        self.assertAlmostEqual(tool_res["inventory_position"], 210.0, places=8)
        self.assertAlmostEqual(tool_res["order_quantity"], 200.0, places=8)
        self.assertTrue(verify_calculation("fixed_period_review", tool_res["inputs"], tool_res)["passed"])

        # End-to-end NL agent test with explicit safety stock + OH/OO/BO
        q_ss = (
            "Under a fixed-period review system, review period is 10 days and lead time is 4 days. "
            "Demand is 25 units per day and safety stock is 60 units. "
            "We have 180 units on-hand inventory, 40 units on-order inventory, and 10 units in backorders. "
            "How much should we order?"
        )
        resp_ss = self.agent.run(q_ss)
        self.assertEqual(resp_ss.status, "completed")
        self.assertEqual(resp_ss.identification.model_id, "fixed_period_review")
        self.assertTrue(resp_ss.verification.passed)
        self.assertAlmostEqual(resp_ss.calculation["target_inventory_level"], 410.0, places=6)
        self.assertAlmostEqual(resp_ss.calculation["order_quantity"], 200.0, places=6)

        # End-to-end NL agent test with direct inventory position
        q_ip = (
            "Under a periodic review policy, review period is 7 days, lead time is 3 days, "
            "demand averages 40 units/day with demand standard deviation of 6 units, service level is 95%, "
            "and inventory position is 250 units."
        )
        resp_ip = self.agent.run(q_ip)
        self.assertEqual(resp_ip.status, "completed")
        self.assertEqual(resp_ip.identification.model_id, "fixed_period_review")
        self.assertAlmostEqual(resp_ip.calculation["inventory_position"], 250.0, places=6)
        self.assertTrue(resp_ip.verification.passed)

    def test_newsvendor_synonyms_and_direct_shortage_excess(self) -> None:
        # Variant 1: passive purchase phrasing + selling price + salvage value + normal distribution
        q1 = (
            "Seasonal jackets are purchased at $45 per unit, have a selling price of $110, "
            "and leftover units have a salvage value of $25. Demand follows a normal distribution "
            "with mean demand of 300 units and demand standard deviation of 50 units."
        )
        resp1 = self.agent.run(q1)
        self.assertEqual(resp1.status, "completed")
        self.assertEqual(resp1.identification.model_id, "newsvendor")
        self.assertAlmostEqual(resp1.calculation["underage_cost"], 65.0, places=6)
        self.assertAlmostEqual(resp1.calculation["overage_cost"], 20.0, places=6)
        self.assertAlmostEqual(resp1.calculation["critical_ratio"], 65.0 / 85.0, places=6)
        self.assertTrue(resp1.verification.passed)

        # Variant 2: direct shortage cost and excess cost + N(mu, sigma)
        q2 = (
            "In a newsvendor setting, shortage cost is $15 per unit and excess cost is $5 per unit. "
            "Demand distribution is N(100, 20). What is the optimal order quantity?"
        )
        resp2 = self.agent.run(q2)
        self.assertEqual(resp2.status, "completed")
        self.assertEqual(resp2.identification.model_id, "newsvendor")
        self.assertAlmostEqual(resp2.calculation["underage_cost"], 15.0, places=6)
        self.assertAlmostEqual(resp2.calculation["overage_cost"], 5.0, places=6)
        self.assertAlmostEqual(resp2.calculation["critical_ratio"], 0.75, places=6)
        self.assertTrue(resp2.verification.passed)

    def test_partial_inputs_never_report_present_parameter_as_missing(self) -> None:
        resp_nv = self.agent.run(
            "In a newsvendor problem, selling price is $50, mean demand is 200, demand standard deviation is 25."
        )
        self.assertEqual(resp_nv.status, "missing_inputs")
        self.assertIn("selling_price", resp_nv.extraction.params)
        self.assertIn("unit_cost", resp_nv.extraction.missing_required)
        self.assertNotIn("selling_price", " ".join(resp_nv.extraction.missing_required))


class TestStudentVerificationOutputs(unittest.TestCase):
    """Verify that every calculation tool and agent explanation returns all 8 verification sections."""

    REQUIRED_KEYS = (
        "inputs_extracted",
        "variable_definitions",
        "formula",
        "formula_with_substituted_values",
        "intermediate_calculations",
        "final_answer",
        "verification_checks",
        "assumptions",
    )

    def _assert_has_all_eight_sections(self, result: Dict[str, Any]) -> None:
        for k in self.REQUIRED_KEYS:
            self.assertIn(k, result, f"Result dictionary missing required key: {k!r}")
            val = result[k]
            self.assertTrue(val is not None, f"Required section {k!r} must not be None")
            if isinstance(val, (dict, list, str)):
                self.assertTrue(len(val) > 0, f"Required section {k!r} must not be empty")

        self.assertIn("verification_passed", result)
        self.assertTrue(result["verification_passed"], "All verification checks must pass for valid inputs")
        self.assertTrue(all(c["passed"] for c in result["verification_checks"]))

    def test_eoq_returns_all_eight_sections(self) -> None:
        res = calculate_eoq(annual_demand=12000, ordering_cost=150, holding_cost=4)
        self._assert_has_all_eight_sections(res)

        # Check variable definitions
        self.assertIn("D", res["variable_definitions"])
        self.assertIn("S", res["variable_definitions"])
        self.assertIn("H", res["variable_definitions"])
        self.assertIn("Q*", res["variable_definitions"])

        # Check formulas
        self.assertIn("optimal_order_quantity", res["formula"])
        self.assertIn("sqrt", res["formula"]["optimal_order_quantity"])

        # Check formula with substituted values
        sub = res["formula_with_substituted_values"]["optimal_order_quantity"]
        self.assertIn("12,000", sub)
        self.assertIn("150", sub)
        self.assertIn("4", sub)
        self.assertIn("948.68", sub)

    def test_quantity_discount_returns_all_eight_sections(self) -> None:
        tiers = [
            {"min_qty": 1, "max_qty": 499, "unit_price": 12.0},
            {"min_qty": 500, "max_qty": 999, "unit_price": 11.5},
            {"min_qty": 1000, "max_qty": None, "unit_price": 11.0},
        ]
        res = calculate_quantity_discount(
            annual_demand=10000, ordering_cost=80, tiers=tiers, holding_rate=0.20
        )
        self._assert_has_all_eight_sections(res)
        self.assertIn("tier_evaluations", res["formula_with_substituted_values"])
        self.assertEqual(len(res["formula_with_substituted_values"]["tier_evaluations"]), 3)
        self.assertIn("optimal_decision", res["formula_with_substituted_values"])

    def test_rop_returns_all_eight_sections(self) -> None:
        # Test variable demand and variable lead time
        res = calculate_rop(
            demand_mean=100,
            demand_std=15,
            lead_time=4,
            lead_time_std=1,
            service_level=0.95,
            time_unit="day",
        )
        self._assert_has_all_eight_sections(res)
        self.assertIn("expected_lead_time_demand", res["formula"])
        self.assertIn("lead_time_demand_std", res["formula"])
        self.assertIn("safety_stock", res["formula"])
        self.assertIn("reorder_point", res["formula"])

        # Substituted values
        sub_rop = res["formula_with_substituted_values"]["reorder_point"]
        self.assertIn("ROP =", sub_rop)

        # Test deterministic case
        res_det = calculate_rop(
            demand_mean=50,
            demand_std=0.0,
            lead_time=3,
            lead_time_std=0.0,
        )
        self._assert_has_all_eight_sections(res_det)
        self.assertEqual(res_det["safety_stock"], 0.0)

    def test_periodic_review_returns_all_eight_sections(self) -> None:
        # Stochastic case
        res = calculate_periodic_review(
            review_period=14,
            lead_time=3,
            demand_mean=50,
            demand_std=8,
            service_level=0.95,
            on_hand_inventory=220,
            on_order_inventory=50,
            time_unit="day",
        )
        self._assert_has_all_eight_sections(res)
        self.assertIn("protection_period", res["formula"])
        self.assertIn("target_inventory_level", res["formula"])
        self.assertIn("order_quantity", res["formula"])

        # Explicit safety stock case
        res_ss = calculate_periodic_review(
            review_period=10,
            lead_time=4,
            demand_mean=25,
            safety_stock=60,
            inventory_position=210,
            time_unit="day",
        )
        self._assert_has_all_eight_sections(res_ss)
        self.assertEqual(res_ss["safety_stock"], 60.0)

    def test_newsvendor_returns_all_eight_sections(self) -> None:
        # Normal distribution
        res_norm = calculate_newsvendor(
            selling_price=6.0,
            unit_cost=2.4,
            salvage_value=0.5,
            demand_mean=180,
            demand_std=30,
            distribution="normal",
        )
        self._assert_has_all_eight_sections(res_norm)
        self.assertIn("underage_cost", res_norm["formula_with_substituted_values"])
        self.assertIn("overage_cost", res_norm["formula_with_substituted_values"])
        self.assertIn("critical_ratio", res_norm["formula_with_substituted_values"])
        self.assertIn("optimal_order_quantity", res_norm["formula_with_substituted_values"])

        # Uniform distribution
        res_uni = calculate_newsvendor(
            underage_cost=10,
            overage_cost=5,
            demand_min=50,
            demand_max=150,
            distribution="uniform",
        )
        self._assert_has_all_eight_sections(res_uni)

        # Discrete distribution
        res_disc = calculate_newsvendor(
            underage_cost=12,
            overage_cost=4,
            demand_outcomes=[(10, 0.2), (20, 0.5), (30, 0.3)],
            distribution="discrete",
        )
        self._assert_has_all_eight_sections(res_disc)

    def test_agent_explanation_contains_all_eight_numbered_sections(self) -> None:
        agent = InventoryDecisionAgent(parser_mode="rule")
        sample_questions = [
            "Our annual demand is 12,000 units. Ordering cost is $150, and holding cost is $4 per unit per year.",
            "Annual demand is 10,000 units, ordering cost is $80 per order, and annual holding cost is 20% of unit price. Price breaks: 1-499 units at $12.00, 500-999 units at $11.50, and 1000+ units at $11.00.",
            "Daily demand averages 120 units with std dev 20. Lead time is 4 days. Service level is 95%.",
            "Review period is 14 days, lead time is 3 days. Demand is 50 units/day with std dev 8. Service level 95%. On hand 220, on order 50.",
            "Loaves sell for $6.00 and cost $2.40. Salvage is $0.50. Demand is normal with mean 180 and std dev 30.",
        ]

        expected_headings = [
            "### 1. Inputs Extracted",
            "### 2. Variable Definitions",
            "### 3. Formula",
            "### 4. Formula with Substituted Values",
            "### 5. Intermediate Calculations",
            "### 6. Final Answer",
            "### 7. Verification Checks",
            "### 8. Assumptions",
        ]

        for q in sample_questions:
            resp = agent.run(q)
            self.assertEqual(resp.status, "completed")
            for heading in expected_headings:
                self.assertIn(
                    heading,
                    resp.explanation,
                    f"Agent explanation missing required student-verifiable section heading: {heading!r}",
                )


class TestInventoryCsvAnalysis(unittest.TestCase):
    """Test CSV data validation and pandas inventory analysis calculations."""

    @classmethod
    def setUpClass(cls) -> None:
        from validate_abcd_project import _ensure_pandas_available

        cls.pd, _ = _ensure_pandas_available()

    def test_csv_validation_missing_columns(self) -> None:
        from app import analyze_inventory_dataframe

        # Missing 'Category' and 'Unit Cost'
        df_invalid = self.pd.DataFrame({"Product": ["Widget A", "Widget B"], "Units": [100, 200]})
        result = analyze_inventory_dataframe(df_invalid)

        self.assertFalse(result["valid"])
        self.assertIn("Category", result["missing_columns"])
        self.assertIn("Unit Cost", result["missing_columns"])
        self.assertIn("Missing required column(s)", result["error_message"])
        self.assertEqual(result["num_rows"], 2)
        self.assertEqual(result["num_columns"], 2)

    def test_csv_full_analysis_calculations(self) -> None:
        from app import analyze_inventory_dataframe

        df_valid = self.pd.DataFrame({
            "Product": ["SKU-101", "SKU-202", "SKU-303", "SKU-404"],
            "Category": ["Electronics", "Electronics", "Hardware", "Hardware"],
            "Units": [120, 350, 80, 250],
            "Unit Cost": [45.0, 15.0, 10.0, 5.0],
        })

        result = analyze_inventory_dataframe(df_valid)

        self.assertTrue(result["valid"])
        self.assertEqual(result["missing_columns"], [])
        self.assertEqual(result["num_rows"], 4)
        self.assertEqual(result["num_columns"], 4)

        # 1. Total Units: 120 + 350 + 80 + 250 = 800
        self.assertAlmostEqual(result["total_units"], 800.0, places=4)

        # 2. Average Unit Cost: (45 + 15 + 10 + 5) / 4 = 18.75
        self.assertAlmostEqual(result["average_unit_cost"], 18.75, places=4)

        # 3. Product with highest units: SKU-202 (350 units)
        self.assertEqual(result["highest_units_product"], "SKU-202")
        self.assertAlmostEqual(result["highest_units_count"], 350.0, places=4)

        # 4. Group by Category
        cat_df = result["units_by_category"]
        cat_dict = dict(zip(cat_df["Category"], cat_df["Units"]))
        self.assertAlmostEqual(cat_dict["Electronics"], 470.0, places=4)
        self.assertAlmostEqual(cat_dict["Hardware"], 330.0, places=4)

        # 5. Calculated column: Inventory Value = Units * Unit Cost
        analyzed_df = result["dataframe"]
        self.assertIn("Inventory Value", analyzed_df.columns)
        expected_values = [5400.0, 5250.0, 800.0, 1250.0]
        actual_values = list(analyzed_df["Inventory Value"])
        for exp, act in zip(expected_values, actual_values):
            self.assertAlmostEqual(exp, float(act), places=4)

        # 6. Total Inventory Value: 5400 + 5250 + 800 + 1250 = 12,700
        self.assertAlmostEqual(result["total_inventory_value"], 12700.0, places=4)


class TestMulticriteriaAbcdClassification(unittest.TestCase):
    """Comprehensive tests for BAN 517 Multicriteria ABCD Classification."""

    @classmethod
    def setUpClass(cls) -> None:
        from validate_abcd_project import _ensure_pandas_available

        cls.pd, _ = _ensure_pandas_available()

    def test_cumulative_percentage_boundaries_continuous_no_rounding(self) -> None:
        from abcd_classification import classify_cumulative_pct

        # Exact continuous boundaries (no pre-rounding)
        self.assertEqual(classify_cumulative_pct(0.0), "A")
        self.assertEqual(classify_cumulative_pct(59.9999), "A")
        self.assertEqual(classify_cumulative_pct(60.0), "A")
        self.assertEqual(classify_cumulative_pct(60.0001), "B")
        self.assertEqual(classify_cumulative_pct(60.4), "B")  # Would fail if pre-rounded to 60!
        self.assertEqual(classify_cumulative_pct(79.9999), "B")
        self.assertEqual(classify_cumulative_pct(80.0), "B")
        self.assertEqual(classify_cumulative_pct(80.0001), "C")
        self.assertEqual(classify_cumulative_pct(80.4), "C")  # Would fail if pre-rounded to 80!
        self.assertEqual(classify_cumulative_pct(89.9999), "C")
        self.assertEqual(classify_cumulative_pct(90.0), "C")
        self.assertEqual(classify_cumulative_pct(90.0001), "D")
        self.assertEqual(classify_cumulative_pct(90.4), "D")  # Would fail if pre-rounded to 90!
        self.assertEqual(classify_cumulative_pct(100.0), "D")

    def test_weighted_score_and_final_class_boundaries(self) -> None:
        from abcd_classification import (
            class_to_score,
            classify_weighted_score,
            compute_weighted_score,
        )

        # Official example: Sales Class A (4) and Hits Class B (3) -> 3.65 -> A
        score_ab = compute_weighted_score(class_to_score("A"), class_to_score("B"))
        self.assertAlmostEqual(score_ab, 3.65, places=9)
        self.assertEqual(classify_weighted_score(score_ab), "A")

        # Continuous weighted-score boundaries
        self.assertEqual(classify_weighted_score(4.00), "A")
        self.assertEqual(classify_weighted_score(3.25), "A")
        self.assertEqual(classify_weighted_score(3.249999), "B")
        self.assertEqual(classify_weighted_score(3.245), "B")
        self.assertEqual(classify_weighted_score(2.50), "B")
        self.assertEqual(classify_weighted_score(2.499999), "C")
        self.assertEqual(classify_weighted_score(2.495), "C")
        self.assertEqual(classify_weighted_score(1.75), "C")
        self.assertEqual(classify_weighted_score(1.749999), "D")
        self.assertEqual(classify_weighted_score(1.00), "D")

        # Verify all 16 discrete (Sales Class, Hits Class) pairs
        expected_matrix = {
            ("A", "A"): (4.00, "A"),
            ("A", "B"): (3.65, "A"),
            ("A", "C"): (3.30, "A"),
            ("A", "D"): (2.95, "B"),
            ("B", "A"): (3.35, "A"),
            ("B", "B"): (3.00, "B"),
            ("B", "C"): (2.65, "B"),
            ("B", "D"): (2.30, "C"),
            ("C", "A"): (2.70, "B"),
            ("C", "B"): (2.35, "C"),
            ("C", "C"): (2.00, "C"),
            ("C", "D"): (1.65, "D"),
            ("D", "A"): (2.05, "C"),
            ("D", "B"): (1.70, "D"),
            ("D", "C"): (1.35, "D"),
            ("D", "D"): (1.00, "D"),
        }
        for (sc, hc), (exp_score, exp_cls) in expected_matrix.items():
            ws = compute_weighted_score(class_to_score(sc), class_to_score(hc))
            self.assertAlmostEqual(ws, exp_score, places=9)
            self.assertEqual(classify_weighted_score(ws), exp_cls)

    def test_independent_sales_and_hits_ranking_and_item_alignment(self) -> None:
        from abcd_classification import classify_abcd_dataframe

        # Inverted Sales vs Hits rankings:
        # Total Sales = 1000:
        #   Item 200: 600 (60% -> A, score 4)
        #   Item 300: 200 (80% -> B, score 3)
        #   Item 400: 100 (90% -> C, score 2)
        #   Item 100: 100 (100% -> D, score 1) -- wait: 400 vs 100 tied at 100, so 100 sorts before 400!
        # Let's use distinct values so ranking inversion is crystal clear:
        # Total Sales = 1000: Item 200 (550->55% A), Item 300 (250->80% B), Item 400 (100->90% C), Item 100 (100? no, 99 & 101)
        # Use Sales = [60, 600, 200, 140] -> Total = 1000:
        #   Sales desc: 200 (600, 60% A), 300 (200, 80% B), 400 (140, 94% D), 100 (60, 100% D)
        # Use Hits  = [600, 50, 150, 200] -> Total = 1000:
        #   Hits desc : 100 (600, 60% A), 400 (200, 80% B), 300 (150, 95% D), 200 (50, 100% D)
        df = self.pd.DataFrame({
            "Item Number": [100, 200, 300, 400],
            "Item Name": ["LowSales-HighHits", "HighSales-LowHits", "Mid-Mid", "LowMid-HighMid"],
            "UoM": ["EA", "EA", "EA", "EA"],
            "Sales ($)": [60.0, 600.0, 200.0, 140.0],
            "Hits": [600.0, 50.0, 150.0, 200.0],
        })

        res = classify_abcd_dataframe(df)
        self.assertTrue(res["valid"])
        out = res["classified_df"]

        # Original item order preserved: [100, 200, 300, 400]
        self.assertEqual(list(out["Item Number"]), [100, 200, 300, 400])

        # Item 100: Sales D (100%, score 1), Hits A (60%, score 4) -> 0.65*1 + 0.35*4 = 2.05 -> C
        self.assertAlmostEqual(out["Cumulative Sales %"][0], 100.0, places=6)
        self.assertEqual(out["Sales Class"][0], "D")
        self.assertAlmostEqual(out["Cumulative Hits %"][0], 60.0, places=6)
        self.assertEqual(out["Hits Class"][0], "A")
        self.assertAlmostEqual(out["Combined Weighted Score"][0], 2.05, places=6)
        self.assertEqual(out["Final ABCD Class"][0], "C")

        # Item 200: Sales A (60%, score 4), Hits D (100%, score 1) -> 0.65*4 + 0.35*1 = 2.95 -> B
        self.assertAlmostEqual(out["Cumulative Sales %"][1], 60.0, places=6)
        self.assertEqual(out["Sales Class"][1], "A")
        self.assertAlmostEqual(out["Cumulative Hits %"][1], 100.0, places=6)
        self.assertEqual(out["Hits Class"][1], "D")
        self.assertAlmostEqual(out["Combined Weighted Score"][1], 2.95, places=6)
        self.assertEqual(out["Final ABCD Class"][1], "B")

        # Item 300: Sales B (80%, score 3), Hits D (95%, score 1) -> 0.65*3 + 0.35*1 = 2.30 -> C
        self.assertAlmostEqual(out["Cumulative Sales %"][2], 80.0, places=6)
        self.assertEqual(out["Sales Class"][2], "B")
        self.assertAlmostEqual(out["Cumulative Hits %"][2], 95.0, places=6)
        self.assertEqual(out["Hits Class"][2], "D")
        self.assertAlmostEqual(out["Combined Weighted Score"][2], 2.30, places=6)
        self.assertEqual(out["Final ABCD Class"][2], "C")

        # Item 400: Sales D (94%, score 1), Hits B (80%, score 3) -> 0.65*1 + 0.35*3 = 1.70 -> D
        self.assertAlmostEqual(out["Cumulative Sales %"][3], 94.0, places=6)
        self.assertEqual(out["Sales Class"][3], "D")
        self.assertAlmostEqual(out["Cumulative Hits %"][3], 80.0, places=6)
        self.assertEqual(out["Hits Class"][3], "B")
        self.assertAlmostEqual(out["Combined Weighted Score"][3], 1.70, places=6)
        self.assertEqual(out["Final ABCD Class"][3], "D")

        self.assertTrue(res["reconciliation"]["all_internal_checks_passed"])

    def test_deterministic_tie_breaking_independent_of_input_order(self) -> None:
        from abcd_classification import classify_abcd_dataframe

        # Items 105 and 102 have identical Sales ($500 each -> total $1000) and Hits (50 each -> total 100)
        # Item 102 must always sort before 105 (50% -> A) and 105 gets 100% -> D regardless of row order!
        df_order_a = self.pd.DataFrame({
            "Item Number": [105, 102],
            "Item Name": ["Item 105", "Item 102"],
            "UoM": ["EA", "EA"],
            "Sales ($)": [500.0, 500.0],
            "Hits": [50.0, 50.0],
        })
        df_order_b = self.pd.DataFrame({
            "Item Number": [102, 105],
            "Item Name": ["Item 102", "Item 105"],
            "UoM": ["EA", "EA"],
            "Sales ($)": [500.0, 500.0],
            "Hits": [50.0, 50.0],
        })

        res_a = classify_abcd_dataframe(df_order_a)["classified_df"]
        res_b = classify_abcd_dataframe(df_order_b)["classified_df"]

        map_a = {res_a["Item Number"][i]: res_a["Final ABCD Class"][i] for i in range(2)}
        map_b = {res_b["Item Number"][i]: res_b["Final ABCD Class"][i] for i in range(2)}

        self.assertEqual(map_a, {102: "A", 105: "D"})
        self.assertEqual(map_b, {102: "A", 105: "D"})

    def test_abcd_validation_missing_columns_and_duplicate_item_numbers(self) -> None:
        from abcd_classification import classify_abcd_dataframe

        df_missing = self.pd.DataFrame({"Item Number": [1], "Sales ($)": [100.0]})
        res_missing = classify_abcd_dataframe(df_missing)
        self.assertFalse(res_missing["valid"])
        self.assertIn("Hits", res_missing["missing_columns"])
        self.assertIn("Item Name", res_missing["missing_columns"])
        self.assertIn("UoM", res_missing["missing_columns"])

        df_dup = self.pd.DataFrame({
            "Item Number": [101, 101],
            "Item Name": ["A", "B"],
            "UoM": ["EA", "EA"],
            "Sales ($)": [100.0, 200.0],
            "Hits": [10.0, 20.0],
        })
        res_dup = classify_abcd_dataframe(df_dup)
        self.assertFalse(res_dup["valid"])
        self.assertIn("Duplicate Item Number", res_dup["error_message"])

    def test_full_project_dataset_reconciliation_against_independent_calculation(self) -> None:
        import os
        from abcd_classification import ABCD_OUTPUT_COLUMNS, classify_abcd_dataframe
        from validate_abcd_project import independent_classify_project_csv

        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ProjectDataV2_UTF8.csv",
        )
        self.assertTrue(os.path.exists(csv_path), f"Required project CSV missing: {csv_path}")

        # 1. Independently compute expected answers using stdlib only (no production code reuse)
        expected = independent_classify_project_csv(csv_path)

        # 2. Execute production classification on DataFrame loaded from ProjectDataV2_UTF8.csv
        df = self.pd.read_csv(csv_path, encoding="utf-8-sig")
        prod = classify_abcd_dataframe(df)

        self.assertTrue(prod["valid"], prod.get("error_message"))
        recon = prod["reconciliation"]
        self.assertTrue(recon["all_internal_checks_passed"])
        self.assertTrue(recon["matches_project_benchmark"])

        # Project benchmark checks
        self.assertEqual(prod["total_items"], 205)
        self.assertAlmostEqual(prod["total_sales"], 82_046_150.0, places=2)
        self.assertAlmostEqual(prod["total_hits"], 23_252.0, places=2)
        self.assertEqual(recon["sum_class_items"], 205)
        self.assertAlmostEqual(recon["sum_class_sales"], 82_046_150.0, places=2)
        self.assertAlmostEqual(recon["sum_class_hits"], 23_252.0, places=2)
        self.assertTrue(recon["every_item_single_class"])

        # Output column structure check
        out_df = prod["classified_df"]
        self.assertEqual(list(out_df.columns), ABCD_OUTPUT_COLUMNS)
        self.assertEqual(len(out_df), 205)

        # Cell-by-cell cross-verification against independent computation
        for i, exp_row in enumerate(expected["rows"]):
            self.assertEqual(str(out_df["Item Number"][i]).strip(), exp_row["Item Number"])
            self.assertEqual(out_df["Item Name"][i], exp_row["Item Name"])
            self.assertEqual(out_df["UoM"][i], exp_row["UoM"])
            self.assertAlmostEqual(float(out_df["Sales ($)"][i]), exp_row["Sales ($)"], places=6)
            self.assertAlmostEqual(float(out_df["Hits"][i]), exp_row["Hits"], places=6)
            self.assertAlmostEqual(float(out_df["Cumulative Sales"][i]), exp_row["Cumulative Sales"], places=6)
            self.assertAlmostEqual(float(out_df["Cumulative Sales %"][i]), exp_row["Cumulative Sales %"], places=6)
            self.assertEqual(out_df["Sales Class"][i], exp_row["Sales Class"])
            self.assertEqual(int(out_df["Sales Score"][i]), exp_row["Sales Score"])
            self.assertAlmostEqual(float(out_df["Cumulative Hits"][i]), exp_row["Cumulative Hits"], places=6)
            self.assertAlmostEqual(float(out_df["Cumulative Hits %"][i]), exp_row["Cumulative Hits %"], places=6)
            self.assertEqual(out_df["Hits Class"][i], exp_row["Hits Class"])
            self.assertEqual(int(out_df["Hits Score"][i]), exp_row["Hits Score"])
            self.assertAlmostEqual(
                float(out_df["Combined Weighted Score"][i]),
                exp_row["Combined Weighted Score"],
                places=6,
            )
            self.assertEqual(out_df["Final ABCD Class"][i], exp_row["Final ABCD Class"])

        # Verify class summary totals match independent summary
        prod_summary = {row["Final ABCD Class"]: row for row in prod["summary_rows"]}
        for cls in ["A", "B", "C", "D"]:
            self.assertEqual(prod_summary[cls]["Item Count"], expected["class_summary"][cls]["count"])
            self.assertAlmostEqual(
                prod_summary[cls]["Total Sales ($)"],
                expected["class_summary"][cls]["sales"],
                places=2,
            )
            self.assertAlmostEqual(
                prod_summary[cls]["Total Hits"],
                expected["class_summary"][cls]["hits"],
                places=2,
            )

        # Verify course policies & safety stock disclaimer are present
        self.assertIn("Weekly", prod["policies"]["A"]["cycle_counting"])
        self.assertIn("95%–98%", prod["policies"]["A"]["cycle_service_level"])
        self.assertIn("Monthly", prod["policies"]["B"]["cycle_counting"])
        self.assertIn("85%–90%", prod["policies"]["B"]["cycle_service_level"])
        self.assertIn("Annual", prod["policies"]["C"]["cycle_counting"])
        self.assertIn("70%–80%", prod["policies"]["C"]["cycle_service_level"])
        self.assertIn("No cycle counting", prod["policies"]["D"]["cycle_counting"])
        self.assertIn("No safety stock", prod["policies"]["D"]["cycle_service_level"])
        self.assertIn("NOT calculated", prod["safety_stock_disclaimer"])


class TestStreamlitCsvIntegration(unittest.TestCase):
    """Test both supported CSV formats through app._render_csv_analysis_section()."""

    @classmethod
    def setUpClass(cls) -> None:
        from validate_abcd_project import _ensure_pandas_available

        cls.pd, _ = _ensure_pandas_available()

    def _make_mock_streamlit(self) -> Any:
        class _DummyCol:
            def __init__(self, log: list):
                self.log = log
            def metric(self, label, value, delta=None):
                self.log.append(("metric", label, value, delta))
            def write(self, text):
                self.log.append(("write", text))
            def markdown(self, text):
                self.log.append(("markdown", text))

        class _DummyExpander:
            def __init__(self, log: list, title: str):
                self.log = log
                self.title = title
            def __enter__(self):
                self.log.append(("expander_enter", self.title))
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        class _DummyColumnConfig:
            @staticmethod
            def NumberColumn(label, format=None):
                return {"label": label, "format": format}

        class MockStreamlit:
            def __init__(self):
                self.events: list = []
                self.column_config = _DummyColumnConfig()
            def header(self, text): self.events.append(("header", text))
            def subheader(self, text): self.events.append(("subheader", text))
            def caption(self, text): self.events.append(("caption", text))
            def markdown(self, text): self.events.append(("markdown", text))
            def write(self, text): self.events.append(("write", text))
            def error(self, text): self.events.append(("error", text))
            def success(self, text): self.events.append(("success", text))
            def info(self, text): self.events.append(("info", text))
            def dataframe(self, df, **kwargs): self.events.append(("dataframe", df, kwargs))
            def download_button(self, **kwargs): self.events.append(("download_button", kwargs))
            def columns(self, n): return [_DummyCol(self.events) for _ in range(n)]
            def expander(self, title): return _DummyExpander(self.events, title)

        return MockStreamlit()

    def test_single_file_uploader_preserved_in_app_source(self) -> None:
        import os

        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertEqual(src.count("file_uploader("), 1, "App must contain exactly one CSV file uploader.")

    def test_streamlit_renders_original_lecture_csv_format(self) -> None:
        import io
        from app import _render_csv_analysis_section

        lecture_csv = io.BytesIO(
            b"Product,Category,Units,Unit Cost\n"
            b"Widget A,Hardware,100,12.50\n"
            b"Widget B,Hardware,250,8.00\n"
            b"Sensor X,Electronics,50,40.00\n"
        )
        mock_st = self._make_mock_streamlit()
        _render_csv_analysis_section(mock_st, lecture_csv)

        errors = [e for e in mock_st.events if e[0] == "error"]
        self.assertEqual(errors, [], f"Unexpected errors for lecture CSV: {errors}")

        headers = [e[1] for e in mock_st.events if e[0] == "header"]
        self.assertIn("Inventory Data Analysis", headers)

        metrics = {e[1]: e[2] for e in mock_st.events if e[0] == "metric"}
        self.assertEqual(metrics["Total Units"], "400")
        self.assertEqual(metrics["Total Inventory Value"], "$5,250.00")
        self.assertEqual(metrics["Product with Highest Units"], "Widget B")

    def test_streamlit_renders_project_abcd_csv_format(self) -> None:
        import io
        import os
        from app import _render_csv_analysis_section

        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ProjectDataV2_UTF8.csv",
        )
        with open(csv_path, "rb") as fh:
            uploaded_buf = io.BytesIO(fh.read())

        mock_st = self._make_mock_streamlit()
        _render_csv_analysis_section(mock_st, uploaded_buf)

        errors = [e for e in mock_st.events if e[0] == "error"]
        self.assertEqual(errors, [], f"Unexpected errors for project ABCD CSV: {errors}")

        headers = [e[1] for e in mock_st.events if e[0] == "header"]
        self.assertIn("Multicriteria ABCD Inventory Classification", headers)

        metrics = {e[1]: e[2] for e in mock_st.events if e[0] == "metric"}
        self.assertEqual(metrics["Total Inventory Items"], "205")
        self.assertEqual(metrics["Total Sales ($)"], "$82,046,150.00")
        self.assertEqual(metrics["Total Hits"], "23,252")
        self.assertEqual(metrics["Reconciliation Status"], "PASSED")

        successes = [e[1] for e in mock_st.events if e[0] == "success"]
        self.assertTrue(any("205" in s and r"\$82,046,150.00" in s and "23,252" in s for s in successes))
        # Confirm NO unescaped $ in ABCD reconciliation message (prevents Streamlit KaTeX mangling)
        import re
        for msg in successes:
            non_code = re.sub(r"`[^`]*`", "", msg)
            self.assertEqual(re.findall(r"(?<!\\)\$", non_code), [], f"Unescaped $ in success banner: {msg}")

        # Confirm both summary table (4 rows) and detailed item-level table (205 rows x 15 cols) rendered cleanly
        dataframes = [e for e in mock_st.events if e[0] == "dataframe"]
        self.assertEqual(len(dataframes), 2, "Expected both ABCD summary table and detailed table.")
        summary_df, summary_kwargs = dataframes[0][1], dataframes[0][2]
        detail_df, detail_kwargs = dataframes[1][1], dataframes[1][2]
        self.assertEqual(len(summary_df), 4)
        self.assertEqual(list(summary_df["Final ABCD Class"]), ["A", "B", "C", "D"])
        self.assertEqual(len(detail_df), 205)
        self.assertEqual(len(detail_df.columns), 15)
        self.assertIn("Item Number", detail_kwargs.get("column_config", {}))

        infos = [e[1] for e in mock_st.events if e[0] == "info"]
        self.assertTrue(any("NOT calculated" in msg for msg in infos))

        downloads = [e[1] for e in mock_st.events if e[0] == "download_button"]
        self.assertEqual(len(downloads), 1)
        csv_text = downloads[0]["data"].decode("utf-8")
        csv_lines = [ln for ln in csv_text.strip().splitlines() if ln.strip()]
        self.assertEqual(len(csv_lines), 206)  # 1 header + 205 items

    def test_streamlit_handles_unrecognized_csv_schema_cleanly(self) -> None:
        import io
        from app import _render_csv_analysis_section

        bad_csv = io.BytesIO(b"Foo,Bar\n1,2\n")
        mock_st = self._make_mock_streamlit()
        _render_csv_analysis_section(mock_st, bad_csv)

        errors = [e[1] for e in mock_st.events if e[0] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("Multicriteria ABCD Classification", errors[0])
        self.assertIn("Standard Inventory Data Analysis", errors[0])

    def test_streamlit_markdown_escaping_and_eoq_purchase_label(self) -> None:
        import re
        from app import SAMPLE_QUESTIONS, _escape_streamlit_markdown

        # 1. Verify escaping helper handles $, Q*, TC(Q*), multiplication *, and preserves code & italics
        sample = (
            "Order cost is $150 and holding is $4; TC(0.98Q)=$3795.51 > TC(Q*)=$3794.73; "
            "`TOC = (12,000 / 948.68) * $150 = $1,897.37/year`; *Not calculated*"
        )
        escaped = _escape_streamlit_markdown(sample)
        self.assertIn(r"\$150", escaped)
        self.assertIn(r"\$4", escaped)
        self.assertIn(r"TC(Q\*)=\$3794.73", escaped)
        self.assertIn("`TOC = (12,000 / 948.68) * $150 = $1,897.37/year`", escaped)
        self.assertIn("*Not calculated*", escaped)
        non_code = re.sub(r"`[^`]*`", "", escaped)
        self.assertEqual(re.findall(r"(?<!\\)\$", non_code), [])

        # 2. Verify ALL 5 sample questions produce 0 unescaped $ outside code spans
        agent = InventoryDecisionAgent(parser_mode="rule")
        for name, q in SAMPLE_QUESTIONS.items():
            resp = agent.run(q)
            self.assertEqual(resp.status, "completed", f"Failed on {name}")
            rendered = _escape_streamlit_markdown(resp.explanation)
            outside = re.sub(r"`[^`]*`", "", rendered)
            self.assertEqual(
                re.findall(r"(?<!\\)\$", outside),
                [],
                f"Unescaped $ found in rendered explanation for {name}",
            )

        # 3. Verify EOQ without unit purchase price indicates purchase-inclusive cost was NOT calculated
        resp_no_price = agent.run(
            "Our annual demand is 12,000 units. Each order costs $150 to place, "
            "and holding cost is $4 per unit per year. What is the EOQ?"
        )
        self.assertEqual(resp_no_price.status, "completed")
        self.assertIn("Total Relevant Inventory Cost (Ordering + Holding):** **$3,794.73/year**", resp_no_price.explanation)
        self.assertIn("Not calculated (no unit purchase price provided)", resp_no_price.explanation)

        # 4. Verify EOQ WITH unit purchase price displays purchase-inclusive Total Annual Cost
        resp_with_price = agent.run(
            "Annual demand is 12,000 units, ordering cost is $150, unit price is $20, "
            "and holding rate is 20% of unit cost. Calculate EOQ."
        )
        self.assertEqual(resp_with_price.status, "completed")
        self.assertIn("Total Annual Cost (incl. Purchase):** **$243,794.73/year**", resp_with_price.explanation)
        self.assertNotIn("Not calculated", resp_with_price.explanation)

    def test_quantity_discount_feasibility_rule_and_whole_unit_recommendations(self) -> None:
        from app import SAMPLE_QUESTIONS

        agent = InventoryDecisionAgent(parser_mode="rule")

        # 1. Quantity Discount feasibility rule wording: evaluates upper feasible boundary (not automatically dominated)
        resp_qd = agent.run(SAMPLE_QUESTIONS["2. Quantity Discount"])
        self.assertEqual(resp_qd.status, "completed")
        rule_text = resp_qd.calculation["formula"]["feasibility_rule"]
        self.assertIn("upper feasible boundary", rule_text)
        self.assertIn("candidate Q = max_qty_j", rule_text)
        self.assertNotIn("dominated", rule_text.lower())
        self.assertAlmostEqual(resp_qd.calculation["optimal_order_quantity"], 1000.0, places=6)
        self.assertAlmostEqual(resp_qd.calculation["total_annual_cost"], 111900.0, places=2)

        # 2. EOQ exact values preserved + theoretical vs. practical whole-unit wording (948.68 -> 949)
        resp_eoq = agent.run(SAMPLE_QUESTIONS["1. Economic Order Quantity (EOQ)"])
        self.assertEqual(resp_eoq.status, "completed")
        self.assertAlmostEqual(resp_eoq.calculation["optimal_order_quantity"], 948.68329805, places=6)
        self.assertAlmostEqual(resp_eoq.calculation["total_relevant_cost"], 3794.73319220, places=6)
        self.assertIn(
            "The theoretical Economic Order Quantity (EOQ) is **948.68 units**",
            resp_eoq.explanation,
        )
        self.assertIn(
            "the practical operational recommendation is **949 units** per order.",
            resp_eoq.explanation,
        )

        # Verify EOQ recommendation is dynamic (not hardcoded) on a different fractional EOQ
        resp_eoq_dyn = agent.run(
            "Annual demand is 8,000 units, ordering cost is $90 per order, and holding cost is $5 per unit per year. Calculate EOQ."
        )
        self.assertEqual(resp_eoq_dyn.status, "completed")
        self.assertIn(
            "The theoretical Economic Order Quantity (EOQ) is **536.66 units**",
            resp_eoq_dyn.explanation,
        )
        self.assertIn(
            "the practical operational recommendation is **537 units** per order.",
            resp_eoq_dyn.explanation,
        )

        # Verify EOQ recommendation when exact EOQ is already a whole number (sqrt(2*5000*75/3) = 500)
        resp_eoq_int = agent.run(
            "Annual demand is 5,000 units, ordering cost is $75 per order, and holding cost is $3 per unit per year. Calculate EOQ."
        )
        self.assertEqual(resp_eoq_int.status, "completed")
        self.assertIn("The theoretical Economic Order Quantity (EOQ) is **500 units**", resp_eoq_int.explanation)
        self.assertIn("already a whole-unit order quantity", resp_eoq_int.explanation)

        # 3. Safety Stock & ROP exact values preserved + rounded-up whole-unit operational recommendation (65.79->66, 545.79->546)
        resp_rop = agent.run(SAMPLE_QUESTIONS["3. Safety Stock & Reorder Point"])
        self.assertEqual(resp_rop.status, "completed")
        self.assertAlmostEqual(resp_rop.calculation["safety_stock"], 65.79414508, places=6)
        self.assertAlmostEqual(resp_rop.calculation["reorder_point"], 545.79414508, places=6)
        self.assertIn("**65.79 units**", resp_rop.explanation)
        self.assertIn("**545.79 units**", resp_rop.explanation)
        self.assertIn("Practical Whole-Unit Operational Recommendation", resp_rop.explanation)
        self.assertIn("**66 units**", resp_rop.explanation)
        self.assertIn("**546 units**", resp_rop.explanation)

        # 4. Fixed-Period Review exact values preserved + rounded-up whole-unit operational recommendation (54.26->55, 904.26->905, 634.26->635)
        resp_fpr = agent.run(SAMPLE_QUESTIONS["4. Fixed-Period Review"])
        self.assertEqual(resp_fpr.status, "completed")
        self.assertAlmostEqual(resp_fpr.calculation["safety_stock"], 54.25524194, places=6)
        self.assertAlmostEqual(resp_fpr.calculation["target_inventory_level"], 904.25524194, places=6)
        self.assertAlmostEqual(resp_fpr.calculation["order_quantity"], 634.25524194, places=6)
        self.assertIn("**54.26 units**", resp_fpr.explanation)
        self.assertIn("**904.26 units**", resp_fpr.explanation)
        self.assertIn("**634.26 units**", resp_fpr.explanation)
        self.assertIn("Practical Whole-Unit Operational Recommendation", resp_fpr.explanation)
        self.assertIn("**55 units**", resp_fpr.explanation)
        self.assertIn("**905 units**", resp_fpr.explanation)
        self.assertIn("**635 units**", resp_fpr.explanation)

        # 5. Newsvendor remains unchanged with exact 191.93 / rounded 192 and expected profit $587.18
        resp_nv = agent.run(SAMPLE_QUESTIONS["5. Newsvendor"])
        self.assertEqual(resp_nv.status, "completed")
        self.assertAlmostEqual(resp_nv.calculation["critical_ratio"], 0.65454545, places=6)
        self.assertAlmostEqual(resp_nv.calculation["optimal_order_quantity"], 191.92864984, places=6)
        self.assertEqual(resp_nv.calculation["rounded_order_quantity"], 192)
        self.assertAlmostEqual(resp_nv.calculation["expected_profit"], 587.17776387, places=6)
        self.assertIn("Order **192 units**", resp_nv.explanation)

    def test_streamlit_section_headings_unnumbered_and_trace_numbered(self) -> None:
        import os

        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as fh:
            src = fh.read()

        # Verify updated unnumbered section headings in exact intended order
        self.assertIn('"Explanation & Recommendation"', src)
        self.assertIn('"Model, Inputs & Tool Details"', src)
        self.assertIn('"Verification Audit"', src)
        self.assertNotIn('"7. Explanation & Recommendation"', src)
        self.assertNotIn('"2–5. Model, Inputs & Tool Details"', src)
        self.assertNotIn('"6. Verification Audit"', src)

        idx_explain = src.index('"Explanation & Recommendation"')
        idx_details = src.index('"Model, Inputs & Tool Details"')
        idx_verify = src.index('"Verification Audit"')
        self.assertLess(idx_explain, idx_details)
        self.assertLess(idx_details, idx_verify)

        # Confirm the 7-step Pipeline Execution Trace retains its numbered steps
        agent = InventoryDecisionAgent(parser_mode="rule")
        resp = agent.run(
            "Our annual demand is 12,000 units. Each order costs $150 to place, and holding cost is $4 per unit per year."
        )
        stages = [step["stage"] for step in resp.pipeline_trace]
        self.assertEqual(len(stages), 7)
        for i, stage in enumerate(stages, start=1):
            self.assertTrue(stage.startswith(f"{i}. "), f"Expected numbered stage {i}, got {stage!r}")


if __name__ == "__main__":
    unittest.main()
