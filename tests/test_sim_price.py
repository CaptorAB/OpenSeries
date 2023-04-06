"""
Test suite for the openseries/sim_price.py module
"""
from typing import Dict, get_type_hints, List
from unittest import TestCase
from pandas import DataFrame

from openseries.sim_price import ReturnSimulation


class TestSimPrice(TestCase):
    """class to run unittests on the module sim_price.py"""

    def test_return_simulations_annotations_and_typehints(self: "TestSimPrice") -> None:
        """Test ReturnSimulation annotations and typehints"""
        returnsimulation_annotations = dict(ReturnSimulation.__annotations__)

        self.assertDictEqual(
            returnsimulation_annotations,
            {
                "number_of_sims": int,
                "trading_days": int,
                "trading_days_in_year": int,
                "mean_annual_return": float,
                "mean_annual_vol": float,
                "df": DataFrame,
            },
        )

        returnsimulation_typehints = get_type_hints(ReturnSimulation)
        self.assertDictEqual(returnsimulation_annotations, returnsimulation_typehints)

    def test_return_simulation_processes(self: "TestSimPrice") -> None:
        """Test ReturnSimulation based on different stochastic processes"""
        args: Dict[str, int | float] = {
            "number_of_sims": 1,
            "trading_days": 2520,
            "mean_annual_return": 0.05,
            "mean_annual_vol": 0.2,
            "seed": 71,
        }
        methods = [
            "from_normal",
            "from_lognormal",
            "from_gbm",
            "from_heston",
            "from_heston_vol",
            "from_merton_jump_gbm",
        ]
        added: List[Dict[str, int | float]] = [
            {},
            {},
            {},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"heston_mu": 0.35, "heston_a": 0.25},
            {"jumps_lamda": 0.00125, "jumps_sigma": 0.001, "jumps_mu": -0.2},
        ]
        target_returns = [
            "0.008917436",
            "0.029000099",
            "-0.011082564",
            "0.067119310",
            "0.101488620",
            "-0.007388824",
        ]
        target_volatilities = [
            "0.200429415",
            "0.200504640",
            "0.200429415",
            "0.263455329",
            "0.440520211",
            "0.210298179",
        ]

        returns = []
        volatilities = []
        for method, adding in zip(methods, added):
            arguments: Dict[str, int | float] = {**args, **adding}
            onesim = getattr(ReturnSimulation, method)(**arguments)
            returns.append(f"{onesim.realized_mean_return:.9f}")
            volatilities.append(f"{onesim.realized_vol:.9f}")

        self.assertListEqual(target_returns, returns)
        self.assertListEqual(target_volatilities, volatilities)

    def test_return_simulation_properties(self: "TestSimPrice") -> None:
        """Test ReturnSimulation properties output"""
        days = 1200
        psim = ReturnSimulation.from_normal(
            number_of_sims=1,
            trading_days=days,
            mean_annual_return=0.05,
            mean_annual_vol=0.1,
            seed=71,
        )

        self.assertIsInstance(psim.results, DataFrame)

        self.assertEqual(psim.results.shape[0], days)

        self.assertEqual(f"{psim.realized_mean_return:.9f}", "0.033493161")

        self.assertEqual(f"{psim.realized_vol:.9f}", "0.096596353")
