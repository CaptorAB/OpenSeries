"""Test suite for the openseries/frame.py module."""

# mypy: disable-error-code="operator,type-arg,arg-type,unused-ignore"
from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal, localcontext
from itertools import product as iter_product
from json import load, loads
from pathlib import Path
from pprint import pformat
from typing import Hashable, Optional, Union, cast
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, Series, date_range, read_excel
from pandas.testing import assert_frame_equal
from requests.exceptions import ConnectionError

# noinspection PyProtectedMember
from openseries._risk import _cvar_down_calc, _var_down_calc
from openseries.datefixer import date_offset_foll
from openseries.frame import (
    OpenFrame,
    constrain_optimized_portfolios,
    efficient_frontier,
    prepare_plot_data,
    sharpeplot,
    simulate_portfolios,
)
from openseries.load_plotly import load_plotly_dict
from openseries.series import OpenTimeSeries
from openseries.types import (
    LiteralFrameProps,
    LiteralPortfolioWeightings,
    ValueType,
)
from tests.test_common_sim import SEED, SIMFRAME, SIMSERIES


class TestOpenFrame(TestCase):

    """class to run unittests on the module frame.py."""

    randomframe: OpenFrame
    randomseries: OpenTimeSeries

    @classmethod
    def setUpClass(cls: type[TestOpenFrame]) -> None:
        """SetUpClass for the TestOpenFrame class."""
        cls.randomseries = SIMSERIES.from_deepcopy()
        cls.randomframe = SIMFRAME.from_deepcopy()

    def test_to_json(self: TestOpenFrame) -> None:
        """Test to_json method."""
        filename = "framesaved.json"
        if Path.home().joinpath("Documents").exists():
            directory = Path.home().joinpath("Documents")
            framefile = directory.joinpath(filename)
        else:
            directory = Path(__file__).resolve().parent
            framefile = directory.joinpath(filename)

        if Path(framefile).exists():
            msg = "test_to_json test case setup failed."
            raise FileExistsError(msg)

        kwargs = [
            {"what_output": "values", "filename": str(framefile)},
            {"what_output": "values", "filename": filename},
            {
                "what_output": "values",
                "filename": filename,
                "directory": directory,
            },
        ]

        for kwarg in kwargs:
            data = self.randomframe.to_json(**kwarg)  # type: ignore[arg-type]
            if [item.get("name") for item in data] != [
                "Asset_0",
                "Asset_1",
                "Asset_2",
                "Asset_3",
                "Asset_4",
            ]:
                msg = "Unexpected data from json"
                raise ValueError(msg)

            if not Path(framefile).exists():
                msg = "json file not created"
                raise FileNotFoundError(msg)

            framefile.unlink()

            if Path(framefile).exists():
                msg = "json file not deleted as intended"
                raise FileExistsError(msg)

        localfile = Path(__file__).resolve().parent.joinpath(filename)

        with patch("pathlib.Path.exists") as mock_doesnotexist:
            mock_doesnotexist.return_value = False
            data = self.randomframe.to_json(what_output="values", filename=filename)

        if [item.get("name") for item in data] != [
            "Asset_0",
            "Asset_1",
            "Asset_2",
            "Asset_3",
            "Asset_4",
        ]:
            msg = "Unexpected data from json"
            raise ValueError(msg)

        localfile.unlink()

        with patch("pathlib.Path.exists") as mock_doesnotexist, patch(
            "pathlib.Path.open",
        ) as mock_donotopen:
            mock_doesnotexist.return_value = True
            mock_donotopen.side_effect = MagicMock()
            data2 = self.randomframe.to_json(what_output="values", filename=filename)

        if [item.get("name") for item in data2] != [
            "Asset_0",
            "Asset_1",
            "Asset_2",
            "Asset_3",
            "Asset_4",
        ]:
            msg = "Unexpected data from json"
            raise ValueError(msg)

    def test_to_json_and_back(self: TestOpenFrame) -> None:
        """Test to_json method and creating an OpenFrame from file data."""
        filename = "frame.json"
        dirpath = Path(__file__).resolve().parent
        framefile = dirpath.joinpath(filename)

        if Path(framefile).exists():
            msg = "test_to_json_and_back test case setup failed."
            raise FileExistsError(msg)

        intended = ["1.640116", "1.354976", "2.287658", "2.045919", "1.169641"]

        data = self.randomframe.to_json(
            what_output="values",
            filename=filename,
            directory=dirpath,
        )

        frame_one = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name=item["name"],
                    dates=item["dates"],
                    values=item["values"],
                    valuetype=item["valuetype"],
                    baseccy=item["currency"],
                    local_ccy=item["local_ccy"],
                )
                for item in data
            ],
        ).to_cumret()

        check_one = [f"{endvalue:.6f}" for endvalue in frame_one.tsdf.iloc[-1]]

        if check_one != intended:
            msg = f"test_to_json_and_back did not output as intended: {check_one}"
            raise ValueError(msg)

        with Path.open(framefile, encoding="utf-8") as jsonfile:
            output = load(jsonfile)

        frame_two = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name=item["name"],
                    dates=item["dates"],
                    values=item["values"],
                    valuetype=item["valuetype"],
                    baseccy=item["currency"],
                    local_ccy=item["local_ccy"],
                )
                for item in output
            ],
        ).to_cumret()

        check_two = [f"{endvalue:.6f}" for endvalue in frame_two.tsdf.iloc[-1]]

        if check_two != intended:
            msg = f"test_to_json_and_back did not output as intended: {check_two}"
            raise ValueError(msg)

        framefile.unlink()

        if Path(framefile).exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

    def test_to_json_and_back_tsdf(self: TestOpenFrame) -> None:
        """Test to_json method and creating an OpenFrame from file data."""
        filename = "frame_tsdf.json"
        dirpath = Path(__file__).resolve().parent
        framefile = dirpath.joinpath(filename)

        if Path(framefile).exists():
            msg = "test_to_json_and_back_tsdf test case setup failed."
            raise FileExistsError(msg)

        intended = ["1.640116", "1.354976", "2.287658", "2.045919", "1.169641"]

        data = self.randomframe.to_json(
            what_output="tsdf",
            filename=filename,
            directory=dirpath,
        )

        frame_one = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name=item["name"],
                    dates=item["dates"],
                    values=item["values"],
                    valuetype=item["valuetype"],
                    baseccy=item["currency"],
                    local_ccy=item["local_ccy"],
                )
                for item in data
            ],
        ).to_cumret()

        check_one = [f"{endvalue:.6f}" for endvalue in frame_one.tsdf.iloc[-1]]

        if check_one != intended:
            msg = (
                "test_to_json_and_back_tsdf did "
                f"not output as intended: {check_one}"
            )
            raise ValueError(msg)

        with Path.open(framefile, encoding="utf-8") as jsonfile:
            output = load(jsonfile)

        frame_two = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name=item["name"],
                    dates=item["dates"],
                    values=item["values"],
                    valuetype=item["valuetype"],
                    baseccy=item["currency"],
                    local_ccy=item["local_ccy"],
                )
                for item in output
            ],
        ).to_cumret()

        check_two = [f"{endvalue:.6f}" for endvalue in frame_two.tsdf.iloc[-1]]

        if check_two != intended:
            msg = (
                "test_to_json_and_back_tsdf did "
                f"not output as intended: {check_two}"
            )
            raise ValueError(msg)

        framefile.unlink()

        if Path(framefile).exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

    def test_to_xlsx(self: TestOpenFrame) -> None:
        """Test to_xlsx method."""
        filename = "trial.xlsx"
        if Path.home().joinpath("Documents").exists():
            basefile = Path.home().joinpath("Documents").joinpath(filename)
        else:
            basefile = Path(__file__).resolve().parent.joinpath(filename)

        if Path(basefile).exists():
            msg = "test_save_to_xlsx test case setup failed."
            raise FileExistsError(msg)

        seriesfile = Path(
            self.randomframe.to_xlsx(filename=filename, sheet_title="boo"),
        ).resolve()

        if basefile != seriesfile:
            msg = "test_save_to_xlsx test case setup failed."
            raise ValueError(msg)

        if not Path(seriesfile).exists():
            msg = "xlsx file not created"
            raise FileNotFoundError(msg)

        seriesfile.unlink()

        directory = Path(__file__).resolve().parent
        seriesfile = Path(
            self.randomframe.to_xlsx(filename="trial.xlsx", directory=directory),
        ).resolve()

        if not Path(seriesfile).exists():
            msg = "xlsx file not created"
            raise FileNotFoundError(msg)

        seriesfile.unlink()

        if Path(seriesfile).exists():
            msg = "xlsx file not deleted as intended"
            raise FileExistsError(msg)

        with pytest.raises(
            expected_exception=NameError,
            match="Filename must end with .xlsx",
        ):
            _ = self.randomframe.to_xlsx(filename="trial.pdf")

        with Path.open(basefile, "w") as fakefile:
            fakefile.write("Hello world")

        with pytest.raises(
            expected_exception=FileExistsError,
            match=f"{filename} already exists.",
        ):
            _ = self.randomframe.to_xlsx(filename=filename, overwrite=False)

        basefile.unlink()

        localfile = Path(__file__).resolve().parent.joinpath(filename)
        with patch("pathlib.Path.exists") as mock_doesnotexist:
            mock_doesnotexist.return_value = False
            seriesfile = Path(self.randomframe.to_xlsx(filename=filename)).resolve()

        if localfile != seriesfile:
            msg = "test_save_to_xlsx test case setup failed."
            raise ValueError(msg)

        dframe = read_excel(
            io=seriesfile,
            header=0,
            index_col=0,
            usecols="A:F",
            skiprows=[1, 2],
            engine="openpyxl",
        )

        df_index = [dejt.date().strftime("%Y-%m-%d") for dejt in dframe.head().index]
        if df_index != [
            "2009-06-30",
            "2009-07-01",
            "2009-07-02",
            "2009-07-03",
            "2009-07-06",
        ]:
            msg = "save_to_xlsx not working as intended."
            raise ValueError(msg)

        seriesfile.unlink()

        with patch("pathlib.Path.exists") as mock_doesnotexist, patch(
            "openpyxl.workbook.workbook.Workbook.save",
        ) as mock_donotopen:
            mock_doesnotexist.return_value = True
            mock_donotopen.side_effect = MagicMock()
            seriesfile2 = Path(self.randomframe.to_xlsx(filename=filename)).resolve()

        if seriesfile2.parts[-2:] != ("Documents", "trial.xlsx"):
            msg = "save_to_xlsx not working as intended."
            raise ValueError(msg)

    def test_calc_range(self: TestOpenFrame) -> None:
        """Test calc_range method."""
        crframe = self.randomframe.from_deepcopy()
        start, end = (
            crframe.first_idx.strftime("%Y-%m-%d"),
            crframe.last_idx.strftime(
                "%Y-%m-%d",
            ),
        )
        rst, ren = crframe.calc_range()

        if [start, end] != [rst.strftime("%Y-%m-%d"), ren.strftime("%Y-%m-%d")]:
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Function calc_range returned earlier date < series start",
        ):
            _, _ = crframe.calc_range(months_offset=125)

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt date < series start",
        ):
            _, _ = crframe.calc_range(from_dt=dt.date(2009, 5, 31))

        with pytest.raises(
            expected_exception=ValueError,
            match="Given to_dt date > series end",
        ):
            _, _ = crframe.calc_range(to_dt=dt.date(2019, 7, 31))

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = crframe.calc_range(
                from_dt=dt.date(2009, 5, 31),
                to_dt=dt.date(2019, 7, 31),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = crframe.calc_range(
                from_dt=dt.date(2009, 7, 31),
                to_dt=dt.date(2019, 7, 31),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Given from_dt or to_dt dates outside series range",
        ):
            _, _ = crframe.calc_range(
                from_dt=dt.date(2009, 5, 31),
                to_dt=dt.date(2019, 5, 31),
            )

        nst, nen = crframe.calc_range(
            from_dt=dt.date(2009, 7, 3),
            to_dt=dt.date(2019, 6, 25),
        )
        if nst != dt.date(2009, 7, 3):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)
        if nen != dt.date(2019, 6, 25):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

        crframe.resample()

        earlier_moved, _ = crframe.calc_range(from_dt=dt.date(2009, 8, 10))
        if earlier_moved != dt.date(2009, 7, 31):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

        _, later_moved = crframe.calc_range(to_dt=dt.date(2009, 8, 20))
        if later_moved != dt.date(2009, 8, 31):
            msg = "Unintended output from calc_range()"
            raise ValueError(msg)

    def test_resample(self: TestOpenFrame) -> None:
        """Test resample method."""
        expected: int = 121
        rs_frame = self.randomframe.from_deepcopy()
        rs_frame.to_cumret()

        before = cast(Series, rs_frame.value_ret).to_dict()

        rs_frame.resample(freq="BME")

        if rs_frame.length != expected:
            msg = "resample() method generated unexpected result"
            raise ValueError(msg)

        after = cast(Series, rs_frame.value_ret).to_dict()
        if before != after:
            msg = "resample() method generated unexpected result"
            raise ValueError(msg)

    def test_resample_to_business_period_ends(self: TestOpenFrame) -> None:
        """Test resample_to_business_period_ends method."""
        rsb_stubs_frame = OpenFrame(
            [
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=121,
                    end_dt=dt.date(2023, 5, 15),
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=123,
                    end_dt=dt.date(2023, 5, 16),
                ).set_new_label("B"),
            ],
        )

        rsb_stubs_frame.resample_to_business_period_ends(freq="BME")
        new_stubs_dates = rsb_stubs_frame.tsdf.index.tolist()

        if new_stubs_dates != [
            dt.date(2023, 1, 15),
            dt.date(2023, 1, 31),
            dt.date(2023, 2, 28),
            dt.date(2023, 3, 31),
            dt.date(2023, 4, 28),
            dt.date(2023, 5, 15),
        ]:
            msg = (
                "resample_to_business_period_ends() method "
                "generated unexpected result"
            )
            raise ValueError(msg)

        rsb_frame = OpenFrame(
            [
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=88,
                    end_dt=dt.date(2023, 4, 28),
                ).set_new_label("A"),
                OpenTimeSeries.from_fixed_rate(
                    rate=0.01,
                    days=8,
                    end_dt=dt.date(2023, 4, 28),
                ).set_new_label("B"),
            ],
        )

        rsb_frame.resample_to_business_period_ends(freq="BME")
        new_dates = rsb_frame.tsdf.index.tolist()

        if new_dates != [
            dt.date(2023, 1, 31),
            dt.date(2023, 2, 28),
            dt.date(2023, 3, 31),
            dt.date(2023, 4, 28),
        ]:
            msg = (
                "resample_to_business_period_ends() method "
                "generated unexpected result"
            )
            raise ValueError(msg)

    def test_max_drawdown_date(self: TestOpenFrame) -> None:
        """Test max_drawdown_date method."""
        mddframe = self.randomframe.from_deepcopy()
        mddframe.to_cumret()

        mdates = cast(Series, mddframe.max_drawdown_date).tolist()

        checkdates = [
            dt.date(2012, 11, 21),
            dt.date(2019, 6, 11),
            dt.date(2015, 7, 24),
            dt.date(2010, 11, 19),
            dt.date(2011, 6, 28),
        ]

        if mdates != checkdates:
            msg = f"max_drawdown_date property generated unexpected result\n{mdates}"
            raise ValueError(msg)

    def test_make_portfolio(self: TestOpenFrame) -> None:
        """Test make_portfolio method."""
        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        mpframe.weights = [1.0 / mpframe.item_count] * mpframe.item_count

        name = "portfolio"
        mptail = mpframe.make_portfolio(name=name).tail()
        mptail = mptail.map(lambda nn: f"{nn:.6f}")

        correct = ["1.731448", "1.729862", "1.730238", "1.726204", "1.727963"]
        wrong = ["1.731448", "1.729862", "1.730238", "1.726204", "1.727933"]
        true_tail = DataFrame(
            columns=[[name], [ValueType.PRICE]],
            index=[
                dt.date(2019, 6, 24),
                dt.date(2019, 6, 25),
                dt.date(2019, 6, 26),
                dt.date(2019, 6, 27),
                dt.date(2019, 6, 28),
            ],
            data=correct,
            dtype="object",
        )
        false_tail = DataFrame(
            columns=[[name], [ValueType.PRICE]],
            index=[
                dt.date(2019, 6, 24),
                dt.date(2019, 6, 25),
                dt.date(2019, 6, 26),
                dt.date(2019, 6, 27),
                dt.date(2019, 6, 28),
            ],
            data=wrong,
            dtype="float64",
        )

        assert_frame_equal(true_tail, mptail, check_exact=True)

        with pytest.raises(expected_exception=AssertionError, match="are different"):
            assert_frame_equal(false_tail, mptail, check_exact=True)

        mpframe.weights = None
        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "OpenFrame weights property must be provided "
                "to run the make_portfolio method."
            ),
        ):
            _ = mpframe.make_portfolio(name=name)

    def test_make_portfolio_weight_strat(self: TestOpenFrame) -> None:
        """Test make_portfolio method with weight_strat."""
        mpframe = self.randomframe.from_deepcopy()
        mpframe.to_cumret()
        name = "portfolio"

        _ = mpframe.make_portfolio(name=name, weight_strat="eq_weights")
        weights: Optional[list[float]] = [0.2, 0.2, 0.2, 0.2, 0.2]
        if weights != mpframe.weights:
            msg = "make_portfolio() equal weight strategy not working as intended."
            raise ValueError(msg)

        with localcontext() as decimal_context:
            decimal_context.rounding = ROUND_HALF_UP

            _ = mpframe.make_portfolio(name=name, weight_strat="inv_vol")
            inv_vol_weights = [
                round(Decimal(wgt), 6) for wgt in cast(list[float], mpframe.weights)
            ]
            if inv_vol_weights != [
                Decimal("0.152977"),
                Decimal("0.206984"),
                Decimal("0.212791"),
                Decimal("0.214929"),
                Decimal("0.212319"),
            ]:
                msg = (
                    "make_portfolio() inverse vol strategy not working as intended."
                    f"output is \n{inv_vol_weights}"
                )
                raise ValueError(msg)

        with pytest.raises(
            expected_exception=NotImplementedError,
            match="Weight strategy not implemented",
        ):
            _ = mpframe.make_portfolio(
                name=name,
                weight_strat=cast(LiteralPortfolioWeightings, "bogus"),
            )

    def test_add_timeseries(self: TestOpenFrame) -> None:
        """Test add_timeseries method."""
        frameas = self.randomframe.from_deepcopy()
        items = int(frameas.item_count)
        frameas.weights = [1 / items] * items
        cols = list(frameas.columns_lvl_zero)
        nbr_cols = int(len(frameas.columns_lvl_zero))
        seriesas = self.randomseries.from_deepcopy()
        seriesas.set_new_label("Asset_6")
        frameas.add_timeseries(seriesas)

        if items + 1 != frameas.item_count:
            msg = "add_timeseries() method did not work as intended."
            raise ValueError(msg)
        if nbr_cols + 1 != len(frameas.columns_lvl_zero):
            msg = "add_timeseries() method did not work as intended."
            raise ValueError(msg)
        if [*cols, "Asset_6"] != frameas.columns_lvl_zero:
            msg = "add_timeseries() method did not work as intended."
            raise ValueError(msg)

    def test_delete_timeseries(self: TestOpenFrame) -> None:
        """Test delete_timeseries method."""
        frame = self.randomframe.from_deepcopy()
        frame.weights = [0.4, 0.1, 0.2, 0.1, 0.2]

        lbl = "Asset_1"
        frame.delete_timeseries(lbl)
        labels = [ff.label for ff in frame.constituents]

        if labels != ["Asset_0", "Asset_2", "Asset_3", "Asset_4"]:
            msg = "delete_timeseries() method did not work as intended."
            raise ValueError(msg)
        if frame.weights != [0.4, 0.2, 0.1, 0.2]:
            msg = "delete_timeseries() method did not work as intended."
            raise ValueError(msg)

    def test_risk_functions_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that risk measures align between OpenFrame and OpenTimeSeries."""
        riskseries = self.randomseries.from_deepcopy()
        riskseries.set_new_label(lvl_zero="Asset_0")
        riskframe = self.randomframe.from_deepcopy()
        riskseries.to_cumret()
        riskframe.to_cumret()

        if riskseries.cvar_down != _cvar_down_calc(
            riskseries.tsdf.iloc[:, 0].tolist(),
        ):
            msg = "CVaR for OpenTimeSeries not equal"
            raise ValueError(msg)
        if riskseries.var_down != _var_down_calc(riskseries.tsdf.iloc[:, 0].tolist()):
            msg = "VaR for OpenTimeSeries not equal"
            raise ValueError(msg)

        if cast(Series, riskframe.cvar_down).iloc[0] != _cvar_down_calc(
            riskframe.tsdf.iloc[:, 0],
        ):
            msg = "CVaR for OpenFrame not equal"
            raise ValueError(msg)
        if cast(Series, riskframe.var_down).iloc[0] != _var_down_calc(
            riskframe.tsdf.iloc[:, 0],
        ):
            msg = "VaR for OpenFrame not equal"
            raise ValueError(msg)

        if cast(Series, riskframe.cvar_down).iloc[0] != _cvar_down_calc(
            riskframe.tsdf,
        ):
            msg = "CVaR for OpenFrame not equal"
            raise ValueError(msg)
        if cast(Series, riskframe.var_down).iloc[0] != _var_down_calc(
            riskframe.tsdf,
        ):
            msg = "VaR for OpenFrame not equal"
            raise ValueError(msg)

    def test_methods_same_as_opentimeseries(self: TestOpenFrame) -> None:
        """Test that method results align between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.value_to_ret()
        sameframe = self.randomframe.from_deepcopy()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        methods = [
            "rolling_return",
            "rolling_vol",
            "rolling_var_down",
            "rolling_cvar_down",
        ]
        for method in methods:
            assert_frame_equal(
                getattr(sameseries, method)(),
                getattr(sameframe, method)(column=0),
            )

        cumseries = sameseries.from_deepcopy()
        cumframe = sameframe.from_deepcopy()

        cumseries.value_to_log()
        cumframe.value_to_log()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.value_to_ret()
        sameframe.value_to_ret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.resample()
        sameframe.resample()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

        sameseries.value_to_diff()
        sameframe.value_to_diff()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf.iloc[:, 0].to_frame())

    def test_calc_methods_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that calc results align between OpenFrame and OpenTimeSeries."""
        sames = self.randomseries.from_deepcopy()
        sames.to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = self.randomframe.from_deepcopy()
        for serie in samef.constituents:
            serie.to_cumret()
        samef.to_cumret()

        methods_to_compare = [
            "arithmetic_ret_func",
            "cvar_down_func",
            "downside_deviation_func",
            "geo_ret_func",
            "kurtosis_func",
            "max_drawdown_func",
            "positive_share_func",
            "skew_func",
            "vol_from_var_func",
            "target_weight_from_var",
            "value_ret_func",
            "var_down_func",
            "vol_func",
            "worst_func",
            "z_score_func",
        ]
        for method in methods_to_compare:
            if (
                f"{getattr(sames, method)(months_from_last=12):.11f}"
                != f"{float(getattr(samef, method)(months_from_last=12).iloc[0]):.11f}"
            ):
                msg = (
                    f"Calc method {method} not aligned between "
                    "OpenTimeSeries and OpenFrame"
                )
                raise ValueError(msg)

    def test_ratio_methods_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that ratio methods align between OpenFrame and OpenTimeSeries."""
        sames = self.randomseries.from_deepcopy()
        sames.to_cumret()
        sames.set_new_label(lvl_zero="Asset_0")
        samef = self.randomframe.from_deepcopy()
        samef.to_cumret()

        smf_vrf = cast(
            Series,
            samef.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12),
        ).iloc[0]
        if (
            f"{sames.ret_vol_ratio_func(riskfree_rate=0.0, months_from_last=12):.11f}"
            != f"{smf_vrf:.11f}"
        ):
            msg = (
                "ret_vol_ratio_func() not aligned between "
                "OpenTimeSeries and OpenFrame"
            )
            raise ValueError(msg)

        smf_srf = cast(
            Series,
            samef.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12),
        ).iloc[0]
        if (
            f"{sames.sortino_ratio_func(riskfree_rate=0.0, months_from_last=12):.11f}"
            != f"{smf_srf:.11f}"
        ):
            msg = (
                "sortino_ratio_func() not aligned between "
                "OpenTimeSeries and OpenFrame"
            )
            raise ValueError(msg)

    def test_measures_same_as_opentimeseries(self: TestOpenFrame) -> None:
        """Test that measure results align between OpenFrame and OpenTimeSeries."""
        frame_0 = self.randomframe.from_deepcopy()
        for zerie in frame_0.constituents:
            zerie.to_cumret()
        frame_0.to_cumret()

        common_calc_props = [
            "arithmetic_ret",
            "cvar_down",
            "downside_deviation",
            "geo_ret",
            "kurtosis",
            "max_drawdown",
            "max_drawdown_cal_year",
            "positive_share",
            "ret_vol_ratio",
            "skew",
            "sortino_ratio",
            "value_ret",
            "var_down",
            "vol",
            "vol_from_var",
            "worst",
            "worst_month",
            "z_score",
        ]

        for prop in common_calc_props:
            result = getattr(frame_0, prop).tolist()
            rounded = [f"{item:.10f}" for item in result]
            roundmeasure = [
                f"{getattr(serie, prop):.10f}" for serie in frame_0.constituents
            ]
            if rounded != roundmeasure:
                msg = (
                    f"Property {prop} not aligned between "
                    "OpenTimeSeries and OpenFrame"
                )
                raise ValueError(msg)

    def test_properties_same_as_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that property results align between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        common_props_to_compare = ["periods_in_a_year", "yearfrac"]
        for comnprop in common_props_to_compare:
            if getattr(sameseries, comnprop) != getattr(sameframe, comnprop):
                msg = (
                    f"Property {comnprop} not aligned between "
                    "OpenTimeSeries and OpenFrame"
                )
                raise ValueError(msg)

    def test_keeping_attributes_aligned_vs_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that attributes are aligned between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        common_calc_props = [
            "arithmetic_ret",
            "cvar_down",
            "downside_deviation",
            "geo_ret",
            "kurtosis",
            "max_drawdown",
            "max_drawdown_cal_year",
            "omega_ratio",
            "positive_share",
            "ret_vol_ratio",
            "skew",
            "sortino_ratio",
            "value_ret",
            "var_down",
            "vol",
            "vol_from_var",
            "worst",
            "worst_month",
            "z_score",
        ]

        common_props = ["periods_in_a_year", "yearfrac", "max_drawdown_date"]

        common_attributes = [
            "length",
            "span_of_days",
            "first_idx",
            "last_idx",
            "tsdf",
        ]

        series_attributes = [
            "values",
            "local_ccy",
            "timeseries_id",
            "instrument_id",
            "currency",
            "isin",
            "dates",
            "name",
            "valuetype",
            "label",
            "domestic",
            "countries",
        ]

        pydantic_basemodel_attributes = [
            "model_extra",
            "model_fields",
            "model_config",
            "model_computed_fields",
            "model_fields_set",
        ]

        frame_attributes = [
            "constituents",
            "columns_lvl_zero",
            "columns_lvl_one",
            "item_count",
            "weights",
            "first_indices",
            "last_indices",
            "lengths_of_items",
            "span_of_days_all",
        ]

        frame_calc_props = ["correl_matrix"]

        series_props = [
            a
            for a in dir(sameseries)
            if not a.startswith("_") and not callable(getattr(sameseries, a))
        ]
        series_compared = set(series_props).symmetric_difference(
            set(
                common_calc_props
                + common_props
                + common_attributes
                + pydantic_basemodel_attributes
                + series_attributes,
            ),
        )
        if len(series_compared) != 0:
            msg = f"Difference is: {series_compared}"
            raise ValueError(msg)

        frame_props = [
            a
            for a in dir(sameframe)
            if not a.startswith("_") and not callable(getattr(sameframe, a))
        ]
        frame_compared = set(frame_props).symmetric_difference(
            set(
                common_calc_props
                + common_props
                + common_attributes
                + pydantic_basemodel_attributes
                + frame_attributes
                + frame_calc_props,
            ),
        )
        if len(frame_compared) != 0:
            msg = f"Difference is: {frame_compared}"
            raise ValueError(msg)

    def test_keeping_methods_aligned_vs_opentimeseries(
        self: TestOpenFrame,
    ) -> None:
        """Test that methods are aligned between OpenFrame and OpenTimeSeries."""
        sameseries = self.randomseries.from_deepcopy()
        sameseries.to_cumret()
        sameframe = self.randomframe.from_deepcopy()
        sameframe.to_cumret()

        pydantic_basemodel_methods = [
            "dict",
            "copy",
            "model_post_init",
            "model_rebuild",
            "model_dump_json",
            "model_validate_json",
            "model_validate_strings",
            "model_copy",
            "model_dump",
            "model_validate",
            "model_construct",
            "model_parametrized_name",
            "model_json_schema",
            "parse_obj",
            "update_forward_refs",
            "parse_file",
            "schema",
            "construct",
            "schema_json",
            "parse_raw",
            "from_orm",
            "json",
            "validate",
        ]

        common_calc_methods = [
            "arithmetic_ret_func",
            "cvar_down_func",
            "downside_deviation_func",
            "geo_ret_func",
            "kurtosis_func",
            "max_drawdown_func",
            "omega_ratio_func",
            "positive_share_func",
            "ret_vol_ratio_func",
            "skew_func",
            "sortino_ratio_func",
            "target_weight_from_var",
            "value_ret_func",
            "var_down_func",
            "vol_from_var_func",
            "vol_func",
            "worst_func",
            "z_score_func",
        ]

        common_methods = [
            "align_index_to_local_cdays",
            "all_properties",
            "calc_range",
            "from_deepcopy",
            "plot_bars",
            "plot_series",
            "resample",
            "resample_to_business_period_ends",
            "return_nan_handle",
            "rolling_return",
            "rolling_vol",
            "rolling_cvar_down",
            "rolling_var_down",
            "to_cumret",
            "to_drawdown_series",
            "to_json",
            "to_xlsx",
            "value_nan_handle",
            "value_ret_calendar_period",
            "value_to_diff",
            "value_to_log",
            "value_to_ret",
        ]

        series_createmethods = [
            "from_arrays",
            "from_df",
            "from_fixed_rate",
        ]

        series_unique = [
            "ewma_vol_func",
            "from_1d_rate_to_cumret",
            "pandas_df",
            "running_adjustment",
            "set_new_label",
            "setup_class",
            "dates_and_values_validate",
        ]

        frame_unique = [
            "add_timeseries",
            "beta",
            "delete_timeseries",
            "ewma_risk",
            "rolling_info_ratio",
            "info_ratio_func",
            "jensen_alpha",
            "tracking_error_func",
            "capture_ratio_func",
            "ord_least_squares_fit",
            "make_portfolio",
            "merge_series",
            "relative",
            "rolling_corr",
            "rolling_beta",
            "trunc_frame",
        ]

        series_methods = [
            a
            for a in dir(sameseries)
            if not a.startswith("_") and callable(getattr(sameseries, a))
        ]
        series_compared = set(series_methods).symmetric_difference(
            set(
                pydantic_basemodel_methods
                + common_calc_methods
                + common_methods
                + series_createmethods
                + series_unique,
            ),
        )
        if len(series_compared) != 0:
            msg = f"Difference is: {series_compared}"
            raise ValueError(msg)

        frame_methods = [
            a
            for a in dir(sameframe)
            if not a.startswith("_") and callable(getattr(sameframe, a))
        ]
        frame_compared = set(frame_methods).symmetric_difference(
            set(
                pydantic_basemodel_methods
                + common_calc_methods
                + common_methods
                + frame_unique,
            ),
        )
        if len(frame_compared) != 0:
            msg = f"Difference is: {frame_compared}"
            raise ValueError(msg)

    def test_value_to_log(self: TestOpenFrame) -> None:
        """Test value_to_log method."""
        logframe = self.randomframe.from_deepcopy()
        logframe.to_cumret()

        aaframe = logframe.tsdf.map(lambda item: f"{item:.12f}")
        bbdict = aaframe.to_dict(orient="list")
        b4_log = [bbdict[key] for key in bbdict]

        logframe.value_to_log()

        ccframe = logframe.tsdf.map(lambda item: f"{item:.12f}")
        eedict = ccframe.to_dict(orient="list")
        middle_log = [eedict[key] for key in eedict]

        if b4_log == middle_log:
            msg = "Method value_to_log() did not work as intended."
            raise ValueError(msg)

    def test_correl_matrix(self: TestOpenFrame) -> None:
        """Test correl_matrix method."""
        corrframe = self.randomframe.from_deepcopy()
        corrframe.to_cumret()
        dict1 = corrframe.correl_matrix.map(lambda nn: f"{nn:.10f}").to_dict()

        dict2 = {
            "Asset_0": {
                "Asset_0": "1.0000000000",
                "Asset_1": "0.0239047199",
                "Asset_2": "-0.0077164998",
                "Asset_3": "0.0212300816",
                "Asset_4": "-0.0432032137",
            },
            "Asset_1": {
                "Asset_0": "0.0239047199",
                "Asset_1": "1.0000000000",
                "Asset_2": "-0.0027966077",
                "Asset_3": "-0.0275509319",
                "Asset_4": "-0.0078836695",
            },
            "Asset_2": {
                "Asset_0": "-0.0077164998",
                "Asset_1": "-0.0027966077",
                "Asset_2": "1.0000000000",
                "Asset_3": "-0.0138919271",
                "Asset_4": "0.0103654979",
            },
            "Asset_3": {
                "Asset_0": "0.0212300816",
                "Asset_1": "-0.0275509319",
                "Asset_2": "-0.0138919271",
                "Asset_3": "1.0000000000",
                "Asset_4": "0.0264748547",
            },
            "Asset_4": {
                "Asset_0": "-0.0432032137",
                "Asset_1": "-0.0078836695",
                "Asset_2": "0.0103654979",
                "Asset_3": "0.0264748547",
                "Asset_4": "1.0000000000",
            },
        }

        if dict1 != dict2:
            msg = f"Unexpected result(s) from method correl_matrix()\n{pformat(dict1)}"
            raise ValueError(msg)

    def test_plot_series(self: TestOpenFrame) -> None:
        """Test plot_series method."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
        fig_json = loads(cast(str, fig.to_json()))

        for i in range(plotframe.item_count):
            rawdata = [f"{x:.11f}" for x in plotframe.tsdf.iloc[1:5, i]]
            fig_data = [f"{x:.11f}" for x in fig_json["data"][i]["y"][1:5]]
            if rawdata != fig_data:
                msg = "Unaligned data between original and data in Figure."
                raise ValueError(msg)

        fig_last, _ = plotframe.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
        )
        fig_last_json = loads(cast(str, fig_last.to_json()))
        rawlast = plotframe.tsdf.iloc[-1, -1]
        figlast = fig_last_json["data"][-1]["y"][0]
        if f"{figlast:.12f}" != f"{rawlast:.12f}":
            msg = "Unaligned data between original and data in Figure."
            raise ValueError(msg)

        fig_last_fmt, _ = plotframe.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
            tick_fmt=".3%",
        )
        fig_last_fmt_json = loads(cast(str, fig_last_fmt.to_json()))
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]

        if last_fmt != "Last 116.964%":
            msg = f"Unaligned data in Figure: '{last_fmt}'"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Must provide same number of labels as items in frame.",
        ):
            _, _ = plotframe.plot_series(auto_open=False, labels=["a", "b"])

        _, logo = load_plotly_dict()

        fig_logo, _ = plotframe.plot_series(
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(cast(str, fig_logo.to_json()))

        if logo == {}:
            if fig_logo_json["layout"]["images"][0] != logo:
                msg = "plot_series add_logo argument not setup correctly"
                raise ValueError(msg)
        elif fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "plot_series add_logo argument not setup correctly"
            raise ValueError(msg)

        fig_nologo, _ = plotframe.plot_series(
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(cast(str, fig_nologo.to_json()))
        if fig_nologo_json["layout"].get("images", None):
            msg = "plot_series add_logo argument not setup correctly"
            raise ValueError(msg)

    def test_plot_series_filefolders(self: TestOpenFrame) -> None:
        """Test plot_series method with different file folder options."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        directory = Path(__file__).resolve().parent
        _, figfile = plotframe.plot_series(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile[:5] == "<div>":
            msg = "plot_series method not working as intended"
            raise ValueError(msg)

        _, divstring = plotframe.plot_series(auto_open=False, output_type="div")
        if divstring[:5] != "<div>" or divstring[-6:] != "</div>":
            msg = "Html div section not created"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = True
            mockhomefig, _ = plotframe.plot_series(
                auto_open=False,
                output_type="div",
            )
            mockhomefig_json = loads(cast(str, mockhomefig.to_json()))

        if mockhomefig_json["data"][0]["name"] != "Asset_0":
            msg = "plot_series method not working as intended"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = False
            _, mockfile = plotframe.plot_series(
                filename="seriesfile.html",
                auto_open=False,
            )
            mockfilepath = Path(mockfile).resolve()

        if mockfilepath.parts[-2:] != ("tests", "seriesfile.html"):
            msg = "plot_series method not working as intended"
            raise ValueError(msg)

        mockfilepath.unlink()

    def test_plot_bars(self: TestOpenFrame) -> None:
        """Test plot_bars method."""
        plotframe = self.randomframe.from_deepcopy()

        fig_keys = ["hovertemplate", "name", "type", "x", "y"]
        fig, _ = plotframe.plot_bars(auto_open=False, output_type="div")
        fig_json = loads(cast(str, fig.to_json()))
        made_fig_keys = list(fig_json["data"][0].keys())
        made_fig_keys.sort()
        if made_fig_keys != fig_keys:
            msg = "Data in Figure not as intended."
            raise ValueError(msg)

        for i in range(plotframe.item_count):
            rawdata = [f"{x:.11f}" for x in plotframe.tsdf.iloc[1:5, i]]
            fig_data = [f"{x:.11f}" for x in fig_json["data"][i]["y"][1:5]]
            if rawdata != fig_data:
                msg = "Unaligned data between original and data in Figure."
                raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Must provide same number of labels as items in frame.",
        ):
            _, _ = plotframe.plot_bars(auto_open=False, labels=["a", "b"])

        overlayfig, _ = plotframe.plot_bars(
            auto_open=False,
            output_type="div",
            mode="overlay",
        )
        overlayfig_json = loads(cast(str, overlayfig.to_json()))

        fig_keys.append("opacity")
        if sorted(overlayfig_json["data"][0].keys()) != sorted(fig_keys):
            msg = "Data in Figure not as intended."
            raise ValueError(msg)

        _, logo = load_plotly_dict()

        fig_logo, _ = plotframe.plot_bars(
            auto_open=False,
            add_logo=True,
            output_type="div",
        )
        fig_logo_json = loads(cast(str, fig_logo.to_json()))

        if logo == {}:
            if fig_logo_json["layout"]["images"][0] != logo:
                msg = "plot_bars add_logo argument not setup correctly"
                raise ValueError(msg)
        elif fig_logo_json["layout"]["images"][0]["source"] != logo["source"]:
            msg = "plot_bars add_logo argument not setup correctly"
            raise ValueError(msg)

        fig_nologo, _ = plotframe.plot_bars(
            auto_open=False,
            add_logo=False,
            output_type="div",
        )
        fig_nologo_json = loads(cast(str, fig_nologo.to_json()))
        if fig_nologo_json["layout"].get("images", None):
            msg = "plot_bars add_logo argument not setup correctly"
            raise ValueError(msg)

    def test_plot_bars_filefolders(self: TestOpenFrame) -> None:
        """Test plot_bars method with different file folder options."""
        plotframe = self.randomframe.from_deepcopy()

        directory = Path(__file__).resolve().parent
        _, figfile = plotframe.plot_bars(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile[:5] == "<div>":
            msg = "plot_bars method not working as intended"
            raise ValueError(msg)

        _, divstring = plotframe.plot_bars(auto_open=False, output_type="div")
        if divstring[:5] != "<div>" or divstring[-6:] != "</div>":
            msg = "Html div section not created"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = True
            mockhomefig, _ = plotframe.plot_bars(
                auto_open=False,
                output_type="div",
            )
            mockhomefig_json = loads(cast(str, mockhomefig.to_json()))

        if mockhomefig_json["data"][0]["name"] != "Asset_0":
            msg = "plot_bars method not working as intended"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = False
            _, mockfile = plotframe.plot_bars(
                filename="barfile.html",
                auto_open=False,
            )
            mockfilepath = Path(mockfile).resolve()

        if mockfilepath.parts[-2:] != ("tests", "barfile.html"):
            msg = "plot_bars method not working as intended"
            raise ValueError(msg)

        mockfilepath.unlink()

    def test_plot_methods_mock_logo_url_fail(self: TestOpenFrame) -> None:
        """Test plot_series and plot_bars methods with mock logo file URL fail."""
        plotframe = self.randomframe.from_deepcopy()
        plotframe.to_cumret()

        with patch("requests.head") as mock_conn_error:
            mock_conn_error.side_effect = ConnectionError()

            seriesfig, _ = plotframe.plot_series(
                auto_open=False,
                add_logo=True,
                output_type="div",
            )
            seriesfig_json = loads(cast(str, seriesfig.to_json()))
            if seriesfig_json["layout"]["images"][0].get("source", None):
                msg = "plot_series add_logo argument not setup correctly"
                raise ValueError(msg)

            barfig, _ = plotframe.plot_bars(
                auto_open=False,
                add_logo=True,
                output_type="div",
            )
            barfig_json = loads(cast(str, barfig.to_json()))
            if barfig_json["layout"]["images"][0].get("source", None):
                msg = "plot_bars add_logo argument not setup correctly"
                raise ValueError(msg)

            with self.assertLogs() as plotseries_context:
                _, _ = plotframe.plot_series(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotseries_context.output[0]
            ):
                msg = (
                    "plot_series() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

            with self.assertLogs() as plotbars_context:
                _, _ = plotframe.plot_bars(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotbars_context.output[0]
            ):
                msg = (
                    "plot_bars() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

        with patch("requests.head") as mock_statuscode:
            mock_statuscode.return_value.status_code = 400

            with self.assertLogs() as plotseries_context:
                _, _ = plotframe.plot_series(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotseries_context.output[0]
            ):
                msg = (
                    "plot_series() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

            with self.assertLogs() as plotbars_context:
                _, _ = plotframe.plot_bars(auto_open=False, output_type="div")
            if (
                "WARNING:root:Failed to add logo image from URL"
                not in plotbars_context.output[0]
            ):
                msg = (
                    "plot_bars() method did not warn as "
                    "expected when logo URL not working"
                )
                raise ValueError(msg)

        with patch("requests.head") as mock_statuscode:
            mock_statuscode.return_value.status_code = 200

            fig, _ = plotframe.plot_series(auto_open=False, output_type="div")
            fig_json = loads(cast(str, fig.to_json()))

            for i in range(plotframe.item_count):
                rawdata = [f"{x:.11f}" for x in plotframe.tsdf.iloc[1:5, i]]
                fig_data = [f"{x:.11f}" for x in fig_json["data"][i]["y"][1:5]]
                if rawdata != fig_data:
                    msg = "Unaligned data between original and data in Figure."
                    raise ValueError(msg)

    def test_passed_empty_list(self: TestOpenFrame) -> None:
        """Test warning on object construct with empty list."""
        with self.assertLogs() as contextmgr:
            OpenFrame([])
        if contextmgr.output != ["WARNING:root:OpenFrame() was passed an empty list."]:
            msg = "OpenFrame failed to log warning about empty input list."
            raise ValueError(msg)

    def test_trunc_frame_both(self: TestOpenFrame) -> None:
        """Test trunc_frame method."""
        series_long = self.randomseries.from_deepcopy()
        series_long.set_new_label("Long")
        tmp_series = self.randomseries.from_deepcopy()
        series_short = OpenTimeSeries.from_df(
            tmp_series.tsdf.loc[
                cast(int, dt.date(2017, 6, 27)) : cast(  # type: ignore[index]
                    int,
                    dt.date(2018, 6, 27),
                ),
                ("Asset_0", ValueType.PRICE),
            ],
        )
        series_short.set_new_label("Short")
        frame = OpenFrame([series_long, series_short])

        firsts = [
            dt.date(2017, 6, 27),
            dt.date(2017, 6, 27),
        ]
        lasts = [
            dt.date(2018, 6, 27),
            dt.date(2018, 6, 27),
        ]

        if firsts == frame.first_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise ValueError(msg)
        if lasts == frame.last_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise ValueError(msg)

        frame.trunc_frame()

        if firsts != frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)
        if lasts != frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

        trunced = [dt.date(2017, 12, 29), dt.date(2018, 3, 29)]

        frame.trunc_frame(start_cut=trunced[0], end_cut=trunced[1])

        if trunced != [frame.first_idx, frame.last_idx]:
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

    def test_trunc_frame_before_after(self: TestOpenFrame) -> None:
        """Test trunc_frame method."""
        series_long = self.randomseries.from_deepcopy()
        series_long.set_new_label("Long")
        tmp_series = self.randomseries.from_deepcopy()
        series_short = OpenTimeSeries.from_df(
            tmp_series.tsdf.loc[
                cast(int, dt.date(2017, 6, 27)) : cast(  # type: ignore[index]
                    int,
                    dt.date(2018, 6, 27),
                ),
                ("Asset_0", ValueType.PRICE),
            ],
        )
        series_short.set_new_label("Short")
        frame = OpenFrame([series_long, series_short])

        firsts = [
            dt.date(2017, 6, 27),
            dt.date(2017, 6, 27),
        ]
        lasts = [
            dt.date(2018, 6, 27),
            dt.date(2018, 6, 27),
        ]

        if firsts == frame.first_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise ValueError(msg)
        if lasts == frame.last_indices.tolist():
            msg = "trunc_frame() test not set up as intended."
            raise ValueError(msg)

        before_frame = frame.from_deepcopy()
        after_frame = frame.from_deepcopy()

        before_frame.trunc_frame(where="before")

        if firsts != before_frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)
        if lasts == before_frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

        after_frame.trunc_frame(where="after")

        if firsts == after_frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)
        if lasts != after_frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

        before_frame.trunc_frame(where="after")
        after_frame.trunc_frame(where="before")

        if firsts != after_frame.first_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)
        if lasts != before_frame.last_indices.tolist():
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

    def test_trunc_frame_start_fail(self: TestOpenFrame) -> None:
        """Test trunc_frame method start fail scenario."""
        frame = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["a"],
                        data=[1, 2, 3, 4, 5],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["b"],
                        data=[6, 7, 8, 9, 10],
                        index=[
                            "2022-10-01",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["c"],
                        data=[11, 12, 13, 14, 15],
                        index=[
                            "2022-10-02",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["d"],
                        data=[16, 17, 18, 19, 20],
                        index=[
                            "2022-10-01",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-06",
                            "2022-10-07",
                        ],
                    ),
                ),
            ],
        )
        with self.assertLogs("root", level="WARNING") as logs:
            frame.trunc_frame()
        if (
            "WARNING:root:One or more constituents "
            "still not truncated to same start dates."
        ) not in logs.output[0]:
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

    def test_trunc_frame_end_fail(self: TestOpenFrame) -> None:
        """Test trunc_frame method end fail scenario."""
        frame = OpenFrame(
            [
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["a"],
                        data=[1, 2, 3, 4, 5],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["b"],
                        data=[6, 7, 8, 9],
                        index=[
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["c"],
                        data=[10, 11, 12, 13, 14],
                        index=[
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                            "2022-10-07",
                        ],
                    ),
                ),
                OpenTimeSeries.from_df(
                    dframe=DataFrame(
                        columns=["d"],
                        data=[15, 16, 17, 18, 19],
                        index=[
                            "2022-10-01",
                            "2022-10-02",
                            "2022-10-03",
                            "2022-10-04",
                            "2022-10-05",
                        ],
                    ),
                ),
            ],
        )
        with self.assertLogs("root", level="WARNING") as logs:
            frame.trunc_frame()
        if (
            "WARNING:root:One or more constituents "
            "still not truncated to same end dates."
        ) not in logs.output[0]:
            msg = "Method trunc_frame() did not work as intended."
            raise ValueError(msg)

    def test_merge_series(self: TestOpenFrame) -> None:
        """Test merge_series method."""
        aframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="Asset_one",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
                OpenTimeSeries.from_arrays(
                    name="Asset_two",
                    dates=[
                        "2022-08-11",
                        "2022-08-12",
                        "2022-08-13",
                        "2022-08-14",
                        "2022-08-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
            ],
        )
        bframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="Asset_one",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
                OpenTimeSeries.from_arrays(
                    name="Asset_two",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-16",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
            ],
        )

        oldlabels = list(aframe.columns_lvl_zero)
        aframe.tsdf = aframe.tsdf.rename(
            columns={
                "Asset_one": "Asset_three",
                "Asset_two": "Asset_four",
            },
            level=0,
        )
        newlabels = list(aframe.columns_lvl_zero)

        b4df = aframe.tsdf.copy()

        if oldlabels == newlabels:
            msg = "Setup to test merge_series() did not work as intended."
            raise ValueError(msg)

        aframe.merge_series(how="outer")
        labelspostmerge = list(aframe.columns_lvl_zero)

        assert_frame_equal(b4df, aframe.tsdf, check_exact=True)

        if newlabels != labelspostmerge:
            msg = "Method merge_series() did not work as intended."
            raise ValueError(msg)

        bframe.merge_series(how="inner")
        blist = [d.strftime("%Y-%m-%d") for d in bframe.tsdf.index]
        if blist != [
            "2022-07-11",
            "2022-07-12",
            "2022-07-13",
            "2022-07-14",
        ]:
            msg = "Method merge_series() did not work as intended."
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=Exception,
            match=(
                "Merging OpenTimeSeries DataFrames with argument "
                "how=inner produced an empty DataFrame."
            ),
        ):
            aframe.merge_series(how="inner")

    def test_all_properties(self: TestOpenFrame) -> None:
        """Test all_properties method."""
        prop_index = [
            "Simple return",
            "Geometric return",
            "Arithmetic return",
            "Volatility",
            "Downside deviation",
            "Return vol ratio",
            "Sortino ratio",
            "Omega ratio",
            "Z-score",
            "Skew",
            "Kurtosis",
            "Positive share",
            "VaR 95.0%",
            "CVaR 95.0%",
            "Imp vol from VaR 95%",
            "Worst",
            "Worst month",
            "Max drawdown",
            "Max drawdown date",
            "Max drawdown in cal yr",
            "first indices",
            "last indices",
            "observations",
            "span of days",
        ]
        apframe = self.randomframe.from_deepcopy()
        apframe.to_cumret()
        result = apframe.all_properties()
        result_index = result.index.tolist()

        if not isinstance(result, DataFrame):
            msg = "Method all_properties() not working as intended."
            raise TypeError(msg)

        result_arg = apframe.all_properties(
            properties=cast(list[LiteralFrameProps], ["geo_ret"]),
        )
        if not isinstance(result_arg, DataFrame):
            msg = "Method all_properties() not working as intended."
            raise TypeError(msg)

        if set(prop_index) != set(result_index):
            msg = "Method all_properties() output not as intended."
            raise ValueError(msg)

        result_values = {}
        for value in result.index:
            if isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], float):
                result_values[value] = (
                    f"{result.loc[value, ('Asset_0', ValueType.PRICE)]:.10f}"
                )
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], int):
                result_values[value] = cast(
                    str,
                    result.loc[value, ("Asset_0", ValueType.PRICE)],
                )
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], dt.date):
                result_values[value] = cast(
                    dt.date,
                    result.loc[value, ("Asset_0", ValueType.PRICE)],
                ).strftime("%Y-%m-%d")
            else:
                msg = f"all_properties returned unexpected type {type(value)}"
                raise TypeError(
                    msg,
                )
        expected_values = {
            "Arithmetic return": "0.0585047569",
            "CVaR 95.0%": "-0.0123803429",
            "Downside deviation": "0.0667228073",
            "Geometric return": "0.0507567099",
            "Imp vol from VaR 95%": "0.0936737165",
            "Kurtosis": "696.0965168893",
            "Max drawdown": "-0.1314808074",
            "Max drawdown in cal yr": "-0.1292814491",
            "Max drawdown date": "2012-11-21",
            "Omega ratio": "1.0983709757",
            "Positive share": "0.5057745918",
            "Return vol ratio": "0.4162058331",
            "Simple return": "0.6401159258",
            "Skew": "19.1911712502",
            "Sortino ratio": "0.8768329634",
            "VaR 95.0%": "-0.0097182152",
            "Volatility": "0.1405668835",
            "Worst": "-0.0191572882",
            "Worst month": "-0.0581245494",
            "Z-score": "0.3750685522",
            "first indices": "2009-06-30",
            "last indices": "2019-06-28",
            "observations": 2512,
            "span of days": 3650,
        }

        if result_values != expected_values:
            msg = (
                "Method all_properties() results "
                f"not as expected.\n{pformat(result_values)}"
            )
            raise ValueError(msg)

        with pytest.raises(expected_exception=ValueError, match="Invalid string: boo"):
            _ = apframe.all_properties(
                properties=cast(list[LiteralFrameProps], ["geo_ret", "boo"]),
            )

    def test_align_index_to_local_cdays(self: TestOpenFrame) -> None:
        """Test align_index_to_local_cdays method."""
        d_range = [d.date() for d in date_range(start="2022-06-01", end="2022-06-15")]
        asim = [1.0] * len(d_range)
        adf = DataFrame(
            data=asim,
            index=d_range,
            columns=[["Asset_a"], [ValueType.PRICE]],
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype=ValueType.PRICE)
        bseries = OpenTimeSeries.from_df(adf, valuetype=ValueType.PRICE)
        bseries.set_new_label("Asset_b")
        aframe = OpenFrame([aseries, bseries])

        midsummer = dt.date(2022, 6, 6)
        if midsummer not in d_range:
            msg = "Midsummer not in date range"
            raise ValueError(msg)

        aframe.align_index_to_local_cdays()
        if midsummer in aframe.tsdf.index:
            msg = "Midsummer in date range"
            raise ValueError(msg)

    def test_rolling_info_ratio(self: TestOpenFrame) -> None:
        """Test rolling_info_ratio method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_info_ratio(long_column=0, short_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "-0.10944130198",
            "-0.12182987211",
            "-0.02845717747",
            "-0.04862442310",
            "-0.02521145077",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_info_ratio() not as intended\n{values}"
            raise ValueError(msg)

        simdata_fxd_per_yr = frame.rolling_info_ratio(
            long_column=0,
            short_column=1,
            periods_in_a_year_fixed=251,
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "-0.10952238381",
            "-0.12192013228",
            "-0.02847826056",
            "-0.04866044751",
            "-0.02523012919",
        ]

        if values_fxd_per_yr != checkdata_fxd_per_yr:
            msg = (
                "Result from method rolling_info_ratio()"
                f" not as intended\n{values_fxd_per_yr}"
            )
            raise ValueError(msg)

    def test_rolling_beta(self: TestOpenFrame) -> None:
        """Test rolling_beta method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()
        simdata = frame.rolling_beta(asset_column=0, market_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.11870239224",
            "0.13743313660",
            "0.10203126346",
            "0.11658696118",
            "0.09445337602",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_beta() not as intended\n{values}"
            raise ValueError(msg)

    def test_tracking_error_func(self: TestOpenFrame) -> None:
        """Test tracking_error_func method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdataa = frame.tracking_error_func(base_column=-1)

        if f"{simdataa.iloc[0]:.10f}" != "0.1611858846":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdataa.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatab = frame.tracking_error_func(
            base_column=-1,
            periods_in_a_year_fixed=251,
        )

        if f"{simdatab.iloc[0]:.10f}" != "0.1610665551":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdatab.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatac = frame.tracking_error_func(base_column=("Asset_4", ValueType.PRICE))

        if f"{simdatac.iloc[0]:.10f}" != "0.1611858846":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdatac.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        if f"{simdataa.iloc[0]:.10f}" != f"{simdatac.iloc[0]:.10f}":
            msg = (
                "Result from tracking_error_func() not "
                f"as expected: '{simdataa.iloc[0]:.10f}' "
                f"versus '{simdatac.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = frame.tracking_error_func(
                base_column=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_info_ratio_func(self: TestOpenFrame) -> None:
        """Test info_ratio_func method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdataa = frame.info_ratio_func(base_column=-1)

        if f"{simdataa.iloc[0]:.10f}" != "0.3141560406":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdataa.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatab = frame.info_ratio_func(base_column=-1, periods_in_a_year_fixed=251)

        if f"{simdatab.iloc[0]:.10f}" != "0.3139234639":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdatab.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        simdatac = frame.info_ratio_func(base_column=("Asset_4", ValueType.PRICE))

        if f"{simdatac.iloc[0]:.10f}" != "0.3141560406":
            msg = (
                f"Result from info_ratio_func() not "
                f"as expected: '{simdatac.iloc[0]:.10f}'"
            )
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = frame.info_ratio_func(
                base_column=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_rolling_corr(self: TestOpenFrame) -> None:
        """Test rolling_corr method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_corr(first_column=0, second_column=1)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.17883816332",
            "0.20010259575",
            "0.14454747431",
            "0.16212311694",
            "0.12819083799",
        ]

        if values != checkdata:
            msg = "Result from method rolling_corr() not as intended."
            raise ValueError(msg)

    def test_rolling_vol(self: TestOpenFrame) -> None:
        """Test rolling_vol method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_vol(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.06592263886",
            "0.06849943405",
            "0.07131196245",
            "0.07266936036",
            "0.07194915928",
        ]

        if values != checkdata:
            msg = "Result from method rolling_vol() not as intended."
            raise ValueError(msg)

        simdata_fxd_per_yr = frame.rolling_vol(
            column=0,
            observations=21,
            periods_in_a_year_fixed=251,
        )

        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5, 0]]
        checkdata_fxd_per_yr = [
            "0.06587383487",
            "0.06844872241",
            "0.07125916863",
            "0.07261556163",
            "0.07189589373",
        ]

        if values_fxd_per_yr != checkdata_fxd_per_yr:
            msg = "Result from method rolling_vol() not as intended."
            raise ValueError(msg)

    def test_rolling_return(self: TestOpenFrame) -> None:
        """Test rolling_return method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_return(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[:5, 0]]
        checkdata = [
            "0.00700872599",
            "-0.00261771394",
            "0.00348938855",
            "0.00809475448",
            "0.00383062672",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_return() not as intended\n{values}"
            raise ValueError(msg)

    def test_rolling_cvar_down(self: TestOpenFrame) -> None:
        """Test rolling_cvar_down method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_cvar_down(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[-14:-9, 0]]
        checkdata = [
            "-0.01044706636",
            "-0.00974280167",
            "-0.00974280167",
            "-0.01139997783",
            "-0.01077418053",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_cvar_down() not as intended\n{values}"
            raise ValueError(msg)

    def test_rolling_var_down(self: TestOpenFrame) -> None:
        """Test rolling_var_down method."""
        frame = self.randomframe.from_deepcopy()
        frame.to_cumret()

        simdata = frame.rolling_var_down(column=0, observations=21)

        values = [f"{v:.11f}" for v in simdata.iloc[-5:, 0]]
        checkdata = [
            "-0.01243135668",
            "-0.01243135668",
            "-0.01243135668",
            "-0.01243135668",
            "-0.01243135668",
        ]

        if values != checkdata:
            msg = f"Result from method rolling_var_down() not as intended\n{values}"
            raise ValueError(msg)

    def test_label_uniqueness(self: TestOpenFrame) -> None:
        """Test label uniqueness."""
        aseries = self.randomseries.from_deepcopy()
        bseries = self.randomseries.from_deepcopy()

        with pytest.raises(
            expected_exception=ValueError,
            match="TimeSeries names/labels must be unique",
        ):
            OpenFrame([aseries, bseries])

        bseries.set_new_label("other_name")
        uframe = OpenFrame([aseries, bseries])

        if uframe.columns_lvl_zero != ["Asset_0", "other_name"]:
            msg = "Fix of non-unique labels unsuccessful."
            raise ValueError(msg)

    def test_capture_ratio(self: TestOpenFrame) -> None:
        """
        Test the capture_ratio_func method.

        Source: 'Capture Ratios: A Popular Method of Measuring Portfolio Performance
        in Practice', Don R. Cox and Delbert C. Goff, Journal of Economics and
        Finance Education (Vol 2 Winter 2013).
        https://www.economics-finance.org/jefe/volume12-2/11ArticleCox.pdf.
        """
        asset = OpenTimeSeries.from_arrays(
            name="asset",
            baseccy="USD",
            valuetype=ValueType.RTRN,
            dates=[
                "2007-12-31",
                "2008-01-31",
                "2008-02-29",
                "2008-03-31",
                "2008-04-30",
                "2008-05-31",
                "2008-06-30",
                "2008-07-31",
                "2008-08-31",
                "2008-09-30",
                "2008-10-31",
                "2008-11-30",
                "2008-12-31",
                "2009-01-31",
                "2009-02-28",
                "2009-03-31",
                "2009-04-30",
                "2009-05-31",
                "2009-06-30",
                "2009-07-31",
                "2009-08-31",
                "2009-09-30",
                "2009-10-31",
                "2009-11-30",
                "2009-12-31",
            ],
            values=[
                0.0,
                -0.0436,
                -0.0217,
                -0.0036,
                0.0623,
                0.0255,
                -0.0555,
                -0.0124,
                -0.0088,
                -0.0643,
                -0.1897,
                -0.0781,
                0.0248,
                -0.0666,
                -0.0993,
                0.0853,
                0.1028,
                0.0634,
                -0.0125,
                0.0762,
                0.0398,
                0.048,
                -0.0052,
                0.0592,
                0.0195,
            ],
        )
        indxx = OpenTimeSeries.from_arrays(
            name="indxx",
            valuetype=ValueType.RTRN,
            baseccy="USD",
            dates=[
                "2007-12-31",
                "2008-01-31",
                "2008-02-29",
                "2008-03-31",
                "2008-04-30",
                "2008-05-31",
                "2008-06-30",
                "2008-07-31",
                "2008-08-31",
                "2008-09-30",
                "2008-10-31",
                "2008-11-30",
                "2008-12-31",
                "2009-01-31",
                "2009-02-28",
                "2009-03-31",
                "2009-04-30",
                "2009-05-31",
                "2009-06-30",
                "2009-07-31",
                "2009-08-31",
                "2009-09-30",
                "2009-10-31",
                "2009-11-30",
                "2009-12-31",
            ],
            values=[
                0.0,
                -0.06,
                -0.0325,
                -0.0043,
                0.0487,
                0.013,
                -0.0843,
                -0.0084,
                0.0145,
                -0.0891,
                -0.168,
                -0.0718,
                0.0106,
                -0.0843,
                -0.1065,
                0.0876,
                0.0957,
                0.0559,
                0.002,
                0.0756,
                0.0361,
                0.0373,
                -0.0186,
                0.06,
                0.0193,
            ],
        )
        cframe = OpenFrame([asset, indxx]).to_cumret()

        upp = cframe.capture_ratio_func(ratio="up")
        down = cframe.capture_ratio_func(ratio="down")
        both = cframe.capture_ratio_func(ratio="both")

        if f"{upp.iloc[0]:.12f}" != "1.063842457805":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)
        if f"{down.iloc[0]:.12f}" != "0.922188852957":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)
        if f"{both.iloc[0]:.12f}" != "1.153605852417":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        upfixed = cframe.capture_ratio_func(ratio="up", periods_in_a_year_fixed=12)

        if f"{upfixed.iloc[0]:.12f}" != "1.063217236138":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        if f"{upp.iloc[0]:.2f}" != f"{upfixed.iloc[0]:.2f}":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        uptuple = cframe.capture_ratio_func(
            ratio="up",
            base_column=("indxx", ValueType.PRICE),
        )

        if f"{uptuple.iloc[0]:.12f}" != "1.063842457805":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        if f"{upp.iloc[0]:.12f}" != f"{uptuple.iloc[0]:.12f}":
            msg = "Result from capture_ratio_func() not as expected."
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="base_column should be a tuple",
        ):
            _ = cframe.capture_ratio_func(
                ratio="up",
                base_column=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_georet_exceptions(self: TestOpenFrame) -> None:
        """Test georet property raising exceptions on bad input data."""
        geoframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="geoseries1",
                    dates=["2022-07-01", "2023-07-01"],
                    values=[1.0, 1.1],
                ),
                OpenTimeSeries.from_arrays(
                    name="geoseries2",
                    dates=["2022-07-01", "2023-07-01"],
                    values=[1.0, 1.2],
                ),
            ],
        )
        if [f"{gr:.5f}" for gr in cast(Series, geoframe.geo_ret)] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise ValueError(msg)

        if [f"{gr:.5f}" for gr in cast(Series, geoframe.geo_ret_func())] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise ValueError(msg)

        geoframe.add_timeseries(
            OpenTimeSeries.from_arrays(
                name="geoseries3",
                dates=["2022-07-01", "2023-07-01"],
                values=[0.0, 1.1],
            ),
        )

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret_func()

        geoframe.delete_timeseries(lvl_zero_item="geoseries3")

        if [f"{gr:.5f}" for gr in cast(Series, geoframe.geo_ret)] != [
            "0.10007",
            "0.20015",
        ]:
            msg = "Unexpected result from property geo_ret"
            raise ValueError(msg)

        geoframe.add_timeseries(
            OpenTimeSeries.from_arrays(
                name="geoseries4",
                dates=["2022-07-01", "2023-07-01"],
                values=[1.0, -1.1],
            ),
        )
        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = geoframe.geo_ret_func()

    def test_value_nan_handle(self: TestOpenFrame) -> None:
        """Test value_nan_handle method."""
        nanframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="nanseries1",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[1.1, 1.0, 0.8, 1.1, 1.0],
                ),
                OpenTimeSeries.from_arrays(
                    name="nanseries2",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    values=[2.1, 2.0, 1.8, 2.1, 2.0],
                ),
            ],
        )
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[2, 0] = None
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.value_nan_handle(method="drop")

        if dropframe.tsdf.iloc[:, 0].tolist() != [1.1, 1.0, 1.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)
        if dropframe.tsdf.iloc[:, 1].tolist() != [2.1, 2.0, 2.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)

        fillframe = nanframe.from_deepcopy()
        fillframe.value_nan_handle(method="fill")

        if fillframe.tsdf.iloc[:, 0].tolist() != [1.1, 1.0, 1.0, 1.1, 1.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)
        if fillframe.tsdf.iloc[:, 1].tolist() != [2.1, 2.0, 1.8, 1.8, 2.0]:
            msg = "Method value_nan_handle() not working as intended"
            raise ValueError(msg)

    def test_return_nan_handle(self: TestOpenFrame) -> None:
        """Test return_nan_handle method."""
        nanframe = OpenFrame(
            [
                OpenTimeSeries.from_arrays(
                    name="nanseries1",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    valuetype=ValueType.RTRN,
                    values=[0.1, 0.05, 0.03, 0.01, 0.04],
                ),
                OpenTimeSeries.from_arrays(
                    name="nanseries2",
                    dates=[
                        "2022-07-11",
                        "2022-07-12",
                        "2022-07-13",
                        "2022-07-14",
                        "2022-07-15",
                    ],
                    valuetype=ValueType.RTRN,
                    values=[0.01, 0.04, 0.02, 0.11, 0.06],
                ),
            ],
        )
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[2, 0] = None
        # noinspection PyTypeChecker
        nanframe.tsdf.iloc[3, 1] = None
        dropframe = nanframe.from_deepcopy()
        dropframe.return_nan_handle(method="drop")

        if dropframe.tsdf.iloc[:, 0].tolist() != [0.1, 0.05, 0.04]:
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)
        if dropframe.tsdf.iloc[:, 1].tolist() != [0.01, 0.04, 0.06]:
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)

        fillframe = nanframe.from_deepcopy()
        fillframe.return_nan_handle(method="fill")

        if fillframe.tsdf.iloc[:, 0].tolist() != [0.1, 0.05, 0.0, 0.01, 0.04]:
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)
        if fillframe.tsdf.iloc[:, 1].tolist() != [0.01, 0.04, 0.02, 0.0, 0.06]:
            msg = "Method return_nan_handle() not working as intended"
            raise ValueError(msg)

    def test_relative(self: TestOpenFrame) -> None:
        """Test relative method."""
        rframe = self.randomframe.from_deepcopy()
        rframe.to_cumret()
        sframe = self.randomframe.from_deepcopy()
        sframe.to_cumret()
        series_before = 5
        series_after = 6

        if rframe.item_count != series_before:
            msg = "Method relative() base case not as intended"
            raise ValueError(msg)

        rframe.relative()

        if rframe.item_count != series_after:
            msg = "Method relative() not working as intended"
            raise ValueError(msg)

        if rframe.tsdf.shape[1] != series_after:
            msg = "Method relative() not working as intended"
            raise ValueError(msg)

        if rframe.constituents[-1].label != "Asset_0_over_Asset_1":
            msg = "Method relative() not working as intended"
            raise ValueError(msg)

        if rframe.columns_lvl_zero[-1] != "Asset_0_over_Asset_1":
            msg = "Method relative() not working as intended"
            raise ValueError(msg)

        rframe.tsdf.iloc[:, -1] = rframe.tsdf.iloc[:, -1].add(1.0)

        sframe.relative(base_zero=False)

        rflist = [f"{rret:.11f}" for rret in rframe.tsdf.iloc[:, -1]]
        sflist = [f"{rret:.11f}" for rret in sframe.tsdf.iloc[:, -1]]

        if rflist != sflist:
            msg = "Method relative() not working as intended"
            raise ValueError(msg)

    def test_to_cumret(self: TestOpenFrame) -> None:
        """Test to_cumret method."""
        rseries = self.randomseries.from_deepcopy()
        rseries.value_to_ret()
        rrseries = rseries.from_deepcopy()
        rrseries.set_new_label(lvl_zero="Rasset")

        cseries = self.randomseries.from_deepcopy()
        cseries.set_new_label(lvl_zero="Basset")
        ccseries = cseries.from_deepcopy()
        ccseries.set_new_label(lvl_zero="Casset")

        mframe = OpenFrame([rseries, cseries])
        cframe = OpenFrame([cseries, ccseries])
        rframe = OpenFrame([rseries, rrseries])

        if mframe.columns_lvl_one != [ValueType.RTRN, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        if cframe.columns_lvl_one != [ValueType.PRICE, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        cframe_lvl_one = list(cframe.columns_lvl_one)

        if rframe.columns_lvl_one != [ValueType.RTRN, ValueType.RTRN]:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        mframe.to_cumret()
        cframe.to_cumret()
        rframe.to_cumret()

        if mframe.columns_lvl_one != [ValueType.PRICE, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        if cframe_lvl_one != cframe.columns_lvl_one:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        if rframe.columns_lvl_one != [ValueType.PRICE, ValueType.PRICE]:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

        fmt = "{:.8f}"

        frame_0 = self.randomframe.from_deepcopy()
        frame_0.to_cumret()
        frame_0.value_to_ret()
        frame_0.tsdf = frame_0.tsdf.map(fmt.format)
        dict_toframe_0 = frame_0.tsdf.to_dict()

        frame_1 = self.randomframe.from_deepcopy()
        frame_1.tsdf = frame_1.tsdf.map(fmt.format)
        dict_toframe_1 = frame_1.tsdf.to_dict()

        if dict_toframe_0 != dict_toframe_1:
            msg = "Method to_cumret() not working as intended"
            raise ValueError(msg)

    def test_miscellaneous(self: TestOpenFrame) -> None:
        """Test miscellaneous methods."""
        zero_str: str = "0"
        zero_float: float = 0.0
        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()

        methods = [
            "arithmetic_ret_func",
            "vol_func",
            "vol_from_var_func",
            "downside_deviation_func",
            "target_weight_from_var",
        ]
        for methd in methods:
            no_fixed = getattr(mframe, methd)()
            fixed = getattr(mframe, methd)(periods_in_a_year_fixed=252)
            for nofix, fix in zip(no_fixed, fixed):
                if f"{100*abs(nofix-fix):.0f}" != zero_str:
                    msg = (
                        "Difference with or without "
                        "fixed periods in year is too great"
                    )
                    raise ValueError(msg)
        for methd in methods:
            dated = getattr(mframe, methd)(
                from_date=mframe.first_idx,
                to_date=mframe.last_idx,
            )
            undated = getattr(mframe, methd)()
            for ddat, undat in zip(dated, undated):
                if f"{ddat:.10f}" != f"{undat:.10f}":
                    msg = (
                        f"Method {methd} with and without date "
                        "arguments returned inconsistent results"
                    )
                    raise ValueError(msg)

        ret = [f"{rr:.9f}" for rr in cast(Series, mframe.value_ret_func())]
        if ret != [
            "0.640115926",
            "0.354975641",
            "1.287658441",
            "1.045918527",
            "0.169641332",
        ]:
            msg = f"Results from value_ret_func() not as expected\n{ret}"
            raise ValueError(msg)

        impvol = [
            f"{iv:.11f}"
            for iv in cast(Series, mframe.vol_from_var_func(drift_adjust=False))
        ]
        impvoldrifted = [
            f"{iv:.11f}"
            for iv in cast(Series, mframe.vol_from_var_func(drift_adjust=True))
        ]

        if impvol != [
            "0.09367371648",
            "0.09357837907",
            "0.09766514288",
            "0.09903125918",
            "0.10121521823",
        ]:
            msg = f"Results from vol_from_var_func() not as expected\n{impvol}"
            raise ValueError(msg)

        if impvoldrifted != [
            "0.09591621674",
            "0.09495303438",
            "0.10103656927",
            "0.10197016748",
            "0.10201301292",
        ]:
            msg = f"Results from vol_from_var_func() not as expected\n{impvoldrifted}"
            raise ValueError(msg)

        # noinspection PyTypeChecker
        mframe.tsdf.iloc[0, 2] = zero_float

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = mframe.value_ret

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = mframe.value_ret_func()

    def test_value_ret_calendar_period(self: TestOpenFrame) -> None:
        """Test value_ret_calendar_period method."""
        vrcseries = self.randomseries.from_deepcopy()
        vrcseries.to_cumret()
        vrcframe = self.randomframe.from_deepcopy()
        vrcframe.to_cumret()

        vrfs_y = vrcseries.value_ret_func(
            from_date=dt.date(2017, 12, 29),
            to_date=dt.date(2018, 12, 28),
        )
        vrff_y = vrcframe.value_ret_func(
            from_date=dt.date(2017, 12, 29),
            to_date=dt.date(2018, 12, 28),
        )
        vrffl_y = [f"{rr:.11f}" for rr in cast(Series, vrff_y)]

        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        vrvrcf_y = vrcframe.value_ret_calendar_period(year=2018)
        vrvrcfl_y = [f"{rr:.11f}" for rr in cast(Series, vrvrcf_y)]

        if f"{vrfs_y:.11f}" != f"{vrvrcs_y:.11f}":
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

        if vrffl_y != vrvrcfl_y:
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dt.date(2018, 4, 30),
            to_date=dt.date(2018, 5, 31),
        )
        vrff_ym = vrcframe.value_ret_func(
            from_date=dt.date(2018, 4, 30),
            to_date=dt.date(2018, 5, 31),
        )
        vrffl_ym = [f"{rr:.11f}" for rr in cast(Series, vrff_ym)]

        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        vrvrcf_ym = vrcframe.value_ret_calendar_period(year=2018, month=5)
        vrvrcfl_ym = [f"{rr:.11f}" for rr in cast(Series, vrvrcf_ym)]

        if f"{vrfs_ym:.11f}" != f"{vrvrcs_ym:.11f}":
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

        if vrffl_ym != vrvrcfl_ym:
            msg = "value_ret_func() and value_ret_calendar_period() inconsistent"
            raise ValueError(msg)

    def test_to_drawdown_series(self: TestOpenFrame) -> None:
        """Test to_drawdown_series method."""
        mframe = self.randomframe.from_deepcopy()
        mframe.to_cumret()
        ddown = [f"{dmax:.11f}" for dmax in cast(Series, mframe.max_drawdown)]
        mframe.to_drawdown_series()
        ddownserie = [f"{dmax:.11f}" for dmax in mframe.tsdf.min()]

        if ddown != ddownserie:
            msg = "Method to_drawdown_series() not working as intended"
            raise ValueError(msg)

    def test_ord_least_squares_fit(self: TestOpenFrame) -> None:
        """Test ord_least_squares_fit method."""
        oframe = self.randomframe.from_deepcopy()
        oframe.to_cumret()
        oframe.value_to_log()

        fsframe = self.randomframe.from_deepcopy()
        fsframe.to_cumret()
        fsframe.ord_least_squares_fit(y_column=0, x_column=1, fitted_series=True)

        if fsframe.columns_lvl_zero[-1] != oframe.columns_lvl_zero[0]:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise ValueError(msg)

        if fsframe.columns_lvl_one[-1] != oframe.columns_lvl_zero[1]:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise ValueError(msg)

        results = []
        for i in range(oframe.item_count):
            for j in range(oframe.item_count):
                tmp = oframe.ord_least_squares_fit(
                    y_column=i,
                    x_column=j,
                    fitted_series=False,
                )
                results.append(f"{float(tmp.params.iloc[0]):.11f}")

        results_tuple = []
        k_tuple: Hashable
        l_tuple: Hashable
        for k_tuple in oframe.tsdf:
            for l_tuple in oframe.tsdf:
                tmp = oframe.ord_least_squares_fit(
                    y_column=cast(tuple[str, ValueType], k_tuple),
                    x_column=cast(tuple[str, ValueType], l_tuple),
                    fitted_series=False,
                )
                results_tuple.append(f"{float(tmp.params.iloc[0]):.11f}")

        if results != results_tuple:
            msg = "Method ord_least_squares_fit() not working as intended"
            raise ValueError(msg)

        if results != [
            "1.00000000000",
            "1.37862724760",
            "0.56028811567",
            "0.69603804963",
            "0.46400465988",
            "0.65027591851",
            "1.00000000000",
            "0.38009173429",
            "0.46384576412",
            "0.22803974704",
            "1.66624847545",
            "2.39643672361",
            "1.00000000000",
            "1.20106034055",
            "0.40261757732",
            "1.35912552734",
            "1.92021284114",
            "0.78861141738",
            "1.00000000000",
            "0.70469281021",
            "0.09471012260",
            "0.09868105726",
            "0.02763366151",
            "0.07366264502",
            "1.00000000000",
        ]:
            msg = f"Method ord_least_squares_fit() not working as intended\n{results}"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="x_column should be a tuple",
        ):
            _ = oframe.ord_least_squares_fit(
                y_column=0,
                x_column=cast(Union[tuple[str, ValueType], int], "string"),
                fitted_series=False,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="y_column should be a tuple",
        ):
            _ = oframe.ord_least_squares_fit(
                y_column=cast(Union[tuple[str, ValueType], int], "string"),
                x_column=1,
                fitted_series=False,
            )

    def test_beta(self: TestOpenFrame) -> None:
        """Test beta method."""
        bframe = self.randomframe.from_deepcopy()
        bframe.to_cumret()
        bframe.resample("7D")
        results = [
            f"{bframe.beta(asset=comb[0], market=comb[1]):.11f}"
            for comb in iter_product(
                range(bframe.item_count),
                range(bframe.item_count),
            )
        ]
        results_tuple = []
        for comb in iter_product(bframe.tsdf, bframe.tsdf):
            beta = bframe.beta(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{beta:.11f}")

        if results != results_tuple:
            msg = "Unexpected results from method beta()"
            raise ValueError(msg)

        if results != [
            "1.00000000000",
            "1.29922733925",
            "0.70818608993",
            "0.67350090937",
            "1.10828592932",
            "0.58286749395",
            "1.00000000000",
            "0.44923557262",
            "0.41611038337",
            "0.66683747762",
            "1.26184019179",
            "1.78421184238",
            "1.00000000000",
            "0.92762168147",
            "1.60509866122",
            "1.29388135250",
            "1.78188690161",
            "1.00016164109",
            "1.00000000000",
            "1.61604657978",
            "0.48322440146",
            "0.64808557090",
            "0.39277316259",
            "0.36677070860",
            "1.00000000000",
        ]:
            msg = f"Unexpected results from method beta()\n{results}"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = bframe.beta(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = bframe.beta(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_beta_returns_input(self: TestOpenFrame) -> None:
        """Test beta method with returns input."""
        bframe = self.randomframe.from_deepcopy()
        bframe.resample("7D")
        results = [
            f"{bframe.beta(asset=comb[0], market=comb[1]):.11f}"
            for comb in iter_product(
                range(bframe.item_count),
                range(bframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(bframe.tsdf, bframe.tsdf):
            beta = bframe.beta(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{beta:.11f}")

        if results != results_tuple:
            msg = "Unexpected results from method beta()"
            raise ValueError(msg)

        if results != [
            "1.00000000000",
            "0.08324853413",
            "-0.10099866224",
            "0.01462541518",
            "-0.16899363037",
            "0.01377634252",
            "1.00000000000",
            "0.01993819305",
            "-0.03779993992",
            "0.01254650184",
            "-0.01853961987",
            "0.02211636067",
            "1.00000000000",
            "-0.06352408467",
            "-0.01478713969",
            "0.00229000338",
            "-0.03576528518",
            "-0.05418525628",
            "1.00000000000",
            "0.05247274217",
            "-0.02629986958",
            "0.01179909172",
            "-0.01253667106",
            "0.05215417741",
            "1.00000000000",
        ]:
            msg = f"Unexpected results from method beta()\n{results}"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = bframe.beta(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = bframe.beta(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_jensen_alpha(self: TestOpenFrame) -> None:
        """Test jensen_alpha method."""
        jframe = self.randomframe.from_deepcopy()
        jframe.to_cumret()
        jframe.resample("7D")
        results = [
            f"{jframe.jensen_alpha(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(jframe.item_count),
                range(jframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(jframe.tsdf, jframe.tsdf):
            alpha = jframe.jensen_alpha(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{alpha:.9f}")

        if results != results_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        if results != [
            "0.000000000",
            "0.011275539",
            "-0.011480230",
            "0.002038749",
            "0.031602774",
            "0.000692798",
            "0.000000000",
            "-0.009171994",
            "0.000198488",
            "0.018763801",
            "0.023746109",
            "0.033640184",
            "0.000000000",
            "0.020761658",
            "0.060163598",
            "0.006548865",
            "0.018119801",
            "-0.015604040",
            "0.000000000",
            "0.044390287",
            "-0.007493292",
            "-0.002651295",
            "-0.017441772",
            "-0.009460482",
            "0.000000000",
        ]:
            msg = f"Unexpected results from method jensen_alpha()\n{results}"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

        ninemth = date_offset_foll(jframe.last_idx, months_offset=-9, adjust=True)
        shortframe = jframe.trunc_frame(start_cut=ninemth)
        shortframe.to_cumret()
        sresults = [
            f"{shortframe.jensen_alpha(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(shortframe.item_count),
                range(shortframe.item_count),
            )
        ]

        sresults_tuple: list[str] = []
        for comb in iter_product(shortframe.tsdf, shortframe.tsdf):
            alpha = shortframe.jensen_alpha(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            sresults_tuple.append(f"{alpha:.9f}")

        if sresults != sresults_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        if sresults != [
            "0.000000000",
            "-0.007428381",
            "-0.015244859",
            "-0.028785495",
            "-0.054945887",
            "-0.025064707",
            "0.000000000",
            "-0.014783332",
            "0.024615935",
            "0.048853995",
            "0.024840001",
            "0.012827133",
            "0.000000000",
            "0.003855844",
            "-0.066858420",
            "0.064449240",
            "0.058748466",
            "0.061290014",
            "0.000000000",
            "0.013305765",
            "0.141205760",
            "0.128274302",
            "0.123668318",
            "0.068585817",
            "0.000000000",
        ]:
            msg = f"Unexpected results from method jensen_alpha()\n{sresults}"
            raise ValueError(msg)

    def test_jensen_alpha_returns_input(self: TestOpenFrame) -> None:
        """Test jensen_alpha method with returns input."""
        jframe = self.randomframe.from_deepcopy()
        jframe.resample("7D")
        results = [
            f"{jframe.jensen_alpha(asset=comb[0], market=comb[1]):.9f}"
            for comb in iter_product(
                range(jframe.item_count),
                range(jframe.item_count),
            )
        ]

        results_tuple = []
        for comb in iter_product(jframe.tsdf, jframe.tsdf):
            alpha = jframe.jensen_alpha(
                asset=comb[0],  # type: ignore[arg-type]
                market=comb[1],  # type: ignore[arg-type]
            )
            results_tuple.append(f"{alpha:.9f}")

        if results != results_tuple:
            msg = "Unexpected results from method jensen_alpha()"
            raise ValueError(msg)

        if results != [
            "0.000000000",
            "0.000791942",
            "0.000768260",
            "0.000773600",
            "0.000772380",
            "-0.000185997",
            "0.000000000",
            "-0.000173494",
            "-0.000165599",
            "-0.000174919",
            "-0.000075575",
            "-0.000086110",
            "0.000000000",
            "-0.000073705",
            "-0.000090422",
            "0.000254537",
            "0.000250048",
            "0.000251442",
            "0.000000000",
            "0.000257860",
            "-0.000008959",
            "-0.000027335",
            "-0.000030532",
            "-0.000042772",
            "0.000000000",
        ]:
            msg = f"Unexpected results from method jensen_alpha()\n{results}"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=TypeError,
            match="asset should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=cast(Union[tuple[str, ValueType], int], "string"),
                market=1,
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="market should be a tuple",
        ):
            _ = jframe.jensen_alpha(
                asset=0,
                market=cast(Union[tuple[str, ValueType], int], "string"),
            )

    def test_ewma_risk(self: TestOpenFrame) -> None:
        """Test ewma_risk method."""
        eframe = self.randomframe.from_deepcopy()
        eframe.to_cumret()
        edf = eframe.ewma_risk()

        list_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 0]]
        list_two = [f"{e:.11f}" for e in edf.head(10).iloc[:, 1]]
        corr_one = [f"{e:.11f}" for e in edf.head(10).iloc[:, 2]]

        if list_one != [
            "0.06250431742",
            "0.06208916909",
            "0.06022552031",
            "0.05840562180",
            "0.05812960782",
            "0.05791392918",
            "0.05691361221",
            "0.06483761105",
            "0.06421248171",
            "0.06260970713",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_one}"
            raise ValueError(msg)

        if list_two != [
            "0.06783309941",
            "0.06799180977",
            "0.06592070398",
            "0.06491357256",
            "0.06990142770",
            "0.06794101498",
            "0.06878554654",
            "0.06979371620",
            "0.07423575130",
            "0.07469953921",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_two}"
            raise ValueError(msg)

        if corr_one != [
            "-0.00018950439",
            "0.02743890783",
            "0.02746345872",
            "0.02900303362",
            "0.07459903397",
            "0.08052954645",
            "0.09959709234",
            "0.00357025403",
            "-0.03875791933",
            "-0.02293443963",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{corr_one}"
            raise ValueError(msg)

    def test_ewma_risk_set_columns(self: TestOpenFrame) -> None:
        """Test ewma_risk method on specified columns."""
        eframe = self.randomframe.from_deepcopy()
        eframe.to_cumret()
        fdf = eframe.ewma_risk(
            first_column=3,
            second_column=4,
            periods_in_a_year_fixed=251,
        )
        list_three = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 0]]
        list_four = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 1]]
        corr_two = [f"{f:.11f}" for f in fdf.head(10).iloc[:, 2]]

        if list_three != [
            "0.07361260347",
            "0.08380092327",
            "0.08392771222",
            "0.08501304571",
            "0.08362635928",
            "0.08175874942",
            "0.07996606071",
            "0.07806839320",
            "0.07604456321",
            "0.07468607166",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_three}"
            raise ValueError(msg)

        if list_four != [
            "0.10703091058",
            "0.10476082189",
            "0.10165576633",
            "0.10346063549",
            "0.10280828404",
            "0.09967975602",
            "0.09693631061",
            "0.10848146972",
            "0.11025146341",
            "0.11658815836",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{list_four}"
            raise ValueError(msg)

        if corr_two != [
            "-0.00112585517",
            "0.03499961968",
            "0.03901924990",
            "-0.00845828372",
            "-0.02665539891",
            "-0.02696637211",
            "-0.02152694854",
            "-0.04779886556",
            "-0.05984710080",
            "-0.08603979121",
        ]:
            msg = f"Unexpected results from method ewma_risk()\n{corr_two}"
            raise ValueError(msg)

    def test_simulate_portfolios(self: TestOpenFrame) -> None:
        """Test function simulate_portfolios."""
        simulations = 1000

        spframe = self.randomframe.from_deepcopy()

        result_returns = simulate_portfolios(
            simframe=spframe,
            num_ports=simulations,
            seed=SEED,
        )

        if result_returns.shape != (simulations, spframe.item_count + 3):
            msg = "Function simulate_portfolios not working as intended"
            raise ValueError(msg)

        return_least_vol = f"{result_returns.loc[:, 'stdev'].min():.7f}"
        return_where_least_vol = (
            f"{result_returns.loc[result_returns['stdev'].idxmin()]['ret']:.7f}"
        )

        if (return_least_vol, return_where_least_vol) != ("0.0476395", "0.0568173"):
            msg = (
                "Function simulate_portfolios not working as intended"
                f"\n{(return_least_vol, return_where_least_vol)}"
            )
            raise ValueError(msg)

        spframe.to_cumret()
        result_values = simulate_portfolios(
            simframe=spframe,
            num_ports=simulations,
            seed=SEED,
        )

        if result_values.shape != (simulations, spframe.item_count + 3):
            msg = "Function simulate_portfolios not working as intended"
            raise ValueError(msg)

        value_least_vol = f"{result_values.loc[:, 'stdev'].min():.7f}"
        value_where_least_vol = (
            f"{result_values.loc[result_values['stdev'].idxmin()]['ret']:.7f}"
        )

        if (value_least_vol, value_where_least_vol) != ("0.0476489", "0.0568400"):
            msg = (
                "Function simulate_portfolios not working as intended"
                f"\n{(value_least_vol, value_where_least_vol)}"
            )
            raise ValueError(msg)

    def test_efficient_frontier(self: TestOpenFrame) -> None:
        """Test function efficient_frontier."""
        simulations = 1000
        points = 20

        eframe = self.randomframe.from_deepcopy()

        frnt, _, _ = efficient_frontier(
            eframe=eframe,
            num_ports=simulations,
            seed=SEED,
            frontier_points=points,
            tweak=False,
        )

        if frnt.shape != (points, eframe.item_count + 4):
            msg = "Function efficient_frontier not working as intended"
            raise ValueError(msg)

        eframe.to_cumret()

        frontier, result, optimal = efficient_frontier(
            eframe=eframe,
            num_ports=simulations,
            seed=SEED,
            frontier_points=points,
            tweak=False,
        )

        if frontier.shape != (points, eframe.item_count + 4):
            msg = "Function efficient_frontier not working as intended"
            raise ValueError(msg)

        frt_most_sharpe = f"{frontier.loc[:, 'sharpe'].max():.9f}"
        frt_return_where_most_sharpe = (
            f"{frontier.loc[frontier['sharpe'].idxmax()]['ret']:.9f}"
        )

        if (frt_most_sharpe, frt_return_where_most_sharpe) != (
            "1.302486911",
            "0.068289998",
        ):
            msg = (
                "Function efficient_frontier not working as intended"
                f"\n{(frt_most_sharpe, frt_return_where_most_sharpe)}"
            )
            raise ValueError(msg)

        sim_least_vol = f"{result.loc[:, 'stdev'].min():.9f}"
        sim_return_where_least_vol = (
            f"{result.loc[result['stdev'].idxmin()]['ret']:.9f}"
        )

        if (sim_least_vol, sim_return_where_least_vol) != (
            "0.047639486",
            "0.056817349",
        ):
            msg = (
                "Function efficient_frontier not working as intended"
                f"\n{(sim_least_vol, sim_return_where_least_vol)}"
            )
            raise ValueError(msg)

        optlist = [round(Decimal(wgt), 6) for wgt in cast(list[float], optimal)]
        total = sum(optimal[3:])

        if round(total, 7) != 1.0:
            msg = f"Function efficient_frontier not working as intended\n{total}"
            raise ValueError(msg)

        if optlist != [
            Decimal("0.068444"),
            Decimal("0.052547"),
            Decimal("1.302525"),
            Decimal("0.116616"),
            Decimal("0.140094"),
            Decimal("0.352682"),
            Decimal("0.312324"),
            Decimal("0.078283"),
        ]:
            msg = f"Function efficient_frontier not working as intended\n{optlist}"
            raise ValueError(msg)

    def test_constrain_optimized_portfolios(self: TestOpenFrame) -> None:
        """Test function constrain_optimized_portfolios."""
        simulations = 1000
        upper_bound = 1.0
        org_port_name = "Current Portfolio"

        std_frame = self.randomframe.from_deepcopy()
        std_frame.to_cumret()
        std_frame.weights = [1 / std_frame.item_count] * std_frame.item_count
        assets_std = OpenTimeSeries.from_df(std_frame.make_portfolio(org_port_name))

        minframe, minseries, maxframe, maxseries = constrain_optimized_portfolios(
            data=std_frame,
            serie=assets_std,
            portfolioname=org_port_name,
            simulations=simulations,
            upper_bound=upper_bound,
        )

        if round(sum(minframe.weights), 7) != 1.0:
            msg = (
                "Function constrain_optimized_portfolios not working as "
                f"intended\n{round(sum(minframe.weights), 7)}"
            )
            raise ValueError(msg)

        minframe_weights = [f"{minw:.7f}" for minw in list(minframe.weights)]
        if minframe_weights != [
            "0.1150421",
            "0.1854466",
            "0.2743087",
            "0.2572628",
            "0.1679398",
        ]:
            msg = (
                "Function constrain_optimized_portfolios not "
                f"working as intended\n{minframe_weights}"
            )
            raise ValueError(msg)

        if (
            f"{minseries.arithmetic_ret - assets_std.arithmetic_ret:.7f}"
            != "0.0047669"
        ):
            msg = (
                "Optimization did not find better return with similar vol\n"
                f"{minseries.arithmetic_ret - assets_std.arithmetic_ret:.7f}"
            )

            raise ValueError(msg)

        if round(sum(maxframe.weights), 7) != 1.0:
            msg = (
                "Function constrain_optimized_portfolios not working as "
                f"intended\n{round(sum(maxframe.weights), 7)}"
            )
            raise ValueError(msg)

        maxframe_weights = [f"{maxw:.7f}" for maxw in list(maxframe.weights)]
        if maxframe_weights != [
            "0.1152015",
            "0.1721200",
            "0.2971957",
            "0.2724543",
            "0.1430285",
        ]:
            msg = (
                "Function constrain_optimized_portfolios not "
                f"working as intended\n{maxframe_weights}"
            )
            raise ValueError(msg)

        if f"{assets_std.vol - maxseries.vol:.7f}" != "0.0000714":
            msg = (
                "Optimization did not find better return with similar vol\n"
                f"{assets_std.vol - maxseries.vol:.7f}"
            )

            raise ValueError(msg)

    def test_sharpeplot(self: TestOpenFrame) -> None:  # noqa: C901
        """Test function sharpeplot."""
        simulations = 1000
        points = 20

        spframe = self.randomframe.from_deepcopy()
        spframe.to_cumret()
        current = OpenTimeSeries.from_df(
            spframe.make_portfolio(
                name="Current Portfolio",
                weight_strat="eq_weights",
            ),
        )

        frontier, simulated, optimum = efficient_frontier(
            eframe=spframe,
            num_ports=simulations,
            seed=SEED,
            frontier_points=points,
            tweak=False,
        )

        plotframe = prepare_plot_data(
            assets=spframe,
            current=current,
            optimized=optimum,
        )

        figure_title_no_text, _ = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=True,
            auto_open=False,
            output_type="div",
        )

        fig_json_title_no_text = loads(cast(str, figure_title_no_text.to_json()))
        if "Risk and Return" not in fig_json_title_no_text["layout"]["title"]["text"]:
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        figure_title_text, _ = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=True,
            titletext="Awesome title",
            auto_open=False,
            output_type="div",
        )

        fig_json_title_text = loads(cast(str, figure_title_text.to_json()))
        if fig_json_title_text["layout"]["title"]["text"] != "Awesome title":
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        figure, _ = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=False,
            auto_open=False,
            output_type="div",
        )

        fig_json = loads(cast(str, figure.to_json()))

        if "text" in fig_json["layout"]["title"]:
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        names = [item["name"] for item in fig_json["data"]]

        if names != [
            "simulated portfolios",
            "Efficient frontier",
            "Asset_0",
            "Asset_1",
            "Asset_2",
            "Asset_3",
            "Asset_4",
            "Max Sharpe Portfolio",
            "Current Portfolio",
        ]:
            msg = f"Function sharpeplot not working as intended\n{names}"
            raise ValueError(msg)

        directory = Path(__file__).resolve().parent
        _, figfile = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=False,
            auto_open=False,
            output_type="file",
            directory=directory,
        )

        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "html file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "html file not deleted as intended"
            raise FileExistsError(msg)

        if figfile[:5] == "<div>":
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        _, divstring = sharpeplot(
            sim_frame=simulated,
            line_frame=frontier,
            point_frame=plotframe,
            point_frame_mode="markers+text",
            title=False,
            auto_open=False,
            output_type="div",
        )
        if divstring[:5] != "<div>" or divstring[-6:] != "</div>":
            msg = "Html div section not created"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = True
            mockhomefig, _ = sharpeplot(
                sim_frame=simulated,
                line_frame=frontier,
                point_frame=plotframe,
                point_frame_mode="markers+text",
                title=False,
                auto_open=False,
                output_type="div",
            )
            mockhomefig_json = loads(cast(str, mockhomefig.to_json()))

        if mockhomefig_json["data"][0]["name"] != "simulated portfolios":
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        with patch("pathlib.Path.exists") as mock_userfolderexists:
            mock_userfolderexists.return_value = False
            _, mockfile = sharpeplot(
                sim_frame=simulated,
                line_frame=frontier,
                point_frame=plotframe,
                point_frame_mode="markers+text",
                title=False,
                auto_open=False,
                output_type="file",
                filename="seriesfile.html",
            )
            mockfilepath = Path(mockfile).resolve()

        if mockfilepath.parts[-2:] != ("tests", "seriesfile.html"):
            msg = "sharpeplot method not working as intended"
            raise ValueError(msg)

        mockfilepath.unlink()
