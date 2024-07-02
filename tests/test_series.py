"""Test suite for the openseries/series.py module."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from json import load, loads
from pathlib import Path
from pprint import pformat
from typing import Union, cast
from unittest import TestCase

import pytest
from pandas import DataFrame, DatetimeIndex, Series, date_range
from pydantic import ValidationError

# noinspection PyProtectedMember
from openseries.series import (
    OpenTimeSeries,
    _check_if_none,
    timeseries_chain,
)
from openseries.types import (
    CountriesType,
    LiteralSeriesProps,
    ValueType,
)
from tests.test_common_sim import SIMSERIES


class NewTimeSeries(OpenTimeSeries):

    """class to test correct pass-through of classes."""

    extra_info: str = "cool"


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "valuetype",
    [ValueType.PRICE, "Price(Close)"],
)
def test_opentimeseries_valid_valuetype(valuetype: ValueType) -> None:
    """Pytest on valid valuetype as input."""
    if not isinstance(
        OpenTimeSeries.from_arrays(
            name="Asset",
            dates=["2023-01-01", "2023-01-02"],
            valuetype=valuetype,
            values=[1.0, 1.1],
        ),
        OpenTimeSeries,
    ):
        msg = "Valid valuetype input rendered unexpected error"
        raise TypeError(msg)


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "valuetype",
    [None, "Price", 12, 1.2],
)
def test_opentimeseries_invalid_valuetype(valuetype: ValueType) -> None:
    """Pytest on invalid valuetype as input."""
    with pytest.raises(
        expected_exception=ValidationError,
        match="type=enum|type=string_type",
    ):
        OpenTimeSeries.from_arrays(
            name="Asset",
            dates=["2023-01-01", "2023-01-02"],
            valuetype=valuetype,
            values=[1.0, 1.1],
        )


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "currency",
    ["SE", True, "12", 1, None],
)
def test_opentimeseries_invalid_currency(currency: str) -> None:
    """Pytest on invalid currency code as input for currency."""
    with pytest.raises(
        expected_exception=ValidationError,
        match="type=string_too_short|type=string_type",
    ):
        OpenTimeSeries.from_arrays(
            name="Asset",
            baseccy=currency,
            dates=["2023-01-01", "2023-01-02"],
            valuetype=ValueType.PRICE,
            values=[1.0, 1.1],
        )


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "domestic",
    ["SE", True, "12", 1, None],
)
def test_opentimeseries_invalid_domestic(domestic: str) -> None:
    """Pytest on invalid currency code as input for domestic."""
    serie = OpenTimeSeries.from_arrays(
        name="Asset",
        dates=["2023-01-01", "2023-01-02"],
        values=[1.0, 1.1],
    )
    with pytest.raises(
        expected_exception=ValidationError,
        match="type=string_too_short|type=string_type",
    ):
        serie.domestic = domestic


@pytest.mark.parametrize(  # type: ignore[misc, unused-ignore]
    "countries",
    ["SEK", True, "12", 1, None, ["SEK"], [True], ["12"], [1], [None], []],
)
def test_opentimeseries_invalid_countries(countries: CountriesType) -> None:
    """Pytest on invalid country codes as input."""
    serie = OpenTimeSeries.from_arrays(
        name="Asset",
        dates=["2023-01-01", "2023-01-02"],
        values=[1.0, 1.1],
    )
    with pytest.raises(
        expected_exception=ValidationError,
        match="type=set_type|type=string_type",
    ):
        serie.countries = countries


class TestOpenTimeSeries(TestCase):

    """class to run unittests on the module series.py."""

    randomseries: OpenTimeSeries
    random_properties: dict[str, Union[dt.date, int, float]]

    @classmethod
    def setUpClass(cls: type[TestOpenTimeSeries]) -> None:
        """SetUpClass for the TestOpenTimeSeries class."""
        cls.randomseries = SIMSERIES.from_deepcopy()

        cls.random_properties = cls.randomseries.all_properties().to_dict()[
            ("Asset_0", ValueType.PRICE)
        ]

    def test_setup_class(self: TestOpenTimeSeries) -> None:
        """Test setup_class method."""
        with pytest.raises(
            expected_exception=ValidationError,
            match="String should have at least 3 characters",
        ):
            OpenTimeSeries.setup_class(domestic_ccy="12")

        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be a valid string",
        ):
            OpenTimeSeries.setup_class(domestic_ccy=cast(str, 12))

        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be a valid list|String should match pattern",
        ):
            OpenTimeSeries.setup_class(countries="12")

        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be a valid string",
        ):
            OpenTimeSeries.setup_class(
                countries=cast(CountriesType, ["SE", cast(str, 12)]),
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="2 validation errors for Countries",
        ):
            OpenTimeSeries.setup_class(countries=cast(CountriesType, ["SE", "12"]))

        with pytest.raises(
            expected_exception=ValidationError,
            match="2 validation errors for Countries",
        ):
            OpenTimeSeries.setup_class(countries=cast(CountriesType, None))

        OpenTimeSeries.setup_class(domestic_ccy="USD", countries="US")
        if OpenTimeSeries.domestic != "USD":
            msg = "Method setup_class() not working as intended"
            raise ValueError(msg)

        if OpenTimeSeries.countries != "US":
            msg = "Method setup_class() not working as intended"
            raise ValueError(msg)

    def test_invalid_dates(self: TestOpenTimeSeries) -> None:
        """Test invalid dates as input."""
        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be a valid string",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=["2023-01-01", cast(str, None)],
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="must be called with a collection of some kind",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=cast(list[str], None),
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=TypeError,
            match="must be called with a collection of some kind",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=cast(list[str], "2023-01-01"),
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Unknown datetime string format",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=["2023-01-bb", "2023-01-02"],
                values=[1.0, 1.1],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Shape of passed values is",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=[],
                values=[1.0, 1.1],
            )

    def test_invalid_values(self: TestOpenTimeSeries) -> None:
        """Test invalid values as input."""
        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be a valid number",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=["2023-01-01", "2023-01-02"],
                values=[1.0, cast(float, None)],
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be a valid list",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=["2023-01-01", "2023-01-02"],
                values=cast(list[float], None),
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be a valid list",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=["2023-01-01", "2023-01-02"],
                values=cast(list[float], 1.0),
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="could not convert string to float",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=["2023-01-01", "2023-01-02"],
                values=[1.0, cast(float, "bb")],
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="There must be at least 1 value",
        ):
            OpenTimeSeries.from_arrays(
                name="Asset_0",
                dates=["2023-01-01", "2023-01-02"],
                values=[],
            )

    def test_dates_values_length_mismatch(self: TestOpenTimeSeries) -> None:
        """Test dates and values input."""
        with pytest.raises(
            expected_exception=ValidationError,
            match="Number of dates and values passed do not match",
        ):
            OpenTimeSeries(
                name="Asset_0",
                timeseries_id="",
                instrument_id="",
                valuetype=ValueType.PRICE,
                currency="SEK",
                local_ccy=True,
                dates=["2023-01-01", "2023-01-02"],
                values=[1.0, 1.1, 1.05],
                tsdf=DataFrame(
                    data=[1.0, 1.1],
                    index=[
                        deyt.date()
                        for deyt in DatetimeIndex(["2023-01-01", "2023-01-02"])
                    ],
                    columns=[["Asset_0"], [ValueType.PRICE]],
                    dtype="float64",
                ),
            )
        with pytest.raises(
            expected_exception=ValidationError,
            match="Number of dates and values passed do not match",
        ):
            OpenTimeSeries(
                name="Asset_0",
                timeseries_id="",
                instrument_id="",
                valuetype=ValueType.PRICE,
                currency="SEK",
                local_ccy=True,
                dates=["2023-01-01", "2023-01-02", "2023-01-03"],
                values=[1.0, 1.1],
                tsdf=DataFrame(
                    data=[1.0, 1.1],
                    index=[
                        deyt.date()
                        for deyt in DatetimeIndex(["2023-01-01", "2023-01-02"])
                    ],
                    columns=[["Asset_0"], [ValueType.PRICE]],
                    dtype="float64",
                ),
            )
        with pytest.raises(
            expected_exception=ValidationError,
            match="Number of dates and values passed do not match",
        ):
            OpenTimeSeries(
                name="Asset_0",
                timeseries_id="",
                instrument_id="",
                valuetype=ValueType.PRICE,
                currency="SEK",
                local_ccy=True,
                dates=["2023-01-01", "2023-01-02"],
                values=[1.0, 1.1],
                tsdf=DataFrame(),
            )

    def test_duplicates_handling(self: TestOpenTimeSeries) -> None:
        """Test duplicate handling."""
        dates = [
            "2017-05-29",
            "2017-05-30",
            "2017-05-31",
            "2017-06-01",
            "2017-06-01",
        ]
        values = [
            100.0000,
            100.0978,
            100.2821,
            100.1741,
            100.4561,
        ]

        with pytest.raises(
            expected_exception=ValidationError,
            match="Dates are not unique",
        ):
            _ = OpenTimeSeries.from_arrays(
                name="Series",
                dates=dates,
                values=values,
            )

    def test_valid_tsdf(self: TestOpenTimeSeries) -> None:
        """Test valid pandas.DataFrame property."""
        dframe = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[["Asset_0"], [ValueType.PRICE]],
            dtype="float64",
        )
        serie = Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name=("Asset_0", ValueType.PRICE),
            dtype="float64",
        )
        data = {
            "timeseries_id": "",
            "currency": "SEK",
            "dates": [
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            "instrument_id": "",
            "isin": "",
            "local_ccy": True,
            "name": "Asset_0",
            "values": [1.0, 1.01, 0.99, 1.015, 1.003],
            "valuetype": ValueType.PRICE,
        }
        df_data = {"tsdf": dframe, **data}
        serie_data = {"tsdf": serie, **data}

        df_obj = OpenTimeSeries(**df_data)  # type: ignore[arg-type,unused-ignore]
        if list(df_obj.tsdf.to_numpy()) != df_obj.values:  # noqa: PD011
            msg = "Raw values and DataFrame values not matching"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValidationError,
            match="Input should be an instance of DataFrame",
        ):
            OpenTimeSeries(**serie_data)  # type: ignore[arg-type,unused-ignore]

    def test_create_from_arrays(self: TestOpenTimeSeries) -> None:
        """Test from_arrays construct method."""
        arrseries = OpenTimeSeries.from_arrays(
            name="arrseries",
            dates=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            values=[1.0, 1.01, 0.99, 1.015, 1.003],
        )
        if not isinstance(arrseries, OpenTimeSeries):
            msg = "Method from_arrays() not working as intended"
            raise TypeError(msg)

    def test_create_from_pd_dataframe(self: TestOpenTimeSeries) -> None:
        """Test construct from pandas.DataFrame."""
        df1 = DataFrame(
            data=[
                [1.0, 1.0],
                [1.01, 0.98],
                [0.99, 1.004],
                [1.015, 0.976],
                [1.003, 0.982],
            ],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=["Asset_0", "Asset_1"],
            dtype="float64",
        )
        df2 = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[["Asset_0"], [ValueType.PRICE]],
            dtype="float64",
        )
        df3 = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[[""], [ValueType.PRICE]],
            dtype="float64",
        )
        df4 = DataFrame(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            columns=[["Asset_0"], [None]],
            dtype="float64",
        )

        df1series = OpenTimeSeries.from_df(dframe=df1, column_nmbr=1)
        df2series = OpenTimeSeries.from_df(dframe=df2, column_nmbr=0)

        if not isinstance(df1series, OpenTimeSeries):
            msg = "Method from_df() not working as intended"
            raise TypeError(msg)
        if not isinstance(df2series, OpenTimeSeries):
            msg = "Method from_df() not working as intended"
            raise TypeError(msg)

        with self.assertLogs() as contextmgr:
            _ = OpenTimeSeries.from_df(dframe=df3, column_nmbr=0)

        if contextmgr.output != ["WARNING:root:Label missing. Adding: Series"]:
            msg = "OpenTimeSeries failed to log warning about label missing."
            raise ValueError(msg)

        with self.assertLogs() as contextmgr:
            _ = OpenTimeSeries.from_df(dframe=df4, column_nmbr=0)
        if contextmgr.output != [
            "WARNING:root:valuetype missing. Adding: Price(Close)",
        ]:
            msg = "OpenTimeSeries failed to log warning about valuetype missing."
            raise ValueError(msg)

        df3series = OpenTimeSeries.from_df(dframe=df3, column_nmbr=0)
        df4series = OpenTimeSeries.from_df(dframe=df4, column_nmbr=0)

        if not isinstance(df3series, OpenTimeSeries):
            msg = "Method from_df() not working as intended"
            raise TypeError(msg)
        if not isinstance(df4series, OpenTimeSeries):
            msg = "Method from_df() not working as intended"
            raise TypeError(msg)

    def test_create_from_pd_series(self: TestOpenTimeSeries) -> None:
        """Test construct from pandas.Series."""
        serie = Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name="Asset_0",
            dtype="float64",
        )
        sen = Series(
            data=[1.0, 1.01, 0.99, 1.015, 1.003],
            index=[
                "2019-06-24",
                "2019-06-25",
                "2019-06-26",
                "2019-06-27",
                "2019-06-28",
            ],
            name=("Asset_0", ValueType.PRICE),
            dtype="float64",
        )

        seseries = OpenTimeSeries.from_df(dframe=serie)
        senseries = OpenTimeSeries.from_df(dframe=sen)

        if not isinstance(seseries, OpenTimeSeries):
            msg = "Method from_df() not working as intended"
            raise TypeError(msg)
        if not isinstance(senseries, OpenTimeSeries):
            msg = "Method from_df() not working as intended"
            raise TypeError(msg)
        if seseries.label != senseries.label:
            msg = "Method from_df() not working as intended"
            raise ValueError(msg)

    def test_check_if_none(self: TestOpenTimeSeries) -> None:
        """Test _check_if_none function."""
        if not _check_if_none(None):
            msg = "Method _check_if_none() not working as intended"
            raise ValueError(msg)
        if _check_if_none(0.0):
            msg = "Method _check_if_none() not working as intended"
            raise ValueError(msg)

    def test_to_json(self: TestOpenTimeSeries) -> None:
        """Test to_json method."""
        filename = "seriessaved.json"
        if Path.home().joinpath("Documents").exists():
            directory = Path.home().joinpath("Documents")
            seriesfile = directory.joinpath(filename)
        else:
            directory = Path(__file__).resolve().parent
            seriesfile = directory.joinpath(filename)

        if Path(seriesfile).exists():
            msg = "test_to_json test case setup failed."
            raise FileExistsError(msg)

        kwargs = [
            {
                "what_output": "values",
                "filename": str(seriesfile),
            },
            {
                "what_output": "values",
                "filename": seriesfile,
            },
            {
                "what_output": "values",
                "filename": seriesfile,
                "directory": directory,
            },
        ]

        for kwarg in kwargs:
            data = self.randomseries.to_json(**kwarg)  # type: ignore[arg-type]
            if [item.get("name") for item in data] != ["Asset_0"]:
                msg = "Unexpected data from json"
                raise ValueError(msg)

            if not Path(seriesfile).exists():
                msg = "json file not created"
                raise FileNotFoundError(msg)

            seriesfile.unlink()

            if Path(seriesfile).exists():
                msg = "json file not deleted as intended"
                raise FileExistsError(msg)

    def test_to_json_and_back(self: TestOpenTimeSeries) -> None:
        """Test to_json method and creating an OpenTimeSeries from file data."""
        filename = "series.json"
        dirpath = Path(__file__).resolve().parent
        seriesfile = dirpath.joinpath(filename)

        if Path(seriesfile).exists():
            msg = "test_to_json_and_back test case setup failed."
            raise FileExistsError(msg)

        intended = "1.640116"

        data = self.randomseries.to_json(
            what_output="values",
            filename=filename,
            directory=dirpath,
        )

        series_one = next(
            OpenTimeSeries.from_arrays(
                name=item["name"],  # type: ignore[arg-type,unused-ignore]
                dates=item["dates"],  # type: ignore[arg-type,unused-ignore]
                values=item["values"],  # type: ignore[arg-type,unused-ignore]
                valuetype=ValueType.RTRN,
                baseccy=item["currency"],  # type: ignore[arg-type,unused-ignore]
                local_ccy=item["local_ccy"],  # type: ignore[arg-type,unused-ignore]
            ).to_cumret()
            for item in data
        )

        if f"{series_one.tsdf.iloc[-1, 0]:.6f}" != intended:
            msg = (
                "test_to_json_and_back did not output as intended: "
                f"{series_one.tsdf.iloc[-1, 0]:.6f}"
            )
            raise ValueError(msg)

        with Path.open(seriesfile, encoding="utf-8") as jsonfile:
            output = load(jsonfile)

        series_two = next(
            OpenTimeSeries.from_arrays(
                name=item["name"],
                dates=item["dates"],
                values=item["values"],
                valuetype=ValueType.RTRN,
                baseccy=item["currency"],
                local_ccy=item["local_ccy"],
            ).to_cumret()
            for item in output
        )

        if f"{series_two.tsdf.iloc[-1, 0]:.6f}" != intended:
            msg = (
                "test_to_json_and_back did not output as intended: "
                f"{series_two.tsdf.iloc[-1, 0]:.6f}"
            )
            raise ValueError(msg)

        if not Path(seriesfile).exists():
            msg = "json file not created"
            raise FileNotFoundError(msg)

        seriesfile.unlink()

        if Path(seriesfile).exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

    def test_to_json_and_back_tsdf(self: TestOpenTimeSeries) -> None:
        """Test to_json method and creating an OpenTimeSeries from file data."""
        filename = "series_tsdf.json"
        dirpath = Path(__file__).resolve().parent
        seriesfile = dirpath.joinpath(filename)

        if Path(seriesfile).exists():
            msg = "test_to_json_and_back_tsdf test case setup failed."
            raise FileExistsError(msg)

        intended = "1.640116"

        data = self.randomseries.to_json(
            what_output="tsdf",
            filename=filename,
            directory=dirpath,
        )

        series_one = next(
            OpenTimeSeries.from_arrays(
                name=item["name"],  # type: ignore[arg-type,unused-ignore]
                dates=item["dates"],  # type: ignore[arg-type,unused-ignore]
                values=item["values"],  # type: ignore[arg-type,unused-ignore]
                valuetype=item["valuetype"],  # type: ignore[arg-type,unused-ignore]
                baseccy=item["currency"],  # type: ignore[arg-type,unused-ignore]
                local_ccy=item["local_ccy"],  # type: ignore[arg-type,unused-ignore]
            ).to_cumret()
            for item in data
        )

        if f"{series_one.tsdf.iloc[-1, 0]:.6f}" != intended:
            msg = (
                "test_to_json_and_back_tsdf did not output as intended: "
                f"{series_one.tsdf.iloc[-1, 0]:.6f}"
            )
            raise ValueError(msg)

        with Path.open(seriesfile, encoding="utf-8") as jsonfile:
            output = load(jsonfile)

        series_two = next(
            OpenTimeSeries.from_arrays(
                name=item["name"],
                dates=item["dates"],
                values=item["values"],
                valuetype=item["valuetype"],
                baseccy=item["currency"],
                local_ccy=item["local_ccy"],
            ).to_cumret()
            for item in output
        )

        if f"{series_two.tsdf.iloc[-1, 0]:.6f}" != intended:
            msg = (
                "test_to_json_and_back_tsdf did not output as intended: "
                f"{series_two.tsdf.iloc[-1, 0]:.6f}"
            )
            raise ValueError(msg)

        if not Path(seriesfile).exists():
            msg = "json file not created"
            raise FileNotFoundError(msg)

        seriesfile.unlink()

        if Path(seriesfile).exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

    def test_create_from_fixed_rate(self: TestOpenTimeSeries) -> None:
        """Test from_fixed_rate construct method."""
        fixseries_one = OpenTimeSeries.from_fixed_rate(
            rate=0.03,
            days=756,
            end_dt=dt.date(2019, 6, 30),
        )
        if not isinstance(fixseries_one, OpenTimeSeries):
            msg = "Method from_fixed_rate() not working as intended"
            raise TypeError(msg)

        rnd_series = self.randomseries.from_deepcopy()
        fixseries_two = OpenTimeSeries.from_fixed_rate(
            rate=0.03,
            d_range=DatetimeIndex(rnd_series.tsdf.index),
        )
        if not isinstance(fixseries_two, OpenTimeSeries):
            msg = "Method from_fixed_rate() not working as intended"
            raise TypeError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="If d_range is not provided both days and end_dt must be.",
        ):
            _ = OpenTimeSeries.from_fixed_rate(rate=0.03)

        with pytest.raises(
            expected_exception=ValueError,
            match="If d_range is not provided both days and end_dt must be.",
        ):
            _ = OpenTimeSeries.from_fixed_rate(rate=0.03, days=30)

    def test_periods_in_a_year(self: TestOpenTimeSeries) -> None:
        """Test periods_in_a_year property."""
        calc = len(self.randomseries.dates) / (
            (self.randomseries.last_idx - self.randomseries.first_idx).days / 365.25
        )

        if calc != self.randomseries.periods_in_a_year:
            msg = "Property periods_in_a_year returned unexpected result"
            raise ValueError(msg)
        if f"{251.3720547945:.10f}" != f"{self.randomseries.periods_in_a_year:.10f}":
            msg = "Property periods_in_a_year returned unexpected result"
            raise ValueError(msg)

        all_prop = self.random_properties["periods_in_a_year"]
        if f"{all_prop:.10f}" != f"{self.randomseries.periods_in_a_year:.10f}":
            msg = "Property periods_in_a_year returned unexpected result"
            raise ValueError(msg)

    def test_yearfrac(self: TestOpenTimeSeries) -> None:
        """Test yearfrac property."""
        if f"{9.99315537303:.11f}" != f"{self.randomseries.yearfrac:.11f}":
            msg = "Property periods_in_a_year returned unexpected result"
            raise ValueError(msg)

        all_prop = self.random_properties["yearfrac"]
        if f"{all_prop:.11f}" != f"{self.randomseries.yearfrac:.11f}":
            msg = "Property periods_in_a_year returned unexpected result"
            raise ValueError(msg)

    def test_resample(self: TestOpenTimeSeries) -> None:
        """Test resample method."""
        rs_series = self.randomseries.from_deepcopy()
        intended_length: int = 121

        before = rs_series.value_ret

        rs_series.resample(freq="BME")

        if rs_series.length != intended_length:
            msg = "Method resample() not working as intended"
            raise ValueError(msg)

        if before != rs_series.value_ret:
            msg = "Method resample() not working as intended"
            raise ValueError(msg)

    def test_resample_to_business_period_ends(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test resample_to_business_period_ends method."""
        rsb_stubs_series = OpenTimeSeries.from_fixed_rate(
            rate=0.01,
            days=121,
            end_dt=dt.date(2023, 5, 15),
        )

        rsb_stubs_series.resample_to_business_period_ends(freq="BME")
        new_stubs_dates = rsb_stubs_series.tsdf.index.tolist()

        if new_stubs_dates != [
            dt.date(2023, 1, 15),
            dt.date(2023, 1, 31),
            dt.date(2023, 2, 28),
            dt.date(2023, 3, 31),
            dt.date(2023, 4, 28),
            dt.date(2023, 5, 15),
        ]:
            msg = "Method resample_to_business_period_ends() not working as intended"
            raise ValueError(msg)

        rsb_series = OpenTimeSeries.from_fixed_rate(
            rate=0.01,
            days=88,
            end_dt=dt.date(2023, 4, 28),
        )

        rsb_series.resample_to_business_period_ends(freq="BME")
        new_dates = rsb_series.tsdf.index.tolist()

        if new_dates != [
            dt.date(2023, 1, 31),
            dt.date(2023, 2, 28),
            dt.date(2023, 3, 31),
            dt.date(2023, 4, 28),
        ]:
            msg = "Method resample_to_business_period_ends() not working as intended"
            raise ValueError(msg)

    def test_calc_range_output(self: TestOpenTimeSeries) -> None:
        """Test output consistency after calc_range applied."""
        cseries = self.randomseries.from_deepcopy()

        dates = cseries.calc_range(months_offset=48)

        if [
            dates[0].strftime("%Y-%m-%d"),
            dates[1].strftime("%Y-%m-%d"),
        ] != ["2015-06-26", "2019-06-28"]:
            msg = "Method calc_range() not working as intended"
            raise ValueError(msg)

        dates = self.randomseries.calc_range(from_dt=dt.date(2016, 6, 30))

        if [
            dates[0].strftime("%Y-%m-%d"),
            dates[1].strftime("%Y-%m-%d"),
        ] != ["2016-06-30", "2019-06-28"]:
            msg = "Method calc_range() not working as intended"
            raise ValueError(msg)

        gr_0 = cseries.vol_func(months_from_last=48)

        cseries.model_config.update({"validate_assignment": False})
        cseries.dates = cseries.dates[-1008:]
        cseries.values = list(cseries.values)[-1008:]
        cseries.model_config.update({"validate_assignment": True})
        cseries.pandas_df()
        cseries.set_new_label(lvl_one=ValueType.RTRN)
        cseries.to_cumret()

        gr_1 = cseries.vol

        if f"{gr_0:.13f}" != f"{gr_1:.13f}":
            msg = "Method calc_range() not working as intended"
            raise ValueError(msg)

    def test_value_to_diff(self: TestOpenTimeSeries) -> None:
        """Test value_to_diff method."""
        diffseries = self.randomseries.from_deepcopy()
        diffseries.value_to_diff()
        values = [f"{nn[0]:.10f}" for nn in diffseries.tsdf.to_numpy()[:10]]
        checkdata = [
            "0.0000000000",
            "0.0034863455",
            "-0.0004729821",
            "-0.0003396582",
            "0.0033967554",
            "0.0034594087",
            "0.0024188282",
            "-0.0088325018",
            "-0.0033775007",
            "-0.0017085753",
        ]

        if values != checkdata:
            msg = f"Result from method value_to_diff() not as intended\n{values}"
            raise ValueError(msg)

    def test_value_to_ret(self: TestOpenTimeSeries) -> None:
        """Test value_to_ret method."""
        retseries = self.randomseries.from_deepcopy()
        retseries.value_to_ret()
        values = [f"{nn[0]:.10f}" for nn in retseries.tsdf.to_numpy()[:10]]
        checkdata = [
            "0.0000000000",
            "0.0034863455",
            "-0.0004713388",
            "-0.0003386378",
            "0.0033876977",
            "0.0034385352",
            "0.0023959946",
            "-0.0087282110",
            "-0.0033670084",
            "-0.0017090219",
        ]

        if values != checkdata:
            msg = f"Result from method value_to_ret() not as intended\n{values}"
            raise ValueError(msg)

    def test_valute_to_log(self: TestOpenTimeSeries) -> None:
        """Test value_to_log method."""
        logseries = self.randomseries.from_deepcopy()
        logseries.value_to_log()
        values = [f"{nn[0]:.10f}" for nn in logseries.tsdf.to_numpy()[:10]]
        checkdata = [
            "0.0000000000",
            "0.0034802823",
            "0.0030088324",
            "0.0026701373",
            "0.0060521096",
            "0.0094847466",
            "0.0118778754",
            "0.0031113505",
            "-0.0002613391",
            "-0.0019718230",
        ]

        if values != checkdata:
            msg = f"Result from method value_to_log() not as intended\n{values}"
            raise ValueError(msg)

    def test_all_properties(self: TestOpenTimeSeries) -> None:
        """Test all_properties method."""
        prop_index = [
            "vol",
            "last_idx",
            "geo_ret",
            "first_idx",
            "max_drawdown",
            "periods_in_a_year",
            "z_score",
            "downside_deviation",
            "worst",
            "value_ret",
            "ret_vol_ratio",
            "worst_month",
            "max_drawdown_date",
            "arithmetic_ret",
            "skew",
            "cvar_down",
            "sortino_ratio",
            "omega_ratio",
            "positive_share",
            "kurtosis",
            "vol_from_var",
            "max_drawdown_cal_year",
            "yearfrac",
            "var_down",
            "length",
            "span_of_days",
        ]
        apseries = self.randomseries.from_deepcopy()
        apseries.to_cumret()
        result = apseries.all_properties()

        result_index = result.index.tolist()
        if set(prop_index) != set(result_index):
            msg = "Method all_properties() not working as intended"
            raise ValueError(msg)

        result_values = {}
        for value in result.index:
            if isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], float):
                result_values[value] = (
                    f"{result.loc[value, ('Asset_0', ValueType.PRICE)]:.10f}"
                )
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], int):
                result_values[value] = result.loc[  # type: ignore[assignment,unused-ignore]
                    value,
                    ("Asset_0", ValueType.PRICE),
                ]
            elif isinstance(result.loc[value, ("Asset_0", ValueType.PRICE)], dt.date):
                result_values[value] = cast(
                    dt.date,
                    result.loc[
                        value,
                        ("Asset_0", ValueType.PRICE),
                    ],
                ).strftime("%Y-%m-%d")
            else:
                msg = f"all_properties returned unexpected type {type(value)}"
                raise TypeError(
                    msg,
                )
        expected_values = {
            "arithmetic_ret": "0.0585047569",
            "cvar_down": "-0.0123803429",
            "downside_deviation": "0.0667228073",
            "first_idx": "2009-06-30",
            "geo_ret": "0.0507567099",
            "kurtosis": "696.0965168893",
            "last_idx": "2019-06-28",
            "length": 2512,
            "max_drawdown": "-0.1314808074",
            "max_drawdown_cal_year": "-0.1292814491",
            "max_drawdown_date": "2012-11-21",
            "omega_ratio": "1.0983709757",
            "periods_in_a_year": "251.3720547945",
            "positive_share": "0.5057745918",
            "ret_vol_ratio": "0.4162058331",
            "skew": "19.1911712502",
            "sortino_ratio": "0.8768329634",
            "span_of_days": 3650,
            "value_ret": "0.6401159258",
            "var_down": "-0.0097182152",
            "vol": "0.1405668835",
            "vol_from_var": "0.0936737165",
            "worst": "-0.0191572882",
            "worst_month": "-0.0581245494",
            "yearfrac": "9.9931553730",
            "z_score": "0.3750685522",
        }

        if result_values != expected_values:
            msg = (
                "Unexpected results from "
                f"all_properties() method\n{pformat(result_values)}"
            )
            raise ValueError(msg)

        props = apseries.all_properties(properties=["geo_ret", "vol"])
        if not isinstance(props, DataFrame):
            msg = "Method all_properties() not working as intended"
            raise TypeError(msg)

        with pytest.raises(expected_exception=ValueError, match="Invalid string: boo"):
            _ = apseries.all_properties(
                cast(list[LiteralSeriesProps], ["geo_ret", "boo"]),
            )

    def test_all_calc_properties(self: TestOpenTimeSeries) -> None:
        """Test all calculated properties."""
        checks = {
            "cvar_down": "-0.0123803429",
            "downside_deviation": "0.0667228073",
            "geo_ret": "0.0507567099",
            "kurtosis": "696.0965168893",
            "max_drawdown": "-0.1314808074",
            "max_drawdown_cal_year": "-0.1292814491",
            "positive_share": "0.5057745918",
            "ret_vol_ratio": "0.4162058331",
            "skew": "19.1911712502",
            "sortino_ratio": "0.8768329634",
            "value_ret": "0.6401159258",
            "var_down": "-0.0097182152",
            "vol": "0.1405668835",
            "vol_from_var": "0.0936737165",
            "worst": "-0.0191572882",
            "worst_month": "-0.0581245494",
            "z_score": "0.3750685522",
        }
        audit = {}
        loop_msg = ""
        for c_key, c_value in checks.items():
            audit.update({c_key: f"{getattr(self.randomseries, c_key):.10f}"})
            if c_value != f"{getattr(self.randomseries, c_key):.10f}":
                loop_msg += (
                    f"\nDifference in {c_key}: "
                    f"'{Decimal(getattr(self.randomseries, c_key)):.10f}'"
                )
            if round(
                Decimal(cast(float, self.random_properties[c_key])),
                10,
            ) != round(
                Decimal(getattr(self.randomseries, c_key)),
                10,
            ):
                msg = (
                    f"Difference in {c_key}: "
                    f"{self.random_properties[c_key]:.10f}"
                    " versus "
                    f"{getattr(self.randomseries, c_key):.10f}"
                )
                raise ValueError(msg)
        if loop_msg != "":
            loop_msg += f"\n{pformat(audit)}"
            raise ValueError(loop_msg)

    def test_all_calc_functions(self: TestOpenTimeSeries) -> None:
        """Test all calculation methods."""
        excel_geo_ret = (1.640115925775493 / 1.4387489280838568) ** (
            1 / ((dt.date(2019, 6, 28) - dt.date(2015, 6, 26)).days / 365.25)
        ) - 1
        checks = {
            "arithmetic_ret_func": "0.03770656022",
            "cvar_down_func": "-0.01265870645",
            "downside_deviation_func": "0.06871856382",
            "geo_ret_func": f"{excel_geo_ret:.11f}",
            "kurtosis_func": "-0.07991363073",
            "max_drawdown_func": "-0.12512526696",
            "positive_share_func": "0.50744786495",
            "ret_vol_ratio_func": "0.37802191976",
            "skew_func": "0.03894541564",
            "sortino_ratio_func": "0.54870995728",
            "value_ret_func": "0.13995978990",
            "var_down_func": "-0.01032629793",
            "vol_func": "0.09974702060",
            "vol_from_var_func": "0.09959111838",
            "worst_func": "-0.01915728825",
            "z_score_func": "0.54204277867",
        }
        audit = {}
        msg = ""
        for c_key, c_value in checks.items():
            calc = f"{getattr(self.randomseries, c_key)(months_from_last=48):.11f}"
            audit.update({c_key: calc})
            if (
                c_value
                != f"{getattr(self.randomseries, c_key)(months_from_last=48):.11f}"
            ):
                msg += (
                    f"Difference in {c_key}: "
                    f"'{getattr(self.randomseries, c_key)(months_from_last=48):.11f}'"
                )
        if msg != "":
            msg += f"\n{pformat(audit)}"
            raise ValueError(msg)

        func = "value_ret_calendar_period"
        if f"{getattr(self.randomseries, func)(year=2019):.12f}" != "0.039890004088":
            msg = (
                f"Unexpected result from method {func}(): "
                f"'{getattr(self.randomseries, func)(year=2019):.12f}'"
            )
            raise ValueError(msg)

    def test_max_drawdown_date(self: TestOpenTimeSeries) -> None:
        """Test max_drawdown_date property."""
        if self.randomseries.max_drawdown_date != dt.date(2012, 11, 21):
            msg = (
                "Unexpected max_drawdown_date: "
                f"'{self.randomseries.max_drawdown_date}'"
            )
            raise ValueError(msg)

        all_prop = self.random_properties["max_drawdown_date"]
        if self.randomseries.max_drawdown_date != all_prop:
            msg = (
                "Unexpected max_drawdown_date: "
                f"'{self.randomseries.max_drawdown_date}'"
            )
            raise ValueError(msg)

    def test_running_adjustment(self: TestOpenTimeSeries) -> None:
        """Test running_adjustment method."""
        adjustedseries = self.randomseries.from_deepcopy()
        adjustedseries.running_adjustment(0.05)

        if f"{cast(float, adjustedseries.tsdf.iloc[-1, 0]):.10f}" != "2.7036984198":
            msg = (
                "Unexpected result from running_adjustment(): "
                f"'{cast(float, adjustedseries.tsdf.iloc[-1, 0]):.10f}'"
            )
            raise ValueError(msg)
        adjustedseries_returns = self.randomseries.from_deepcopy()
        adjustedseries_returns.value_to_ret()
        adjustedseries_returns.running_adjustment(0.05)

        if (
            f"{cast(float, adjustedseries_returns.tsdf.iloc[-1, 0]):.10f}"
            != "0.0036950612"
        ):
            msg = (
                "Unexpected result from running_adjustment(): "
                f"'{cast(float, adjustedseries_returns.tsdf.iloc[-1, 0]):.10f}'"
            )
            raise ValueError(msg)

        adjustedseries_returns.to_cumret()
        if (
            f"{cast(float, adjustedseries.tsdf.iloc[-1, 0]):.10f}"
            != f"{cast(float, adjustedseries_returns.tsdf.iloc[-1, 0]):.10f}"
        ):
            msg = (
                "Unexpected result from running_adjustment(): "
                f"'{cast(float, adjustedseries.tsdf.iloc[-1, 0]):.10f}' versus "
                f"'{cast(float, adjustedseries_returns.tsdf.iloc[-1, 0]):.10f}'"
            )
            raise ValueError(msg)

    def test_timeseries_chain(self: TestOpenTimeSeries) -> None:
        """Test timeseries_chain function."""
        full_series = self.randomseries.from_deepcopy()
        full_values = [f"{nn:.10f}" for nn in full_series.tsdf.iloc[:, 0].tolist()]

        front_series = OpenTimeSeries.from_df(full_series.tsdf.iloc[:126])

        back_series = OpenTimeSeries.from_df(
            full_series.tsdf.iloc[
                cast(int, full_series.tsdf.index.get_loc(front_series.last_idx)) :
            ],
        )
        full_series.tsdf.index.get_loc(front_series.last_idx)
        chained_series = timeseries_chain(front_series, back_series)
        chained_values = [f"{nn:.10f}" for nn in list(chained_series.values)]

        if full_series.dates != chained_series.dates:
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

        if full_values != chained_values:
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

        pushed_date = front_series.last_idx + dt.timedelta(days=10)
        no_overlap_series = OpenTimeSeries.from_df(
            full_series.tsdf.loc[cast(int, pushed_date) :],
        )
        with pytest.raises(
            expected_exception=ValueError,
            match="Timeseries dates must overlap to allow them to be chained.",
        ):
            _ = timeseries_chain(front_series, no_overlap_series)

        front_series_two = OpenTimeSeries.from_df(full_series.tsdf.iloc[:136])
        front_series_two.resample(freq="8D")

        if back_series.first_idx in front_series_two.tsdf.index:
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

        new_chained_series = timeseries_chain(front_series_two, back_series)
        if not isinstance(new_chained_series, OpenTimeSeries):
            msg = "Function timeseries_chain() not working as intended"
            raise TypeError(msg)

        front_series_three = OpenTimeSeries.from_df(full_series.tsdf.iloc[:136])
        front_series_three.resample(freq="10D")

        if back_series.first_idx in front_series_three.tsdf.index:
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Failed to find a matching date between series",
        ):
            _ = timeseries_chain(front_series_three, back_series)

    def test_timeserieschain_newclass(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test correct pass-through of classes in timeseries_chain."""
        base_series_one = self.randomseries.from_deepcopy()

        sub_series_one = NewTimeSeries.from_arrays(
            name="sub_series_one",
            dates=base_series_one.dates,
            values=list(base_series_one.tsdf.iloc[:, 0].to_numpy()),
        )
        base_series_two = OpenTimeSeries.from_arrays(
            name="base_series_two",
            dates=[
                "2019-06-28",
                "2019-06-29",
                "2019-06-30",
            ],
            values=[
                1.0,
                1.009,
                1.011,
            ],
        )
        sub_series_two = NewTimeSeries.from_arrays(
            name="sub_series_two",
            dates=[
                "2019-06-28",
                "2019-06-29",
                "2019-06-30",
            ],
            values=[
                1.0,
                1.009,
                1.011,
            ],
        )
        if (
            sub_series_one.extra_info  # type: ignore[attr-defined, unused-ignore]
            != "cool"
        ):
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

        new_base = timeseries_chain(front=base_series_one, back=base_series_two)
        new_sub = timeseries_chain(front=sub_series_one, back=sub_series_two)

        if not isinstance(new_base, OpenTimeSeries):
            msg = "Function timeseries_chain() not working as intended"
            raise TypeError(msg)

        if not isinstance(new_sub, NewTimeSeries):
            msg = "Function timeseries_chain() not working as intended"
            raise TypeError(msg)

        if isinstance(new_base, NewTimeSeries):
            msg = "Function timeseries_chain() not working as intended"
            raise TypeError(msg)

        if new_sub.__class__.__subclasscheck__(OpenTimeSeries):
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

        if new_base.dates != new_sub.dates:
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

        if new_base.values != new_sub.values:  # noqa: PD011
            msg = "Function timeseries_chain() not working as intended"
            raise ValueError(msg)

    def test_chained_methods_newclass(self: TestOpenTimeSeries) -> None:
        """Test that chained methods on subclass returns subclass and not baseclass."""
        cseries = self.randomseries.from_deepcopy()
        if not isinstance(cseries, OpenTimeSeries):
            msg = "chained methods on subclass not working as intended"
            raise TypeError(msg)

        copyseries = NewTimeSeries.from_arrays(
            name="moo",
            dates=cseries.dates,
            values=list(cseries.tsdf.iloc[:, 0].to_numpy()),
        )
        if not isinstance(copyseries, NewTimeSeries):
            msg = "chained methods on subclass not working as intended"
            raise TypeError(msg)

        copyseries.set_new_label("boo").running_adjustment(0.001).resample(
            "BME",
        ).value_to_ret()
        if not isinstance(copyseries, NewTimeSeries):
            msg = "chained methods on subclass not working as intended"
            raise TypeError(msg)

    def test_plot_series(self: TestOpenTimeSeries) -> None:
        """Test plot_series method."""
        plotseries = self.randomseries.from_deepcopy()

        directory = Path(__file__).resolve().parent
        _, figfile = plotseries.plot_series(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "json file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

        fig, _ = plotseries.plot_series(auto_open=False, output_type="div")
        fig_json = loads(cast(str, fig.to_json()))
        rawdata = [f"{x:.11f}" for x in plotseries.tsdf.iloc[1:5, 0]]
        fig_data = [f"{x:.11f}" for x in fig_json["data"][0]["y"][1:5]]
        if rawdata != fig_data:
            msg = "Unaligned data between original and data in Figure."
            raise ValueError(msg)

        fig_last, _ = plotseries.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
        )
        fig_last_json = loads(cast(str, fig_last.to_json()))
        last = fig_last_json["data"][-1]["y"][0]

        if f"{last:.10f}" != "1.6401159258":
            msg = f"Unaligned data between original and data in Figure: '{last:.10f}'"
            raise ValueError(msg)

        if fig_last_json["data"][-1]["hovertemplate"] != "%{y}<br>%{x|%Y-%m-%d}":
            msg = "plot_series hovertemplate not as expected"
            raise ValueError(msg)

        fig_last_fmt, _ = plotseries.plot_series(
            auto_open=False,
            output_type="div",
            show_last=True,
            tick_fmt=".3%",
        )
        fig_last_fmt_json = loads(cast(str, fig_last_fmt.to_json()))
        last_fmt = fig_last_fmt_json["data"][-1]["text"][0]

        if last_fmt != "Last 164.012%":
            msg = f"Unaligned data between original and data in Figure: '{last_fmt}'"
            raise ValueError(msg)

        if (
            fig_last_fmt_json["data"][-1]["hovertemplate"]
            != "%{y:.3%}<br>%{x|%Y-%m-%d}"
        ):
            msg = "plot_series hovertemplate not as expected"
            raise ValueError(msg)

    def test_plot_bars(self: TestOpenTimeSeries) -> None:
        """Test plot_bars method."""
        barseries = self.randomseries.from_deepcopy()
        barseries.resample(freq="BME").value_to_ret()
        rawdata = [f"{x:.11f}" for x in barseries.tsdf.iloc[1:5, 0]]

        directory = Path(__file__).resolve().parent
        _, figfile = barseries.plot_bars(auto_open=False, directory=directory)
        plotfile = Path(figfile).resolve()
        if not plotfile.exists():
            msg = "json file not created"
            raise FileNotFoundError(msg)

        plotfile.unlink()
        if plotfile.exists():
            msg = "json file not deleted as intended"
            raise FileExistsError(msg)

        fig_keys = ["hovertemplate", "name", "type", "x", "y"]
        fig, _ = barseries.plot_bars(auto_open=False, output_type="div")
        fig_json = loads(cast(str, fig.to_json()))
        fig_data = [f"{x:.11f}" for x in fig_json["data"][0]["y"][1:5]]
        if rawdata != fig_data:
            msg = "Unaligned data between original and data in Figure."
            raise ValueError(msg)

        made_fig_keys = list(fig_json["data"][0].keys())
        made_fig_keys.sort()
        if made_fig_keys != fig_keys:
            msg = "Data in Figure not as intended."
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Must provide same number of labels as items in frame.",
        ):
            _, _ = barseries.plot_bars(auto_open=False, labels=["a", "b"])

        overlayfig, _ = barseries.plot_bars(
            auto_open=False,
            output_type="div",
            mode="overlay",
        )
        overlayfig_json = loads(cast(str, overlayfig.to_json()))

        fig_keys.append("opacity")
        if sorted(overlayfig_json["data"][0].keys()) != sorted(fig_keys):
            msg = "Data in Figure not as intended."
            raise ValueError(msg)

    def test_align_index_to_local_cdays(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test align_index_to_local_cdays method."""
        d_range = [d.date() for d in date_range(start="2020-06-15", end="2020-06-25")]
        asim = [1.0] * len(d_range)
        adf = DataFrame(
            data=asim,
            index=d_range,
            columns=[["Asset_0"], [ValueType.PRICE]],
        )
        aseries = OpenTimeSeries.from_df(adf, valuetype=ValueType.PRICE)

        if aseries.countries != "SE":
            msg = "Base case test_align_index_to_local_cdays not set up as intended"
            raise ValueError(msg)

        midsummer = dt.date(2020, 6, 19)
        if midsummer not in d_range:
            msg = "Date range generation not run as intended"
            raise ValueError(msg)

        aseries.align_index_to_local_cdays()
        if midsummer in aseries.tsdf.index:
            msg = "Method align_index_to_local_cdays() not working as intended"
            raise ValueError(msg)

    def test_ewma_vol_func(self: TestOpenTimeSeries) -> None:
        """Test ewma_vol_func method."""
        simdata = self.randomseries.ewma_vol_func()
        values = [f"{v:.11f}" for v in simdata.iloc[:5]]
        checkdata = [
            "0.06250431742",
            "0.06208916909",
            "0.06022552031",
            "0.05840562180",
            "0.05812960782",
        ]

        if values != checkdata:
            msg = f"Result from method ewma_vol_func() not as intended\n{values}"
            raise ValueError(msg)

        simdata_fxd_per_yr = self.randomseries.ewma_vol_func(
            periods_in_a_year_fixed=251,
        )
        values_fxd_per_yr = [f"{v:.11f}" for v in simdata_fxd_per_yr.iloc[:5]]
        checkdata_fxd_per_yr = [
            "0.06245804409",
            "0.06204320311",
            "0.06018093403",
            "0.05836238283",
            "0.05808657319",
        ]

        if values_fxd_per_yr != checkdata_fxd_per_yr:
            msg = (
                "Result from method ewma_vol_func() "
                f"not as intended\n{values_fxd_per_yr}"
            )
            raise ValueError(msg)

    def test_downside_deviation(self: TestOpenTimeSeries) -> None:
        """
        Test downside_deviation_func method.

        Source: https://www.investopedia.com/terms/d/downside-deviation.asp.
        """
        dd_asset = OpenTimeSeries.from_arrays(
            name="asset",
            valuetype=ValueType.RTRN,
            baseccy="USD",
            dates=[
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
                "2016-12-31",
                "2017-12-31",
                "2018-12-31",
                "2019-12-31",
            ],
            values=[
                0.0,
                -0.02,
                0.16,
                0.31,
                0.17,
                -0.11,
                0.21,
                0.26,
                -0.03,
                0.38,
            ],
        ).to_cumret()

        mar = 0.01
        downdev = dd_asset.downside_deviation_func(
            min_accepted_return=mar,
            periods_in_a_year_fixed=1,
        )

        if f"{downdev:.10f}" != "0.0433333333":
            msg = f"Unexpected result from downside_deviation_func() {downdev:.10f}"
            raise ValueError(msg)

    def test_omega_ratio(self: TestOpenTimeSeries) -> None:
        """
        Test omega_ratio_func method.

        Source:
        https://breakingdownfinance.com/finance-topics/
        performance-measurement/omega-ratio/
        """
        or_asset = OpenTimeSeries.from_arrays(
            name="asset",
            valuetype=ValueType.RTRN,
            baseccy="USD",
            dates=[
                "1999-12-31",
                "2000-12-31",
                "2001-12-31",
                "2002-12-31",
                "2003-12-31",
                "2004-12-31",
                "2005-12-31",
                "2006-12-31",
                "2007-12-31",
                "2008-12-31",
                "2009-12-31",
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
                "2016-12-31",
                "2017-12-31",
            ],
            values=[
                1.0000,
                1.0422,
                0.9722,
                1.1002,
                1.0067,
                1.2290,
                1.5040,
                1.6755,
                1.9732,
                2.4217,
                2.6145,
                2.6636,
                2.9385,
                3.3328,
                2.9826,
                2.9775,
                3.1627,
                3.0460,
                3.8614,
            ],
        )

        mar = 0.03
        omega = or_asset.omega_ratio_func(
            min_accepted_return=mar,
        )

        if f"{omega:.10f}" != "3.1163413842":
            msg = f"Unexpected result from omega_ratio_func(): {omega:.10f}"
            raise ValueError(msg)

    def test_validations(self: TestOpenTimeSeries) -> None:
        """Test input validations."""
        valid_instrument_id = "58135911b239b413482758c9"
        invalid_instrument_id_one = "58135911b239b413482758c"
        invalid_instrument_id_two = "5_135911b239b413482758c9"
        valid_timeseries_id = "5813595971051506189ba416"
        invalid_timeseries_id_one = "5813595971051506189ba41"
        invalid_timeseries_id_two = "5_13595971051506189ba416"

        basecase = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2017-05-29"],
            values=[100.0],
        )
        if basecase.dates != ["2017-05-29"]:
            msg = "Validations base case setup failed"
            raise ValueError(msg)

        if basecase.values != [100.0]:  # noqa: PD011
            msg = "Validations base case setup failed"
            raise ValueError(msg)

        basecase.countries = cast(CountriesType, ["SE", "US"])
        if basecase.countries != {"SE", "US"}:
            msg = "Validations base case setup failed"
            raise ValueError(msg)

        basecase.countries = cast(CountriesType, ["SE", "SE"])
        if basecase.countries != {"SE"}:
            msg = "Validations base case setup failed"
            raise ValueError(msg)

        with pytest.raises(
            expected_exception=ValueError,
            match="Shape of passed values is",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                timeseries_id=valid_timeseries_id,
                instrument_id=valid_instrument_id,
                dates=[],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="There must be at least 1 value",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[],
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=invalid_timeseries_id_one,
                instrument_id=valid_instrument_id,
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                name="asset",
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=invalid_timeseries_id_two,
                instrument_id=valid_instrument_id,
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=valid_timeseries_id,
                instrument_id=invalid_instrument_id_one,
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="String should match pattern",
        ):
            OpenTimeSeries.from_arrays(
                timeseries_id=valid_timeseries_id,
                instrument_id=invalid_instrument_id_two,
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValidationError,
            match="Dates are not unique",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-29",
                ],
                values=[
                    100.0,
                    100.0978,
                ],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="Shape of passed values is",
        ):
            OpenTimeSeries.from_arrays(
                name="asset",
                dates=[
                    "2017-05-29",
                    "2017-05-30",
                ],
                values=[
                    100.0,
                    100.0978,
                    100.2821,
                ],
            )

    def test_from_1d_rate_to_cumret(self: TestOpenTimeSeries) -> None:
        """Test from_1d_rate_to_cumret method."""
        tms = OpenTimeSeries.from_arrays(
            name="asset",
            valuetype=ValueType.RTRN,
            dates=[
                "2022-12-05",
                "2022-12-06",
                "2022-12-07",
                "2022-12-08",
                "2022-12-09",
                "2022-12-12",
                "2022-12-13",
                "2022-12-14",
                "2022-12-15",
                "2022-12-16",
                "2022-12-19",
            ],
            values=[
                0.02434,
                0.02434,
                0.02434,
                0.02434,
                0.02434,
                0.02434,
                0.02434,
                0.02434,
                0.02434,
                0.02434,
                0.02434,
            ],
        )
        ave_rate = f"{tms.tsdf.mean().iloc[0]:.5f}"
        if ave_rate != "0.02434":
            msg = "from_1d_rate_to_cumret() base case setup failed"
            raise ValueError(msg)

        tms.from_1d_rate_to_cumret()

        val_ret = f"{tms.value_ret:.5f}"
        if val_ret != "0.00093":
            msg = "Unexpected result from from_1d_rate_to_cumret()"
            raise ValueError(msg)

    def test_geo_ret_value_ret_exceptions(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test georet property raising exceptions on bad input data."""
        geoseries = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2022-07-01", "2023-07-01"],
            values=[
                1.0,
                1.1,
            ],
        )
        if f"{geoseries.geo_ret:.7f}" != "0.1000718":
            msg = "Property geo_ret base case setup failed"
            raise ValueError(msg)

        if f"{geoseries.geo_ret_func():.7f}" != "0.1000718":
            msg = "Method geo_ret_func() base case setup failed"
            raise ValueError(msg)

        zeroseries = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2022-07-01", "2023-07-01"],
            values=[
                0.0,
                1.1,
            ],
        )
        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = zeroseries.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = zeroseries.geo_ret_func()

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = zeroseries.value_ret

        with pytest.raises(
            expected_exception=ValueError,
            match="Simple return cannot be calculated due to an",
        ):
            _ = zeroseries.value_ret_func()

        negseries = OpenTimeSeries.from_arrays(
            name="asset",
            dates=["2022-07-01", "2023-07-01"],
            values=[
                1.0,
                -0.1,
            ],
        )

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = negseries.geo_ret

        with pytest.raises(
            expected_exception=ValueError,
            match=(
                "Geometric return cannot be calculated due to an "
                "initial value being zero or a negative value."
            ),
        ):
            _ = negseries.geo_ret_func()

    def test_miscellaneous(self: TestOpenTimeSeries) -> None:
        """Test miscellaneous methods."""
        mseries = self.randomseries.from_deepcopy()
        zero_str: str = "0"

        methods = [
            "arithmetic_ret_func",
            "vol_func",
            "vol_from_var_func",
            "downside_deviation_func",
            "target_weight_from_var",
        ]
        for methd in methods:
            no_fixed = getattr(mseries, methd)()
            fixed = getattr(mseries, methd)(periods_in_a_year_fixed=252)
            if f"{100*abs(no_fixed-fixed):.0f}" != zero_str:
                msg = "Difference with or without fixed periods in year is too great"
                raise ValueError(msg)

        impvol = mseries.vol_from_var_func(drift_adjust=False)
        if f"{impvol:.12f}" != "0.093673716476":
            msg = "Unexpected result from method vol_from_var_func(): '{impvol:.12f}'"
            raise ValueError(msg)

        impvoldrifted = mseries.vol_from_var_func(drift_adjust=True)
        if f"{impvoldrifted:.12f}" != "0.095916216736":
            msg = (
                "Unexpected result from method vol_from_var_func(): "
                f"'{impvoldrifted:.12f}'"
            )
            raise ValueError(msg)

    def test_value_ret_calendar_period(
        self: TestOpenTimeSeries,
    ) -> None:
        """Test value_ret_calendar_period method."""
        vrcseries = self.randomseries.from_deepcopy()

        vrfs_y = vrcseries.value_ret_func(
            from_date=dt.date(2017, 12, 29),
            to_date=dt.date(2018, 12, 28),
        )
        vrvrcs_y = vrcseries.value_ret_calendar_period(year=2018)
        if f"{vrfs_y:.11f}" != f"{vrvrcs_y:.11f}":
            msg = (
                "Results from value_ret_func() and value_ret_calendar_period() "
                "not matching as expected"
            )
            raise ValueError(msg)

        vrfs_ym = vrcseries.value_ret_func(
            from_date=dt.date(2018, 4, 30),
            to_date=dt.date(2018, 5, 31),
        )
        vrvrcs_ym = vrcseries.value_ret_calendar_period(year=2018, month=5)
        if f"{vrfs_ym:.11f}" != f"{vrvrcs_ym:.11f}":
            msg = (
                "Results from value_ret_func() and value_ret_calendar_period() "
                "not matching as expected"
            )
            raise ValueError(msg)

    def test_to_drawdown_series(self: TestOpenTimeSeries) -> None:
        """Test to_drawdown_series method."""
        mseries = self.randomseries.from_deepcopy()
        ddvalue = mseries.max_drawdown
        mseries.to_drawdown_series()
        ddserievalue = float((mseries.tsdf.min()).iloc[0])
        if f"{ddvalue:.11f}" != f"{ddserievalue:.11f}":
            msg = (
                "Results from property max_drawdown and to_drawdown_series() "
                "not matching as expected"
            )
            raise ValueError(msg)

    def test_set_new_label(self: TestOpenTimeSeries) -> None:
        """Test set_new_label method."""
        lseries = self.randomseries.from_deepcopy()

        if cast(tuple[str, str], lseries.tsdf.columns[0]) != (
            "Asset_0",
            ValueType.PRICE,
        ):
            msg = "set_new_label() base case not working as intended"
            raise ValueError(msg)

        lseries.set_new_label(lvl_zero="zero")
        if lseries.tsdf.columns[0][0] != "zero":
            msg = "Method set_new_label() base case not working as intended"
            raise ValueError(msg)

        lseries.set_new_label(lvl_one=ValueType.RTRN)
        if lseries.tsdf.columns[0][1] != ValueType.RTRN:
            msg = "Method set_new_label() base case not working as intended"
            raise ValueError(msg)

        lseries.set_new_label(lvl_zero="two", lvl_one=ValueType.PRICE)
        if cast(tuple[str, str], lseries.tsdf.columns[0]) != ("two", ValueType.PRICE):
            msg = "Method set_new_label() base case not working as intended"
            raise ValueError(msg)

        lseries.set_new_label(delete_lvl_one=True)
        if lseries.tsdf.columns[0] != "two":
            msg = "Method set_new_label() base case not working as intended"
            raise ValueError(msg)
