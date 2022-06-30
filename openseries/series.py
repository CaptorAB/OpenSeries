# -*- coding: utf-8 -*-
import copy
import datetime as dt
import json
import jsonschema
from jsonschema.exceptions import ValidationError
import math
import numpy as np
import os
import pandas as pd
from pandas.core.series import Series
from pandas.tseries.offsets import CDay
from pathlib import Path
import plotly.graph_objs as go
from plotly.offline import plot
from stdnum import isin as isincode
from stdnum.exceptions import ValidationError as StdnumValidationError
import scipy.stats as ss
from typing import Union, List, TypedDict


from openseries.captor_open_api_sdk import CaptorOpenApiService
from openseries.datefixer import date_offset_foll, date_fix
from openseries.load_plotly import load_plotly_dict
from openseries.risk import (
    cvar_down,
    var_down,
    drawdown_series,
    drawdown_details,
)
from openseries.sweden_holidays import SwedenHolidayCalendar, holidays_sw


class TimeSerie(TypedDict, total=False):
    """Class to hold the type of input data for the OpenTimeSeries class."""

    _id: str
    instrumentId: str
    currency: str
    dates: List[str]
    domestic: str
    name: str
    isin: str
    label: str
    schema: dict
    sweden: SwedenHolidayCalendar
    valuetype: str
    values: List[float]
    local_ccy: bool
    tsdf: pd.DataFrame


class OpenTimeSeries(object):

    _id: str
    instrumentId: str
    currency: str
    dates: List[str]
    domestic: str
    name: str
    isin: str
    label: str
    schema: dict
    sweden: SwedenHolidayCalendar
    valuetype: str
    values: List[float]
    local_ccy: bool
    tsdf: pd.DataFrame

    @classmethod
    def setup_class(cls):
        """Sets the domestic currency and calendar of the user.

        The values default to SEK and a calendar for Sweden designed by Captor Fund Management.
        We have no plans to support other calendars within this project.
        """

        cls.domestic = "SEK"
        cls.sweden = SwedenHolidayCalendar(holidays_sw)

    def __init__(self, d: TimeSerie):
        """Instantiates an object of the class OpenTimeSeries \n

        The data can have daily frequency, but not more frequent.

        Parameters
        ----------
        d: TimeSerie
            A subclass of TypedDict with the required and optional parameters

        Returns
        -------
        OpenTimeSeries
            Object of the class OpenTimeSeries
        """

        schema_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "openseries.json"
        )
        with open(file=schema_file, mode="r", encoding="utf-8") as f:
            series_schema = json.load(f)

        try:
            jsonschema.validate(instance=d, schema=series_schema)
        except ValidationError as e:
            raise Exception(d.get("_id", None), d.get("name", None), e)

        if d.get("isin", None):
            try:
                isincode.validate(d["isin"])
            except StdnumValidationError as ee:
                raise Exception("Provided ISIN code is invalid.", ee)

        self.__dict__ = d

        if self.name != "":
            self.label = self.name

        self.pandas_df()

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            A representation of an OpenTimeSeries object
        """

        return (
            "{}(label={}, _id={}, valuetype={}, currency={}, start={}, "
            "end={})".format(
                self.__class__.__name__,
                self.label,
                self._id,
                self.valuetype,
                self.currency,
                self.first_idx.strftime("%Y-%m-%d"),
                self.last_idx.strftime("%Y-%m-%d"),
            )
        )

    @classmethod
    def from_open_api(
        cls,
        timeseries_id: str,
        label: str = "series",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ):
        """Fetches timeseries data from the Captor API /opentimeseries endpoint

        Parameters
        ----------
        timeseries_id: str
            Captor database timeseries id
        label : str, default: "series"
            Name for the timeseries
        baseccy : str, default: "SEK"
            The currency of the timeseries
        local_ccy: bool, default: True
            True if timeseries has not been converted into baseccy and False otherwise

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        captor = CaptorOpenApiService()
        data = captor.get_timeseries(timeseries_id)

        output = TimeSerie(
            _id=data["id"],
            name=label,
            currency=baseccy,
            instrumentId="",
            isin="",
            local_ccy=local_ccy,
            valuetype=data["type"],
            dates=data["dates"],
            values=[float(val) for val in data["values"]],
        )

        return cls(d=output)

    @classmethod
    def from_open_nav(
        cls, isin: str, valuetype: str = "Price(Close)", local_ccy: bool = True
    ):
        """Fetches timeseries data from the Captor API /nav endpoint

        Parameters
        ----------
        isin: str
            Captor database timeseries id
        valuetype : str, default: "Price(Close)"
            Type of timeseries
        local_ccy: bool, default: True
            True if timeseries has not been converted into baseccy and False otherwise

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        captor = CaptorOpenApiService()
        data = captor.get_nav(isin=isin)

        output = TimeSerie(
            _id=data["_id"],
            name=data["longName"],
            currency=data["currency"],
            instrumentId="",
            isin=isin,
            local_ccy=local_ccy,
            valuetype=valuetype,
            dates=data["dates"],
            values=[float(val) for val in data["navPerUnit"]],
        )

        return cls(d=output)

    @classmethod
    def from_open_fundinfo(
        cls,
        isin: str,
        report_date: dt.date = None,
        valuetype: str = "Price(Close)",
        local_ccy: bool = True,
    ):
        """Fetches timeseries data from the Captor API /fundinfo endpoint

        Parameters
        ----------
        isin: str
            Captor database timeseries id
        report_date : datetime.date, optional
            reportDate parameter. Defaults to today if not set
        valuetype : str, default: "Price(Close)"
            Type of timeseries
        local_ccy: bool, default: True
            True if timeseries has not been converted into baseccy and False otherwise

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        captor = CaptorOpenApiService()
        data = captor.get_fundinfo(isins=[isin], report_date=report_date)

        fundinfo = data[0]["classes"][0]

        if isin != fundinfo["isin"]:
            raise Exception(
                "Method OpenTimeSeries.from_open_fundinfo() returned {} instead of intended {}".format(
                    fundinfo["isin"], isin
                )
            )

        output = TimeSerie(
            _id="",
            name=fundinfo["name"],
            currency=fundinfo["navCurrency"],
            instrumentId="",
            isin=fundinfo["isin"],
            local_ccy=local_ccy,
            valuetype=valuetype,
            dates=fundinfo["returnTimeSeries"]["dates"],
            values=[float(val) for val in fundinfo["returnTimeSeries"]["values"]],
        )

        return cls(d=output)

    @classmethod
    def from_df(
        cls,
        df: Union[pd.DataFrame, pd.Series],
        column_nmbr: int = 0,
        valuetype: str = "Price(Close)",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ):
        """Creates a timeseries from a Pandas DataFrame or Series

        Parameters
        ----------
        df: Union[pd.DataFrame, pd.Series]
            Pandas DataFrame or Series
        column_nmbr : int, default: 0
            Using iloc[:, column_nmbr] to pick column
        valuetype : str, default: "Price(Close)"
            Name for the timeseries
        baseccy : str, default: "SEK"
            The currency of the timeseries
        local_ccy: bool, default: True
            True if timeseries has not been converted into baseccy and False otherwise

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if isinstance(df, Series):
            if isinstance(df.name, tuple):
                label, _ = df.name
            else:
                label = df.name
            values = df.values.tolist()
        else:
            values = df.iloc[:, column_nmbr].tolist()
            if isinstance(df.columns, pd.MultiIndex):
                label = df.columns.get_level_values(0).values[column_nmbr]
                valuetype = df.columns.get_level_values(1).values[column_nmbr]
            else:
                label = df.columns.values[column_nmbr]
        dates = [date_fix(d).strftime("%Y-%m-%d") for d in df.index]
        output = TimeSerie(
            _id="",
            currency=baseccy,
            instrumentId="",
            isin="",
            local_ccy=local_ccy,
            name=label,
            valuetype=valuetype,
            dates=dates,
            values=values,
        )

        return cls(d=output)

    @classmethod
    def from_frame(
        cls,
        frame,
        label: str,
        valuetype: str = "Price(Close)",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ):
        """Creates a timeseries from an openseries.frame.OpenFrame

        Parameters
        ----------
        frame: OpenFrame
            openseries.frame.OpenFrame
        label : str
            Name for the timeseries
        valuetype : str, default: "Price(Close)"
            Name for the timeseries
        baseccy : str, default: "SEK"
            The currency of the timeseries
        local_ccy: bool, default: True
            True if timeseries has not been converted into baseccy and False otherwise

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        df = frame.tsdf.loc[:, (label, valuetype)]
        dates = [date_fix(d).strftime("%Y-%m-%d") for d in df.index]

        output = TimeSerie(
            _id="",
            currency=baseccy,
            instrumentId="",
            isin="",
            local_ccy=local_ccy,
            name=df.name[0],
            valuetype=df.name[1],
            dates=dates,
            values=df.values.tolist(),
        )

        return cls(d=output)

    def from_deepcopy(self):
        """Creates a copy of an OpenTimeSeries object

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        return copy.deepcopy(self)

    @classmethod
    def from_fixed_rate(
        cls,
        rate: float,
        date_range: pd.DatetimeIndex = None,
        days: int = None,
        end_dt: dt.date = None,
        label: str = "Series",
        valuetype: str = "Price(Close)",
        baseccy: str = "SEK",
        local_ccy: bool = True,
    ):
        """Creates a timeseries from a series of values accruing with a given fixed rate

        Providing a date_range of type Pandas.DatetimeIndex takes priority over
        providing a combination of days and an end date.

        Parameters
        ----------
        rate: float
            The accrual rate
        date_range: pd.DatetimeIndex, optional
            A given range of dates
        days: int, optional
            Number of days to generate when date_range not provided. Must be combined with end_dt
        end_dt: datetime.date, optional
            End date of date range to generate when date_range not provided. Must be combined with days
        label : str
            Name for the timeseries
        valuetype : str, default: "Price(Close)"
            Name for the timeseries
        baseccy : str, default: "SEK"
            The currency of the timeseries
        local_ccy: bool, default: True
            True if timeseries has not been converted into baseccy and False otherwise

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        if date_range is None:
            date_range = pd.DatetimeIndex(
                [d.date() for d in pd.date_range(periods=days, end=end_dt, freq="D")]
            )
        deltas = np.array([i.days for i in date_range[1:] - date_range[:-1]])
        array = list(np.cumprod(np.insert(1 + deltas * rate / 365, 0, 1.0)))
        date_range = [d.strftime("%Y-%m-%d") for d in date_range]

        output = TimeSerie(
            _id="",
            name=label,
            currency=baseccy,
            instrumentId="",
            isin="",
            local_ccy=local_ccy,
            valuetype=valuetype,
            dates=date_range,
            values=array,
        )

        return cls(d=output)

    def to_json(self, filename: str, directory: str = None) -> dict:
        """Dumps timeseries data into a json file

        The label and tsdf parameters are deleted before the json file is saved

        Parameters
        ----------
        filename: str
            Filename including filetype
        directory: str, optional
            File folder location
        Returns
        -------
        dict
            A dictionary
        """
        if not directory:
            directory = os.path.dirname(os.path.abspath(__file__))

        data = self.__dict__

        cleaner_list = ["label", "tsdf"]
        for item in cleaner_list:
            data.pop(item)

        with open(os.path.join(directory, filename), "w") as ff:
            json.dump(data, ff, indent=2, sort_keys=False)

        return data

    def pandas_df(self):
        """Sets the .tsdf parameter as a Pandas DataFrame from the .dates and .values lists

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """
        df = pd.DataFrame(data=self.values, index=self.dates, dtype="float64")
        df.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        df.index = [d.date() for d in pd.DatetimeIndex(df.index)]

        df.sort_index(inplace=True)
        self.tsdf = df

        return self

    def calc_range(
        self,
        months_offset: int = None,
        from_dt: Union[dt.date, None] = None,
        to_dt: Union[dt.date, None] = None,
    ) -> (dt.date, dt.date):
        """Creates user defined date range

        Parameters
        ----------
        months_offset: int, optional
            Number of months offset as positive integer. Overrides use of from_date and to_date
        from_dt: datetime.date, optional
            Specific from date
        to_dt: datetime.date, optional
            Specific from date

        Returns
        -------
        (datetime.date, datetime.date)
            Start and end date of the chosen date range
        """

        self.setup_class()
        if months_offset is not None or from_dt is not None or to_dt is not None:
            if months_offset is not None:
                earlier = date_offset_foll(
                    self.last_idx,
                    calendar=CDay(calendar=self.sweden),
                    months_offset=-months_offset,
                )
                assert (
                    earlier >= self.first_idx
                ), "Function calc_range returned earlier date < series start"
                later = self.last_idx
            else:
                if from_dt is not None and to_dt is None:
                    assert from_dt >= self.first_idx, (
                        "Function calc_range returned earlier date < " "series start"
                    )
                    earlier, later = from_dt, self.last_idx
                elif from_dt is None and to_dt is not None:
                    assert (
                        to_dt <= self.last_idx
                    ), "Function calc_range returned later date > series end"
                    earlier, later = self.first_idx, to_dt
                elif from_dt is not None and to_dt is not None:
                    assert to_dt <= self.last_idx and from_dt >= self.first_idx, (
                        "Function calc_range returned dates outside " "series range"
                    )
                    earlier, later = from_dt, to_dt
                else:
                    earlier, later = from_dt, to_dt

            earlier = date_fix(earlier)
            later = date_fix(later)

            while not self.tsdf.index.isin([earlier]).any():
                earlier -= dt.timedelta(days=1)

            while not self.tsdf.index.isin([later]).any():
                later += dt.timedelta(days=1)

        else:
            earlier, later = self.first_idx, self.last_idx

        return earlier, later

    def align_index_to_local_cdays(self):
        """Changes the index of the associated Pandas DataFrame .tsdf to align with
        local calendar business days

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.setup_class()
        date_range = [
            d.date()
            for d in pd.date_range(
                start=self.tsdf.first_valid_index(),
                end=self.tsdf.last_valid_index(),
                freq=CDay(calendar=self.sweden),
            )
        ]

        self.tsdf = self.tsdf.reindex(date_range, method=None, copy=False)

        return self

    def all_properties(self, properties: list = None) -> pd.DataFrame:
        """Calculates the chosen timeseries properties

        Parameters
        ----------
        properties: list, optional
            The properties to calculate. Defaults to calculating all available.

        Returns
        -------
        pandas.DataFrame
            Pandas DataFrame
        """

        if not properties:
            properties = [
                "value_ret",
                "geo_ret",
                "arithmetic_ret",
                "vol",
                "downside_deviation",
                "ret_vol_ratio",
                "sortino_ratio",
                "z_score",
                "skew",
                "kurtosis",
                "positive_share",
                "var_down",
                "cvar_down",
                "vol_from_var",
                "worst",
                "worst_month",
                "max_drawdown_cal_year",
                "max_drawdown",
                "max_drawdown_date",
                "first_idx",
                "last_idx",
                "length",
                "span_of_days",
                "yearfrac",
                "periods_in_a_year",
            ]

        pdf = pd.DataFrame.from_dict(
            {x: getattr(self, x) for x in properties}, orient="index"
        )

        pdf.columns = self.tsdf.columns

        return pdf

    @property
    def length(self) -> int:
        """
        Returns
        -------
        int
            Number of observations
        """

        return len(self.tsdf.index)

    @property
    def first_idx(self) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The first date in the timeseries
        """

        return self.tsdf.index[0]

    @property
    def last_idx(self) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The last date in the timeseries
        """

        return self.tsdf.index[-1]

    @property
    def span_of_days(self) -> int:
        """
        Returns
        -------
        int
            Number of days from the first date to the last
        """

        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self) -> float:
        """
        Returns
        -------
        float
            Length of the timeseries expressed in years assuming all years have 365.25 days
        """

        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self) -> float:
        """
        Returns
        -------
        float
            The average number of observations per year
        """

        return self.length / self.yearfrac

    @property
    def geo_ret(self) -> float:
        """https://www.investopedia.com/terms/c/cagr.asp

        Returns
        -------
        float
            Geometric annualized return
        """

        if float(self.tsdf.loc[self.first_idx]) == 0.0:
            raise Exception("First data point == 0.0")
        return float(
            (self.tsdf.loc[self.last_idx] / self.tsdf.loc[self.first_idx])
            ** (1 / self.yearfrac)
            - 1
        )

    def geo_ret_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """https://www.investopedia.com/terms/c/cagr.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Geometric annualized return
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        if float(self.tsdf.loc[earlier]) == 0.0:
            raise Exception("First data point == 0.0")

        return float(
            (self.tsdf.loc[later] / self.tsdf.loc[earlier]) ** (1 / fraction) - 1
        )

    @property
    def arithmetic_ret(self) -> float:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp

        Returns
        -------
        float
            Arithmetic annualized log return
        """

        return float(np.log(self.tsdf).diff().mean() * self.periods_in_a_year)

    def arithmetic_ret_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        periods_in_a_year_fixed: int = None,
    ) -> float:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        float
            Arithmetic annualized log return
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[earlier:later].count(numeric_only=True)
            time_factor = how_many / fraction
        return float(np.log(self.tsdf.loc[earlier:later]).diff().mean() * time_factor)

    @property
    def value_ret(self) -> float:
        """
        Returns
        -------
        float
            Simple return
        """

        if float(self.tsdf.iloc[0]) == 0.0:
            raise Exception("First data point == 0.0")
        return float(self.tsdf.iloc[-1] / self.tsdf.iloc[0] - 1)

    def value_ret_func(
        self,
        logret: bool = False,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """
        Parameters
        ----------
        logret : bool, optional
            True for log return and False for simple return
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Simple return
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if float(self.tsdf.loc[earlier]) == 0.0:
            raise Exception("First data point == 0.0")
        if logret:
            ret = np.log(self.tsdf.loc[later] / self.tsdf.loc[earlier])
        else:
            ret = self.tsdf.loc[later] / self.tsdf.loc[earlier] - 1
        return float(ret)

    def value_ret_calendar_period(self, year: int, month: int = None) -> float:
        """
        Parameters
        ----------
        year : int
            Calendar year of the period to calculate.
        month : int, optional
            Calendar month of the period to calculate.

        Returns
        -------
        float
            Simple return for a specific calendar period
        """

        caldf = self.tsdf.copy()
        caldf.index = pd.DatetimeIndex(caldf.index)
        if month is None:
            period = str(year)
        else:
            period = "-".join([str(year), str(month).zfill(2)])
        rtn = caldf.copy().pct_change()
        rtn = rtn.loc[period] + 1
        return float(rtn.apply(np.cumprod, axis="index").iloc[-1] - 1)

    @property
    def vol(self) -> float:
        """Based on Pandas .std() which is the equivalent of stdev.s([...]) in MS Excel \n
        https://www.investopedia.com/terms/v/volatility.asp

        Returns
        -------
        float
            Annualized volatility
        """

        return float(self.tsdf.pct_change().std() * np.sqrt(self.periods_in_a_year))

    def vol_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        periods_in_a_year_fixed: int = None,
    ) -> float:
        """Based on Pandas .std() which is the equivalent of stdev.s([...]) in MS Excel \n
        https://www.investopedia.com/terms/v/volatility.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        float
            Annualized volatility
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[earlier:later].count(numeric_only=True)
            time_factor = how_many / fraction

        return float(
            self.tsdf.loc[earlier:later].pct_change().std() * np.sqrt(time_factor)
        )

    @property
    def downside_deviation(self) -> float:
        """The standard deviation of returns that are below a Minimum Accepted Return of zero.
        It is used to calculate the Sortino Ratio \n
        https://www.investopedia.com/terms/d/downside-deviation.asp

        Returns
        -------
        float
            Downside deviation
        """

        dddf = self.tsdf.pct_change()

        return float(
            math.sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / self.length)
            * np.sqrt(self.periods_in_a_year)
        )

    def downside_deviation_func(
        self,
        min_accepted_return: float = 0.0,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        periods_in_a_year_fixed: int = None,
    ) -> float:
        """The standard deviation of returns that are below a Minimum Accepted Return of zero.
        It is used to calculate the Sortino Ratio \n
        https://www.investopedia.com/terms/d/downside-deviation.asp

        Parameters
        ----------
        min_accepted_return : float, optional
            The annualized Minimum Accepted Return (MAR)
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        float
            Downside deviation
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        how_many = float(
            self.tsdf.loc[earlier:later].pct_change().count(numeric_only=True)
        )
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction

        dddf = (
            self.tsdf.loc[earlier:later]
            .pct_change()
            .sub(min_accepted_return / time_factor)
        )

        return float(
            math.sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / how_many)
            * np.sqrt(time_factor)
        )

    @property
    def ret_vol_ratio(self) -> float:
        """
        Returns
        -------
        float
            Ratio of geometric return and annualized volatility
        """

        return self.geo_ret / self.vol

    def ret_vol_ratio_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        riskfree_rate: float = 0.0,
    ) -> float:
        """The ratio of geometric return and annualized volatility or, if risk free return
        provided, Sharpe ratio calculated as ( geometric return - risk free return )
        / volatility. The latter ratio implies that the riskfree asset has
        zero volatility \n
        https://www.investopedia.com/terms/s/sharperatio.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        riskfree_rate : float, optional
            The return of the zero volatility asset used to calculate Sharpe ratio

        Returns
        -------
        float
            Ratio of geometric return and annualized volatility or,
            if risk free return provided, Sharpe ratio
        """

        return (
            self.geo_ret_func(months_from_last, from_date, to_date) - riskfree_rate
        ) / self.vol_func(months_from_last, from_date, to_date)

    @property
    def sortino_ratio(self) -> float:
        """https://www.investopedia.com/terms/s/sortinoratio.asp

        Returns
        -------
        float
            Ratio of geometric return and downside deviation with a riskfree rate of zero
        """

        return self.geo_ret / self.downside_deviation

    def sortino_ratio_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        riskfree_rate: float = 0.0,
    ) -> float:
        """The Sortino ratio calculated as ( geometric return - risk free return )
        / downside deviation. The ratio implies that the riskfree asset has
        zero volatility, and a minimum acceptable return of zero \n
        https://www.investopedia.com/terms/s/sortinoratio.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        riskfree_rate : float, optional
            The return of the zero volatility asset

        Returns
        -------
        float
            The Sortino ratio calculated as ( geometric return - risk free return ) / downside deviation
        """

        return (
            self.geo_ret_func(months_from_last, from_date, to_date) - riskfree_rate
        ) / self.downside_deviation_func(
            min_accepted_return=0.0,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
        )

    @property
    def z_score(self) -> float:
        """https://www.investopedia.com/terms/z/zscore.asp

        Returns
        -------
        float
            Z-score as (last return - mean return) / standard deviation of returns.
        """

        return float(
            (self.tsdf.pct_change().iloc[-1] - self.tsdf.pct_change().mean())
            / self.tsdf.pct_change().std()
        )

    def z_score_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """https://www.investopedia.com/terms/z/zscore.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Z-score as (last return - mean return) / standard deviation of returns
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        part = self.tsdf.loc[earlier:later].pct_change().copy()
        return float((part.iloc[-1] - part.mean()) / part.std())

    @property
    def max_drawdown(self) -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        float
            Maximum drawdown without any limit on date range
        """

        return float((self.tsdf / self.tsdf.expanding(min_periods=1).max()).min() - 1)

    @property
    def max_drawdown_date(self) -> dt.date:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        datetime.date
            Date when the maximum drawdown occurred
        """

        mdddf = self.tsdf.copy()
        mdddf.index = pd.DatetimeIndex(mdddf.index)
        mdd_date = (
            (mdddf / mdddf.expanding(min_periods=1).max())
            .idxmin()
            .values[0]
            .astype(dt.datetime)
        )
        return dt.datetime.fromtimestamp(mdd_date / 1e9).date()

    def max_drawdown_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Maximum drawdown without any limit on date range
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(
            (
                self.tsdf.loc[earlier:later]
                / self.tsdf.loc[earlier:later].expanding(min_periods=1).max()
            ).min()
            - 1
        )

    @property
    def max_drawdown_cal_year(self) -> float:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        float
            Maximum drawdown in a single calendar year.
        """

        return float(
            self.tsdf.groupby([pd.DatetimeIndex(self.tsdf.index).year])
            .apply(lambda x: (x / x.expanding(min_periods=1).max()).min() - 1)
            .min()
        )

    @property
    def worst(self) -> float:
        """
        Returns
        -------
        float
            Most negative percentage change
        """

        return float(self.tsdf.pct_change().min())

    @property
    def worst_month(self) -> float:
        """
        Returns
        -------
        float
            Most negative month
        """

        resdf = self.tsdf.copy()
        resdf.index = pd.DatetimeIndex(resdf.index)
        return float(resdf.resample("BM").last().pct_change().min())

    def worst_func(
        self,
        observations: int = 1,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """
        Parameters
        ----------
        observations: int, default: 1
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Most negative percentage change over a rolling number of observations within
            a chosen date range
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(
            self.tsdf.loc[earlier:later]
            .pct_change()
            .rolling(observations, min_periods=observations)
            .sum()
            .min()
        )

    @property
    def positive_share(self) -> float:
        """
        Returns
        -------
        float
            The share of percentage changes that are greater than zero
        """
        pos = self.tsdf.pct_change()[1:][
            self.tsdf.pct_change()[1:].values >= 0.0
        ].count()
        tot = self.tsdf.pct_change()[1:].count()
        return float(pos / tot)

    def positive_share_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """
        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            The share of percentage changes that are greater than zero
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        period = self.tsdf.loc[earlier:later].copy()
        return float(
            period[period.pct_change().ge(0.0)].count(numeric_only=True)
            / period.pct_change().count(numeric_only=True)
        )

    @property
    def skew(self) -> float:
        """https://www.investopedia.com/terms/s/skewness.asp

        Returns
        -------
        float
            Skew of the return distribution
        """

        return float(
            ss.skew(self.tsdf.pct_change().values, bias=True, nan_policy="omit")
        )

    def skew_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """https://www.investopedia.com/terms/s/skewness.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Skew of the return distribution
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(
            ss.skew(
                self.tsdf.loc[earlier:later].pct_change(),
                bias=True,
                nan_policy="omit",
            )
        )

    @property
    def kurtosis(self) -> float:
        """https://www.investopedia.com/terms/k/kurtosis.asp

        Returns
        -------
        float
            Kurtosis of the return distribution
        """
        return float(
            ss.kurtosis(
                self.tsdf.pct_change(),
                fisher=True,
                bias=True,
                nan_policy="omit",
            )
        )

    def kurtosis_func(
        self,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """https://www.investopedia.com/terms/k/kurtosis.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Kurtosis of the return distribution
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)

        return float(
            ss.kurtosis(
                self.tsdf.loc[earlier:later].pct_change(),
                fisher=True,
                bias=True,
                nan_policy="omit",
            )
        )

    @property
    def cvar_down(self, level: float = 0.95) -> float:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

        Parameters
        ----------
        level: float, default: 0.95
            The sought CVaR level

        Returns
        -------
        float
            Downside Conditional Value At Risk "CVaR"
        """

        items = self.tsdf.iloc[:, 0].pct_change().count()
        return (
            self.tsdf.iloc[:, 0]
            .pct_change()
            .sort_values()
            .iloc[: int(math.ceil((1 - level) * items))]
            .mean()
        )

    def cvar_down_func(
        self,
        level: float = 0.95,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
    ) -> float:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

        Parameters
        ----------
        level: float, default: 0.95
            The sought CVaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        float
            Downside Conditional Value At Risk "CVaR"
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        how_many = (
            self.tsdf.loc[earlier:later, self.tsdf.columns.values[0]]
            .pct_change()
            .count()
        )
        return (
            self.tsdf.loc[earlier:later, self.tsdf.columns.values[0]]
            .pct_change()
            .sort_values()
            .iloc[: int(math.ceil((1 - level) * how_many))]
            .mean()
        )

    @property
    def var_down(self, level: float = 0.95, interpolation: str = "lower") -> float:
        """Downside Value At Risk, "VaR". The equivalent of
        percentile.inc([...], 1-level) over returns in MS Excel \n
        https://www.investopedia.com/terms/v/var.asp

        Parameters
        ----------

        level: float, default: 0.95
            The sought VaR level
        interpolation: str, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
            Default value is linear

        Returns
        -------
        float
            Downside Value At Risk
        """

        return float(
            self.tsdf.pct_change().quantile(1 - level, interpolation=interpolation)
        )

    def var_down_func(
        self,
        level: float = 0.95,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        interpolation: str = "lower",
    ) -> float:
        """https://www.investopedia.com/terms/v/var.asp
        Downside Value At Risk, "VaR". The equivalent of
        percentile.inc([...], 1-level) over returns in MS Excel.

        Parameters
        ----------

        level: float, default: 0.95
            The sought VaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        interpolation: str, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
            Default value is linear

        Returns
        -------
        float
            Downside Value At Risk
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        return float(
            self.tsdf.loc[earlier:later]
            .pct_change()
            .quantile(q=1 - level, interpolation=interpolation)
        )

    @property
    def vol_from_var(self, level: float = 0.95, interpolation: str = "lower") -> float:
        """
        Parameters
        ----------

        level: float, default: 0.95
            The sought VaR level
        interpolation: str, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
            Default value is linear

        Returns
        -------
        float
            Implied annualized volatility from the Downside VaR using the
            assumption that returns are normally distributed.
        """

        return float(
            -np.sqrt(self.periods_in_a_year)
            * self.var_down_func(level, interpolation=interpolation)
            / ss.norm.ppf(level)
        )

    def vol_from_var_func(
        self,
        level: float = 0.95,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        interpolation: str = "lower",
        drift_adjust: bool = False,
        periods_in_a_year_fixed: int = None,
    ) -> float:
        """
        Parameters
        ----------

        level: float, default: 0.95
            The sought VaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        interpolation: str, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
            Default value is linear
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        float
            Implied annualized volatility from the Downside VaR using the
            assumption that returns are normally distributed.
        """

        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[earlier:later].count(numeric_only=True)
            time_factor = how_many / fraction
        if drift_adjust:
            return float(
                (-np.sqrt(time_factor) / ss.norm.ppf(level))
                * (
                    self.var_down_func(
                        level,
                        months_from_last,
                        from_date,
                        to_date,
                        interpolation,
                    )
                    - self.tsdf.loc[earlier:later].pct_change().sum()
                    / len(self.tsdf.loc[earlier:later].pct_change())
                )
            )
        else:
            return float(
                -np.sqrt(time_factor)
                * self.var_down_func(
                    level, months_from_last, from_date, to_date, interpolation
                )
                / ss.norm.ppf(level)
            )

    def target_weight_from_var(
        self,
        target_vol: float = 0.175,
        min_leverage_local: float = 0.0,
        max_leverage_local: float = 99999.0,
        level: float = 0.95,
        months_from_last: int = None,
        from_date: dt.date = None,
        to_date: dt.date = None,
        interpolation: str = "lower",
        drift_adjust: bool = False,
        periods_in_a_year_fixed: int = None,
    ) -> float:
        """A position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 -> target met

        Parameters
        ----------
        target_vol: float, default: 0.175
            Target Volatility
        min_leverage_local: float, default: 0.0
            A minimum adjustment factor
        max_leverage_local: float, default: 99999.0
            A maximum adjustment factor
        level: float, default: 0.95
            The sought VaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        interpolation: str, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
            Default value is linear
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        float
            A position weight multiplier from the ratio between a VaR implied
            volatility and a given target volatility. Multiplier = 1.0 -> target met
        """

        return max(
            min_leverage_local,
            min(
                target_vol
                / self.vol_from_var_func(
                    level=level,
                    months_from_last=months_from_last,
                    from_date=from_date,
                    to_date=to_date,
                    interpolation=interpolation,
                    drift_adjust=drift_adjust,
                    periods_in_a_year_fixed=periods_in_a_year_fixed,
                ),
                max_leverage_local,
            ),
        )

    def value_to_ret(self, logret: bool = False):
        """Converts a valueseries into a returnseries.

        Parameters
        ----------
        logret: bool, default: False
            True for log return and False for simple return.
            Log return is the equivalent of LN(value[t] / value[t-1]) in MS excel.

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if logret:
            self.tsdf = np.log(self.tsdf).diff()
        else:
            self.tsdf = self.tsdf.pct_change()
        self.tsdf.iloc[0] = 0
        self.valuetype = "Return(Total)"
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        return self

    def value_to_diff(self, periods: int = 1):
        """Converts a valueseries to a series of its period differences

        Parameters
        ----------
        periods: int, default: 1
            The number of periods between observations over which difference is calculated

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf = self.tsdf.diff(periods=periods)
        self.tsdf.iloc[0] = 0
        self.valuetype = "Return(Total)"
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        return self

    def value_to_log(self, reverse: bool = False):
        """Converts a valueseries into logarithmic return series \n
        Equivalent to LN(value[t] / value[t=0]) in MS Excel

        Parameters
        ----------
        reverse: bool, default: False
            Allows for a reversal of the conversion.
            I.e. converting a logarithmic return series into a valueseries

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if reverse:
            self.tsdf = np.exp(self.tsdf)
            self.valuetype = "Price(Close)"
            self.tsdf.columns = pd.MultiIndex.from_product(
                [[self.label], [self.valuetype]]
            )
        else:
            self.tsdf = np.log(self.tsdf / self.tsdf.iloc[0])
            self.valuetype = "Return(Total)"
            self.tsdf.columns = pd.MultiIndex.from_product(
                [[self.label], [self.valuetype]]
            )
        return self

    def to_cumret(self):
        """Converts a returnseries into a cumulative valueseries

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if not any(
            [
                True if x == "Return(Total)" else False
                for x in self.tsdf.columns.get_level_values(1).values
            ]
        ):
            self.value_to_ret()
        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.cumprod(axis=0) / self.tsdf.iloc[0]
        self.valuetype = "Price(Close)"
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        return self

    def resample(self, freq: str = "BM"):
        """Resamples the timeseries frequency

        Parameters
        ----------
        freq: str, default "BM"
            Valid values https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf.index = pd.DatetimeIndex(self.tsdf.index)
        self.tsdf = self.tsdf.resample(freq).last()
        self.tsdf.index = [d.date() for d in pd.DatetimeIndex(self.tsdf.index)]
        return self

    def to_drawdown_series(self):
        """Converts the timeseries into a drawdown series

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        self.tsdf = drawdown_series(self.tsdf)
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], ["Drawdowns"]])
        return self

    def drawdown_details(self) -> pd.DataFrame:
        """
        Returns
        -------
        Pandas.DataFrame
            Calculates 'Max Drawdown', 'Start of drawdown', 'Date of bottom',
            'Days from start to bottom', & 'Average fall per day'
        """

        dddf = self.tsdf.copy()
        dddf.index = pd.DatetimeIndex(dddf.index)
        return drawdown_details(dddf).to_frame()

    def rolling_vol(
        self, observations: int = 21, periods_in_a_year_fixed: int = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        observations: int, default: 21
            Number of observations in the overlapping window.
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        Pandas.DataFrame
            Rolling annualised volatilities
        """

        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            time_factor = self.periods_in_a_year
        df = self.tsdf.pct_change().copy()
        voldf = df.rolling(observations, min_periods=observations).std() * np.sqrt(
            time_factor
        )
        voldf.dropna(inplace=True)
        voldf.columns = pd.MultiIndex.from_product(
            [[self.label], ["Rolling volatility"]]
        )
        return voldf

    def rolling_return(self, observations: int = 21) -> pd.DataFrame:
        """
        Parameters
        ----------
        observations: int, default: 21
            Number of observations in the overlapping window.

        Returns
        -------
        Pandas.DataFrame
            Rolling returns
        """

        retdf = (
            self.tsdf.pct_change().rolling(observations, min_periods=observations).sum()
        )
        retdf.columns = pd.MultiIndex.from_product([[self.label], ["Rolling returns"]])
        return retdf.dropna()

    def rolling_cvar_down(
        self, level: float = 0.95, observations: int = 252
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        level: float, default: 0.95
            The sought Conditional Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.

        Returns
        -------
        Pandas.DataFrame
            Rolling annualized downside CVaR
        """

        cvardf = self.tsdf.rolling(observations, min_periods=observations).apply(
            lambda x: cvar_down(x, level=level)
        )
        cvardf = cvardf.dropna()
        cvardf.columns = pd.MultiIndex.from_product([[self.label], ["Rolling CVaR"]])
        return cvardf

    def rolling_var_down(
        self,
        level: float = 0.95,
        observations: int = 252,
        interpolation: str = "lower",
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        level: float, default: 0.95
            The sought Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.
        interpolation: str, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
            Default value is linear

        Returns
        -------
        Pandas.DataFrame
           Rolling annualized downside Value At Risk "VaR"
        """

        vardf = self.tsdf.rolling(observations, min_periods=observations).apply(
            lambda x: var_down(x, level=level, interpolation=interpolation)
        )
        vardf = vardf.dropna()
        vardf.columns = pd.MultiIndex.from_product([[self.label], ["Rolling VaR"]])
        return vardf

    def value_nan_handle(self, method: str = "fill"):
        """Handling of missing values in a valueseries

        Parameters
        ----------
        method: str, default: "fill"
            Method used to handle NaN. Either fill with last known or drop

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        assert method in [
            "fill",
            "drop",
        ], "Method must be either fill or drop passed as string."
        if method == "fill":
            self.tsdf.fillna(method="ffill", inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def return_nan_handle(self, method: str = "fill"):
        """Handling of missing values in a returnseries

        Parameters
        ----------
        method: str, default: "fill"
            Method used to handle NaN. Either fill with zero or drop

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        assert method in [
            "fill",
            "drop",
        ], "Method must be either fill or drop passed as string."
        if method == "fill":
            self.tsdf.fillna(value=0.0, inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def running_adjustment(self, adjustment: float, days_in_year: int = 365):
        """Adds (+) or subtracts (-) a fee from the timeseries return

        Parameters
        ----------
        adjustment: float
            Fee to add or subtract
        days_in_year: int, default: 365
            The calculation divisor and
            assumed number of days in a calendar year

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if any(
            [
                True if x == "Return(Total)" else False
                for x in self.tsdf.columns.get_level_values(1).values
            ]
        ):
            ra_df = self.tsdf.copy()
        else:
            ra_df = self.tsdf.pct_change().copy()
        ra_df.dropna(inplace=True)
        prev = date_fix(str(self.first_idx))
        dates: list = [prev]
        values: list = [float(self.tsdf.iloc[0])]
        for idx, row in ra_df.iterrows():
            idx = date_fix(str(idx))
            dates.append(idx)
            values.append(
                values[-1]
                * (1 + float(row) + adjustment * (idx - prev).days / days_in_year)
            )
            prev = idx
        self.tsdf = pd.DataFrame(data=values, index=dates)
        self.valuetype = "Price(Close)"
        self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        self.tsdf.index = [d.date() for d in pd.DatetimeIndex(self.tsdf.index)]
        return self

    def set_new_label(
        self,
        lvl_zero: str = None,
        lvl_one: str = None,
        delete_lvl_one: bool = False,
    ):
        """Sets the column labels of the .tsdf Pandas Dataframe associated
        with the timeseries

        Parameters
        ----------
        lvl_zero: str, optional
            New level zero label
        lvl_one: str, optional
            New level one label
        delete_lvl_one: bool, default: False
            If True the level one label is deleted

        Returns
        -------
        OpenTimeSeries
            An OpenTimeSeries object
        """

        if lvl_zero is None and lvl_one is None:
            self.tsdf.columns = pd.MultiIndex.from_product(
                [[self.label], [self.valuetype]]
            )
        elif lvl_zero is not None and lvl_one is None:
            self.tsdf.columns = pd.MultiIndex.from_product(
                [[lvl_zero], [self.valuetype]]
            )
            self.label = lvl_zero
        elif lvl_zero is None and lvl_one is not None:
            self.tsdf.columns = pd.MultiIndex.from_product([[self.label], [lvl_one]])
            self.valuetype = lvl_one
        else:
            self.tsdf.columns = pd.MultiIndex.from_product([[lvl_zero], [lvl_one]])
            self.label, self.valuetype = lvl_zero, lvl_one
        if delete_lvl_one:
            self.tsdf.columns = self.tsdf.columns.droplevel(level=1)
        return self

    def plot_series(
        self,
        mode: str = "lines",
        tick_fmt: str = None,
        directory: str = None,
        size_array: list = None,
        auto_open: bool = True,
        add_logo: bool = True,
        show_last: bool = False,
        output_type: str = "file",
    ) -> (go.Figure, str):
        """Creates a Plotly Figure

        To scale the bubble size, use the attribute sizeref.
        We recommend using the following formula to calculate a sizeref value:
        sizeref = 2. * max(array of size values) / (desired maximum marker size ** 2)

        Parameters
        ----------
        mode: str, default: "lines"
            The type of scatter to use. lines, markers or lines+markers
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        directory: str, optional
            Directory where Plotly html file is saved
        size_array: list, optional
            List of values that will set bubble sizes if mode is markers
        auto_open: bool, default: True
            Determines whether or not to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        show_last: bool, default: False
            If True the last data point is highlighted as red dot with a label
        output_type: str, default: "file"
            file or div

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """

        if not directory:
            directory = os.path.join(str(Path.home()), "Documents")
        filename = self.label.replace("/", "").replace("#", "").replace(" ", "").upper()
        plotfile = os.path.join(os.path.abspath(directory), "{}.html".format(filename))

        assert mode in [
            "lines",
            "markers",
            "both",
        ], "Style must be specified as lines, markers or both."
        if mode == "both":
            mode = "lines+markers"

        values = [float(x) for x in self.tsdf.iloc[:, 0].tolist()]

        if size_array:
            sizer = 2.0 * max(size_array) / (90.0**2)
            text_array = [f"{x:.2%}" for x in size_array]
        else:
            sizer = None
            text_array = None

        data = [
            go.Scatter(
                x=self.tsdf.index,
                y=values,
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                line=dict(width=2.5, color="rgb(33, 134, 197)", dash="solid"),
                marker=dict(size=size_array, sizemode="area", sizeref=sizer, sizemin=4),
                text=text_array,
                mode=mode,
                name=self.label,
            )
        ]

        fig, logo = load_plotly_dict()
        fig["data"] = data
        figure = go.Figure(fig)
        figure.update_layout(yaxis=dict(tickformat=tick_fmt))

        if add_logo:
            figure.add_layout_image(logo)

        if show_last is True:
            if tick_fmt:
                txt = "Last " + "{:" + "{}".format(tick_fmt) + "}"
            else:
                txt = "Last {}"

            figure.add_scatter(
                x=[self.last_idx],
                y=[self.tsdf.iloc[-1, 0]],
                mode="markers + text",
                marker={"color": "red", "size": 12},
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                showlegend=False,
                text=[txt.format(self.tsdf.iloc[-1, 0])],
                textposition="top center",
            )

        plot(
            figure,
            filename=plotfile,
            auto_open=auto_open,
            link_text="",
            include_plotlyjs="cdn",
            config=fig["config"],
            output_type=output_type,
        )

        return figure, plotfile


def timeseries_chain(front, back, old_fee: float = 0.0) -> OpenTimeSeries:
    """Chain two timeseries together

    Parameters
    ----------
    front: OpenTimeSeries
        Earlier series to chain with
    back: OpenTimeSeries
        Later series to chain with
    old_fee: bool, default: False
        Fee to apply to earlier series

    Returns
    -------
    OpenTimeSeries
        An OpenTimeSeries object
    """
    old = front.from_deepcopy()
    old.running_adjustment(old_fee)
    new = back.from_deepcopy()

    olddf = old.tsdf.copy()
    dates = [x.strftime("%Y-%m-%d") for x in olddf.index if x < new.first_idx]
    values = np.array([float(x) for x in old.tsdf.values][: len(dates)])
    values = list(values * float(new.tsdf.iloc[0]) / float(olddf.loc[new.first_idx]))

    dates.extend([x.strftime("%Y-%m-%d") for x in new.tsdf.index])
    values.extend([float(x) for x in new.tsdf.values])

    new_dict = dict(new.__dict__)
    cleaner_list = ["label", "tsdf"]
    for item in cleaner_list:
        new_dict.pop(item)
    new_dict.update(dates=dates, values=values)
    return type(back)(new_dict)
