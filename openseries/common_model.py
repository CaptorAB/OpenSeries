"""Defining the CommonModel class."""
from __future__ import annotations

import datetime as dt
from inspect import stack
from json import dump
from math import ceil
from pathlib import Path
from secrets import choice
from string import ascii_letters
from typing import Any, Optional, Union, cast

from numpy import log, sqrt
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series
from plotly.graph_objs import Figure  # type: ignore[import-untyped,unused-ignore]
from plotly.io import to_html  # type: ignore[import-untyped,unused-ignore]
from plotly.offline import plot  # type: ignore[import-untyped,unused-ignore]
from pydantic import BaseModel, ConfigDict, DirectoryPath
from scipy.stats import (  # type: ignore[import-untyped,unused-ignore]
    kurtosis,
    skew,
)

from openseries._risk import (
    _cvar_down_calc,
    _drawdown_series,
    _var_down_calc,
    _var_implied_vol_and_target_func,
)
from openseries.datefixer import _get_calc_range
from openseries.load_plotly import load_plotly_dict
from openseries.types import (
    DaysInYearType,
    LiteralBarPlotMode,
    LiteralLinePlotMode,
    LiteralNanMethod,
    LiteralPlotlyJSlib,
    LiteralPlotlyOutput,
    LiteralQuantileInterp,
    ValueType,
)


class CommonModel(BaseModel):

    """Declare CommonModel."""

    tsdf: DataFrame = DataFrame(dtype="float64")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
    )

    @property
    def length(self: CommonModel) -> int:
        """
        Number of observations.

        Returns
        -------
        int
            Number of observations
        """
        return len(self.tsdf.index)

    @property
    def first_idx(self: CommonModel) -> dt.date:
        """
        The first date in the timeseries.

        Returns
        -------
        datetime.date
            The first date in the timeseries
        """
        return cast(dt.date, self.tsdf.index[0])

    @property
    def last_idx(self: CommonModel) -> dt.date:
        """
        The last date in the timeseries.

        Returns
        -------
        datetime.date
            The last date in the timeseries
        """
        return cast(dt.date, self.tsdf.index[-1])

    @property
    def span_of_days(self: CommonModel) -> int:
        """
        Number of days from the first date to the last.

        Returns
        -------
        int
            Number of days from the first date to the last
        """
        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self: CommonModel) -> float:
        """
        Length of series expressed in years assuming all years have 365.25 days.

        Returns
        -------
        float
            Length of the timeseries expressed in years assuming all years
            have 365.25 days
        """
        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self: CommonModel) -> float:
        """
        The average number of observations per year.

        Returns
        -------
        float
            The average number of observations per year
        """
        return self.length / self.yearfrac

    @property
    def max_drawdown_cal_year(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Maximum drawdown in a single calendar year.
        """
        years = Index(d.year for d in self.tsdf.index)
        mddc = (
            self.tsdf.groupby(years)
            .apply(
                lambda prices: (prices / prices.expanding(min_periods=1).max()).min()
                - 1,
            )
            .min()
        )
        if self.tsdf.shape[1] == 1:
            return float(mddc.iloc[0])
        return Series(
            data=mddc,
            index=self.tsdf.columns,
            name="Max Drawdown in cal yr",
            dtype="float64",
        )

    @property
    def geo_ret(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/c/cagr.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Compounded Annual Growth Rate (CAGR)
        """
        return self.geo_ret_func()

    @property
    def arithmetic_ret(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/a/arithmeticmean.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Annualized arithmetic mean of returns
        """
        return self.arithmetic_ret_func()

    @property
    def value_ret(self: CommonModel) -> Union[float, Series[float]]:
        """
        Simple return.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Simple return
        """
        return self.value_ret_func()

    @property
    def vol(self: CommonModel) -> Union[float, Series[float]]:
        """
        Annualized volatility.

        Based on Pandas .std() which is the equivalent of stdev.s([...]) in MS Excel.
        https://www.investopedia.com/terms/v/volatility.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Annualized volatility
        """
        return self.vol_func()

    @property
    def downside_deviation(self: CommonModel) -> Union[float, Series[float]]:
        """
        Downside Deviation.

        Standard deviation of returns that are below a Minimum Accepted Return
        of zero. It is used to calculate the Sortino Ratio.
        https://www.investopedia.com/terms/d/downside-deviation.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Downside deviation
        """
        min_accepted_return: float = 0.0
        return self.downside_deviation_func(min_accepted_return=min_accepted_return)

    @property
    def ret_vol_ratio(self: CommonModel) -> Union[float, Series[float]]:
        """
        Ratio of annualized arithmetic mean of returns and annualized volatility.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility.
        """
        riskfree_rate: float = 0.0
        return self.ret_vol_ratio_func(riskfree_rate=riskfree_rate)

    @property
    def sortino_ratio(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/s/sortinoratio.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Sortino ratio calculated as the annualized arithmetic mean of returns
            / downside deviation. The ratio implies that the riskfree asset has zero
            volatility, and a minimum acceptable return of zero.
        """
        riskfree_rate: float = 0.0
        minimum_accepted_return: float = 0.0
        return self.sortino_ratio_func(
            riskfree_rate=riskfree_rate,
            min_accepted_return=minimum_accepted_return,
        )

    @property
    def z_score(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/z/zscore.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Z-score as (last return - mean return) / standard deviation of returns.
        """
        return self.z_score_func()

    @property
    def max_drawdown(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Maximum drawdown without any limit on date range
        """
        return self.max_drawdown_func()

    @property
    def max_drawdown_date(self: CommonModel) -> Union[dt.date, Series[dt.date]]:
        """
        Date when the maximum drawdown occurred.

        https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Returns
        -------
        Union[datetime.date, pandas.Series[dt.date]]
            Date when the maximum drawdown occurred
        """
        mdddf = self.tsdf.copy()
        mdddf.index = DatetimeIndex(mdddf.index)
        result = (mdddf / mdddf.expanding(min_periods=1).max()).idxmin().dt.date

        if self.tsdf.shape[1] == 1:
            return result.iloc[0]
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Max drawdown date",
            dtype="datetime64[ns]",
        ).dt.date

    @property
    def worst(self: CommonModel) -> Union[float, Series[float]]:
        """
        Most negative percentage change.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Most negative percentage change
        """
        observations: int = 1
        return self.worst_func(observations=observations)

    @property
    def positive_share(self: CommonModel) -> Union[float, Series[float]]:
        """
        The share of percentage changes that are greater than zero.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            The share of percentage changes that are greater than zero
        """
        return self.positive_share_func()

    @property
    def skew(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/s/skewness.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Skew of the return distribution
        """
        return self.skew_func()

    @property
    def kurtosis(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/k/kurtosis.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Kurtosis of the return distribution
        """
        return self.kurtosis_func()

    @property
    def cvar_down(self: CommonModel) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Downside 95% Conditional Value At Risk "CVaR"
        """
        level: float = 0.95
        return self.cvar_down_func(level=level)

    @property
    def var_down(self: CommonModel) -> Union[float, Series[float]]:
        """
        Downside 95% Value At Risk (VaR).

        The equivalent of percentile.inc([...], 1-level) over returns in MS Excel.
        https://www.investopedia.com/terms/v/var.asp.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Downside 95% Value At Risk (VaR)
        """
        level: float = 0.95
        interpolation: LiteralQuantileInterp = "lower"
        return self.var_down_func(level=level, interpolation=interpolation)

    @property
    def vol_from_var(self: CommonModel) -> Union[float, Series[float]]:
        """
        Implied annualized volatility from Downside 95% Value at Risk.

        Assumes that returns are normally distributed.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Implied annualized volatility from the Downside 95% VaR using the
            assumption that returns are normally distributed.
        """
        level: float = 0.95
        interpolation: LiteralQuantileInterp = "lower"
        return self.vol_from_var_func(level=level, interpolation=interpolation)

    def value_to_log(self: CommonModel) -> CommonModel:
        """
        Series of values converted into logarithmic weighted series.

        Equivalent to LN(value[t] / value[t=0]) in Excel.

        Returns
        -------
        self
            An object of the same class
        """
        self.tsdf = DataFrame(
            data=log(self.tsdf / self.tsdf.iloc[0]),
            index=self.tsdf.index,
            columns=self.tsdf.columns,
        )
        return self

    def value_nan_handle(
        self: CommonModel,
        method: LiteralNanMethod = "fill",
    ) -> CommonModel:
        """
        Handle missing values in a valueseries.

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with last known or drop

        Returns
        -------
        self
            An object of the same class
        """
        if method == "fill":
            self.tsdf = self.tsdf.ffill()
        else:
            self.tsdf = self.tsdf.dropna()
        return self

    def return_nan_handle(
        self: CommonModel,
        method: LiteralNanMethod = "fill",
    ) -> CommonModel:
        """
        Handle missing values in a returnseries.

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with zero or drop

        Returns
        -------
        self
            An object of the same class
        """
        if method == "fill":
            self.tsdf = self.tsdf.fillna(value=0.0)
        else:
            self.tsdf = self.tsdf.dropna()
        return self

    def to_drawdown_series(self: CommonModel) -> CommonModel:
        """
        Convert timeseries into a drawdown series.

        Returns
        -------
        self
            An object of the same class
        """
        for serie in self.tsdf:
            self.tsdf.loc[:, serie] = _drawdown_series(  # type: ignore[index]
                prices=self.tsdf.loc[:, serie],  # type: ignore[index]
            )
        return self

    def to_json(
        self: CommonModel,
        filename: str,
        directory: Optional[DirectoryPath] = None,
    ) -> list[dict[str, Union[str, bool, ValueType, list[str], list[float]]]]:
        """
        Dump timeseries data into a json file.

        The label and tsdf parameters are deleted before the json file is saved

        Parameters
        ----------
        filename: str
            Filename including filetype
        directory: DirectoryPath, optional
            File folder location

        Returns
        -------
        list[Dict[str, Union[str, bool, ValueType, list[str], list[float]]]]
            A list of dictionaries with the raw original data of the series
        """
        if directory:
            dirpath = Path(directory).resolve()
        elif Path.home().joinpath("Documents").exists():
            dirpath = Path.home().joinpath("Documents")
        else:
            dirpath = Path(stack()[1].filename).parent

        cleaner_list = ["label", "tsdf"]
        data = dict(self.__dict__)
        output = []
        if "label" in data:
            for item in cleaner_list:
                data.pop(item)
            output.append(dict(data))
        else:
            for serie in cast(list[Any], data.get("constituents")):
                itemdata = dict(serie.__dict__)
                for item in cleaner_list:
                    itemdata.pop(item)
                output.append(dict(itemdata))

        with Path.open(
            dirpath.joinpath(filename),
            "w",
            encoding="utf-8",
        ) as jsonfile:
            dump(output, jsonfile, indent=2, sort_keys=False)

        return output

    def to_xlsx(
        self: CommonModel,
        filename: str,
        sheet_title: Optional[str] = None,
        directory: Optional[DirectoryPath] = None,
        *,
        overwrite: bool = True,
    ) -> str:
        """
        Save .tsdf DataFrame to an Excel spreadsheet file.

        Parameters
        ----------
        filename: str
            Filename that should include .xlsx
        sheet_title: str, optional
            Name of the sheet in the Excel file
        directory: DirectoryPath, optional
            The file directory where the Excel file is saved.
        overwrite: bool, default: True
            Flag whether to overwrite an existing file

        Returns
        -------
        str
            The Excel file path
        """
        if filename[-5:].lower() != ".xlsx":
            msg = "Filename must end with .xlsx"
            raise NameError(msg)

        if directory:
            dirpath = Path(directory).resolve()
        elif Path.home().joinpath("Documents").exists():
            dirpath = Path.home().joinpath("Documents")
        else:
            dirpath = Path(stack()[1].filename).parent

        sheetfile = dirpath.joinpath(filename)

        wrkbook = Workbook()
        wrksheet = wrkbook.active

        if sheet_title:
            cast(Worksheet, wrksheet).title = sheet_title

        for row in dataframe_to_rows(df=self.tsdf, index=True, header=True):
            cast(Worksheet, wrksheet).append(row)

        if not overwrite and Path(sheetfile).exists():
            msg = f"{sheetfile!s} already exists."
            raise FileExistsError(msg)

        wrkbook.save(sheetfile)

        return str(sheetfile)

    def plot_bars(
        self: CommonModel,
        mode: LiteralBarPlotMode = "group",
        tick_fmt: Optional[str] = None,
        filename: Optional[str] = None,
        directory: Optional[DirectoryPath] = None,
        labels: Optional[list[str]] = None,
        output_type: LiteralPlotlyOutput = "file",
        include_plotlyjs: LiteralPlotlyJSlib = "cdn",
        *,
        auto_open: bool = True,
        add_logo: bool = True,
    ) -> tuple[Figure, str]:
        """
        Create a Plotly Bar Figure.

        Parameters
        ----------
        self.tsdf: pandas.DataFrame
            The timeseries self.tsdf
        mode: LiteralBarPlotMode
            The type of bar to use
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        filename: str, optional
            Name of the Plotly html file
        directory: DirectoryPath, optional
            Directory where Plotly html file is saved
        labels: list[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type
        include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
            Determines how the plotly.js library is included in the output
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot

        Returns
        -------
        tuple[plotly.go.Figure, str]
            Plotly Figure and html filename with location
        """
        if labels:
            if len(labels) != self.tsdf.shape[1]:
                msg = "Must provide same number of labels as items in frame."
                raise ValueError(
                    msg,
                )
        else:
            labels = list(self.tsdf.columns.get_level_values(0))

        if directory:
            dirpath = Path(directory).resolve()
        elif Path.home().joinpath("Documents").exists():
            dirpath = Path.home().joinpath("Documents")
        else:
            dirpath = Path(stack()[1].filename).parent

        if not filename:
            filename = "".join(choice(ascii_letters) for _ in range(6)) + ".html"
        plotfile = dirpath.joinpath(filename)

        fig, logo = load_plotly_dict()
        figure = Figure(fig)

        opacity = 0.7 if mode == "overlay" else None

        hovertemplate = (
            f"%{{y:{tick_fmt}}}<br>%{{x|{'%Y-%m-%d'}}}"
            if tick_fmt
            else "%{y}<br>%{x|%Y-%m-%d}"
        )

        for item in range(self.tsdf.shape[1]):
            figure.add_bar(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate=hovertemplate,
                name=labels[item],
                opacity=opacity,
            )
        figure.update_layout(barmode=mode, yaxis={"tickformat": tick_fmt})

        if add_logo:
            figure.add_layout_image(logo)

        if output_type == "file":
            plot(
                figure_or_data=figure,
                filename=str(plotfile),
                auto_open=auto_open,
                auto_play=False,
                link_text="",
                include_plotlyjs=include_plotlyjs,
                config=fig["config"],
                output_type=output_type,
            )
            string_output = str(plotfile)
        else:
            div_id = filename.split(sep=".")[0]
            string_output = to_html(
                fig=figure,
                config=fig["config"],
                auto_play=False,
                include_plotlyjs=include_plotlyjs,
                full_html=False,
                div_id=div_id,
            )

        return figure, string_output

    def plot_series(  # noqa: C901
        self: CommonModel,
        mode: LiteralLinePlotMode = "lines",
        tick_fmt: Optional[str] = None,
        filename: Optional[str] = None,
        directory: Optional[DirectoryPath] = None,
        labels: Optional[list[str]] = None,
        output_type: LiteralPlotlyOutput = "file",
        include_plotlyjs: LiteralPlotlyJSlib = "cdn",
        *,
        auto_open: bool = True,
        add_logo: bool = True,
        show_last: bool = False,
    ) -> tuple[Figure, str]:
        """
        Create a Plotly Scatter Figure.

        Parameters
        ----------
        self.tsdf: pandas.DataFrame
            The timeseries self.tsdf
        mode: LiteralLinePlotMode, default: "lines"
            The type of scatter to use
        tick_fmt: str, optional
            None, '%', '.1%' depending on number of decimals to show
        filename: str, optional
            Name of the Plotly html file
        directory: DirectoryPath, optional
            Directory where Plotly html file is saved
        labels: list[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type
        include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
            Determines how the plotly.js library is included in the output
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        show_last: bool, default: False
            If True the last self.tsdf point is highlighted as red dot with a label

        Returns
        -------
        tuple[plotly.go.Figure, str]
            Plotly Figure and html filename with location
        """
        if labels:
            if len(labels) != self.tsdf.shape[1]:
                msg = "Must provide same number of labels as items in frame."
                raise ValueError(
                    msg,
                )
        else:
            labels = list(self.tsdf.columns.get_level_values(0))

        if directory:
            dirpath = Path(directory).resolve()
        elif Path.home().joinpath("Documents").exists():
            dirpath = Path.home().joinpath("Documents")
        else:
            dirpath = Path(stack()[1].filename).parent

        if not filename:
            filename = "".join(choice(ascii_letters) for _ in range(6)) + ".html"
        plotfile = dirpath.joinpath(filename)

        fig, logo = load_plotly_dict()
        figure = Figure(fig)

        hovertemplate = (
            f"%{{y:{tick_fmt}}}<br>%{{x|{'%Y-%m-%d'}}}"
            if tick_fmt
            else "%{y}<br>%{x|%Y-%m-%d}"
        )

        for item in range(self.tsdf.shape[1]):
            figure.add_scatter(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate=hovertemplate,
                line={"width": 2.5, "dash": "solid"},
                mode=mode,
                name=labels[item],
            )
        figure.update_layout(yaxis={"tickformat": tick_fmt})

        if show_last is True:
            txt = f"Last {{:{tick_fmt}}}" if tick_fmt else "Last {}"

            for item in range(self.tsdf.shape[1]):
                figure.add_scatter(
                    x=[self.tsdf.iloc[:, item].index[-1]],
                    y=[self.tsdf.iloc[-1, item]],
                    mode="markers + text",
                    marker={"color": "red", "size": 12},
                    hovertemplate=hovertemplate,
                    showlegend=False,
                    name=labels[item],
                    text=[txt.format(self.tsdf.iloc[-1, item])],
                    textposition="top center",
                )

        if add_logo:
            figure.add_layout_image(logo)

        if output_type == "file":
            plot(
                figure_or_data=figure,
                filename=str(plotfile),
                auto_open=auto_open,
                auto_play=False,
                link_text="",
                include_plotlyjs=include_plotlyjs,
                config=fig["config"],
                output_type=output_type,
            )
            string_output = str(plotfile)
        else:
            div_id = filename.split(sep=".")[0]
            string_output = to_html(
                fig=figure,
                config=fig["config"],
                auto_play=False,
                include_plotlyjs=include_plotlyjs,
                full_html=False,
                div_id=div_id,
            )

        return figure, string_output

    def arithmetic_ret_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Union[float, Series[float]]:
        """
        https://www.investopedia.com/terms/a/arithmeticmean.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Annualized arithmetic mean of returns
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            fraction = (later - earlier).days / 365.25
            how_many = self.tsdf.loc[
                cast(int, earlier) : cast(int, later),
                self.tsdf.columns.to_numpy()[0],
            ].count()
            time_factor = how_many / fraction

        result = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .mean()
            * time_factor
        )

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Arithmetic return",
            dtype="float64",
        )

    def vol_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Union[float, Series[float]]:
        """
        Annualized volatility.

        Based on Pandas .std() which is the equivalent of stdev.s([...]) in MS Excel.
        https://www.investopedia.com/terms/v/volatility.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Annualized volatility
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            fraction = (later - earlier).days / 365.25
            how_many = (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].count().iloc[0]
            )
            time_factor = how_many / fraction

        data = self.tsdf.loc[cast(int, earlier) : cast(int, later)]
        result = (
            data.pct_change(fill_method=cast(str, None)).std().mul(sqrt(time_factor))
        )

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Volatility",
            dtype="float64",
        )

    def vol_from_var_func(
        self: CommonModel,
        level: float = 0.95,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        interpolation: LiteralQuantileInterp = "lower",
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
        *,
        drift_adjust: bool = False,
    ) -> Union[float, Series[float]]:
        """
        Implied annualized volatility.

        Implied annualized volatility from the Downside VaR using the assumption
        that returns are normally distributed.

        Parameters
        ----------
        level: float, default: 0.95
            The sought VaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        interpolation: LiteralQuantileInterp, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Implied annualized volatility from the Downside VaR using the
            assumption that returns are normally distributed.
        """
        return _var_implied_vol_and_target_func(
            data=self.tsdf,
            level=level,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            interpolation=interpolation,
            drift_adjust=drift_adjust,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

    def target_weight_from_var(
        self: CommonModel,
        target_vol: float = 0.175,
        level: float = 0.95,
        min_leverage_local: float = 0.0,
        max_leverage_local: float = 99999.0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        interpolation: LiteralQuantileInterp = "lower",
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
        *,
        drift_adjust: bool = False,
    ) -> Union[float, Series[float]]:
        """
        Target weight from VaR.

        A position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 -> target met.

        Parameters
        ----------
        target_vol: float, default: 0.175
            Target Volatility
        level: float, default: 0.95
            The sought VaR level
        min_leverage_local: float, default: 0.0
            A minimum adjustment factor
        max_leverage_local: float, default: 99999.0
            A maximum adjustment factor
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        interpolation: LiteralQuantileInterp, default: "lower"
            type of interpolation in Pandas.DataFrame.quantile() function.
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return

        Returns
        -------
        Union[float, Pandas.Series[float]]
            A position weight multiplier from the ratio between a VaR implied
            volatility and a given target volatility. Multiplier = 1.0 -> target met
        """
        return _var_implied_vol_and_target_func(
            data=self.tsdf,
            target_vol=target_vol,
            level=level,
            min_leverage_local=min_leverage_local,
            max_leverage_local=max_leverage_local,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            interpolation=interpolation,
            drift_adjust=drift_adjust,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

    def cvar_down_func(
        self: CommonModel,
        level: float = 0.95,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Downside Conditional Value At Risk "CVaR".

        https://www.investopedia.com/terms/c/conditional_value_at_risk.asp.

        Parameters
        ----------
        level: float, default: 0.95
            The sought CVaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Downside Conditional Value At Risk "CVaR"
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        cvar_df = self.tsdf.loc[cast(int, earlier) : cast(int, later)].copy(deep=True)
        result = [
            cvar_df.loc[:, x]  # type: ignore[call-overload,index]
            .pct_change(fill_method=cast(str, None))
            .sort_values()
            .iloc[
                : int(
                    ceil(
                        (1 - level)
                        * cvar_df.loc[:, x]  # type: ignore[index]
                        .pct_change(fill_method=cast(str, None))
                        .count(),
                    ),
                )
            ]
            .mean()
            for x in self.tsdf
        ]
        if self.tsdf.shape[1] == 1:
            return float(result[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name=f"CVaR {level:.1%}",
            dtype="float64",
        )

    def downside_deviation_func(
        self: CommonModel,
        min_accepted_return: float = 0.0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Union[float, Series[float]]:
        """
        Downside Deviation.

        The standard deviation of returns that are below a Minimum Accepted
        Return of zero. It is used to calculate the Sortino Ratio.
        https://www.investopedia.com/terms/d/downside-deviation.asp.

        Parameters
        ----------
        min_accepted_return : float, optional
            The annualized Minimum Accepted Return (MAR)
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Downside deviation
        """
        zero: float = 0.0
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        how_many = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .count(numeric_only=True)
        )
        if periods_in_a_year_fixed:
            time_factor = Series(
                data=[float(periods_in_a_year_fixed)] * how_many.shape[0],
                index=self.tsdf.columns,
                dtype="float64",
            )
        else:
            fraction = (later - earlier).days / 365.25
            time_factor = how_many.div(fraction)

        dddf = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .sub(min_accepted_return / time_factor)
        )

        result = sqrt((dddf[dddf < zero] ** 2).sum() / how_many) * sqrt(
            time_factor,
        )

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Downside deviation",
            dtype="float64",
        )

    def geo_ret_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Compounded Annual Growth Rate (CAGR).

        https://www.investopedia.com/terms/c/cagr.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Compounded Annual Growth Rate (CAGR)
        """
        zero: float = 0.0
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        fraction = (later - earlier).days / 365.25

        if (
            zero in self.tsdf.loc[earlier].tolist()
            or self.tsdf.loc[[earlier, later]].lt(0.0).any().any()
        ):
            msg = (
                "Geometric return cannot be calculated due to "
                "an initial value being zero or a negative value."
            )
            raise ValueError(
                msg,
            )

        result = (self.tsdf.loc[later] / self.tsdf.loc[earlier]) ** (1 / fraction) - 1

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Geometric return",
            dtype="float64",
        )

    def skew_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Skew of the return distribution.

        https://www.investopedia.com/terms/s/skewness.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Skew of the return distribution
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = skew(
            a=self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .to_numpy(),
            bias=True,
            nan_policy="omit",
        )

        if self.tsdf.shape[1] == 1:
            return float(result[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Skew",
            dtype="float64",
        )

    def kurtosis_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Kurtosis of the return distribution.

        https://www.investopedia.com/terms/k/kurtosis.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Kurtosis of the return distribution
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = kurtosis(
            self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change(
                fill_method=cast(str, None),
            ),
            fisher=True,
            bias=True,
            nan_policy="omit",
        )

        if self.tsdf.shape[1] == 1:
            return float(result[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Kurtosis",
            dtype="float64",
        )

    def max_drawdown_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        min_periods: int = 1,
    ) -> Union[float, Series[float]]:
        """
        Maximum drawdown without any limit on date range.

        https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        min_periods: int, default: 1
            Smallest number of observations to use to find the maximum drawdown

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Maximum drawdown without any limit on date range
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            / self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .expanding(min_periods=min_periods)
            .max()
        ).min() - 1
        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Max Drawdown",
            dtype="float64",
        )

    def positive_share_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Calculate share of percentage changes that are greater than zero.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Calculate share of percentage changes that are greater than zero
        """
        zero: float = 0.0
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        pos = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))[1:][
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change(
                    fill_method=cast(str, None),
                )[1:]
                > zero
            ]
            .count()
        )
        tot = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))[1:]
            .count()
        )
        share = pos / tot
        if self.tsdf.shape[1] == 1:
            return float(share.iloc[0])
        return Series(
            data=share,
            index=self.tsdf.columns,
            name="Positive Share",
            dtype="float64",
        )

    def ret_vol_ratio_func(
        self: CommonModel,
        riskfree_rate: float = 0.0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Union[float, Series[float]]:
        """
        Ratio between arithmetic mean of returns and annualized volatility.

        The ratio of annualized arithmetic mean of returns and annualized
        volatility or, if riskfree return provided, Sharpe ratio calculated
        as ( geometric return - risk-free return ) / volatility. The latter ratio
        implies that the riskfree asset has zero volatility.
        https://www.investopedia.com/terms/s/sharperatio.asp.

        Parameters
        ----------
        riskfree_rate : float
            The return of the zero volatility asset used to calculate Sharpe ratio
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility or, if risk-free return provided, Sharpe ratio
        """
        ratio = Series(
            self.arithmetic_ret_func(
                months_from_last=months_from_last,
                from_date=from_date,
                to_date=to_date,
                periods_in_a_year_fixed=periods_in_a_year_fixed,
            )
            - riskfree_rate,
        ) / self.vol_func(
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        if self.tsdf.shape[1] == 1:
            return float(ratio.iloc[0])
        return Series(
            data=ratio,
            index=self.tsdf.columns,
            name="Return vol ratio",
            dtype="float64",
        )

    def sortino_ratio_func(
        self: CommonModel,
        riskfree_rate: float = 0.0,
        min_accepted_return: float = 0.0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Union[float, Series[float]]:
        """
        Sortino Ratio.

        The Sortino ratio calculated as ( return - risk free return )
        / downside deviation. The ratio implies that the riskfree asset has zero
        volatility, and a minimum acceptable return of zero. The ratio is
        calculated using the annualized arithmetic mean of returns.
        https://www.investopedia.com/terms/s/sortinoratio.asp.

        Parameters
        ----------
        riskfree_rate : float
            The return of the zero volatility asset
        min_accepted_return : float, optional
            The annualized Minimum Accepted Return (MAR)
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Sortino ratio calculated as ( return - riskfree return ) /
            downside deviation (std dev of returns below MAR)
        """
        ratio = Series(
            self.arithmetic_ret_func(
                months_from_last=months_from_last,
                from_date=from_date,
                to_date=to_date,
                periods_in_a_year_fixed=periods_in_a_year_fixed,
            )
            - riskfree_rate,
        ) / self.downside_deviation_func(
            min_accepted_return=min_accepted_return,
            months_from_last=months_from_last,
            from_date=from_date,
            to_date=to_date,
            periods_in_a_year_fixed=periods_in_a_year_fixed,
        )

        if self.tsdf.shape[1] == 1:
            return float(ratio.iloc[0])
        return Series(
            data=ratio,
            index=self.tsdf.columns,
            name="Sortino ratio",
            dtype="float64",
        )

    def value_ret_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Calculate simple return.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Calculate simple return
        """
        zero: float = 0.0
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if zero in self.tsdf.iloc[0].tolist():
            msg = (
                "Simple return cannot be calculated due to "
                f"an initial value being zero. ({self.tsdf.head(3)})"
            )
            raise ValueError(
                msg,
            )

        result = self.tsdf.loc[later] / self.tsdf.loc[earlier] - 1

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Simple return",
            dtype="float64",
        )

    def value_ret_calendar_period(
        self: CommonModel,
        year: int,
        month: Optional[int] = None,
    ) -> Union[float, Series[float]]:
        """
        Calculate simple return for a specific calendar period.

        Parameters
        ----------
        year : int
            Calendar year of the period to calculate.
        month : int, optional
            Calendar month of the period to calculate.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Calculate simple return for a specific calendar period
        """
        if month is None:
            period = str(year)
        else:
            period = "-".join([str(year), str(month).zfill(2)])
        vrdf = self.tsdf.copy()
        vrdf.index = DatetimeIndex(vrdf.index)
        resultdf = DataFrame(vrdf.pct_change(fill_method=cast(str, None)))
        result = resultdf.loc[period] + 1
        cal_period = result.cumprod(axis="index").iloc[-1] - 1
        if self.tsdf.shape[1] == 1:
            return float(cal_period.iloc[0])
        return Series(
            data=cal_period,
            index=self.tsdf.columns,
            name=period,
            dtype="float64",
        )

    def var_down_func(
        self: CommonModel,
        level: float = 0.95,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        interpolation: LiteralQuantileInterp = "lower",
    ) -> Union[float, Series[float]]:
        """
        Downside Value At Risk, "VaR".

        The equivalent of percentile.inc([...], 1-level) over returns in MS Excel.
        https://www.investopedia.com/terms/v/var.asp.

        Parameters
        ----------
        level: float, default: 0.95
            The sought VaR level
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        interpolation: LiteralQuantileInterp, default: "lower"
            Type of interpolation in Pandas.DataFrame.quantile() function.

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Downside Value At Risk
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .quantile(1 - level, interpolation=interpolation)
        )

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name=f"VaR {level:.1%}",
            dtype="float64",
        )

    def worst_func(
        self: CommonModel,
        observations: int = 1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Most negative percentage change over a rolling number of observations.

        Parameters
        ----------
        observations: int, default: 1
            Number of observations over which to measure the worst outcome
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Most negative percentage change over a rolling number of observations
            within a chosen date range
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change(fill_method=cast(str, None))
            .rolling(observations, min_periods=observations)
            .sum()
            .min()
        )

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Worst",
            dtype="float64",
        )

    def z_score_func(
        self: CommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series[float]]:
        """
        Z-score as (last return - mean return) / standard deviation of returns.

        https://www.investopedia.com/terms/z/zscore.asp.

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date

        Returns
        -------
        Union[float, Pandas.Series[float]]
            Z-score as (last return - mean return) / standard deviation of returns
        """
        earlier, later = _get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        zscframe = self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change(
            fill_method=cast(str, None),
        )
        result = (zscframe.iloc[-1] - zscframe.mean()) / zscframe.std()

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Z-score",
            dtype="float64",
        )

    def rolling_cvar_down(
        self: CommonModel,
        column: int = 0,
        level: float = 0.95,
        observations: int = 252,
    ) -> DataFrame:
        """
        Calculate rolling annualized downside CVaR.

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        level: float, default: 0.95
            The sought Conditional Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.

        Returns
        -------
        Pandas.DataFrame
            Calculate rolling annualized downside CVaR
        """
        cvar_label = cast(tuple[str], self.tsdf.iloc[:, column].name)[0]
        cvarseries = (
            self.tsdf.iloc[:, column]
            .rolling(observations, min_periods=observations)
            .apply(lambda x: _cvar_down_calc(x, level=level))
        )
        cvardf = cvarseries.dropna().to_frame()
        cvardf.columns = MultiIndex.from_arrays([[cvar_label], ["Rolling CVaR"]])

        return cvardf

    def rolling_return(
        self: CommonModel,
        column: int = 0,
        observations: int = 21,
    ) -> DataFrame:
        """
        Calculate rolling returns.

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        observations: int, default: 21
            Number of observations in the overlapping window.

        Returns
        -------
        Pandas.DataFrame
            Calculate rolling returns
        """
        ret_label = cast(tuple[str], self.tsdf.iloc[:, column].name)[0]
        retseries = (
            self.tsdf.iloc[:, column]
            .pct_change(fill_method=cast(str, None))
            .rolling(observations, min_periods=observations)
            .sum()
        )
        retdf = retseries.dropna().to_frame()
        retdf.columns = MultiIndex.from_arrays([[ret_label], ["Rolling returns"]])

        return retdf

    def rolling_var_down(
        self: CommonModel,
        column: int = 0,
        level: float = 0.95,
        observations: int = 252,
        interpolation: LiteralQuantileInterp = "lower",
    ) -> DataFrame:
        """
        Calculate rolling annualized downside Value At Risk "VaR".

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        level: float, default: 0.95
            The sought Value At Risk level
        observations: int, default: 252
            Number of observations in the overlapping window.
        interpolation: LiteralQuantileInterp, default: "lower"
            Type of interpolation in Pandas.DataFrame.quantile() function.

        Returns
        -------
        Pandas.DataFrame
           Calculate rolling annualized downside Value At Risk "VaR"
        """
        var_label = cast(tuple[str], self.tsdf.iloc[:, column].name)[0]
        varseries = (
            self.tsdf.iloc[:, column]
            .rolling(observations, min_periods=observations)
            .apply(
                lambda x: _var_down_calc(x, level=level, interpolation=interpolation),
            )
        )
        vardf = varseries.dropna().to_frame()
        vardf.columns = MultiIndex.from_arrays([[var_label], ["Rolling VaR"]])

        return vardf

    def rolling_vol(
        self: CommonModel,
        column: int = 0,
        observations: int = 21,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> DataFrame:
        """
        Calculate rolling annualised volatilities.

        Parameters
        ----------
        column: int, default: 0
            Position as integer of column to calculate
        observations: int, default: 21
            Number of observations in the overlapping window.
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Pandas.DataFrame
            Calculate rolling annualised volatilities
        """
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = self.periods_in_a_year
        vol_label = cast(tuple[str, ValueType], self.tsdf.iloc[:, column].name)[0]
        dframe = self.tsdf.iloc[:, column].pct_change(fill_method=cast(str, None))
        volseries = dframe.rolling(
            observations,
            min_periods=observations,
        ).std() * sqrt(
            time_factor,
        )
        voldf = volseries.dropna().to_frame()
        voldf.columns = MultiIndex.from_arrays(
            [
                [vol_label],
                ["Rolling volatility"],
            ],
        )

        return DataFrame(voldf)
