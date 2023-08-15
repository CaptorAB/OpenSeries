"""
Defining common properties
"""
import datetime as dt
from json import dump
from pathlib import Path
from random import choices
from string import ascii_letters
from os import path
from typing import Any, cast, Dict, List, Optional, Tuple, TypeVar, Union
from math import ceil
from numpy import cumprod, log, sqrt
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pandas import DataFrame, DatetimeIndex, Series
from plotly.graph_objs import Figure
from plotly.offline import plot
from scipy.stats import kurtosis, norm, skew

from openseries.risk import drawdown_series
from openseries.types import (
    LiteralBarPlotMode,
    LiteralLinePlotMode,
    LiteralPlotlyOutput,
    LiteralNanMethod,
    ValueType,
)
from openseries.load_plotly import load_plotly_dict
from openseries.common_tools import get_calc_range
from openseries.types import LiteralQuantileInterp


TypeCommonModel = TypeVar("TypeCommonModel", bound="CommonModel")


def var_implied_vol_and_target_func(
    data: DataFrame,
    level: float,
    target_vol: Optional[float] = None,
    min_leverage_local: float = 0.0,
    max_leverage_local: float = 99999.0,
    months_from_last: Optional[int] = None,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    interpolation: LiteralQuantileInterp = "lower",
    drift_adjust: bool = False,
    periods_in_a_year_fixed: Optional[int] = None,
) -> Union[float, Series]:
    """A position weight multiplier from the ratio between a VaR implied
    volatility and a given target volatility. Multiplier = 1.0 -> target met

    Parameters
    ----------
    data: DataFrame
        Timeseries data
    level: float
        The sought VaR level
    target_vol: Optional[float]
        Target Volatility
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
    drift_adjust: bool, default: False
        An adjustment to remove the bias implied by the average return
    periods_in_a_year_fixed : int, optional
        Allows locking the periods-in-a-year to simplify test cases and
        comparisons

    Returns
    -------
    Union[float, Pandas.Series]
        A position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 -> target met
    """
    earlier, later = get_calc_range(
        data=data,
        months_offset=months_from_last,
        from_dt=from_date,
        to_dt=to_date,
    )
    if periods_in_a_year_fixed:
        time_factor = float(periods_in_a_year_fixed)
    else:
        fraction = (later - earlier).days / 365.25
        how_many = data.loc[cast(int, earlier) : cast(int, later)].count().iloc[0]
        time_factor = how_many / fraction
    if drift_adjust:
        imp_vol = (-sqrt(time_factor) / norm.ppf(level)) * (
            data.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
            .quantile(1 - level, interpolation=interpolation)
            - data.loc[cast(int, earlier) : cast(int, later)].pct_change().sum()
            / len(data.loc[cast(int, earlier) : cast(int, later)].pct_change())
        )
    else:
        imp_vol = (
            -sqrt(time_factor)
            * data.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
            .quantile(1 - level, interpolation=interpolation)
            / norm.ppf(level)
        )

    if target_vol:
        result = imp_vol.apply(
            lambda x: max(min_leverage_local, min(target_vol / x, max_leverage_local))
        )
        label = "Weight from target vol"
    else:
        result = imp_vol
        label = f"Imp vol from VaR {level:.0%}"

    if data.shape[1] == 1:
        return float(result.iloc[0])
    return Series(
        data=result,
        index=data.columns,
        name=label,
        dtype="float64",
    )


class CommonModel:
    """CommonModel declared"""

    tsdf: DataFrame

    @property
    def length(self: TypeCommonModel) -> int:
        """
        Returns
        -------
        int
            Number of observations
        """

        return len(self.tsdf.index)

    @property
    def first_idx(self: TypeCommonModel) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The first date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[0])

    @property
    def last_idx(self: TypeCommonModel) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The last date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[-1])

    @property
    def span_of_days(self: TypeCommonModel) -> int:
        """
        Returns
        -------
        int
            Number of days from the first date to the last
        """

        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self: TypeCommonModel) -> float:
        """
        Returns
        -------
        float
            Length of the timeseries expressed in years assuming all years
            have 365.25 days
        """

        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self: TypeCommonModel) -> float:
        """
        Returns
        -------
        float
            The average number of observations per year
        """

        return self.length / self.yearfrac

    @property
    def max_drawdown_cal_year(self: TypeCommonModel) -> Union[float, Series]:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Returns
        -------
        Union[float, Pandas.Series]
            Maximum drawdown in a single calendar year.
        """
        years = [d.year for d in self.tsdf.index]
        result = (
            self.tsdf.groupby(years)
            .apply(
                lambda prices: (prices / prices.expanding(min_periods=1).max()).min()
                - 1
            )
            .min()
        )
        result.name = "Max Drawdown in cal yr"
        result = result.astype("float64")
        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return result

    def value_to_log(self: TypeCommonModel) -> TypeCommonModel:
        """Converts a valueseries into logarithmic weighted series \n
        Equivalent to LN(value[t] / value[t=0]) in MS Excel

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
        self: TypeCommonModel, method: LiteralNanMethod = "fill"
    ) -> TypeCommonModel:
        """Handling of missing values in a valueseries

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
            self.tsdf.fillna(method="pad", inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def return_nan_handle(
        self: TypeCommonModel, method: LiteralNanMethod = "fill"
    ) -> TypeCommonModel:
        """Handling of missing values in a returnseries

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
            self.tsdf.fillna(value=0.0, inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def to_drawdown_series(self: TypeCommonModel) -> TypeCommonModel:
        """Converts timeseries into a drawdown series

        Returns
        -------
        self
            An object of the same class
        """

        for serie in self.tsdf:
            self.tsdf.loc[:, serie] = drawdown_series(self.tsdf.loc[:, serie])
        return self

    def to_json(
        self: TypeCommonModel, filename: str, directory: Optional[str] = None
    ) -> List[Dict[str, Union[str, bool, ValueType, List[str], List[float]]]]:
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
        List[Dict[str, Union[str, bool, ValueType, List[str], List[float]]]]
            A list of dictionaries with the raw original data of the series
        """
        if not directory:
            directory = path.dirname(path.abspath(__file__))

        cleaner_list = ["label", "tsdf"]
        data = self.__dict__

        output = []
        if "label" in data:
            for item in cleaner_list:
                data.pop(item)
            output.append(dict(data))
        else:
            series = [
                serie.__dict__ for serie in cast(List[Any], data.get("constituents"))
            ]
            for data in series:
                for item in cleaner_list:
                    data.pop(item)
                output.append(data)

        with open(path.join(directory, filename), "w", encoding="utf-8") as jsonfile:
            dump(data, jsonfile, indent=2, sort_keys=False)

        return output

    def to_xlsx(
        self: TypeCommonModel,
        filename: str,
        sheet_title: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> str:
        """Saves the data in the .tsdf DataFrame to an Excel spreadsheet file

        Parameters
        ----------
        filename: str
            Filename that should include .xlsx
        sheet_title: str, optional
            Name of the sheet in the Excel file
        directory: str, optional
            The file directory where the Excel file is saved.
        Returns
        -------
        str
            The Excel file path
        """

        if filename[-5:].lower() != ".xlsx":
            raise NameError("Filename must end with .xlsx")
        if directory:
            sheetfile = path.join(directory, filename)
        else:
            script_path = path.abspath(__file__)
            sheetfile = path.join(path.dirname(script_path), filename)

        wrkbook = Workbook()
        wrksheet = wrkbook.active

        if sheet_title:
            wrksheet.title = sheet_title

        for row in dataframe_to_rows(df=self.tsdf, index=True, header=True):
            wrksheet.append(row)

        wrkbook.save(sheetfile)

        return sheetfile

    def plot_bars(
        self: TypeCommonModel,
        mode: LiteralBarPlotMode = "group",
        tick_fmt: Optional[str] = None,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        labels: Optional[List[str]] = None,
        auto_open: bool = True,
        add_logo: bool = True,
        output_type: LiteralPlotlyOutput = "file",
    ) -> Tuple[Figure, str]:
        """Creates a Plotly Bar Figure

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
        directory: str, optional
            Directory where Plotly html file is saved
        labels: List[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """
        if labels:
            assert (
                len(labels) == self.tsdf.shape[1]
            ), "Must provide same number of labels as items in frame."
        else:
            labels = list(self.tsdf.columns.get_level_values(0))
        if not directory:
            directory = path.join(str(Path.home()), "Documents")
        if not filename:
            filename = "".join(choices(ascii_letters, k=6)) + ".html"
        plotfile = path.join(path.abspath(directory), filename)

        if mode == "overlay":
            opacity = 0.7
        else:
            opacity = None

        fig, logo = load_plotly_dict()
        figure = Figure(fig)
        for item in range(self.tsdf.shape[1]):
            figure.add_bar(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                name=labels[item],
                opacity=opacity,
            )
        figure.update_layout(barmode=mode, yaxis={"tickformat": tick_fmt})

        if add_logo:
            figure.add_layout_image(logo)

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

    def plot_series(
        self: TypeCommonModel,
        mode: LiteralLinePlotMode = "lines",
        tick_fmt: Optional[str] = None,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        labels: Optional[List[str]] = None,
        auto_open: bool = True,
        add_logo: bool = True,
        show_last: bool = False,
        output_type: LiteralPlotlyOutput = "file",
    ) -> Tuple[Figure, str]:
        """Creates a Plotly Figure

        To scale the bubble size, use the attribute sizeref.
        We recommend using the following formula to calculate a sizeref value:
        sizeref = 2. * max(array of size values) / (desired maximum marker size ** 2)

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
        directory: str, optional
            Directory where Plotly html file is saved
        labels: List[str], optional
            A list of labels to manually override using the names of
            the input self.tsdf
        auto_open: bool, default: True
            Determines whether to open a browser window with the plot
        add_logo: bool, default: True
            If True a Captor logo is added to the plot
        show_last: bool, default: False
            If True the last self.tsdf point is highlighted as red dot with a label
        output_type: LiteralPlotlyOutput, default: "file"
            Determines output type

        Returns
        -------
        (plotly.go.Figure, str)
            Plotly Figure and html filename with location
        """

        if labels:
            assert (
                len(labels) == self.tsdf.shape[1]
            ), "Must provide same number of labels as items in frame."
        else:
            labels = list(self.tsdf.columns.get_level_values(0))
        if not directory:
            directory = path.join(str(Path.home()), "Documents")
        if not filename:
            filename = "".join(choices(ascii_letters, k=6)) + ".html"
        plotfile = path.join(path.abspath(directory), filename)

        fig, logo = load_plotly_dict()
        figure = Figure(fig)
        for item in range(self.tsdf.shape[1]):
            figure.add_scatter(
                x=self.tsdf.index,
                y=self.tsdf.iloc[:, item],
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                line={"width": 2.5, "dash": "solid"},
                mode=mode,
                name=labels[item],
            )
        figure.update_layout(yaxis={"tickformat": tick_fmt})

        if add_logo:
            figure.add_layout_image(logo)

        if show_last is True:
            if tick_fmt:
                txt = f"Last {{:{tick_fmt}}}"
            else:
                txt = "Last {}"

            for item in range(self.tsdf.shape[1]):
                figure.add_scatter(
                    x=[self.tsdf.iloc[:, item].index[-1]],
                    y=[self.tsdf.iloc[-1, item]],
                    mode="markers + text",
                    marker={"color": "red", "size": 12},
                    hovertemplate="%{y}<br>%{x|%Y-%m-%d}",
                    showlegend=False,
                    name=labels[item],
                    text=[txt.format(self.tsdf.iloc[-1, item])],
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

    def arithmetic_ret_func(
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/a/arithmeticmean.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series]
            Annualized arithmetic mean of returns
        """

        earlier, later = get_calc_range(
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
                cast(int, earlier) : cast(int, later), self.tsdf.columns.values[0]
            ].count()
            time_factor = how_many / fraction

        result = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change().mean()
            * time_factor
        )

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            name="Arithmetic return",
            dtype="float64",
        )

    def vol_func(
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Union[float, Series]:
        """Based on Pandas .std() which is the equivalent of stdev.s([...])
        in MS Excel \n
        https://www.investopedia.com/terms/v/volatility.asp

        Parameters
        ----------
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        Union[float, Pandas.Series]
            Annualized volatility
        """

        earlier, later = get_calc_range(
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

        result = self.tsdf.loc[
            cast(int, earlier) : cast(int, later)
        ].pct_change().std() * sqrt(time_factor)

        if self.tsdf.shape[1] == 1:
            return float(result[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Volatility",
            dtype="float64",
        )

    def vol_from_var_func(
        self: TypeCommonModel,
        level: float = 0.95,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        interpolation: LiteralQuantileInterp = "lower",
        drift_adjust: bool = False,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Union[float, Series]:
        """
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
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series]
            Implied annualized volatility from the Downside VaR using the
            assumption that returns are normally distributed.
        """
        return var_implied_vol_and_target_func(
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
        self: TypeCommonModel,
        target_vol: float = 0.175,
        level: float = 0.95,
        min_leverage_local: float = 0.0,
        max_leverage_local: float = 99999.0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        interpolation: LiteralQuantileInterp = "lower",
        drift_adjust: bool = False,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Union[float, Series]:
        """A position weight multiplier from the ratio between a VaR implied
        volatility and a given target volatility. Multiplier = 1.0 -> target met

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
        drift_adjust: bool, default: False
            An adjustment to remove the bias implied by the average return
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series]
            A position weight multiplier from the ratio between a VaR implied
            volatility and a given target volatility. Multiplier = 1.0 -> target met
        """
        return var_implied_vol_and_target_func(
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
        self: TypeCommonModel,
        level: float = 0.95,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

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
        Union[float, Pandas.Series]
            Downside Conditional Value At Risk "CVaR"
        """

        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        cvar_df = self.tsdf.loc[cast(int, earlier) : cast(int, later)].copy(deep=True)
        result = [
            cvar_df.loc[:, x]
            .pct_change()
            .sort_values()
            .iloc[: int(ceil((1 - level) * cvar_df.loc[:, x].pct_change().count()))]
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
        self: TypeCommonModel,
        min_accepted_return: float = 0.0,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Union[float, Series]:
        """The standard deviation of returns that are below a Minimum Accepted
        Return of zero.
        It is used to calculate the Sortino Ratio \n
        https://www.investopedia.com/terms/d/downside-deviation.asp

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
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series]
            Downside deviation
        """

        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        how_many = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
            .count(numeric_only=True)
        )
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction

        dddf = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
            .sub(min_accepted_return / time_factor)
        )

        result = sqrt((dddf[dddf < 0.0] ** 2).sum() / how_many) * sqrt(time_factor)

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Downside deviation",
            dtype="float64",
        )

    def geo_ret_func(
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/c/cagr.asp

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
        Union[float, Pandas.Series]
            Compounded Annual Growth Rate (CAGR)
        """
        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        fraction = (later - earlier).days / 365.25

        if (
            0.0 in self.tsdf.loc[earlier].tolist()
            or self.tsdf.loc[[earlier, later]].lt(0.0).any().any()
        ):
            raise ValueError(
                "Geometric return cannot be calculated due to an initial "
                "value being zero or a negative value."
            )

        result = (self.tsdf.iloc[-1] / self.tsdf.iloc[0]) ** (1 / fraction) - 1

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Geometric return",
            dtype="float64",
        )

    def kurtosis_func(
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/k/kurtosis.asp

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
        Union[float, Pandas.Series]
            Kurtosis of the return distribution
        """

        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = kurtosis(
            self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change(),
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
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        min_periods: int = 1,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

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
        Union[float, Pandas.Series]
            Maximum drawdown without any limit on date range
        """
        earlier, later = get_calc_range(
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
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """The share of percentage changes that are greater than zero

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
        Union[float, Pandas.Series]
            The share of percentage changes that are greater than zero
        """
        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        pos = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()[1:][
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change()[1:]
                > 0.0
            ]
            .count()
        )
        tot = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()[1:]
            .count()
        )
        result = pos / tot
        result.name = "Positive Share"
        result = result.astype("float64")
        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return result

    def ret_vol_ratio_func(
        self: TypeCommonModel,
        riskfree_rate: Optional[float] = 0.0,
        riskfree_column: int = -1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Union[float, Series]:
        """The ratio of annualized arithmetic mean of returns and annualized
        volatility or, if riskfree return provided, Sharpe ratio calculated
        as ( geometric return - risk-free return ) / volatility. The latter ratio
        implies that the riskfree asset has zero volatility. \n
        https://www.investopedia.com/terms/s/sharperatio.asp

        Parameters
        ----------
        riskfree_rate : float, optional
            The return of the zero volatility asset used to calculate Sharpe ratio
        riskfree_column : int, default: -1
            The return of the zero volatility asset used to calculate Sharpe ratio
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series]
            Ratio of the annualized arithmetic mean of returns and annualized
            volatility or,
            if risk-free return provided, Sharpe ratio
        """

        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            how_many = (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[:, 0].count()
            )
            fraction = (later - earlier).days / 365.25
            time_factor = how_many / fraction

        ratios = []
        if riskfree_rate is None:
            if isinstance(riskfree_column, int):
                riskfree = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                    :, riskfree_column
                ]
                riskfree_item = self.tsdf.iloc[:, riskfree_column].name
            else:
                raise ValueError("base_column argument should be an integer.")

            for item in self.tsdf:
                if item == riskfree_item:
                    ratios.append(0.0)
                else:
                    longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                        :, item
                    ]
                    ret = float(longdf.pct_change().mean() * time_factor)
                    riskfree_ret = float(riskfree.pct_change().mean() * time_factor)
                    vol = float(longdf.pct_change().std() * sqrt(time_factor))
                    ratios.append((ret - riskfree_ret) / vol)
        else:
            for item in self.tsdf:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :, item
                ]
                ret = float(longdf.pct_change().mean() * time_factor)
                vol = float(longdf.pct_change().std() * sqrt(time_factor))
                ratios.append((ret - riskfree_rate) / vol)

        if self.tsdf.shape[1] == 1:
            return ratios[0]
        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name="Return vol ratio",
            dtype="float64",
        )

    def skew_func(
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/s/skewness.asp

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
        Union[float, Pandas.Series]
            Skew of the return distribution
        """
        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = skew(
            a=self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change().values,
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

    def sortino_ratio_func(
        self: TypeCommonModel,
        riskfree_rate: Optional[float] = 0.0,
        riskfree_column: int = -1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[int] = None,
    ) -> Union[float, Series]:
        """The Sortino ratio calculated as ( return - risk free return )
        / downside deviation. The ratio implies that the riskfree asset has zero
        volatility, and a minimum acceptable return of zero. The ratio is
        calculated using the annualized arithmetic mean of returns. \n
        https://www.investopedia.com/terms/s/sortinoratio.asp

        Parameters
        ----------
        riskfree_rate : float, optional
            The return of the zero volatility asset
        riskfree_column : int, default: -1
            The return of the zero volatility asset used to calculate Sharpe ratio
        months_from_last : int, optional
            number of months offset as positive integer. Overrides use of from_date
            and to_date
        from_date : datetime.date, optional
            Specific from date
        to_date : datetime.date, optional
            Specific to date
        periods_in_a_year_fixed : int, optional
            Allows locking the periods-in-a-year to simplify test cases and
            comparisons

        Returns
        -------
        Union[float, Pandas.Series]
            Sortino ratio calculated as ( return - riskfree return ) /
            downside deviation
        """
        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        how_many = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[:, 0].count()
        )
        fraction = (later - earlier).days / 365.25

        if periods_in_a_year_fixed:
            time_factor = periods_in_a_year_fixed
        else:
            time_factor = how_many / fraction

        ratios = []
        if riskfree_rate is None:
            if isinstance(riskfree_column, int):
                riskfree = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                    :, riskfree_column
                ]
                riskfree_item = self.tsdf.iloc[:, riskfree_column].name
            else:
                raise ValueError("base_column argument should be an integer.")

            for item in self.tsdf:
                if item == riskfree_item:
                    ratios.append(0.0)
                else:
                    longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                        :, item
                    ]
                    ret = float(longdf.pct_change().mean() * time_factor)
                    riskfree_ret = float(riskfree.pct_change().mean() * time_factor)
                    dddf = longdf.pct_change()
                    downdev = float(
                        sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / how_many)
                        * sqrt(time_factor)
                    )
                    ratios.append((ret - riskfree_ret) / downdev)

        else:
            for item in self.tsdf:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :, item
                ]
                ret = float(longdf.pct_change().mean() * time_factor)
                dddf = longdf.pct_change()
                downdev = float(
                    sqrt((dddf[dddf.values < 0.0].values ** 2).sum() / how_many)
                    * sqrt(time_factor)
                )
                ratios.append((ret - riskfree_rate) / downdev)

        if self.tsdf.shape[1] == 1:
            return ratios[0]
        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name="Sortino ratio",
            dtype="float64",
        )

    def value_ret_func(
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """
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
        Union[float, Pandas.Series]
            Simple return
        """

        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        if 0.0 in self.tsdf.iloc[0].tolist():
            raise ValueError(
                f"Simple return cannot be calculated due to an "
                f"initial value being zero. ({self.tsdf.head(3)})"
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
        self: TypeCommonModel, year: int, month: Optional[int] = None
    ) -> Union[float, Series]:
        """
        Parameters
        ----------
        year : int
            Calendar year of the period to calculate.
        month : int, optional
            Calendar month of the period to calculate.

        Returns
        -------
        Union[float, Pandas.Series]
            Simple return for a specific calendar period
        """

        if month is None:
            period = str(year)
        else:
            period = "-".join([str(year), str(month).zfill(2)])
        vrdf = self.tsdf.copy()
        vrdf.index = DatetimeIndex(vrdf.index)
        result = vrdf.pct_change().copy()
        result = result.loc[period] + 1
        result = result.apply(cumprod, axis="index").iloc[-1] - 1
        result.name = period
        result = result.astype("float64")
        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return result

    def var_down_func(
        self: TypeCommonModel,
        level: float = 0.95,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        interpolation: LiteralQuantileInterp = "lower",
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/v/var.asp
        Downside Value At Risk, "VaR". The equivalent of
        percentile.inc([...], 1-level) over returns in MS Excel.

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
        Union[float, Pandas.Series]
            Downside Value At Risk
        """
        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
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
        self: TypeCommonModel,
        observations: int = 1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """Most negative percentage change over a rolling number of observations
        within a chosen date range

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
        Union[float, Pandas.Series]
            Most negative percentage change over a rolling number of observations
            within a chosen date range
        """
        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        result = (
            self.tsdf.loc[cast(int, earlier) : cast(int, later)]
            .pct_change()
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
        self: TypeCommonModel,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
    ) -> Union[float, Series]:
        """https://www.investopedia.com/terms/z/zscore.asp

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
        Union[float, Pandas.Series]
            Z-score as (last return - mean return) / standard deviation of returns
        """

        earlier, later = get_calc_range(
            data=self.tsdf,
            months_offset=months_from_last,
            from_dt=from_date,
            to_dt=to_date,
        )
        zscframe = self.tsdf.loc[cast(int, earlier) : cast(int, later)].pct_change()
        result = (zscframe.iloc[-1] - zscframe.mean()) / zscframe.std()

        if self.tsdf.shape[1] == 1:
            return float(result.iloc[0])
        return Series(
            data=result,
            index=self.tsdf.columns,
            name="Z-score",
            dtype="float64",
        )
