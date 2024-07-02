"""Defining the OpenFrame class."""

# mypy: disable-error-code="index,assignment"
from __future__ import annotations

import datetime as dt
from copy import deepcopy
from functools import reduce
from inspect import stack
from logging import warning
from pathlib import Path
from typing import Callable, Optional, Union, cast

import statsmodels.api as sm  # type: ignore[import-untyped,unused-ignore]
from numpy import (
    append,
    array,
    cov,
    cumprod,
    divide,
    dot,
    float64,
    inf,
    isinf,
    linspace,
    log,
    nan,
    sqrt,
    square,
    std,
    zeros,
)
from numpy import (
    sum as npsum,
)
from numpy.typing import NDArray
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Int64Dtype,
    MultiIndex,
    Series,
    concat,
    merge,
)
from plotly.graph_objs import Figure  # type: ignore[import-untyped,unused-ignore]
from plotly.io import to_html  # type: ignore[import-untyped,unused-ignore]
from plotly.offline import plot  # type: ignore[import-untyped,unused-ignore]
from pydantic import DirectoryPath, field_validator
from scipy.optimize import minimize  # type: ignore[import-untyped,unused-ignore]

# noinspection PyProtectedMember
from statsmodels.regression.linear_model import (  # type: ignore[import-untyped,unused-ignore]
    OLSResults,
)
from typing_extensions import Self

from openseries._common_model import _CommonModel
from openseries.datefixer import do_resample_to_business_period_ends
from openseries.load_plotly import load_plotly_dict
from openseries.series import OpenTimeSeries
from openseries.simulation import _random_generator
from openseries.types import (
    CountriesType,
    DaysInYearType,
    LiteralBizDayFreq,
    LiteralCaptureRatio,
    LiteralFrameProps,
    LiteralHowMerge,
    LiteralLinePlotMode,
    LiteralOlsFitCovType,
    LiteralOlsFitMethod,
    LiteralPandasReindexMethod,
    LiteralPlotlyJSlib,
    LiteralPlotlyOutput,
    LiteralPortfolioWeightings,
    LiteralTrunc,
    OpenFramePropertiesList,
    ValueType,
)

__all__ = [
    "OpenFrame",
    "constrain_optimized_portfolios",
    "efficient_frontier",
    "prepare_plot_data",
    "sharpeplot",
    "simulate_portfolios",
]


# noinspection PyUnresolvedReferences
class OpenFrame(_CommonModel):

    """
    OpenFrame objects hold OpenTimeSeries in the list constituents.

    The intended use is to allow comparisons across these timeseries.

    Parameters
    ----------
    constituents: list[OpenTimeSeries]
        List of objects of Class OpenTimeSeries
    weights: list[float], optional
        List of weights in float format.

    Returns
    -------
    OpenFrame
        Object of the class OpenFrame

    """

    constituents: list[OpenTimeSeries]
    tsdf: DataFrame = DataFrame(dtype="float64")
    weights: Optional[list[float]] = None

    # noinspection PyMethodParameters
    @field_validator("constituents")  # type: ignore[misc]
    def _check_labels_unique(
        cls: OpenFrame,  # noqa: N805
        tseries: list[OpenTimeSeries],
    ) -> list[OpenTimeSeries]:
        """Pydantic validator ensuring that OpenFrame labels are unique."""
        labls = [x.label for x in tseries]
        if len(set(labls)) != len(labls):
            msg = "TimeSeries names/labels must be unique"
            raise ValueError(msg)
        return tseries

    def __init__(
        self: Self,
        constituents: list[OpenTimeSeries],
        weights: Optional[list[float]] = None,
    ) -> None:
        """
        OpenFrame objects hold OpenTimeSeries in the list constituents.

        The intended use is to allow comparisons across these timeseries.

        Parameters
        ----------
        constituents: list[OpenTimeSeries]
            List of objects of Class OpenTimeSeries
        weights: list[float], optional
            List of weights in float format.

        Returns
        -------
        OpenFrame
            Object of the class OpenFrame

        """
        super().__init__(  # type: ignore[call-arg]
            constituents=constituents,
            weights=weights,
        )

        self.constituents = constituents
        self.weights = weights
        self._set_tsdf()

    def _set_tsdf(self: Self) -> None:
        """Set the tsdf DataFrame."""
        if self.constituents is not None and len(self.constituents) != 0:
            self.tsdf = reduce(
                lambda left, right: concat([left, right], axis="columns", sort=True),
                [x.tsdf for x in self.constituents],
            )
        else:
            warning("OpenFrame() was passed an empty list.")

    def from_deepcopy(self: Self) -> Self:
        """
        Create copy of the OpenFrame object.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        return deepcopy(self)

    def merge_series(
        self: Self,
        how: LiteralHowMerge = "outer",
    ) -> Self:
        """
        Merge index of Pandas Dataframes of the constituent OpenTimeSeries.

        Parameters
        ----------
        how: LiteralHowMerge, default: "outer"
            The Pandas merge method.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        lvl_zero = list(self.columns_lvl_zero)
        self.tsdf = reduce(
            lambda left, right: merge(
                left=left,
                right=right,
                how=how,
                left_index=True,
                right_index=True,
            ),
            [x.tsdf for x in self.constituents],
        )

        mapper = dict(zip(self.columns_lvl_zero, lvl_zero))
        self.tsdf = self.tsdf.rename(columns=mapper, level=0)

        if self.tsdf.empty:
            msg = (
                "Merging OpenTimeSeries DataFrames with "
                f"argument how={how} produced an empty DataFrame."
            )
            raise ValueError(msg)

        if how == "inner":
            for xerie in self.constituents:
                xerie.tsdf = xerie.tsdf.loc[self.tsdf.index]
        return self

    def all_properties(
        self: Self,
        properties: Optional[list[LiteralFrameProps]] = None,
    ) -> DataFrame:
        """
        Calculate chosen timeseries properties.

        Parameters
        ----------
        properties: list[LiteralFrameProps], optional
            The properties to calculate. Defaults to calculating all available.

        Returns
        -------
        pandas.DataFrame
            Properties of the contituent OpenTimeSeries

        """
        if properties:
            props = OpenFramePropertiesList(*properties)
            prop_list = [getattr(self, x) for x in props]
        else:
            prop_list = [
                getattr(self, x) for x in OpenFramePropertiesList.allowed_strings
            ]
        return cast(DataFrame, concat(prop_list, axis="columns").T)

    @property
    def lengths_of_items(self: Self) -> Series[int]:
        """
        Number of observations of all constituents.

        Returns
        -------
        Pandas.Series[int]
            Number of observations of all constituents

        """
        return Series(
            data=[int(self.tsdf.loc[:, d].count()) for d in self.tsdf],
            index=self.tsdf.columns,
            name="observations",
            dtype=Int64Dtype(),
        )

    @property
    def item_count(self: Self) -> int:
        """
        Number of constituents.

        Returns
        -------
        int
            Number of constituents

        """
        return len(self.constituents)

    @property
    def columns_lvl_zero(self: Self) -> list[str]:
        """
        Level 0 values of the MultiIndex columns in the .tsdf DataFrame.

        Returns
        -------
        list[str]
            Level 0 values of the MultiIndex columns in the .tsdf DataFrame

        """
        return list(self.tsdf.columns.get_level_values(0))

    @property
    def columns_lvl_one(self: Self) -> list[ValueType]:
        """
        Level 1 values of the MultiIndex columns in the .tsdf DataFrame.

        Returns
        -------
        list[ValueType]
            Level 1 values of the MultiIndex columns in the .tsdf DataFrame

        """
        return list(self.tsdf.columns.get_level_values(1))

    @property
    def first_indices(self: Self) -> Series[dt.date]:
        """
        The first dates in the timeseries of all constituents.

        Returns
        -------
        Pandas.Series[dt.date]
            The first dates in the timeseries of all constituents

        """
        return Series(
            data=[i.first_idx for i in self.constituents],
            index=self.tsdf.columns,
            name="first indices",
            dtype="datetime64[ns]",
        ).dt.date

    @property
    def last_indices(self: Self) -> Series[dt.date]:
        """
        The last dates in the timeseries of all constituents.

        Returns
        -------
        Pandas.Series[dt.date]
            The last dates in the timeseries of all constituents

        """
        return Series(
            data=[i.last_idx for i in self.constituents],
            index=self.tsdf.columns,
            name="last indices",
            dtype="datetime64[ns]",
        ).dt.date

    @property
    def span_of_days_all(self: Self) -> Series[int]:
        """
        Number of days from the first date to the last for all items in the frame.

        Returns
        -------
        Pandas.Series[int]
            Number of days from the first date to the last for all
            items in the frame.

        """
        return Series(
            data=[c.span_of_days for c in self.constituents],
            index=self.tsdf.columns,
            name="span of days",
            dtype=Int64Dtype(),
        )

    def value_to_ret(self: Self) -> Self:
        """
        Convert series of values into series of returns.

        Returns
        -------
        OpenFrame
            The returns of the values in the series

        """
        self.tsdf = self.tsdf.pct_change(fill_method=cast(str, None))
        self.tsdf.iloc[0] = 0
        new_labels = [ValueType.RTRN] * self.item_count
        arrays = [self.tsdf.columns.get_level_values(0), new_labels]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def value_to_diff(self: Self, periods: int = 1) -> Self:
        """
        Convert series of values to series of their period differences.

        Parameters
        ----------
        periods: int, default: 1
            The number of periods between observations over which difference
            is calculated

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        self.tsdf = self.tsdf.diff(periods=periods)
        self.tsdf.iloc[0] = 0
        new_labels = [ValueType.RTRN] * self.item_count
        arrays = [self.tsdf.columns.get_level_values(0), new_labels]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def to_cumret(self: Self) -> Self:
        """
        Convert series of returns into cumulative series of values.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        if any(
            x == ValueType.PRICE
            for x in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            self.value_to_ret()

        self.tsdf = self.tsdf.add(1.0)
        self.tsdf = self.tsdf.apply(cumprod, axis="index") / self.tsdf.iloc[0]
        new_labels = [ValueType.PRICE] * self.item_count
        arrays = [self.tsdf.columns.get_level_values(0), new_labels]
        self.tsdf.columns = MultiIndex.from_arrays(arrays)
        return self

    def resample(
        self: Self,
        freq: Union[LiteralBizDayFreq, str] = "BME",
    ) -> Self:
        """
        Resample the timeseries frequency.

        Parameters
        ----------
        freq: Union[LiteralBizDayFreq, str], default "BME"
            The date offset string that sets the resampled frequency

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        self.tsdf.index = DatetimeIndex(self.tsdf.index)
        self.tsdf = self.tsdf.resample(freq).last()
        self.tsdf.index = Index(d.date() for d in DatetimeIndex(self.tsdf.index))
        for xerie in self.constituents:
            xerie.tsdf.index = DatetimeIndex(xerie.tsdf.index)
            xerie.tsdf = xerie.tsdf.resample(freq).last()
            xerie.tsdf.index = Index(
                dejt.date() for dejt in DatetimeIndex(xerie.tsdf.index)
            )

        return self

    def resample_to_business_period_ends(
        self: Self,
        freq: LiteralBizDayFreq = "BME",
        countries: CountriesType = "SE",
        method: LiteralPandasReindexMethod = "nearest",
    ) -> Self:
        """
        Resamples timeseries frequency to the business calendar month end dates.

        Stubs left in place. Stubs will be aligned to the shortest stub.

        Parameters
        ----------
        freq: LiteralBizDayFreq, default "BME"
            The date offset string that sets the resampled frequency
        countries: CountriesType, default: "SE"
            (List of) country code(s) according to ISO 3166-1 alpha-2
            to create a business day calendar used for date adjustments
        method: LiteralPandasReindexMethod, default: nearest
            Controls the method used to align values across columns

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        head: Series[float] = self.tsdf.loc[self.first_indices.max()].copy()
        tail: Series[float] = self.tsdf.loc[self.last_indices.min()].copy()
        dates = do_resample_to_business_period_ends(
            data=self.tsdf,
            head=head,
            tail=tail,
            freq=freq,
            countries=countries,
        )
        self.tsdf = self.tsdf.reindex([deyt.date() for deyt in dates], method=method)
        for xerie in self.constituents:
            xerie.tsdf = xerie.tsdf.reindex(
                [deyt.date() for deyt in dates],
                method=method,
            )
        return self

    def ewma_risk(
        self: Self,
        lmbda: float = 0.94,
        day_chunk: int = 11,
        dlta_degr_freedms: int = 0,
        first_column: int = 0,
        second_column: int = 1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> DataFrame:
        """
        Exponentially Weighted Moving Average Volatilities and Correlation.

        Exponentially Weighted Moving Average (EWMA) for Volatilities and
        Correlation. https://www.investopedia.com/articles/07/ewma.asp.

        Parameters
        ----------
        lmbda: float, default: 0.94
            Scaling factor to determine weighting.
        day_chunk: int, default: 11
            Sampling the data which is assumed to be daily.
        dlta_degr_freedms: int, default: 0
            Variance bias factor taking the value 0 or 1.
        first_column: int, default: 0
            Column of first timeseries.
        second_column: int, default: 1
            Column of second timeseries.
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
        Pandas.DataFrame
            Series volatilities and correlation

        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        if periods_in_a_year_fixed is None:
            fraction = (later - earlier).days / 365.25
            how_many = (
                self.tsdf.loc[cast(int, earlier) : cast(int, later)].count().iloc[0]
            )
            time_factor = how_many / fraction
        else:
            time_factor = periods_in_a_year_fixed

        corr_label = (
            cast(tuple[str, str], self.tsdf.iloc[:, first_column].name)[0]
            + "_VS_"
            + cast(tuple[str, str], self.tsdf.iloc[:, second_column].name)[0]
        )
        cols = [
            cast(tuple[str, str], self.tsdf.iloc[:, first_column].name)[0],
            cast(tuple[str, str], self.tsdf.iloc[:, second_column].name)[0],
        ]

        data = self.tsdf.loc[cast(int, earlier) : cast(int, later)].copy()

        for rtn in cols:
            data[rtn, ValueType.RTRN] = (
                data.loc[:, (rtn, ValueType.PRICE)].apply(log).diff()
            )

        raw_one = [
            data.loc[:, (cols[0], ValueType.RTRN)]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]
        raw_two = [
            data.loc[:, (cols[1], ValueType.RTRN)]
            .iloc[1:day_chunk]
            .std(ddof=dlta_degr_freedms)
            * sqrt(time_factor),
        ]
        raw_cov = [
            cov(
                m=data.loc[:, (cols[0], ValueType.RTRN)].iloc[1:day_chunk].to_numpy(),
                y=data.loc[:, (cols[1], ValueType.RTRN)].iloc[1:day_chunk].to_numpy(),
                ddof=dlta_degr_freedms,
            )[0][1],
        ]
        raw_corr = [raw_cov[0] / (2 * raw_one[0] * raw_two[0])]

        for _, row in data.iloc[1:].iterrows():
            tmp_raw_one = sqrt(
                square(row.loc[cols[0], ValueType.RTRN]) * time_factor * (1 - lmbda)
                + square(raw_one[-1]) * lmbda,
            )
            tmp_raw_two = sqrt(
                square(row.loc[cols[1], ValueType.RTRN]) * time_factor * (1 - lmbda)
                + square(raw_two[-1]) * lmbda,
            )
            tmp_raw_cov = (
                row.loc[cols[0], ValueType.RTRN]
                * row.loc[cols[1], ValueType.RTRN]
                * time_factor
                * (1 - lmbda)
                + raw_cov[-1] * lmbda
            )
            tmp_raw_corr = tmp_raw_cov / (2 * tmp_raw_one * tmp_raw_two)
            raw_one.append(tmp_raw_one)
            raw_two.append(tmp_raw_two)
            raw_cov.append(tmp_raw_cov)
            raw_corr.append(tmp_raw_corr)

        return DataFrame(
            index=[*cols, corr_label],
            columns=data.index,
            data=[raw_one, raw_two, raw_corr],
        ).T

    @property
    def correl_matrix(self: Self) -> DataFrame:
        """
        Correlation matrix.

        Returns
        -------
        Pandas.DataFrame
            Correlation matrix

        """
        corr_matrix = self.tsdf.pct_change(fill_method=cast(str, None)).corr(
            method="pearson",
            min_periods=1,
        )
        corr_matrix.columns = corr_matrix.columns.droplevel(level=1)
        corr_matrix.index = corr_matrix.index.droplevel(level=1)
        corr_matrix.index.name = "Correlation"
        return corr_matrix

    def add_timeseries(
        self: Self,
        new_series: OpenTimeSeries,
    ) -> Self:
        """
        To add an OpenTimeSeries object.

        Parameters
        ----------
        new_series: OpenTimeSeries
            The timeseries to add

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        self.constituents += [new_series]
        # noinspection PyUnreachableCode
        self.tsdf = concat([self.tsdf, new_series.tsdf], axis="columns", sort=True)
        return self

    def delete_timeseries(self: Self, lvl_zero_item: str) -> Self:
        """
        To delete an OpenTimeSeries object.

        Parameters
        ----------
        lvl_zero_item: str
            The .tsdf column level 0 value of the timeseries to delete

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        if self.weights:
            new_c, new_w = [], []
            for serie, weight in zip(self.constituents, self.weights):
                if serie.label != lvl_zero_item:
                    new_c.append(serie)
                    new_w.append(weight)
            self.constituents = new_c
            self.weights = new_w
        else:
            self.constituents = [
                item for item in self.constituents if item.label != lvl_zero_item
            ]
        self.tsdf = self.tsdf.drop(lvl_zero_item, axis="columns", level=0)
        return self

    def trunc_frame(
        self: Self,
        start_cut: Optional[dt.date] = None,
        end_cut: Optional[dt.date] = None,
        where: LiteralTrunc = "both",
    ) -> Self:
        """
        Truncate DataFrame such that all timeseries have the same time span.

        Parameters
        ----------
        start_cut: datetime.date, optional
            New first date
        end_cut: datetime.date, optional
            New last date
        where: LiteralTrunc, default: both
            Determines where dataframe is truncated also when start_cut
            or end_cut is None.

        Returns
        -------
        OpenFrame
            An OpenFrame object

        """
        if not start_cut and where in ["before", "both"]:
            start_cut = self.first_indices.max()
        if not end_cut and where in ["after", "both"]:
            end_cut = self.last_indices.min()
        self.tsdf = self.tsdf.sort_index()
        self.tsdf = self.tsdf.truncate(before=start_cut, after=end_cut, copy=False)

        for xerie in self.constituents:
            xerie.tsdf = xerie.tsdf.truncate(
                before=start_cut,
                after=end_cut,
                copy=False,
            )
        if len(set(self.first_indices)) != 1:
            msg = (
                f"One or more constituents still "
                f"not truncated to same start dates.\n"
                f"{self.tsdf.head()}"
            )
            warning(msg=msg)
        if len(set(self.last_indices)) != 1:
            msg = (
                f"One or more constituents still "
                f"not truncated to same end dates.\n"
                f"{self.tsdf.tail()}"
            )
            warning(msg=msg)
        return self

    def relative(
        self: Self,
        long_column: int = 0,
        short_column: int = 1,
        *,
        base_zero: bool = True,
    ) -> None:
        """
        Calculate cumulative relative return between two series.

        Parameters
        ----------
        long_column: int, default: 0
            Column number of timeseries bought
        short_column: int, default: 1
            Column number of timeseries sold
        base_zero: bool, default: True
            If set to False 1.0 is added to allow for a capital base and
            to allow a volatility calculation

        """
        rel_label = (
            cast(tuple[str, str], self.tsdf.iloc[:, long_column].name)[0]
            + "_over_"
            + cast(tuple[str, str], self.tsdf.iloc[:, short_column].name)[0]
        )
        if base_zero:
            self.tsdf[rel_label, ValueType.RELRTRN] = (
                self.tsdf.iloc[:, long_column] - self.tsdf.iloc[:, short_column]
            )
        else:
            self.tsdf[rel_label, ValueType.RELRTRN] = (
                1.0 + self.tsdf.iloc[:, long_column] - self.tsdf.iloc[:, short_column]
            )
        self.constituents += [
            OpenTimeSeries.from_df(self.tsdf.iloc[:, -1]),
        ]

    def tracking_error_func(
        self: Self,
        base_column: Union[tuple[str, ValueType], int] = -1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Series[float]:
        """
        Tracking Error.

        Calculates Tracking Error which is the standard deviation of the
        difference between the fund and its index returns.
        https://www.investopedia.com/terms/t/trackingerror.asp.

        Parameters
        ----------
        base_column: Union[tuple[str, ValueType], int], default: -1
            Column of timeseries that is the denominator in the ratio.
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
        Pandas.Series[float]
            Tracking Errors

        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        if isinstance(base_column, tuple):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                :,
                base_column,
            ]
            short_item = base_column
            short_label = cast(
                tuple[str, ValueType],
                self.tsdf.loc[:, base_column].name,
            )[0]
        elif isinstance(base_column, int):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                :,
                base_column,
            ]
            short_item = self.tsdf.iloc[
                :,
                base_column,
            ].name
            short_label = cast(tuple[str, str], self.tsdf.iloc[:, base_column].name)[0]
        else:
            msg = "base_column should be a tuple[str, ValueType] or an integer."
            raise TypeError(
                msg,
            )

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = float(shortdf.count() / fraction)

        terrors = []
        for item in self.tsdf:
            if item == short_item:
                terrors.append(0.0)
            else:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :,
                    item,
                ]
                # noinspection PyTypeChecker
                relative = 1.0 + longdf - shortdf
                vol = float(
                    relative.pct_change(fill_method=cast(str, None)).std()
                    * sqrt(time_factor),
                )
                terrors.append(vol)

        return Series(
            data=terrors,
            index=self.tsdf.columns,
            name=f"Tracking Errors vs {short_label}",
            dtype="float64",
        )

    def info_ratio_func(
        self: Self,
        base_column: Union[tuple[str, ValueType], int] = -1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Series[float]:
        """
        Information Ratio.

        The Information Ratio equals ( fund return less index return ) divided
        by the Tracking Error. And the Tracking Error is the standard deviation of
        the difference between the fund and its index returns.
        The ratio is calculated using the annualized arithmetic mean of returns.

        Parameters
        ----------
        base_column: Union[tuple[str, ValueType], int], default: -1
            Column of timeseries that is the denominator in the ratio.
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
        Pandas.Series[float]
            Information Ratios

        """
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        if isinstance(base_column, tuple):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                :,
                base_column,
            ]
            short_item = base_column
            short_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, base_column].name,
            )[0]
        elif isinstance(base_column, int):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                :,
                base_column,
            ]
            short_item = self.tsdf.iloc[
                :,
                base_column,
            ].name
            short_label = cast(tuple[str, str], self.tsdf.iloc[:, base_column].name)[0]
        else:
            msg = "base_column should be a tuple[str, ValueType] or an integer."
            raise TypeError(
                msg,
            )

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = float(shortdf.count() / fraction)

        ratios = []
        for item in self.tsdf:
            if item == short_item:
                ratios.append(0.0)
            else:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :,
                    item,
                ]
                # noinspection PyTypeChecker
                relative = 1.0 + longdf - shortdf
                ret = float(
                    relative.pct_change(fill_method=cast(str, None)).mean()
                    * time_factor,
                )
                vol = float(
                    relative.pct_change(fill_method=cast(str, None)).std()
                    * sqrt(time_factor),
                )
                ratios.append(ret / vol)

        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name=f"Info Ratios vs {short_label}",
            dtype="float64",
        )

    def capture_ratio_func(  # noqa: C901
        self: Self,
        ratio: LiteralCaptureRatio,
        base_column: Union[tuple[str, ValueType], int] = -1,
        months_from_last: Optional[int] = None,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> Series[float]:
        """
        Capture Ratio.

        The Up (Down) Capture Ratio is calculated by dividing the CAGR
        of the asset during periods that the benchmark returns are positive (negative)
        by the CAGR of the benchmark during the same periods.
        CaptureRatio.BOTH is the Up ratio divided by the Down ratio.
        Source: 'Capture Ratios: A Popular Method of Measuring Portfolio Performance
        in Practice', Don R. Cox and Delbert C. Goff, Journal of Economics and
        Finance Education (Vol 2 Winter 2013).
        https://www.economics-finance.org/jefe/volume12-2/11ArticleCox.pdf.

        Parameters
        ----------
        ratio: LiteralCaptureRatio
            The ratio to calculate
        base_column: Union[tuple[str, ValueType], int], default: -1
            Column of timeseries that is the denominator in the ratio.
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
        Pandas.Series[float]
            Capture Ratios

        """
        loss_limit: float = 0.0
        earlier, later = self.calc_range(months_from_last, from_date, to_date)
        fraction = (later - earlier).days / 365.25

        if isinstance(base_column, tuple):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                :,
                base_column,
            ]
            short_item = base_column
            short_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, base_column].name,
            )[0]
        elif isinstance(base_column, int):
            shortdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].iloc[
                :,
                base_column,
            ]
            short_item = self.tsdf.iloc[
                :,
                base_column,
            ].name
            short_label = cast(tuple[str, str], self.tsdf.iloc[:, base_column].name)[0]
        else:
            msg = "base_column should be a tuple[str, ValueType] or an integer."
            raise TypeError(
                msg,
            )

        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = float(shortdf.count() / fraction)

        ratios = []
        for item in self.tsdf:
            if item == short_item:
                ratios.append(0.0)
            else:
                longdf = self.tsdf.loc[cast(int, earlier) : cast(int, later)].loc[
                    :,
                    item,
                ]
                if ratio == "up":
                    uparray = (
                        longdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_rtrn = uparray.prod() ** (1 / (len(uparray) / time_factor)) - 1
                    upidxarray = (
                        shortdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_idx_return = (
                        upidxarray.prod() ** (1 / (len(upidxarray) / time_factor)) - 1
                    )
                    ratios.append(up_rtrn / up_idx_return)
                elif ratio == "down":
                    downarray = (
                        longdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_return = (
                        downarray.prod() ** (1 / (len(downarray) / time_factor)) - 1
                    )
                    downidxarray = (
                        shortdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_idx_return = (
                        downidxarray.prod() ** (1 / (len(downidxarray) / time_factor))
                        - 1
                    )
                    ratios.append(down_return / down_idx_return)
                elif ratio == "both":
                    uparray = (
                        longdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_rtrn = uparray.prod() ** (1 / (len(uparray) / time_factor)) - 1
                    upidxarray = (
                        shortdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            > loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    up_idx_return = (
                        upidxarray.prod() ** (1 / (len(upidxarray) / time_factor)) - 1
                    )
                    downarray = (
                        longdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_return = (
                        downarray.prod() ** (1 / (len(downarray) / time_factor)) - 1
                    )
                    downidxarray = (
                        shortdf.pct_change(fill_method=cast(str, None))[
                            shortdf.pct_change(fill_method=cast(str, None)).to_numpy()
                            < loss_limit
                        ]
                        .add(1)
                        .to_numpy()
                    )
                    down_idx_return = (
                        downidxarray.prod() ** (1 / (len(downidxarray) / time_factor))
                        - 1
                    )
                    ratios.append(
                        (up_rtrn / up_idx_return) / (down_return / down_idx_return),
                    )

        if ratio == "up":
            resultname = f"Up Capture Ratios vs {short_label}"
        elif ratio == "down":
            resultname = f"Down Capture Ratios vs {short_label}"
        else:
            resultname = f"Up-Down Capture Ratios vs {short_label}"

        return Series(
            data=ratios,
            index=self.tsdf.columns,
            name=resultname,
            dtype="float64",
        )

    def beta(
        self: Self,
        asset: Union[tuple[str, ValueType], int],
        market: Union[tuple[str, ValueType], int],
        dlta_degr_freedms: int = 1,
    ) -> float:
        """
        Market Beta.

        Calculates Beta as Co-variance of asset & market divided by Variance
        of the market. https://www.investopedia.com/terms/b/beta.asp.

        Parameters
        ----------
        asset: Union[tuple[str, ValueType], int]
            The column of the asset
        market: Union[tuple[str, ValueType], int]
            The column of the market against which Beta is measured
        dlta_degr_freedms: int, default: 1
            Variance bias factor taking the value 0 or 1.

        Returns
        -------
        float
            Beta as Co-variance of x & y divided by Variance of x

        """
        if all(
            x_value == ValueType.RTRN
            for x_value in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            if isinstance(asset, tuple):
                y_value = self.tsdf.loc[:, asset]
            elif isinstance(asset, int):
                y_value = self.tsdf.iloc[:, asset]
            else:
                msg = "asset should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )
            if isinstance(market, tuple):
                x_value = self.tsdf.loc[:, market]
            elif isinstance(market, int):
                x_value = self.tsdf.iloc[:, market]
            else:
                msg = "market should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )
        else:
            if isinstance(asset, tuple):
                y_value = log(
                    self.tsdf.loc[:, asset] / self.tsdf.loc[:, asset].iloc[0],
                )
            elif isinstance(asset, int):
                y_value = log(
                    self.tsdf.iloc[:, asset] / cast(float, self.tsdf.iloc[0, asset]),
                )
            else:
                msg = "asset should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )
            if isinstance(market, tuple):
                x_value = log(
                    self.tsdf.loc[:, market] / self.tsdf.loc[:, market].iloc[0],
                )
            elif isinstance(market, int):
                x_value = log(
                    self.tsdf.iloc[:, market] / cast(float, self.tsdf.iloc[0, market]),
                )
            else:
                msg = "market should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )

        covariance = cov(y_value, x_value, ddof=dlta_degr_freedms)
        beta = covariance[0, 1] / covariance[1, 1]

        return float(beta)

    def ord_least_squares_fit(
        self: Self,
        y_column: Union[tuple[str, ValueType], int],
        x_column: Union[tuple[str, ValueType], int],
        method: LiteralOlsFitMethod = "pinv",
        cov_type: LiteralOlsFitCovType = "nonrobust",
        *,
        fitted_series: bool = True,
    ) -> OLSResults:
        """
        Ordinary Least Squares fit.

        Performs a linear regression and adds a new column with a fitted line
        using Ordinary Least Squares fit
        https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html.

        Parameters
        ----------
        y_column: Union[tuple[str, ValueType], int]
            The column level values of the dependent variable y
        x_column: Union[tuple[str, ValueType], int]
            The column level values of the exogenous variable x
        method: LiteralOlsFitMethod, default: pinv
            Method to solve least squares problem
        cov_type: LiteralOlsFitCovType, default: nonrobust
            Covariance estimator
        fitted_series: bool, default: True
            If True the fit is added as a new column in the .tsdf Pandas.DataFrame

        Returns
        -------
        OLSResults
            The Statsmodels regression output

        """
        if isinstance(y_column, tuple):
            y_value = self.tsdf.loc[:, y_column]
            y_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, y_column].name,
            )[0]
        elif isinstance(y_column, int):
            y_value = self.tsdf.iloc[:, y_column]
            y_label = cast(tuple[str, str], self.tsdf.iloc[:, y_column].name)[0]
        else:
            msg = "y_column should be a tuple[str, ValueType] or an integer."
            raise TypeError(
                msg,
            )

        if isinstance(x_column, tuple):
            x_value = self.tsdf.loc[:, x_column]
            x_label = cast(
                tuple[str, str],
                self.tsdf.loc[:, x_column].name,
            )[0]
        elif isinstance(x_column, int):
            x_value = self.tsdf.iloc[:, x_column]
            x_label = cast(tuple[str, str], self.tsdf.iloc[:, x_column].name)[0]
        else:
            msg = "x_column should be a tuple[str, ValueType] or an integer."
            raise TypeError(
                msg,
            )

        results = sm.OLS(y_value, x_value).fit(method=method, cov_type=cov_type)
        if fitted_series:
            self.tsdf[y_label, x_label] = results.predict(x_value)

        return cast(OLSResults, results)

    def jensen_alpha(  # noqa: C901
        self: Self,
        asset: Union[tuple[str, ValueType], int],
        market: Union[tuple[str, ValueType], int],
        riskfree_rate: float = 0.0,
        dlta_degr_freedms: int = 1,
    ) -> float:
        """
        Jensen's alpha.

        The Jensen's measure, or Jensen's alpha, is a risk-adjusted performance
        measure that represents the average return on a portfolio or investment,
        above or below that predicted by the capital asset pricing model (CAPM),
        given the portfolio's or investment's beta and the average market return.
        This metric is also commonly referred to as simply alpha.
        https://www.investopedia.com/terms/j/jensensmeasure.asp.

        Parameters
        ----------
        asset: Union[tuple[str, ValueType], int]
            The column of the asset
        market: Union[tuple[str, ValueType], int]
            The column of the market against which Jensen's alpha is measured
        riskfree_rate : float, default: 0.0
            The return of the zero volatility riskfree asset
        dlta_degr_freedms: int, default: 1
            Variance bias factor taking the value 0 or 1.

        Returns
        -------
        float
            Jensen's alpha

        """
        full_year = 1.0
        if all(
            x == ValueType.RTRN
            for x in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            if isinstance(asset, tuple):
                asset_log = self.tsdf.loc[:, asset]
                asset_cagr = asset_log.mean()
            elif isinstance(asset, int):
                asset_log = self.tsdf.iloc[:, asset]
                asset_cagr = asset_log.mean()
            else:
                msg = "asset should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )
            if isinstance(market, tuple):
                market_log = self.tsdf.loc[:, market]
                market_cagr = market_log.mean()
            elif isinstance(market, int):
                market_log = self.tsdf.iloc[:, market]
                market_cagr = market_log.mean()
            else:
                msg = "market should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )
        else:
            if isinstance(asset, tuple):
                asset_log = log(
                    self.tsdf.loc[:, asset] / self.tsdf.loc[:, asset].iloc[0],
                )
                if self.yearfrac > full_year:
                    asset_cagr = (
                        self.tsdf.loc[:, asset].iloc[-1]
                        / self.tsdf.loc[:, asset].iloc[0]
                    ) ** (1 / self.yearfrac) - 1
                else:
                    asset_cagr = (
                        self.tsdf.loc[:, asset].iloc[-1]
                        / self.tsdf.loc[:, asset].iloc[0]
                        - 1
                    )
            elif isinstance(asset, int):
                asset_log = log(
                    self.tsdf.iloc[:, asset] / cast(float, self.tsdf.iloc[0, asset]),
                )
                if self.yearfrac > full_year:
                    asset_cagr = (
                        cast(float, self.tsdf.iloc[-1, asset])
                        / cast(float, self.tsdf.iloc[0, asset])
                    ) ** (1 / self.yearfrac) - 1
                else:
                    asset_cagr = (
                        cast(float, self.tsdf.iloc[-1, asset])
                        / cast(float, self.tsdf.iloc[0, asset])
                        - 1
                    )
            else:
                msg = "asset should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )
            if isinstance(market, tuple):
                market_log = log(
                    self.tsdf.loc[:, market] / self.tsdf.loc[:, market].iloc[0],
                )
                if self.yearfrac > full_year:
                    market_cagr = (
                        self.tsdf.loc[:, market].iloc[-1]
                        / self.tsdf.loc[:, market].iloc[0]
                    ) ** (1 / self.yearfrac) - 1
                else:
                    market_cagr = (
                        self.tsdf.loc[:, market].iloc[-1]
                        / self.tsdf.loc[:, market].iloc[0]
                        - 1
                    )
            elif isinstance(market, int):
                market_log = log(
                    self.tsdf.iloc[:, market] / cast(float, self.tsdf.iloc[0, market]),
                )
                if self.yearfrac > full_year:
                    market_cagr = (
                        cast(float, self.tsdf.iloc[-1, market])
                        / cast(float, self.tsdf.iloc[0, market])
                    ) ** (1 / self.yearfrac) - 1
                else:
                    market_cagr = (
                        cast(float, self.tsdf.iloc[-1, market])
                        / cast(float, self.tsdf.iloc[0, market])
                        - 1
                    )
            else:
                msg = "market should be a tuple[str, ValueType] or an integer."
                raise TypeError(
                    msg,
                )

        covariance = cov(asset_log, market_log, ddof=dlta_degr_freedms)
        beta = covariance[0, 1] / covariance[1, 1]

        return float(asset_cagr - riskfree_rate - beta * (market_cagr - riskfree_rate))

    def make_portfolio(
        self: Self,
        name: str,
        weight_strat: Optional[LiteralPortfolioWeightings] = None,
    ) -> DataFrame:
        """
        Calculate a basket timeseries based on the supplied weights.

        Parameters
        ----------
        name: str
            Name of the basket timeseries
        weight_strat: LiteralPortfolioWeightings, optional
            weight calculation strategies

        Returns
        -------
        Pandas.DataFrame
            A basket timeseries

        """
        if self.weights is None and weight_strat is None:
            msg = (
                "OpenFrame weights property must be provided "
                "to run the make_portfolio method."
            )
            raise ValueError(
                msg,
            )
        dframe = self.tsdf.copy()
        if not any(
            x == ValueType.RTRN
            for x in self.tsdf.columns.get_level_values(1).to_numpy()
        ):
            dframe = dframe.pct_change(fill_method=cast(str, None))
            dframe.iloc[0] = 0
        if weight_strat:
            if weight_strat == "eq_weights":
                self.weights = [1.0 / self.item_count] * self.item_count
            elif weight_strat == "inv_vol":
                vol = divide(1.0, std(dframe, axis=0, ddof=1))
                vol[isinf(vol)] = nan
                self.weights = list(divide(vol, vol.sum()))
            else:
                msg = "Weight strategy not implemented"
                raise NotImplementedError(msg)
        return DataFrame(
            data=dframe.dot(other=array(self.weights)).add(1.0).cumprod(),
            index=self.tsdf.index,
            columns=[[name], [ValueType.PRICE]],
            dtype="float64",
        )

    def rolling_info_ratio(
        self: Self,
        long_column: int = 0,
        short_column: int = 1,
        observations: int = 21,
        periods_in_a_year_fixed: Optional[DaysInYearType] = None,
    ) -> DataFrame:
        """
        Calculate rolling Information Ratio.

        The Information Ratio equals ( fund return less index return ) divided by
        the Tracking Error. And the Tracking Error is the standard deviation of the
        difference between the fund and its index returns.

        Parameters
        ----------
        long_column: int, default: 0
            Column of timeseries that is the numerator in the ratio.
        short_column: int, default: 1
            Column of timeseries that is the denominator in the ratio.
        observations: int, default: 21
            The length of the rolling window to use is set as number of observations.
        periods_in_a_year_fixed : DaysInYearType, optional
            Allows locking the periods-in-a-year to simplify test cases and comparisons

        Returns
        -------
        Pandas.DataFrame
            Rolling Information Ratios

        """
        long_label = cast(
            tuple[str, str],
            self.tsdf.iloc[:, long_column].name,
        )[0]
        short_label = cast(
            tuple[str, str],
            self.tsdf.iloc[:, short_column].name,
        )[0]
        ratio_label = f"{long_label} / {short_label}"
        if periods_in_a_year_fixed:
            time_factor = float(periods_in_a_year_fixed)
        else:
            time_factor = self.periods_in_a_year

        relative = (
            1.0 + self.tsdf.iloc[:, long_column] - self.tsdf.iloc[:, short_column]
        )

        retseries = (
            relative.pct_change(fill_method=cast(str, None))
            .rolling(observations, min_periods=observations)
            .sum()
        )
        retdf = retseries.dropna().to_frame()

        voldf = relative.pct_change(fill_method=cast(str, None)).rolling(
            observations,
            min_periods=observations,
        ).std() * sqrt(time_factor)
        voldf = voldf.dropna().to_frame()

        ratiodf = (retdf.iloc[:, 0] / voldf.iloc[:, 0]).to_frame()
        ratiodf.columns = [[ratio_label], ["Information Ratio"]]

        return DataFrame(ratiodf)

    def rolling_beta(
        self: Self,
        asset_column: int = 0,
        market_column: int = 1,
        observations: int = 21,
        dlta_degr_freedms: int = 1,
    ) -> DataFrame:
        """
        Calculate rolling Market Beta.

        Calculates Beta as Co-variance of asset & market divided by Variance
        of the market. https://www.investopedia.com/terms/b/beta.asp.

        Parameters
        ----------
        asset_column: int, default: 0
            Column of timeseries that is the asset.
        market_column: int, default: 1
            Column of timeseries that is the market.
        observations: int, default: 21
            The length of the rolling window to use is set as number of observations.
        dlta_degr_freedms: int, default: 1
            Variance bias factor taking the value 0 or 1.

        Returns
        -------
        Pandas.DataFrame
            Rolling Betas

        """
        market_label = cast(tuple[str, str], self.tsdf.iloc[:, market_column].name)[0]
        asset_label = cast(tuple[str, str], self.tsdf.iloc[:, asset_column].name)[0]
        beta_label = f"{asset_label} / {market_label}"

        rolling: DataFrame = self.tsdf.copy()
        rolling = rolling.pct_change(fill_method=cast(str, None)).rolling(
            observations,
            min_periods=observations,
        )

        rcov = rolling.cov(ddof=dlta_degr_freedms)
        rcov = rcov.dropna()

        rollbetaseries = rcov.iloc[:, asset_column].xs(
            market_label,
            level=1,
        ) / rcov.iloc[
            :,
            market_column,
        ].xs(
            market_label,
            level=1,
        )
        rollbeta = rollbetaseries.to_frame()
        rollbeta.index = rollbeta.index.droplevel(level=1)
        rollbeta.columns = MultiIndex.from_arrays([[beta_label], ["Beta"]])

        return rollbeta

    def rolling_corr(
        self: Self,
        first_column: int = 0,
        second_column: int = 1,
        observations: int = 21,
    ) -> DataFrame:
        """
        Calculate rolling Correlation.

        Calculates correlation between two series. The period with
        at least the given number of observations is the first period calculated.

        Parameters
        ----------
        first_column: int, default: 0
            The position as integer of the first timeseries to compare
        second_column: int, default: 1
            The position as integer of the second timeseries to compare
        observations: int, default: 21
            The length of the rolling window to use is set as number of observations

        Returns
        -------
        Pandas.DataFrame
            Rolling Correlations

        """
        corr_label = (
            cast(tuple[str, str], self.tsdf.iloc[:, first_column].name)[0]
            + "_VS_"
            + cast(tuple[str, str], self.tsdf.iloc[:, second_column].name)[0]
        )
        first_series = (
            self.tsdf.iloc[:, first_column]
            .pct_change(fill_method=cast(str, None))[1:]
            .rolling(observations, min_periods=observations)
        )
        second_series = self.tsdf.iloc[:, second_column].pct_change(
            fill_method=cast(str, None),
        )[1:]
        corrdf = first_series.corr(other=second_series).dropna().to_frame()
        corrdf.columns = MultiIndex.from_arrays(
            [
                [corr_label],
                ["Rolling correlation"],
            ],
        )

        return DataFrame(corrdf)


def simulate_portfolios(
    simframe: OpenFrame,
    num_ports: int,
    seed: int,
) -> DataFrame:
    """
    Generate random weights for simulated portfolios.

    Parameters
    ----------
    simframe: OpenFrame
        Return data for portfolio constituents
    num_ports: int
        Number of possible portfolios to simulate
    seed: int
        The seed for the random process

    Returns
    -------
    pandas.DataFrame
        The resulting data

    """
    copi = simframe.from_deepcopy()

    if any(
        x == ValueType.PRICE for x in copi.tsdf.columns.get_level_values(1).to_numpy()
    ):
        copi.value_to_ret()
        log_ret = copi.tsdf.copy()[1:]
    else:
        log_ret = copi.tsdf.copy()

    log_ret.columns = log_ret.columns.droplevel(level=1)

    randomizer = _random_generator(seed=seed)

    all_weights = zeros((num_ports, simframe.item_count))
    ret_arr = zeros(num_ports)
    vol_arr = zeros(num_ports)
    sharpe_arr = zeros(num_ports)

    for x in range(num_ports):
        weights = array(randomizer.random(simframe.item_count))
        weights = weights / npsum(weights)
        all_weights[x, :] = weights

        vol_arr[x] = sqrt(
            dot(
                weights.T,
                dot(log_ret.cov() * simframe.periods_in_a_year, weights),
            ),
        )

        ret_arr[x] = npsum(log_ret.mean() * weights * simframe.periods_in_a_year)

        sharpe_arr[x] = ret_arr[x] / vol_arr[x]

    # noinspection PyUnreachableCode
    simdf = concat(
        [
            DataFrame({"stdev": vol_arr, "ret": ret_arr, "sharpe": sharpe_arr}),
            DataFrame(all_weights, columns=simframe.columns_lvl_zero),
        ],
        axis="columns",
    )
    simdf = simdf.replace([inf, -inf], nan)
    return simdf.dropna()


# noinspection PyUnusedLocal
def efficient_frontier(  # noqa: C901
    eframe: OpenFrame,
    num_ports: int = 5000,
    seed: int = 71,
    upperbounds: float = 1.0,
    frontier_points: int = 200,
    *,
    tweak: bool = True,
) -> tuple[DataFrame, DataFrame, NDArray[float64]]:
    """
    Identify an efficient frontier.

    Parameters
    ----------
    eframe: OpenFrame
        Portfolio data
    num_ports: int, default: 5000
        Number of possible portfolios to simulate
    seed: int, default: 71
        The seed for the random process
    upperbounds: float, default: 1.0
        The largest allowed allocation to a single asset
    frontier_points: int, default: 200
        number of points along frontier to optimize
    tweak: bool, default: True
        cutting the frontier to exclude multiple points with almost the same risk

    Returns
    -------
    tuple[DataFrame, DataFrame, NDArray[float]]
        The efficient frontier data, simulation data and optimal portfolio

    """
    if eframe.weights is None:
        eframe.weights = [1.0 / eframe.item_count] * eframe.item_count

    copi = eframe.from_deepcopy()

    if any(
        x == ValueType.PRICE for x in copi.tsdf.columns.get_level_values(1).to_numpy()
    ):
        copi.value_to_ret()
        log_ret = copi.tsdf.copy()[1:]
    else:
        log_ret = copi.tsdf.copy()

    log_ret.columns = log_ret.columns.droplevel(level=1)

    simulated = simulate_portfolios(simframe=copi, num_ports=num_ports, seed=seed)

    frontier_min = simulated.loc[simulated["stdev"].idxmin()]["ret"]
    arithmetic_mean = log_ret.mean() * copi.periods_in_a_year
    frontier_max = 0.0
    if isinstance(arithmetic_mean, Series):
        frontier_max = arithmetic_mean.max()

    def _check_sum(weights: NDArray[float64]) -> float64:
        return cast(float64, npsum(weights) - 1)

    def _get_ret_vol_sr(
        lg_ret: DataFrame,
        weights: NDArray[float64],
        per_in_yr: float,
    ) -> NDArray[float64]:
        ret = npsum(lg_ret.mean() * weights) * per_in_yr
        volatility = sqrt(dot(weights.T, dot(lg_ret.cov() * per_in_yr, weights)))
        sr = ret / volatility
        return cast(NDArray[float64], array([ret, volatility, sr]))

    def _diff_return(
        lg_ret: DataFrame,
        weights: NDArray[float64],
        per_in_yr: float,
        poss_return: float,
    ) -> float64:
        return cast(
            float64,
            _get_ret_vol_sr(lg_ret=lg_ret, weights=weights, per_in_yr=per_in_yr)[0]
            - poss_return,
        )

    def _neg_sharpe(weights: NDArray[float64]) -> float64:
        return cast(
            float64,
            _get_ret_vol_sr(
                lg_ret=log_ret,
                weights=weights,
                per_in_yr=eframe.periods_in_a_year,
            )[2]
            * -1,
        )

    def _minimize_volatility(
        weights: NDArray[float64],
    ) -> float64:
        return cast(
            float64,
            _get_ret_vol_sr(
                lg_ret=log_ret,
                weights=weights,
                per_in_yr=eframe.periods_in_a_year,
            )[1],
        )

    constraints = {"type": "eq", "fun": _check_sum}
    bounds = tuple((0, upperbounds) for _ in range(eframe.item_count))
    init_guess = array(eframe.weights)

    opt_results = minimize(
        fun=_neg_sharpe,
        x0=init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimal = _get_ret_vol_sr(
        lg_ret=log_ret,
        weights=opt_results.x,
        per_in_yr=eframe.periods_in_a_year,
    )

    frontier_y = linspace(start=frontier_min, stop=frontier_max, num=frontier_points)
    frontier_x = []
    frontier_weights = []

    for possible_return in frontier_y:
        cons = cast(
            dict[str, Union[str, Callable[[float, NDArray[float64]], float64]]],
            (
                {"type": "eq", "fun": _check_sum},
                {
                    "type": "eq",
                    "fun": lambda w, poss_return=possible_return: _diff_return(
                        lg_ret=log_ret,
                        weights=w,
                        per_in_yr=eframe.periods_in_a_year,
                        poss_return=poss_return,
                    ),
                },
            ),
        )

        result = minimize(
            fun=_minimize_volatility,
            x0=init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        frontier_x.append(result["fun"])
        frontier_weights.append(result["x"])

    # noinspection PyUnreachableCode
    line_df = concat(
        [
            DataFrame(data=frontier_weights, columns=eframe.columns_lvl_zero),
            DataFrame({"stdev": frontier_x, "ret": frontier_y}),
        ],
        axis="columns",
    )
    line_df["sharpe"] = line_df.ret / line_df.stdev

    limit_small = 0.0001
    line_df = line_df.mask(line_df.abs() < limit_small, 0.0)
    line_df["text"] = line_df.apply(
        lambda c: "<br><br>Weights:<br>"
        + "<br>".join(
            [f"{c[nm]:.1%}  {nm}" for nm in eframe.columns_lvl_zero],
        ),
        axis="columns",
    )

    if tweak:
        limit_tweak = 0.001
        line_df["stdev_diff"] = line_df.stdev.pct_change()
        line_df = line_df.loc[line_df.stdev_diff.abs() > limit_tweak]
        line_df = line_df.drop(columns="stdev_diff")

    return line_df, simulated, append(optimal, opt_results.x)


def constrain_optimized_portfolios(
    data: OpenFrame,
    serie: OpenTimeSeries,
    portfolioname: str = "Current Portfolio",
    simulations: int = 10000,
    curve_points: int = 200,
    upper_bound: float = 0.25,
) -> tuple[OpenFrame, OpenTimeSeries, OpenFrame, OpenTimeSeries]:
    """
    Constrain optimized portfolios to those that improve on the current one.

    Parameters
    ----------
    data: OpenFrame
        Portfolio data
    serie: OpenTimeSeries
        A
    portfolioname: str, default: "Current Portfolio"
        Name of the portfolio
    simulations: int, default: 10000
        Number of possible portfolios to simulate
    curve_points: int, default: 200
        Number of optimal portfolios on the efficient frontier
    upper_bound: float, default: 0.25
        The largest allowed allocation to a single asset

    Returns
    -------
    tuple[OpenFrame, OpenTimeSeries, OpenFrame, OpenTimeSeries]
        The constrained optimal portfolio data

    """
    lr_frame = data.from_deepcopy()
    mv_frame = data.from_deepcopy()

    front_frame, sim_frame, optimal = efficient_frontier(
        eframe=data,
        num_ports=simulations,
        frontier_points=curve_points,
        upperbounds=upper_bound,
    )

    condition_least_ret = front_frame.ret > serie.arithmetic_ret
    # noinspection PyArgumentList
    least_ret_frame = front_frame[condition_least_ret].sort_values(by="stdev")
    least_ret_port = least_ret_frame.iloc[0]
    least_ret_port_name = f"Minimize vol & target return of {portfolioname}"
    least_ret_weights = [least_ret_port[c] for c in lr_frame.columns_lvl_zero]
    lr_frame.weights = least_ret_weights
    resleast = OpenTimeSeries.from_df(lr_frame.make_portfolio(least_ret_port_name))

    condition_most_vol = front_frame.stdev < serie.vol
    # noinspection PyArgumentList
    most_vol_frame = front_frame[condition_most_vol].sort_values(
        by="ret",
        ascending=False,
    )
    most_vol_port = most_vol_frame.iloc[0]
    most_vol_port_name = f"Maximize return & target risk of {portfolioname}"
    most_vol_weights = [most_vol_port[c] for c in mv_frame.columns_lvl_zero]
    mv_frame.weights = most_vol_weights
    resmost = OpenTimeSeries.from_df(mv_frame.make_portfolio(most_vol_port_name))

    return lr_frame, resleast, mv_frame, resmost


def prepare_plot_data(
    assets: OpenFrame,
    current: OpenTimeSeries,
    optimized: NDArray[float64],
) -> DataFrame:
    """
    Prepare date to be used as point_frame in the sharpeplot function.

    Parameters
    ----------
    assets: OpenFrame
        Portfolio data with individual assets and a weighted portfolio
    current: OpenTimeSeries
        The current or initial portfolio based on given weights
    optimized: DataFrame
        Data optimized with the efficient_frontier method

    Returns
    -------
    DataFrame
        The data prepared with mean returns, volatility and weights

    """
    txt = "<br><br>Weights:<br>" + "<br>".join(
        [
            f"{wgt:.1%}  {nm}"
            for wgt, nm in zip(
                cast(list[float], assets.weights),
                assets.columns_lvl_zero,
            )
        ],
    )

    opt_text_list = [
        f"{wgt:.1%}  {nm}" for wgt, nm in zip(optimized[3:], assets.columns_lvl_zero)
    ]
    opt_text = "<br><br>Weights:<br>" + "<br>".join(opt_text_list)
    vol: Series[float] = assets.vol
    plotframe = DataFrame(
        data=[
            assets.arithmetic_ret,
            vol,
            Series(
                data=[""] * assets.item_count,
                index=vol.index,
            ),
        ],
        index=["ret", "stdev", "text"],
    )
    plotframe.columns = plotframe.columns.droplevel(level=1)
    plotframe["Max Sharpe Portfolio"] = [optimized[0], optimized[1], opt_text]
    plotframe[current.label] = [current.arithmetic_ret, current.vol, txt]

    return plotframe


def sharpeplot(  # noqa: C901
    sim_frame: DataFrame = None,
    line_frame: DataFrame = None,
    point_frame: DataFrame = None,
    point_frame_mode: LiteralLinePlotMode = "markers",
    filename: Optional[str] = None,
    directory: Optional[DirectoryPath] = None,
    titletext: Optional[str] = None,
    output_type: LiteralPlotlyOutput = "file",
    include_plotlyjs: LiteralPlotlyJSlib = "cdn",
    *,
    title: bool = True,
    add_logo: bool = True,
    auto_open: bool = True,
) -> tuple[Figure, str]:
    """
    Create scatter plot coloured by Sharpe Ratio.

    Parameters
    ----------
    sim_frame: DataFrame, optional
        Data from the simulate_portfolios method.
    line_frame: DataFrame, optional
        Data from the efficient_frontier method.
    point_frame: DataFrame, optional
        Data to highlight current and efficient portfolios.
    point_frame_mode: LiteralLinePlotMode, default: markers
        Which type of scatter to use.
    filename: str, optional
        Name of the Plotly html file
    directory: DirectoryPath, optional
        Directory where Plotly html file is saved
    titletext: str, optional
        Text for the plot title
    output_type: LiteralPlotlyOutput, default: "file"
        Determines output type
    include_plotlyjs: LiteralPlotlyJSlib, default: "cdn"
        Determines how the plotly.js library is included in the output
    title: bool, default: True
        Whether to add standard plot title
    add_logo: bool, default: True
        Whether to add Captor logo
    auto_open: bool, default: True
        Determines whether to open a browser window with the plot

    Returns
    -------
    Figure
        The scatter plot with simulated and optimized results

    """
    returns = []
    risk = []

    if directory:
        dirpath = Path(directory).resolve()
    elif Path.home().joinpath("Documents").exists():
        dirpath = Path.home().joinpath("Documents")
    else:
        dirpath = Path(stack()[1].filename).parent

    if not filename:
        filename = "sharpeplot.html"
    plotfile = dirpath.joinpath(filename)

    fig, logo = load_plotly_dict()
    figure = Figure(fig)

    if sim_frame is not None:
        returns.extend(list(sim_frame.loc[:, "ret"]))
        risk.extend(list(sim_frame.loc[:, "stdev"]))
        figure.add_scatter(
            x=sim_frame.loc[:, "stdev"],
            y=sim_frame.loc[:, "ret"],
            hoverinfo="skip",
            marker={
                "size": 10,
                "opacity": 0.5,
                "color": sim_frame.loc[:, "sharpe"],
                "colorscale": "Jet",
                "reversescale": True,
                "colorbar": {"thickness": 20, "title": "Ratio<br>ret / vol"},
            },
            mode="markers",
            name="simulated portfolios",
        )
    if line_frame is not None:
        returns.extend(list(line_frame.loc[:, "ret"]))
        risk.extend(list(line_frame.loc[:, "stdev"]))
        figure.add_scatter(
            x=line_frame.loc[:, "stdev"],
            y=line_frame.loc[:, "ret"],
            text=line_frame.loc[:, "text"],
            xhoverformat=".2%",
            yhoverformat=".2%",
            hovertemplate="Return %{y}<br>Vol %{x}%{text}",
            hoverlabel_align="right",
            line={"width": 2.5, "dash": "solid"},
            mode="lines",
            name="Efficient frontier",
        )

    colorway = cast(dict[str, list[str]], fig["layout"]).get("colorway")[
        : len(point_frame.columns)
    ]

    if point_frame is not None:
        for col, clr in zip(point_frame.columns, colorway):
            returns.extend([point_frame.loc["ret", col]])
            risk.extend([point_frame.loc["stdev", col]])
            figure.add_scatter(
                x=[point_frame.loc["stdev", col]],
                y=[point_frame.loc["ret", col]],
                xhoverformat=".2%",
                yhoverformat=".2%",
                hovertext=[point_frame.loc["text", col]],
                hovertemplate="Return %{y}<br>Vol %{x}%{hovertext}",
                hoverlabel_align="right",
                marker={"size": 20, "color": clr},
                mode=point_frame_mode,
                name=col,
                text=col,
                textfont={"size": 14},
                textposition="bottom center",
            )

    figure.update_layout(
        xaxis={"tickformat": ".1%"},
        xaxis_title="volatility",
        yaxis={
            "tickformat": ".1%",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        yaxis_title="annual return",
        showlegend=False,
    )
    if title:
        if titletext is None:
            titletext = "<b>Risk and Return</b><br>"
        figure.update_layout(title={"text": titletext, "font": {"size": 32}})

    if add_logo:
        figure.add_layout_image(logo)

    if output_type == "file":
        plot(
            figure_or_data=figure,
            filename=str(plotfile),
            auto_open=auto_open,
            auto_play=False,
            link_text="",
            include_plotlyjs=cast(bool, include_plotlyjs),
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
            include_plotlyjs=cast(bool, include_plotlyjs),
            full_html=False,
            div_id=div_id,
        )

    return figure, string_output
