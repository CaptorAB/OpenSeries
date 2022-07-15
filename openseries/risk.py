# -*- coding: utf-8 -*-
"""
Source:
https://github.com/pmorissette/ffn/blob/master/ffn/core.py
"""
import datetime as dt
import math
from typing import List

import numpy as np
import pandas as pd


def cvar_down(
    data: pd.DataFrame | pd.Series | List[float], level: float = 0.95
) -> float:
    """https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

    Parameters
    ----------
    data: pd.DataFrame | List[float]
        The data to perform the calculation over
    level: float, default: 0.95
        The sought CVaR level

    Returns
    -------
    float
        Downside Conditional Value At Risk "CVaR"
    """

    if isinstance(data, pd.DataFrame):
        clean = np.nan_to_num(data.iloc[:, 0])
    else:
        clean = np.nan_to_num(data)
    ret = clean[1:] / clean[:-1] - 1
    array = np.sort(ret)
    return float(np.mean(array[: int(math.ceil(len(array) * (1 - level)))]))


def var_down(
    data: pd.DataFrame | pd.Series | List[float],
    level: float = 0.95,
    interpolation: str = "lower",
) -> float:
    """Downside Value At Risk, "VaR". The equivalent of
    percentile.inc([...], 1-level) over returns in MS Excel \n
    https://www.investopedia.com/terms/v/var.asp

    Parameters
    ----------
    data: pd.DataFrame | List[float]
        The data to perform the calculation over
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

    if isinstance(data, pd.DataFrame):
        clean = np.nan_to_num(data.iloc[:, 0])
    else:
        clean = np.nan_to_num(data)
    ret = clean[1:] / clean[:-1] - 1
    result = np.quantile(ret, 1 - level, method=interpolation)
    return result


def drawdown_series(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calculates https://www.investopedia.com/terms/d/drawdown.asp
    This returns a series representing a drawdown.
    When the price is at all-time highs, the drawdown
    is 0. However, when prices are below high watermarks,
    the drawdown series = current / hwm - 1
    The max drawdown can be obtained by simply calling .min()
    on the result (since the drawdown series is negative)
    Method ignores all gaps of NaN's in the price series.
    :param prices: (Series or DataFrame) Series of prices.
    """
    # make a copy so that we don't modify original data
    drawdown = prices.copy()

    # Fill NaN's with previous values
    drawdown = drawdown.fillna(method="ffill")

    # Ignore problems with NaN's in the beginning
    drawdown[np.isnan(drawdown)] = -np.Inf

    # Rolling maximum
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.0
    return drawdown


def calc_max_drawdown(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calculates the max drawdown of a price series. If you want the
    actual drawdown series, please use to_drawdown_series.
    :param prices: (Series or DataFrame) Series of prices.
    """
    return (prices / prices.expanding(min_periods=1).max()).min() - 1


def max_drawdown_date(prices: pd.DataFrame | pd.Series) -> dt.date:
    """
    Date when Max drawdown occurred.
    """
    mdd_date = (
        (prices / prices.expanding(min_periods=1).max())
        .idxmin()
        .values[0]
        .astype(dt.datetime)
    )
    return dt.datetime.fromtimestamp(mdd_date / 1e9).date()


def drawdown_details(prices: pd.DataFrame | pd.Series) -> pd.Series:
    """

    :param prices:
    """
    mdate = max_drawdown_date(prices)
    md = float(calc_max_drawdown(prices))
    dd = prices.copy()
    drwdwn = drawdown_series(dd).loc[:mdate]
    drwdwn.sort_index(ascending=False, inplace=True)
    sdate = drwdwn[drwdwn == 0.0].idxmax().values[0].astype(dt.datetime)
    sdate = dt.datetime.fromtimestamp(sdate / 1e9).date()
    duration = (mdate - sdate).days
    ret_per_day = md / duration
    df = pd.Series(
        data=[md, sdate, mdate, duration, ret_per_day],
        index=[
            "Max Drawdown",
            "Start of drawdown",
            "Date of bottom",
            "Days from start to bottom",
            "Average fall per day",
        ],
        name="Drawdown details",
    )
    return df
