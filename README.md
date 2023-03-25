<img src="https://sales.captor.se/captor_logo_sv_1600_icketransparent.png" alt="Captor
Fund Management AB"
width="81" height="100" align="left" float="right"/><br/>

<br><br>

# OpenSeries

[![PyPI version](https://img.shields.io/pypi/v/openseries.svg)](https://pypi.org/project/openseries/)
[![Python version](https://img.shields.io/pypi/pyversions/openseries.svg)](https://docs.python.org/3.10/)
[![GitHub Action Test Suite](https://github.com/CaptorAB/OpenSeries/actions/workflows/TestsWithSplit.yml/badge.svg)](https://github.com/CaptorAB/OpenSeries/actions/workflows/TestsWithSplit.yml)
[![Coverage](https://cdn.jsdelivr.net/gh/CaptorAB/OpenSeries@master/coverage.svg)](https://github.com/CaptorAB/OpenSeries/actions/workflows/TestsWithSplit.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**OpenSeries** is a project with tools to perform timeseries analysis on a single
asset or a group of assets. It is solely made for daily or less frequent data.

<span style="font-size:2em;">[Changelog](https://github.com/CaptorAB/OpenSeries/blob/master/CHANGELOG.md)</span>

To install:

```
pip install openseries
```

An overview of an OpenTimeSeries object is shown in the below example. Most of the
time an object will be constructed from the constructing class methods. The
OpenTimeSeries and OpenFrame classes are both subclasses of
the [Pydantic BaseModel](https://docs.pydantic.dev/usage/models/).

```
from pandas import DataFrame, DatetimeIndex
from openseries.series import OpenTimeSeries, ValueType
```

```
series = OpenTimeSeries(
    timeseriesId="",
    instrumentId="",
    currency="SEK",
    dates=["2020-09-03", "2020-09-04", "2020-09-07", "2020-09-08", "2020-09-09"],
    name="Timeseries",
    label="Timeseries",
    valuetype=ValueType.PRICE,
    values=[114.9965, 114.8355, 114.8694, 115.1131, 114.8643],
    local_ccy=True,
    tsdf=DataFrame(
        data=[114.9965, 114.8355, 114.8694, 115.1131, 114.8643],
        index=[
            d.date()
            for d in DatetimeIndex(
                [
                    "2020-09-03",
                    "2020-09-04",
                    "2020-09-07",
                    "2020-09-08",
                    "2020-09-09",
                ]
            )
        ],
        columns=[["Timeseries"], [ValueType.PRICE]],
        dtype="float64",
    ),
)
```

To make use of some tools available in the [Pandas](https://pandas.pydata.org/) library
the [OpenTimeSeries](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py)
and [OpenFrame](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/frame.py)
classes have an attribute `tsdf`
which is a DataFrame constructed from the raw data in the lists `dates` and `values`.

## Table of Contents

- [Files / Modules in the project](#files-in-the-project)
- [Class methods used to construct an OpenTimeSeries](#class-methods-used-to-construct-an-opentimeseries-object)
- [OpenTimeSeries non-numeric properties](#non-numeric-or--helper--properties-that-apply-only-to-the-opentimeseries-class)
- [OpenFrame non-numeric properties](#non-numeric-or--helper--properties-that-apply-only-to-the-openframe-class)
- [Non-numeric properties for both classes](#non-numeric-or--helper--properties-that-apply-to-both-the-opentimeseries-and-the-openframe-class)
- [OpenTimeSeries only methods](#methods-that-apply-only-to-the-opentimeseries-class)
- [OpenFrame only methods](#methods-that-apply-only-to-the-openframe-class)
- [Methods for both classes](#methods-that-apply-to-both-the-opentimeseries-and-the-openframe-class)
- [Numeric properties for both classes](#numeric-properties-available-for-individual-opentimeseries-or-on-all-series-in-an-openframe-)
- [Numeric methods with period arguments for both classes](#methods-below-are-identical-to-the-numeric-properties-above)
- [Development Instructions](#development-instructions)

### Files in the project

| File                                                                                                             | Description                                                                                                                                                                                         |
|:-----------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [series.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py)                             | Defines the class _OpenTimeSeries_ for managing and analyzing a single timeseries. The module also defines a function `timeseries_chain` that can be used to chain two timeseries objects together. |
| [frame.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/frame.py)                               | Defines the class _OpenFrame_ for managing a group of timeseries, and e.g. calculate a portfolio timeseries from a rebalancing strategy between timeseries.                                         |
| [datefixer.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/datefixer.py)                       | Date utilities. Please read the docstring of each function for its description.                                                                                                                     |
| [plotly_layouts.json](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/plotly_layouts.json)         | [Plotly](https://plotly.com/python/) defaults used in the `plot_bars` and `plot_series` methods.                                                                                                    |
| [plotly_captor_logo.json](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/plotly_captor_logo.json) | JSON with a link to the Captor logo used in the `plot_bars` and `plot_series` methods.                                                                                                              |
| [risk.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/risk.py)                                 | Methods used to calculate VaR, CVaR and drawdowns.                                                                                                                                                  |
| [sim_price.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/sim_price.py)                       | Simulates OpenTimeSeries from different stochastic processes.                                                                                                                                       |
| [stoch_processes.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/stoch_processes.py)           | Generates stochastic processes used in the `sim_price.py` module.                                                                                                                                   |
| [types.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/types.py)                               | Contains all bespoke typing.                                                                                                                                                                        |
| [exceptions.py](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/exceptions.py)                     | Sub-classed exceptions to provide better user feedback on errors.                                                                                                                                   |

### Class methods used to construct an [OpenTimeSeries](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py) object.

| Method                 | Applies to                    | Description                                                                                              |
|:-----------------------|:------------------------------|:---------------------------------------------------------------------------------------------------------|
| `from_df`              | `OpenTimeSeries`              | Class method to create an OpenTimeSeries object from a pandas.DataFrame column.                          |
| `from_frame`           | `OpenTimeSeries`              | Class method to create a new OpenTimeSeries object from a series within an OpenFrame.                    |
| `from_fixed_rate`      | `OpenTimeSeries`              | Class method to create an OpenTimeSeries object from a fixed rate, number of days and an end date.       |
| `parse_obj`            | `OpenTimeSeries`              | A method inherited from the Pydantic BaseModel to construct an object from a `dict`.                     |
| `from_deepcopy`        | `OpenTimeSeries`, `OpenFrame` | Creates a copy of an OpenTimeSeries object.                                                              |

### Non-numeric or "helper" properties that apply only to the [OpenTimeSeries](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py) class.

| Property       | type            | Applies to       | Description                                                                                                                                  |
|:---------------|:----------------|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| `timeseriesId` | `str`           | `OpenTimeSeries` | Placeholder for database identifier for the timeseries. Can be left as empty string.                                                         |
| `instrumentId` | `str`           | `OpenTimeSeries` | Placeholder for database identifier for the instrument associated with the timeseries. Can be left as empty string.                          |
| `dates`        | `List[str]`     | `OpenTimeSeries` | Dates of the timeseries. Not edited by any method to allow reversion to original.                                                            |
| `values`       | `List[float]`   | `OpenTimeSeries` | Values of the timeseries. Not edited by any method to allow reversion to original.                                                           |
| `currency`     | `str`           | `OpenTimeSeries` | Currency of the timeseries. Only used if conversion/hedging methods are added.                                                               |
| `domestic`     | `str`           | `OpenTimeSeries` | Domestic currency of the user / investor. Only used if conversion/hedging methods are added.                                                 |
| `local_ccy`    | `bool`          | `OpenTimeSeries` | Indicates if series should be in its local currency or the domestic currency of the user. Only used if conversion/hedging methods are added. |
| `name`         | `str`           | `OpenTimeSeries` | An identifier field.                                                                                                                         |
| `isin`         | `str`           | `OpenTimeSeries` | ISIN code of the associated instrument. If any.                                                                                              |
| `label`        | `str`           | `OpenTimeSeries` | Field used in outputs. Derived from name as default.                                                                                         |
| `countries`    | `list` or `str` | `OpenTimeSeries` | (List of) country code(s) according to ISO 3166-1 alpha-2 used to generate business days.                                                    |
| `valuetype`    | `ValueType`     | `OpenTimeSeries` | Field identifies the type of values in the series. ValueType is an Enum.                                                                     |

### Non-numeric or "helper" properties that apply only to the [OpenFrame](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/frame.py) class.

| Property           | type                   | Applies to  | Description                                                              |
|:-------------------|:-----------------------|:------------|:-------------------------------------------------------------------------|
| `constituents`     | `List[OpenTimeSeries]` | `OpenFrame` | A list of the OpenTimeSeries that make up an OpenFrame.                  |
| `columns_lvl_zero` | `list`                 | `OpenFrame` | A list of the level zero column names in the OpenFrame pandas.DataFrame. |
| `columns_lvl_one`  | `list`                 | `OpenFrame` | A list of the level one column names in the OpenFrame pandas.DataFrame.  |
| `item_count`       | `int`                  | `OpenFrame` | Number of columns in the OpenFrame pandas.DataFrame.                     |
| `weights`          | `List[float]`          | `OpenFrame` | Weights used in the method `make_portfolio`.                             |
| `first_indices`    | `pandas.Series`        | `OpenFrame` | First dates of all the series in the OpenFrame.                          |
| `last_indices`     | `pandas.Series`        | `OpenFrame` | Last dates of all the series in the OpenFrame.                           |
| `lengths_of_items` | `pandas.Series`        | `OpenFrame` | Number of items in each of the series in the OpenFrame.                  |
| `span_of_days_all` | `pandas.Series`        | `OpenFrame` | Number of days from the first to the last in each of the series.         |

### Non-numeric or "helper" properties that apply to both the [OpenTimeSeries](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py) and the [OpenFrame](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/frame.py) class.

| Property            | type                             | Applies to                    | Description                                                                       |
|:--------------------|:---------------------------------|:------------------------------|:----------------------------------------------------------------------------------|
| `first_idx`         | `datetime.date`                  | `OpenTimeSeries`, `OpenFrame` | First date of the series.                                                         |
| `last_idx`          | `datetime.date`                  | `OpenTimeSeries`, `OpenFrame` | Last date of the series.                                                          |
| `length`            | `int`                            | `OpenTimeSeries`, `OpenFrame` | Number of items in the series.                                                    |
| `span_of_days`      | `int`                            | `OpenTimeSeries`, `OpenFrame` | Number of days from the first to the last date in the series.                     |
| `tsdf`              | `pandas.DataFrame`               | `OpenTimeSeries`, `OpenFrame` | The Pandas DataFrame which gets edited by the class methods.                      |
| `max_drawdown_date` | `datetime.date`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Date when the maximum drawdown occurred.                                          |
| `periods_in_a_year` | `float`                          | `OpenTimeSeries`, `OpenFrame` | The number of observations in an average year for all days in the data.           |
| `yearfrac`          | `float`                          | `OpenTimeSeries`, `OpenFrame` | Length of timeseries expressed as np.float64 fraction of a year with 365.25 days. |

### Methods that apply only to the [OpenTimeSeries](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py) class.

| Method                           | Applies to       | Description                                                                                                                                    |
|:---------------------------------|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| `setup_class`                    | `OpenTimeSeries` | Class method that defines the `domestic` home currency and the `countries` home countries attributes.                                          |
| `to_json`                        | `OpenTimeSeries` | Method to export the OpenTimeSeries `__dict__` to a json file.                                                                                 |
| `pandas_df`                      | `OpenTimeSeries` | Method to create the `tsdf` pandas.DataFrame from the `dates` and `values`.                                                                    |
| `set_new_label`                  | `OpenTimeSeries` | Method to change the pandas.DataFrame column MultiIndex.                                                                                       |
| `running_adjustment`             | `OpenTimeSeries` | Adjusts the series performance with a `float` factor.                                                                                          |
| `ewma_vol_func`                  | `OpenTimeSeries` | Returns a `pandas.Series` with volatility based on [Exponentially Weighted Moving Average](https://www.investopedia.com/articles/07/ewma.asp). |
| `from_1d_rate_to_cumret`         | `OpenTimeSeries` | Converts a series of 1-day rates into a cumulative valueseries.                                                                                |
| `dates_not_empty`                | `OpenTimeSeries` | Pydantic root_validator method to validate that dates are not empty.                                                                           |
| `values_not_empty`               | `OpenTimeSeries` | Pydantic root_validator method to validate that values are not emptys.                                                                         |
| `check_dates_values_same_length` | `OpenTimeSeries` | Pydantic root_validator method to validate that date and value arrays are the same length.                                                     |
| `check_values_match`             | `OpenTimeSeries` | Pydantic root_validator method to validate that the value array matches the values in the DataFrame when object is constructed.                |
| `check_isincode`                 | `OpenTimeSeries` | Pydantic root_validator method to validate the ISIN code if provided.                                                                          |

### Methods that apply only to the [OpenFrame](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/frame.py) class.

| Method                  | Applies to  | Description                                                                                                                                                       |
|:------------------------|:------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `merge_series`          | `OpenFrame` | Merges the Pandas Dataframes of the constituent OpenTimeSeries.                                                                                                   |
| `trunc_frame`           | `OpenFrame` | Truncates the OpenFrame to a common period.                                                                                                                       |
| `add_timeseries`        | `OpenFrame` | Adds a given OpenTimeSeries to the OpenFrame.                                                                                                                     |
| `delete_timeseries`     | `OpenFrame` | Deletes an OpenTimeSeries from the OpenFrame.                                                                                                                     |
| `relative`              | `OpenFrame` | Calculates a new series that is the relative performance of two others.                                                                                           |
| `make_portfolio`        | `OpenFrame` | Calculates a portfolio timeseries from series and weights.                                                                                                        |
| `ord_least_squares_fit` | `OpenFrame` | Performs a regression and an [Ordinary Least Squares](https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html) fit.                              |
| `beta`                  | `OpenFrame` | Calculates [Beta](https://www.investopedia.com/terms/b/beta.asp) of an asset relative a market.                                                                   |
| `jensen_alpha`          | `OpenFrame` | Calculates [Jensen's Alpha](https://www.investopedia.com/terms/j/jensensmeasure.asp) of an asset relative a market.                                               |
| `tracking_error_func`   | `OpenFrame` | Calculates the [tracking errors](https://www.investopedia.com/terms/t/trackingerror.asp) relative to a selected series in the OpenFrame.                          |
| `info_ratio_func`       | `OpenFrame` | Calculates the [information ratios](https://www.investopedia.com/terms/i/informationratio.asp) relative to a selected series in the OpenFrame.                    |
| `capture_ratio_func`    | `OpenFrame` | Calculates up, down and up/down [capture ratios](https://www.investopedia.com/terms/d/down-market-capture-ratio.asp) relative to a selected series.               |
| `rolling_info_ratio`    | `OpenFrame` | Returns a pandas.DataFrame with the rolling [information ratio](https://www.investopedia.com/terms/i/informationratio.asp) between two series.                    |
| `rolling_beta`          | `OpenFrame` | Returns a pandas.DataFrame with the rolling [Beta](https://www.investopedia.com/terms/b/beta.asp) of an asset relative a market.                                  |
| `rolling_corr`          | `OpenFrame` | Calculates and adds a series of rolling [correlations](https://www.investopedia.com/terms/c/correlation.asp) between two other series.                            |
| `ewma_risk`             | `OpenFrame` | Returns a `pandas.DataFrame` with volatility and correlation based on [Exponentially Weighted Moving Average](https://www.investopedia.com/articles/07/ewma.asp). |

### Methods that apply to both the [OpenTimeSeries](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py) and the [OpenFrame](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/frame.py) class.

| Method                             | Applies to                    | Description                                                                                                                                            |
|:-----------------------------------|:------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `align_index_to_local_cdays`       | `OpenTimeSeries`, `OpenFrame` | Aligns the series dates to a business calendar. Defaults to Sweden.                                                                                    |
| `resample`                         | `OpenTimeSeries`, `OpenFrame` | Resamples the series to a specific frequency.                                                                                                          |
| `resample_to_business_period_ends` | `OpenTimeSeries`, `OpenFrame` | Resamples the series to month-end dates with monthly, quarterly or annual frequency.                                                                   |
| `value_nan_handle`                 | `OpenTimeSeries`, `OpenFrame` | Fills `Nan` in a value series with the preceding non-Nan value.                                                                                        |
| `return_nan_handle`                | `OpenTimeSeries`, `OpenFrame` | Replaces `Nan` in a return series with a 0.0 `float`.                                                                                                  |
| `to_cumret`                        | `OpenTimeSeries`, `OpenFrame` | Converts a return series into a value series and/or resets a value series to be rebased from 1.0.                                                      |
| `value_to_ret`                     | `OpenTimeSeries`, `OpenFrame` | Converts a value series into a percentage return series.                                                                                               |
| `value_to_diff`                    | `OpenTimeSeries`, `OpenFrame` | Converts a value series into a series of differences.                                                                                                  |
| `value_to_log`                     | `OpenTimeSeries`, `OpenFrame` | Converts a value series into a logarithmic return series.                                                                                              |
| `value_ret_calendar_period`        | `OpenTimeSeries`, `OpenFrame` | Returns the series simple return for a specific calendar period.                                                                                       |
| `plot_series`                      | `OpenTimeSeries`, `OpenFrame` | Opens a HTML [Plotly Scatter](https://plotly.com/python/line-and-scatter/) plot of the series in a browser window.                                     |
| `plot_bars`                        | `OpenTimeSeries`, `OpenFrame` | Opens a HTML [Plotly Bar](https://plotly.com/python/bar-charts/) plot of the series in a browser window.                                               |
| `drawdown_details`                 | `OpenTimeSeries`, `OpenFrame` | Returns detailed drawdown characteristics.                                                                                                             |
| `to_drawdown_series`               | `OpenTimeSeries`, `OpenFrame` | Converts the series into drawdown series.                                                                                                              |
| `rolling_return`                   | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling returns.                                                                                                       |
| `rolling_vol`                      | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling volatilities.                                                                                                  |
| `rolling_var_down`                 | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling VaR figures.                                                                                                   |
| `rolling_cvar_down`                | `OpenTimeSeries`, `OpenFrame` | Returns a pandas.DataFrame with rolling CVaR figures.                                                                                                  |
| `calc_range`                       | `OpenTimeSeries`, `OpenFrame` | Returns the start and end dates of a range from specific period definitions. Used by the below numeric methods and not meant to be used independently. |

### Numeric properties available for individual [OpenTimeSeries](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/series.py) or on all series in an [OpenFrame](https://github.com/CaptorAB/OpenSeries/blob/master/openseries/frame.py).

| Property                | type                     | Applies to                    | Description                                                                                                                                                                                                             |
|:------------------------|:-------------------------|:------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `all_properties`        | `pandas.DataFrame`       | `OpenTimeSeries`, `OpenFrame` | Returns most of the properties in one go.                                                                                                                                                                               |
| `arithmetic_ret`        | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Annualized arithmetic mean of returns](https://www.investopedia.com/terms/a/arithmeticmean.asp).                                                                                                                       |
| `geo_ret`               | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Compound Annual Growth Rate(CAGR)](https://www.investopedia.com/terms/c/cagr.asp), a specific implementation of geometric mean.                                                                                        |
| `value_ret`             | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Simple return from first to last observation.                                                                                                                                                                           |
| `vol`                   | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Annualized [volatility](https://www.investopedia.com/terms/v/volatility.asp). Pandas .std() is the equivalent of stdev.s([...]) in MS excel.                                                                            |
| `downside_deviation`    | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Downside deviation](https://www.investopedia.com/terms/d/downside-deviation.asp) is the volatility of all negative return observations. Minimum Accepted Return (MAR) set to zero.                                     |
| `ret_vol_ratio`         | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Ratio of arithmetic mean return and annualized volatility. It is the [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp) with the riskfree rate set to zero.                                           |
| `sortino_ratio`         | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp) is the arithmetic mean return divided by the downside deviation. This attribute assumes that the riskfree rate and the MAR are both zero.    |
| `var_down`              | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Value At Risk](https://www.investopedia.com/terms/v/var.asp), "VaR". The equivalent of percentile.inc([...], 1-level) over returns in MS Excel. For other confidence levels use the corresponding method. |
| `cvar_down`             | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Conditional Value At Risk](https://www.investopedia.com/terms/c/conditional_value_at_risk.asp), "CVaR". For other confidence levels use the corresponding method.                                         |
| `worst`                 | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Most negative percentage change of a single observation.                                                                                                                                                                |
| `worst_month`           | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Most negative month.                                                                                                                                                                                                    |
| `max_drawdown`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Maximum drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp).                                                                                                                                      |
| `max_drawdown_cal_year` | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Max drawdown in a single calendar year.                                                                                                                                                                                 |
| `positive_share`        | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The share of percentage changes that are positive.                                                                                                                                                                      |
| `vol_from_var`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Implied annualized volatility from the Downside VaR using the assumption that returns are normally distributed.                                                                                                         |
| `skew`                  | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Skew](https://www.investopedia.com/terms/s/skewness.asp) of the return distribution.                                                                                                                                   |
| `kurtosis`              | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Kurtosis](https://www.investopedia.com/terms/k/kurtosis.asp) of the return distribution.                                                                                                                               |
| `z_score`               | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Z-score](https://www.investopedia.com/terms/z/zscore.asp) as (last return - mean return) / standard deviation of returns.                                                                                              |
| `correl_matrix`         | `pandas.DataFrame`       | `OpenFrame`                   | A correlation matrix.                                                                                                                                                                                                   |

### Methods below are identical to the Numeric Properties above.

_They are simply methods that take different date or length inputs to return the
properties for subset periods._

| Method                    | type                     | Applies to                    | Description                                                                                                                                                                                                                                                    |
|:--------------------------|:-------------------------|:------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arithmetic_ret_func`     | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Annualized arithmetic mean of returns](https://www.investopedia.com/terms/a/arithmeticmean.asp).                                                                                                                                                              |
| `geo_ret_func`            | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Compound Annual Growth Rate(CAGR)](https://www.investopedia.com/terms/c/cagr.asp), a specific implementation of geometric mean.                                                                                                                               |
| `value_ret_func`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Simple return from first to last observation.                                                                                                                                                                                                                  |
| `vol_func`                | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Annualized [volatility](https://www.investopedia.com/terms/v/volatility.asp). Pandas .std() is the equivalent of stdev.s([...]) in MS excel.                                                                                                                   |
| `downside_deviation_func` | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Downside deviation](https://www.investopedia.com/terms/d/downside-deviation.asp) is the volatility of all negative return observations. MAR and riskfree rate can be set.                                                                                     |
| `ret_vol_ratio_func`      | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Ratio of arithmetic mean return and annualized volatility. It is the [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp) with the riskfree rate set to zero. A riskfree rate can be set as a float or a series chosen for the frame function. |
| `sortino_ratio_func`      | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp) is the arithmetic mean return divided by the downside deviation. A riskfree rate can be set as a float or a series chosen for the frame function. MAR is set to zero.               |
| `var_down_func`           | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Value At Risk](https://www.investopedia.com/terms/v/var.asp), "VaR". The equivalent of percentile.inc([...], 1-level) over returns in MS Excel. Default is 95% confidence level.                                                                 |
| `cvar_down_func`          | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Downside 95% [Conditional Value At Risk](https://www.investopedia.com/terms/c/conditional_value_at_risk.asp), "CVaR". Default is 95% confidence level.                                                                                                         |
| `worst_func`              | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Most negative percentage change for a given number of observations (default=1).                                                                                                                                                                                |
| `max_drawdown_func`       | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Maximum drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp).                                                                                                                                                                             |
| `positive_share_func`     | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | The share of percentage changes that are positive.                                                                                                                                                                                                             |
| `vol_from_var_func`       | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | Implied annualized volatility from the Downside VaR using the assumption that returns are normally distributed.                                                                                                                                                |
| `skew_func`               | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Skew](https://www.investopedia.com/terms/s/skewness.asp) of the return distribution.                                                                                                                                                                          |
| `kurtosis_func`           | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Kurtosis](https://www.investopedia.com/terms/k/kurtosis.asp) of the return distribution.                                                                                                                                                                      |
| `z_score_func`            | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | [Z-score](https://www.investopedia.com/terms/z/zscore.asp) as (last return - mean return) / standard deviation of returns.                                                                                                                                     |
| `target_weight_from_var`  | `float`, `pandas.Series` | `OpenTimeSeries`, `OpenFrame` | A position target weight from the ratio between a VaR implied volatility and a given target volatility.                                                                                                                                                        |

## Development instructions

These instructions assume that you
have [Python 3.10](https://docs.python.org/3.10/) and
[Poetry](https://python-poetry.org/) installed on your machine and that you are OK
to install this project in a virtual environment. If not, feel free to do it your
own way.

### In Windows Powershell execute the below commands:

```
git clone https://github.com/CaptorAB/OpenSeries.git
cd OpenSeries
python3 -m venv ./venv
./source_me.ps1
pip install poetry==1.4.0
poetry install --with dev
pre-commit install
```

### In a Mac Terminal/Linux execute the below commands:

```
git clone https://github.com/CaptorAB/OpenSeries.git
cd OpenSeries
make
source source_me
make install
```

### Testing and Linting / Type-checking:

```
make test
make lint
```
The silenced error codes can be found  in the project's
[.flake8](https://github.com/CaptorAB/OpenSeries/blob/master/.flake8) and
[pyproject.toml](https://github.com/CaptorAB/OpenSeries/blob/master/pyproject.toml)
files. Flake8 linting is embedded in the pre-commit hook but not mypy.
