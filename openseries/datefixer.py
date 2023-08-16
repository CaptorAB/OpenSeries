"""
Date related utilities
"""
from __future__ import annotations
import datetime as dt
from typing import Optional, Union
from dateutil.relativedelta import relativedelta
from holidays import country_holidays, list_supported_countries
from numpy import array, busdaycalendar, datetime64, is_busday, where, timedelta64
from pandas import DataFrame, date_range, Timestamp, Series, DatetimeIndex, concat
from pandas.tseries.offsets import CustomBusinessDay

from openseries.types import (
    CountriesType,
    TradingDaysType,
    LiteralBizDayFreq,
    LiteralPandasResampleConvention,
)


def holiday_calendar(
    startyear: int,
    endyear: int,
    countries: CountriesType = "SE",
    custom_holidays: Optional[
        Union[
            dict[Union[dt.date, dt.datetime, str, float, int], str],
            list[Union[dt.date, dt.datetime, str, float, int]],
            Union[dt.date, dt.datetime, str, float, int],
        ]
    ] = None,
) -> busdaycalendar:
    """Function to generate a business calendar

    Parameters
    ----------
    startyear: int
        First year in date range generated
    endyear: int
        Last year in date range generated
    countries: list[str] | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: Union[
        dict[Union[dt.date, dt.datetime, str, float, int], str],
        list[Union[dt.date, dt.datetime, str, float, int]],
        Union[dt.date, dt.datetime, str, float, int]], optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    numpy.busdaycalendar
        Numpy busdaycalendar object
    """
    startyear -= 1
    endyear += 1
    if startyear == endyear:
        endyear += 1
    years = list(range(startyear, endyear))

    if isinstance(countries, str) and countries in list_supported_countries():
        staging = country_holidays(country=countries, years=years)
        if custom_holidays is not None:
            staging.update(custom_holidays)
        hols = array(sorted(staging.keys()), dtype="datetime64[D]")
    elif isinstance(countries, list) and all(
        country in list_supported_countries() for country in countries
    ):
        country: str
        countryholidays: list[Union[dt.date, str]] = []
        for i, country in enumerate(countries):
            staging = country_holidays(country=country, years=years)
            if i == 0 and custom_holidays is not None:
                staging.update(custom_holidays)
            countryholidays += list(staging)
        hols = array(sorted(list(set(countryholidays))), dtype="datetime64[D]")
    else:
        raise ValueError(
            "Argument countries must be a string country code or a list "
            "of string country codes according to ISO 3166-1 alpha-2."
        )

    return busdaycalendar(holidays=hols)


def date_fix(
    fixerdate: Union[str, dt.date, dt.datetime, datetime64, Timestamp],
) -> dt.date:
    """Function to parse from different date formats into datetime.date

    Parameters
    ----------
    fixerdate: str | datetime.date | datetime.datetime |
    numpy.datetime64 | pandas.Timestamp
        The data item to parse

    Returns
    -------
    datetime.date
        Parsed date
    """

    if isinstance(fixerdate, (Timestamp, dt.datetime)):
        return fixerdate.date()
    if isinstance(fixerdate, dt.date):
        return fixerdate
    if isinstance(fixerdate, datetime64):
        unix_epoch = datetime64(0, "s")
        one_second = timedelta64(1, "s")
        seconds_since_epoch = (fixerdate - unix_epoch) / one_second
        return dt.datetime.utcfromtimestamp(float(seconds_since_epoch)).date()
    if isinstance(fixerdate, str):
        return dt.datetime.strptime(fixerdate, "%Y-%m-%d").date()
    raise TypeError(
        f"Unknown date format {str(fixerdate)} of "
        f"type {str(type(fixerdate))} encountered"
    )


def date_offset_foll(
    raw_date: Union[str, dt.date, dt.datetime, datetime64, Timestamp],
    months_offset: int = 12,
    adjust: bool = False,
    following: bool = True,
    countries: CountriesType = "SE",
    custom_holidays: Optional[
        Union[
            dict[Union[dt.date, dt.datetime, str, float, int], str],
            list[Union[dt.date, dt.datetime, str, float, int]],
            Union[dt.date, dt.datetime, str, float, int],
        ]
    ] = None,
) -> dt.date:
    """Function to offset dates according to a given calendar

    Parameters
    ----------
    raw_date: str | datetime.date | datetime.datetime | numpy.datetime64 |
    pandas.Timestamp
        The date to offset from
    months_offset: int, default: 12
        Number of months as integer
    adjust: bool, default: False
        Determines if offset should adjust for business days
    following: bool, default: True
        Determines if days should be offset forward (following) or backward
    countries: list[str] | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: Union[
        dict[Union[dt.date, dt.datetime, str, float, int], str],
        list[Union[dt.date, dt.datetime, str, float, int]],
        Union[dt.date, dt.datetime, str, float, int]], optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        Off-set date
    """

    raw_date = date_fix(raw_date)
    month_delta = relativedelta(months=months_offset)

    if following:
        day_delta = relativedelta(days=1)
    else:
        day_delta = relativedelta(days=-1)

    new_date = raw_date + month_delta

    if adjust:
        startyear = min([raw_date.year, new_date.year])
        endyear = max([raw_date.year, new_date.year])
        calendar = holiday_calendar(
            startyear=startyear,
            endyear=endyear,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        while not is_busday(dates=new_date, busdaycal=calendar):
            new_date += day_delta

    return new_date


def get_previous_business_day_before_today(
    today: Optional[dt.date] = None,
    countries: CountriesType = "SE",
    custom_holidays: Optional[
        Union[
            dict[Union[dt.date, dt.datetime, str, float, int], str],
            list[Union[dt.date, dt.datetime, str, float, int]],
            Union[dt.date, dt.datetime, str, float, int],
        ]
    ] = None,
) -> dt.date:
    """Function to bump backwards to find the previous business day before today

    Parameters
    ----------
    today: datetime.date, optional
        Manual input of the day from where the previous business day is found
    countries: list[str] | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: Union[
        dict[Union[dt.date, dt.datetime, str, float, int], str],
        list[Union[dt.date, dt.datetime, str, float, int]],
        Union[dt.date, dt.datetime, str, float, int]], optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        The previous business day
    """

    if today is None:
        today = dt.date.today()

    return date_offset_foll(
        today - dt.timedelta(days=1),
        countries=countries,
        custom_holidays=custom_holidays,
        months_offset=0,
        adjust=True,
        following=False,
    )


def offset_business_days(
    ddate: dt.date,
    days: int,
    countries: CountriesType = "SE",
    custom_holidays: Optional[
        Union[
            dict[Union[dt.date, dt.datetime, str, float, int], str],
            list[Union[dt.date, dt.datetime, str, float, int]],
            Union[dt.date, dt.datetime, str, float, int],
        ]
    ] = None,
) -> dt.date:
    """Function to bump a date by business days instead of calendar days.
    It first adjusts to a valid business day and then bumps with given
    number of business days from there

    Parameters
    ----------
    ddate: datetime.date
        A starting date that does not have to be a business day
    days: int
        The number of business days to offset from the business day that is
        the closest preceding the day given
    countries: list[str] | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: Union[
        dict[Union[dt.date, dt.datetime, str, float, int], str],
        list[Union[dt.date, dt.datetime, str, float, int]],
        Union[dt.date, dt.datetime, str, float, int]], optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        The new offset business day
    """
    if days <= 0:
        scaledtoyeardays = int((days * 372 / 250) // 1) - 365
        ndate = ddate + dt.timedelta(days=scaledtoyeardays)
        calendar = holiday_calendar(
            startyear=ndate.year,
            endyear=ddate.year,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        local_bdays: list[dt.date] = [
            bday.date()
            for bday in date_range(
                periods=abs(scaledtoyeardays),
                end=ddate,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]
    else:
        scaledtoyeardays = int((days * 372 / 250) // 1) + 365
        ndate = ddate + dt.timedelta(days=scaledtoyeardays)
        calendar = holiday_calendar(
            startyear=ddate.year,
            endyear=ndate.year,
            countries=countries,
            custom_holidays=custom_holidays,
        )
        local_bdays = [
            bday.date()
            for bday in date_range(
                start=ddate,
                periods=scaledtoyeardays,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

    while ddate not in local_bdays:
        if days <= 0:
            ddate -= dt.timedelta(days=1)
        else:
            ddate += dt.timedelta(days=1)

    idx = where(array(local_bdays) == ddate)[0]

    return date_fix(local_bdays[idx[0] + days])


def generate_calender_date_range(
    trading_days: TradingDaysType,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    countries: CountriesType = "SE",
) -> list[dt.date]:
    """Generates a list of business day calender dates

        Parameters
        ----------
        trading_days: TradingDaysType
            Number of days to generate
        start: datetime.date, optional
            Date when the range starts
        end: datetime.date, optional
            Date when the range ends
        countries: CountriesType, default: "SE"
            (List of) country code(s) according to ISO 3166-1 alpha-2

    Returns
    -------
    list[dt.date]
        List of business day calender dates
    """
    if start and not end:
        tmp_range = date_range(
            start=start, periods=trading_days * 365 // 252, freq="D"
        )
        calendar = holiday_calendar(
            startyear=start.year, endyear=tmp_range[-1].year, countries=countries
        )
        return [
            d.date()
            for d in date_range(
                start=start,
                periods=trading_days,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

    if end and not start:
        tmp_range = date_range(end=end, periods=trading_days * 365 // 252, freq="D")
        calendar = holiday_calendar(
            startyear=tmp_range[0].year, endyear=end.year, countries=countries
        )
        return [
            d.date()
            for d in date_range(
                end=end,
                periods=trading_days,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

    raise ValueError(
        "Provide one of start or end date, but not both. "
        "Date range is inferred from number of trading days."
    )


def align_dataframe_to_local_cdays(
    data: DataFrame, countries: CountriesType = "SE"
) -> DataFrame:
    """Changes the index of the associated Pandas DataFrame .tsdf to align with
    local calendar business days

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    countries: list[str] | str, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2

    Returns
    -------
    pandas.DataFrame
        An OpenFrame object
    """
    startyear = data.index[0].year
    endyear = data.index[-1].year
    calendar = holiday_calendar(
        startyear=startyear, endyear=endyear, countries=countries
    )

    d_range = [
        d.date()
        for d in date_range(
            start=data.first_valid_index(),
            end=data.last_valid_index(),
            freq=CustomBusinessDay(calendar=calendar),
        )
    ]
    return data.reindex(d_range, method=None, copy=False)


def do_resample_to_business_period_ends(
    data: DataFrame,
    head: Series,
    tail: Series,
    freq: LiteralBizDayFreq,
    countries: CountriesType,
    convention: LiteralPandasResampleConvention,
) -> DatetimeIndex:
    """Resamples timeseries frequency to the business calendar
    month end dates of each period while leaving any stubs
    in place. Stubs will be aligned to the shortest stub

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    head: pandas:Series
        Data point at maximum first date of all series
    tail: pandas:Series
        Data point at minimum last date of all series
    freq: LiteralBizDayFreq
        The date offset string that sets the resampled frequency
    countries: CountriesType
        (List of) country code(s) according to ISO 3166-1 alpha-2
        to create a business day calendar used for date adjustments
    convention: LiteralPandasResampleConvention
        Controls whether to use the start or end of `rule`.

    Returns
    -------
    Pandas.DatetimeIndex
        A date range aligned to business period ends
    """

    head = head.to_frame().T
    tail = tail.to_frame().T
    data.index = DatetimeIndex(data.index)
    data = data.resample(rule=freq, convention=convention).last()
    data.drop(index=data.index[-1], inplace=True)
    data.index = [d.date() for d in DatetimeIndex(data.index)]

    if head.index[0] not in data.index:
        data = concat([data, head])

    if tail.index[0] not in data.index:
        data = concat([data, tail])

    data.sort_index(inplace=True)

    dates = DatetimeIndex(
        [data.index[0]]
        + [
            date_offset_foll(
                dt.date(d.year, d.month, 1)
                + relativedelta(months=1)
                - dt.timedelta(days=1),
                countries=countries,
                months_offset=0,
                adjust=True,
                following=False,
            )
            for d in data.index[1:-1]
        ]
        + [data.index[-1]]
    )
    dates = dates.drop_duplicates()
    return dates


def get_calc_range(
    data: DataFrame,
    months_offset: Optional[int] = None,
    from_dt: Optional[dt.date] = None,
    to_dt: Optional[dt.date] = None,
) -> tuple[dt.date, dt.date]:
    """Creates user defined date range

    Parameters
    ----------
    data: pandas.DataFrame
        The data with the date range
    months_offset: int, optional
        Number of months offset as positive integer. Overrides use of from_date
        and to_date
    from_dt: datetime.date, optional
        Specific from date
    to_dt: datetime.date, optional
        Specific from date

    Returns
    -------
    Tuple[datetime.date, datetime.date]
        Start and end date of the chosen date range
    """
    earlier, later = data.index[0], data.index[-1]
    if months_offset is not None or from_dt is not None or to_dt is not None:
        if months_offset is not None:
            earlier = date_offset_foll(
                raw_date=data.index[-1],
                months_offset=-months_offset,
                adjust=False,
                following=True,
            )
            assert (
                earlier >= data.index[0]
            ), "Function calc_range returned earlier date < series start"
            later = data.index[-1]
        else:
            if from_dt is not None and to_dt is None:
                assert (
                    from_dt >= data.index[0]
                ), "Function calc_range returned earlier date < series start"
                earlier, later = from_dt, data.index[-1]
            elif from_dt is None and to_dt is not None:
                assert (
                    to_dt <= data.index[-1]
                ), "Function calc_range returned later date > series end"
                earlier, later = data.index[0], to_dt
            elif from_dt is not None and to_dt is not None:
                assert (
                    to_dt <= data.index[-1] and from_dt >= data.index[0]
                ), "Function calc_range returned dates outside series range"
                earlier, later = from_dt, to_dt
        while earlier not in data.index.tolist():
            earlier -= dt.timedelta(days=1)
        while later not in data.index.tolist():
            later += dt.timedelta(days=1)

    return earlier, later
