"""Date related utilities."""

from __future__ import annotations

import datetime as dt
from typing import Optional, Union, cast

from dateutil.relativedelta import relativedelta
from holidays import (
    country_holidays,
    list_supported_countries,
)
from numpy import array, busdaycalendar, datetime64, is_busday, where
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timestamp,
    concat,
    date_range,
)
from pandas.tseries.offsets import CustomBusinessDay
from pydantic import PositiveInt

from openseries.types import (
    CountriesType,
    DateType,
    HolidayType,
    LiteralBizDayFreq,
)

__all__ = [
    "date_fix",
    "date_offset_foll",
    "do_resample_to_business_period_ends",
    "generate_calendar_date_range",
    "get_previous_business_day_before_today",
    "holiday_calendar",
    "offset_business_days",
]


def holiday_calendar(
    startyear: int,
    endyear: int,
    countries: CountriesType = "SE",
    custom_holidays: Optional[HolidayType] = None,
) -> busdaycalendar:
    """
    Generate a business calendar.

    Parameters
    ----------
    startyear: int
        First year in date range generated
    endyear: int
        Last year in date range generated
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    numpy.busdaycalendar
        Generate a business calendar

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
        hols = array(sorted(set(countryholidays)), dtype="datetime64[D]")
    else:
        msg = (
            "Argument countries must be a string country code or "
            "a list of string country codes according to ISO 3166-1 alpha-2."
        )
        raise ValueError(
            msg,
        )

    return busdaycalendar(holidays=hols)


def date_fix(
    fixerdate: DateType,
) -> dt.date:
    """
    Parse different date formats into datetime.date.

    Parameters
    ----------
    fixerdate: DateType
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
        return (
            dt.datetime.strptime(str(fixerdate)[:10], "%Y-%m-%d")
            .astimezone()
            .date()
        )
    if isinstance(fixerdate, str):
        return dt.datetime.strptime(fixerdate, "%Y-%m-%d").astimezone().date()
    msg = f"Unknown date format {fixerdate!s} of type {type(fixerdate)!s} encountered"
    raise TypeError(
        msg,
    )


def date_offset_foll(
    raw_date: DateType,
    months_offset: int = 12,
    countries: CountriesType = "SE",
    custom_holidays: Optional[HolidayType] = None,
    *,
    adjust: bool = False,
    following: bool = True,
) -> dt.date:
    """
    Offset dates according to a given calendar.

    Parameters
    ----------
    raw_date: DateType
        The date to offset from
    months_offset: int, default: 12
        Number of months as integer
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]
    adjust: bool, default: False
        Determines if offset should adjust for business days
    following: bool, default: True
        Determines if days should be offset forward (following) or backward

    Returns
    -------
    datetime.date
        Off-set date

    """
    raw_date = date_fix(raw_date)
    month_delta = relativedelta(months=months_offset)

    day_delta = relativedelta(days=1) if following else relativedelta(days=-1)

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
    custom_holidays: Optional[HolidayType] = None,
) -> dt.date:
    """
    Bump date backwards to find the previous business day.

    Parameters
    ----------
    today: datetime.date, optional
        Manual input of the day from where the previous business day is found
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        The previous business day

    """
    if today is None:
        today = dt.datetime.now().astimezone().date()

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
    custom_holidays: Optional[HolidayType] = None,
) -> dt.date:
    """
    Bump date by business days.

    It first adjusts to a valid business day and then bumps with given
    number of business days from there.

    Parameters
    ----------
    ddate: datetime.date
        A starting date that does not have to be a business day
    days: int
        The number of business days to offset from the business day that is given
        If days is set as anything other than an integer its value is set to zero
    countries: CountriesType, default: "SE"
        (List of) country code(s) according to ISO 3166-1 alpha-2
    custom_holidays: HolidayType, optional
        Argument where missing holidays can be added as
        {"2021-02-12": "Jack's birthday"} or ["2021-02-12"]

    Returns
    -------
    datetime.date
        The new offset business day

    """
    try:
        days = int(days)
    except TypeError:
        days = 0

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

    return cast(dt.date, local_bdays[idx[0] + days])


def generate_calendar_date_range(
    trading_days: PositiveInt,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    countries: CountriesType = "SE",
) -> list[dt.date]:
    """
    Generate a list of business day calendar dates.

    Parameters
    ----------
    trading_days: PositiveInt
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
        List of business day calendar dates

    """
    if start and not end:
        tmp_range = date_range(
            start=start,
            periods=trading_days * 365 // 252,
            freq="D",
        )
        calendar = holiday_calendar(
            startyear=start.year,
            endyear=tmp_range[-1].year,
            countries=countries,
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
            startyear=tmp_range[0].year,
            endyear=end.year,
            countries=countries,
        )
        return [
            d.date()
            for d in date_range(
                end=end,
                periods=trading_days,
                freq=CustomBusinessDay(calendar=calendar),
            )
        ]

    msg = (
        "Provide one of start or end date, but not both. "
        "Date range is inferred from number of trading days."
    )
    raise ValueError(
        msg,
    )


def do_resample_to_business_period_ends(
    data: DataFrame,
    head: Series[float],
    tail: Series[float],
    freq: LiteralBizDayFreq,
    countries: CountriesType,
) -> DatetimeIndex:
    """
    Resample timeseries frequency to business calendar month end dates.

    Stubs left in place. Stubs will be aligned to the shortest stub.

    Parameters
    ----------
    data: pandas.DataFrame
        The timeseries data
    head: pandas:Series[float]
        Data point at maximum first date of all series
    tail: pandas:Series[float]
        Data point at minimum last date of all series
    freq: LiteralBizDayFreq
        The date offset string that sets the resampled frequency
    countries: CountriesType
        (List of) country code(s) according to ISO 3166-1 alpha-2
        to create a business day calendar used for date adjustments

    Returns
    -------
    Pandas.DatetimeIndex
        A date range aligned to business period ends

    """
    newhead = head.to_frame().T
    newtail = tail.to_frame().T
    data.index = DatetimeIndex(data.index)
    data = data.resample(rule=freq).last()
    data = data.drop(index=data.index[-1])
    data.index = Index(d.date() for d in DatetimeIndex(data.index))

    if newhead.index[0] not in data.index:
        # noinspection PyUnreachableCode
        data = concat([data, newhead])

    if newtail.index[0] not in data.index:
        # noinspection PyUnreachableCode
        data = concat([data, newtail])

    data = data.sort_index()

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
        + [data.index[-1]],
    )
    return dates.drop_duplicates()
