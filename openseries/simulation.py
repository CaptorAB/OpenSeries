"""Defining the ReturnSimulation class."""

from __future__ import annotations

import datetime as dt
from typing import Optional, cast

from numpy import multiply, sqrt
from numpy.random import PCG64, Generator, SeedSequence
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    concat,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)
from typing_extensions import Self

from openseries.datefixer import generate_calendar_date_range
from openseries.types import (
    CountriesType,
    DaysInYearType,
    ValueType,
)

__all__ = ["ReturnSimulation"]


def _random_generator(seed: Optional[int]) -> Generator:
    """
    Make a Numpy Random Generator object.

    Parameters
    ----------
    seed: int, optional
        Random seed

    Returns
    -------
    numpy.random.Generator
        Numpy random process generator

    """
    ss = SeedSequence(entropy=seed)
    bg = PCG64(seed=cast(Optional[int], ss))
    return Generator(bit_generator=bg)


class ReturnSimulation(BaseModel):

    """
    The class ReturnSimulation allows for simulating financial timeseries.

    Parameters
    ----------
    number_of_sims : PositiveInt
        Number of simulations to generate
    trading_days: PositiveInt
        Total number of days to simulate
    trading_days_in_year : DaysInYearType
        Number of trading days used to annualize
    mean_annual_return : float
        Mean annual return of the distribution
    mean_annual_vol : PositiveFloat
        Mean annual standard deviation of the distribution
    dframe: pandas.DataFrame
        Pandas DataFrame object holding the resulting values
    jumps_lamda: NonNegativeFloat, default: 0.0
        This is the probability of a jump happening at each point in time
    jumps_sigma: NonNegativeFloat, default: 0.0
        This is the volatility of the jump size
    jumps_mu: float, default: 0.0
        This is the average jump size
    seed: int, optional
        Seed for random process initiation

    """

    number_of_sims: PositiveInt
    trading_days: PositiveInt
    trading_days_in_year: DaysInYearType
    mean_annual_return: float
    mean_annual_vol: PositiveFloat
    dframe: DataFrame
    jumps_lamda: NonNegativeFloat = 0.0
    jumps_sigma: NonNegativeFloat = 0.0
    jumps_mu: float = 0.0
    seed: Optional[int] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
    )

    @property
    def results(self: Self) -> DataFrame:
        """
        Simulation data.

        Returns
        -------
        pandas.DataFrame
            Simulation data

        """
        return self.dframe.add(1.0).cumprod(axis="columns").T

    @property
    def realized_mean_return(self: Self) -> float:
        """
        Annualized arithmetic mean of returns.

        Returns
        -------
        float
            Annualized arithmetic mean of returns

        """
        return cast(
            float,
            (
                self.results.pct_change(fill_method=cast(str, None)).mean()
                * self.trading_days_in_year
            ).iloc[0],
        )

    @property
    def realized_vol(self: Self) -> float:
        """
        Annualized volatility.

        Returns
        -------
        float
            Annualized volatility

        """
        return cast(
            float,
            (
                self.results.pct_change(fill_method=cast(str, None)).std()
                * sqrt(self.trading_days_in_year)
            ).iloc[0],
        )

    @classmethod
    def from_normal(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
        randomizer: Optional[Generator] = None,
    ) -> ReturnSimulation:
        """
        Create a Normal distribution simulation.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize
        randomizer: numpy.random.Generator, optional
            Random process generator

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation

        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        returns = randomizer.normal(
            loc=mean_annual_return / trading_days_in_year,
            scale=mean_annual_vol / sqrt(trading_days_in_year),
            size=(number_of_sims, trading_days),
        )

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=returns, dtype="float64"),
            seed=seed,
        )

    @classmethod
    def from_lognormal(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
        randomizer: Optional[Generator] = None,
    ) -> ReturnSimulation:
        """
        Create a Lognormal distribution simulation.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize
        randomizer: numpy.random.Generator, optional
            Random process generator

        Returns
        -------
        ReturnSimulation
            Lognormal distribution simulation

        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        returns = (
            randomizer.lognormal(
                mean=mean_annual_return / trading_days_in_year,
                sigma=mean_annual_vol / sqrt(trading_days_in_year),
                size=(number_of_sims, trading_days),
            )
            - 1
        )

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=returns, dtype="float64"),
            seed=seed,
        )

    @classmethod
    def from_gbm(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
        randomizer: Optional[Generator] = None,
    ) -> ReturnSimulation:
        """
        Create a Geometric Brownian Motion simulation.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType, default: 252
            Number of trading days used to annualize
        randomizer: numpy.random.Generator, optional
            Random process generator

        Returns
        -------
        ReturnSimulation
            Geometric Brownian Motion simulation

        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        drift = (mean_annual_return - 0.5 * mean_annual_vol**2.0) * (
            1.0 / trading_days_in_year
        )

        normal_mean = 0.0
        wiener = randomizer.normal(
            loc=normal_mean,
            scale=sqrt(1.0 / trading_days_in_year) * mean_annual_vol,
            size=(number_of_sims, trading_days),
        )

        returns = drift + wiener

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=returns, dtype="float64"),
            seed=seed,
        )

    @classmethod
    def from_merton_jump_gbm(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        trading_days: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        seed: int,
        jumps_lamda: NonNegativeFloat,
        jumps_sigma: NonNegativeFloat = 0.0,
        jumps_mu: float = 0.0,
        trading_days_in_year: DaysInYearType = 252,
        randomizer: Optional[Generator] = None,
    ) -> ReturnSimulation:
        """
        Create a Merton Jump-Diffusion model simulation.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        seed: int
            Seed for random process initiation
        jumps_lamda: NonNegativeFloat
            This is the probability of a jump happening at each point in time
        jumps_sigma: NonNegativeFloat, default: 0.0
            This is the volatility of the jump size
        jumps_mu: float, default: 0.0
            This is the average jump size
        trading_days_in_year: DaysInYearType, default: 252
            Number of trading days used to annualize
        randomizer: numpy.random.Generator, optional
            Random process generator

        Returns
        -------
        ReturnSimulation
            Merton Jump-Diffusion model simulation

        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        normal_mean = 0.0
        wiener = randomizer.normal(
            loc=normal_mean,
            scale=sqrt(1.0 / trading_days_in_year) * mean_annual_vol,
            size=(number_of_sims, trading_days),
        )

        poisson_jumps = multiply(
            randomizer.poisson(
                lam=jumps_lamda * (1.0 / trading_days_in_year),
                size=(number_of_sims, trading_days),
            ),
            randomizer.normal(
                loc=jumps_mu,
                scale=jumps_sigma,
                size=(number_of_sims, trading_days),
            ),
        )

        drift = (
            mean_annual_return
            - 0.5 * mean_annual_vol**2.0
            - jumps_lamda * (jumps_mu + jumps_sigma**2.0)
        ) * (1.0 / trading_days_in_year)

        returns = poisson_jumps + drift + wiener

        returns[:, 0] = 0.0

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            jumps_lamda=jumps_lamda,
            jumps_sigma=jumps_sigma,
            jumps_mu=jumps_mu,
            dframe=DataFrame(data=returns, dtype="float64"),
            seed=seed,
        )

    def to_dataframe(
        self: Self,
        name: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        countries: CountriesType = "SE",
    ) -> DataFrame:
        """
        Create a pandas.DataFrame from simulation(s).

        Parameters
        ----------
        name: str
            Name label of the serie(s)
        start: datetime.date, optional
            Date when the simulation starts
        end: datetime.date, optional
            Date when the simulation ends
        countries: CountriesType, default: "SE"
            (List of) country code(s) according to ISO 3166-1 alpha-2

        Returns
        -------
        pandas.DataFrame
            The simulation(s) data

        """
        d_range = generate_calendar_date_range(
            trading_days=self.trading_days,
            start=start,
            end=end,
            countries=countries,
        )

        if self.number_of_sims == 1:
            sdf = self.dframe.iloc[0].T.to_frame()
            sdf.index = Index(d_range)
            sdf.columns = MultiIndex.from_arrays(
                [
                    [name],
                    [ValueType.RTRN],
                ],
            )
            return sdf

        fdf = DataFrame()
        for item in range(self.number_of_sims):
            sdf = self.dframe.iloc[item].T.to_frame()
            sdf.index = Index(d_range)
            sdf.columns = MultiIndex.from_arrays(
                [
                    [f"{name}_{item}"],
                    [ValueType.RTRN],
                ],
            )
            fdf = concat([fdf, sdf], axis="columns", sort=True)
        return fdf
